"""Python wrapper: build a C++ UlyssesGroup from a torch ProcessGroup (bootstrap is pure C++)."""

from __future__ import annotations

import os
import warnings
from typing import Callable

import torch
import torch.distributed as dist

# NVSHMEM reads NVSHMEM_SYMMETRIC_SIZE when it (re)initializes -- i.e. when the first LIVE group is
# constructed (destroying the last group finalizes NVSHMEM, so the next group re-initializes with a
# fresh size). While any group is alive the heap keeps its size; track that to warn instead of
# failing at some later nvshmem_align. destroy() keeps the count in step with the C++ group count.
_live_groups = 0
_heap_bytes = 0


class AsyncA2AHandle:
    """Result of an async a2a: the collective runs on the group's comm stream; wait() makes the
    CALLER's current stream wait for it (GPU-side event wait, host does not block) and returns the
    output. WHAT the output is depends on which entry point made the handle:
    ``all_to_all_single_4d_async`` gives a tensor the caller owns, and
    ``all_to_all_single_4d_borrowed_async`` a view of the tag-scoped symmetric buffer that the
    next call with that tag overwrites."""

    def __init__(self, out, ev_done: torch.cuda.Event):
        self._out = out
        self._ev_done = ev_done

    def wait(self):
        torch.cuda.current_stream().wait_event(self._ev_done)
        return self._out


class UlyssesGroup:
    """Ulysses all-to-all group over the NVSHMEM symmetric heap (single-node NVLink P2P).

    Wraps the C++ ``UlyssesGroup`` custom class: construction broadcasts the NVSHMEM unique id
    via ``torch.distributed``, initializes NVSHMEM, and reserves a symmetric-heap pool that all
    collectives allocate their outputs from (buffers reused per tag+size+dtype).

    ``process_group`` may be a strided subgroup -- that is what 2-D parallelism needs (tp=2 x
    ulysses-sp=4 on 8 GPUs: the sp groups are {0,2,4,6} and {1,3,5,7}) -- but construction stays
    collective over the WHOLE JOB, never over just that subgroup: the NVSHMEM bootstrap,
    ``nvshmem_team_split_strided`` and every symmetric-heap allocation are collectives over all
    PEs. So EVERY rank must build its group together, and concurrently-live groups must keep
    allocating in step afterwards -- the first call carrying a given tag+size+dtype allocates
    from the shared heap, so both sp groups have to issue the same shapes in the same order.
    """

    def __init__(
        self,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
        initial_pool_bytes: int = 2 << 30,
    ) -> None:
        """
        Args:
            process_group: bootstrap process group; ``None`` uses ``dist.group.WORLD``. Its ranks
                must be evenly strided (any stride); pass the ulysses subgroup here under 2-D
                parallelism.
            device: this rank's CUDA device; ``None`` uses the current device.
            initial_pool_bytes: NVSHMEM symmetric-heap reservation (default 2 GiB); every
                collective's output buffer is carved from this pool.
        """
        pg = process_group if process_group is not None else dist.group.WORLD
        self.pg = pg
        self.rank = dist.get_rank(pg)
        self.world_size = dist.get_world_size(pg)
        self.peer_global_ranks = list(dist.get_process_group_ranks(pg))
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        torch.cuda.set_device(device)

        # Reservation must be set via env before NVSHMEM init; it takes effect only when NVSHMEM
        # (re)initializes, which happens while no other group is alive.
        global _live_groups, _heap_bytes
        if _live_groups == 0:
            os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str(int(initial_pool_bytes))
            _heap_bytes = int(initial_pool_bytes)
        elif int(initial_pool_bytes) > _heap_bytes:
            warnings.warn(
                f"initial_pool_bytes={int(initial_pool_bytes)} exceeds the NVSHMEM heap sized by "
                f"the first live UlyssesGroup ({_heap_bytes} B); the extra bytes may not be backed "
                "(size the first group's pool for all concurrently-live groups)",
                stacklevel=2,
            )
        # P2P direct writes do not need NVLS (NVLink SHARP multicast); on some nodes its
        # multicast heap mapping fails and segfaults, so disable by default for cross-node
        # robustness (overridable via env).
        os.environ.setdefault("NVSHMEM_DISABLE_NVLS", "1")
        # This op is single-node NVLink P2P only; on nodes with IB NICs, NVSHMEM tries to init
        # the IB remote transport and segfaults, so disable remote transport by default
        # (verified on H200+IB nodes: init SIGSEGVs otherwise).
        os.environ.setdefault("NVSHMEM_REMOTE_TRANSPORT", "none")

        cls = torch.classes.fast_ulysses.UlyssesGroup
        if dist.get_rank() == 0:
            uid = cls.get_uniqueid()
        else:
            uid = [0] * cls.uniqueid_nints()
        uid_t = torch.tensor(uid, dtype=torch.int64, device=device)
        # Generated on GLOBAL rank 0 and broadcast on WORLD -- NOT on ``pg``, even when ``pg`` is a
        # subgroup. init_world below bootstraps ONE NVSHMEM job of dist.get_world_size() PEs, and
        # every PE of it must join with the SAME id. Narrowing this broadcast to ``pg`` would give
        # each subgroup its own id, and a job-sized bootstrap fed two ids never completes.
        dist.broadcast(uid_t, src=0, group=dist.group.WORLD)
        cls.init_world(uid_t.tolist(), dist.get_rank(), dist.get_world_size())

        dist.barrier(group=pg)
        self._group = cls(
            [int(r) for r in self.peer_global_ranks],
            int(self.rank),
            int(device.index),
            int(initial_pool_bytes),
        )
        dist.barrier(group=pg)
        _live_groups += 1
        self._destroyed = False

        # Dedicated high-priority stream for the ASYNC collectives (sync calls run directly on the
        # caller's stream -- routing them through here costs two event hops per call, ~0.27 ms
        # measured, comparable to the a2a itself). This stream and the caller's are NOT ordered
        # against each other, which is safe because the fast_barrier state is per TAG: only a
        # tag's own calls have to stay ordered (see all_to_all_single_4d_async).
        # High priority lets the comm kernels get SM slots under concurrent compute.
        _, greatest = torch.cuda.Stream.priority_range()
        self._comm_stream = torch.cuda.Stream(device=device, priority=greatest)
        # Persistent staging buffers for async inputs, keyed (tag, shape, dtype):
        # (buffer, release_event). See _stage_input.
        self._staging: dict[tuple, tuple[torch.Tensor, torch.cuda.Event]] = {}

    def _stage_input(self, x: torch.Tensor, tag: str) -> tuple[torch.Tensor, torch.cuda.Event]:
        """Copy x into the persistent per-(tag, shape, dtype) staging buffer on the CALLER's
        stream and return (staging, release_event). The comm stream reads only the staging
        copy, so the caller's tensor is never retained cross-stream. (record_stream would pin
        every freed input until the comm stream catches up -- with a host running many layers
        ahead that inflates the allocator's reserved pool by the whole in-flight window.)
        Reuse waits GPU-side for the previous collective on the same key to finish reading;
        consecutive uses of one tag are a full layer apart, so this does not stall."""
        key = (tag, tuple(x.shape), x.dtype)
        entry = self._staging.get(key)
        if entry is None:
            entry = (torch.empty_like(x), torch.cuda.Event())
            self._staging[key] = entry
        else:
            torch.cuda.current_stream().wait_event(entry[1])
        entry[0].copy_(x)
        return entry

    def _launch_on_comm_stream(self, releases: list[torch.cuda.Event], fn: Callable):
        """Run a collective on the group's comm stream: comm stream waits for the caller's current
        stream (staged inputs ready -- and, since the ready-event trails everything already
        submitted, any earlier consumer of the same tag's buffer), runs fn, records the staging
        release events (comm stream done reading), and returns (result, done_event)."""
        cur = torch.cuda.current_stream()
        ev_ready = torch.cuda.Event()
        ev_ready.record(cur)
        self._comm_stream.wait_event(ev_ready)
        with torch.cuda.stream(self._comm_stream):
            out = fn()
        for ev in releases:
            ev.record(self._comm_stream)
        ev_done = torch.cuda.Event()
        ev_done.record(self._comm_stream)
        return out, ev_done

    def all_to_all_single_4d(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        tag: str = "",
        out: torch.Tensor | None = None,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> torch.Tensor:
        """4D all-to-all: mode0 scatters heads / gathers sequence, mode1 inverts it. Returns a
        tensor the CALLER OWNS -- an ordinary tensor with no lifetime rules attached.

        The result is copied out of the tag's symmetric window into ``out`` (validated: CUDA,
        contiguous, matching dtype and shape) or into a freshly allocated tensor. That copy is
        a device-to-device pass over the output, issued on the caller's stream behind the
        closing handshake, and it is a stage of ``all_to_all_single_4d_timed`` -- measure it
        rather than guess. ``all_to_all_single_4d_borrowed`` is this call without it.

        The transfer rides the DMA engines (zero SM), so it overlaps compute that an
        SM-resident collective cannot. Collective -- every rank MUST issue the SAME
        (shape, mode) call sequence, or the group hangs; sync, async, copying and borrowed
        calls all count in that one sequence.

        A tag names one symmetric window and one barrier state, so a tag's calls must stay
        ORDERED, which on one stream they are. Concurrently-live RESULTS do not need distinct
        tags on this form -- each is copied out before the next call on that tag is issued, and
        the opening barrier is what keeps a peer out of the window until it has been. They do
        need distinct tags on ``all_to_all_single_4d_borrowed``, so q/k/v usually get them
        anyway; the cost of one more tag is one more window on the symmetric heap.

        ``seq_splits[p]`` / ``head_splits[p]`` are rank p's sequence and head shard: pass BOTH
        or NEITHER, identical on every rank, and they must match the shape actually handed in
        (checked). Passing neither means even shards and the divisibility that implies -- which
        is what a caller who pads the sequence to a multiple of world_size gets today.
        Full contract: docs/API.md.
        """
        return torch.ops.fast_ulysses.all_to_all_single_4d(
            self._group, x.contiguous(), mode, tag, seq_splits, head_splits, out
        )

    def all_to_all_single_4d_borrowed(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        tag: str = "",
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> torch.Tensor:
        """The same collective, except the result IS the tag's symmetric window: no copy-out.

        Shapes, splits and the collective call sequence are exactly ``all_to_all_single_4d``'s.
        What is extra is a lifetime contract that NOTHING IN THIS LIBRARY ENFORCES -- no check,
        no assertion, no debug mode:

        * The result is valid only until the next call carrying this tag. That call's transfer
          writes the same bytes, and this result silently becomes that one.
        * Consume it on the stream that produced it, before that next call. Reading it on
          another stream is yours to synchronise.
        * Do not read it after ``destroy()``; the memory is freed.
        * ``.clone()``, or any op that produces a new tensor, is how you keep it.

        Cross-rank safety IS handled: a peer cannot overwrite this window until every rank has
        reached the next call's opening barrier, and your reads are ordered ahead of that
        barrier on your stream (tests/distributed/a2a_window_race.py is what says so).

        This is a separate method rather than a flag so that the borrow is visible at the call
        site. When in doubt use ``all_to_all_single_4d``, which copies and has no rules.

        No ``barrier`` parameter, deliberately: a deferred sync result would be an unreadable
        view with nothing left to publish it. The async form does expose it.
        """
        return torch.ops.fast_ulysses.all_to_all_single_4d_borrowed(
            self._group, x.contiguous(), mode, tag, True, seq_splits, head_splits
        )

    def all_to_all_single_4d_async(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        tag: str = "",
        out: torch.Tensor | None = None,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> AsyncA2AHandle:
        """Async form of ``all_to_all_single_4d``, on the group's comm stream; handle.wait()
        makes the caller's stream wait (GPU-side) and returns the tensor the caller owns. Same
        collective contract as the sync call, ``out`` / ``seq_splits`` / ``head_splits``
        included.

        Ordering is per TAG, not per group: the barrier flags and epoch counter are keyed by
        tag (ulysses_group.cuh BarrierState), so an outstanding handle must be wait()ed before
        the next call with THAT tag, while a call on another tag may run concurrently on an
        unordered stream (tests/distributed/a2a_overlapping_barriers.py builds exactly that).

        No ``barrier`` parameter here: deferring the closing handshake would leave the copy-out
        reading the window before the peers' writes had landed. It lives on
        ``all_to_all_single_4d_borrowed_async``. Full contract: docs/API.md.
        """
        x, ev_free = self._stage_input(x.contiguous(), tag)
        y, ev_done = self._launch_on_comm_stream(
            [ev_free],
            lambda: torch.ops.fast_ulysses.all_to_all_single_4d(
                self._group, x, mode, tag, seq_splits, head_splits, out
            ),
        )
        # Two streams touch the output: one allocated it, the other wrote it. Whichever did not
        # allocate it is a cross-stream use, and the caching allocator has to know before it may
        # recycle the block. Registering both covers either origin -- a caller-supplied ``out``
        # belongs to the caller's stream, one allocated inside the op belongs to the comm stream
        # -- and the allocator ignores a block's own stream, so one of the two is a no-op.
        y.record_stream(self._comm_stream)
        y.record_stream(torch.cuda.current_stream())
        return AsyncA2AHandle(y, ev_done)

    def all_to_all_single_4d_borrowed_async(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        tag: str = "",
        barrier: bool = True,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> AsyncA2AHandle:
        """Async form of ``all_to_all_single_4d_borrowed``: handle.wait() returns the WINDOW
        VIEW, under the same unenforced rules, with "the stream that produced it" being the
        caller's stream from wait() onwards.

        barrier=False defers only the CLOSING handshake to a later barrier=True call on the
        same stream, so several calls share one -- until then the deferred call's output view
        is not safe to read, only the barrier-carrying handle's wait() implies peers' writes
        arrived, and all ranks must use the identical barrier pattern. Full contract:
        docs/API.md.
        """
        x, ev_free = self._stage_input(x.contiguous(), tag)
        out, ev_done = self._launch_on_comm_stream(
            [ev_free],
            lambda: torch.ops.fast_ulysses.all_to_all_single_4d_borrowed(
                self._group, x, mode, tag, barrier, seq_splits, head_splits
            ),
        )
        return AsyncA2AHandle(out, ev_done)

    def all_to_all_single_4d_timed(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        tag: str = "",
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run the COPYING collective and return ``(output, {stage: ms})``.

        Stages are ``barrier_in`` (writers wait for readers), ``transfer`` (the peer copies,
        with this rank's own share running underneath them on the caller's stream),
        ``barrier_out`` (readers wait for writers) and ``copy_out`` (window -> the caller's
        tensor, the one stage ``all_to_all_single_4d_borrowed`` does not pay). Strictly ordered
        on one stream, so they sum to the whole call.

        **Benchmark only.** Reading the events synchronises the device, which the normal entry
        point never does.
        """
        out, stages = torch.ops.fast_ulysses.all_to_all_single_4d_timed(
            self._group, x.contiguous(), mode, tag, seq_splits, head_splits
        )
        return out, {
            "barrier_in": float(stages[0]),
            "transfer": float(stages[1]),
            "barrier_out": float(stages[2]),
            "copy_out": float(stages[3]),
        }

    def destroy(self) -> None:
        """Release the symmetric-heap resources (collective: ALL ranks must call together)."""
        if self._destroyed:
            return
        # Drain the comm stream first: dist.barrier only syncs the caller's current stream, so an
        # unwaited async a2a could still be writing the buffers nvshmem_free is about to release.
        self._comm_stream.synchronize()
        self._staging.clear()
        dist.barrier(group=self.pg)
        self._group.destroy()
        self._destroyed = True
        global _live_groups
        _live_groups -= 1
