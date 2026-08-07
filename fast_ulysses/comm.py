"""Python wrapper: build a C++ UlyssesGroup from a torch ProcessGroup (bootstrap is pure C++)."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import AsyncCollectiveTensor

from . import _C

# NVSHMEM reads NVSHMEM_SYMMETRIC_SIZE when it (re)initializes -- i.e. when the first LIVE group is
# constructed (destroying the last group finalizes NVSHMEM, so the next group re-initializes with a
# fresh size). While any group is alive the heap keeps its size; track that to warn instead of
# failing at some later allocation. destroy() keeps the count in step with the C++ group count.
_live_groups = 0
_heap_bytes = 0


class CompletedHandle:
    """What the async calls return on a build whose libtorch has no ``c10d::register_work``
    (probed by CMakeLists.txt; see ``csrc/work.h``). There is no registry to bind the completion
    event to, so the C++ side has already made the caller's stream wait on it -- correct, but
    with none of the overlap the async path exists for. ``wait()`` therefore only unwraps.

    A different type from the ``AsyncCollectiveTensor`` the normal path returns, so that
    difference is visible rather than silent; ``wait()`` is the surface both share."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def wait(self) -> torch.Tensor:
        return self._tensor

    def __repr__(self) -> str:
        return f"CompletedHandle({tuple(self._tensor.shape)}, {self._tensor.dtype})"


class UlyssesGroup:
    """Ulysses all-to-all group over the NVSHMEM symmetric heap (single-node GPU-to-GPU P2P).

    Wraps the C++ ``UlyssesGroup`` custom class: construction broadcasts the NVSHMEM unique id
    via ``torch.distributed``, initializes NVSHMEM, and reserves a symmetric-heap pool that all
    collectives allocate their outputs from (buffers reused per tag+size+dtype).

    ``process_group`` may be a strided subgroup -- that is what 2-D parallelism needs (tp=2 x
    ulysses-sp=4 on 8 GPUs: the sp groups are {0,2,4,6} and {1,3,5,7}) -- but construction stays
    collective over the WHOLE JOB, never over just that subgroup: the NVSHMEM bootstrap and
    ``nvshmem_team_split_strided`` are collectives over all PEs. So EVERY rank must build its
    group together.

    Every pair of ranks in the group must be P2P-mappable, which covers NVLink and PCIe alike,
    including pairs on different root complexes. An unreachable pair is refused at construction.

    Call ``reserve`` once afterwards to declare the windows this process will use; that takes
    symmetric allocation off the call path, after which concurrently-live groups may issue
    different shapes in different orders.
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
            initial_pool_bytes: the pool, taken in full by one symmetric allocation at
                construction (default 2 GiB); every collective's window is an offset into it.
                This is COMMITTED, not a cap -- size it for what the process will use.
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
        #
        # The heap is sized ABOVE the pool because the pool is now a single nvshmem_align of the
        # whole `initial_pool_bytes` (see csrc/symmetric_pool.cuh), and NVSHMEM keeps its own
        # bookkeeping in the same heap -- asking for 100% of it need not succeed.
        global _live_groups, _heap_bytes
        if _live_groups == 0:
            os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str(int(initial_pool_bytes) + (64 << 20))
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
        # This op is single-node P2P only. Where an IB NIC is present NVSHMEM will otherwise
        # bring up the IB remote transport during init, which it does not survive here, so turn
        # off a transport this operator never uses.
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
        release events (comm stream done reading), then binds a completion event on the comm
        stream to the result. Returns (result, registered), where ``registered`` is False on a
        build without torch's work registry -- the C++ side has then already made the caller's
        stream wait (see csrc/work.h).

        The binding call sits OUTSIDE the stream context deliberately: that fallback wait has to
        land on the caller's stream, which is the current one only out here."""
        cur = torch.cuda.current_stream()
        ev_ready = torch.cuda.Event()
        ev_ready.record(cur)
        self._comm_stream.wait_event(ev_ready)
        with torch.cuda.stream(self._comm_stream):
            out = fn()
        for ev in releases:
            ev.record(self._comm_stream)
        registered = _C.register_stream_completion(out, self._comm_stream.cuda_stream)
        return out, registered

    def reserve(
        self,
        calls: Sequence[Mapping[str, object]],
        *,
        allow_growth: bool = False,
    ) -> None:
        """Pre-size the symmetric windows for the calls this group will make, then seal the pool.

        Each entry describes one intended call: ``tag``, ``shape`` (the 4D input shape), and
        optionally ``mode`` (default 0), ``dtype`` (default bfloat16), ``seq_splits`` and
        ``head_splits``. Give each tag the LARGEST shape it will ever see -- windows are matched
        by capacity, so that covers every smaller call on the same tag.

        This is an optimisation and a guard, not a requirement: the pool is one symmetric
        allocation taken at construction, and every window is an offset into it, so no call
        allocates. What sealing adds is that a shape drifting upward becomes an error rather
        than costing one window per growth (growth does not reclaim the offset it outgrew), and
        that ranks disagreeing about what they allocate are caught -- local offsets line up only
        while every rank hands them out in the same order.

        COLLECTIVE OVER THE WHOLE JOB: every rank must call this with the same entries in the
        same order, including ranks in a different group. Groups of equal size reserving the same
        shapes get the same window sizes, which is what keeps the allocation in step.

        Once sealed the groups may diverge -- different shapes, modes and call ORDER -- as long
        as every call fits a declared capacity. ``a2a_subgroup_divergent.py`` is the test, and
        docs/API.md records what divergent allocation does without a reserve.

        With ``allow_growth=False`` a later undeclared call raises instead of allocating; pass
        True for the old behaviour.
        """
        for call in calls:
            torch.ops.fast_ulysses.reserve(
                self._group,
                str(call["tag"]),
                list(call["shape"]),  # type: ignore[arg-type]
                int(call.get("mode", 0)),  # type: ignore[arg-type]
                call.get("dtype", torch.bfloat16),
                call.get("seq_splits"),
                call.get("head_splits"),
            )
        if not allow_growth:
            self._group.seal_pool()

    def barrier_epoch(self, tag: str) -> int:
        """TESTS: the tag's device-side handshake counter. 0 before the tag's first call.

        Synchronises the device to read it, so it is not for a hot path. It lets a test assert
        that a handshake ADVANCED rather than infer it from whether the data came out torn --
        torn data is a sufficient signal, not a necessary one.
        """
        return self._group.barrier_epoch(tag)

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
    ) -> torch.Tensor | CompletedHandle:
        """Async form of ``all_to_all_single_4d``, on the group's comm stream. Returns an
        ``AsyncCollectiveTensor`` wrapping the tensor the caller owns; same collective contract
        as the sync call, ``out`` / ``seq_splits`` / ``head_splits`` included.

        ``result.wait()`` returns the plain tensor -- and so does the first use of the result by
        any aten op, which waits BY ITSELF. Either way the wait is the caller's current stream
        waiting on the comm stream's completion event: GPU-side, no host block. A view op
        (``view``, ``reshape``, slicing) is the exception: it does not wait, it re-wraps.

        Wait on -- or use -- every result. torch's registry holds the completion until something
        pops it, so one that is simply dropped leaves an entry behind, and torch prints a count
        of the survivors at process exit.

        Ordering is per TAG, not per group: the barrier flags and epoch counter are keyed by
        tag (ulysses_group.cuh BarrierState), so an outstanding result must be waited before
        the next call with THAT tag, while a call on another tag may run concurrently on an
        unordered stream (tests/distributed/a2a_overlapping_barriers.py builds exactly that).

        No ``barrier`` parameter here: deferring the closing handshake would leave the copy-out
        reading the window before the peers' writes had landed. It lives on
        ``all_to_all_single_4d_borrowed_async``. Full contract: docs/API.md.
        """
        x, ev_free = self._stage_input(x.contiguous(), tag)
        y, registered = self._launch_on_comm_stream(
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
        # On the plain tensor, before wrapping: any aten op on the wrapper would wait first.
        y.record_stream(self._comm_stream)
        y.record_stream(torch.cuda.current_stream())
        return AsyncCollectiveTensor(y) if registered else CompletedHandle(y)

    def all_to_all_single_4d_borrowed_async(
        self,
        x: torch.Tensor,
        *,
        mode: int = 0,
        tag: str = "",
        barrier: bool = True,
        seq_splits: list[int] | None = None,
        head_splits: list[int] | None = None,
    ) -> torch.Tensor | CompletedHandle:
        """Async form of ``all_to_all_single_4d_borrowed``: an ``AsyncCollectiveTensor`` over the
        WINDOW VIEW, under the same unenforced rules, with "the stream that produced it" being
        the caller's stream from the wait onwards -- ``.wait()``, or the first aten op on the
        result, whichever comes first (``.clone()``, the documented way to keep a borrowed
        result, is one such op).

        The wait binds to that CALL, not to the window: each borrowed result is a fresh
        ``at::from_blob`` view with its own storage, which is what torch's registry keys on, so
        waiting still says nothing about whether a later call with this tag has overwritten the
        bytes. Unchanged from the sync form -- see ``all_to_all_single_4d_borrowed``.

        barrier=False defers only the CLOSING handshake to a later barrier=True call on the
        same stream, so several calls share one -- until then the deferred call's output view
        is not safe to read, only the barrier-carrying result's wait implies peers' writes
        arrived, and all ranks must use the identical barrier pattern. Full contract:
        docs/API.md.
        """
        x, ev_free = self._stage_input(x.contiguous(), tag)
        out, registered = self._launch_on_comm_stream(
            [ev_free],
            lambda: torch.ops.fast_ulysses.all_to_all_single_4d_borrowed(
                self._group, x, mode, tag, barrier, seq_splits, head_splits
            ),
        )
        return AsyncCollectiveTensor(out) if registered else CompletedHandle(out)

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
