"""torchrun worker: is the copy-engine payload visible when the flag announcing it arrives?

    torchrun --nproc_per_node=8 tests/distributed/a2a_ce_flag_ordering.py

A call writes the payload into the peers' windows with ``cudaMemcpy2DAsync`` -- copy engines --
and then announces itself with a ``st.release.sys`` store from the one-block barrier kernel on the
same stream. The reader spins on that flag with ``ld.acquire.sys`` and then reads the payload.

Two different engines write to the same remote memory. Stream order says the copy has COMPLETED
before the kernel launches, but "completed" is a statement about the source; whether it means the
bytes are visible at the DESTINATION is the question. Carried over from the reference
implementation, which loaded the same question: the CUDA API reference defines a memcpy's
completion purely as a host-side property; the Programming Guide's one cross-device ordering
guarantee (3.4.2.1) is scoped to the NULL stream and to when commands START, and the next sentence
withdraws it for an async copy in a non-default stream; PTX 8.5's ``.release`` covers "prior
operations from the current thread" and ``.sys`` scope is a set of THREADS -- a copy-engine
transfer is neither, so the barrier kernel's release does not cover the payload at all. Neither
NVSHMEM's nor NCCL's CE paths use this shape: all of them keep the peer-visible flag on the data's
path (``cuStreamWriteValue64``, or a ``cudaMemcpyAsync`` of the flag on the transfer stream).
So a pass is EVIDENCE for this machine and this shape, not a proof. The documented-safe change is
to write the flag with a ``cudaMemcpyAsync`` on the transfer stream instead of from the kernel.

If the ordering does not hold, a reader that is ALREADY WAITING when the flag lands sees the
previous call's payload in part of the buffer. The worker maximises exactly that:

  * BORROWED results -- this extension always returns the symmetric-heap view, and the check reads
    it directly. Never ``.clone()`` it here: a device-to-device copy between the barrier and the
    first read is time the writes could use to drain, which would hide the very thing being
    tested.
  * A LARGE payload, so the copy engine is still busy when the flag is issued.
  * SKEW IN THE TRANSFER. rank 0 is held back by a ballast GEMM chain, so its CE copies start
    late, while every other rank is already spinning in the CLOSING barrier and reads the instant
    rank 0's flag arrives.
  * A distinct constant per iteration, so a stale byte is unmistakable rather than plausible. It
    CYCLES in 1..128 rather than being the iteration number, because bfloat16 has 8 significant
    bits: ``float(i)`` is exact only up to 256 and above that adjacent iterations collide on one
    bf16 value, which is precisely the comparison a tear has to survive. (The reference worker
    used ``float(i)`` over 300 iterations and is blind from 257 on; it failed at iteration 2, so
    it never showed.)

WHY THE PRE-CALL BALLAST IS NOT BLIND HERE, AND WHEN IT WILL BECOME SO. The reference's first
version of this test skewed the ranks the same way and was blind: it passed even with the closing
barrier deleted, because its call OPENS with a barrier that re-aligns everyone before any data
moves. It had to skew the TRANSFER instead, with uneven shards -- which this extension cannot do
from Python: ``A2APlan`` carries per-rank splits, but ``check_uniform_args`` builds even ones only
and no entry point exposes them. Rebuild this worker on uneven shards once they are reachable.
Meanwhile a pre-call ballast does reach the transfer here, because ``all_to_all_single_4d`` has NO
opening barrier: it is copies, then ``fast_barrier``. That is the same missing barrier
tests/distributed/a2a_window_race.py probes -- so if an opening barrier is ever added, THIS WORKER
GOES BLIND. Re-run the negative control after any barrier change.

READING A FAILURE: the missing opening barrier means a tear here has TWO possible causes, and the
constants the log prints tell them apart. A flag-ordering violation reads the residue that was in
the window BEFORE this call's writes became visible -- the PREVIOUS iteration's constant, the
``[i-1, i]`` mixture the reference measured with its closing barrier deleted. A window race (a
peer's NEXT call overwriting us while we read) reads the FOLLOWING iteration's constant. Only the
first is this worker's subject; the second is a2a_window_race.py's and would be a finding against
that gap, not against the copy-engine ordering.

Deviation from the reference worker: it breaks out of the loop on the first tear. Here the loop
runs to the end, because a rank that leaves early stops issuing collectives and its peers hang in
``fast_barrier`` -- a 600 s timeout instead of a reported failure.

NEGATIVE CONTROL: delete ``group->fast_barrier(stream, tag)`` in fast_ulysses/csrc/bindings.cpp
and rebuild; the reference implementation's equivalent control failed at iteration 2 with 62.9M
stale elements reading [i-1, i]. Without a rebuild, the same control is one line here: replace the
``group.all_to_all_single_4d(x, mode=0, tag="ord")`` call below with
``group.all_to_all_single_4d_async(x, mode=0, tag="ord", barrier=False).wait()`` -- a
``barrier=False`` handle's wait orders this rank's own work only, so the reads become
unsynchronised and must report stale constants within a few iterations. If the worker still
passes with either control applied, it is testing nothing and a pass means nothing.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup


def log(msg: str) -> None:
    print(f"[rank {dist.get_rank()}] {msg}", flush=True)


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    group = UlyssesGroup(process_group=dist.group.WORLD, initial_pool_bytes=2 << 30)

    # ~200 MB per rank at ws=4 (~400 MB at ws=8): the copy engines are still draining when the
    # barrier kernel publishes the flag.
    b, s_local, d = 1, 8192, 384
    x = torch.empty((b, s_local, 8 * ws, d), dtype=torch.bfloat16, device=dev)
    mb = x.numel() * x.element_size() / 1e6
    ballast = torch.randn((4096, 4096), dtype=torch.bfloat16, device=dev)

    # Warm the tag: the first use allocates and collectively registers the symmetric buffer, which
    # serialises the ranks and would hide the skew this worker depends on.
    x.fill_(0.0)
    group.all_to_all_single_4d(x, mode=0, tag="ord")
    torch.cuda.synchronize()
    dist.barrier()

    iters, stale, first_bad = 300, 0, 0
    for i in range(1, iters + 1):
        v = float(1 + i % 128)  # bf16-exact and distinct from every neighbouring call; see above
        x.fill_(v)

        # rank 0's copies start late; everyone else is already spinning in the closing barrier.
        if rank == 0:
            for _ in range(8):
                ballast = ballast @ ballast * 1e-4

        y = group.all_to_all_single_4d(x, mode=0, tag="ord")
        # Enqueued directly behind the barrier on this stream, reading the window itself. The host
        # sync inside .item() happens after this comparison has already read it, so it cannot mask
        # a stale read.
        bad = int((y != v).sum().item())
        if bad:
            stale += bad
            if not first_bad:
                first_bad = i
                seen = torch.unique(y.float()).tolist()[:6]
                log(f"iteration {i}: {bad} elements are not {v}; saw {seen}")

    if stale:
        log(
            f"FAILURE: copy-engine payload was not visible when the flag arrived -- {stale} "
            f"stale elements, first at iteration {first_bad}"
        )
    else:
        log(f"{iters} skewed calls, {mb:.0f} MB per call on this rank, stayed coherent")

    verdict = torch.tensor([int(stale > 0)], device=dev)
    dist.all_reduce(verdict)
    if rank == 0:
        print("CE_FLAG_ORDER " + ("PASS" if verdict.item() == 0 else "FAIL"), flush=True)
    group.destroy()
    dist.destroy_process_group()
    if verdict.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
