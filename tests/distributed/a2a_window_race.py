"""torchrun worker: does a peer's NEXT call overwrite our window while we are still reading it?

    torchrun --nproc_per_node=8 tests/distributed/a2a_window_race.py

Every result is a view of the tag's single symmetric-heap buffer, so the caller reads exactly the
memory the peers write. The closing barrier of call N (bindings.cpp: CE copies, then
``fast_barrier``) proves every peer's call-N writes have LANDED. It proves nothing about every
peer having FINISHED READING its own window for call N -- so a fast peer's call N+1 transfer can
start writing into our window while our call-N read is still in flight.

WHAT THIS EXTENSION HAS TODAY: only that closing barrier. The reference implementation
(custom_nccl_op) also OPENS every call with a barrier, before it writes anything -- writers wait
for readers -- and that barrier sits at the START of the next call rather than at the end of this
one precisely because a borrowed result is read by the caller, at a time the operator never sees.
fast_ulysses has no opening barrier, so this worker is a live PROBE of that gap, not a regression
test of a working feature. If it reports a torn window, the finding is that
``all_to_all_single_4d`` needs a second ``fast_barrier`` before ``launch_a2a_ce``.

The adversarial timing is the whole worker; the assertion is trivial:
  * rank 0 is the SLOW READER. The ballast sits between the call and the read, exactly where a
    real caller's compute sits, so rank 0 is still holding the window while its peers -- released
    by the very same barrier -- have already run ahead into call i+1 and started copying into it.
  * every rank fills its input with the same constant ``i``, so a correct output is uniformly
    ``i`` and a torn one contains two different constants: unmistakable, not plausible noise.
  * the per-iteration check is a device-side compare plus a host-only ``.item()``. NOTHING
    re-aligns the ranks inside the loop; adding a ``dist.barrier()`` or a ``torch.cuda`` group
    sync there makes this worker blind -- it would then pass on a build with no barriers at all.

Deviation from the reference worker: it breaks out of the loop on the first tear. Here the loop
runs to the end, because a rank that leaves early stops issuing collectives and its peers hang in
``fast_barrier`` -- a 600 s timeout instead of a reported failure.

NEGATIVE CONTROL: delete ``group->fast_barrier(stream, tag)`` in fast_ulysses/csrc/bindings.cpp
and rebuild. With no handshake at all, iteration 1 or 2 must report a large number of elements
holding a neighbouring call's constant (the log prints the distinct values it saw). If it still
passes with that line gone, the timing above has stopped being adversarial on this machine and
the worker is worthless as written -- fix the worker before trusting a pass.
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
    group = UlyssesGroup(process_group=dist.group.WORLD, initial_pool_bytes=1 << 30)

    # mode 1 (gather heads / scatter sequence): the output is one flat window and the check is a
    # single pass over all of it -- the widest window for a peer to land in. Big enough that the
    # read itself takes real time.
    b, s_local, n_local, d = 1, 4096, 8, 128
    x = torch.empty((b, s_local * ws, n_local, d), dtype=torch.bfloat16, device=dev)
    ballast = torch.randn((4096, 4096), dtype=torch.bfloat16, device=dev)

    # Warm the tag: the first use allocates and collectively registers the symmetric buffer, which
    # serialises the ranks and would hide the skew this worker depends on.
    x.fill_(0.0)
    group.all_to_all_single_4d(x, mode=1, tag="race")
    torch.cuda.synchronize()
    dist.barrier()

    iters, torn, first_bad = 40, 0, 0
    for i in range(1, iters + 1):
        x.fill_(float(i))
        y = group.all_to_all_single_4d(x, mode=1, tag="race")

        # Rank 0 keeps holding the borrowed window while its peers race into call i+1, whose
        # copies target this very buffer. Alternating keeps the ranks oscillating rather than
        # settling into a steady lag.
        if rank == 0 and i % 2 == 0:
            for _ in range(8):
                ballast = ballast @ ballast * 1e-4

        # Device-side check; .item() syncs this rank's host only, so the ranks stay skewed.
        bad = int((y != float(i)).sum().item())
        if bad:
            torn += bad
            if not first_bad:
                first_bad = i
                seen = torch.unique(y.float()).tolist()[:6]
                log(f"iteration {i}: {bad} elements are not {i}; saw {seen}")

    if torn:
        log(
            f"FAILURE: window torn -- {torn} elements carried a neighbouring call's value, "
            f"first at iteration {first_bad}"
        )
    else:
        log(f"{iters} skewed same-tag iterations stayed coherent")

    verdict = torch.tensor([int(torn > 0)], device=dev)
    dist.all_reduce(verdict)
    if rank == 0:
        print("WINDOW_RACE " + ("PASS" if verdict.item() == 0 else "FAIL"), flush=True)
    group.destroy()
    dist.destroy_process_group()
    if verdict.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
