# Design notes

[English](DESIGN.md) · [中文](zh/DESIGN.md)

Why the code is shaped the way it is, and what it rests on that is not guaranteed. Contracts are in
[API.md](API.md); numbers are in [BENCHMARK.md](BENCHMARK.md).

## The transfer

Peer windows are written with plain `cudaMemcpy2D/3DAsync` into addresses obtained from
`nvshmem_ptr`. That uses the copy engines and **no SMs**, which is the point: an SM-resident
collective cannot get a block slot while GEMMs hold every SM, and this one does not need one.

The sequence/head relayout is expressed as source and destination strides on those copies, so it
costs nothing beyond the transfer that had to happen anyway. That is why the baseline's two permute
kernels do not appear on our side at all.

All the addressing lives in `csrc/a2a_plan.cpp`, which has no CUDA and no NVSHMEM in it. The layout
contract can therefore be tested on a machine with no GPU (`tests/test_plan.py`), which is also the
only correctness check CI can run.

Uneven shards are the general case; even splits are `seq_splits = [s/P] * P`. There is one code
path, so there is one thing to get right.

## One symmetric allocation

The pool takes the whole symmetric heap in the constructor with a single `nvshmem_align`, and
`acquire()` only ever hands out offsets into it.

This is not tidiness. `nvshmem_align` is collective and synchronizes the CUDA stream internally,
while `barrier=False` deliberately leaves a spin barrier in flight. Allocating on the call path
therefore parks the host inside `nvshmem_align`, where it can no longer issue the publish its peers
are spinning for — and their hosts are parked in the same place. A circular wait.

Local offsets line up across ranks only because every rank hands them out in the same order, which
the SPMD call contract already requires. `reserve()` + `seal()` turn a violation of that order into
an error instead of one rank addressing another rank's window.

## The barrier

A one-block spin kernel, publishing with a release store and waiting with an acquire load, over a
device-resident epoch counter. The epoch is on the device rather than computed on the host so that
a CUDA-graph capture replays correctly — a host-computed epoch would bake a constant into the graph.

`cuStreamWriteValue64` / `cuStreamWaitValue64` would remove the last kernel launch from the path and
were tried. Two things against them: they measured worse under concurrent compute, which is the
regression this operator exists to avoid, and the waiting form needs a remote-write-flush device
attribute that much of the target hardware does not have. The spin kernel's inline PTX needs only
`sm_70`.

## What this rests on that is not documented

**A completed copy-engine write is visible at the destination by the time a later kernel's release
store announcing it arrives.** No vendor document says so:

- the CUDA API reference defines memcpy completion as a *host-side* property;
- the Programming Guide's cross-device ordering guarantee is scoped to the NULL stream and is
  withdrawn for async copies on a non-default stream;
- PTX scopes `.release` to "prior operations from the current thread", which a copy-engine transfer
  is not;
- neither NVSHMEM nor NCCL pairs a host-issued CE transfer with an SM release store; both keep the
  flag on the data's own path.

It holds in testing, which is evidence and not a guarantee.
`tests/distributed/a2a_ce_flag_ordering.py` tests it and
`tests/distributed/a2a_ce_fault_injection.py` is the negative control that keeps that test honest —
it arms the failure on every run, so the test cannot silently stop testing.

## Two NVSHMEM entry points are not the documented ones

`nvshmemx_hostlib_init_attr` is used instead of the inline `nvshmemx_init_attr`, and
`nvshmemx_hostlib_finalize` instead of `nvshmem_finalize`. The inline versions call
`nvshmemi_init_thread` / `nvshmemi_finalize`, which live only in the static `libnvshmem_device.a`.
Linking that clashes with the version node of torch's own bundled NVSHMEM and surfaces as
`undefined symbol: nvshmem_selected_device_transport`. The `hostlib_` entries are exported directly
by the host shared library; NVSHMEM's own Python unique-id path uses them too.

This is also why only the host library is linked and `CUDA_SEPARABLE_COMPILATION` is off: no
device-side `nvshmem_*` call exists in these kernels, so nothing needs the device library.

## Constraints that are stated but not enforced

**Live groups must partition the job.** Two enforcement attempts were written and removed rather
than left in place looking like protection:

1. A *local* check of the PE sets already built cannot see the violation — the divergence is present
   in the first construction, before any second group exists.
2. An *all-gather* of the PE sets cannot either. The gather is collective over the world, while only
   group members reach the constructor; the ranks that join no group never arrive, so the gather
   hangs in place of the split it was meant to protect.

A check every rank calls would work, so it cannot live in a constructor only members call. Not
built. `tests/distributed/a2a_overlapping_groups.py` demonstrates the failure and is deliberately
not registered, because running it hangs.

**A borrowed result stays valid only until the next call with that tag.** Nothing enforces it. What
*is* enforced is the narrower case where an input or `out` overlaps the window it is about to fill;
`check_window_aliasing` compares intervals, which also covers the round trip
`y = a2a_borrowed(x, tag="t"); a2a_borrowed(y, mode=1, tag="t")` — a shape a caller reaches for
naturally, and one that used to work by accident of an older pool key.

## The async result

`all_to_all_single_4d_async` returns an `AsyncCollectiveTensor` registered against torch's work
registry, so the first aten op on the result waits by itself. The registry keys on the output's
**storage**, not its address: each call's `at::from_blob` view carries a fresh storage, so a
registry entry belongs to that call and never to the window it aliases.

When the registry is unavailable in the linked libtorch (a build-time `nm` probe decides this, and
`build_info()["has_work_registry"]` reports it), the same functions return a handle with an explicit
`.wait()` instead. A result that is simply dropped leaves an entry behind; torch prints a count of
the survivors at process exit.

The sync collectives stay on the caller's stream rather than the comm stream. Routing them through
it would cost two event hops per call, which is comparable to the collective itself.
