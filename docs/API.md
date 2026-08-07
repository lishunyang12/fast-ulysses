# API Reference

Everything is exposed from the top-level package:

```python
from fast_ulysses import UlyssesGroup, CompletedHandle
```

Shape conventions used throughout: `b` batch, `d` head dim, `ws = world_size`,
`s_local = s_global / ws`, `n_local = n_global / ws`.

## `UlyssesGroup(process_group=None, device=None, initial_pool_bytes=2<<30)`

| Parameter | Type | Meaning |
| --- | --- | --- |
| `process_group` | `torch.distributed.ProcessGroup` or `None` | Bootstrap process group; `None` uses `dist.group.WORLD`. A subgroup is allowed if its ranks are **evenly strided** (they become an NVSHMEM strided team) — e.g. the sp slice `{0,2,4,6}` of a tp2 × sp4 mesh. A non-arithmetic rank list raises. |
| `device` | `torch.device` or `None` | This rank's CUDA device; `None` uses the current device. |
| `initial_pool_bytes` | `int` | The pool, default `2<<30` (2 GiB). Taken **in full** by one symmetric allocation at construction — it is committed, not a cap — and every collective's window is an offset into it, one per `tag`+dtype. The heap is sized by the **first live** group; a later, larger request only warns, and destroying all groups lets the next one re-size. |

Construction broadcasts the NVSHMEM unique id over `torch.distributed` and is collective over the
**whole job**, not over `process_group`: the NVSHMEM bootstrap and the team split are all-PE
collectives. **Every rank must construct its group together** — under 2-D parallelism that means
both sp groups at once.

Every pair of ranks in a group must be P2P-mappable (`nvshmem_ptr` non-null, which follows
`cudaDeviceCanAccessPeer`). That covers NVLink and PCIe alike, including pairs on different root
complexes of a two-socket box. A pair that is not reachable is refused at construction, naming the
pair; `fast-ulysses doctor` prints the full matrix in advance.

## `reserve(calls, *, allow_growth=False) -> None`

Pre-size every symmetric window this process will use, then seal the pool.

Each entry is a mapping with `tag`, `shape` (the 4D **input** shape), and optionally `mode`
(default 0), `dtype` (default `bfloat16`), `seq_splits` and `head_splits`. Give each tag the
largest shape it will see — windows match by capacity, so that covers every smaller call on it.

```python
group.reserve([
    {"tag": "qkv", "shape": (b, s_local, n_global, d), "mode": 0},
    {"tag": "qkv", "shape": (b, s_global, n_local, d), "mode": 1},
])
```

This is an optimisation and a guard, not a requirement: the pool takes one symmetric allocation
at construction and hands out local offsets, so no call allocates. What sealing adds is that a
shape drifting upward becomes an error instead of costing one window per growth (growth does
not reclaim the offset it outgrew), and that ranks disagreeing about what they allocate are
caught — local offsets only line up while every rank hands them out in the same order.

**Collective over the whole job**: every rank calls it with the same entries in the same order,
including ranks in a different group. Once sealed, groups may then diverge freely — different
shapes, modes and call order — as long as every call fits a declared capacity.

`allow_growth=True` keeps the previous behaviour, where an undeclared call allocates.

> Why the pool is one allocation: `nvshmem_align` is collective and synchronizes the CUDA stream
> internally, while the flag barrier is a spin kernel that `barrier=False` deliberately leaves
> in flight. Allocating mid-run therefore parked the host inside `nvshmem_align`, where it could
> no longer issue the publish its peers were spinning for — and their hosts were parked in the
> same place. At `world_size=8` the deferred-group workload hung 12 of 12 runs; taking the
> allocation off the call path, 0 of 15.

## `all_to_all_single_4d(x, *, mode=0, tag="", out=None) -> Tensor`

Returns a tensor the **caller owns**. This is the default; use it unless a profile says the
copy-out below matters.

| Parameter | Type | Meaning |
| --- | --- | --- |
| `x` | `Tensor` | 4D CUDA tensor, `float16`/`bfloat16`; `.contiguous()` is applied internally. |
| `mode` | `int` | `0` (scatter heads / gather sequence) or `1` (its inverse). |
| `tag` | `str` | Labels the symmetric-heap staging buffer (one per `tag`+dtype, kept at its high-water capacity) and the barrier state that goes with it. **A tag's calls must stay ordered** — on one stream they are. Concurrently-live *results* do not need distinct tags on this form (each is copied out before the next call on that tag is issued); on the borrowed form below they do. |
| `out` | `Tensor` or `None` | Optional preallocated destination, to avoid an allocation per call. Validated: CUDA, contiguous, dtype equal to `x`'s, shape equal to the output shape below. `None` allocates with `at::empty`. |

**Copy-out.** The transfer lands in the tag's symmetric window; the call then copies the window
into `out` (or into the fresh allocation) with one flat device-to-device copy, on the caller's
stream behind the closing handshake, and returns that. Every rank's result is dense from the
window base, so there is nothing pitched about the copy — it moves exactly the result's bytes.

What that buys is the absence of rules: the returned tensor may outlive the next call with this
tag, be read on another stream, be handed to the allocator, or survive `destroy()`. What it costs
is a stage of `all_to_all_single_4d_timed` (`copy_out`) — measure it on your shapes;
`benchmark/bench_stages.py` prints it as a share of the call.

**Input / output shapes**

| mode | input `x` | output |
| --- | --- | --- |
| 0 | `(b, s_local, n_global, d)` | `(b, s_global, n_local, d)` |
| 1 | `(b, s_global, n_local, d)` | `(b, s_local, n_global, d)` |

**Uneven shards (`seq_splits` / `head_splits`)**

`seq_splits[p]` is rank p's local sequence length and `head_splits[p]` its head count. Pass
**both or neither** (one alone is an error), **identical on every rank**, and matching the shape
actually handed in — the entry point cross-checks them against `x` and raises rather than
producing a silently wrong result.

- Passing neither keeps the even behaviour above, including the divisibility requirement on the
  scattered axis (`n_global % ws` for mode 0, `s_global % ws` for mode 1).
- Passing them is what lets a caller drop sequence padding: with `seq_splits`, shards may differ
  in length (by one token in the unpadded case, or arbitrarily). `s_global` becomes
  `sum(seq_splits)` and the output length follows.
- The symmetric window is allocated for the **largest** rank's output, since the allocation is
  collective and every rank must ask for the same size; each rank's own result is a dense prefix
  of it. Splits are rank-uniform, so no rank has to communicate to compute that maximum.

**Transport**

There is exactly ONE transport, on the copy engines: pitched `cudaMemcpy2D/3DAsync` into peers'
window addresses, the remote peers serialised onto one transfer stream and this rank's own share
on the caller's stream, joined back with events, then the flag barrier. The addressing comes from
the host-side plan (`csrc/a2a_plan.cpp`).

No launch config, no kernel-path selection, no autotune, so first calls are collective-safe by
construction — a rank-local timing decision taken inside a collective would diverge the group.
The copy engines use no SMs, so the transfer keeps moving while compute holds every SM slot;
an SM-resident collective waits for the compute to drain instead. Figures:
[docs/BENCHMARK.md](BENCHMARK.md).

Cost of the transport: ~`world_size` memcpy launches per call, a few µs each — that is the floor
for small shapes.

Caveat, stated where it belongs: that a completed peer memcpy is VISIBLE at the destination when
the barrier's flag store arrives is not a documented CUDA guarantee, only measured behaviour —
the long note above the closing barrier in `csrc/bindings.cpp` and
`tests/distributed/a2a_ce_flag_ordering.py` are the whole of the evidence.

**Collective hard constraints (violating them hangs the whole group)**

- All ranks must call with the **same `(shape, mode, tag)` sequence** — a mismatch forks the
  symmetric-heap allocations (a new `tag+shape+dtype` allocates collectively) and the barriers.
- Sync and async calls **both count** in the sequence (sync runs on the caller's stream, async on
  the comm stream), and so do copying and borrowed calls — there is one sequence, not two.

**Barrier ordering contract (per TAG, not per group)**

The handshake state — flag buffer AND epoch counter — is per tag and lives on the device
(`BarrierState` in `csrc/ulysses_group.cuh`).

- **Within a tag the calls must be ordered**: wait for an outstanding async result before issuing
  the next call carrying that tag. The epoch protocol needs every rank to number a tag's
  handshakes identically, and only program order within the tag gives that.
- **Across tags they need not be**: an outstanding async call on one tag and a call on another tag
  may be in flight together on unordered streams. That is
  `tests/distributed/a2a_overlapping_barriers.py`, which builds exactly that shape (and which
  warms each tag serially first, because a tag's FIRST use also runs an all-PE allocation).

## `all_to_all_single_4d_borrowed(x, *, mode=0, tag="") -> Tensor`

The same collective without the copy-out: **the result IS the tag's symmetric window.** Same
parameters as above minus `out` (there is nothing to copy into), same shapes, same uneven splits,
same collective constraints.

In exchange it carries a lifetime contract that **nothing in this library enforces** — no check,
no assertion, no debug mode:

- The result is valid **only until the next call carrying this tag**. That call's transfer writes
  the same bytes, and this result silently becomes that one.
- **Consume it on the stream that produced it**, before that next call. Reading it on another
  stream is yours to synchronise.
- **Do not read it after `destroy()`** — the memory is freed.
- `.clone()`, or any op producing a new tensor, is how you keep it.

Cross-rank safety **is** handled: a peer cannot overwrite this window until every rank has reached
the next call's opening barrier, and your reads are ordered ahead of that barrier on your stream.
`tests/distributed/a2a_window_race.py` is the evidence for that, and
`tests/distributed/a2a_copy_out.py` is the evidence that the rules above are real — it lets a
borrowed result be clobbered on purpose and checks that a copied one is not.

It is a separate function rather than a flag on `all_to_all_single_4d` so that the borrow is
visible at the call site. When in doubt use the copying form, which has no rules.

No `barrier` parameter on this sync form: a deferred result would be an unreadable view with
nothing left to publish it.

## `all_to_all_single_4d_async(x, *, mode=0, tag="", out=None) -> AsyncCollectiveTensor`

Submits the copying collective to the group's high-priority comm stream and returns immediately,
wrapping the tensor the caller owns in a
`torch.distributed._functional_collectives.AsyncCollectiveTensor`. Constraints identical to the
sync call.

`result.wait()` returns the plain tensor — and so does the **first use of the result by any aten
op**, which waits by itself; that is why this is a wrapper subclass and not a handle of our own.
Either way the wait is the **caller's** current stream waiting on the comm stream's completion
event: GPU-side, the host does not block. A **view op** (`view`, `reshape`, slicing) is the
exception — it does not wait, it re-wraps.

**Wait on, or use, every result.** The completion is a `c10d::Work` in torch's registry, keyed by
the output's storage, and only a wait pops it; a result that is merely dropped leaves its entry
(and its CUDA event) behind, and torch prints a count of the survivors at process exit.

`out=` is the one hole in "you cannot forget": you already hold that tensor, and reading it
directly goes nowhere near the registry. Read the returned wrapper, not your `out`.

**Builds without `c10d::register_work`**: `CMakeLists.txt` probes `libtorch_cpu` for the symbol at
configure time. Without it there is no registry to bind to, so `csrc/work.cpp` makes the caller's
stream wait on the event before returning and both async calls hand back a `CompletedHandle`
instead — same `.wait()`, correct results, no overlap. A different type, so the degradation is
visible rather than silent.

The output is registered with both streams (`Tensor.record_stream`), because one of them allocated
it and the other wrote it, and the caching allocator must not recycle the block while the other is
still using it.

**Input staging** (both async forms): the input is copied on the caller's stream into a persistent
per-`(tag, shape, dtype)` staging buffer and the comm stream reads only the copy, so `x` is
never retained cross-stream (no `record_stream` — a host running many layers ahead would
otherwise pin every freed input and inflate the allocator's reserved pool by the whole
in-flight window). Costs one device copy per call and `tags × tensor size` resident memory.

**Mixing with sync calls** (both async forms): allowed on DIFFERENT tags, with nothing ordering
the two streams — see the per-tag contract above. On the SAME tag, wait for the outstanding
result first.

## `all_to_all_single_4d_borrowed_async(x, *, mode=0, tag="", barrier=True) -> AsyncCollectiveTensor`

The async borrowed form; the wrapped tensor is the **window view**, under the rules above, with
"the stream that produced it" being the caller's stream from the wait onwards — `.wait()`, or the
first aten op on the result, whichever comes first. `.clone()`, the documented way to keep a
borrowed result, is one such op.

The wait binds to that **call**, not to the window: every borrowed result is a fresh `from_blob`
view with its own storage, which is what the registry keys on, so waiting still says nothing about
whether a later call with this tag has already overwritten the bytes. That is the sync form's
lifetime contract, unchanged.

**Grouped handshake (`barrier=False`)**: every call OPENS with a handshake (writers wait for
readers, `csrc/bindings.cpp`) and that one is not optional; `barrier=False` defers only the
CLOSING one, to a later `barrier=True` call **on the same stream** — so several async calls (e.g.
q, k, v of one layer, distinct tags) share one closing handshake, removing N-1 of the 2N per
group. Publication is by stream order, not by tag: only waiting for the barrier-carrying result
implies peers' writes have landed, and waiting for a `barrier=False` result orders this rank's own
work only — its output view is not safe to read until then. All ranks must use the identical
barrier pattern.

This flag exists only on the borrowed form. A copying call with a deferred closing handshake
would copy the window out before the peers' writes had landed, so there is nothing to defer.

> This flag used to deadlock at `world_size=8`, for the reason recorded under `reserve()`. Fixed
> by making the pool one allocation. Two things worth keeping from chasing it: the hang rate was
> so timing-sensitive that the same workload measured 1/12, 5/12 and 12/12 on builds differing
> only in comments, so a single clean run proves nothing; and a hung run leaves eight ranks
> holding GPU memory that the harness does not always reap, after which everything on that node
> hangs earlier and looks far worse.

## `destroy() -> None`

Releases the symmetric-heap resources (drain comm stream, `dist.barrier`, destroy). All ranks must
call it together. Dropping a group without `destroy()` leaks the heap with a warning — the
teardown is collective, so it cannot run from GC.

---

# Environment variables

Set by `UlyssesGroup.__init__` (before NVSHMEM init):

| Variable | Value | Why |
| --- | --- | --- |
| `NVSHMEM_SYMMETRIC_SIZE` | `initial_pool_bytes` + 64 MiB | Heap reservation, set before NVSHMEM init. Sized above the pool because the pool is one allocation of the whole thing and NVSHMEM keeps its bookkeeping in the same heap. |
| `NVSHMEM_DISABLE_NVLS` | `1` (setdefault) | P2P direct writes don't need NVLS; its multicast heap mapping segfaults on some nodes. |
| `NVSHMEM_REMOTE_TRANSPORT` | `none` (setdefault) | Single-node op; the IB remote transport segfaults NVSHMEM init on IB-equipped nodes. |

Read by the library / build / tests:

| Variable | Where | Meaning |
| --- | --- | --- |
| `FAST_ULYSSES_CUDA_ARCH` | build (`setup.py`) | Target compute capabilities, `;`-separated. Default `80;90;100;120`. |
| `FAST_ULYSSES_CMAKE_ARGS` | build (`setup.py`) | Extra CMake `-D` flags (see docs/INSTALL.md Troubleshooting). |
| `FAST_ULYSSES_TEST_NPROC` | `tests/test_multigpu.py` | Overrides the torchrun process count (e.g. odd world sizes). |
