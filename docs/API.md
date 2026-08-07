# API Reference

[English](API.md) · [中文](zh/API.md)

Everything is exposed from the top level: `from fast_ulysses import UlyssesGroup, CompletedHandle`.
Shape names: `b` batch, `d` head dim, `ws = world_size`; `s_local` / `n_local` are **this rank's**
sequence and head shard, `s_global = sum(seq_splits)`, `n_global = sum(head_splits)`. Uneven shards
are the general case — `s_local = s_global / ws` is the even special case.

## Collective hard constraints

**Violating any of these hangs the whole group.** Nothing raises, nothing times out.

- **Every rank must issue the same `(shape, mode, tag)` sequence**, counting sync and async,
  copying and borrowed calls alike — there is one sequence, not two. A mismatch forks the
  symmetric-heap windows (a new `tag`+capacity+dtype allocates collectively) and the barriers.
- **Within a tag the calls must stay ordered**, since the epoch protocol needs every rank to number
  a tag's handshakes identically: wait for an outstanding async result before the next call on that
  tag. **Across tags they need not be** — two tags may be in flight on unordered streams.
- **Construction, `reserve()` and `destroy()` are collective over the whole job**, not over
  `process_group`; under 2-D parallelism that means both sp groups at once.
- **A `barrier=False` pattern must be identical on every rank.**

## `UlyssesGroup(process_group=None, device=None, initial_pool_bytes=2<<30)`

| Parameter | Type | Meaning |
| --- | --- | --- |
| `process_group` | `ProcessGroup` or `None` | Bootstrap process group; `None` uses `dist.group.WORLD`. A subgroup is allowed if its ranks are **evenly strided** (they become an NVSHMEM strided team) — e.g. the sp slice `{0,2,4,6}` of a tp2 × sp4 mesh. A non-arithmetic rank list raises. |
| `device` | `torch.device` or `None` | This rank's CUDA device; `None` uses the current device. |
| `initial_pool_bytes` | `int` | The pool, default `2<<30` (2 GiB). Taken **in full** by one symmetric allocation at construction — committed, not a cap — and every collective's window is an offset into it. The heap is sized by the **first live** group; a later, larger request only warns, and destroying all groups lets the next one re-size. |

Every pair of ranks must be P2P-mappable (`nvshmem_ptr` non-null, following
`cudaDeviceCanAccessPeer`) — NVLink and PCIe alike, two-socket boxes included. An unreachable pair
is refused at construction, naming the pair; `fast-ulysses doctor` prints the matrix in advance.

## `reserve(calls, *, allow_growth=False) -> None`

Pre-size every symmetric window this process will use, then seal the pool. Each entry is a mapping
with `tag`, `shape` (the 4D **input** shape) and optionally `mode` (0), `dtype` (`bfloat16`),
`seq_splits`, `head_splits`. Windows match by capacity: give each tag the largest shape it will see.

```python
group.reserve([{"tag": "qkv", "shape": (b, s_local, n_global, d), "mode": 0},
               {"tag": "qkv", "shape": (b, s_global, n_local, d), "mode": 1}])
```

Once sealed, an **undeclared call raises** instead of allocating, so a shape drifting upward is an
error rather than an abandoned window per growth; `allow_growth=True` skips the seal. Same entries,
same order, on every rank — after that, groups may diverge as long as each call fits a capacity.

## Shapes, splits and tags

| mode | input `x` | output |
| --- | --- | --- |
| 0 — scatter heads, gather sequence | `(b, s_local, n_global, d)` | `(b, s_global, n_local, d)` |
| 1 — the inverse | `(b, s_global, n_local, d)` | `(b, s_local, n_global, d)` |

`seq_splits[p]` is rank p's sequence length and `head_splits[p]` its head count. Pass **both or
neither**, **identical on every rank**, matching the shape handed in. Neither means even shards and
the scattered axis must divide (`n_global % ws` for mode 0, `s_global % ws` for mode 1); both lets
shards differ arbitrarily, which is what lets a caller drop sequence padding. The window is sized
for the **largest** rank's output (the allocation is collective) and each result is a dense prefix.

A `tag` names one symmetric window (per `tag`+capacity+dtype, at its high-water capacity) plus its
handshake state — flag buffer and epoch counter, per tag, on the device (`BarrierState` in
`csrc/ulysses_group.cuh`). The window lives from the tag's first call to `destroy()` and every call
on the tag overwrites it, so concurrently-live borrowed results need distinct tags; copies do not.

## Raises

`RuntimeError`, from validation that runs **before the call's first barrier**, so a rejected
argument leaves no rank waiting on peers that did not reject it: `x` not 4D or not CUDA, dtype not
`float16`/`bfloat16`, `d * elem_size` not 16 B-aligned, `mode` not 0 or 1, `world_size` outside
`[1, 8]`; one of `seq_splits` / `head_splits` without the other, or splits contradicting `x`'s
shape; no splits and the scattered axis does not divide; `out` not contiguous CUDA, or its dtype or
shape not the output's; `x` or `out` overlapping the tag's window; a call over a sealed capacity.

## `all_to_all_single_4d(x, *, mode=0, tag="", out=None, seq_splits=None, head_splits=None) -> Tensor`

**The default.** Returns a tensor the caller owns, with no lifetime rules: it may outlive the next
call with this tag, be read on another stream, be handed to the allocator, or survive `destroy()`.
`x` is 4D CUDA `float16`/`bfloat16`, `.contiguous()` applied internally; `out` is an optional
preallocated destination, validated as above, and `None` allocates.

The transfer lands in the tag's window and one flat device-to-device copy moves it out on the
caller's stream behind the closing handshake — that copy is the price, reported as the `copy_out`
stage of `all_to_all_single_4d_timed`. There is one transport, always: pitched
`cudaMemcpy2D/3DAsync` into peers' window addresses from a host-side plan (`csrc/a2a_plan.cpp`),
with no launch config and no autotune, so first calls are collective-safe by construction. Figures:
[docs/BENCHMARK.md](BENCHMARK.md).

## `all_to_all_single_4d_borrowed(x, *, mode=0, tag="", seq_splits=None, head_splits=None) -> Tensor`

The same collective without the copy-out: **the result IS the tag's symmetric window.** Same shapes,
splits and collective constraints; no `out`, and no `barrier` — a deferred sync result would be an
unreadable view with nothing left to publish it. In exchange, a **lifetime contract that nothing in
this library enforces** — no check, no assertion, no debug mode:

- Valid **only until the next call carrying this tag**. That call's transfer writes the same bytes,
  and this result silently becomes that one.
- **Consume it on the stream that produced it**, before that next call. Reading it on another
  stream is yours to synchronise.
- **Do not read it after `destroy()`** — the memory is freed.
- `.clone()`, or any op producing a new tensor, is how you keep it.

Cross-rank safety **is** handled: no peer can overwrite this window until every rank has reached the
next call's opening barrier, and your reads are ordered ahead of it on your stream
(`tests/distributed/a2a_window_race.py`, `a2a_copy_out.py`). When in doubt use the copying form.

## `all_to_all_single_4d_async(x, *, mode=0, tag="", out=None, seq_splits=None, head_splits=None)`

Submits the copying collective to the group's high-priority comm stream and returns immediately,
wrapping the caller-owned tensor in an `AsyncCollectiveTensor`; same arguments and same collective
contract as the sync call. `result.wait()` returns the plain tensor, and so does the **first use of
the result by any aten op** — either way the caller's current stream waits on the comm stream's
completion event, GPU-side, and the host does not block. A **view op** does not wait, it re-wraps.

**Wait on, or use, every result.** A dropped one leaves its entry in torch's work registry, and its
CUDA event, behind; torch prints a count of the survivors at exit. `out=` is the one hole — reading
your own `out` never touches the registry, so read the returned wrapper. On a `libtorch_cpu` with no
`c10d::register_work` there is no registry to bind to, and both async forms return a
`CompletedHandle`: same `.wait()`, correct results, no overlap. A distinct type, so it is visible.

**Both async forms** stage the input on the caller's stream into a persistent per-`(tag, shape,
dtype)` buffer only the comm stream reads, so `x` is never retained cross-stream; that costs one
device copy per call and `tags × tensor size` resident. Mixing with sync calls: different tags only.

## `all_to_all_single_4d_borrowed_async(x, *, mode=0, tag="", barrier=True, seq_splits=None, head_splits=None)`

An `AsyncCollectiveTensor` over the **window view**, under the lifetime rules above, with "the
stream that produced it" being the caller's stream from the wait onwards. The wait binds to that
**call**, not to the window — each borrowed result is a fresh view with its own storage, which is
what the registry keys on — so it says nothing about a later call on this tag overwriting the bytes.

**Grouped handshake (`barrier=False`)**: every call OPENS with a handshake (writers wait for
readers), which is not optional; `barrier=False` defers only the CLOSING one, to a later
`barrier=True` call **on the same stream**, so several async calls — q, k, v of one layer, distinct
tags — share one closing handshake, removing N-1 of the 2N per group. Publication is by stream
order, not by tag, so a `barrier=False` result's view is not safe to read until the barrier-carrying
result is waited on. **All ranks must use the identical pattern.** The flag exists only on the
borrowed form: a copying call would copy the window out before the peers' writes had landed.

## `destroy() -> None`

Releases the symmetric-heap resources (drain comm stream, `dist.barrier`, destroy); all ranks must
call it together. Dropping a group without it leaks the heap with a warning: teardown is collective.

## Also on `UlyssesGroup`

| Entry point | Purpose |
| --- | --- |
| `all_to_all_single_4d_timed(x, *, mode=0, tag="", seq_splits=None, head_splits=None)` | The copying call, returning `(output, {barrier_in, transfer, barrier_out, copy_out} in ms)`. **Benchmark only**: reading the events synchronises the device. |
| `barrier_epoch(tag) -> int` | The tag's device-side handshake counter, 0 before the tag's first call. **Tests only**: reading it synchronises the device. |

## `make_group(process_group=None, device=None, initial_pool_bytes=2<<30, prefer="auto")`

Returns whichever group class is faster on this machine. Collective, like either constructor.

- `prefer="auto"` (default) — `TorchUlyssesGroup` when the group's GPUs span more than one CPU
  socket, `UlyssesGroup` otherwise. A socket layout that cannot be determined counts as "does not
  span", since that is the single-socket case.
- `prefer="fast"` / `prefer="torch"` — force one.

`result.fallback` is `True` for `TorchUlyssesGroup` and `False` for `UlyssesGroup`.

`spans_sockets(process_group=None) -> bool | None` is the same check on its own. It is collective:
each rank can only read its own device's NUMA node, so they are gathered. `None` means the kernel
did not report a node for at least one device.

## `TorchUlyssesGroup(process_group=None, device=None, initial_pool_bytes=...)`

The four collectives above, implemented with `torch.distributed`, for the one topology where that
is faster. Bit-exact with `UlyssesGroup` on every entry point and both shape families.

What relaxes: results are always owned, so the borrowed forms carry no lifetime rule; `tag` is
ignored, since nothing is reused between calls; the async forms complete before returning, so
there is no overlap to gain; `reserve()` and `destroy()` are no-ops. The collective call-sequence
contract still applies, because `torch.distributed` has its own.

# Environment variables

`UlyssesGroup.__init__` sets `NVSHMEM_SYMMETRIC_SIZE`, and `NVSHMEM_DISABLE_NVLS` /
`NVSHMEM_REMOTE_TRANSPORT` by `setdefault`, before NVSHMEM init. Those and the build variables are
documented in [docs/INSTALL.md](INSTALL.md).
