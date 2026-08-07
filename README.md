<div align="center">

# fast-ulysses

**Ulysses sequence-parallel all-to-all as a torch custom op, moved by the GPU copy engines over NVSHMEM symmetric memory.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/triple-mu/fast-ulysses/blob/master/LICENSE)

[English](https://github.com/triple-mu/fast-ulysses/blob/master/README.md) · [中文](https://github.com/triple-mu/fast-ulysses/blob/master/README.zh.md)

</div>

Ulysses sequence parallelism shards a long sequence across GPUs: one all-to-all before attention
trades the sequence shard for a head shard, a second trades it back. For long-sequence video DiT
workloads those two collectives are the critical communication.

This ships that 4D all-to-all as `torch.ops.fast_ulysses.all_to_all_single_4d`. The transfer is
pitched `cudaMemcpy2D/3DAsync` straight into peers' symmetric-heap addresses, so it **uses no SMs**
and runs while compute kernels hold every slot, and the sequence/head relayout is expressed as
copy strides instead of two separate permute kernels.

## What it does

- **Zero-SM transfer.** Remote copies are serialised on one stream; this rank's own share goes on
  the caller's stream.
- **Relayout for free.** The addressing is a host-side plan (`a2a_plan.cpp`), testable without a
  GPU, so there is no launch config and no autotune.
- **Copy by default, borrow explicitly.** `all_to_all_single_4d` returns a tensor the caller owns.
  `all_to_all_single_4d_borrowed` returns the symmetric window itself — one copy faster, valid only
  until the next call with that tag.
- **Async.** Returns an `AsyncCollectiveTensor`; the first aten op on it waits by itself.
  `barrier=False` lets several borrowed calls (one layer's q/k/v) share one closing handshake.
- **Uneven shards.** Per-rank `seq_splits` / `head_splits`, which is what lets a caller drop
  sequence padding. Even splits are the special case, not a separate path.
- **Reserve, then seal.** `reserve()` pre-sizes every window the process will use; afterwards an
  undeclared call raises instead of allocating in the middle of a collective.

Limits: single node, `world_size ∈ [1, 8]` including odd sizes; `float16` / `bfloat16`;
`d * elem_size` 16-byte aligned; every pair of GPUs in a group must be P2P-mappable
(`fast-ulysses doctor` prints the matrix, and an unreachable pair is refused at construction).

Fusion examples (QK RMSNorm + RoPE inside the scatter kernel) live on the
`examples/qk-norm-rope-fusion` branch.

## Results

8 GPUs, one exclusively allocated node per row, same container and the same `.so` for all four
architectures. `base` is `torch.distributed` permute + `all_to_all_single` + permute; `raw` is
`all_to_all_single` alone, without the relayout that makes its result usable. Wan 720p, bf16, ms.

| GPU | fabric | base | ours | vs base | transfer | vs raw |
|---|---|---|---|---|---|---|
| A100-SXM4-80GB | NVLink | 2.865 | **1.670** | **1.72×** | 1.227 | 1.12× |
| H200 | NVSwitch | 1.575 | **0.855** | **1.84×** | 0.683 | 1.22× |
| B200 | NVLink | 1.193 | **0.554** | **2.15×** | 0.402 | 1.22× |

Across three generations the shape is the same: 47–60% of the baseline is relayout that costs us
nothing, and the transfer alone beats a bare `all_to_all_single` by 1.12–1.37×. The collective
hides essentially completely under a concurrent GEMM chain (86% on B200, ~105% on A100). Dropping
the sequence padding is free (1.00×); the baseline pays 5–8% for the same change.

Full stage-by-stage tables, five machines: [docs/BENCHMARK.md](https://github.com/triple-mu/fast-ulysses/blob/master/docs/BENCHMARK.md).

## Where it does not win

**A group spanning two CPU sockets on a PCIe machine.** Inside one socket the operator is
1.4–2.2×, as on NVLink. Across a socket boundary it is about 0.62× of `torch.distributed` —
not because our transfer is slow, but because `all_to_all_single` does not use direct GPU P2P
there. It routes around the socket boundary through the InfiniBand NICs or through host shared
memory; we always write peer memory directly. Deny NCCL that bypass and we are 3.8–4.9× faster on
the same path. `fast-ulysses doctor` reports when a group spans sockets;
[docs/BENCHMARK.md](https://github.com/triple-mu/fast-ulysses/blob/master/docs/BENCHMARK.md#two-socket-pcie) has the measurements.

If you have to run on both kinds of machine, `make_group` picks for you:

```python
from fast_ulysses import make_group

group = make_group(process_group=dist.group.WORLD, initial_pool_bytes=1 << 30)
# group.fallback is True when it chose torch.distributed because the GPUs span sockets
```

It returns a `TorchUlyssesGroup` — the same four entry points on `torch.distributed` — when the
group spans sockets, and `UlyssesGroup` otherwise. `prefer="fast"` / `prefer="torch"` force it.
The two are bit-exact on every entry point and both shape families, so the caller keeps one code
path; the fallback simply has no overlap to gain and no lifetime rule on its results.

## Install

Requires **PyTorch 2.10+**, **CUDA 12.8+ or 13**, and sm80 / sm90 / sm100 / sm120. NVSHMEM 3.4.5+
comes from the `nvidia-nvshmem-cu1x` wheel torch already depends on, so there is nothing to install
separately.

```bash
pip install fast-ulysses                                  # newest torch, from PyPI
pip install -e . --no-build-isolation                     # from source, all four arches
FAST_ULYSSES_CUDA_ARCH=90 pip install -e . --no-build-isolation   # one arch, much faster
```

Wheels for other torch versions, and what to do when the import fails:
[docs/INSTALL.md](https://github.com/triple-mu/fast-ulysses/blob/master/docs/INSTALL.md).

## Quick start

`torchrun --nproc_per_node=2 example.py`:

```python
import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup

dist.init_process_group("nccl")
rank, ws = dist.get_rank(), dist.get_world_size()
lr = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(lr)

group = UlyssesGroup(process_group=dist.group.WORLD, initial_pool_bytes=1 << 30)

# mode 0: (b, s_local, n_global, d) -> (b, s_global, n_local, d)
b, s_local, d = 2, 16, 128
x = torch.randn(b, s_local, 4 * ws, d, dtype=torch.bfloat16, device=f"cuda:{lr}")

# Every rank must issue the same (shape, mode, tag) sequence.
out = group.all_to_all_single_4d(x, mode=0, tag="demo")
assert out.shape == (b, s_local * ws, 4, d)

group.destroy()
dist.destroy_process_group()
```

## API

| API | Summary |
| --- | --- |
| `UlyssesGroup(process_group=None, device=None, initial_pool_bytes=2<<30)` | Collective: NVSHMEM init + symmetric-heap pool. |
| `group.reserve(calls, *, allow_growth=False)` | Pre-size every window, then seal. |
| `group.all_to_all_single_4d(x, *, mode=0, tag="", out=None)` | The default. Returns a tensor the caller owns. |
| `group.all_to_all_single_4d_borrowed(x, *, mode=0, tag="")` | No copy-out; the result IS the window. |
| `group.all_to_all_single_4d_async(...)` | The default op on a high-priority comm stream. |
| `group.all_to_all_single_4d_borrowed_async(..., barrier=True)` | The borrowed op on that stream. |
| `group.destroy()` | Release symmetric-heap resources (collective). |
| `fast-ulysses doctor` | Build, devices, P2P matrix, socket layout. |

Why the code is shaped this way, and what it rests on that is not guaranteed: [docs/DESIGN.md](https://github.com/triple-mu/fast-ulysses/blob/master/docs/DESIGN.md).

Shapes, tag semantics, the barrier ordering contract, and the collective hard constraints —
violating the rank-uniform call sequence hangs the group: [docs/API.md](https://github.com/triple-mu/fast-ulysses/blob/master/docs/API.md).

## Testing

```bash
pytest                                                            # auto-skips below 2 GPUs
torchrun --nproc_per_node=8 tests/distributed/a2a_correctness.py  # one worker directly
```

Development setup: [docs/DEVELOP.md](https://github.com/triple-mu/fast-ulysses/blob/master/docs/DEVELOP.md).

## License

Apache-2.0. See [LICENSE](https://github.com/triple-mu/fast-ulysses/blob/master/LICENSE).
