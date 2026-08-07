<div align="center">

# fast-ulysses

**Ulysses sequence-parallel all-to-all as a torch custom op — NVSHMEM symmetric heap + GPU-to-GPU P2P, no NCCL on the hot path.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

</div>

## Why fast-ulysses?

Ulysses sequence parallelism (DeepSpeed-Ulysses) shards long sequences across GPUs: one all-to-all before attention trades the sequence shard for a head shard, a second one trades back. For long-sequence / video DiT workloads (Wan, HunyuanVideo, ...) these two all-to-alls are the critical communication.

`fast_ulysses` ships this 4D all-to-all as a standalone **torch custom op** (`torch.ops.fast_ulysses.all_to_all_single_4d`) that bypasses NCCL inside the node: the transfer stages through the NVSHMEM symmetric heap, data goes straight into peer memory over P2P, and a custom flag barrier synchronizes ranks — no host round-trip, no NCCL collective on the hot path.

## Features

- **One transfer path, on the copy engines**: pitched `cudaMemcpy2D/3DAsync` straight into peers' symmetric-heap addresses — **zero SM usage**, so the transfer runs at full link bandwidth while compute kernels hold every SM slot. The peer copies are serialised onto one stream and this rank's own share goes on the caller's stream.
- **No launch config, no autotune**: the addressing is a host-side plan, so first calls are collective-safe by construction ([docs/API.md](docs/API.md)).
- **Copy by default, borrow explicitly**: `all_to_all_single_4d` hands back a tensor the caller owns and has no lifetime rules; `all_to_all_single_4d_borrowed` hands back the symmetric window itself, which is faster by one copy of the output and valid only until the next call with that tag. The borrow is a separate function so it is visible at the call site ([docs/API.md](docs/API.md)).
- **Grouped handshakes**: `barrier=False` lets several async borrowed a2as (e.g. one layer's q/k/v) share one CLOSING handshake — each call still opens with its own ([docs/API.md](docs/API.md)).
- **Fusion examples** (QK RMSNorm + RoPE in the scatter kernel, standalone `rms_norm` / `rope` / `norm_rope`) live on the `examples/qk-norm-rope-fusion` branch.
- Single node, `world_size ∈ [1, 8]` (odd sizes included). The requirement is that every pair of
  GPUs in a group is P2P-mappable, not that they share an NVSwitch fabric; `fast-ulysses doctor`
  prints the matrix, and a group spanning an unreachable pair is refused at construction.
  **Correct on PCIe, but only FAST within one socket** — a group crossing a socket boundary is
  slower than `torch.distributed`, badly so at 8 ranks
  ([docs/BENCHMARK.md](docs/BENCHMARK.md#the-limit-crossing-a-socket-boundary)).
- **Reserve, then seal**: `reserve()` pre-sizes every window the process will use, after which an
  undeclared call is an error rather than a symmetric allocation in the middle of a collective
  ([docs/API.md](docs/API.md)).
- Even splits by default (`s` and `n` divisible by `world_size`), or per-rank `seq_splits` / `head_splits` for uneven shards — which is what lets a caller drop sequence padding ([docs/API.md](docs/API.md)). `mode=0` enters attention, `mode=1` leaves it.
- `float16` / `bfloat16`; `d * elem_size` 16-byte aligned.

## Installation

Requirements: **PyTorch 2.13+**, **CUDA 12 or 13**, and a GPU from sm80 (A100) / sm90 (H100/H200) / sm100 (B200) / sm120. **NVSHMEM 3.4.5+** is used, and torch already depends on it (`nvidia-nvshmem-cu13`), so no separate install is needed — the build finds torch's copy automatically; `NVSHMEM_HOME` overrides it.

```bash
pip install -e . --no-build-isolation                    # torch's NVSHMEM, all four arches
FAST_ULYSSES_CUDA_ARCH=90 pip install -e . --no-build-isolation   # one arch, much faster
```

- `NVSHMEM_HOME` (optional): install root containing `include/nvshmem.h`. Omit it to use torch's own copy.
- `FAST_ULYSSES_CUDA_ARCH` (optional): target compute capabilities, `;`-separated (default `80;90;100;120`).
- `--no-build-isolation`: link the already-installed PyTorch.

Docker setup, fabric-less nodes, and troubleshooting: [docs/INSTALL.md](docs/INSTALL.md).

## Quick Start

Save as `example.py` and run with `torchrun --nproc_per_node=2 example.py`:

```python
import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)

    group = UlyssesGroup(process_group=dist.group.WORLD, initial_pool_bytes=1 << 30)

    # mode0: input (b, s_local, n_global, d) -> output (b, s_global, n_local, d)
    b, s_local, d = 2, 16, 128
    n_global = 4 * ws  # must be divisible by world_size
    x = torch.randn(b, s_local, n_global, d, dtype=torch.bfloat16, device=dev)

    # All ranks must issue the same (shape, mode, tag) call sequence.
    out = group.all_to_all_single_4d(x, mode=0, tag="demo")
    assert out.shape == (b, s_local * ws, n_global // ws, d)
    if rank == 0:
        print(f"ws={ws} in={tuple(x.shape)} out={tuple(out.shape)}", flush=True)

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

## API at a Glance

| API | Summary |
| --- | --- |
| `UlyssesGroup(process_group=None, device=None, initial_pool_bytes=2<<30)` | Collective construction: NVSHMEM init + symmetric-heap pool. |
| `group.reserve(calls, *, allow_growth=False)` | Pre-size every window the process will use, then seal: an undeclared call raises instead of allocating mid-collective. |
| `group.all_to_all_single_4d(x, *, mode=0, tag="", out=None)` | **The default.** Uniform 4D all-to-all on the copy engines, returning a tensor the caller owns. |
| `group.all_to_all_single_4d_borrowed(x, *, mode=0, tag="")` | Same op without the copy-out: the result IS the symmetric window, valid only until the next call with that tag, and nothing enforces that. |
| `group.all_to_all_single_4d_async(...) -> AsyncCollectiveTensor` | The default op on a high-priority comm stream. `.wait()` returns the plain tensor, and so does the first aten op on it — which waits by itself, so forgetting is not a data race. |
| `group.all_to_all_single_4d_borrowed_async(..., barrier=True) -> AsyncCollectiveTensor` | The borrowed op on that stream; `barrier=False` groups calls under one closing handshake. |
| `group.destroy()` | Release symmetric-heap resources (collective). |
| `fast-ulysses doctor` | Report the build, the devices, and pairwise P2P reachability. Run it before blaming the operator. |

Shapes, tag semantics, the per-tag barrier ordering contract, and the **collective hard constraints** (violating the rank-uniform call sequence hangs the whole group): [docs/API.md](docs/API.md).

## Benchmarks

4×H200 NVLink, exclusive GPUs, `world_size=4`, bf16, medians in ms. `base` is
`torch.distributed` permute + `all_to_all_single` + permute; `raw` is `all_to_all_single`
alone, without the relayout that makes its result usable.

| shape | MB/rank | base | ours | vs base | transfer only | vs raw |
|---|---|---|---|---|---|---|
| Wan 720p (s=75827, h=40, d=384) | 582 | 2.786 | **1.156** | **2.41×** | 1.133 | **1.19×** |
| Wan 480p (s=32987, h=40, d=384) | 253 | 1.272 | **0.530** | **2.40×** | 0.507 | **1.28×** |
| MiniMax-H3 (s=38051, h=56, d=384) | 409 | 2.034 | **0.916** | **2.22×** | 0.895 | **1.12×** |

Two separable wins. **Over half the baseline's time is its two permutes** (51–53%), which is SM
work competing with the compute the collective is meant to hide behind; ours folds the relayout
into the copies' addressing, so it costs nothing. And **the transfer alone beats a bare
`all_to_all_single`** on the same bytes.

Per-call overheads, same runs: the two barriers total 21–23 µs (under 2%), and host-side
submission is ~40 µs and flat in the batch dimension.

**Dropping the sequence padding is free here.** With per-rank `seq_splits` the uneven path costs
the same as the even one (1.00× across three shapes and `b ∈ {1,2,4}`), because uneven is the
general case in the plan and even is a special case of it. The baseline pays 5–8% for the same
change, since shards of unequal length force it off its flat `all_to_all_single` path.

Measured again at 8 ranks on A100, H200 and B200 — one node each, same container, same venv, and
the same `.so` — the advantage holds across three generations at 1.66–2.15×, and the collective
hides essentially completely under a concurrent GEMM chain (86% on B200, ~105% on A100).

**It does not hold across a socket boundary.** Within one socket of a PCIe box the operator is
1.86×, as expected; spanning both sockets of that box it is 7× *slower* than `torch.distributed` at
8 ranks. The cause is the algorithm, not the link — see
[docs/BENCHMARK.md](docs/BENCHMARK.md#the-limit-crossing-a-socket-boundary).

Full stage-by-stage tables, all four generations: [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Testing

```bash
pytest                     # torchrun-wrapped multi-GPU suites; auto-skip below 2 GPUs
torchrun --nproc_per_node=8 tests/distributed/a2a_correctness.py   # direct worker invocation
```

Development setup (pre-commit, formatting, layout): [docs/DEVELOP.md](docs/DEVELOP.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
