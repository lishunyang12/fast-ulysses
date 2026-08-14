# fast-ulysses

Minimal equal-split Ulysses all-to-all. The caller sees contiguous `[B, S, H, D]` in and out and
never permutes: the relayout is folded into the transfer, so neither side runs a pack or an
unpack kernel. On the 8-GPU host below that is 1.04-1.18x NCCL's bare `all_to_all_single` and
1.21-1.36x NCCL with the two permutes it needs, per `benchmark_report.md`.

Supported:

- one rank per GPU, with 1, 2, 4, or 8 GPUs;
- contiguous `[B, S, H, D]` FP16/BF16 tensors;
- equal splits and inference only;
- batch size 1 on the 8-GPU mlx5 path;
- forward `[B, S_local, H_global, D] -> [B, S_global, H_local, D]`;
- reverse `[B, S_global, H_local, D] -> [B, S_local, H_global, D]`.

There is no varlen, uneven split, autograd, internal pool, async work wrapper, plan cache, CUDA Graph,
or release-wheel machinery.

## Install

```bash
FAST_ULYSSES_CUDA_ARCH=120 pip install -e . --no-build-isolation
```

The architecture is detected from the current GPU when `FAST_ULYSSES_CUDA_ARCH` is not set.
The build also links the system `libibverbs` and `libmlx5` libraries.

## Use

```python
from fast_ulysses import UlyssesGroup

group = UlyssesGroup()
output = group.allocate_output(x, mode=0)  # collective; allocate once
group.exchange(x, output, mode=0)
group.release_output(output)               # collective; every rank in step
group.destroy()
```

`backend()` names the transport. `p2p` writes into the peers' symmetric-memory windows with the
copy engines. A peer copy carries flat runs only: the relayout is a device-local copy, staged and
pipelined against the sends, because a strided copy across a link runs at 60% of a flat one while
a device-local strided copy is unaffected.

`mlx5` is chosen on an 8-GPU host with a per-GPU NIC. Direct P2P across a socket boundary does
not scale with concurrency there -- eight ranks pushing at once get 19.9 GB/s each on those links
against 47.8 GB/s within a PCIe switch -- so the far half goes over the NICs instead. Those
writes use interleaved MKeys, and the NIC gathers or scatters the strided `[S,H,D]` slices
itself, so no pack or unpack is needed on either side. The closest NIC is selected from sysfs.

Set `FAST_ULYSSES_DISABLE_RDMA=1` to use CUDA P2P only. To override NIC discovery, set all eight
rank-local devices explicitly, for example:

```bash
export FAST_ULYSSES_NICS=mlx5_2,mlx5_3,mlx5_0,mlx5_1,mlx5_6,mlx5_7,mlx5_4,mlx5_5
```

## Benchmark

`benchmark.py` checks results against NCCL before timing. It reports:

- `raw`: pre-packed NCCL `all_to_all_single`, communication only;
- `layout`: preallocated NCCL pack + communication + unpack;
- `fast`: direct P2P into the final layout;
- `GB/s`: per-rank remote-payload throughput, equivalent to NCCL bus bandwidth for all-to-all;
- `vs raw` and `vs layout`: baseline latency divided by fast latency.

For `N` ranks, NCCL algorithm bandwidth is `bus GB/s * N / (N - 1)`, and aggregate remote
throughput is `bus GB/s * N`. The Markdown report includes both values for raw NCCL.

Every case runs untimed warmup iterations first. Ranks are aligned outside the timed region before
each iteration. Each iteration records the slowest rank; the table is the median across trials.

```bash
torchrun --standalone --nproc_per_node=8 benchmark.py \
  --seq-len 37824 --num-heads 56 --head-dim 128 \
  --report benchmark_report.md
```

`seq-len` is the global sequence length, not the per-rank length. The defaults are 10 warmup calls,
one measured call per trial, and the median of 20 trials.

## Test

```bash
torchrun --standalone --nproc_per_node=8 test_correctness.py
FAST_ULYSSES_DISABLE_RDMA=1 torchrun --standalone --nproc_per_node=4 test_correctness.py
```

Shapes and dtypes against the NCCL reference, the rejection paths, and 400 back-to-back calls
with one rank skewed. That last check arms itself: the same pattern runs once over raw peer
copies with no barrier and has to tear. A run whose control stays clean reports BLIND.
