# fast-ulysses

Minimal equal-split Ulysses all-to-all using PyTorch symmetric memory and direct CUDA peer writes.

Supported:

- one rank per GPU, with 1, 2, 4, or 8 GPUs;
- contiguous `[B, S, H, D]` FP16/BF16 tensors;
- equal splits and inference only;
- forward `[B, S_local, H_global, D] -> [B, S_global, H_local, D]`;
- reverse `[B, S_global, H_local, D] -> [B, S_local, H_global, D]`.

There is no varlen, uneven split, autograd, internal pool, async work wrapper, plan cache, CUDA Graph,
or release-wheel machinery.

## Install

```bash
source /workspace/sgl-env/bin/activate
FAST_ULYSSES_CUDA_ARCH=100 pip install -e . --no-build-isolation
```

The architecture is detected from the current GPU when `FAST_ULYSSES_CUDA_ARCH` is not set.

## Use

```python
from fast_ulysses import UlyssesGroup

group = UlyssesGroup()
output = group.allocate_output(x, mode=0)  # collective; allocate once
group.exchange(x, output, mode=0)
group.destroy()
```

On GPUs with native peer atomics, barriers stay on the selected CUDA stream. On PCIe systems the
payload is still direct P2P, but the wrapper uses blocking process-group barriers and stream
synchronization so it never waits in a persistent GPU spin kernel.

## Benchmark

`benchmark.py` checks results against NCCL before timing. It reports:

- `raw ms`: pre-packed NCCL `all_to_all_single`, communication only;
- `layout ms`: preallocated NCCL pack + communication + unpack;
- `fast ms`: direct P2P into the final layout;
- `vs raw` and `vs layout`: baseline latency divided by fast latency.

Every case runs untimed warmup iterations first. Ranks are aligned outside the timed region before
each iteration. Each iteration records the slowest rank; the table is the median across trials.

```bash
torchrun --standalone --nproc_per_node=8 benchmark.py \
  --seq-len 37824 --num-heads 56 --head-dim 128
```

`seq-len` is the global sequence length, not the per-rank length. The defaults are 10 warmup calls,
one measured call per trial, and the median of 20 trials.
