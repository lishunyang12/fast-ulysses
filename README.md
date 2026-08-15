# fast-ulysses

Minimal equal-split Ulysses all-to-all with no layout, pack, unpack, or staging tensors.

Supported:

- one rank per GPU, with 1, 2, 4, or 8 GPUs;
- contiguous `[B, S, H, D]` FP16/BF16 tensors;
- equal splits and inference only;
- batch size 1 on the 8-GPU mlx5 path, where `heads * head_dim * itemsize` summed over all
  ranks must also fit 65535 bytes -- the MKey stride field. Larger shapes are refused;
  `FAST_ULYSSES_DISABLE_RDMA=1` runs them on the CUDA P2P backend;
- mode 0 `[B, S_local, H_global, D] -> [B, S_global, H_local, D]`;
- mode 1 `[B, S_global, H_local, D] -> [B, S_local, H_global, D]`.

There is no varlen, uneven split, autograd, async work wrapper, plan cache, CUDA Graph,
or release-wheel machinery.

## Install

```bash
FAST_ULYSSES_CUDA_ARCH=100 python -m pip install -e .
```

Both editable entry points are supported in the active environment:

```bash
python -m pip install -e .
python setup.py develop
```

The architecture is detected with `nvidia-smi` when `FAST_ULYSSES_CUDA_ARCH` is not set. The build
locates Torch CMake files directly in the active virtual environment, so PEP 517 build isolation
does not install or import a second copy of PyTorch.
It uses up to 32 parallel jobs by default; override that with `CMAKE_BUILD_PARALLEL_LEVEL` or
`FAST_ULYSSES_BUILD_JOBS`. If `ccache` is on `PATH`, it is enabled automatically for both C++ and
CUDA. `FAST_ULYSSES_CCACHE=/path/to/ccache` selects it explicitly, while
`FAST_ULYSSES_CCACHE=0` disables it.
The build also links the system `libibverbs` and `libmlx5` libraries.

For example:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=32 FAST_ULYSSES_CUDA_ARCH=100 python -m pip install -e .
```

## Use

```python
from fast_ulysses import UlyssesGroup

group = UlyssesGroup()
output = group.all_to_all_4d(x, mode=0)
group.destroy()
```

The first call for each `(mode, shape, dtype)` collectively creates a registered output workspace;
later calls reuse it automatically. The returned workspace is overwritten by the next call with
the same key. `allocate_output` and an explicit `out` remain available only when two results of
the same geometry must stay live at once. `destroy()` releases all registered workspaces; there is
no per-output release step.

On the supported 8-GPU PCIe host, same-socket transfers use CUDA IPC pointers. Cross-socket
transfers use mlx5 interleaved MKeys: the NIC gathers or scatters the strided `[S,H,D]` slices
directly, so the application tensor layout never changes. The closest NIC is selected from sysfs.

Set `FAST_ULYSSES_DISABLE_RDMA=1` to use CUDA P2P only. To override NIC discovery, set all eight
rank-local devices explicitly, for example:

```bash
export FAST_ULYSSES_NICS=mlx5_2,mlx5_3,mlx5_0,mlx5_1,mlx5_6,mlx5_7,mlx5_4,mlx5_5
```

## Test

`test_correctness.py` runs under torchrun. It covers every supported shape and dtype against the
NCCL reference, the rejection paths, and back-to-back calls with one rank per quad deliberately
skewed. That last check is armed: the same pattern runs once over raw peer copies with no barrier
at all, and must tear. A run whose control stays clean prints `BLIND` and fails, because it proved
nothing.

```bash
torchrun --standalone --nproc_per_node=8 test_correctness.py
FAST_ULYSSES_DISABLE_RDMA=1 torchrun --standalone --nproc_per_node=8 test_correctness.py
```

Run both: the two backends synchronise differently, and only the second one exercises batch > 1.

## Benchmark

`benchmark.py` checks results against NCCL before timing. It reports:

- `raw`: pre-packed NCCL `all_to_all_single`, communication only;
- `layout`: preallocated NCCL pack + communication + unpack;
- `fast`: direct P2P into the final layout through the automatic output pool;
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
