# fast-ulysses minimal

A small inference-only Ulysses all-to-all built from PyTorch symmetric memory and direct CUDA peer
copies. Each rank obtains every output window's remote base pointer and writes its shard directly.

The implementation deliberately supports only the common equal-split 4D case:

```text
mode 0: [B, S_local,  H_global, D] -> [B, S_global, H_local,  D]
mode 1: [B, S_global, H_local,  D] -> [B, S_local,  H_global, D]
```

FP16 and BF16 are supported. Sequence/heads must divide the group size. This branch is
inference-only and has no varlen, autograd, internal buffer pool, async result wrapper, lend mode,
or CUDA Graph integration.

```python
from fast_ulysses import UlyssesGroup

group = UlyssesGroup()
out = group.allocate_output(x, mode=0)  # collective; do this outside the loop
group.exchange(x, out, mode=0)
group.destroy()
```

`exchange(..., stream=stream)` submits on a caller-owned CUDA stream. On native-atomic fabrics
(normally NVLink/NVSwitch), barriers and copies remain device-side and the call may overlap compute.
The caller owns cross-stream event ordering and tensor lifetime. On PCIe, the wrapper uses blocking
process-group barriers around synchronized P2P copies to avoid GPU spin deadlocks; PCIe overlap is
not supported.

Build from source:

```bash
pip install -e . --no-build-isolation
```

Correctness and benchmark:

```bash
torchrun --standalone --nproc_per_node=8 test/distributed/correctness.py
torchrun --standalone --nproc_per_node=8 benchmark/bench_a2a.py
```

The benchmark defaults to global sequence length 37824, 56 heads, and head dimension 128. Override
them with `--seq-len`, `--num-heads`, and `--head-dim`; `--common-shapes` runs the bundled three-shape
set. The sequence length is global, not per rank.
