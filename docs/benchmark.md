# Benchmark

`benchmark/bench_a2a.py` compares raw NCCL, NCCL with layout conversion, and the minimal direct-P2P
path in both directions. Timings are max-rank wall-clock latency, then the median across trials.

The default global shape is `(37824, 56, 128)` in BF16; the sequence length is divided across
ranks. Use `--seq-len`, `--num-heads`, and `--head-dim` to override it, or `--common-shapes` to run
`(37824, 56, 128)`, `(75600, 40, 128)`, and `(32760, 40, 128)`.
