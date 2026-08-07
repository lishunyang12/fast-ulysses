# Benchmarks

All numbers below were produced under `scripts/exclusive.sh`, which refuses to start until the
requested GPUs are free, binds the run to exactly those GPUs, samples throughout, and prints
`EXCLUSIVE` or `CONTENDED`. A `CONTENDED` run is not a slow result, it is not a result. Numbers
from different machines or time windows are not compared against each other.

Measured on 4×H200 (NVLink), CUDA 13, PyTorch 2.13, NVSHMEM 3.4.5, bf16, `world_size=4`,
`mode=0`, medians over 25–30 iterations.

Shapes are the attention inputs of two real models, QKV packed into one collective so the last
dim is `3 * head_dim`. `s` includes a 227-token text tail, so it does not divide by the group
size — the case sequence padding exists to hide.

| label | s | heads | 3·head_dim | MB/rank |
|---|---|---|---|---|
| Wan 720p | 75827 | 40 | 384 | 582 |
| Wan 480p | 32987 | 40 | 384 | 253 |
| MiniMax-H3 | 38051 | 56 | 384 | 409 |

## Where the time goes

`BASE` is `torch.distributed`'s path (permute, `all_to_all_single`, permute). `raw` is
`all_to_all_single` alone — same bytes, no relayout, result in the wrong layout, so it is a
transport floor rather than an alternative.

| shape | perm_in | a2a | perm_out | BASE | barr_in | transfer | barr_out | OURS | raw | CE/raw | relayout% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Wan 720p | 0.707 | 1.315 | 0.764 | **2.786** | 0.011 | 1.133 | 0.012 | **1.156** | 1.343 | **1.19×** | 52.8% |
| Wan 480p | 0.316 | 0.619 | 0.336 | **1.272** | 0.013 | 0.507 | 0.010 | **0.530** | 0.651 | **1.28×** | 51.3% |
| MiniMax-H3 | 0.501 | 0.994 | 0.540 | **2.034** | 0.010 | 0.895 | 0.011 | **0.916** | 1.006 | **1.12×** | 51.1% |

Three things a total does not show:

- **Over half the baseline's time is relayout, not communication** (51–53%). Those permutes are
  SM work, competing with the compute the collective is supposed to hide behind. Our copies
  express the same relayout as source and destination strides, so it costs nothing.
- **The transfer alone beats a bare `all_to_all_single`** by 1.12–1.28× on identical bytes.
- **The barriers are not the bottleneck**: 21–23 µs for both, under 2% of a call.

Reproduce: `./scripts/exclusive.sh <gpus> -- torchrun --nproc_per_node=4 benchmark/bench_stages.py`

## Serialised copies, own share on the caller's stream

Giving every peer its own stream is the intuitive design and is measurably wrong. What made it
look right is that a naive serialisation also serialises this rank's own share, which crosses no
link.

| | Wan 720p | Wan 480p | MiniMax-H3 |
|---|---|---|---|
| one stream per peer | 2.273 | 1.018 | 1.683 |
| everything on one stream | 1.345 | 0.631 | 1.049 |
| **remote serialised, own share on the caller's stream** | **1.175** | **0.542** | **0.932** |

Concurrent copies contend for the same egress and each runs slower than it would with the link to
itself: serialising is worth 1.7×. Keeping the own share on the caller's stream recovers the
local/remote overlap the stream pool existed for, for another 12–14%, with no extra stream or
event. All three variants pass the correctness suite.

## Removing the sequence padding

Rounding a sequence up to a multiple of the group size and padding the tail keeps every rank the
same length, which is what lets the baseline stay on its flat path. The padded tokens then ride
through attention and through the collective on every layer of every step. Per-rank `seq_splits`
accepts shards differing by one token instead.

| shape | b | base padded | base unpadded | base cost | ours padded | ours unpadded | ours cost |
|---|---|---|---|---|---|---|---|
| Wan 720p | 1 | 2.786 | 3.011 | 1.08× | 1.172 | 1.172 | **1.00×** |
| Wan 480p | 1 | 1.274 | 1.371 | 1.08× | 0.539 | 0.542 | **1.01×** |
| MiniMax-H3 | 1 | 2.027 | 2.170 | 1.07× | 0.932 | 0.931 | **1.00×** |
| Wan 720p | 4 | 11.194 | 11.768 | 1.05× | 4.501 | 4.512 | **1.00×** |

The pad is at most `world_size - 1` tokens, so the byte counts are nearly identical and the ratio
is the number to read. Uneven is the general case in the plan and even is `seq_splits = [s/P] * P`,
so both take the same path at the same cost. The baseline cannot stay on its flat
`all_to_all_single` once shards differ at all — it needs split sizes, a per-peer reshape and a
`cat` — and pays 5–8% for a one-token difference.

Reproduce: `./scripts/exclusive.sh <gpus> -- torchrun --nproc_per_node=4 benchmark/bench_padding.py`

## Batch dimension

Device time scales linearly with `b`; host-side submission does not, because the batch is folded
into one `cudaMemcpy3DAsync` per peer instead of `b`.

| b | Wan 720p device | submit (µs) |
|---|---|---|
| 1 | 1.172 | 39.9 |
| 2 | 2.285 | 41.2 |
| 4 | 4.501 | 40.0 |

Only multi-row copies are fused: folding a single-row flat memcpy into a 3D copy puts it on the
strided path and is slower.

## Alternatives tried and not adopted

Kept here rather than in comments, so the code says what it does and this says why the
alternatives were dropped. Same machine and conditions as above unless noted.

| alternative | result |
|---|---|
| One stream per peer, and everything on one stream | See the table above; both are slower than remote-serialised with the own share on the caller's stream. |
| Sequential peer order instead of XOR-shift | XOR-shift pairs ranks without coordination; measured in the sibling NCCL implementation at ~14%, not re-run here. |
| `cudaMemcpy3DBatchAsync` | 0.82 ms, and 1.35 ms with `cudaMemcpyFlagPreferOverlapWithCompute`, against the plain `cudaMemcpy3DAsync` used instead. Also rejects the legacy default stream with "invalid argument". |
| Fusing single-row copies into 3D | 0.67 → 2.24 ms at `b=2`. Fusing is applied only to multi-row copies for this reason. |
| Contiguous per-sender segments instead of strided writes (mode 1) | Strided runs ~9% below contiguous (352.8 vs 389.7 GB/s in the sibling implementation). Not available here regardless: the window IS the returned tensor, so there is no local pass to interleave afterwards. |
| `cuStreamWriteValue64` / `cuStreamWaitValue64` barrier instead of the spin kernel | Concurrent-GEMM overlap fell from +34% to −28% (`WRITE_VALUE_DEFAULT`) and −15% (`NO_MEMORY_BARRIER`). The waiting form also needs `CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES`, which is 0 on much of the target hardware, while the spin kernel's inline PTX is available from sm_70 up. |

## What is not measured here

- **End-to-end model impact.** These are microbenchmarks — warm L2, no neighbours competing for
  bandwidth. Removing the padding saves attention work and memory this cannot see; the tables
  only show that the collective does not become more expensive.
- **Overlap with compute.** The zero-SM property is structural, not measured in this file.
- **Beyond `world_size = 8`,** or across nodes.
