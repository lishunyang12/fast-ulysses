# Benchmarks

All numbers below were produced under `scripts/exclusive.sh`, which refuses to start until the
requested GPUs are free, binds the run to exactly those GPUs, samples throughout, and prints
`EXCLUSIVE` or `CONTENDED`. A `CONTENDED` run is not a slow result, it is not a result. Numbers
from different machines or time windows are not compared against each other.

Two machines appear below and their numbers are NOT comparable: with more ranks each holds a
smaller shard, so `MB/rank` differs. Every section says which it is.

- **4×H200** (NVLink), CUDA 13, PyTorch 2.13, NVSHMEM 3.4.5, bf16, `mode=0`, medians over
  25–30 iterations. This is the default for every section that does not say otherwise.
- **8×H200** (NVSwitch, one exclusively allocated node), CUDA 13.3, PyTorch 2.13.0+cu130,
  NVSHMEM 3.4.5 (torch's own wheel), same shapes and iteration counts.

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

## The same two, at 8 ranks

One whole H200 node, allocated exclusively, health-gated. Each rank holds half the shard it holds
at 4 ranks, so `MB/rank` halves and these rows do not belong in the tables above.

| shape | MB/rank | BASE | OURS | vs BASE | transfer | raw | CE/raw | relayout% | copyout% |
|---|---|---|---|---|---|---|---|---|---|
| Wan 720p | 291 | 1.575 | **0.855** | **1.84×** | 0.683 | 0.833 | **1.22×** | 47.3% | 16.8% |
| Wan 480p | 127 | 0.712 | **0.428** | **1.66×** | 0.320 | 0.394 | **1.23×** | 46.9% | 15.2% |
| MiniMax-H3 | 205 | 1.125 | **0.633** | **1.78×** | 0.494 | 0.613 | **1.24×** | 47.0% | 16.4% |

The advantage over the baseline is smaller than at 4 ranks (1.66–1.84× against 2.22–2.41×) and
the advantage over bare `all_to_all_single` is steadier (1.22–1.24× against 1.12–1.28×). Both
follow from the same thing: half the bytes per rank, twice the peers, so the fixed per-call costs
weigh more and the relayout the baseline pays weighs slightly less (47% here against 51–53%).

Padding, same run:

| shape | pad | base pad→unpad | ours pad→unpad | ours vs base |
|---|---|---|---|---|
| Wan 720p | 5 | 1.06× | **0.99×** | 1.77× |
| Wan 480p | 5 | 1.07× | **0.95×** | 1.59× |
| MiniMax-H3 | 5 | 1.07× | **0.99×** | 1.74× |

Dropping the padding stays free — at 8 ranks it is marginally *cheaper*, which is what the ratio
being at or just under 1.00 means: the pad is 5 tokens, so this is the measurement floor rather
than a real gain. The baseline still pays 6–7%.

Reproduce: `./scripts/exclusive.sh <gpus> -- torchrun --nproc_per_node=8 benchmark/bench_stages.py`
(and `bench_padding.py`)

## Four generations, one binary

One the cluster node per generation, 8 GPUs each, allocated exclusively and health-gated. The only
thing that changes between rows is the GPU and its fabric: same container image
(`nvcr.io/nvidia/pytorch:26.07-py3`), same venv (PyTorch 2.13.0+cu130, NVSHMEM from torch's own
wheel), and literally the same `_C…so` — it carries SASS for all four architectures, so nothing was
rebuilt between arms.

`world_size=8`, bf16, `mode=0`, medians over 25–30 iterations, ms.

| GPU | fabric | shape | perm_in | a2a | perm_out | **BASE** | barr_in | transfer | barr_out | copy_out | **OURS** | vs BASE | raw | CE/raw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A100-SXM4-80GB | NVLink | Wan 720p | 0.740 | 1.354 | 0.771 | 2.865 | 0.027 | 1.227 | 0.081 | 0.336 | **1.670** | **1.72×** | 1.369 | 1.12× |
| A100-SXM4-80GB | NVLink | Wan 480p | 0.292 | 0.630 | 0.292 | 1.213 | 0.020 | 0.474 | 0.019 | 0.146 | **0.660** | **1.84×** | 0.650 | 1.37× |
| A100-SXM4-80GB | NVLink | MiniMax-H3 | 0.457 | 0.939 | 0.469 | 1.865 | 0.020 | 0.719 | 0.015 | 0.241 | **0.995** | **1.87×** | 0.988 | 1.37× |
| H200 | NVSwitch | Wan 720p | 0.359 | 0.831 | 0.385 | 1.575 | 0.009 | 0.683 | 0.020 | 0.144 | **0.855** | **1.84×** | 0.833 | 1.22× |
| H200 | NVSwitch | Wan 480p | 0.162 | 0.378 | 0.172 | 0.712 | 0.011 | 0.320 | 0.032 | 0.065 | **0.428** | **1.66×** | 0.394 | 1.23× |
| H200 | NVSwitch | MiniMax-H3 | 0.255 | 0.596 | 0.274 | 1.125 | 0.012 | 0.494 | 0.023 | 0.104 | **0.633** | **1.78×** | 0.613 | 1.24× |
| B200 | NVLink | Wan 720p | 0.346 | 0.478 | 0.369 | 1.193 | 0.013 | 0.402 | 0.039 | 0.099 | **0.554** | **2.15×** | 0.491 | 1.22× |
| B200 | NVLink | Wan 480p | 0.157 | 0.230 | 0.164 | 0.551 | 0.011 | 0.197 | 0.050 | 0.047 | **0.305** | **1.81×** | 0.245 | 1.25× |
| B200 | NVLink | MiniMax-H3 | 0.247 | 0.349 | 0.262 | 0.858 | 0.014 | 0.288 | 0.034 | 0.075 | **0.410** | **2.09×** | 0.363 | 1.26× |
| RTX PRO 6000 | **PCIe, 2 sockets** | Wan 720p | 0.452 | 16.675 | 0.440 | 17.567 | 0.076 | 119.216 | 0.203 | 0.358 | **119.853** | **0.15×** | 16.700 | **0.14×** |
| RTX PRO 6000 | **PCIe, 2 sockets** | Wan 480p | 0.203 | 7.793 | 0.190 | 8.186 | 0.033 | 52.447 | 0.007 | 0.141 | **52.628** | **0.16×** | 7.313 | **0.14×** |
| RTX PRO 6000 | **PCIe, 2 sockets** | MiniMax-H3 | 0.321 | 12.178 | 0.316 | 12.816 | 0.046 | 83.279 | 0.101 | 0.246 | **83.672** | **0.15×** | 11.779 | **0.14×** |

On all three NVLink generations the shape of the win is the same and it does not depend on the
generation: the relayout is 47–60% of the baseline and costs us nothing, and the transfer itself
beats a bare `all_to_all_single` by 1.12–1.37×. The absolute numbers track the fabric — B200 moves
the same bytes in a third of A100's time — but the *ratio* does not move much, which is the point:
the advantage is structural, not a property of one machine.

The last three rows are the exception, and they are dealt with below.

### Where the benefit comes from, per generation

| GPU | relayout, share of BASE | copy_out, share of OURS | both barriers | host submit |
|---|---|---|---|---|
| A100-SXM4-80GB | 48–53% | 20–24% | 0.035–0.108 ms | 87 µs |
| H200 | 47% | 15–17% | 0.029–0.043 ms | 42 µs |
| B200 | 58–60% | 15–18% | 0.045–0.052 ms | 47 µs |
| RTX PRO 6000 | 5% | 0.3% | 0.04–0.28 ms | 42–50 µs |

The relayout share **rises** from A100 to B200 (48–53% → 58–60%) even though B200 is faster at
everything. The permutes are SM work and did not speed up as much as the communication did, so the
part we eliminate grew. On the PCIe box the share collapses to 5% for the opposite reason — the
communication there is so slow that nothing else matters.

### Hiding the collective under compute

`hidden% = (serial − concurrent) / a2a_alone`, against a concurrent 3-GEMM chain shaped like
to_q/k/v. This is the claim the zero-SM design exists to support, and it had never been measured.

| GPU | gemm alone | a2a alone | hidden | samples |
|---|---|---|---|---|
| A100-SXM4-80GB | 5.6–6.2 ms | 0.54–0.59 ms | **99–109%, median 105%** | 5 runs |
| B200 | 0.94 ms | 0.249 ms | **86%** | 1 run |
| RTX PRO 6000 | 3.5 ms | 37.1 ms | **5%** | 1 run |

Read >100% as "fully hidden": the metric can exceed 100 because the serial arrangement pays a
launch/sync cost the concurrent one avoids (on A100, `serial − gemm_alone` is 0.83 ms against an
`a2a_alone` of 0.55 ms). It is not 105% of the transfer being hidden.

B200's 86% is lower than A100's because the collective is large relative to the GEMM there
(a2a/gemm = 0.27 against A100's 0.09) — there is simply less GEMM to hide under. The PCIe row's 5%
is a consequence of the transfer being 37 ms against a 3.5 ms GEMM; nothing can hide that.

H200 is missing from this table: `bench_ce` was added to the sweep after the H200 arm had already
run, and the H200 queue did not free up again in time.

## The limit: crossing a socket boundary

The PCIe rows above are not a PCIe result, they are a **cross-socket** result, and the difference
matters. Same node, same binary, only the GPU set changes:

| GPUs | topology between them | transfer | CE/raw |
|---|---|---|---|
| 0,1,2,3 | all on NUMA 0 (`PIX`/`NODE`) | 8.4 ms | **1.86×** |
| 0,1,4,5 | two per socket (`SYS` pairs present) | 24.3 ms | 0.96× |
| 0–7 | four per socket | 119.2 ms | **0.14×** |

**Within one socket the operator behaves exactly as it does on NVLink** — 1.86×, in the same band
as the three NVLink generations. It is crossing the socket that breaks it, and the breakage is
super-linear: parity at 4 ranks, 7× slower at 8.

Two explanations were tested on that node and **refuted**:

- *The PCIe link is slow.* It is not: 54.7 GB/s intra-socket and 28.7 GB/s cross-socket, measured
  with a flat `cudaMemcpy` between two devices. Fully serialising all seven peer copies at those
  rates predicts ~6.5 ms, not 119.
- *The pitched (strided) copies are the problem.* They are not: at the row width this shape
  actually uses (3840 B with a 30720 B pitch), a pitched copy runs at 54.78 GB/s against a flat
  copy's 54.6 — no measurable difference.

What is left, and fits the super-linear shape, is the **algorithm**: a direct point-to-point
all-to-all puts every rank's traffic for every remote peer onto the wire simultaneously, so at 8
ranks there are 16 flows crossing one shared inter-socket link. `all_to_all_single` is topology
aware and routes to minimise crossings. On a fabric where every peer is equidistant that advantage
does not exist and we win; on one where peer distance varies by 2× and the far path is shared, it
dominates.

**Not fixed, and not attempted.** A topology-aware plan — ordering or grouping the peer copies by
distance, or staging cross-socket traffic through one rank per socket — is the obvious direction,
but nothing here has been designed or measured. Until then: on a multi-socket PCIe box, either keep
the group inside one socket, or use `torch.distributed`.

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
- **Overlap with compute on H200.** Measured on A100, B200 and RTX PRO 6000 (see above); the
  H200 arm predates that measurement.
- **Beyond `world_size = 8`,** or across nodes.
