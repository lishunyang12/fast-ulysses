# Benchmarks

[English](BENCHMARK.md) · [中文](zh/BENCHMARK.md)

Every number here was taken under `scripts/exclusive.sh`, which refuses to start until the
requested GPUs are free, binds the run to them, samples throughout, and prints `EXCLUSIVE` or
`CONTENDED`. A `CONTENDED` run is not a result. Numbers from different machines are not compared.

bf16, `mode=0`, medians over 25–30 iterations, milliseconds.

| machine | fabric | notes |
|---|---|---|
| 4×H200, 8×H200 | NVLink / NVSwitch | the default for sections that do not say otherwise |
| 8×A100-SXM4-80GB | NVLink | |
| 8×B200 | NVLink | |
| 8×RTX PRO 6000 | PCIe, 2 sockets | Intel Xeon 8480+, GPUs 4/4 across NUMA nodes |
| 8×RTX PRO 5000 | PCIe, 2 sockets | AMD EPYC 9575F, GPUs 4/4 across NUMA nodes |

Shapes are the attention inputs of two real models, QKV packed into one collective so the last dim
is `3 * head_dim`. `s` includes a 227-token text tail, so it does not divide by the group size.

| label | s | heads | 3·head_dim | MB/rank at ws=4 | at ws=8 |
|---|---|---|---|---|---|
| Wan 720p | 75827 | 40 | 384 | 582 | 291 |
| Wan 480p | 32987 | 40 | 384 | 253 | 127 |
| MiniMax-H3 | 38051 | 56 | 384 | 409 | 205 |

## Where the time goes

`BASE` is `torch.distributed`'s path (permute, `all_to_all_single`, permute). `raw` is
`all_to_all_single` alone — same bytes, no relayout, result in the wrong layout, so it is a
transport floor rather than an alternative. 4×H200.

| shape | perm_in | a2a | perm_out | BASE | barr_in | transfer | barr_out | OURS | raw | CE/raw | relayout% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Wan 720p | 0.707 | 1.315 | 0.764 | **2.786** | 0.011 | 1.133 | 0.012 | **1.156** | 1.343 | **1.19×** | 52.8% |
| Wan 480p | 0.316 | 0.619 | 0.336 | **1.272** | 0.013 | 0.507 | 0.010 | **0.530** | 0.651 | **1.28×** | 51.3% |
| MiniMax-H3 | 0.501 | 0.994 | 0.540 | **2.034** | 0.010 | 0.895 | 0.011 | **0.916** | 1.006 | **1.12×** | 51.1% |

Over half the baseline is relayout, not communication. Those permutes are SM work; our copies
express the same relayout as source and destination strides, so it costs nothing. Both barriers
together are 21–23 µs.

Reproduce: `./scripts/exclusive.sh <gpus> -- torchrun --nproc_per_node=4 benchmark/bench_stages.py`

## Five machines, one binary

One node per machine, 8 GPUs each, allocated exclusively and health-gated. Same container image,
same venv, and literally the same `_C…so` — it carries SASS for all four architectures, so nothing
was rebuilt between arms. `world_size=8`.

| GPU | fabric | shape | a2a | **BASE** | transfer | copy_out | **OURS** | vs BASE | raw | CE/raw |
|---|---|---|---|---|---|---|---|---|---|---|
| A100-SXM4-80GB | NVLink | Wan 720p | 1.354 | 2.865 | 1.227 | 0.336 | **1.670** | **1.72×** | 1.369 | 1.12× |
| A100-SXM4-80GB | NVLink | Wan 480p | 0.630 | 1.213 | 0.474 | 0.146 | **0.660** | **1.84×** | 0.650 | 1.37× |
| A100-SXM4-80GB | NVLink | MiniMax-H3 | 0.939 | 1.865 | 0.719 | 0.241 | **0.995** | **1.87×** | 0.988 | 1.37× |
| H200 | NVSwitch | Wan 720p | 0.831 | 1.575 | 0.683 | 0.144 | **0.855** | **1.84×** | 0.833 | 1.22× |
| H200 | NVSwitch | Wan 480p | 0.378 | 0.712 | 0.320 | 0.065 | **0.428** | **1.66×** | 0.394 | 1.23× |
| H200 | NVSwitch | MiniMax-H3 | 0.596 | 1.125 | 0.494 | 0.104 | **0.633** | **1.78×** | 0.613 | 1.24× |
| B200 | NVLink | Wan 720p | 0.478 | 1.193 | 0.402 | 0.099 | **0.554** | **2.15×** | 0.491 | 1.22× |
| B200 | NVLink | Wan 480p | 0.230 | 0.551 | 0.197 | 0.047 | **0.305** | **1.81×** | 0.245 | 1.25× |
| B200 | NVLink | MiniMax-H3 | 0.349 | 0.858 | 0.288 | 0.075 | **0.410** | **2.09×** | 0.363 | 1.26× |
| RTX PRO 6000 | PCIe, 2 sockets | Wan 720p | 14.080 | 14.970 | 22.390 | 0.358 | 23.754 | 0.63× | 13.957 | 0.62× |
| RTX PRO 6000 | PCIe, 2 sockets | MiniMax-H3 | 9.955 | 10.591 | 15.824 | 0.246 | 16.513 | 0.64× | 9.875 | 0.62× |
| RTX PRO 5000 | PCIe, 2 sockets | Wan 720p | 7.692 | 8.831 | 12.010 | 0.481 | 13.365 | 0.66× | 7.687 | 0.64× |
| RTX PRO 5000 | PCIe, 2 sockets | Wan 480p | 3.316 | 3.814 | 5.034 | 0.197 | 5.542 | 0.69× | 3.363 | 0.67× |
| RTX PRO 5000 | PCIe, 2 sockets | MiniMax-H3 | 5.443 | 6.252 | 8.230 | 0.332 | 8.941 | 0.70× | 5.395 | 0.66× |

On all three NVLink generations the shape of the win is the same: the relayout is 47–60% of the
baseline and costs us nothing, and the transfer beats a bare `all_to_all_single` by 1.12–1.37×. The
absolute numbers track the fabric — B200 moves the same bytes in a third of A100's time — but the
ratio barely moves.

The two PCIe machines are a group spanning both sockets, which is a different situation; see
"Two-socket PCIe" below.

> An earlier run on a different node of the same RTX PRO 6000 model reported `transfer = 119.2 ms`
> (CE/raw 0.14×) for Wan 720p at 8 ranks. It does not reproduce: a second node of that model gives
> 22.4 ms, and its 4-rank rows match the first node's. That row has been withdrawn.

### Where the benefit comes from

| GPU | relayout, share of BASE | copy_out, share of OURS | both barriers |
|---|---|---|---|
| A100-SXM4-80GB | 48–53% | 20–24% | 0.035–0.108 ms |
| H200 | 47% | 15–17% | 0.029–0.043 ms |
| B200 | 58–60% | 15–18% | 0.045–0.052 ms |
| RTX PRO 6000 / 5000 | 6–13% | 1.5–3.7% | 0.04–1.0 ms |

The relayout share **rises** from A100 to B200 even though B200 is faster at everything: the
permutes are SM work and did not speed up as much as the communication did, so the part we
eliminate grew. On the PCIe boxes it collapses because the communication dominates everything.

### Hiding the collective under compute

`hidden% = (serial − concurrent) / a2a_alone`, against a concurrent 3-GEMM chain shaped like
to_q/k/v. This is the claim the zero-SM design exists to support.

| GPU | gemm alone | a2a alone | hidden | samples |
|---|---|---|---|---|
| A100-SXM4-80GB | 5.6–6.2 ms | 0.54–0.59 ms | **99–109%, median 105%** | 5 runs |
| B200 | 0.94 ms | 0.249 ms | **86%** | 1 run |

Read >100% as "fully hidden": the metric can exceed 100 because the serial arrangement pays a
launch cost the concurrent one avoids (on A100, `serial − gemm_alone` is 0.83 ms against an
`a2a_alone` of 0.55 ms). B200's 86% is lower because the collective is large relative to the GEMM
there (a2a/gemm 0.27 against A100's 0.09) — less GEMM to hide under.

H200 is missing: `bench_ce` was added to the sweep after that arm ran and the queue did not free up
again. The RTX PRO 6000 figure came from the withdrawn run above and has been dropped with it.

## Two-socket PCIe

Two machines, both 8 GPUs split 4/4 across NUMA nodes, both fully P2P-reachable in
`nvidia-smi topo -p2p rw`. Same binary; only the GPU set changes. Wan 720p `transfer`:

| GPUs | RTX PRO 5000 (AMD) | RTX PRO 6000 (Intel) |
|---|---|---|
| 0,1,2,3 — one socket | 14.2 ms, **1.39×** | 8.4 ms, **2.20×** |
| 0,1,4,5 — split | 18.4 ms, 0.62× | 24.2 ms, 0.95× |
| 0–7 | 12.0 ms, 0.64× | 22.4 ms, 0.62× |

**Within one socket the operator behaves as it does on NVLink.** Across a socket boundary it is
about 0.62×, on both machines and both CPU vendors.

The reason is not that our transfer is slow — it is that **`all_to_all_single` does not use direct
GPU P2P across the socket boundary and we always do.** `NCCL_DEBUG=INFO` shows the bypass it picks:
`NET/IB/…/GDRDMA` through the InfiniBand NICs on the AMD machine, `SHM/direct/direct` through host
shared memory on the Intel machine, which has only two NICs and both on socket 0.

Taking the bypass away makes it explicit (AMD machine, Wan 720p):

| GPUs | `raw`, IB on | `raw`, IB off | ours, IB on | ours, IB off | CE/raw |
|---|---|---|---|---|---|
| 0,1,4,5 | 11.425 | **90.057** | 18.342 | 18.380 | 0.62× → **4.90×** |
| 0–7 | 7.663 | **46.836** | 12.372 | 12.380 | 0.62× → **3.78×** |

Our numbers do not move, because we never used the NIC. **On the same PCIe path we are 3.8–4.9×
faster than `all_to_all_single`**; it wins across sockets by not using that path.

The link measurements behind this, 64 MiB flat peer copies:

| path | AMD, 1 pair | AMD, 4 pairs | Intel, 1 pair | Intel, 4 pairs |
|---|---|---|---|---|
| same switch | 53.6 | 212.7 | 55.2 | 216.8 |
| same socket, across a host bridge | 53.0 | 106.9 | 54.9 | 216.1 |
| **across sockets** | 48.8 | **96.7** | 29.1 | **30.1** |
| across sockets, via pinned host | 53.4 | 106.7 | 47.9 | **61.4** |

Cross-socket P2P on the Intel machine does not scale with concurrency at all — four concurrent
pairs move no more than one. Routing through pinned host memory is 2× better there, and 10% better
on the AMD machine; that is what NCCL's SHM transport does.

Two other candidates were tested and refuted:

- **Pitched copies.** At the row widths this actually issues (1536–15360 B inside a 30720 B pitch),
  a pitched copy runs at 1.00–1.10× a flat one — never slower, on either path.
- **More streams.** One stream per peer is identical to one stream for all of them. A GPU's fan-out
  is capped by its own uplink (50 GB/s AMD, 37 GB/s Intel), not by contention between destinations.

The ceiling: for this access pattern the link is effectively half duplex — both directions together
aggregate the same as one (factor 1.96–1.99). At 8 ranks each GPU moves 254.6 MB out and as much
in, so 509 MB at 52 GB/s predicts 9.8 ms against a measured 12.0. **The operator is at ~82% of that
ceiling**; there is nothing left to schedule.

**Not fixed.** Matching NCCL across sockets means a shared-host-memory transport with a second
handshake — a new transport worth 0.62× → ~1.15× on the Intel machine, ~10% on the AMD one, and
nothing on NVLink. `fast-ulysses doctor` reports when a group spans sockets.

## Serialised copies, own share on the caller's stream

| | Wan 720p | Wan 480p | MiniMax-H3 |
|---|---|---|---|
| one stream per peer | 2.273 | 1.018 | 1.683 |
| everything on one stream | 1.345 | 0.631 | 1.049 |
| **remote serialised, own share on the caller's stream** | **1.175** | **0.542** | **0.932** |

On NVLink, concurrent copies contend for the same egress: serialising is worth 1.7×. Keeping the
own share on the caller's stream recovers the local/remote overlap for another 12–14%. All three
variants pass the correctness suite. (On PCIe the stream count makes no difference either way —
see above.)

## Removing the sequence padding

Rounding a sequence up to a multiple of the group size keeps every rank the same length, which is
what lets the baseline stay on its flat path; the padded tokens then ride through attention and the
collective on every layer of every step. Per-rank `seq_splits` accepts shards differing by one
token instead. 4×H200.

| shape | b | base padded | base unpadded | base cost | ours padded | ours unpadded | ours cost |
|---|---|---|---|---|---|---|---|
| Wan 720p | 1 | 2.786 | 3.011 | 1.08× | 1.172 | 1.172 | **1.00×** |
| Wan 480p | 1 | 1.274 | 1.371 | 1.08× | 0.539 | 0.542 | **1.01×** |
| MiniMax-H3 | 1 | 2.027 | 2.170 | 1.07× | 0.932 | 0.931 | **1.00×** |
| Wan 720p | 4 | 11.194 | 11.768 | 1.05× | 4.501 | 4.512 | **1.00×** |

The pad is at most `world_size - 1` tokens, so the ratio is the number to read. Uneven is the
general case in the plan and even is `seq_splits = [s/P] * P`, so both take the same path at the
same cost. The baseline cannot stay on flat `all_to_all_single` once shards differ at all — it
needs split sizes, a per-peer reshape and a `cat` — and pays 5–8% for one token. At 8 ranks the
baseline still pays 6–7% and ours stays at 0.95–0.99×.

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

| alternative | result |
|---|---|
| One stream per peer, and everything on one stream | Both slower than remote-serialised with the own share on the caller's stream (table above). |
| Sequential peer order instead of XOR-shift | XOR-shift pairs ranks without coordination; measured in the sibling NCCL implementation at ~14%, not re-run here. |
| `cudaMemcpy3DBatchAsync` | 0.82 ms, and 1.35 ms with `cudaMemcpyFlagPreferOverlapWithCompute`, against the plain `cudaMemcpy3DAsync` used instead. Also rejects the legacy default stream. |
| Fusing single-row copies into 3D | 0.67 → 2.24 ms at `b=2`. Fusion is applied only to multi-row copies for this reason. |
| Contiguous per-sender segments instead of strided writes | Strided runs ~9% below contiguous in the sibling implementation. Not available here regardless: the window IS the returned tensor, so there is no local pass to interleave afterwards. |
| `cuStreamWriteValue64` / `cuStreamWaitValue64` instead of the spin kernel | Concurrent-GEMM overlap fell from +34% to −28% and −15% depending on flags. The waiting form also needs `CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES`, which is 0 on much of the target hardware, while the spin kernel's inline PTX is available from sm_70 up. |
| Cross-socket staging through pinned host memory | 2× the link bandwidth on the Intel machine, 10% on the AMD one — but it needs the receiver to pull, so it is a new transport with a second handshake, not a scheduling change. Not built. |

## What is not measured here

- **End-to-end model impact.** These are microbenchmarks — warm L2, no neighbours competing for
  bandwidth. Removing the padding saves attention work and memory this cannot see.
- **Overlap with compute on H200.**
- **Beyond `world_size = 8`,** or across nodes.
