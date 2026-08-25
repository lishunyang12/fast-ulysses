# MiniMax H3 packed-flat test plan

This plan measures whether local packing followed by flat peer copies repairs the pitched-copy
collapse seen on PCIe, and whether that operator result survives MiniMax H3 integration.

## Target configuration

The first target is the validated four-card RTX PRO 5000 layout:

- four 72 GiB Blackwell GPUs on one NUMA node;
- TP2 x Ulysses2, so two Ulysses groups communicate concurrently;
- physical GPU order `0,2,1,3` on the reference node, making the strided Ulysses groups physical
  pairs `(0,1)` and `(2,3)`;
- MiniMax H3 FL2VA, BF16, cuDNN attention, 1344x768, 5 seconds, seed 1101;
- no CPU or layerwise offload, VAE patch parallelism four.

Always inspect `nvidia-smi topo -m`. Override `GPU_IDS` and `NUMA_NODE` instead of copying the
reference IDs to a different host.

## Comparisons

| name | implementation | purpose |
| --- | --- | --- |
| `nccl` | permute + `all_to_all_single` + permute | production baseline |
| `pitched-owned` | pitched peer copies plus flat copy-out | PCIe control |
| `packed-owned` | local pack + flat peer copies + local mode-1 unpack | PCIe fallback |
| `auto-owned` | first-shape autotune plus owned outputs | topology-aware candidate |

`pitched-zero` and `auto-zero` remain explicit diagnostics, not default E2E candidates. On the
reference PCIe H3 run, `pitched-zero` entered all four persistent Q/K/V/O buffers but then made no
forward progress; the following TP all-reduce timed out after 602 seconds. The operator-only loop
did not expose this interaction, so no zero-copy production speedup may be claimed until an E2E
screen completes and matches the NCCL output.

The H3 block benchmark uses TP-local heads. H3 has 56 model heads; TP2 leaves 28 heads in each
Ulysses2 group. It issues three independent `[B,S/U,28,128]` mode-0 calls for Q/K/V and one
`[B,S,14,128]` mode-1 call for O. The older `d=384` fused-QKV row is useful for transport
decomposition but is not an end-to-end predictor for vLLM-Omni.

## Phases

### 1. Topology and fabric ceiling

Archive GPU PCI bus IDs, NUMA distances, `nvidia-smi topo -m`, and flat peer-copy bandwidth. Run
the two physical Ulysses pairs separately, then both groups concurrently. This distinguishes a
bad link from collective scheduling or PCIe-switch contention.

### 2. Operator decomposition

For each physical pair record:

- standard permute + NCCL + permute;
- raw NCCL with prepared layout;
- local pack;
- flat peer copies without barrier;
- production packed mode 0, owned and zero-copy output;
- production packed mode 1.

Then run the TP2 x Ulysses2 H3 block benchmark five times. It reports slowest-process p50, p95,
and p99 plus the projected 50-block x 50-step communication total.

### 3. End to end

Restart the server for every backend. Keep model commit, prompt, shape, seed, steps, attention
backend, parallelism, and compile mode identical. Exclude two warmups. Record three measured
requests, stage headers, GPU memory/utilization samples, server logs, and decoded video/audio
FrameMD5.

The auto variant measures pitched and packed on the first mode-0 and mode-1 shapes, compares the
slowest rank, and keeps pitched unless packed is at least 5% faster. Startup autotuning is absorbed
by the warmups and must not be included in measured request latency. An explicit zero-copy run
assigns independent persistent symmetric outputs to Q, K, V, and O and is a correctness/liveness
diagnostic until the E2E hang above is resolved.

`RUN_LEVEL=screen` uses five steps to reject broken or losing paths. `RUN_LEVEL=full` uses 50
steps for the reportable result. Do not publish the screen result as a production speedup.

### 4. Profile

After the screen passes, collect one two-step Nsight or torch-profiler trace per backend. Attribute
time to input pack, peer copies, barriers, mode-1 unpack/copy-out, NCCL kernels, and layout kernels.
The trace must explain the wall-clock change; latency alone is insufficient.

## Acceptance gates

Packed-flat proceeds to the full E2E run only when all of these hold:

1. Every distributed round trip is exact.
2. Flat peer copies recover at least 70% of the pair's flat-link ceiling.
3. `packed_owned_block` beats `nccl_block` by at least 10% in at least four of five exclusive runs.
4. p95 does not regress by more than 10% relative to packed p50.
5. The server log confirms the requested backend and auto selection; an explicit zero-copy
   diagnostic must also confirm every allocation. Silent NCCL fallback is a hard failure.
6. Decoded video and audio FrameMD5 match the NCCL baseline.
7. Full-run denoise and E2E latency both improve; a VAE-only or startup-only change is not a
   communication result.

## One-command runner

From the fast-ulysses checkout:

```bash
WORK_ROOT=/lustre/raplab/client/sylarl/minimax-h3-native \
GPU_IDS=0,2,1,3 NUMA_NODE=0 \
bash benchmark/h3_packing/run_pro5000_suite.sh all
```

The default is the five-step screen. Run the reportable 50-step E2E after it passes:

```bash
WORK_ROOT=/lustre/raplab/client/sylarl/minimax-h3-native \
GPU_IDS=0,2,1,3 NUMA_NODE=0 RUN_LEVEL=full \
bash benchmark/h3_packing/run_pro5000_suite.sh e2e
```

Results are written under `WORK_ROOT/results/h3-packing-<UTC timestamp>`. Set `RESULT_ROOT` to an
explicit directory when setup, microbench, and E2E are launched as separate scheduler jobs.
