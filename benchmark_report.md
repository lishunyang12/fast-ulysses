# fast-ulysses benchmark report

- world_size: 8
- dtype: bfloat16
- backend: pcie
- NCCL_P2P_LEVEL: <unset>
- warmup: 10 calls/case
- measurement: 1 call(s)/trial, 20 trials, slowest rank then median

All bandwidths are decimal GB/s. `bus` counts only bytes sent to remote ranks; `aggregate` is the sum across all ranks.

| Shape | Dir | Raw NCCL ms | NCCL alg GB/s | NCCL bus GB/s | NCCL aggregate GB/s | NCCL + layout ms | Layout GB/s | Fast ms | Fast GB/s | Raw / fast | Layout / fast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 37824,56,128 | fwd | 1.793 | 37.80 | 33.08 | 264.61 | 2.087 | 28.41 | 2.979 | 19.91 | 0.60x | 0.70x |
| 37824,56,128 | rev | 1.800 | 37.66 | 32.95 | 263.59 | 2.041 | 29.06 | 3.042 | 19.50 | 0.59x | 0.67x |
