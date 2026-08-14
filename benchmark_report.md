# fast-ulysses benchmark report

- world_size: 8
- dtype: bfloat16
- backend: mlx5
- NCCL_P2P_LEVEL: <unset>
- warmup: 10 calls/case
- measurement: 1 call(s)/trial, 12 trials, slowest rank then median

All bandwidths are decimal GB/s. `bus` counts only bytes sent to remote ranks; `aggregate` is the sum across all ranks.

| Shape | Dir | Raw NCCL ms | NCCL alg GB/s | NCCL bus GB/s | NCCL aggregate GB/s | NCCL + layout ms | Layout GB/s | Fast ms | Fast GB/s | Raw / fast | Layout / fast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 37824,56,128 | fwd | 1.779 | 38.09 | 33.33 | 266.65 | 2.076 | 28.57 | 1.709 | 34.69 | 1.04x | 1.21x |
| 37824,56,128 | rev | 1.800 | 37.66 | 32.96 | 263.65 | 2.045 | 29.00 | 1.583 | 37.46 | 1.14x | 1.29x |
| 75600,40,128 | fwd | 2.545 | 38.02 | 33.27 | 266.15 | 2.940 | 28.80 | 2.163 | 39.15 | 1.18x | 1.36x |
| 75600,40,128 | rev | 2.555 | 37.87 | 33.13 | 265.07 | 2.908 | 29.12 | 2.204 | 38.42 | 1.16x | 1.32x |
| 32760,40,128 | fwd | 1.168 | 35.91 | 31.43 | 251.40 | 1.301 | 28.20 | 0.993 | 36.95 | 1.18x | 1.31x |
| 32760,40,128 | rev | 1.177 | 35.63 | 31.18 | 249.43 | 1.265 | 29.01 | 1.004 | 36.54 | 1.17x | 1.26x |
