#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace ulysses {

void launch_equal_a2a(const void* src,
                      const std::vector<uint64_t>& peer_ptrs,
                      int mode,
                      int64_t batch,
                      int64_t axis1,
                      int64_t axis2,
                      int64_t head_dim,
                      int64_t element_size,
                      int rank,
                      cudaStream_t stream);

void fast_barrier(cudaStream_t stream,
                  const std::vector<uint64_t>& flag_ptrs,
                  int rank,
                  uint64_t epoch);

}  // namespace ulysses
