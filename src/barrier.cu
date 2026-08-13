#include <fast_ulysses/common.hpp>
#include <fast_ulysses/transfer.hpp>

namespace ulysses {
namespace {

struct PeerFlags { uint64_t ptr[8]; };

__device__ __forceinline__ void publish(uint64_t* address, uint64_t value)
{
    asm volatile("red.release.sys.global.max.u64 [%0], %1;" ::
                 "l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ uint64_t acquire(const uint64_t* address)
{
    uint64_t value;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) :
                 "l"(address) : "memory");
    return value;
}

__global__ void barrier_kernel(uint64_t* local,
                               PeerFlags peers,
                               int world_size,
                               int rank,
                               uint64_t epoch)
{
    const int peer = threadIdx.x;
    if (peer >= world_size) return;
    publish(reinterpret_cast<uint64_t*>(peers.ptr[peer]) + rank, epoch);
    while (acquire(local + peer) < epoch) { }
}

}  // namespace

void fast_barrier(cudaStream_t stream,
                  const std::vector<uint64_t>& flag_ptrs,
                  int rank,
                  uint64_t epoch)
{
    if (flag_ptrs.size() <= 1) return;
    PeerFlags peers{};
    for (size_t i = 0; i < flag_ptrs.size(); ++i) peers.ptr[i] = flag_ptrs[i];
    barrier_kernel<<<1, 32, 0, stream>>>(
        reinterpret_cast<uint64_t*>(flag_ptrs[rank]), peers,
        static_cast<int>(flag_ptrs.size()), rank, epoch);
    ULYSSES_CUDA_CHECK(cudaGetLastError());
}

}  // namespace ulysses
