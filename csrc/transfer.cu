#include "fast_ulysses.h"

namespace ulysses {
namespace {

struct PeerFlags { uint64_t ptr[8]; };

__device__ __forceinline__ void publish(uint64_t* address, uint64_t value)
{
    // Every (writer, reader) pair owns its own slot, so this needs release ordering but not
    // atomicity. An atomic would also refuse to work: P2P native atomics are absent on PCIe.
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::
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

// mode 0: gather the head block that belongs to `peer` into a contiguous staging slot. The
// input's (batch, seq) rows are contiguous, so every batch element is covered by one copy.
void pack(uint8_t* stage, const uint8_t* input, int peer, const Dims& dims, cudaStream_t stream)
{
    const int64_t width = dims.heads / dims.world_size * dims.dim * dims.element_size;
    const int64_t pitch = dims.heads * dims.dim * dims.element_size;
    FU_CUDA_CHECK(cudaMemcpy2DAsync(stage + peer * slice_bytes(dims), width,
                                    input + peer * width, pitch, width,
                                    dims.batch * dims.seq, cudaMemcpyDeviceToDevice, stream));
}

// mode 0: the staged slot lands on the peer's output rows that belong to this rank.
void send_forward(uint64_t peer_output,
                  const uint8_t* stage,
                  int peer,
                  const Dims& dims,
                  cudaStream_t stream)
{
    const int64_t chunk = dims.seq * (dims.heads / dims.world_size) * dims.dim *
                          dims.element_size;
    const uint8_t* source = stage + peer * slice_bytes(dims);
    auto* destination = reinterpret_cast<uint8_t*>(peer_output);
    for (int64_t b = 0; b < dims.batch; ++b)
        FU_CUDA_CHECK(cudaMemcpyAsync(destination + (b * dims.world_size + dims.rank) * chunk,
                                      source + b * chunk, chunk, cudaMemcpyDefault, stream));
}

// mode 1: this rank's sequence slice for `peer` is already contiguous, so it goes straight into
// the peer's staging slot without a local pass.
void send_reverse(uint64_t peer_stage,
                  const uint8_t* input,
                  int peer,
                  const Dims& dims,
                  cudaStream_t stream)
{
    const int64_t chunk = dims.seq / dims.world_size * dims.heads * dims.dim *
                          dims.element_size;
    auto* destination = reinterpret_cast<uint8_t*>(peer_stage) + dims.rank * slice_bytes(dims);
    for (int64_t b = 0; b < dims.batch; ++b)
        FU_CUDA_CHECK(cudaMemcpyAsync(destination + b * chunk,
                                      input + (b * dims.world_size + peer) * chunk, chunk,
                                      cudaMemcpyDefault, stream));
}

// This rank's own share never crosses a link, so it moves from the input to the output directly.
void local_share(uint8_t* output, const uint8_t* input, const Dims& dims, cudaStream_t stream)
{
    if (dims.mode == 0) {
        const int64_t width = dims.heads / dims.world_size * dims.dim * dims.element_size;
        const int64_t pitch = dims.heads * dims.dim * dims.element_size;
        const int64_t chunk = dims.seq * width;
        for (int64_t b = 0; b < dims.batch; ++b)
            FU_CUDA_CHECK(cudaMemcpy2DAsync(
                output + (b * dims.world_size + dims.rank) * chunk, width,
                input + b * dims.seq * pitch + dims.rank * width, pitch, width, dims.seq,
                cudaMemcpyDeviceToDevice, stream));
    } else {
        const int64_t local_seq = dims.seq / dims.world_size;
        const int64_t width = dims.heads * dims.dim * dims.element_size;
        const int64_t pitch = width * dims.world_size;
        for (int64_t b = 0; b < dims.batch; ++b)
            FU_CUDA_CHECK(cudaMemcpy2DAsync(
                output + b * local_seq * pitch + dims.rank * width, pitch,
                input + (b * dims.world_size + dims.rank) * local_seq * width, width, width,
                local_seq, cudaMemcpyDeviceToDevice, stream));
    }
}

// One peer's share written straight into that peer's output, relayout and all.
void copy_strided(uint64_t peer_output, const uint8_t* input, int peer, const Dims& dims,
                  cudaStream_t stream)
{
    auto* destination = reinterpret_cast<uint8_t*>(peer_output);
    const int64_t element = dims.dim * dims.element_size;
    for (int64_t b = 0; b < dims.batch; ++b) {
        int64_t source_offset, destination_offset, source_pitch, destination_pitch, width, rows;
        if (dims.mode == 0) {
            const int64_t local_heads = dims.heads / dims.world_size;
            source_offset = (b * dims.seq * dims.heads + peer * local_heads) * element;
            destination_offset =
                (b * dims.seq * dims.world_size + dims.rank * dims.seq) * local_heads * element;
            source_pitch = dims.heads * element;
            destination_pitch = local_heads * element;
            width = local_heads * element;
            rows = dims.seq;
        } else {
            const int64_t local_seq = dims.seq / dims.world_size;
            source_offset = (b * dims.seq + peer * local_seq) * dims.heads * element;
            destination_offset =
                (b * local_seq * dims.heads * dims.world_size + dims.rank * dims.heads) *
                element;
            source_pitch = dims.heads * element;
            destination_pitch = dims.heads * dims.world_size * element;
            width = dims.heads * element;
            rows = local_seq;
        }
        FU_CUDA_CHECK(cudaMemcpy2DAsync(destination + destination_offset, destination_pitch,
                                        input + source_offset, source_pitch, width, rows,
                                        cudaMemcpyDefault, stream));
    }
}

}  // namespace

void barrier(cudaStream_t stream,
             const std::vector<uint64_t>& flags,
             int rank,
             uint64_t epoch)
{
    if (flags.size() <= 1) return;
    PeerFlags peers{};
    for (size_t i = 0; i < flags.size(); ++i) peers.ptr[i] = flags[i];
    barrier_kernel<<<1, 32, 0, stream>>>(reinterpret_cast<uint64_t*>(flags[rank]), peers,
                                         flags.size(), rank, epoch);
    FU_CUDA_CHECK(cudaGetLastError());
}

int64_t slice_bytes(const Dims& dims)
{
    return dims.batch * dims.seq * dims.heads * dims.dim * dims.element_size /
           dims.world_size;
}

void exchange_peers(const void* input,
                    void* output,
                    void* stage,
                    const std::vector<uint64_t>& output_peers,
                    const std::vector<uint64_t>& stage_peers,
                    const std::vector<int>& peers,
                    const Dims& dims,
                    cudaStream_t caller,
                    cudaStream_t pack_stream,
                    cudaStream_t send_stream,
                    const std::vector<cudaEvent_t>& events)
{
    const auto* source = static_cast<const uint8_t*>(input);
    auto* staging = static_cast<uint8_t*>(stage);
    TORCH_CHECK(peers.size() + 2 <= events.size(), "not enough events for the peer list");
    const cudaEvent_t ready = events[events.size() - 2];
    const cudaEvent_t done = events.back();

    FU_CUDA_CHECK(cudaEventRecord(ready, caller));
    FU_CUDA_CHECK(cudaStreamWaitEvent(send_stream, ready, 0));
    if (dims.mode == 0) {
        FU_CUDA_CHECK(cudaStreamWaitEvent(pack_stream, ready, 0));
        // Pack peer i+1 while peer i is on the wire: the local pass is an order of magnitude
        // faster than the link, so it disappears behind it after the first slot.
        for (size_t i = 0; i < peers.size(); ++i) {
            pack(staging, source, peers[i], dims, pack_stream);
            FU_CUDA_CHECK(cudaEventRecord(events[i], pack_stream));
            FU_CUDA_CHECK(cudaStreamWaitEvent(send_stream, events[i], 0));
            send_forward(output_peers[peers[i]], staging, peers[i], dims, send_stream);
        }
    } else {
        for (int peer : peers)
            send_reverse(stage_peers[peer], source, peer, dims, send_stream);
    }
    local_share(static_cast<uint8_t*>(output), source, dims, caller);
    FU_CUDA_CHECK(cudaEventRecord(done, send_stream));
    FU_CUDA_CHECK(cudaStreamWaitEvent(caller, done, 0));
}

void exchange_peers_direct(const void* input,
                           const std::vector<uint64_t>& output_peers,
                           const std::vector<int>& peers,
                           const Dims& dims,
                           cudaStream_t stream)
{
    const auto* source = static_cast<const uint8_t*>(input);
    for (int peer : peers) copy_strided(output_peers[peer], source, peer, dims, stream);
    copy_strided(output_peers[dims.rank], source, dims.rank, dims, stream);
}

void unpack_peers(void* output,
                  const void* stage,
                  const std::vector<int>& peers,
                  const Dims& dims,
                  cudaStream_t stream)
{
    const int64_t width = dims.heads * dims.dim * dims.element_size;
    const int64_t pitch = width * dims.world_size;
    const int64_t rows = dims.batch * dims.seq / dims.world_size;
    for (int peer : peers)
        FU_CUDA_CHECK(cudaMemcpy2DAsync(static_cast<uint8_t*>(output) + peer * width, pitch,
                                        static_cast<const uint8_t*>(stage) +
                                            peer * slice_bytes(dims),
                                        width, width, rows, cudaMemcpyDeviceToDevice, stream));
}

}  // namespace ulysses
