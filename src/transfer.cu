#include <fast_ulysses/common.hpp>
#include <fast_ulysses/transfer.hpp>

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
                      cudaStream_t stream)
{
    const int ws = static_cast<int>(peer_ptrs.size());
    const auto* source = static_cast<const uint8_t*>(src);
    const int64_t row_bytes = (mode == 0 ? axis2 / ws : axis2) *
                              head_dim * element_size;

    auto copy_peer = [&](int peer) {
        for (int64_t b = 0; b < batch; ++b) {
            int64_t src_offset;
            int64_t dst_offset;
            int64_t src_pitch;
            int64_t dst_pitch;
            int64_t rows;
            if (mode == 0) {
                const int64_t s_local = axis1;
                const int64_t h_global = axis2;
                const int64_t h_local = h_global / ws;
                src_offset = ((b * s_local * h_global) + peer * h_local) *
                             head_dim * element_size;
                dst_offset = ((b * s_local * ws + rank * s_local) * h_local) *
                             head_dim * element_size;
                src_pitch = h_global * head_dim * element_size;
                dst_pitch = h_local * head_dim * element_size;
                rows = s_local;
            } else {
                const int64_t s_global = axis1;
                const int64_t s_local = s_global / ws;
                const int64_t h_local = axis2;
                const int64_t h_global = h_local * ws;
                src_offset = ((b * s_global + peer * s_local) * h_local) *
                             head_dim * element_size;
                dst_offset = ((b * s_local * h_global) + rank * h_local) *
                             head_dim * element_size;
                src_pitch = h_local * head_dim * element_size;
                dst_pitch = h_global * head_dim * element_size;
                rows = s_local;
            }
            auto* destination = reinterpret_cast<uint8_t*>(peer_ptrs[peer]);
            ULYSSES_CUDA_CHECK(cudaMemcpy2DAsync(
                destination + dst_offset, static_cast<size_t>(dst_pitch),
                source + src_offset, static_cast<size_t>(src_pitch),
                static_cast<size_t>(row_bytes), static_cast<size_t>(rows),
                cudaMemcpyDefault, stream));
        }
    };

    for (int step = 1; step < ws; ++step) {
        const int peer = rank ^ step;
        if (peer < ws) copy_peer(peer);
    }
    if ((ws & (ws - 1)) != 0) {
        for (int peer = 0; peer < ws; ++peer)
            if (peer != rank && (peer ^ rank) >= ws) copy_peer(peer);
    }
    copy_peer(rank);
}

}  // namespace ulysses
