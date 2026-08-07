#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <map>
#include <nvshmem.h>
#include <string>
#include <tuple>
#include <vector>

namespace ulysses {

class SymmetricHeapPool {
public:
    // reserved_bytes: per-group cap (must be <= NVSHMEM_SYMMETRIC_SIZE reserved at init).
    SymmetricHeapPool(int64_t reserved_bytes, int world_size, std::vector<int> peer_global_pes);

    struct Buffer {
        void*                 sym_base;
        int64_t               nbytes;
        std::vector<uint64_t> peer_ptrs;  // nvshmem_ptr(sym_base, peer_global_pe)
    };

    // Reuse on (tag,numel,dtype) hit; otherwise collectively allocate a new segment and register it.
    //
    // `numel` is a CAPACITY, not a shape: nvshmem_align is a collective, so on a miss every rank
    // must ask for the same size AND miss together. Both are why the key is not the caller's
    // output shape -- under uneven splits that differs per rank, and two calls whose shapes
    // happen to collide on one rank but not on another would fork the hit/miss pattern and hang.
    // Callers lay their own view over Buffer::sym_base, or copy out of it (see
    // all_to_all_single_4d_borrowed and all_to_all_single_4d).
    const Buffer& acquire(int64_t numel, c10::ScalarType dtype, const std::string& tag);

    // Terminal collective op: before calling, release all views built over acquire()'d buffers and
    // ensure no A2A/collective is in flight, since this nvshmem_free's the segments those views alias.
    void destroy();  // nvshmem_free all segments + clear registry

private:
    using Key = std::tuple<std::string, int64_t, c10::ScalarType>;
    int64_t               reserved_, used_ = 0;
    int                   world_size_;
    std::vector<int>      peer_global_pes_;
    std::vector<void*>    segments_;
    std::map<Key, Buffer> registry_;
    bool                  destroyed_ = false;
};

}  // namespace ulysses
