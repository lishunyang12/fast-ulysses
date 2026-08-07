#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ulysses {

// One symmetric allocation for the whole group, carved up locally.
//
// The constructor takes the entire pool with a single nvshmem_align; acquire() only ever returns
// an offset into it. Nothing on the call path allocates.
//
// That is not tidiness, it is the fix for a deadlock. nvshmem_align is collective and
// synchronizes the CUDA stream internally, while the flag barrier is a spin kernel that
// barrier=False deliberately leaves in flight. Allocating mid-run therefore parks the host inside
// nvshmem_align, where it can no longer issue the publish its peers are spinning for -- and their
// hosts are parked in the same place. docs/API.md, under reserve(), has the measurement.
//
// What this rests on instead: every rank hands out offsets in the SAME ORDER, so a tag lands at
// the same offset in every rank's slab and nvshmem_ptr on the slab base translates it. That order
// is already what the SPMD call contract requires; seal() is what turns a violation of it into an
// error rather than into one rank addressing another's window.
class SymmetricHeapPool {
public:
    // reserved_bytes: the whole pool, allocated here (must be <= NVSHMEM_SYMMETRIC_SIZE).
    // Collective. Every peer must be P2P-mappable; an unreachable pair is refused, named.
    SymmetricHeapPool(int64_t reserved_bytes, int world_size, std::vector<int> peer_global_pes);

    struct Buffer {
        void*                 sym_base;
        int64_t               numel;      // capacity, in elements of the dtype it was made for
        std::vector<uint64_t> peer_ptrs;  // this window's address in each peer's slab
    };

    // Reuse a tag's window when it is big enough; carve a new one when it is not.
    //
    // `numel` is a CAPACITY, not a shape. The key is (tag, dtype) and the match is capacity >=
    // requested, so a tag costs one window at its high-water mark rather than one per distinct
    // size it has ever seen; keying on exact capacity meant a benchmark sweeping three shapes
    // under one tag exhausted the pool. It also keeps the key off the caller's output shape,
    // which under uneven splits differs per rank and would fork the hit/miss pattern.
    //
    // Callers lay their own view over Buffer::sym_base, or copy out of it (see
    // all_to_all_single_4d_borrowed and all_to_all_single_4d).
    const Buffer& acquire(int64_t numel, c10::ScalarType dtype, const std::string& tag);

    // After this, a miss in acquire() is an error instead of a new window.
    //
    // Two things it catches. A shape that drifts upward: growth does not reclaim the offset it
    // outgrew, so the drift silently costs one window per growth. And ranks that stop agreeing on
    // what they allocate, which is the assumption the local offsets rest on.
    void seal()
    {
        sealed_ = true;
    }

    // Terminal collective op: before calling, release all views built over acquire()'d buffers and
    // ensure no A2A/collective is in flight, since this nvshmem_free's the slab those views alias.
    void destroy();

private:
    using Key = std::pair<std::string, c10::ScalarType>;
    int64_t               reserved_, used_ = 0;
    int                   world_size_;
    std::vector<int>      peer_global_pes_;
    void*                 slab_ = nullptr;  // the one allocation
    std::vector<uint64_t> slab_peer_;       // nvshmem_ptr(slab_, peer_global_pe)
    std::map<Key, Buffer> registry_;
    bool                  destroyed_ = false;
    bool                  sealed_    = false;
};

}  // namespace ulysses
