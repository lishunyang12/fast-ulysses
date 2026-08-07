#pragma once
#include "a2a_plan.h"
#include "symmetric_pool.cuh"
#include "ulysses_common.cuh"
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <nvshmem.h>
#include <string>
#include <torch/custom_class.h>
#include <vector>

namespace ulysses {

// Per-group CE (copy-engine) transfer resources: one stream per peer for the memcpy
// fan-out. Created lazily by UlyssesGroup::ce_resources(), released in destroy(). Serial
// use only (same contract as the config caches). Join events are deliberately NOT pooled
// here -- see the fresh-event note in launch_a2a_ce.
struct CEResources {
    std::vector<cudaStream_t> streams;
};

// CE transfer path: issues `plan.ops` as a per-peer cudaMemcpy2D/3DAsync fan-out over
// ce.streams, joined back to `stream` with events. The addressing comes from build_plan
// (a2a_plan.h); `peer_ptrs[p]` is the base of peer p's symmetric window, which op offsets are
// relative to. The caller appends the flag barrier (no nvshmem quiet needed: these are not
// NVSHMEM proxy writes). Rationale: all_to_all_ce.cu file header.
void launch_a2a_ce(const void*                  src,
                   const std::vector<uint64_t>& peer_ptrs,
                   const A2APlan&               plan,
                   const CEResources&           ce,
                   int                          rank,
                   cudaStream_t                 stream);

class UlyssesGroup: public torch::CustomClassHolder {
public:
    static int64_t              uniqueid_nints();  // ceil(sizeof(nvshmemx_uniqueid_t)/8)
    static std::vector<int64_t> get_uniqueid();    // rank0 only
    static void init_world(std::vector<int64_t> uid_ints, int64_t global_rank, int64_t global_nranks);  // idempotent

    UlyssesGroup(std::vector<int64_t> peer_global_pes, int64_t my_rank, int64_t device_id, int64_t reserved_bytes);
    ~UlyssesGroup() override;

    int64_t rank() const
    {
        return my_rank_;
    }
    int64_t world_size() const
    {
        return world_size_;
    }
    void destroy();

    SymmetricHeapPool& pool()
    {
        return *pool_;
    }
    nvshmem_team_t team() const
    {
        return team_;
    }

    // Custom single-node NVLink flag barrier: replaces the slow nvshmem sync (~280us) that falls back on
    // hardware without NVLS fabric. No nvshmem quiet is needed (or would help) before it: the transport
    // issues raw cudaMemcpy2DAsync into nvshmem_ptr addresses, and quiet orders NVSHMEM operations, which
    // those are not -- their completion is joined onto `stream` inside launch_a2a_ce and the flag store is
    // stream-ordered after that. See the closing-barrier comment in bindings.cpp. No-op when world_size==1.
    //
    // `tag` picks the barrier state, of which there is ONE SET PER TAG -- see BarrierState. It is the
    // caller's tag, i.e. the same one that names the output buffer.
    void fast_barrier(cudaStream_t stream, const std::string& tag);

    // Lazily create (world_size streams + events) and return the CE transfer resources.
    const CEResources& ce_resources();

private:
    int                                my_rank_, world_size_, device_id_;
    std::vector<int>                   peer_global_pes_;
    nvshmem_team_t                     team_;
    bool                               owns_team_ = false;
    bool                               destroyed_ = false;
    std::unique_ptr<SymmetricHeapPool> pool_;

    // CE transfer resources (lazy; see ce_resources()).
    CEResources ce_;
    bool        ce_ready_ = false;

    // fast_barrier state: symmetric flag buffer (uint64[ws]) + monotonic epoch. ONE SET PER TAG.
    //
    // The epoch protocol needs every rank to assign epochs to the same handshakes in the same order,
    // and program order gives that only WITHIN a tag: the contract says an outstanding async result
    // must be waited on before the next call with THAT tag, and says nothing about other tags. An
    // async call on the comm stream and a sync call on the caller's stream, on different tags, run on
    // unordered streams; with one counter for the whole group their barrier kernels interleave
    // differently on different ranks, so a rank ends up waiting on an epoch its peer published for the
    // OTHER collective -- which says nothing about the transfer it actually cares about, and the
    // window is read before it is written. Measured in the NCCL reference build of this operator (4
    // ranks): a non-atomic per-group increment HUNG at 2000 iterations; making it atomic fixed the
    // hang and NOT the race (torn results at iteration 47 / 87 / 126). Per tag there is nothing to
    // interleave.
    //
    // The flags are a symmetric-heap buffer like the data (the tag goes in the pool name); the epoch
    // is this rank's own device memory, never addressed by a peer, and device-side so that a captured
    // graph advances it on replay -- see ulysses_barrier_kernel.
    struct BarrierState {
        void*                 my_flags = nullptr;  // this rank's flag base
        std::vector<uint64_t> peer_flags;          // per-peer flag base (including self)
        uint64_t*             epoch = nullptr;     // device counter, incremented by the kernel
    };
    std::map<std::string, BarrierState> barriers_;

    // This tag's barrier state, created on first use (collective pool alloc + zeroed flags).
    const BarrierState& barrier_state_(const std::string& tag, cudaStream_t stream);
};

}  // namespace ulysses
