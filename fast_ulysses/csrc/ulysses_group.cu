#include "ulysses_group.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/torch.h>

namespace ulysses {

static bool g_world_inited = false;
static int  g_live_groups  = 0;

// ---- Custom NVLink flag barrier ----
// Flag layout: each rank holds uint64 flags[ws] PER TAG. On arrival, rank r publishes epoch into
// every peer p's flags[r] (P2P, release/sys MAX), then spins until its own flags[0..ws-1] all >= epoch
// (acquire/sys). Monotonically increasing epoch means no reset and no ABA. Strict lockstep within a
// tag (SPMD collective) keeps epoch identical across ranks throughout.
struct BarPeers {
    uint64_t p[8];
};

__global__ void ulysses_barrier_kernel(uint64_t* local, BarPeers peers, int ws, int rank, uint64_t* epoch_counter)
{
    // The epoch is advanced ON THE DEVICE, not passed in from the host. That is what makes a captured
    // call replayable: a host-computed epoch is a constant baked into the graph, so every replay
    // announces the same value, the peer flags already hold it from the previous replay, and the wait
    // is satisfied by stale state -- a handshake that silently does nothing. (In the NCCL reference
    // build of this operator that corrupted 260k elements on the FIRST replay.)
    //
    // atomicAdd rather than a plain read-modify-write: barrier state is per tag and a tag's calls are
    // ordered, so two of these are never resident on one counter in correct use; the atomic keeps a
    // caller who breaks that contract from losing an epoch outright instead of failing.
    __shared__ uint64_t epoch;
    if (threadIdx.x == 0)
        epoch = atomicAdd(reinterpret_cast<unsigned long long*>(epoch_counter), 1ULL) + 1;
    __syncthreads();  // must precede the early-out: every launched thread has to reach it

    int t = threadIdx.x;
    if (t >= ws)
        return;
    uint64_t* remote = reinterpret_cast<uint64_t*>(peers.p[t]) + rank;  // peer t's flags[rank]
    red_release_sys_max_u64(remote, epoch);
    uint64_t  v;
    uint64_t* mine = local + t;  // own flags[t] (written by peer t)
    do {
        v = ld_acquire_sys_u64(mine);
    } while (v < epoch);
}

const UlyssesGroup::BarrierState& UlyssesGroup::barrier_state_(const std::string& tag, cudaStream_t stream)
{
    auto it = barriers_.find(tag);
    if (it != barriers_.end())
        return it->second;

    BarrierState st;
    // The tag goes in the pool name, so each tag's handshake gets its own flags -- see BarrierState.
    const auto& buf = pool_->acquire({static_cast<int64_t>(world_size_)}, at::kLong, "__ulysses_sync__" + tag);
    st.my_flags     = buf.sym_base;
    st.peer_flags   = buf.peer_ptrs;
    // Epoch counter: this rank's own device memory, never addressed by a peer.
    ULYSSES_CUDA_CHECK(cudaMalloc(&st.epoch, sizeof(uint64_t)));
    ULYSSES_CUDA_CHECK(cudaMemsetAsync(st.epoch, 0, sizeof(uint64_t), stream));  // first kernel publishes 1
    ULYSSES_CUDA_CHECK(cudaMemsetAsync(st.my_flags, 0, world_size_ * sizeof(uint64_t), stream));
    // One slow sync: ensure all ranks finish clearing before anyone writes a flag (otherwise the
    // clear could overwrite an already-written epoch).
    nvshmemx_barrier_on_stream(team_, stream);
    return barriers_.emplace(tag, std::move(st)).first->second;
}

void UlyssesGroup::fast_barrier(cudaStream_t stream, const std::string& tag)
{
    if (world_size_ == 1)
        return;
    const BarrierState& st = barrier_state_(tag, stream);
    BarPeers            peers;
    for (int i = 0; i < world_size_; ++i)
        peers.p[i] = st.peer_flags[i];
    // NOTE (measured on 4xH200, CUDA 13.3): a cuStreamWriteValue64/WaitValue64 variant of
    // this barrier -- attractive on paper because the CE a2a path would then never need an
    // SM slot -- was tried and REVERTED: stream memops interfered with concurrent GEMMs
    // (overlap hidden% dropped from +34% to -28% with WRITE_VALUE_DEFAULT and -15% with
    // NO_MEMORY_BARRIER). The 1-block spin kernel only needs an SM slot at a kernel
    // boundary and empirically wins under compute pressure.
    ulysses_barrier_kernel<<<1, 32, 0, stream>>>(
        reinterpret_cast<uint64_t*>(st.my_flags), peers, world_size_, my_rank_, st.epoch);
    ULYSSES_CUDA_CHECK(cudaGetLastError());  // catch a barrier-kernel launch failure
}

int64_t UlyssesGroup::uniqueid_nints()
{
    return static_cast<int64_t>((sizeof(nvshmemx_uniqueid_t) + 7) / 8);
}

std::vector<int64_t> UlyssesGroup::get_uniqueid()
{
    nvshmemx_uniqueid_t uid;
    std::memset(&uid, 0, sizeof(uid));
    TORCH_CHECK(nvshmemx_get_uniqueid(&uid) == 0, "nvshmemx_get_uniqueid failed");
    std::vector<int64_t> out(uniqueid_nints(), 0);
    std::memcpy(out.data(), &uid, sizeof(uid));
    return out;
}

void UlyssesGroup::init_world(std::vector<int64_t> uid_ints, int64_t global_rank, int64_t global_nranks)
{
    if (g_world_inited)
        return;
    TORCH_CHECK(static_cast<int64_t>(uid_ints.size()) >= uniqueid_nints(), "uid_ints too short");
    nvshmemx_uniqueid_t uid;
    std::memcpy(&uid, uid_ints.data(), sizeof(uid));
    // Use INITIALIZER, not memset(0): it stamps the version field of attr/args/uid_args.
    // nvshmemx_set_attr_uniqueid_args does not write version, and hostlib_init_attr dispatches the V2
    // path based on attr.args.version, so the version must be stamped first (inline nvshmemx_init_attr
    // auto-stamps when version is invalid; here we explicitly substitute that step).
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    TORCH_CHECK(
        nvshmemx_set_attr_uniqueid_args(static_cast<int>(global_rank), static_cast<int>(global_nranks), &uid, &attr)
            == 0,
        "nvshmemx_set_attr_uniqueid_args failed");
    // DEVIATION: use the host-lib direct entry nvshmemx_hostlib_init_attr instead of
    // inline nvshmemx_init_attr. The inline version calls nvshmemi_init_thread, a symbol that lives only
    // in static libnvshmem_device.a; linking it clashes with the NVSHMEM version node of torch's bundled
    // libtorch_nvshmem.so (undefined symbol nvshmem_selected_device_transport). hostlib_init_attr is the
    // equivalent entry exported directly by the host shared library (NVSHMEM's own python UID path uses it).
    TORCH_CHECK(nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr) == 0,
                "nvshmemx_hostlib_init_attr failed");
    g_world_inited = true;
}

UlyssesGroup::UlyssesGroup(std::vector<int64_t> peer_global_pes,
                           int64_t              my_rank,
                           int64_t              device_id,
                           int64_t              reserved_bytes):
    my_rank_(static_cast<int>(my_rank)),
    world_size_(static_cast<int>(peer_global_pes.size())),
    device_id_(static_cast<int>(device_id))
{
    TORCH_CHECK(g_world_inited, "init_world must be called before constructing UlyssesGroup");
    TORCH_CHECK(
        world_size_ >= 1 && world_size_ <= 8, "world_size must be in [1, 8] (single-node NVLink), got ", world_size_);
    ULYSSES_CUDA_CHECK(cudaSetDevice(device_id_));
    peer_global_pes_.reserve(world_size_);
    for (auto pe : peer_global_pes)
        peer_global_pes_.push_back(static_cast<int>(pe));

    // team: the whole world -> TEAM_WORLD; else the strided slice of it that this PE set describes.
    // The stride comes from the PE list, so the 2-D parallel layout works: tp=2 x ulysses-sp=4 on 8
    // GPUs gives the sp groups {0,2,4,6} and {1,3,5,7}, stride 2, and both are live at once.
    //
    // Each PE passes its OWN triplet below, which is not the shape the API is built for: NVSHMEM's
    // own nvshmem_team_split_2d gets its two axis teams by looping split_strided with EVERY PE in
    // EVERY call, non-members included. What the one-call-per-PE form here rests on was READ out of
    // NVSHMEM 3.7.2's team_internal.cpp and has NOT been run: nvshmemi_team_split_strided wraps its
    // body in PARENT-team collectives -- a barrier before, quiet + team_sync + an error allreduce --
    // while the body, the AND-reduce that picks the team index, runs over the CHILD triplet alone.
    // Hence the contract comm.py states: the live groups must PARTITION the job, every PE in
    // exactly one and all of them constructing together, so those parent-team collectives still
    // pair up one for one. A rank that builds no group, or a second one by itself, hangs the job
    // inside the split. Two disjoint teams may come back with the SAME team index -- the psync that
    // index names is each PE's own, so teams sharing no PE never meet in it.
    const int gpes          = nvshmem_n_pes();
    bool      is_full_world = (world_size_ == gpes);
    for (int i = 0; i < world_size_ && is_full_world; ++i)
        if (peer_global_pes_[i] != i)
            is_full_world = false;
    if (is_full_world) {
        team_      = NVSHMEM_TEAM_WORLD;
        owns_team_ = false;
    }
    else {
        const int   start  = peer_global_pes_[0];
        const int   stride = world_size_ > 1 ? peer_global_pes_[1] - start : 1;
        std::string pes;  // the offending list, named by both checks below (<= 8 entries)
        for (int pe : peer_global_pes_)
            pes += std::to_string(pe) + " ";
        // A PE set that is not an arithmetic progression is not a strided team and has no other
        // representation here -- refuse it by name instead of splitting some other PE set.
        TORCH_CHECK(stride >= 1, "process-group ranks must be increasing; got [", pes, "]");
        for (int i = 1; i < world_size_; ++i)
            TORCH_CHECK(peer_global_pes_[i] == start + i * stride,
                        "process-group ranks must be evenly strided (they become an NVSHMEM strided team); got [",
                        pes,
                        "], which breaks stride ",
                        stride,
                        " at index ",
                        i,
                        " (rank ",
                        peer_global_pes_[i],
                        ", expected ",
                        start + i * stride,
                        ")");
        nvshmem_team_config_t cfg;
        std::memset(&cfg, 0, sizeof(cfg));
        TORCH_CHECK(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, start, stride, world_size_, &cfg, 0, &team_) == 0,
                    "nvshmem_team_split_strided(start=",
                    start,
                    ", stride=",
                    stride,
                    ", size=",
                    world_size_,
                    ") failed");
        owns_team_ = true;
    }

    pool_ = std::make_unique<SymmetricHeapPool>(reserved_bytes, world_size_, peer_global_pes_);
    ++g_live_groups;
}

const CEResources& UlyssesGroup::ce_resources()
{
    if (!ce_ready_) {
        ce_.streams.resize(world_size_);
        for (int i = 0; i < world_size_; ++i)
            ULYSSES_CUDA_CHECK(cudaStreamCreateWithFlags(&ce_.streams[i], cudaStreamNonBlocking));
        ce_ready_ = true;
    }
    return ce_;
}

void UlyssesGroup::destroy()
{
    if (destroyed_)
        return;
    if (ce_ready_) {
        // Unchecked teardown calls, matching the rest of destroy(): the caller already
        // guarantees quiescence (comm.py drains the comm stream + dist.barrier first).
        for (auto s : ce_.streams) {
            cudaStreamSynchronize(s);
            cudaStreamDestroy(s);
        }
        ce_       = {};
        ce_ready_ = false;
    }
    // Epoch counters only (unchecked, like the stream teardown above); the flags they go with are
    // pool segments and go away with pool_->destroy().
    for (auto& [tag, st] : barriers_)
        cudaFree(st.epoch);
    barriers_.clear();
    if (pool_)
        pool_->destroy();
    if (owns_team_)
        nvshmem_team_destroy(team_);
    destroyed_ = true;
    if (--g_live_groups == 0 && g_world_inited) {
        // DEVIATION: nvshmem_finalize() is inline and calls nvshmemi_finalize() (again only in static
        // device.a). Use nvshmemx_hostlib_finalize() exported by the host shared library instead.
        nvshmemx_hostlib_finalize();
        g_world_inited = false;
    }
}

UlyssesGroup::~UlyssesGroup()
{
    // No collective teardown from the destructor: destroy() calls nvshmem_free / team_destroy /
    // hostlib_finalize, all collective, while GC / interpreter-exit timing differs across ranks --
    // a rank destructing alone would hang the group. Leak and warn instead; explicit destroy()
    // (the Python wrapper guards it with dist.barrier) is the supported path.
    if (!destroyed_)
        std::cerr << "[fast_ulysses] UlyssesGroup dropped without destroy(); leaking symmetric-heap "
                     "resources (call group.destroy() on all ranks)"
                  << std::endl;
}

}  // namespace ulysses
