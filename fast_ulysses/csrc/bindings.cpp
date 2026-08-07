#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "a2a_plan.h"
#include "ulysses_common.cuh"
#include "ulysses_group.cuh"

namespace ulysses {

// Smoke test: include NVSHMEM headers + reference its types to prove the
// compile/host-link path works; no runtime init required.
int64_t nvshmem_uniqueid_nbytes()
{
    return static_cast<int64_t>(sizeof(nvshmemx_uniqueid_t));
}

namespace {

// Validation + dims for the uniform 4D a2a entry point. The input must already be contiguous.
//
// Even splits are handed to the plan in its general per-rank form -- seq_splits = [s/ws]*ws,
// head_splits = [n/ws]*ws -- so build_plan has no even/uneven distinction to get wrong. This
// entry point accepts nothing else: the divisibility checks below are what make the uniform
// assumption (every rank the same shard) true, and the symmetric pool's collective alloc
// depends on every rank computing the same output size.
A2ADims check_uniform_args(const at::Tensor& input, int64_t mode, int ws, int rank)
{
    TORCH_CHECK(input.is_cuda() && input.dim() == 4, "input must be a 4D CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "dtype must be float16 or bfloat16");
    const int64_t x1   = input.size(1);
    const int64_t x2   = input.size(2);
    const int64_t d    = input.size(3);
    const int64_t elem = input.element_size();
    TORCH_CHECK((d * elem) % 16 == 0, "d*elem_size must be 16B-aligned");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");
    A2ADims dims;
    dims.b          = input.size(0);
    dims.d          = d;
    dims.rank       = rank;
    dims.world_size = ws;
    if (mode == 0) {
        // x1 is this rank's sequence shard, x2 the global head count.
        TORCH_CHECK(x2 % ws == 0, "n_global must be divisible by world_size");
        dims.seq_splits.assign(ws, x1);
        dims.head_splits.assign(ws, x2 / ws);
    }
    else {
        // x1 is the global sequence length, x2 this rank's head shard.
        TORCH_CHECK(x1 % ws == 0, "s_global must be divisible by world_size");
        dims.seq_splits.assign(ws, x1 / ws);
        dims.head_splits.assign(ws, x2);
    }
    return dims;
}

}  // namespace

// The transfer rides the DMA engines: zero SM usage, so it overlaps compute that starves an
// SM-resident collective. This is the only transport -- the kernel and TMA paths, and the
// runtime autotune that chose between them, were removed: they cannot overlap a dependent
// GEMM chain (which never yields a block slot), and the autotune timed candidates INSIDE the
// collective, so any rank that ranked them differently diverged from the rest. Full
// rationale: all_to_all_ce.cu file header.
at::Tensor all_to_all_single_4d(
    const c10::intrusive_ptr<UlyssesGroup>& group, at::Tensor input, int64_t mode, std::string tag, bool barrier)
{
    input        = input.contiguous();
    const int ws = static_cast<int>(group->world_size());
    TORCH_CHECK(ws >= 1 && ws <= 8, "world_size must be in [1, 8] (single-node NVLink), got ", ws);
    const A2ADims dims = check_uniform_args(input, mode, ws, static_cast<int>(group->rank()));
    // Every byte offset comes from the plan; the transport only turns its ops into memcpy
    // calls. a2a_plan.cpp is host-only and covered by tests/test_plan.py without a GPU.
    const A2APlan plan = build_plan(dims, static_cast<int>(mode), input.element_size());

    const at::cuda::CUDAGuard guard(input.device());
    cudaStream_t              stream = at::cuda::getCurrentCUDAStream();

    const auto& buf = group->pool().acquire(plan.output_shape, input.scalar_type(), tag);

    // WRITERS WAIT FOR READERS, before writing anything.
    //
    // The window is single-buffered per tag, so this call is about to overwrite what the
    // previous call with this tag produced -- and a peer may still be reading its own copy of
    // it. The closing barrier below proves everyone's WRITES landed; it proves nothing about
    // everyone having finished READING. Without this one, a fast peer's next transfer lands in
    // our window while we are still consuming the last result.
    //
    // That is not hypothetical here: a2a_window_race.py and a2a_overlapping_barriers.py both
    // failed on it, the latter at iteration 1, before this was added.
    //
    // It guards the START of a call rather than the end of the previous one because this
    // operator returns the window ITSELF -- the caller does the reading, at a time the
    // operator never sees. Sitting here it covers both: the caller's reads are ordered ahead
    // of it on the same stream, and no peer can write until every rank has reached it.
    //
    // Costs one handshake per call. Measured at 8-14 us in custom_nccl_op, under 1% of a
    // model-sized collective; not re-measured here.
    group->fast_barrier(stream, tag);

    launch_a2a_ce(input.data_ptr(), buf.peer_ptrs, plan, group->ce_resources(), stream);
    // No nvshmemx_quiet: the transfers are CE memcpy operations, not NVSHMEM proxy writes, so
    // quiet -- which orders NVSHMEM operations -- would not cover them anyway. Their
    // completion is joined onto `stream` inside launch_a2a_ce, and the flag barrier's store is
    // stream-ordered after that.
    //
    // Whether a completed peer memcpy is VISIBLE at the destination when a later kernel's
    // store arrives is NOT documented: the CUDA API reference defines memcpy completion as a
    // host-side property, the Programming Guide's cross-device ordering guarantee is
    // NULL-stream-scoped and withdrawn for async copies in a non-default stream, and PTX 8.5
    // scopes .release to "prior operations from the current thread" -- which a copy engine
    // transfer is not. Neither NVSHMEM nor NCCL pairs a host-issued CE transfer with an SM
    // release store; both keep the flag on the data's path. custom_nccl_op measured the
    // ordering holding on H200 and on a PCIe box under a test with a working negative control,
    // but that is evidence, not a guarantee. See a2a_ce_flag_ordering.py.
    //
    // barrier=false defers the closing handshake to a later barrier=true call on the same
    // stream; until then the output views are NOT safe to read.
    if (barrier)
        group->fast_barrier(stream, tag);
    return buf.view;
}

}  // namespace ulysses

TORCH_LIBRARY(fast_ulysses, m)
{
    m.def("nvshmem_uniqueid_nbytes() -> int");
    m.impl("nvshmem_uniqueid_nbytes", &ulysses::nvshmem_uniqueid_nbytes);

    m.class_<ulysses::UlyssesGroup>("UlyssesGroup")
        .def(torch::init<std::vector<int64_t>, int64_t, int64_t, int64_t>())
        .def("rank", &ulysses::UlyssesGroup::rank)
        .def("world_size", &ulysses::UlyssesGroup::world_size)
        .def("destroy", &ulysses::UlyssesGroup::destroy)
        .def_static("uniqueid_nints", &ulysses::UlyssesGroup::uniqueid_nints)
        .def_static("get_uniqueid", &ulysses::UlyssesGroup::get_uniqueid)
        .def_static("init_world", &ulysses::UlyssesGroup::init_world);

    m.def("all_to_all_single_4d(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
          "Tensor input, int mode, str tag, bool barrier=True) -> Tensor");
    m.impl("all_to_all_single_4d", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_single_4d);
}

// Python `import _C` needs PyInit__C; TORCH_LIBRARY already registered at dlopen time.
PYBIND11_MODULE(_C, m) {}
