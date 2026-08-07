#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <optional>
#include <torch/extension.h>
#include <torch/library.h>

#include <tuple>

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

// Validation + dims for the 4D a2a entry point. The input must already be contiguous.
//
// The plan works in per-rank splits and treats uneven as the general case, so this function's
// only job is to decide what the splits ARE: the caller's, or -- when it passes none -- the even
// ones that today's shapes imply. Sequence shards are uneven in production (sglang's
// build_shard_plan pads the sequence to a multiple of sp_size purely so every rank holds the
// same length; passing the true per-rank lengths here is what lets that padding go away). Head
// shards are always even in practice, but they cost nothing extra: the plan takes both.
A2ADims make_dims(const at::Tensor&                          input,
                  int64_t                                    mode,
                  int                                        ws,
                  int                                        rank,
                  const std::optional<std::vector<int64_t>>& seq_splits,
                  const std::optional<std::vector<int64_t>>& head_splits)
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

    if (seq_splits.has_value() || head_splits.has_value()) {
        // One without the other has no meaning: the plan needs the WHOLE group's geometry, and
        // the missing half cannot be inferred from a shape that is itself already sharded.
        TORCH_CHECK(seq_splits.has_value() && head_splits.has_value(),
                    "pass both seq_splits and head_splits, or neither");
        dims.seq_splits  = *seq_splits;
        dims.head_splits = *head_splits;
        dims.validate();  // length/sign checks, before indexing by rank below
        // Cross-check the declared splits against the tensor actually handed in, so a caller
        // that mis-shards gets an error here instead of a silently corrupt result.
        const int64_t expect_x1 = (mode == 0) ? dims.seq_splits[rank] : dims.seq_total();
        const int64_t expect_x2 = (mode == 0) ? dims.head_total() : dims.head_splits[rank];
        TORCH_CHECK(x1 == expect_x1 && x2 == expect_x2,
                    "input is [",
                    dims.b,
                    ", ",
                    x1,
                    ", ",
                    x2,
                    ", ",
                    d,
                    "] but the splits imply [",
                    dims.b,
                    ", ",
                    expect_x1,
                    ", ",
                    expect_x2,
                    ", ",
                    d,
                    "]");
        return dims;
    }

    // No splits given: the even special case, which the shape alone determines only if the
    // scattered axis divides. The other axis is already sharded on entry, so it never has to.
    if (mode == 0) {
        // x1 is this rank's sequence shard, x2 the global head count.
        TORCH_CHECK(x2 % ws == 0, "n_global must be divisible by world_size (or pass head_splits)");
        dims.seq_splits.assign(ws, x1);
        dims.head_splits.assign(ws, x2 / ws);
    }
    else {
        // x1 is the global sequence length, x2 this rank's head shard.
        TORCH_CHECK(x1 % ws == 0, "s_global must be divisible by world_size (or pass seq_splits)");
        dims.seq_splits.assign(ws, x1 / ws);
        dims.head_splits.assign(ws, x2);
    }
    return dims;
}

// Validated input, the plan, and the tensor the result will be copied into. `output` is left
// UNDEFINED for a borrowed call: its result is the window itself, so there is nothing to copy
// into. Shared by every entry point so they cannot drift apart on validation.
struct Prepared {
    at::Tensor x;
    at::Tensor output;
    A2APlan    plan;
};

// Everything this does runs BEFORE the call's first collective (a tag's first acquire() calls
// nvshmem_align, then fast_barrier), so a rejected argument leaves no rank waiting on peers
// that did not reject it. Under SPMD every rank rejects the same call.
Prepared prepare(const c10::intrusive_ptr<UlyssesGroup>&    group,
                 const at::Tensor&                          input,
                 int64_t                                    mode,
                 const std::optional<std::vector<int64_t>>& seq_splits,
                 const std::optional<std::vector<int64_t>>& head_splits,
                 const std::optional<at::Tensor>&           out,
                 bool                                       borrowed)
{
    const int ws = static_cast<int>(group->world_size());
    TORCH_CHECK(ws >= 1 && ws <= 8, "world_size must be in [1, 8] (single-node NVLink), got ", ws);

    Prepared prepared;
    prepared.x         = input.contiguous();
    const A2ADims dims = make_dims(prepared.x, mode, ws, static_cast<int>(group->rank()), seq_splits, head_splits);
    // Every byte offset comes from the plan; the transport only turns its ops into memcpy
    // calls. a2a_plan.cpp is host-only and covered by tests/test_plan.py without a GPU.
    prepared.plan = build_plan(dims, static_cast<int>(mode), prepared.x.element_size());

    if (borrowed) {
        return prepared;
    }
    if (out.has_value()) {
        prepared.output = *out;
        TORCH_CHECK(prepared.output.is_cuda() && prepared.output.is_contiguous(),
                    "out must be a contiguous CUDA tensor");
        TORCH_CHECK(prepared.output.scalar_type() == prepared.x.scalar_type(),
                    "out has dtype ",
                    prepared.output.scalar_type(),
                    ", expected ",
                    prepared.x.scalar_type());
        TORCH_CHECK(prepared.output.sizes() == at::IntArrayRef(prepared.plan.output_shape),
                    "out has shape ",
                    prepared.output.sizes(),
                    ", expected ",
                    at::IntArrayRef(prepared.plan.output_shape));
    }
    else {
        prepared.output = at::empty(prepared.plan.output_shape, prepared.x.options());
    }
    return prepared;
}

// Barrier, transfer, barrier: everything the call does to the symmetric window, ordered on
// `stream`. Returns the tag's window, whose base holds this rank's result densely.
//
// The transfer rides the DMA engines: zero SM usage, so it overlaps compute that starves an
// SM-resident collective. This is the only transport -- the kernel and TMA paths, and the
// runtime autotune that chose between them, were removed: they cannot overlap a dependent
// GEMM chain (which never yields a block slot), and the autotune timed candidates INSIDE the
// collective, so any rank that ranked them differently diverged from the rest. Full
// rationale: all_to_all_ce.cu file header.
const SymmetricHeapPool::Buffer& transfer_on_stream(const c10::intrusive_ptr<UlyssesGroup>& group,
                                                    const Prepared&                         prepared,
                                                    const std::string&                      tag,
                                                    bool                                    barrier,
                                                    cudaStream_t                            stream)
{
    // The window is sized for the largest rank (plan.window_numel); this rank's own result is a
    // dense prefix of it, so the borrowed view and the copy-out below are both built from
    // plan.output_shape, not from the window's capacity. Sizing the window per rank instead
    // would break the collective alloc -- see A2APlan::window_numel and
    // SymmetricHeapPool::acquire.
    const auto& buf = group->pool().acquire(prepared.plan.window_numel, prepared.x.scalar_type(), tag);
    // Reading and writing the same window would be silent corruption, so refuse it.
    //
    // The pool keys on (tag, capacity, dtype) rather than shape, because under uneven splits
    // each rank's output shape differs and a shape key would fork a collective allocation --
    // a hang. The cost is that under EVEN splits the two modes of one tag now collide:
    // b*s_total*n_me*d == b*s_me*n_total*d identically. So `y = a2a_borrowed(x, mode=0,
    // tag="t"); a2a_borrowed(y, mode=1, tag="t")` hands the transport the very buffer every
    // peer is writing. That used to work by accident of the shape key. docs/API.md already
    // forbids concurrently-live results sharing a tag, but a round trip does not look like
    // one, and it is exactly the shape a usp-style caller reaches for.
    //
    // Only a BORROWED result can reach this check; a copied one is the caller's own memory,
    // which is the whole reason the copying form is the default.
    TORCH_CHECK(prepared.x.data_ptr() != buf.sym_base,
                "input aliases tag '",
                tag,
                "'s symmetric buffer: feeding a borrowed result straight back in under the "
                "same tag would read the window every peer is concurrently writing. Use a "
                "different tag for the second call, or the copying entry point.");

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
    // It guards the START of a call rather than the end of the previous one because a BORROWED
    // result is read by the caller, at a time the operator never sees. Sitting here it covers
    // that and the copying form's own copy-out alike: either read is ordered ahead of it on
    // the same stream, and no peer can write until every rank has reached it.
    //
    // Costs one handshake per call. Measured at 8-14 us in custom_nccl_op, under 1% of a
    // model-sized collective; not re-measured here.
    group->fast_barrier(stream, tag);

    launch_a2a_ce(prepared.x.data_ptr(),
                  buf.peer_ptrs,
                  prepared.plan,
                  group->ce_resources(),
                  static_cast<int>(group->rank()),
                  stream);
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
    // stream; until then the window is NOT safe to read. Only the borrowed entry points expose
    // it: a deferred copying call would copy the window out before the peers' writes landed.
    if (barrier) {
        group->fast_barrier(stream, tag);
    }
    return buf;
}

// Window -> the caller's tensor, ordered after the closing barrier on the same stream. Every
// rank's result is dense from the window base (A2APlan::output_shape), so this is one flat
// device-to-device copy rather than anything pitched.
//
// This rank's own share travels through the window like every peer's, so it is copied out
// here too. Routing it straight from the input to the output instead -- which is what
// custom_nccl_op's plan does -- would save it one HBM round trip, but it needs a second plan
// shape because the borrowed form still requires it in the window. Not done, not measured.
void copy_out(const Prepared& prepared, const SymmetricHeapPool::Buffer& buf, cudaStream_t stream)
{
    ULYSSES_CUDA_CHECK(cudaMemcpyAsync(prepared.output.data_ptr(),
                                       buf.sym_base,
                                       static_cast<size_t>(prepared.output.numel() * prepared.output.element_size()),
                                       cudaMemcpyDeviceToDevice,
                                       stream));
}

}  // namespace

// THE DEFAULT: the result is an ordinary tensor the caller owns, with no rules attached. It may
// outlive the next call with this tag, be read on another stream, or survive destroy(). The
// price is the copy-out above.
at::Tensor all_to_all_single_4d(const c10::intrusive_ptr<UlyssesGroup>&    group,
                                const at::Tensor&                          input,
                                int64_t                                    mode,
                                std::string                                tag,
                                const std::optional<std::vector<int64_t>>& seq_splits,
                                const std::optional<std::vector<int64_t>>& head_splits,
                                const std::optional<at::Tensor>&           out)
{
    const at::cuda::CUDAGuard guard(input.device());
    const Prepared            prepared = prepare(group, input, mode, seq_splits, head_splits, out, /*borrowed=*/false);
    cudaStream_t              stream   = at::cuda::getCurrentCUDAStream();
    const SymmetricHeapPool::Buffer& buf = transfer_on_stream(group, prepared, tag, /*barrier=*/true, stream);
    copy_out(prepared, buf, stream);
    return prepared.output;
}

// THE FAST PATH, spelled out at the call site: the result IS the tag's symmetric window. No
// copy-out, and every rule that makes that safe is the caller's to keep -- NOTHING here
// enforces them. What they are: the docstring on
// UlyssesGroup.all_to_all_single_4d_borrowed.
at::Tensor all_to_all_single_4d_borrowed(const c10::intrusive_ptr<UlyssesGroup>&    group,
                                         const at::Tensor&                          input,
                                         int64_t                                    mode,
                                         std::string                                tag,
                                         bool                                       barrier,
                                         const std::optional<std::vector<int64_t>>& seq_splits,
                                         const std::optional<std::vector<int64_t>>& head_splits)
{
    const at::cuda::CUDAGuard guard(input.device());
    const Prepared prepared = prepare(group, input, mode, seq_splits, head_splits, std::nullopt, /*borrowed=*/true);
    const SymmetricHeapPool::Buffer& buf =
        transfer_on_stream(group, prepared, tag, barrier, at::cuda::getCurrentCUDAStream());
    // The pool owns the memory; this is a no-op-deleter view of it.
    return at::from_blob(
        buf.sym_base, prepared.plan.output_shape, [](void*) {}, prepared.x.options());
}

// Benchmark-only: the copying call with CUDA events between its stages, so a caller can see
// where the time goes instead of inferring it from totals. Stages are `barrier_in` (writers
// wait for readers), `transfer` (the peer copies plus this rank's own share, which runs on the
// caller's stream underneath them -- timing those apart would describe a shape the operator
// does not have), `barrier_out` (readers wait for writers) and `copy_out` (window -> the
// caller's tensor, the one stage the borrowed form does not pay). They are strictly ordered on
// one stream, so they sum to the whole call.
//
// Reading the events needs a device synchronise, which the normal entry points never do.
std::tuple<at::Tensor, std::vector<double>>
all_to_all_single_4d_timed(const c10::intrusive_ptr<UlyssesGroup>&    group,
                           const at::Tensor&                          input,
                           int64_t                                    mode,
                           std::string                                tag,
                           const std::optional<std::vector<int64_t>>& seq_splits,
                           const std::optional<std::vector<int64_t>>& head_splits)
{
    const at::cuda::CUDAGuard guard(input.device());
    const Prepared prepared = prepare(group, input, mode, seq_splits, head_splits, std::nullopt, /*borrowed=*/false);
    cudaStream_t   stream   = at::cuda::getCurrentCUDAStream();
    const auto&    buf      = group->pool().acquire(prepared.plan.window_numel, prepared.x.scalar_type(), tag);
    TORCH_CHECK(prepared.x.data_ptr() != buf.sym_base, "input aliases tag '", tag, "'s symmetric buffer");

    cudaEvent_t marks[5];
    for (auto& ev : marks) {
        ULYSSES_CUDA_CHECK(cudaEventCreate(&ev));
    }

    ULYSSES_CUDA_CHECK(cudaEventRecord(marks[0], stream));
    group->fast_barrier(stream, tag);
    ULYSSES_CUDA_CHECK(cudaEventRecord(marks[1], stream));
    launch_a2a_ce(prepared.x.data_ptr(),
                  buf.peer_ptrs,
                  prepared.plan,
                  group->ce_resources(),
                  static_cast<int>(group->rank()),
                  stream);
    ULYSSES_CUDA_CHECK(cudaEventRecord(marks[2], stream));
    group->fast_barrier(stream, tag);
    ULYSSES_CUDA_CHECK(cudaEventRecord(marks[3], stream));
    copy_out(prepared, buf, stream);
    ULYSSES_CUDA_CHECK(cudaEventRecord(marks[4], stream));

    ULYSSES_CUDA_CHECK(cudaEventSynchronize(marks[4]));
    std::vector<double> stages(4, 0.0);
    for (int i = 0; i < 4; ++i) {
        float ms = 0.0F;
        ULYSSES_CUDA_CHECK(cudaEventElapsedTime(&ms, marks[i], marks[i + 1]));
        stages[i] = static_cast<double>(ms);
    }
    for (auto& ev : marks) {
        ULYSSES_CUDA_CHECK(cudaEventDestroy(ev));
    }
    return {prepared.output, stages};
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

    // The default: an ordinary tensor the caller owns; `out` is an optional preallocated
    // destination. No `barrier` flag -- deferring the closing handshake would make the
    // copy-out read the window before the peers' writes had landed.
    m.def("all_to_all_single_4d(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
          "Tensor input, int mode, str tag, int[]? seq_splits=None, int[]? head_splits=None, "
          "Tensor? out=None) -> Tensor");
    m.impl("all_to_all_single_4d", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::all_to_all_single_4d);

    // The result IS the symmetric window: no copy-out, and rules nothing enforces.
    m.def("all_to_all_single_4d_borrowed(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
          "Tensor input, int mode, str tag, bool barrier=True, int[]? seq_splits=None, "
          "int[]? head_splits=None) -> Tensor");
    m.impl("all_to_all_single_4d_borrowed",
           c10::DispatchKey::CompositeExplicitAutograd,
           &ulysses::all_to_all_single_4d_borrowed);

    // Benchmark only: synchronises the device to read its events.
    m.def("all_to_all_single_4d_timed(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
          "Tensor input, int mode, str tag, int[]? seq_splits=None, int[]? head_splits=None) "
          "-> (Tensor, float[])");
    m.impl("all_to_all_single_4d_timed",
           c10::DispatchKey::CompositeExplicitAutograd,
           &ulysses::all_to_all_single_4d_timed);
}

// Python `import _C` needs PyInit__C; TORCH_LIBRARY already registered at dlopen time.
PYBIND11_MODULE(_C, m) {}
