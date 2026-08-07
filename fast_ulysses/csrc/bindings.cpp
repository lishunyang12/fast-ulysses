#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <dlfcn.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <optional>
#include <torch/extension.h>
#include <torch/library.h>

#include <map>
#include <tuple>

#include "a2a_plan.h"
#include "ulysses_common.cuh"
#include "ulysses_group.cuh"
#include "work.h"

namespace ulysses {

namespace {

// Validation + dims for the 4D a2a entry point. The input must already be contiguous. Shape-only
// core, so reserve() can size a window for a call that has no tensor yet.
//
// The plan treats uneven as the general case, so the only decision here is what the splits ARE:
// the caller's, or -- when it passes none -- the even ones the shape implies. Passing true
// per-rank sequence lengths is what lets a caller drop its padding.
A2ADims make_dims_from_shape(at::IntArrayRef                            sizes,
                             c10::ScalarType                            dtype,
                             int64_t                                    mode,
                             int                                        ws,
                             int                                        rank,
                             const std::optional<std::vector<int64_t>>& seq_splits,
                             const std::optional<std::vector<int64_t>>& head_splits)
{
    TORCH_CHECK(sizes.size() == 4, "input must be 4D, got ", sizes.size(), " dims");
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "dtype must be float16 or bfloat16");
    const int64_t x1   = sizes[1];
    const int64_t x2   = sizes[2];
    const int64_t d    = sizes[3];
    const int64_t elem = static_cast<int64_t>(c10::elementSize(dtype));
    TORCH_CHECK((d * elem) % 16 == 0, "d*elem_size must be 16B-aligned");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");
    A2ADims dims;
    dims.b          = sizes[0];
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

A2ADims make_dims(const at::Tensor&                          input,
                  int64_t                                    mode,
                  int                                        ws,
                  int                                        rank,
                  const std::optional<std::vector<int64_t>>& seq_splits,
                  const std::optional<std::vector<int64_t>>& head_splits)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    return make_dims_from_shape(input.sizes(), input.scalar_type(), mode, ws, rank, seq_splits, head_splits);
}

// Validated input, the plan, and the tensor the result will be copied into. `output` is left
// UNDEFINED for a borrowed call: its result is the window itself, so there is nothing to copy
// into. Shared by every entry point so they cannot drift apart on validation.
struct Prepared {
    at::Tensor x;
    at::Tensor output;
    A2APlan    plan;
};

// Byte intervals [a, a+a_bytes) and [b, b+b_bytes) intersect.
bool intervals_overlap(const void* a, int64_t a_bytes, const void* b, int64_t b_bytes)
{
    const auto* pa = static_cast<const char*>(a);
    const auto* pb = static_cast<const char*>(b);
    return pa < pb + b_bytes && pb < pa + a_bytes;
}

// Refuse a call whose input or `out` shares bytes with the window it is about to fill.
//
// The pool keys on (tag, capacity, dtype) rather than shape, because under uneven splits each
// rank's output shape differs and a shape key would make ranks carve different windows. The
// cost is that under EVEN splits the two modes of one tag collide: b*s_total*n_me*d ==
// b*s_me*n_total*d identically. So `y = a2a_borrowed(x, mode=0, tag="t"); a2a_borrowed(y,
// mode=1, tag="t")` hands the transport the very buffer every peer is writing. That used to work
// by accident of the shape key. docs/API.md already forbids concurrently-live results sharing a
// tag, but a round trip does not look like one, and it is exactly the shape a usp-style caller
// reaches for.
//
// Only a BORROWED result can reach either check; a copied one is the caller's own memory, which
// is the whole reason the copying form is the default. A non-contiguous view of the window does
// not reach it either -- prepare() calls .contiguous(), which copies it out of the window first.
//
// Intervals rather than `data_ptr() == sym_base`: a borrowed result sliced on its batch axis
// (y[1:2]) is contiguous, starts past the base, and is still inside the window, so pointer
// equality let it through. Passing a borrowed result as `out` is the same hazard on the
// copy-out side, and had no check at all.
void check_window_aliasing(const Prepared& prepared, const SymmetricHeapPool::Buffer& buf, const std::string& tag)
{
    const int64_t window_bytes = buf.numel * prepared.x.element_size();
    TORCH_CHECK(!intervals_overlap(prepared.x.data_ptr(), prepared.x.nbytes(), buf.sym_base, window_bytes),
                "input overlaps tag '",
                tag,
                "'s symmetric window: it would be read while every peer writes it. Use a "
                "different tag for the second call, or the copying entry point.");
    if (prepared.output.defined()) {
        TORCH_CHECK(
            !intervals_overlap(prepared.output.data_ptr(), prepared.output.nbytes(), buf.sym_base, window_bytes),
            "out overlaps tag '",
            tag,
            "'s symmetric window: the copy-out would read and write the same bytes. "
            "Pass a tensor outside the symmetric heap.");
    }
}

// Everything this does runs BEFORE the call's first collective (fast_barrier), so a rejected
// argument leaves no rank waiting on peers that did not reject it. Under SPMD every rank
// rejects the same call.
Prepared prepare(const c10::intrusive_ptr<UlyssesGroup>&    group,
                 const at::Tensor&                          input,
                 int64_t                                    mode,
                 const std::optional<std::vector<int64_t>>& seq_splits,
                 const std::optional<std::vector<int64_t>>& head_splits,
                 const std::optional<at::Tensor>&           out,
                 bool                                       borrowed)
{
    const int ws = static_cast<int>(group->world_size());
    TORCH_CHECK(ws >= 1 && ws <= 8, "world_size must be in [1, 8] (one node), got ", ws);

    Prepared prepared;
    prepared.x         = input.contiguous();
    const A2ADims dims = make_dims(prepared.x, mode, ws, static_cast<int>(group->rank()), seq_splits, head_splits);
    prepared.plan      = build_plan(dims, static_cast<int>(mode), prepared.x.element_size());

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
// SM-resident collective (all_to_all_ce.cu file header). It is the only transport, and nothing
// here picks between transports at runtime -- a rank-local timing decision taken inside a
// collective diverges the group.
const SymmetricHeapPool::Buffer& transfer_on_stream(const c10::intrusive_ptr<UlyssesGroup>& group,
                                                    const Prepared&                         prepared,
                                                    const std::string&                      tag,
                                                    bool                                    barrier,
                                                    cudaStream_t                            stream)
{
    // The window is sized for the largest rank (plan.window_numel); this rank's own result is a
    // dense prefix of it, so the borrowed view and the copy-out below are both built from
    // plan.output_shape, not from the window's capacity. Sizing the window per rank instead
    // would make the ranks disagree on offsets -- see A2APlan::window_numel and
    // SymmetricHeapPool::acquire.
    const auto& buf = group->pool().acquire(prepared.plan.window_numel, prepared.x.scalar_type(), tag);
    check_window_aliasing(prepared, buf, tag);

    // WRITERS WAIT FOR READERS, before writing anything.
    //
    // The window is single-buffered per tag, so this call is about to overwrite what the
    // previous call with this tag produced -- and a peer may still be reading its own copy of
    // it. The closing barrier below proves everyone's WRITES landed; it proves nothing about
    // everyone having finished READING. Without this one, a fast peer's next transfer lands in
    // our window while we are still consuming the last result.
    //
    // Not hypothetical: a2a_window_race.py fails without this, and its docstring is the control.
    //
    // It guards the START of a call rather than the end of the previous one because a BORROWED
    // result is read by the caller, at a time the operator never sees. Sitting here it covers
    // that and the copying form's own copy-out alike: either read is ordered ahead of it on
    // the same stream, and no peer can write until every rank has reached it.
    //
    // Costs one handshake per call; docs/BENCHMARK.md has the figure.
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
    // release store; both keep the flag on the data's path. It holds in testing, which is
    // evidence and not a guarantee -- a2a_ce_fault_injection.py is what keeps that test honest.
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

// Size a tag's window for a call that has not happened yet, so the allocation it would otherwise
// trigger mid-call happens here instead. Rationale and what was measured: SymmetricHeapPool::seal.
//
// Takes the shape and mode of the intended call rather than a byte count: the window is sized
// for the LARGEST rank's output, which only the plan knows (A2APlan::window_numel). Reserving
// the biggest shape a tag will see covers every smaller one, because acquire() matches on
// capacity rather than exact size.
//
// This is a collective: every rank must call it with the same tag, shape, mode and dtype.
void reserve(const c10::intrusive_ptr<UlyssesGroup>&    group,
             std::string                                tag,
             std::vector<int64_t>                       sizes,
             int64_t                                    mode,
             at::ScalarType                             dtype,
             const std::optional<std::vector<int64_t>>& seq_splits,
             const std::optional<std::vector<int64_t>>& head_splits)
{
    const int     ws   = static_cast<int>(group->world_size());
    const A2ADims dims = make_dims_from_shape(
        at::IntArrayRef(sizes), dtype, mode, ws, static_cast<int>(group->rank()), seq_splits, head_splits);
    const A2APlan plan = build_plan(dims, static_cast<int>(mode), static_cast<int64_t>(c10::elementSize(dtype)));
    group->pool().acquire(plan.window_numel, dtype, tag);
    // The tag's barrier flags come out of the same pool, so they have to be reserved too --
    // otherwise sealing turns the tag's first HANDSHAKE into the failure, not its first window.
    group->reserve_barrier(tag, at::cuda::getCurrentCUDAStream());
}

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
    check_window_aliasing(prepared, buf, tag);

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

    m.class_<ulysses::UlyssesGroup>("UlyssesGroup")
        .def(torch::init<std::vector<int64_t>, int64_t, int64_t, int64_t>())
        .def("seal_pool", [](const c10::intrusive_ptr<ulysses::UlyssesGroup>& g) { g->pool().seal(); })
        .def("barrier_epoch", &ulysses::UlyssesGroup::barrier_epoch)  // tests; see the declaration
        .def("destroy", &ulysses::UlyssesGroup::destroy)
        .def_static("uniqueid_nints", &ulysses::UlyssesGroup::uniqueid_nints)
        .def_static("get_uniqueid", &ulysses::UlyssesGroup::get_uniqueid)
        .def_static("init_world", &ulysses::UlyssesGroup::init_world);

    // Collective: pre-size a tag's window so no later call has to allocate. See ulysses::reserve.
    m.def("reserve(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, str tag, int[] sizes, "
          "int mode, ScalarType dtype, int[]? seq_splits=None, int[]? head_splits=None) -> ()");
    m.impl("reserve", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::reserve);

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
PYBIND11_MODULE(_C, m)
{
    // The async entry points are Python-side (comm.py owns the comm stream and the input
    // staging), so this is the one piece that has to be C++: c10d::Work is a C++ interface.
    // Not a torch op -- it takes a raw stream handle and mutates a process-wide registry,
    // neither of which belongs in a dispatcher schema. csrc/work.h has the contract.
    m.def("register_stream_completion", [](const at::Tensor& tensor, int64_t comm_stream) {
        return ulysses::register_stream_completion(tensor, reinterpret_cast<cudaStream_t>(comm_stream));
    });

    // TESTS ONLY. Underscored, and not a torch op, because arming it deliberately breaks the
    // operator: it is the negative control for a2a_ce_flag_ordering.py. See all_to_all_ce.cu.
    m.def("_set_ce_fault", &ulysses::set_ce_fault);

    // What this .so actually is. Diagnostics only -- nothing on the collective path reads it.
    //
    // The two that answer real questions: `cuda_arch_list`, because a .so without this GPU's
    // arch fails at load with a message that does not mention arch; and `nvshmem_loaded_from`,
    // resolved with dladdr rather than reported from the build, because torch ships its own
    // libnvshmem and which one won is exactly what a coexistence problem turns on.
    m.def("build_info", []() {
        Dl_info                            info{};
        std::map<std::string, std::string> out;
        out["version"]            = "0.1.0";
        out["cuda_arch_list"]     = FAST_ULYSSES_CUDA_ARCH_LIST;
        out["nvshmem_build_home"] = FAST_ULYSSES_NVSHMEM_HOME;
        out["nvshmem_loaded_from"] =
            dladdr(reinterpret_cast<void*>(&nvshmem_ptr), &info) && info.dli_fname ? info.dli_fname : "unknown";
        return out;
    });
}
