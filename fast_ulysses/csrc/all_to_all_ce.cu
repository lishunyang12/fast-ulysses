// CE (copy-engine) transfer path for the uniform 4D all-to-all: per-peer pitched
// cudaMemcpy2D/3DAsync fan-out on dedicated streams, joined back to the launching stream
// with events. The data movement uses DMA engines and zero SMs, so -- unlike the
// SM-resident scatter or the (1-block) TMA kernel, which cannot get a block slot while
// e.g. cuBLAS nvjet GEMMs hold every SM -- it runs at full NVLink bandwidth concurrently
// with compute. Measured (exclusive 4xH100/4xH200, Wan ws=4): standalone ~0.69ms
// (209 GB/s, vs 385 GB/s for a bare pitched peer memcpy), but 93-94% of it overlaps a
// concurrent GEMM chain where the kernel paths overlap ~25-38% -- net exposed time
// ~0.05ms/call vs ~0.36-0.38ms.
//
// This file computes no offsets: the (peer, offset, pitch, rows) addressing is built by
// build_plan in a2a_plan.cpp, which is host-only and tested without a GPU (tests/test_plan.py).
// A plan op may carry the whole batch (CopyOp::depth), in which case it issues as one
// cudaMemcpy3DAsync instead of b cudaMemcpy2DAsync calls; at b == 1 the plan never fuses, so
// the calls are exactly the ones this path has always issued.
//
// NOTE (measured on CUDA 13.3, exclusive GPUs; alternatives tried and REVERTED):
// - single-stream serial submission: local copy loses its overlap with the remote
//   copies -- 0.82ms standalone, no upside elsewhere;
// - cudaMemcpy3DBatchAsync: the driver's pitched BATCH path (many copies in one call, not
//   the plain cudaMemcpy3DAsync used above) is itself slow (0.82ms, and 1.35ms with
//   cudaMemcpyFlagPreferOverlapWithCompute); it also rejects the LEGACY default stream with
//   "invalid argument" (explicit streams required).
// The per-peer stream pool below beat both on standalone time at equal hiding.
#include "a2a_plan.h"
#include "ulysses_group.cuh"

namespace ulysses {

namespace {

// One CopyOp -> one CUDA call. depth > 1 is the batch dimension folded in by the plan;
// cudaMemcpy3DParms takes no slice stride and derives one as pitch * ysize, which is why
// push_batched only fuses when the strides divide the pitches (a2a_plan.cpp).
void issue_copy(void* dst, const void* src, const CopyOp& op, cudaStream_t stream)
{
    if (op.depth <= 1) {
        ULYSSES_CUDA_CHECK(cudaMemcpy2DAsync(dst,
                                             static_cast<size_t>(op.dst_pitch),
                                             src,
                                             static_cast<size_t>(op.src_pitch),
                                             static_cast<size_t>(op.width),
                                             static_cast<size_t>(op.rows),
                                             cudaMemcpyDefault,
                                             stream));
        return;
    }
    cudaMemcpy3DParms parms = {};
    parms.srcPtr            = make_cudaPitchedPtr(const_cast<void*>(src),
                                       static_cast<size_t>(op.src_pitch),
                                       static_cast<size_t>(op.width),
                                       static_cast<size_t>(op.src_slice / op.src_pitch));
    parms.dstPtr            = make_cudaPitchedPtr(dst,
                                       static_cast<size_t>(op.dst_pitch),
                                       static_cast<size_t>(op.width),
                                       static_cast<size_t>(op.dst_slice / op.dst_pitch));
    parms.extent =
        make_cudaExtent(static_cast<size_t>(op.width), static_cast<size_t>(op.rows), static_cast<size_t>(op.depth));
    parms.kind = cudaMemcpyDefault;
    ULYSSES_CUDA_CHECK(cudaMemcpy3DAsync(&parms, stream));
}

}  // namespace

void launch_a2a_ce(const void*                  src,
                   const std::vector<uint64_t>& peer_ptrs,
                   const A2APlan&               plan,
                   const CEResources&           ce,
                   cudaStream_t                 stream)
{
    const int      ws        = static_cast<int>(peer_ptrs.size());
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    // Fan out: every CE stream waits for the launching stream (inputs ready), copies its
    // peer's slice, and the launching stream joins all copies before the caller's barrier.
    // The shared source-egress NVLink port caps aggregate bandwidth regardless of stream
    // count; the pool's value is keeping the LOCAL copy concurrent with the remote ones.
    //
    // FRESH events every call -- do not hoist them into CEResources. Re-recording a shared
    // event that still has in-flight stream waits (deep enqueue-ahead: many deferred
    // barrier=False groups queued behind the device) lets a pending wait resolve against a
    // LATER record whose completion depends on this very stream progressing -- a circular
    // wait that deadlocks the group (reproduced at ws=2 with a few undrained groups).
    // Create/destroy is a few us per call and depth-safe: the waits capture the dependency
    // at call time, and destroy defers until the event retires.
    cudaEvent_t ready;
    ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    ULYSSES_CUDA_CHECK(cudaEventRecord(ready, stream));
    for (int p = 0; p < ws; ++p) {
        cudaStream_t cs = ce.streams[p];
        ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(cs, ready, 0));
        for (const CopyOp& op : plan.ops) {
            if (op.peer != p) {
                continue;
            }
            issue_copy(reinterpret_cast<uint8_t*>(peer_ptrs[p]) + op.dst_offset, src_bytes + op.src_offset, op, cs);
        }
        cudaEvent_t done;
        ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        ULYSSES_CUDA_CHECK(cudaEventRecord(done, cs));
        ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(stream, done, 0));
        ULYSSES_CUDA_CHECK(cudaEventDestroy(done));
    }
    ULYSSES_CUDA_CHECK(cudaEventDestroy(ready));
}

}  // namespace ulysses
