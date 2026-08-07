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
// NOTE (alternatives tried and REVERTED):
// - cudaMemcpy3DBatchAsync: the driver's pitched BATCH path (many copies in one call, not
//   the plain cudaMemcpy3DAsync used above) is itself slow (0.82ms, and 1.35ms with
//   cudaMemcpyFlagPreferOverlapWithCompute); it also rejects the LEGACY default stream with
//   "invalid argument" (explicit streams required). Measured on CUDA 13.3, exclusive GPUs.
//
// This file used to record a second reverted alternative -- "single-stream serial
// submission: local copy loses its overlap with the remote copies, 0.82ms standalone" --
// and concluded the per-peer stream pool beat it. THAT CONCLUSION IS WRONG on 4xH200 and
// the fan-out it justified has been removed; see the measurements at the transfer below.
// The observation behind it was right (serialising SELF too does lose the local/remote
// overlap) but the fix is to keep self on the caller's stream, not to fan the remote copies
// out.
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
                   int                          rank,
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
    // ONE STREAM for the remote copies, and this rank's OWN share on the CALLER's stream.
    //
    // Fanning out per peer is the intuitive design and this file used to do it. Measured on
    // 4xH200 NVLink, exclusive GPUs, wan-720p / wan-480p / h3, ms per call:
    //
    //     fan out, one stream per peer              2.273  1.018  1.683
    //     everything serialised onto one stream     1.345  0.631  1.049   (1.7x faster)
    //     remote serialised, self on the caller's   1.175  0.542  0.932   (1.9x faster)
    //
    // Concurrent copies contend for the same egress and each runs slower than it would with
    // the link to itself, so serialising them is worth 1.7x on its own. The fan-out's stated
    // purpose was keeping the LOCAL copy concurrent with the remote ones -- that is real, and
    // it is why the file's old note recorded serialising as SLOWER, but it does not need a
    // stream pool: our own share crosses no link, so issuing it on the caller's stream gets
    // the same overlap for another 12-14% and no extra stream or event. custom_nccl_op
    // measured the same two effects independently (a2a_ce.cpp:23-42: 186.8 -> 244.5 GB/s
    // per-GPU egress serialised, and -24/-27% for hoisting self).
    //
    // Peers are visited in XOR-shift order, which nudges ranks into pairing up with no
    // cross-rank coordination; custom_nccl_op measured that alone at +14%, not re-measured
    // here.
    cudaEvent_t ready;
    ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    ULYSSES_CUDA_CHECK(cudaEventRecord(ready, stream));

    auto emit = [&](int p, cudaStream_t on) {
        for (const CopyOp& op : plan.ops) {
            if (op.peer == p) {
                issue_copy(reinterpret_cast<uint8_t*>(peer_ptrs[p]) + op.dst_offset, src_bytes + op.src_offset, op,
                           on);
            }
        }
    };

    cudaStream_t xfer = ce.streams[0];
    ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(xfer, ready, 0));
    for (int k = 1; k < ws; ++k) {
        const int peer = rank ^ k;
        if (peer < ws) {
            emit(peer, xfer);
        }
    }
    // XOR only enumerates every peer when ws is a power of two; sweep for any it missed.
    if ((ws & (ws - 1)) != 0) {
        for (int p = 0; p < ws; ++p) {
            if (p != rank && (p ^ rank) >= ws) {
                emit(p, xfer);
            }
        }
    }
    emit(rank, stream);

    cudaEvent_t done;
    ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    ULYSSES_CUDA_CHECK(cudaEventRecord(done, xfer));
    ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(stream, done, 0));
    ULYSSES_CUDA_CHECK(cudaEventDestroy(done));
    ULYSSES_CUDA_CHECK(cudaEventDestroy(ready));
}

}  // namespace ulysses
