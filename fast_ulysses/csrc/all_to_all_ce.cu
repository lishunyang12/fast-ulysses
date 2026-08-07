// CE (copy-engine) transfer path for the 4D all-to-all: pitched cudaMemcpy2D/3DAsync straight
// into peers' symmetric-heap addresses. The remote copies are serialised onto one stream and
// this rank's own share goes on the caller's stream.
//
// The data movement uses DMA engines and zero SMs, so -- unlike an SM-resident collective, which
// cannot get a block slot while e.g. cuBLAS GEMMs hold every SM -- it runs concurrently with
// compute rather than behind it.
//
// This file computes no offsets: the (peer, offset, pitch, rows) addressing is built by
// build_plan in a2a_plan.cpp, which is host-only and tested without a GPU (tests/test_plan.py).
// A plan op may carry the whole batch (CopyOp::depth), in which case it issues as one
// cudaMemcpy3DAsync instead of b cudaMemcpy2DAsync calls; at b == 1 the plan never fuses, so
// the calls are exactly the ones this path has always issued.
//
// cudaMemcpy3DBatchAsync -- the driver's pitched batch entry point, as opposed to the plain
// cudaMemcpy3DAsync used here -- was tried and is not used: it is slower, and it rejects the
// legacy default stream. docs/BENCHMARK.md has the figures.
#include "a2a_plan.h"
#include "ulysses_group.cuh"

#include <chrono>
#include <thread>

namespace ulysses {

namespace {

// Debug-only, armed from Python. 0 = off, which is the only state a normal build ever sees.
int64_t g_fault_delay_us = 0;

void CUDART_CB delay_payload(void* arg)
{
    std::this_thread::sleep_for(std::chrono::microseconds(reinterpret_cast<int64_t>(arg)));
}

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

// Arm (delay_us > 0) or disarm (0) the "signal before payload" fault. TESTS ONLY.
//
// This operator depends on copy-engine writes being visible at the destination by the time the
// barrier kernel's release store announcing them arrives. That is undocumented, so the test for
// it is worth exactly as much as its negative control -- and the control otherwise means editing
// this file and rebuilding, which is something a person has to remember to do.
//
// Armed, launch_a2a_ce holds the payload on the transfer stream and does not join it back onto
// the caller's stream, so the closing barrier publishes while the bytes are still in flight and
// readers must see stale data. A test that arms it, requires a tear, disarms it and requires
// none therefore carries its own control on every run. See a2a_ce_fault_injection.py.
void set_ce_fault(int64_t delay_us)
{
    g_fault_delay_us = delay_us;
}

void launch_a2a_ce(const void*                  src,
                   const std::vector<uint64_t>& peer_ptrs,
                   const A2APlan&               plan,
                   const CEResources&           ce,
                   int                          rank,
                   cudaStream_t                 stream)
{
    const int      ws        = static_cast<int>(peer_ptrs.size());
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    // FRESH events every call -- do not hoist them into CEResources. Re-recording a shared
    // event that still has in-flight stream waits (deep enqueue-ahead: many deferred
    // barrier=False groups queued behind the device) lets a pending wait resolve against a
    // LATER record whose completion depends on this very stream progressing -- a circular
    // wait that deadlocks the group.
    // Create/destroy is a few us per call and depth-safe: the waits capture the dependency
    // at call time, and destroy defers until the event retires.
    // ONE STREAM for the remote copies, and this rank's OWN share on the CALLER's stream.
    //
    // The remote copies all leave through the same egress, so giving each peer its own stream
    // only makes every copy slower; serialising them removes that contention for free.
    //
    // This rank's own share crosses no link, so it CAN run alongside the remote ones -- and
    // getting that overlap needs no stream of its own, just the caller's. Hence the separate
    // emit below.
    //
    // Peers are visited in XOR-shift order, which pairs ranks up without any cross-rank
    // coordination. docs/BENCHMARK.md has the figures for all three effects.
    cudaEvent_t ready;
    ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    ULYSSES_CUDA_CHECK(cudaEventRecord(ready, stream));

    auto emit = [&](int p, cudaStream_t on) {
        for (const CopyOp& op : plan.ops) {
            if (op.peer == p) {
                issue_copy(reinterpret_cast<uint8_t*>(peer_ptrs[p]) + op.dst_offset, src_bytes + op.src_offset, op, on);
            }
        }
    };

    cudaStream_t xfer = ce.xfer;
    ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(xfer, ready, 0));
    // Fault injection (see set_ce_fault): hold the payload back so the flag cannot be behind it.
    if (g_fault_delay_us > 0) {
        ULYSSES_CUDA_CHECK(cudaLaunchHostFunc(xfer, delay_payload, reinterpret_cast<void*>(g_fault_delay_us)));
    }
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
    // Skipping this join is the fault: the caller's stream reaches the closing barrier, and
    // publishes, without waiting for the remote copies. Everything else about the call is
    // unchanged, so what the test observes is the ordering and nothing else.
    if (g_fault_delay_us == 0) {
        ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(stream, done, 0));
    }
    ULYSSES_CUDA_CHECK(cudaEventDestroy(done));
    ULYSSES_CUDA_CHECK(cudaEventDestroy(ready));
}

}  // namespace ulysses
