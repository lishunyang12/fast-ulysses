#pragma once
// Addressing for the 4D all-to-all, expressed as pitched copies. Pure host arithmetic with no
// CUDA and no NVSHMEM in it, so the layout contract can be tested on its own (see
// tests/test_plan.py, which replays the plan over numpy buffers and compares against torch's
// all_to_all_single + permute). The transport in all_to_all_ce.cu only turns these ops into
// cudaMemcpy calls; it computes no offsets of its own.
//
// UNEVEN splits are the general case: even splits are just seq_splits = [s/P]*P and
// head_splits = [n/P]*P, so there is a single code path to get right. The operator entry point
// (check_uniform_args in bindings.cpp) builds even splits only; the uneven arithmetic is what
// the host tests exercise.
#include <cstdint>
#include <vector>

namespace ulysses {

enum A2AMode : int {
    // [b, seq_splits[me], head_total, d] -> [b, seq_total, head_splits[me], d]
    kScatterHead = 0,
    // the inverse: [b, seq_total, head_splits[me], d] -> [b, seq_splits[me], head_total, d]
    kGatherHead = 1,
};

struct A2ADims {
    int64_t              b          = 0;
    int64_t              d          = 0;
    int                  rank       = 0;
    int                  world_size = 0;
    std::vector<int64_t> seq_splits;   // per-rank local sequence length
    std::vector<int64_t> head_splits;  // per-rank head count

    int64_t              seq_total() const;
    int64_t              head_total() const;
    std::vector<int64_t> seq_offsets() const;   // exclusive prefix sum of seq_splits
    std::vector<int64_t> head_offsets() const;  // exclusive prefix sum of head_splits

    // Throws std::invalid_argument naming the offending field. Call before anything allocates.
    void validate() const;
};

// One pitched copy. Offsets are byte offsets from the base of their buffer -- this rank's input
// tensor and the destination peer's symmetric window -- so the same struct describes a
// cudaMemcpy2D/3DAsync and a host-side replay in a test.
//
// `depth` folds the batch dimension in. Every batch element repeats the same rows x width copy
// at a fixed stride, so b of them can go as ONE cudaMemcpy3DAsync instead of b
// cudaMemcpy2DAsync calls -- same bytes on the device, b-1 fewer launches on the host.
//
// Not every batched op can be expressed this way: cudaMemcpy3DParms derives its slice stride as
// pitch * ysize rather than taking it directly, so the stride has to be a whole multiple of the
// pitch with room for the rows. `push_batched` checks and falls back to separate 2D copies when
// it is not. See a2a_plan.cpp.
struct CopyOp {
    int     peer       = 0;
    int64_t src_offset = 0;
    int64_t dst_offset = 0;
    int64_t src_pitch  = 0;
    int64_t dst_pitch  = 0;
    int64_t width      = 0;  // bytes per row
    int64_t rows       = 0;
    int64_t depth      = 1;  // batch elements folded into this op
    int64_t src_slice  = 0;  // bytes between batch elements, source
    int64_t dst_slice  = 0;  // bytes between batch elements, destination
};

struct A2APlan {
    std::vector<int64_t> output_shape;  // shape of the symmetric window this rank receives into
    std::vector<CopyOp>  ops;           // what THIS rank sends; ops[i].peer says where
};

// The result IS the symmetric window (all_to_all_single_4d returns the window view), so the
// window already holds the output layout and this rank's own share travels through it like
// every other peer's: `ops` covers all world_size destinations and there is no copy-out.
A2APlan build_plan(const A2ADims& dims, int mode, int64_t elem_size);

}  // namespace ulysses
