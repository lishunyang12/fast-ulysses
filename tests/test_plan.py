"""Layout contract for the 4D all-to-all, checked without a GPU or a process group.

The plan is a list of pitched copies (fast_ulysses/csrc/a2a_plan.cpp). Here we execute those
copies over plain byte buffers -- one symmetric window per simulated rank -- and compare the
result against the reference built from ``all_to_all_single`` + permute, which is the semantics
every Ulysses implementation agrees on. If this passes, the addressing is right and any later
failure is a transport or synchronisation bug, not a layout bug.

The window IS the output here: ``all_to_all_single_4d`` returns the window view, so every rank
sends its own share through the window like any other peer's and there is no copy-out stage to
replay.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
# Importing the package registers torch.ops.fast_ulysses; a2a_plan_debug needs no GPU.
pytest.importorskip("fast_ulysses", reason="fast_ulysses._C extension not built")

SCATTER_HEAD = 0
GATHER_HEAD = 1


def build_plan(b, d, rank, world_size, seq_splits, head_splits, mode, elem_size):
    output_shape, ops = torch.ops.fast_ulysses.a2a_plan_debug(
        b, d, rank, world_size, list(seq_splits), list(head_splits), mode, elem_size
    )
    output_shape = list(output_shape)
    return {
        "output_shape": output_shape,
        "output_bytes": int(np.prod(output_shape)) * elem_size,
        "ops": ops.numpy(),
    }


def apply_ops(ops, src, dst):
    for (
        _peer,
        src_off,
        dst_off,
        src_pitch,
        dst_pitch,
        width,
        rows,
        depth,
        src_slice,
        dst_slice,
    ) in ops:
        # depth folds the batch dimension into one op; replaying it is the same copy repeated
        # at a fixed stride, which is exactly what cudaMemcpy3DAsync does.
        for z in range(depth):
            for row in range(rows):
                s = src_off + z * src_slice + row * src_pitch
                t = dst_off + z * dst_slice + row * dst_pitch
                dst[t : t + width] = src[s : s + width]


def run_plan(inputs, plans):
    """Replay the device sequence on plain byte buffers: rank r writes into peer p's window."""
    windows = [np.zeros(p["output_bytes"], dtype=np.uint8) for p in plans]
    for src_rank, plan in enumerate(plans):
        src = inputs[src_rank].reshape(-1).view(np.uint8)
        for op in plan["ops"]:
            apply_ops([op], src, windows[op[0]])
    return windows


def reference_scatter_head(inputs, head_splits):
    """[b, s_r, n_total, d] per rank -> [b, s_total, n_p, d] per rank.

    Rank p's output stacks, over source ranks r in order, the head columns p owns from rank r's
    local sequence shard. That is exactly what all_to_all_single does after the scattered axis
    has been permuted to dim 0.
    """
    head_off = np.cumsum([0] + list(head_splits))
    return [
        np.concatenate(
            [x[:, :, head_off[p] : head_off[p + 1], :] for x in inputs],
            axis=1,
        )
        for p in range(len(inputs))
    ]


def reference_gather_head(inputs, seq_splits):
    """[b, s_total, n_r, d] per rank -> [b, s_p, n_total, d] per rank."""
    seq_off = np.cumsum([0] + list(seq_splits))
    return [
        np.concatenate(
            [x[:, seq_off[p] : seq_off[p + 1], :, :] for x in inputs],
            axis=2,
        )
        for p in range(len(inputs))
    ]


def make_inputs(b, d, seq_splits, head_splits, mode, dtype=np.float32):
    """Distinct values everywhere, so any misplaced byte shows up as a mismatch."""
    seq_total = int(sum(seq_splits))
    head_total = int(sum(head_splits))
    inputs = []
    base = 0
    for r in range(len(seq_splits)):
        if mode == SCATTER_HEAD:
            shape = (b, int(seq_splits[r]), head_total, d)
        else:
            shape = (b, seq_total, int(head_splits[r]), d)
        n = int(np.prod(shape))
        inputs.append((base + np.arange(n, dtype=dtype)).reshape(shape))
        base += n
    return inputs


def check(b, d, seq_splits, head_splits, mode, dtype=np.float32):
    world_size = len(seq_splits)
    elem_size = np.dtype(dtype).itemsize
    inputs = make_inputs(b, d, seq_splits, head_splits, mode, dtype)
    plans = [
        build_plan(b, d, r, world_size, seq_splits, head_splits, mode, elem_size)
        for r in range(world_size)
    ]
    raw = run_plan(inputs, plans)

    if mode == SCATTER_HEAD:
        expected = reference_scatter_head(inputs, head_splits)
    else:
        expected = reference_gather_head(inputs, seq_splits)

    for r in range(world_size):
        got = raw[r].view(dtype).reshape(plans[r]["output_shape"])
        assert got.shape == expected[r].shape, f"rank {r}: {got.shape} != {expected[r].shape}"
        np.testing.assert_array_equal(got, expected[r], err_msg=f"rank {r} mode {mode}")
    return plans


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("b", [1, 3])
@pytest.mark.parametrize("mode", [SCATTER_HEAD, GATHER_HEAD])
def test_even(world_size, b, mode):
    plans = check(b, 4, [5] * world_size, [2] * world_size, mode)
    # The uniform entry point allocates the window with a size-collective nvshmem_align, so
    # every rank must arrive at the same output size.
    assert {tuple(p["output_shape"]) for p in plans} == {tuple(plans[0]["output_shape"])}


@pytest.mark.parametrize("mode", [SCATTER_HEAD, GATHER_HEAD])
@pytest.mark.parametrize("b", [1, 2])
def test_uneven_sequence(mode, b):
    check(b, 3, [7, 5, 5, 4], [2, 2, 2, 2], mode)


@pytest.mark.parametrize("mode", [SCATTER_HEAD, GATHER_HEAD])
@pytest.mark.parametrize("b", [1, 2])
def test_uneven_heads(mode, b):
    check(b, 3, [4, 4, 4, 4], [3, 2, 2, 1], mode)


@pytest.mark.parametrize("mode", [SCATTER_HEAD, GATHER_HEAD])
def test_uneven_both(mode):
    check(2, 5, [9, 4, 6, 1], [4, 1, 3, 2], mode)


@pytest.mark.parametrize("mode", [SCATTER_HEAD, GATHER_HEAD])
def test_empty_shard(mode):
    """A rank holding no sequence at all is legal and must not emit copies for it."""
    check(1, 2, [3, 0, 5, 4], [2, 2, 2, 2], mode)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_dtypes(dtype):
    check(2, 4, [3, 5], [2, 4], SCATTER_HEAD, dtype)


def test_batch_fusion_rule():
    """push_batched folds the batch into ONE op, but only for multi-row copies."""
    multi = build_plan(3, 4, 0, 2, [2, 2], [2, 2], SCATTER_HEAD, 4)
    assert len(multi["ops"]) == 2  # one per peer
    assert set(multi["ops"][:, 7]) == {3}  # depth == b
    single = build_plan(3, 4, 0, 2, [1, 1], [2, 2], SCATTER_HEAD, 4)
    assert len(single["ops"]) == 2 * 3  # rows == 1: b separate 2D copies per peer
    assert set(single["ops"][:, 7]) == {1}


@pytest.mark.parametrize("mode", [SCATTER_HEAD, GATHER_HEAD])
@pytest.mark.parametrize(
    "seq_splits, head_splits",
    [([5, 5, 5, 5], [2, 2, 2, 2]), ([7, 5, 5, 4], [3, 2, 2, 1]), ([9, 4, 6, 1], [4, 1, 3, 2])],
)
def test_fused_ops_survive_cuda_memcpy_3d(mode, seq_splits, head_splits):
    """The one property the replay above cannot check.

    ``apply_ops`` steps the batch with ``src_slice`` directly, so a fused op with a slice
    stride cudaMemcpy3DParms cannot express still replays correctly here -- while the device
    copied the wrong bytes, because cudaMemcpy3DParms takes no slice stride and derives one as
    pitch * ysize (issue_copy passes ysize = slice / pitch). Assert the arithmetic that makes
    that derivation exact, on every op the plan chose to fuse.
    """
    fused = 0
    for rank in range(len(seq_splits)):
        plan = build_plan(3, 4, rank, len(seq_splits), seq_splits, head_splits, mode, 2)
        for op in plan["ops"]:
            _, _, _, src_pitch, dst_pitch, width, rows, depth, src_slice, dst_slice = op
            if depth == 1:
                continue
            fused += 1
            assert rows > 1, "push_batched must not fuse single-row copies"
            assert width <= src_pitch and width <= dst_pitch
            assert src_slice % src_pitch == 0 and dst_slice % dst_pitch == 0
            assert src_slice // src_pitch >= rows and dst_slice // dst_pitch >= rows
    assert fused > 0, "nothing was fused, so this checked nothing"


def test_round_trip():
    """gather(scatter(x)) == x, which is how the two modes are used around attention."""
    b, d, elem = 2, 3, 4
    seq_splits, head_splits = [5, 3, 4, 2], [2, 3, 1, 2]
    world_size = len(seq_splits)

    inputs = make_inputs(b, d, seq_splits, head_splits, SCATTER_HEAD)
    plans = [
        build_plan(b, d, r, world_size, seq_splits, head_splits, SCATTER_HEAD, elem)
        for r in range(world_size)
    ]
    mid_raw = run_plan(inputs, plans)
    mid = [mid_raw[r].view(np.float32).reshape(plans[r]["output_shape"]) for r in range(world_size)]

    back_plans = [
        build_plan(b, d, r, world_size, seq_splits, head_splits, GATHER_HEAD, elem)
        for r in range(world_size)
    ]
    back_raw = run_plan(mid, back_plans)
    for r in range(world_size):
        got = back_raw[r].view(np.float32).reshape(back_plans[r]["output_shape"])
        np.testing.assert_array_equal(got, inputs[r], err_msg=f"round trip rank {r}")


def test_matches_torch_all_to_all_single():
    """Tie the reference back to torch itself, not just to our own numpy restatement.

    Reproduces all_to_all_single's chunk semantics in one process (rank r's j-th chunk goes to
    rank j's r-th slot) around the same permutes sglang's usp.py uses, so the plan is validated
    against the collective every other Ulysses implementation is written to.
    """
    b, d, world_size = 2, 4, 4
    s_local, h_global = 3, 8
    h_local = h_global // world_size

    torch.manual_seed(0)
    full = torch.randn(b, s_local * world_size, h_global, d, dtype=torch.float32)
    inputs = [full.narrow(1, r * s_local, s_local).contiguous() for r in range(world_size)]

    # sglang's _usp_input_all_to_all, head_dim=2: permute heads to dim 0, exchange, then fold
    # world_size next to the sequence axis.
    sends = [x.permute(2, 0, 1, 3).contiguous().flatten() for x in inputs]
    chunks = [list(s.chunk(world_size)) for s in sends]
    torch_out = []
    for p in range(world_size):
        recv = torch.cat([chunks[r][p] for r in range(world_size)])
        y = recv.reshape(world_size, h_local, b, s_local, d)
        torch_out.append(
            y.permute(2, 0, 3, 1, 4).contiguous().reshape(b, s_local * world_size, h_local, d)
        )

    plans = [
        build_plan(
            b, d, r, world_size, [s_local] * world_size, [h_local] * world_size, SCATTER_HEAD, 4
        )
        for r in range(world_size)
    ]
    raw = run_plan([x.numpy() for x in inputs], plans)
    for r in range(world_size):
        got = raw[r].view(np.float32).reshape(plans[r]["output_shape"])
        np.testing.assert_array_equal(got, torch_out[r].numpy(), err_msg=f"rank {r}")


@pytest.mark.parametrize(
    "seq_splits, head_splits, message",
    [
        ([1, 2], [1], "head_splits has"),
        ([1], [1, 2], "seq_splits has"),
        ([1, -1], [1, 1], "negative"),
    ],
)
def test_invalid_splits_rejected(seq_splits, head_splits, message):
    with pytest.raises(Exception, match=message):
        build_plan(1, 2, 0, 2, seq_splits, head_splits, SCATTER_HEAD, 4)


def test_invalid_mode_rejected():
    with pytest.raises(Exception, match="mode must be"):
        build_plan(1, 2, 0, 2, [1, 1], [1, 1], 7, 4)
