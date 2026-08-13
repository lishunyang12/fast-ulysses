"""Benchmark minimal fast-ulysses against the equivalent NCCL layout path."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from functools import partial

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", default=[])
    parser.add_argument("--seq-len", type=int, default=37824)
    parser.add_argument("--num-heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--common-shapes", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    return parser.parse_args()


def parse_shape(text: str) -> tuple[int, int, int]:
    values = tuple(int(v) for v in text.split(","))
    if len(values) != 3:
        raise ValueError("shape must be SEQ,HEADS,HEAD_DIM")
    return values


def timed(fn, warmup: int, iters: int, trials: int, device: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(trials):
        dist.barrier(device_ids=[device])
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(device)
        elapsed = torch.tensor(
            [(time.perf_counter() - start) * 1000 / iters],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(elapsed.item())
    return statistics.median(samples)


def nccl_forward(x: torch.Tensor, recv: torch.Tensor, ws: int) -> torch.Tensor:
    b, s_local, h_global, d = x.shape
    h_local = h_global // ws
    send = x.permute(2, 0, 1, 3).contiguous().flatten()
    dist.all_to_all_single(recv, send)
    return (
        recv.view(ws, h_local, b, s_local, d)
        .permute(2, 0, 3, 1, 4)
        .contiguous()
        .view(b, s_local * ws, h_local, d)
    )


def nccl_reverse(x: torch.Tensor, recv: torch.Tensor, ws: int) -> torch.Tensor:
    b, s_global, h_local, d = x.shape
    s_local = s_global // ws
    h_global = h_local * ws
    send = (
        x.permute(2, 0, 1, 3)
        .contiguous()
        .view(h_local, ws, s_local, b, d)
        .transpose(0, 1)
        .contiguous()
        .view(h_global, s_local, b, d)
        .flatten()
    )
    dist.all_to_all_single(recv, send)
    return recv.view(h_global, s_local, b, d).transpose(0, 2).contiguous()


def keep_result(holder: list[torch.Tensor | None], fn, *args) -> None:
    holder[0] = fn(*args)


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    if args.shape:
        shapes = [parse_shape(s) for s in args.shape]
    elif args.common_shapes:
        shapes = [(37824, 56, 128), (75600, 40, 128), (32760, 40, 128)]
    else:
        shapes = [(args.seq_len, args.num_heads, args.head_dim)]

    group = UlyssesGroup(device=local_rank)
    if rank == 0:
        print(f"# world_size={ws} dtype=bfloat16 backend={group.backend}")
        print(f"# NCCL_P2P_LEVEL={os.getenv('NCCL_P2P_LEVEL', '<unset>')}")
        print(f"{'shape':<20} {'case':<12} {'ms':>10} {'remote GB/s':>14} {'vs NCCL':>10}")

    for seq, heads, dim in shapes:
        if seq % ws or heads % ws:
            if rank == 0:
                print(f"{seq},{heads},{dim}: skipped (not divisible by {ws})")
            continue
        x = torch.randn(
            (1, seq // ws, heads, dim),
            dtype=torch.bfloat16,
            device=local_rank,
        )
        recv_fwd = torch.empty(x.numel(), dtype=x.dtype, device=x.device)
        y_ref = nccl_forward(x, recv_fwd, ws)
        out_fwd = group.allocate_output(x, mode=0)
        group.exchange(x, out_fwd, mode=0)
        if not torch.equal(out_fwd, y_ref):
            raise RuntimeError(f"rank {rank}: forward mismatch for {seq, heads, dim}")

        recv_rev = torch.empty_like(recv_fwd)
        x_ref = nccl_reverse(y_ref, recv_rev, ws)
        out_rev = group.allocate_output(out_fwd, mode=1)
        group.exchange(out_fwd, out_rev, mode=1)
        if not torch.equal(out_rev, x) or not torch.equal(x_ref, x):
            raise RuntimeError(f"rank {rank}: reverse mismatch for {seq, heads, dim}")

        raw_send = torch.empty_like(recv_fwd)
        raw_recv = torch.empty_like(recv_fwd)
        holder: list[torch.Tensor | None] = [None]
        cases = {
            "nccl_raw": partial(dist.all_to_all_single, raw_recv, raw_send),
            "nccl_fwd": partial(keep_result, holder, nccl_forward, x, recv_fwd, ws),
            "fast_fwd": partial(group.exchange, x, out_fwd, mode=0),
            "nccl_rev": partial(keep_result, holder, nccl_reverse, y_ref, recv_rev, ws),
            "fast_rev": partial(group.exchange, out_fwd, out_rev, mode=1),
        }
        results = {
            name: timed(fn, args.warmup, args.iters, args.trials, local_rank)
            for name, fn in cases.items()
        }
        remote_bytes = x.numel() * x.element_size() * (ws - 1) / ws
        if rank == 0:
            label = f"{seq},{heads},{dim}"
            for name, ms in results.items():
                gbps = remote_bytes / (ms / 1000) / 1e9
                base = results["nccl_fwd" if "fwd" in name else "nccl_rev"]
                ratio = base / ms if name != "nccl_raw" else float("nan")
                shown = "-" if name == "nccl_raw" else f"{ratio:.2f}x"
                print(f"{label:<20} {name:<12} {ms:10.3f} {gbps:14.2f} {shown:>10}")

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
