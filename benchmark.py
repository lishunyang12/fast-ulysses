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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--report",
        type=str,
        default="",
        metavar="PATH",
        help="also write a Markdown report to PATH",
    )
    return parser.parse_args()


def parse_shape(text: str) -> tuple[int, int, int]:
    values = tuple(int(v) for v in text.split(","))
    if len(values) != 3:
        raise ValueError("shape must be SEQ,HEADS,HEAD_DIM")
    return values


def timed(fn, warmup: int, iters: int, trials: int, device: int) -> float:
    for _ in range(warmup):
        dist.barrier(device_ids=[device])
        fn()
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(trials):
        total_ms = 0.0
        for _ in range(iters):
            dist.barrier(device_ids=[device])
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize(device)
            elapsed = torch.tensor(
                [(time.perf_counter() - start) * 1000],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            total_ms += elapsed.item()
        samples.append(total_ms / iters)
    return statistics.median(samples)


def gbps(byte_count: float, milliseconds: float) -> float:
    return byte_count / (milliseconds / 1000) / 1e9


def write_report(path: str, metadata: list[str], rows: list[dict]) -> None:
    lines = [
        "# fast-ulysses benchmark report",
        "",
        *[f"- {item}" for item in metadata],
        "",
        "All bandwidths are decimal GB/s. `bus` counts only bytes sent to remote "
        "ranks; `aggregate` is the sum across all ranks.",
        "",
        "| Shape | Mode | Raw NCCL ms | NCCL alg GB/s | NCCL bus GB/s | "
        "NCCL aggregate GB/s | NCCL + layout ms | Layout GB/s | Fast ms | "
        "Fast GB/s | Raw / fast | Layout / fast |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['shape']} | {row['mode']} | {row['raw_ms']:.3f} | "
            f"{row['raw_alg_gbps']:.2f} | {row['raw_bus_gbps']:.2f} | "
            f"{row['raw_aggregate_gbps']:.2f} | {row['layout_ms']:.3f} | "
            f"{row['layout_gbps']:.2f} | {row['fast_ms']:.3f} | "
            f"{row['fast_gbps']:.2f} | {row['raw_ms'] / row['fast_ms']:.2f}x | "
            f"{row['layout_ms'] / row['fast_ms']:.2f}x |"
        )
    with open(path, "w", encoding="utf-8") as report:
        report.write("\n".join(lines) + "\n")


def nccl_mode0(
    x: torch.Tensor,
    send: torch.Tensor,
    recv: torch.Tensor,
    output: torch.Tensor,
    ws: int,
) -> torch.Tensor:
    b, s_local, h_global, d = x.shape
    h_local = h_global // ws
    send.view(h_global, b, s_local, d).copy_(x.permute(2, 0, 1, 3))
    dist.all_to_all_single(recv, send)
    output.view(b, ws, s_local, h_local, d).copy_(
        recv.view(ws, h_local, b, s_local, d).permute(2, 0, 3, 1, 4)
    )
    return output


def nccl_mode1(
    x: torch.Tensor,
    send: torch.Tensor,
    recv: torch.Tensor,
    output: torch.Tensor,
    ws: int,
) -> torch.Tensor:
    b, s_global, h_local, d = x.shape
    s_local = s_global // ws
    send.view(ws, b, s_local, h_local, d).copy_(
        x.view(b, ws, s_local, h_local, d).permute(1, 0, 2, 3, 4)
    )
    dist.all_to_all_single(recv, send)
    output.view(b, s_local, ws, h_local, d).copy_(
        recv.view(ws, b, s_local, h_local, d).permute(1, 2, 0, 3, 4)
    )
    return output


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
    metadata = [
        f"world_size: {ws}",
        "dtype: bfloat16",
        f"backend: {group.backend}",
        f"NCCL_P2P_LEVEL: {os.getenv('NCCL_P2P_LEVEL', '<unset>')}",
        f"warmup: {args.warmup} calls/case",
        f"measurement: {args.iters} call(s)/trial, {args.trials} trials, "
        "slowest rank then median",
    ]
    report_rows = []
    if rank == 0:
        print(f"# world_size={ws} dtype=bfloat16 backend={group.backend}")
        print(f"# NCCL_P2P_LEVEL={os.getenv('NCCL_P2P_LEVEL', '<unset>')}")
        print(
            f"# warmup={args.warmup}/case iters={args.iters}/trial "
            f"trials={args.trials} rank_reduce=max summary=median"
        )
        print(
            "# GB/s = per-rank remote payload (NCCL bus bandwidth); "
            "raw alg GB/s = GB/s * world_size / (world_size - 1)"
        )
        print(
            f"{'shape':<18} {'mode':<6} {'raw ms':>8} {'raw GB/s':>9} "
            f"{'layout ms':>9} {'layout GB/s':>11} {'fast ms':>8} "
            f"{'fast GB/s':>9} {'vs raw':>8} {'vs layout':>10}"
        )

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
        send_mode0 = torch.empty(x.numel(), dtype=x.dtype, device=x.device)
        recv_mode0 = torch.empty_like(send_mode0)
        mode0_ref = torch.empty(
            (1, seq, heads // ws, dim), dtype=x.dtype, device=x.device
        )
        nccl_mode0(x, send_mode0, recv_mode0, mode0_ref, ws)
        out_mode0 = group.exchange(x, mode=0)
        if not torch.equal(out_mode0, mode0_ref):
            raise RuntimeError(f"rank {rank}: mode=0 mismatch for {seq, heads, dim}")

        send_mode1 = torch.empty_like(send_mode0)
        recv_mode1 = torch.empty_like(send_mode0)
        mode1_ref = torch.empty_like(x)
        nccl_mode1(mode0_ref, send_mode1, recv_mode1, mode1_ref, ws)
        out_mode1 = group.exchange(out_mode0, mode=1)
        if not torch.equal(out_mode1, x) or not torch.equal(mode1_ref, x):
            raise RuntimeError(f"rank {rank}: mode=1 mismatch for {seq, heads, dim}")

        raw_recv_mode0 = torch.empty_like(recv_mode0)
        raw_recv_mode1 = torch.empty_like(recv_mode1)
        cases = {
            "raw_mode0": partial(
                dist.all_to_all_single, raw_recv_mode0, send_mode0
            ),
            "layout_mode0": partial(
                nccl_mode0, x, send_mode0, recv_mode0, mode0_ref, ws
            ),
            "fast_mode0": partial(group.exchange, x, mode=0),
            "raw_mode1": partial(
                dist.all_to_all_single, raw_recv_mode1, send_mode1
            ),
            "layout_mode1": partial(
                nccl_mode1, mode0_ref, send_mode1, recv_mode1, mode1_ref, ws
            ),
            "fast_mode1": partial(group.exchange, out_mode0, mode=1),
        }
        results = {
            name: timed(fn, args.warmup, args.iters, args.trials, local_rank)
            for name, fn in cases.items()
        }
        tensor_bytes = x.numel() * x.element_size()
        remote_bytes = tensor_bytes * (ws - 1) / ws
        if rank == 0:
            label = f"{seq},{heads},{dim}"
            for mode in (0, 1):
                raw = results[f"raw_mode{mode}"]
                layout = results[f"layout_mode{mode}"]
                fast = results[f"fast_mode{mode}"]
                raw_bus_gbps = gbps(remote_bytes, raw)
                layout_gbps = gbps(remote_bytes, layout)
                fast_gbps = gbps(remote_bytes, fast)
                print(
                    f"{label:<18} {mode:<6} {raw:8.3f} "
                    f"{raw_bus_gbps:9.2f} {layout:9.3f} {layout_gbps:11.2f} "
                    f"{fast:8.3f} {fast_gbps:9.2f} {raw / fast:7.2f}x "
                    f"{layout / fast:9.2f}x"
                )
                report_rows.append(
                    {
                        "shape": label,
                        "mode": mode,
                        "raw_ms": raw,
                        "raw_alg_gbps": gbps(tensor_bytes, raw),
                        "raw_bus_gbps": raw_bus_gbps,
                        "raw_aggregate_gbps": raw_bus_gbps * ws,
                        "layout_ms": layout,
                        "layout_gbps": layout_gbps,
                        "fast_ms": fast,
                        "fast_gbps": fast_gbps,
                    }
                )

    if rank == 0 and args.report:
        write_report(args.report, metadata, report_rows)
        print(f"# wrote Markdown report: {args.report}")

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
