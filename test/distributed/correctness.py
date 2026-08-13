"""Small distributed correctness check for both directions."""

import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    heads = 2 * ws
    seq = 8 * ws
    x = (
        torch.arange(
            rank * (seq // ws) * heads * 16,
            (rank + 1) * (seq // ws) * heads * 16,
            device=local_rank,
            dtype=torch.float32,
        )
        .to(torch.bfloat16)
        .view(1, seq // ws, heads, 16)
    )

    group = UlyssesGroup(device=local_rank)
    y = group.allocate_output(x, mode=0)
    group.exchange(x, y, mode=0)
    z = group.allocate_output(y, mode=1)
    group.exchange(y, z, mode=1)
    if not torch.equal(x, z):
        raise RuntimeError(f"rank {rank}: round trip failed")
    if rank == 0:
        print(f"PASS world_size={ws} backend={group.backend}")
    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
