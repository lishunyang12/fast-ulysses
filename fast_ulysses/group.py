from __future__ import annotations

import warnings

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


class UlyssesGroup:
    """Equal-split, inference-only Ulysses all-to-all."""

    def __init__(self, process_group=None, device=None):
        self.pg = process_group or dist.group.WORLD
        self.rank = dist.get_rank(self.pg)
        self.world_size = dist.get_world_size(self.pg)
        self.device = torch.device("cuda" if device is None else device)
        if self.device.type != "cuda":
            raise ValueError("device must be CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(self.device)

        local = torch.tensor([self.device.index], device=self.device)
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local, group=self.pg)
        devices = [int(value.item()) for value in gathered]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            symm_mem.enable_symm_mem_for_group(self.pg.group_name)
        self._group = torch.classes.fast_ulysses.UlyssesGroup(
            self.pg.group_name,
            self.rank,
            self.world_size,
            self.device.index,
            devices,
        )
        self.backend = self._group.backend()
        if self.backend == "mlx5":
            self._sync = torch.zeros(1, dtype=torch.int32, device=self.device)
            peers = [None] * self.world_size
            dist.all_gather_object(peers, self._group.connection_info(), group=self.pg)
            self._group.connect(peers)
        self._destroyed = False

    def allocate_output(self, x: torch.Tensor, mode: int = 0) -> torch.Tensor:
        self._check_alive()
        output = self._group.allocate_output(x, mode)
        if self.backend == "mlx5":
            peers = [None] * self.world_size
            dist.all_gather_object(peers, self._group.buffer_info(output), group=self.pg)
            self._group.connect_buffer(output, peers)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.pg, device_ids=[self.device.index])
        return output

    def release_output(self, output: torch.Tensor) -> None:
        """Drop the windows behind `output`. Collective: every rank must release in step."""
        self._check_alive()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.pg, device_ids=[self.device.index])
        self._group.release_output(output)

    def exchange(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        mode: int = 0,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        self._check_alive()
        selected = stream or torch.cuda.current_stream(self.device)
        if torch.device(selected.device) != self.device:
            raise ValueError("stream is on the wrong GPU")
        if self.backend == "mlx5":
            # The mlx5 transport has no handshake of its own, in either direction: nothing tells
            # a peer that its buffer is full, and nothing keeps this call off the buffer a peer
            # is still reading. Both host round trips stand in for one of those. This is a
            # stopgap, not the design, and it costs two collectives per call.
            selected.synchronize()
            dist.all_reduce(self._sync, group=self.pg)
            # The NIC is driven from the host, so an opening barrier only holds it back if the
            # host waits for it. `all_reduce` alone just queues a kernel and returns.
            torch.cuda.synchronize(self.device)
            self._group.exchange(x, output, mode, selected.cuda_stream)
            selected.synchronize()
            dist.all_reduce(self._sync, group=self.pg)
            torch.cuda.synchronize(self.device)
            self._group.flush()
        else:
            self._group.exchange(x, output, mode, selected.cuda_stream)
        return output

    def destroy(self):
        if self._destroyed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.pg, device_ids=[self.device.index])
        self._group.destroy()
        self._destroyed = True

    def _check_alive(self):
        if self._destroyed:
            raise RuntimeError("group is destroyed")


__all__ = ["UlyssesGroup"]
