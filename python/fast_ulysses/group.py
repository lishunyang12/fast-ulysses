"""Small inference-only Ulysses all-to-all over symmetric GPU memory."""

from __future__ import annotations

import warnings

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


class UlyssesGroup:
    """Equal-split 4D Ulysses exchange using direct peer writes.

    The NVLink/native-atomic backend queues device barriers and copies on the
    selected stream.  The PCIe backend uses host-visible process-group barriers
    around synchronized peer copies; it is deliberately blocking and safe.
    """

    def __init__(
        self,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | str | int | None = None,
    ) -> None:
        self.pg = process_group if process_group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.pg)
        self.world_size = dist.get_world_size(self.pg)
        self.device = torch.device("cuda" if device is None else device)
        if self.device.type != "cuda":
            raise ValueError(f"device must be CUDA, got {self.device}")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(self.device)

        local = torch.tensor([self.device.index], dtype=torch.int64, device=self.device)
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local, group=self.pg)
        devices = [int(item.item()) for item in gathered]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            symm_mem.enable_symm_mem_for_group(self.pg.group_name)

        self._handle = torch.classes.fast_ulysses.UlyssesGroup(
            self.pg.group_name,
            self.rank,
            self.world_size,
            self.device.index,
            devices,
        )
        self.backend = self._handle.backend()
        self._destroyed = False

    def allocate_output(self, x: torch.Tensor, *, mode: int = 0) -> torch.Tensor:
        """Collectively allocate a symmetric output buffer for ``x`` and ``mode``."""
        self._check_alive()
        out = self._handle.allocate_output(x, mode)
        # Allocation is outside the hot path.  A host-visible rendezvous here
        # keeps setup free of persistent GPU spin barriers on PCIe systems.
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.pg, device_ids=[self.device.index])
        return out

    def exchange(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        *,
        mode: int = 0,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Write the equal-split Ulysses exchange directly into ``out``.

        ``out`` must come from :meth:`allocate_output`.  On the device backend
        this only submits work; when a different stream is supplied, the caller
        owns event ordering and the lifetime of ``x`` and ``out``.  The PCIe
        backend synchronizes and returns only after every rank has completed.
        """
        self._check_alive()
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("fast-ulysses minimal is inference-only")
        selected = stream or torch.cuda.current_stream(self.device)
        if torch.device(selected.device) != self.device:
            raise ValueError(f"stream is on {selected.device}, expected {self.device}")

        if self.backend == "pcie":
            selected.synchronize()
            dist.barrier(group=self.pg, device_ids=[self.device.index])
            self._handle.exchange(x, out, mode, selected.cuda_stream)
            selected.synchronize()
            dist.barrier(group=self.pg, device_ids=[self.device.index])
        else:
            self._handle.exchange(x, out, mode, selected.cuda_stream)
        return out

    def destroy(self) -> None:
        """Collectively release all symmetric output buffers."""
        if self._destroyed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.pg, device_ids=[self.device.index])
        self._handle.destroy()
        self._destroyed = True

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("UlyssesGroup has been destroyed")


__all__ = ["UlyssesGroup"]
