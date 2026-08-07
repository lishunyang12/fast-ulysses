"""fast_ulysses — Ulysses all-to-all custom op over the NVSHMEM symmetric heap."""

import torch  # noqa: F401  load libtorch before dlopen of _C

try:
    from . import _C  # noqa: F401,E402  trigger TORCH_LIBRARY registration
except ImportError as exc:  # ld.so names a symbol; _diagnose names the cause
    from ._diagnose import explain  # noqa: E402  imported only on the failure path

    raise ImportError(explain(exc)) from exc

from .comm import CompletedHandle, UlyssesGroup  # noqa: E402
from .fallback import TorchUlyssesGroup, make_group, spans_sockets  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "UlyssesGroup",
    "TorchUlyssesGroup",
    "CompletedHandle",
    "make_group",
    "spans_sockets",
]
