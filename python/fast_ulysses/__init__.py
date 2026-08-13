"""Minimal symmetric-memory Ulysses all-to-all."""

try:
    import torch  # noqa: F401

    from . import _C
except ImportError as exc:
    from ._diagnose import explain

    raise ImportError(explain(exc)) from exc

from .group import UlyssesGroup

__version__ = _C.build_info()["version"]
__all__ = ["UlyssesGroup"]
