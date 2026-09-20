"""Hardware capability selection for local model operations."""

from __future__ import annotations

from typing import Any, Literal

DeviceName = Literal["cuda", "mps", "cpu"]


def select_device(torch_module: Any | None = None) -> DeviceName:
    """Return the best available PyTorch device: CUDA, MPS, then CPU.

    Supplying ``torch_module`` keeps this decision testable without importing
    PyTorch or requiring hardware in the test environment.
    """
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return "cpu"

    cuda = getattr(torch_module, "cuda", None)
    if _is_available(cuda):
        return "cuda"

    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    if _is_available(mps):
        return "mps"

    return "cpu"


def _is_available(backend: Any) -> bool:
    is_available = getattr(backend, "is_available", None)
    return callable(is_available) and bool(is_available())
