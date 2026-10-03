"""Hardware capability selection for local model operations."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Literal

DeviceName = Literal["cuda", "mps", "cpu"]
DtypeName = Literal["bfloat16", "float16", "float32"]
MixedPrecision = Literal["bf16", "fp16", "no"]


@dataclass(frozen=True)
class TrainingCapability:
    """The supported precision policy for one selected training device."""

    device: DeviceName
    dtype: DtypeName
    mixed_precision: MixedPrecision


def select_device(torch_module: Any | None = None) -> DeviceName:
    """Return the best available PyTorch device: CUDA, MPS, then CPU.

    Supplying ``torch_module`` keeps this decision testable without importing
    PyTorch or requiring hardware in the test environment.
    """
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
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


def select_training_capability(
    torch_module: Any | None = None,
) -> TrainingCapability:
    """Select the supported training dtype and Accelerate precision mode.

    CUDA uses bfloat16 when the runtime reports support, otherwise float16.
    MPS and CPU deliberately use float32 with Accelerate mixed precision
    disabled because the Stage-3 path does not support mixed precision there.
    """
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except ImportError:
            return TrainingCapability("cpu", "float32", "no")

    device = select_device(torch_module)
    if device == "cuda":
        cuda = getattr(torch_module, "cuda", None)
        if _is_bf16_supported(cuda):
            return TrainingCapability("cuda", "bfloat16", "bf16")
        return TrainingCapability("cuda", "float16", "fp16")
    return TrainingCapability(device, "float32", "no")


def _is_bf16_supported(cuda: Any) -> bool:
    is_bf16_supported = getattr(cuda, "is_bf16_supported", None)
    return callable(is_bf16_supported) and bool(is_bf16_supported())


def _is_available(backend: Any) -> bool:
    is_available = getattr(backend, "is_available", None)
    return callable(is_available) and bool(is_available())
