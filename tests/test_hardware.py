from types import SimpleNamespace

import pytest

from equiroute.hardware import select_device, select_training_capability


class AvailableBackend:
    def __init__(self, available: bool, bf16_supported: bool = False) -> None:
        self.available = available
        self.bf16_supported = bf16_supported

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bf16_supported


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [
        pytest.param(True, True, "cuda", id="cuda-preferred-over-mps"),
        pytest.param(False, True, "mps", id="mps-available"),
        pytest.param(False, False, "cpu", id="cpu-fallback"),
    ],
)
def test_select_device_prefers_available_accelerator(
    cuda_available: bool, mps_available: bool, expected: str
) -> None:
    torch = SimpleNamespace(
        cuda=AvailableBackend(cuda_available),
        backends=SimpleNamespace(mps=AvailableBackend(mps_available)),
    )

    assert select_device(torch) == expected


def test_select_device_falls_back_to_cpu_for_minimal_torch_fake() -> None:
    assert select_device(SimpleNamespace()) == "cpu"


@pytest.mark.parametrize(
    ("device", "bf16_supported", "dtype", "mixed_precision"),
    [
        pytest.param("cuda", True, "bfloat16", "bf16", id="cuda-bfloat16"),
        pytest.param("cuda", False, "float16", "fp16", id="cuda-float16"),
        pytest.param("mps", False, "float32", "no", id="mps-float32"),
        pytest.param("cpu", False, "float32", "no", id="cpu-float32"),
    ],
)
def test_select_training_capability_uses_supported_precision_policy(
    device: str,
    bf16_supported: bool,
    dtype: str,
    mixed_precision: str,
) -> None:
    torch = SimpleNamespace(
        cuda=AvailableBackend(device == "cuda", bf16_supported),
        backends=SimpleNamespace(mps=AvailableBackend(device == "mps")),
    )

    capability = select_training_capability(torch)

    assert capability.device == device
    assert capability.dtype == dtype
    assert capability.mixed_precision == mixed_precision
