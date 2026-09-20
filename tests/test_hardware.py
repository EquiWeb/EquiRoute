from types import SimpleNamespace

import pytest

from equiroute.hardware import select_device


class AvailableBackend:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


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
