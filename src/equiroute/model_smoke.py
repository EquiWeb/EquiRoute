"""Manually invoked FunctionGemma model-loading smoke check.

Run with ``python -m equiroute.model_smoke`` after installing the ``model``
extra and obtaining access to the gated Hugging Face repository.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from .hardware import select_device
from .model import FUNCTIONGEMMA_MODEL_ID, FUNCTIONGEMMA_REVISION

MODEL_ID = FUNCTIONGEMMA_MODEL_ID
MODEL_REVISION = FUNCTIONGEMMA_REVISION
_PROMPT = "Reply with exactly one word: ready."


class ModelSmokeError(RuntimeError):
    """A failure that prevents the FunctionGemma smoke check from running."""


def run_smoke() -> tuple[str, str]:
    """Load the pinned FunctionGemma model and generate one token."""
    torch, transformers = _load_dependencies()
    device = select_device(torch)

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
    except Exception as error:
        raise ModelSmokeError(
            f"Could not download or load {MODEL_ID} at revision {MODEL_REVISION}. "
            "Confirm network access, Hugging Face credentials, and acceptance of "
            "the model's gated license."
        ) from error

    try:
        model.to(device)
        model.eval()
        inputs = tokenizer(_PROMPT, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    except Exception as error:
        raise ModelSmokeError(
            f"Could not run {MODEL_ID} on the selected {device} device. "
            "Use a PyTorch build that supports this device, or run on CPU."
        ) from error

    if generated.shape[-1] <= input_length:
        raise ModelSmokeError(f"{MODEL_ID} did not generate a token on {device}.")

    token = tokenizer.decode(generated[0][input_length:], skip_special_tokens=False)
    return device, token


def main() -> int:
    """Run the smoke check and report its generated token."""
    try:
        device, token = run_smoke()
    except ModelSmokeError as error:
        print(f"Model smoke failed: {error}", file=sys.stderr)
        return 1

    print(f"Generated a token on {device}: {token!r}")
    return 0


def _load_dependencies() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise ModelSmokeError(
            "The model smoke check requires the optional model dependencies. "
            "Install them with `uv sync --extra model`."
        ) from error
    return torch, transformers


if __name__ == "__main__":
    raise SystemExit(main())
