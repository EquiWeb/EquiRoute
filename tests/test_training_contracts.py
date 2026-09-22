from __future__ import annotations

import pytest
from pydantic import ValidationError

from equiroute.model import FUNCTIONGEMMA_MODEL_ID, FUNCTIONGEMMA_REVISION
from equiroute.schemas import TrainingManifest


_FINGERPRINT = "a" * 64


def _completed_manifest(**changes: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "1",
        "status": "completed",
        "inputs": {
            "route_registry_fingerprint": _FINGERPRINT,
            "train": {
                "examples": 36,
                "source_fingerprint": _FINGERPRINT,
                "compiled_fingerprint": _FINGERPRINT,
            },
            "validation": {
                "examples": 9,
                "source_fingerprint": _FINGERPRINT,
                "compiled_fingerprint": _FINGERPRINT,
            },
            "test": {
                "examples": 9,
                "source_fingerprint": _FINGERPRINT,
                "compiled_fingerprint": _FINGERPRINT,
            },
        },
        "resolved_config": {
            "model": {
                "base_model": FUNCTIONGEMMA_MODEL_ID,
                "revision": FUNCTIONGEMMA_REVISION,
            },
            "template_id": "stage2-functiongemma-native-v1",
            "template_fingerprint": _FINGERPRINT,
            "training": {
                "seed": 42,
                "epochs": 1,
                "learning_rate": 0.0002,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "lora_rank": 16,
                "lora_alpha": 32,
                "max_sequence_length": 128,
            },
            "lora": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.0,
                "bias": "none",
            },
            "checkpoints": {
                "evaluation_strategy": "epoch",
                "save_strategy": "epoch",
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "load_best_model_at_end": True,
            },
        },
        "hardware": {
            "device": "mps",
            "dtype": "float32",
            "mixed_precision": "no",
        },
        "checkpoint_selection": {
            "metric": "eval_loss",
            "value": 0.1,
            "path": "continuation/trainer-state/checkpoint-1",
            "global_step": 1,
            "epoch": 1.0,
        },
        "evaluation": {
            "validation": {"examples": 9, "loss": 0.1},
            "test": {"examples": 9, "loss": 0.2},
            "test_used_for_selection": False,
        },
        "artifacts": {
            "merged_model": [{"path": "model/model.safetensors", "sha256": _FINGERPRINT}],
            "adapter": [
                {
                    "path": "continuation/adapter/adapter_model.safetensors",
                    "sha256": _FINGERPRINT,
                }
            ],
        },
    }
    manifest.update(changes)
    return manifest


def test_completed_training_manifest_records_strict_stage_three_provenance() -> None:
    manifest = TrainingManifest.model_validate(_completed_manifest())

    assert manifest.inputs.train.examples == 36
    assert manifest.resolved_config.lora.target_modules == ["q_proj", "v_proj"]
    assert manifest.evaluation is not None
    assert manifest.evaluation.test_used_for_selection is False


@pytest.mark.parametrize(
    "changes",
    [
        {"checkpoint_selection": None},
        {"hardware": {"device": "mps", "dtype": "float16", "mixed_precision": "fp16"}},
        {"resolved_config": {"unexpected": "field"}},
    ],
)
def test_training_manifest_rejects_incomplete_or_unsupported_contracts(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        TrainingManifest.model_validate(_completed_manifest(**changes))
