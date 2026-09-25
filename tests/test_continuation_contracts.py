from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from equiroute.model import FUNCTIONGEMMA_MODEL_ID, FUNCTIONGEMMA_REVISION
from equiroute.schemas import (
    ComparativeEvaluation,
    ContinuationConfig,
    TrainingConfig,
    TrainingManifest,
)

_PARENT_FINGERPRINT = "a" * 64
_CHILD_FINGERPRINT = "b" * 64
_DATA_FINGERPRINT = "c" * 64


def _training_config() -> dict[str, object]:
    return {
        "model": {
            "base_model": FUNCTIONGEMMA_MODEL_ID,
            "revision": FUNCTIONGEMMA_REVISION,
        },
        "routes": "routes.yaml",
        "data": {
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
        },
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
        "output": {"directory": "runs/router", "export": "merged_huggingface"},
    }


def _report(*, child: bool, data_fingerprint: str = _DATA_FINGERPRINT) -> dict[str, object]:
    routes: list[dict[str, object]]
    confusion: list[dict[str, object]]
    if child:
        routes = [
            {
                "name": "alpha",
                "support": 1,
                "predictions": 0,
                "true_positives": 0,
                "precision": None,
                "recall": 0.0,
            },
            {
                "name": "beta",
                "support": 0,
                "predictions": 1,
                "true_positives": 0,
                "precision": 0.0,
                "recall": None,
            },
        ]
        confusion = [{"expected": "alpha", "predicted": {"beta": 1}}, {"expected": "beta", "predicted": {}}]
        route_correct = 0
        argument_correct = 0
        registry_fingerprint = _CHILD_FINGERPRINT
    else:
        routes = [
            {
                "name": "alpha",
                "support": 1,
                "predictions": 1,
                "true_positives": 1,
                "precision": 1.0,
                "recall": 1.0,
            }
        ]
        confusion = [{"expected": "alpha", "predicted": {"alpha": 1}}]
        route_correct = 1
        argument_correct = 1
        registry_fingerprint = _PARENT_FINGERPRINT

    route_accuracy = float(route_correct)
    argument_accuracy = float(argument_correct)
    return {
        "schema_version": "1",
        "artifact": "runs/router",
        "model": "runs/router/model",
        "registry_fingerprint": registry_fingerprint,
        "data": {"examples": 1, "fingerprint": data_fingerprint},
        "config": {},
        "metrics": {
            "examples": 1,
            "valid_decisions": 1,
            "valid_decision_rate": 1.0,
            "route_correct": route_correct,
            "route_accuracy": route_accuracy,
            "argument_correct": argument_correct,
            "argument_accuracy": argument_accuracy,
        },
        "routes": routes,
        "confusion_matrix": confusion,
        "invalid_outputs": [
            {"category": "missing_function_call", "count": 0},
            {"category": "malformed_function_call", "count": 0},
            {"category": "invalid_argument_syntax", "count": 0},
            {"category": "unknown_route", "count": 0},
            {"category": "invalid_arguments", "count": 0},
        ],
        "representative_errors": [],
        "thresholds": [
            {"name": "valid_decision_rate", "minimum": 0.0, "actual": 1.0, "passed": True},
            {"name": "route_accuracy", "minimum": 0.0, "actual": route_accuracy, "passed": True},
            {"name": "argument_accuracy", "minimum": 0.0, "actual": argument_accuracy, "passed": True},
        ],
        "passed": True,
    }


def _comparison() -> dict[str, object]:
    return {
        "regression_data": {"examples": 1, "fingerprint": _DATA_FINGERPRINT},
        "parent": _report(child=False),
        "child": _report(child=True),
        "old_route_names": ["alpha"],
        "max_route_accuracy_drop": 1.0,
        "route_accuracy_drop": 1.0,
        "max_argument_accuracy_drop": 1.0,
        "argument_accuracy_drop": 1.0,
        "passed": True,
    }


def _manifest() -> dict[str, object]:
    config = _training_config()
    return {
        "schema_version": "1",
        "status": "running",
        "inputs": {
            "route_registry_fingerprint": _CHILD_FINGERPRINT,
            "train": {"examples": 1, "source_fingerprint": _DATA_FINGERPRINT, "compiled_fingerprint": _DATA_FINGERPRINT},
            "validation": {"examples": 1, "source_fingerprint": _DATA_FINGERPRINT, "compiled_fingerprint": _DATA_FINGERPRINT},
            "test": {"examples": 1, "source_fingerprint": _DATA_FINGERPRINT, "compiled_fingerprint": _DATA_FINGERPRINT},
        },
        "resolved_config": {
            "model": config["model"],
            "template_id": "stage2-functiongemma-native-v1",
            "template_fingerprint": _DATA_FINGERPRINT,
            "training": config["training"],
            "lora": {"rank": 16, "alpha": 32, "dropout": 0.0, "bias": "none"},
            "checkpoints": {
                "evaluation_strategy": "epoch",
                "save_strategy": "epoch",
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "load_best_model_at_end": True,
            },
        },
        "hardware": {"device": "cpu", "dtype": "float32", "mixed_precision": "no"},
    }


def _lineage() -> dict[str, object]:
    return {
        "parent": {
            "directory": "runs/parent",
            "manifest_sha256": _PARENT_FINGERPRINT,
            "adapter": [{"path": "continuation/adapter/adapter_model.safetensors", "sha256": _PARENT_FINGERPRINT}],
            "registry_fingerprint": _PARENT_FINGERPRINT,
        },
        "registry_change": {
            "parent_registry_fingerprint": _PARENT_FINGERPRINT,
            "child_registry_fingerprint": _CHILD_FINGERPRINT,
            "retained_route_names": ["alpha"],
            "added_routes": [
                {
                    "name": "beta",
                    "description": "Handle beta requests.",
                    "parameters": {"type": "object", "additionalProperties": False},
                }
            ],
        },
        "comparative_evaluation": _comparison(),
    }


def test_continuation_config_is_optional_and_defaults_to_no_regression_drop() -> None:
    legacy = TrainingConfig.model_validate(_training_config())
    continuation = ContinuationConfig.model_validate({"regression": "regression.jsonl"})

    assert legacy.continuation is None
    assert continuation.max_route_accuracy_drop == 0.0
    assert continuation.max_argument_accuracy_drop == 0.0


@pytest.mark.parametrize(
    "document",
    [
        {"regression": ""},
        {"regression": "regression.jsonl", "max_route_accuracy_drop": -0.01},
        {"regression": "regression.jsonl", "max_argument_accuracy_drop": 1.01},
    ],
)
def test_continuation_config_rejects_invalid_regression_gates(
    document: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ContinuationConfig.model_validate(document)


def test_comparative_evaluation_accepts_consistent_regression_evidence() -> None:
    comparison = ComparativeEvaluation.model_validate(_comparison())

    assert comparison.passed is True
    assert comparison.route_accuracy_drop == 1.0


@pytest.mark.parametrize(
    "change",
    [
        lambda document: document["child"].update(
            {"data": {"examples": 1, "fingerprint": _PARENT_FINGERPRINT}}
        ),
        lambda document: document.update({"old_route_names": ["beta"]}),
        lambda document: document.update({"route_accuracy_drop": 0.0}),
        lambda document: document.update({"max_route_accuracy_drop": 0.5}),
    ],
)
def test_comparative_evaluation_rejects_inconsistent_lineage_evidence(
    change: object,
) -> None:
    document = _comparison()
    change(document)  # type: ignore[operator]

    with pytest.raises(ValidationError):
        ComparativeEvaluation.model_validate(document)


def test_training_manifest_preserves_legacy_and_accepts_complete_lineage() -> None:
    legacy = TrainingManifest.model_validate(_manifest())
    child_document = _manifest()
    child_document.update(_lineage())
    child = TrainingManifest.model_validate(child_document)

    assert legacy.parent is None
    assert child.parent is not None
    assert child.comparative_evaluation is not None


@pytest.mark.parametrize(
    "change",
    [
        lambda document: document.pop("comparative_evaluation"),
        lambda document: document["inputs"].update(
            {"route_registry_fingerprint": _PARENT_FINGERPRINT}
        ),
        lambda document: document["comparative_evaluation"].update(
            {"old_route_names": ["beta"]}
        ),
    ],
)
def test_training_manifest_rejects_incomplete_or_inconsistent_lineage(
    change: object,
) -> None:
    document = _manifest()
    document.update(_lineage())
    change(document)  # type: ignore[operator]

    with pytest.raises(ValidationError):
        TrainingManifest.model_validate(document)
