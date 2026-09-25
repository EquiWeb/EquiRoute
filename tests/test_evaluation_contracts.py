from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from equiroute.model import FUNCTIONGEMMA_MODEL_ID, FUNCTIONGEMMA_REVISION
from equiroute.schemas import (
    EvaluationConfig,
    EvaluationMetrics,
    EvaluationReport,
    RouteMetrics,
    TrainingConfig,
)

_FINGERPRINT = "a" * 64


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


def _report() -> dict[str, object]:
    return {
        "schema_version": "1",
        "artifact": "runs/router",
        "model": "runs/router/model",
        "registry_fingerprint": _FINGERPRINT,
        "data": {"examples": 4, "fingerprint": _FINGERPRINT},
        "config": {},
        "metrics": {
            "examples": 4,
            "valid_decisions": 3,
            "valid_decision_rate": 0.75,
            "route_correct": 2,
            "route_accuracy": 0.5,
            "argument_correct": 1,
            "argument_accuracy": 0.25,
        },
        "routes": [
            {
                "name": "alpha",
                "support": 2,
                "predictions": 2,
                "true_positives": 1,
                "precision": 0.5,
                "recall": 0.5,
            },
            {
                "name": "beta",
                "support": 2,
                "predictions": 1,
                "true_positives": 1,
                "precision": 1.0,
                "recall": 0.5,
            },
        ],
        "confusion_matrix": [
            {"expected": "alpha", "predicted": {"alpha": 1}},
            {"expected": "beta", "predicted": {"alpha": 1, "beta": 1}},
        ],
        "invalid_outputs": [
            {"category": "missing_function_call", "count": 1},
            {"category": "malformed_function_call", "count": 0},
            {"category": "invalid_argument_syntax", "count": 0},
            {"category": "unknown_route", "count": 0},
            {"category": "invalid_arguments", "count": 0},
        ],
        "representative_errors": [
            {
                "expected_route": "beta",
                "invalid_category": "missing_function_call",
                "input": "example input",
                "raw_output": "not a call",
                "detail": "no function call was generated",
            }
        ],
        "thresholds": [
            {
                "name": "valid_decision_rate",
                "minimum": 0.0,
                "actual": 0.75,
                "passed": True,
            },
            {
                "name": "route_accuracy",
                "minimum": 0.0,
                "actual": 0.5,
                "passed": True,
            },
            {
                "name": "argument_accuracy",
                "minimum": 0.0,
                "actual": 0.25,
                "passed": True,
            },
        ],
        "passed": True,
    }


def test_evaluation_defaults_preserve_existing_training_config() -> None:
    config = TrainingConfig.model_validate(_training_config())

    assert config.evaluation == EvaluationConfig()
    assert config.evaluation.arguments == "exact"
    assert config.evaluation.redact is False
    assert config.evaluation.max_new_tokens == 128
    assert config.evaluation.thresholds.model_dump() == {
        "valid_decision_rate": 0.0,
        "route_accuracy": 0.0,
        "argument_accuracy": 0.0,
    }


@pytest.mark.parametrize(
    "document",
    [
        {"max_new_tokens": 0},
        {"thresholds": {"valid_decision_rate": -0.01}},
        {"thresholds": {"route_accuracy": 1.01}},
        {"thresholds": {"argument_accuracy": 1.01}},
    ],
)
def test_evaluation_config_rejects_out_of_range_quality_gates(
    document: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        EvaluationConfig.model_validate(document)


def test_evaluation_report_accepts_consistent_semantic_evidence() -> None:
    report = EvaluationReport.model_validate(_report())

    assert report.metrics.route_accuracy == 0.5
    assert report.routes[1].precision == 1.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report["metrics"].update(valid_decisions=5),
        lambda report: report["invalid_outputs"].reverse(),
        lambda report: report["confusion_matrix"][1].update(expected="alpha"),
        lambda report: report["confusion_matrix"][0]["predicted"].update(gamma=1),
        lambda report: report["thresholds"][1].update(actual=0.75),
        lambda report: report.update(passed=False),
    ],
)
def test_evaluation_report_rejects_impossible_evidence(
    mutate: object,
) -> None:
    report = deepcopy(_report())
    mutate(report)  # type: ignore[operator]

    with pytest.raises(ValidationError):
        EvaluationReport.model_validate(report)


def test_redacted_report_rejects_representative_content() -> None:
    report = _report()
    report["config"] = {"redact": True}

    with pytest.raises(ValidationError, match="redacted reports"):
        EvaluationReport.model_validate(report)


def test_evaluation_metric_rates_and_undefined_precision_are_strict() -> None:
    with pytest.raises(ValidationError):
        EvaluationMetrics(
            examples=3,
            valid_decisions=2,
            valid_decision_rate=0.5,
            route_correct=1,
            route_accuracy=1 / 3,
            argument_correct=1,
            argument_accuracy=1 / 3,
        )
    with pytest.raises(ValidationError):
        RouteMetrics(
            name="alpha",
            support=1,
            predictions=0,
            true_positives=0,
            precision=0.0,
            recall=0.0,
        )
