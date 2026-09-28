from __future__ import annotations

from pathlib import Path

import pytest

from equiroute.dataset import _registry_fingerprint
from equiroute.evaluation import EvaluationError, score_completions
from equiroute.io import load_examples, load_route_registry
from equiroute.schemas import DatasetArtifact, EvaluationConfig, EvaluationThresholds


FIXTURES = Path(__file__).parent / "fixtures" / "stage4"


def _inputs():
    registry = load_route_registry(FIXTURES / "routes.yaml")
    examples = load_examples(FIXTURES / "examples.jsonl", registry)
    completions = (
        (FIXTURES / "completions.txt").read_text(encoding="utf-8").splitlines()
    )
    return registry, examples, completions


def _provenance(examples: list[object]) -> DatasetArtifact:
    return DatasetArtifact(examples=len(examples), fingerprint="a" * 64)


def test_scores_fixed_raw_predictions_semantically_and_applies_inclusive_gates() -> (
    None
):
    registry, examples, completions = _inputs()
    config = EvaluationConfig(
        thresholds=EvaluationThresholds(
            valid_decision_rate=0.6,
            route_accuracy=0.5,
            argument_accuracy=0.34,
        )
    )

    report = score_completions(
        examples,
        completions,
        registry,
        config,
        artifact="fixture-artifact",
        data=_provenance(examples),
    )

    assert report.schema_version == "2"
    assert report.artifact == "fixture-artifact"
    assert report.model == "model"
    assert report.registry_fingerprint == _registry_fingerprint(registry)
    assert report.metrics.model_dump() == {
        "examples": 6,
        "valid_decisions": 4,
        "valid_decision_rate": 4 / 6,
        "route_correct": 3,
        "route_accuracy": 3 / 6,
        "argument_correct": 2,
        "argument_accuracy": 2 / 6,
    }
    assert [route.model_dump() for route in report.routes] == [
        {
            "name": "lookup_balance",
            "support": 3,
            "predictions": 2,
            "true_positives": 2,
            "precision": 1.0,
            "recall": 2 / 3,
        },
        {
            "name": "replace_card",
            "support": 2,
            "predictions": 2,
            "true_positives": 1,
            "precision": 0.5,
            "recall": 0.5,
        },
        {
            "name": "transfer_status",
            "support": 1,
            "predictions": 0,
            "true_positives": 0,
            "precision": None,
            "recall": 0.0,
        },
        {
            "name": "unrepresented_route",
            "support": 0,
            "predictions": 0,
            "true_positives": 0,
            "precision": None,
            "recall": None,
        },
    ]
    assert [row.model_dump() for row in report.confusion_matrix] == [
        {
            "expected": "lookup_balance",
            "predicted": {"lookup_balance": 2, "replace_card": 1},
        },
        {"expected": "replace_card", "predicted": {"replace_card": 1}},
        {"expected": "transfer_status", "predicted": {}},
        {"expected": "unrepresented_route", "predicted": {}},
    ]
    assert [output.model_dump() for output in report.invalid_outputs] == [
        {"category": "missing_function_call", "count": 0},
        {"category": "malformed_function_call", "count": 1},
        {"category": "invalid_argument_syntax", "count": 0},
        {"category": "unknown_route", "count": 1},
        {"category": "invalid_arguments", "count": 0},
    ]
    assert [gate.model_dump() for gate in report.thresholds] == [
        {
            "name": "valid_decision_rate",
            "minimum": 0.6,
            "actual": 4 / 6,
            "passed": True,
        },
        {
            "name": "route_accuracy",
            "minimum": 0.5,
            "actual": 3 / 6,
            "passed": True,
        },
        {
            "name": "argument_accuracy",
            "minimum": 0.34,
            "actual": 2 / 6,
            "passed": False,
        },
    ]
    assert report.passed is False
    assert [
        (error.expected_route, error.predicted_route, error.invalid_category)
        for error in report.representative_errors
    ] == [
        ("lookup_balance", "replace_card", None),
        ("lookup_balance", "lookup_balance", None),
        ("replace_card", None, "unknown_route"),
        ("transfer_status", None, "malformed_function_call"),
    ]


def test_rejects_mismatched_iterator_counts_without_truncating() -> None:
    registry, examples, completions = _inputs()

    with pytest.raises(EvaluationError, match="counts differ"):
        score_completions(
            iter(examples),
            iter(completions[:-1]),
            registry,
            EvaluationConfig(),
            artifact="fixture-artifact",
            data=_provenance(examples),
        )


def test_redacts_only_representative_content_not_semantic_evidence() -> None:
    registry, examples, completions = _inputs()

    report = score_completions(
        examples,
        completions,
        registry,
        EvaluationConfig(redact=True),
        artifact="fixture-artifact",
        data=_provenance(examples),
    )

    assert report.metrics.argument_correct == 2
    assert report.invalid_outputs[1].count == 1
    assert all(error.input is None for error in report.representative_errors)
    assert all(error.raw_output is None for error in report.representative_errors)
    assert all(error.detail is None for error in report.representative_errors)
