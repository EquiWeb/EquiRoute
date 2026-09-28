from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from equiroute.errors import ConfigLoadError
from equiroute.io import load_training_config
from equiroute.migrations import (
    SchemaMigrationError,
    migrate_comparative_evaluation,
    migrate_dataset_manifest,
    migrate_dataset_report,
    migrate_training_evaluation,
    migrate_training_manifest,
)
from equiroute.model import FUNCTIONGEMMA_MODEL_ID, FUNCTIONGEMMA_REVISION
from equiroute.schemas import (
    ComparativeEvaluation,
    DatasetManifest,
    DatasetReport,
    TrainingEvaluation,
    TrainingManifest,
)
from equiroute.training import _read_manifest

_FIXTURES = Path(__file__).parent / "fixtures"
_PARENT_FINGERPRINT = "a" * 64
_CHILD_FINGERPRINT = "b" * 64
_DATA_FINGERPRINT = "c" * 64


def _v1_report(*, child: bool) -> dict[str, object]:
    if child:
        routes: list[dict[str, object]] = [
            {
                "name": "alpha",
                "support": 1,
                "predictions": 1,
                "true_positives": 1,
                "precision": 1.0,
                "recall": 1.0,
            },
            {
                "name": "beta",
                "support": 0,
                "predictions": 0,
                "true_positives": 0,
                "precision": None,
                "recall": None,
            },
        ]
        confusion = [
            {"expected": "alpha", "predicted": {"alpha": 1}},
            {"expected": "beta", "predicted": {}},
        ]
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
        registry_fingerprint = _PARENT_FINGERPRINT

    return {
        "schema_version": "1",
        "artifact": "runs/router",
        "model": "runs/router/model",
        "registry_fingerprint": registry_fingerprint,
        "data": {"examples": 1, "fingerprint": _DATA_FINGERPRINT},
        "config": {},
        "metrics": {
            "examples": 1,
            "valid_decisions": 1,
            "valid_decision_rate": 1.0,
            "route_correct": 1,
            "route_accuracy": 1.0,
            "argument_correct": 1,
            "argument_accuracy": 1.0,
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
            {
                "name": "valid_decision_rate",
                "minimum": 0.0,
                "actual": 1.0,
                "passed": True,
            },
            {
                "name": "route_accuracy",
                "minimum": 0.0,
                "actual": 1.0,
                "passed": True,
            },
            {
                "name": "argument_accuracy",
                "minimum": 0.0,
                "actual": 1.0,
                "passed": True,
            },
        ],
        "passed": True,
    }


def _v1_comparison() -> dict[str, object]:
    return {
        "schema_version": "1",
        "regression_data": {"examples": 1, "fingerprint": _DATA_FINGERPRINT},
        "parent": _v1_report(child=False),
        "child": _v1_report(child=True),
        "old_route_names": ["alpha"],
        "max_route_accuracy_drop": 0.0,
        "route_accuracy_drop": 0.0,
        "max_argument_accuracy_drop": 0.0,
        "argument_accuracy_drop": 0.0,
        "passed": True,
    }


def _v1_manifest() -> dict[str, object]:
    return {
        "schema_version": "1",
        "status": "running",
        "inputs": {
            "route_registry_fingerprint": _CHILD_FINGERPRINT,
            "train": {
                "examples": 1,
                "source_fingerprint": _DATA_FINGERPRINT,
                "compiled_fingerprint": _DATA_FINGERPRINT,
            },
            "validation": {
                "examples": 1,
                "source_fingerprint": _DATA_FINGERPRINT,
                "compiled_fingerprint": _DATA_FINGERPRINT,
            },
            "test": {
                "examples": 1,
                "source_fingerprint": _DATA_FINGERPRINT,
                "compiled_fingerprint": _DATA_FINGERPRINT,
            },
        },
        "resolved_config": {
            "model": {
                "base_model": FUNCTIONGEMMA_MODEL_ID,
                "revision": FUNCTIONGEMMA_REVISION,
            },
            "template_id": "stage2-functiongemma-native-v1",
            "template_fingerprint": _DATA_FINGERPRINT,
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
        "parent": {
            "directory": "runs/parent",
            "manifest_sha256": _PARENT_FINGERPRINT,
            "adapter": [
                {
                    "path": "continuation/adapter/adapter_model.safetensors",
                    "sha256": _PARENT_FINGERPRINT,
                }
            ],
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
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                    },
                }
            ],
        },
        "comparative_evaluation": _v1_comparison(),
    }


def test_loads_unversioned_v1_config_without_rewriting_source() -> None:
    source = _FIXTURES / "config.yaml"
    original = source.read_bytes()

    config = load_training_config(source)

    assert config.schema_version == "2"
    assert source.read_bytes() == original


def test_migrates_v1_dataset_documents_and_leaves_v2_idempotent() -> None:
    report = {
        "schema_version": "1",
        "example_count": 1,
        "route_distribution": [
            {"name": "alpha", "total": 1, "train": 1, "validation": 0, "test": 0}
        ],
    }
    manifest = {
        "schema_version": "1",
        "registry_fingerprint": _PARENT_FINGERPRINT,
        "datasets": {"train": {"examples": 1, "fingerprint": _DATA_FINGERPRINT}},
    }

    migrated_report = migrate_dataset_report(report)
    migrated_manifest = migrate_dataset_manifest(manifest)

    assert report["schema_version"] == "1"
    assert manifest["schema_version"] == "1"
    assert DatasetReport.model_validate(migrated_report).schema_version == "2"
    parsed_manifest = DatasetManifest.model_validate(migrated_manifest)
    assert parsed_manifest.schema_version == "2"
    v2_manifest = parsed_manifest.model_dump(mode="json")
    assert migrate_dataset_manifest(v2_manifest) is v2_manifest


@pytest.mark.parametrize("version", ["3", 3, None])
def test_rejects_future_or_nonstring_schema_versions(
    tmp_path: Path, version: object
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        _FIXTURES.joinpath("config.yaml").read_text(encoding="utf-8")
        + f"schema_version: {json.dumps(version)}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError, match="schema_version") as error:
        load_training_config(config)

    assert '"1" and "2"' in str(error.value)


def test_loads_nested_v1_comparative_reports_without_rewriting_manifest(
    tmp_path: Path,
) -> None:
    document = _v1_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    original = path.read_bytes()

    manifest = _read_manifest(path)

    assert manifest.schema_version == "2"
    assert manifest.comparative_evaluation is not None
    assert manifest.comparative_evaluation.parent.schema_version == "2"
    assert manifest.comparative_evaluation.child.schema_version == "2"
    assert document["schema_version"] == "1"
    comparison = document["comparative_evaluation"]
    assert isinstance(comparison, dict)
    assert comparison["parent"]["schema_version"] == "1"
    assert path.read_bytes() == original

    v2_manifest = manifest.model_dump(mode="json")
    assert migrate_training_manifest(v2_manifest) is v2_manifest
    parsed_comparison = ComparativeEvaluation.model_validate(deepcopy(comparison))
    assert parsed_comparison.parent.schema_version == "2"


def test_migrates_v1_loss_and_continuation_evidence_without_mutating_source() -> None:
    loss = {
        "schema_version": "1",
        "validation": {"examples": 9, "loss": 0.1},
        "test": {"examples": 9, "loss": 0.2},
        "test_used_for_selection": False,
    }
    comparison = _v1_comparison()

    migrated_loss = migrate_training_evaluation(loss)
    migrated_comparison = migrate_comparative_evaluation(comparison)

    assert loss["schema_version"] == "1"
    assert comparison["schema_version"] == "1"
    assert TrainingEvaluation.model_validate(loss).schema_version == "2"
    assert ComparativeEvaluation.model_validate(comparison).schema_version == "2"
    assert TrainingEvaluation.model_validate(migrated_loss).schema_version == "2"
    assert (
        ComparativeEvaluation.model_validate(migrated_comparison).schema_version == "2"
    )
    assert migrate_training_evaluation(migrated_loss) is migrated_loss
    assert migrate_comparative_evaluation(migrated_comparison) is migrated_comparison


def test_direct_migration_rejects_nonstring_artifact_version() -> None:
    with pytest.raises(SchemaMigrationError, match="must be a string"):
        migrate_dataset_report({"schema_version": 2})
