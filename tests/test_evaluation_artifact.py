from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import equiroute.evaluation as evaluation
from equiroute.dataset import _registry_fingerprint
from equiroute.evaluation import EvaluationError, evaluate_artifact
from equiroute.io import load_route_registry
from equiroute.model import FUNCTIONGEMMA_MODEL_ID, FUNCTIONGEMMA_REVISION
from equiroute.schemas import TrainingManifest


FIXTURES = Path(__file__).parent / "fixtures"
STAGE4 = FIXTURES / "stage4"
_FINGERPRINT = "a" * 64
_MODEL_HASH = hashlib.sha256(b"model").hexdigest()


def _write_completed_artifact(
    tmp_path: Path, *, registry_fingerprint: str | None = None
) -> Path:
    artifact = tmp_path / "artifact"
    provenance = artifact / "equiroute"
    provenance.mkdir(parents=True)
    shutil.copyfile(STAGE4 / "routes.yaml", provenance / "routes.yaml")
    shutil.copyfile(FIXTURES / "stage3" / "config.yaml", provenance / "run-config.yaml")
    registry = load_route_registry(provenance / "routes.yaml")
    manifest = TrainingManifest.model_validate(
        {
            "schema_version": "1",
            "status": "completed",
            "inputs": {
                "route_registry_fingerprint": registry_fingerprint
                or _registry_fingerprint(registry),
                "train": {
                    "examples": 1,
                    "source_fingerprint": _FINGERPRINT,
                    "compiled_fingerprint": _FINGERPRINT,
                },
                "validation": {
                    "examples": 1,
                    "source_fingerprint": _FINGERPRINT,
                    "compiled_fingerprint": _FINGERPRINT,
                },
                "test": {
                    "examples": 1,
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
                    "lora_rank": 4,
                    "lora_alpha": 8,
                    "max_sequence_length": 128,
                },
                "lora": {
                    "rank": 4,
                    "alpha": 8,
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
                "evaluation": {
                    "max_new_tokens": 7,
                    "thresholds": {"route_accuracy": 0.75},
                },
            },
            "hardware": {"device": "cpu", "dtype": "float32", "mixed_precision": "no"},
            "checkpoint_selection": {
                "metric": "eval_loss",
                "value": 0.1,
                "path": "continuation/trainer-state/checkpoint-1",
                "global_step": 1,
                "epoch": 1.0,
            },
            "evaluation": {
                "validation": {"examples": 1, "loss": 0.1},
                "test": {"examples": 1, "loss": 0.2},
                "test_used_for_selection": False,
            },
            "artifacts": {
                "merged_model": [
                    {"path": "model/model.safetensors", "sha256": _MODEL_HASH}
                ],
                "adapter": [
                    {
                        "path": "continuation/adapter/adapter_model.safetensors",
                        "sha256": _FINGERPRINT,
                    }
                ],
            },
        }
    )
    (provenance / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json")), encoding="utf-8"
    )
    (artifact / "model").mkdir()
    (artifact / "model" / "model.safetensors").write_bytes(b"model")
    return artifact


def test_evaluates_exported_artifact_with_input_only_prompts_and_atomic_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _write_completed_artifact(tmp_path)
    generated = (STAGE4 / "completions.txt").read_text(encoding="utf-8").splitlines()
    calls: list[tuple[Path, list[str], Any]] = []

    def fake_generate(
        model_directory: Path, prompts: list[str], config: Any
    ) -> list[str]:
        calls.append((model_directory, prompts, config))
        return generated

    monkeypatch.setattr(evaluation, "_generate_completions", fake_generate)

    report = evaluate_artifact(artifact, STAGE4 / "examples.jsonl")

    assert calls[0][0] == artifact.resolve() / "model"
    assert calls[0][2].max_new_tokens == 7
    assert len(calls[0][1]) == 6
    assert "Show the pending balance for account A-001." in calls[0][1][0]
    assert all(prompt.endswith("<start_of_turn>model\n") for prompt in calls[0][1])
    assert all("<start_function_call>" not in prompt for prompt in calls[0][1])
    assert report.artifact == str(artifact.resolve())
    assert report.model == "model"
    assert report.data.examples == 6
    assert report.metrics.route_correct == 3
    assert report.config.thresholds.route_accuracy == 0.75
    assert report.passed is False
    persisted = artifact / "equiroute" / "semantic-evaluation.json"
    assert json.loads(persisted.read_text(encoding="utf-8")) == report.model_dump(
        mode="json"
    )


def test_rejects_artifact_registry_provenance_before_model_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _write_completed_artifact(tmp_path, registry_fingerprint="b" * 64)

    def unexpected_generation(*_: object) -> list[str]:
        raise AssertionError(
            "model generation must not begin before provenance validation"
        )

    monkeypatch.setattr(evaluation, "_generate_completions", unexpected_generation)

    with pytest.raises(EvaluationError, match="fingerprint"):
        evaluate_artifact(artifact, STAGE4 / "examples.jsonl")

    assert not (artifact / "equiroute" / "semantic-evaluation.json").exists()


def test_rejects_tampered_export_before_model_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _write_completed_artifact(tmp_path)
    (artifact / "model" / "model.safetensors").write_bytes(b"tampered")

    def unexpected_generation(*_: object) -> list[str]:
        raise AssertionError(
            "model generation must not begin before export verification"
        )

    monkeypatch.setattr(evaluation, "_generate_completions", unexpected_generation)

    with pytest.raises(EvaluationError, match="does not match artifact provenance"):
        evaluate_artifact(artifact, STAGE4 / "examples.jsonl")
