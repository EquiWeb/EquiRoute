from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from equiroute.decisions import DecisionValidationError, validate_decision
from equiroute.errors import LabelingConfigError, SanitizedArtifactLoadError
from equiroute.io import (
    load_labeling_config,
    load_labeling_route_registry,
    load_sanitized_handoff,
    resolve_labeling_paths,
)
from equiroute.migrations import (
    SchemaMigrationError,
    migrate_label_candidate,
    migrate_labeling_config,
    migrate_labeling_manifest,
)
from equiroute.schemas import (
    LabelCandidate,
    LabelingManifest,
    RawInputRow,
    RouteRegistry,
)


def _config() -> dict[str, object]:
    return {
        "schema_version": "2",
        "input": {"directory": "sanitized"},
        "routes": "routes.yaml",
        "output": {"directory": "candidates"},
        "provider": {
            "endpoint": "https://openrouter.ai/api/v1",
            "model": "openai/gpt-4o-mini",
            "credential_env_var": "OPENROUTER_API_KEY",
        },
        "policy_prompt": "Choose exactly one route.",
        "concurrency": 2,
        "rate_limit_per_minute": 30,
        "max_retries": 2,
    }


def _registry() -> RouteRegistry:
    return RouteRegistry.model_validate(
        {
            "routes": [
                {
                    "name": "billing",
                    "description": "Billing help.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }
            ]
        }
    )


def _candidate(*, decision: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "schema_version": "2",
        "provenance": {
            "source_id": "ticket-1",
            "source_line": 1,
            "policy_fingerprint": "a" * 64,
            "registry_fingerprint": "b" * 64,
            "provider_model": "openai/gpt-4o-mini",
            "provider_endpoint": "https://openrouter.ai/api/v1",
            "request_status": "succeeded",
            "response_timestamp": "2026-10-02T12:30:45Z",
            "attempts": 1,
        },
        "status": "labeled",
        "decision": decision or {"name": "billing", "arguments": {}},
    }


def _write_handoff(directory: Path) -> None:
    directory.mkdir()
    row = RawInputRow(
        id="ticket-1",
        input="Sanitized text only.",
        metadata={"_equiroute": {"source_line": 1}, "channel": "email"},
    )
    rows = (
        json.dumps(
            row.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    (directory / "rows.jsonl").write_bytes(rows)
    manifest = {
        "schema_version": "2",
        "source": {"rows": 1, "sha256": "a" * 64},
        "output": {"rows": 1, "sha256": hashlib.sha256(rows).hexdigest()},
        "config_fingerprint": "b" * 64,
        "max_input_bytes": 512,
        "redaction_count": 1,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_loads_strict_labeling_config_and_resolves_only_config_relative_paths(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "labeling.yaml"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    (tmp_path / "routes.yaml").write_text(
        """routes:
  - name: billing
    description: Billing help.
    parameters:
      type: object
      properties: {}
      additionalProperties: false
""",
        encoding="utf-8",
    )

    config = load_labeling_config(config_path)

    assert resolve_labeling_paths(config_path, config) == (
        tmp_path / "sanitized",
        tmp_path / "routes.yaml",
        tmp_path / "candidates",
    )
    assert load_labeling_route_registry(config_path, config) == _registry()


def test_labeling_config_rejects_non_v2_and_never_reflects_credential_value(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "labeling.yaml"
    document = _config()
    document["provider"] = {
        "endpoint": "https://api-key:credential-secret@openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "credential_env_var": "OPENROUTER_API_KEY",
    }
    config_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(LabelingConfigError) as raised:
        load_labeling_config(config_path)

    assert "credential-secret" not in str(raised.value)
    assert raised.value.path == "provider.endpoint"

    document = _config()
    document["schema_version"] = "1"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(LabelingConfigError, match="schema_version"):
        load_labeling_config(config_path)


@pytest.mark.parametrize(
    "document",
    [{}, {"schema_version": 2}, {"schema_version": "1"}, {"schema_version": "3"}],
)
def test_stage8_migrations_are_v2_only_and_non_mutating(
    document: dict[str, object],
) -> None:
    original = dict(document)

    for migration in (
        migrate_labeling_config,
        migrate_label_candidate,
        migrate_labeling_manifest,
    ):
        with pytest.raises(SchemaMigrationError):
            migration(document)

    assert document == original


def test_candidate_is_not_a_training_example_and_uses_safe_rejection_reasons() -> None:
    candidate = LabelCandidate.model_validate(_candidate())

    assert "input" not in candidate.model_dump(mode="json")
    rejected = LabelCandidate.model_validate(
        {
            **_candidate(),
            "status": "rejected",
            "decision": None,
            "rejection_reason": "malformed_response",
        }
    )
    assert rejected.decision is None
    with pytest.raises(ValidationError, match="Extra inputs"):
        LabelCandidate.model_validate({**_candidate(), "input": "raw secret"})
    with pytest.raises(ValidationError, match="rejection_reason"):
        LabelCandidate.model_validate(
            {
                **_candidate(),
                "status": "rejected",
                "decision": None,
                "rejection_reason": "provider message with raw content",
            }
        )


def test_candidate_decision_remains_subject_to_shared_route_validation() -> None:
    candidate = LabelCandidate.model_validate(_candidate(decision={"name": "unknown"}))

    assert candidate.decision is not None
    with pytest.raises(DecisionValidationError, match="unknown route"):
        validate_decision(candidate.decision, _registry())


def test_candidate_requires_positive_source_line() -> None:
    document = _candidate()
    document["provenance"] = {
        **document["provenance"],
        "source_line": 0,
    }

    with pytest.raises(ValidationError, match="source_line"):
        LabelCandidate.model_validate(document)


def test_labeling_manifest_excludes_raw_input_and_credential_fields() -> None:
    manifest = LabelingManifest.model_validate(
        {
            "schema_version": "2",
            "input": {"rows": 1, "sha256": "a" * 64},
            "input_manifest_fingerprint": "b" * 64,
            "output": {"rows": 1, "sha256": "c" * 64},
            "policy_fingerprint": "d" * 64,
            "registry_fingerprint": "e" * 64,
            "provider_model": "openai/gpt-4o-mini",
            "provider_endpoint": "https://openrouter.ai/api/v1",
        }
    )

    assert set(manifest.model_dump()) == {
        "schema_version",
        "input",
        "input_manifest_fingerprint",
        "output",
        "policy_fingerprint",
        "registry_fingerprint",
        "provider_model",
        "provider_endpoint",
    }


def test_verified_handoff_loader_reads_only_stage7_artifact_and_checks_hash_count(
    tmp_path: Path,
) -> None:
    handoff = tmp_path / "sanitized"
    _write_handoff(handoff)

    loaded = load_sanitized_handoff(handoff)

    assert loaded.manifest.output.rows == 1
    assert [(item.row.id, item.line) for item in loaded.rows] == [("ticket-1", 1)]

    (handoff / "rows.jsonl").write_bytes(b'{"input":"raw-secret"}\n')
    with pytest.raises(SanitizedArtifactLoadError) as raised:
        load_sanitized_handoff(handoff)

    assert "raw-secret" not in str(raised.value)
    assert "SHA-256" in raised.value.message
