from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from equiroute.errors import CandidateArtifactLoadError, ReviewRecordLoadError
from equiroute.io import (
    approved_review_rows,
    load_acceptance_config,
    load_gold_quality_config,
    load_labeling_handoff,
    load_review_config,
    load_review_handoff,
    load_review_manifest,
    load_sanitized_handoff,
)
from equiroute.migrations import (
    SchemaMigrationError,
    migrate_acceptance_config,
    migrate_gold_quality_config,
    migrate_review_config,
)
from equiroute.schemas import (
    AcceptedLabelProvenance,
    CandidateValidation,
    LabelCandidate,
    RawInputRow,
    ReviewManifest,
    ReviewRow,
    ReviewerDecision,
    RouteRegistry,
)


def _compact(document: object) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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


def _write_sanitized(directory: Path) -> None:
    directory.mkdir()
    row = RawInputRow(
        id="ticket-1",
        input="Sanitized input for review.",
        metadata={"_equiroute": {"source_line": 7}},
    )
    rows = _compact(row.model_dump(mode="json")) + b"\n"
    (directory / "rows.jsonl").write_bytes(rows)
    manifest = {
        "schema_version": "2",
        "source": {"rows": 1, "sha256": "a" * 64},
        "output": {"rows": 1, "sha256": hashlib.sha256(rows).hexdigest()},
        "config_fingerprint": "b" * 64,
        "max_input_bytes": 512,
        "redaction_count": 1,
    }
    (directory / "manifest.json").write_bytes(_compact(manifest))


def _write_candidates(
    directory: Path, sanitized: Path, registry: RouteRegistry
) -> None:
    directory.mkdir()
    sanitized_rows = (sanitized / "rows.jsonl").read_bytes()
    sanitized_manifest = (sanitized / "manifest.json").read_bytes()
    candidate = {
        "schema_version": "2",
        "provenance": {
            "source_id": "ticket-1",
            "source_line": 7,
            "policy_fingerprint": "c" * 64,
            "registry_fingerprint": hashlib.sha256(
                _compact(registry.model_dump(mode="json"))
            ).hexdigest(),
            "provider_model": "openai/gpt-4o-mini",
            "provider_endpoint": "https://openrouter.ai/api/v1",
            "request_status": "succeeded",
            "response_timestamp": "2026-10-03T12:30:45Z",
            "attempts": 1,
        },
        "status": "labeled",
        "decision": {"name": "billing", "arguments": {}},
    }
    rows = _compact(candidate) + b"\n"
    (directory / "candidates.jsonl").write_bytes(rows)
    manifest = {
        "schema_version": "2",
        "input": {"rows": 1, "sha256": hashlib.sha256(sanitized_rows).hexdigest()},
        "input_manifest_fingerprint": hashlib.sha256(sanitized_manifest).hexdigest(),
        "output": {"rows": 1, "sha256": hashlib.sha256(rows).hexdigest()},
        "policy_fingerprint": "c" * 64,
        "registry_fingerprint": hashlib.sha256(
            _compact(registry.model_dump(mode="json"))
        ).hexdigest(),
        "provider_model": "openai/gpt-4o-mini",
        "provider_endpoint": "https://openrouter.ai/api/v1",
    }
    (directory / "manifest.json").write_bytes(_compact(manifest))


def _immutable_row_bytes(row: ReviewRow) -> bytes:
    document = row.model_dump(mode="json")
    document.pop("review")
    return _compact(document)


def _write_review(
    report: Path, manifest_path: Path, sanitized, candidates, registry: RouteRegistry
) -> None:
    candidate = candidates.rows[0]
    row = ReviewRow(
        schema_version="2",
        source_id="ticket-1",
        input=sanitized.rows[0].row.input,
        candidate=candidate.candidate,
        validation=candidate.validation,
        selected_for_review=True,
        review=ReviewerDecision(
            decision="approved",
            reviewer="reviewer-1",
            reviewed_at="2026-10-03T12:31:45Z",
        ),
    )
    rows = _compact(row.model_dump(mode="json")) + b"\n"
    report.write_bytes(rows)
    manifest = ReviewManifest(
        schema_version="2",
        sanitized={
            "rows": sanitized.manifest.output.rows,
            "sha256": sanitized.manifest.output.sha256,
        },
        sanitized_manifest_fingerprint=hashlib.sha256(
            (sanitized.directory / "manifest.json").read_bytes()
        ).hexdigest(),
        candidates={
            "rows": candidates.manifest.output.rows,
            "sha256": candidates.manifest.output.sha256,
        },
        candidate_manifest_fingerprint=hashlib.sha256(
            (candidates.directory / "manifest.json").read_bytes()
        ).hexdigest(),
        registry_fingerprint=hashlib.sha256(
            _compact(registry.model_dump(mode="json"))
        ).hexdigest(),
        policy_fingerprint=candidates.manifest.policy_fingerprint,
        provider_model=candidates.manifest.provider_model,
        provider_endpoint=candidates.manifest.provider_endpoint,
        rows={"rows": 1, "sha256": hashlib.sha256(rows).hexdigest()},
        immutable_fingerprint=hashlib.sha256(
            _immutable_row_bytes(row) + b"\n"
        ).hexdigest(),
    )
    manifest_path.write_bytes(_compact(manifest.model_dump(mode="json")))


def test_verified_candidate_handoff_binds_stage7_registry_and_provider_provenance(
    tmp_path: Path,
) -> None:
    registry = _registry()
    sanitized_dir = tmp_path / "sanitized"
    candidates_dir = tmp_path / "candidates"
    _write_sanitized(sanitized_dir)
    _write_candidates(candidates_dir, sanitized_dir, registry)

    handoff = load_labeling_handoff(
        candidates_dir,
        sanitized=load_sanitized_handoff(sanitized_dir),
        registry=registry,
    )

    assert handoff.rows[0].validation == CandidateValidation(valid=True)
    assert (
        handoff.rows[0].sha256
        == hashlib.sha256(
            (candidates_dir / "candidates.jsonl").read_bytes()
        ).hexdigest()
    )

    manifest = json.loads((candidates_dir / "manifest.json").read_text())
    manifest["provider_model"] = "different/model"
    (candidates_dir / "manifest.json").write_bytes(_compact(manifest))
    with pytest.raises(CandidateArtifactLoadError):
        load_labeling_handoff(
            candidates_dir,
            sanitized=load_sanitized_handoff(sanitized_dir),
            registry=registry,
        )


def test_review_handoff_allows_only_explicit_approvals_and_protects_raw_input(
    tmp_path: Path,
) -> None:
    registry = _registry()
    sanitized_dir = tmp_path / "sanitized"
    candidates_dir = tmp_path / "candidates"
    _write_sanitized(sanitized_dir)
    _write_candidates(candidates_dir, sanitized_dir, registry)
    sanitized = load_sanitized_handoff(sanitized_dir)
    candidates = load_labeling_handoff(
        candidates_dir, sanitized=sanitized, registry=registry
    )
    report = tmp_path / "review.jsonl"
    manifest_path = tmp_path / "review-manifest.json"
    _write_review(report, manifest_path, sanitized, candidates, registry)

    handoff = load_review_handoff(
        report,
        manifest=load_review_manifest(manifest_path),
        manifest_path=manifest_path,
        sanitized=sanitized,
        candidates=candidates,
        registry=registry,
    )

    assert [item.row.source_id for item in approved_review_rows(handoff)] == [
        "ticket-1"
    ]

    document = json.loads(report.read_text())
    document["input"] = "raw-secret-must-not-escape"
    report.write_bytes(_compact(document) + b"\n")
    with pytest.raises(ReviewRecordLoadError) as raised:
        load_review_handoff(
            report,
            manifest=load_review_manifest(manifest_path),
            manifest_path=manifest_path,
            sanitized=sanitized,
            candidates=candidates,
            registry=registry,
        )
    assert "raw-secret-must-not-escape" not in str(raised.value)


@pytest.mark.parametrize(
    "migration",
    [migrate_review_config, migrate_acceptance_config, migrate_gold_quality_config],
)
def test_stage9_config_migrations_are_v2_only_and_non_mutating(migration) -> None:
    document = {"schema_version": "1", "sanitized": "s"}

    with pytest.raises(SchemaMigrationError):
        migration(document)

    assert document == {"schema_version": "1", "sanitized": "s"}


def test_review_approval_rejects_unselected_or_invalid_candidates() -> None:
    candidate = LabelCandidate.model_validate(
        {
            "schema_version": "2",
            "provenance": {
                "source_id": "ticket-1",
                "source_line": 1,
                "policy_fingerprint": "a" * 64,
                "registry_fingerprint": "b" * 64,
                "provider_model": "model",
                "provider_endpoint": "https://example.test/v1",
                "request_status": "succeeded",
                "response_timestamp": "2026-10-03T12:30:45Z",
                "attempts": 1,
            },
            "status": "labeled",
            "decision": {"name": "billing", "arguments": {}},
        }
    )

    with pytest.raises(ValueError, match="approved review rows"):
        ReviewRow(
            schema_version="2",
            source_id="ticket-1",
            input="sanitized",
            candidate=candidate,
            validation=CandidateValidation(valid=True),
            selected_for_review=False,
            review=ReviewerDecision(
                decision="approved",
                reviewer="reviewer-1",
                reviewed_at="2026-10-03T12:31:45Z",
            ),
        )


def test_stage9_config_loaders_are_v2_only_and_support_review_csv(
    tmp_path: Path,
) -> None:
    review = {
        "schema_version": "2",
        "sanitized": "sanitized",
        "candidates": "candidates",
        "routes": "routes.yaml",
        "output": {"directory": "review"},
        "report_format": "csv",
        "sampling": {"seed": 42, "per_route": {"billing": 3}},
    }
    review_path = tmp_path / "review.yaml"
    review_path.write_bytes(_compact(review))
    assert load_review_config(review_path).report_format == "csv"

    acceptance = {
        "schema_version": "2",
        "sanitized": "sanitized",
        "candidates": "candidates",
        "routes": "routes.yaml",
        "review": "review.jsonl",
        "review_manifest": "review-manifest.json",
        "output": {"directory": "accepted"},
    }
    acceptance_path = tmp_path / "acceptance.yaml"
    acceptance_path.write_bytes(_compact(acceptance))
    assert load_acceptance_config(acceptance_path).schema_version == "2"

    gold = {
        "schema_version": "2",
        "sanitized": "sanitized",
        "candidates": "candidates",
        "routes": "routes.yaml",
        "gold": "gold.jsonl",
        "output": {"directory": "quality"},
    }
    gold_path = tmp_path / "gold.yaml"
    gold_path.write_bytes(_compact(gold))
    assert load_gold_quality_config(gold_path).gold == "gold.jsonl"


@pytest.mark.parametrize(
    "timestamp",
    ["2026-10-03", "2026-10-03T12:31:45"],
)
def test_reviewer_timestamps_require_timezone_qualified_rfc3339(
    timestamp: str,
) -> None:
    with pytest.raises(ValueError):
        ReviewerDecision(
            decision="approved",
            reviewer="reviewer-1",
            reviewed_at=timestamp,
        )

    with pytest.raises(ValueError):
        AcceptedLabelProvenance(
            source_id="ticket-1",
            source_line=1,
            candidate_sha256="a" * 64,
            candidate_artifact_sha256="b" * 64,
            handoff_rows_sha256="c" * 64,
            handoff_manifest_fingerprint="d" * 64,
            registry_fingerprint="e" * 64,
            policy_fingerprint="f" * 64,
            provider_model="model",
            provider_endpoint="https://example.test/v1",
            reviewer="reviewer-1",
            reviewed_at=timestamp,
            review_rows_sha256="0" * 64,
        )


def test_reviewer_timestamps_accept_rfc3339_utc_offsets() -> None:
    timestamp = "2026-10-03T12:31:45.123456+01:00"
    assert (
        ReviewerDecision(
            decision="approved",
            reviewer="reviewer-1",
            reviewed_at=timestamp,
        ).reviewed_at
        == timestamp
    )
    assert (
        AcceptedLabelProvenance(
            source_id="ticket-1",
            source_line=1,
            candidate_sha256="a" * 64,
            candidate_artifact_sha256="b" * 64,
            handoff_rows_sha256="c" * 64,
            handoff_manifest_fingerprint="d" * 64,
            registry_fingerprint="e" * 64,
            policy_fingerprint="f" * 64,
            provider_model="model",
            provider_endpoint="https://example.test/v1",
            reviewer="reviewer-1",
            reviewed_at=timestamp,
            review_rows_sha256="0" * 64,
        ).reviewed_at
        == timestamp
    )
