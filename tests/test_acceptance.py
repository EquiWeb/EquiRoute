from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner


from equiroute.acceptance import accept_approved_labels
from equiroute.cli import app
from equiroute.dataset import validate_dataset
from equiroute.errors import (
    AcceptanceConfigError,
    ExampleLoadError,
    ReviewRecordLoadError,
)
from equiroute.io import load_route_registry
from equiroute.review import review_label_candidates


def _compact(document: object) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_pipeline(directory: Path, *, duplicate_inputs: bool = False) -> Path:
    directory.mkdir(exist_ok=True)
    registry = {
        "routes": [
            {
                "name": "billing",
                "description": "Billing requests.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "technical",
                "description": "Technical requests.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        ]
    }
    registry_bytes = _compact(registry)
    (directory / "routes.yaml").write_bytes(registry_bytes)
    registry_fingerprint = hashlib.sha256(registry_bytes).hexdigest()

    source_rows = [
        {
            "id": f"ticket-{number}",
            "input": "Same sanitized input"
            if duplicate_inputs and number < 3
            else f"Sanitized input {number}",
            "metadata": {"_equiroute": {"source_line": number}},
        }
        for number in range(1, 6)
    ]
    sanitized_bytes = b"".join(_compact(row) + b"\n" for row in source_rows)
    sanitized = directory / "sanitized"
    sanitized.mkdir()
    (sanitized / "rows.jsonl").write_bytes(sanitized_bytes)
    sanitized_manifest = _compact(
        {
            "schema_version": "2",
            "source": {"rows": 5, "sha256": "a" * 64},
            "output": {
                "rows": 5,
                "sha256": hashlib.sha256(sanitized_bytes).hexdigest(),
            },
            "config_fingerprint": "b" * 64,
            "max_input_bytes": 100,
            "redaction_count": 1,
        }
    )
    (sanitized / "manifest.json").write_bytes(sanitized_manifest)

    def provenance(number: int, request_status: str = "succeeded") -> dict[str, object]:
        return {
            "source_id": f"ticket-{number}",
            "source_line": number,
            "policy_fingerprint": "c" * 64,
            "registry_fingerprint": registry_fingerprint,
            "provider_model": "test/model",
            "provider_endpoint": "https://example.test/v1",
            "request_status": request_status,
            "response_timestamp": "2026-10-03T12:30:45Z",
            "attempts": 1,
        }

    candidate_rows = [
        {
            "schema_version": "2",
            "provenance": provenance(1),
            "status": "labeled",
            "decision": {"name": "billing", "arguments": {}},
        },
        {
            "schema_version": "2",
            "provenance": provenance(2),
            "status": "labeled",
            "decision": {"name": "billing", "arguments": {}},
        },
        {
            "schema_version": "2",
            "provenance": provenance(3),
            "status": "labeled",
            "decision": {"name": "technical", "arguments": {}},
        },
        {
            "schema_version": "2",
            "provenance": provenance(4, "refused"),
            "status": "rejected",
            "rejection_reason": "refusal",
        },
        {
            "schema_version": "2",
            "provenance": provenance(5),
            "status": "labeled",
            "decision": {"name": "billing", "arguments": {"unexpected": "value"}},
        },
    ]
    candidate_bytes = b"".join(_compact(row) + b"\n" for row in candidate_rows)
    candidates = directory / "candidates"
    candidates.mkdir()
    (candidates / "candidates.jsonl").write_bytes(candidate_bytes)
    (candidates / "manifest.json").write_bytes(
        _compact(
            {
                "schema_version": "2",
                "input": {
                    "rows": 5,
                    "sha256": hashlib.sha256(sanitized_bytes).hexdigest(),
                },
                "input_manifest_fingerprint": hashlib.sha256(
                    sanitized_manifest
                ).hexdigest(),
                "output": {
                    "rows": 5,
                    "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
                },
                "policy_fingerprint": "c" * 64,
                "registry_fingerprint": registry_fingerprint,
                "provider_model": "test/model",
                "provider_endpoint": "https://example.test/v1",
            }
        )
    )

    review_config = directory / "review.yaml"
    review_config.write_bytes(
        _compact(
            {
                "schema_version": "2",
                "sanitized": "sanitized",
                "candidates": "candidates",
                "routes": "routes.yaml",
                "output": {"directory": "review"},
            }
        )
    )
    review_label_candidates(review_config)
    return review_config


def _review_decisions(directory: Path, decisions: dict[int, str]) -> None:
    path = directory / "review" / "review.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for line, decision in decisions.items():
        review: dict[str, str] = {
            "decision": decision,
            "reviewer": "reviewer-1",
            "reviewed_at": "2026-10-03T12:31:45Z",
        }
        if decision == "rejected":
            review["reason"] = "incorrect_route"
        rows[line - 1]["review"] = review
    path.write_bytes(b"".join(_compact(row) + b"\n" for row in rows))


def _acceptance_config(
    directory: Path, *, quotas: dict[str, object] | None = None, gold: bool = False
) -> Path:
    document: dict[str, object] = {
        "schema_version": "2",
        "sanitized": "sanitized",
        "candidates": "candidates",
        "routes": "routes.yaml",
        "review": "review/review.jsonl",
        "review_manifest": "review/manifest.json",
        "output": {"directory": "accepted"},
    }
    if quotas is not None:
        document["quotas"] = quotas
    if gold:
        gold_rows = [
            {
                "id": "ticket-1",
                "input": "gold",
                "route": {"name": "billing", "arguments": {}},
            },
            {
                "id": "ticket-4",
                "input": "gold",
                "route": {"name": "technical", "arguments": {}},
            },
        ]
        (directory / "gold.jsonl").write_bytes(
            b"".join(_compact(row) + b"\n" for row in gold_rows)
        )
        document["gold"] = "gold.jsonl"
    path = directory / "acceptance.yaml"
    path.write_bytes(_compact(document))
    return path


def test_acceptance_compiles_only_approved_rows_with_complete_provenance(
    tmp_path: Path,
) -> None:
    _write_pipeline(tmp_path)
    _review_decisions(tmp_path, {1: "approved", 2: "rejected"})

    manifest = accept_approved_labels(_acceptance_config(tmp_path, gold=True))

    output = tmp_path / "accepted"
    records = [
        json.loads(line)
        for line in (output / "examples.jsonl").read_text().splitlines()
    ]
    assert [record["id"] for record in records] == ["ticket-1"]
    assert records[0]["input"] == "Sanitized input 1"
    provenance = records[0]["metadata"]["_equiroute"]
    assert provenance == {
        "source_id": "ticket-1",
        "source_line": 1,
        "candidate_sha256": hashlib.sha256(
            (tmp_path / "candidates" / "candidates.jsonl")
            .read_bytes()
            .splitlines(keepends=True)[0]
        ).hexdigest(),
        "candidate_artifact_sha256": manifest.candidates.sha256,
        "handoff_rows_sha256": manifest.sanitized.sha256,
        "handoff_manifest_fingerprint": manifest.sanitized_manifest_fingerprint,
        "registry_fingerprint": manifest.registry_fingerprint,
        "policy_fingerprint": "c" * 64,
        "provider_model": "test/model",
        "provider_endpoint": "https://example.test/v1",
        "reviewer": "reviewer-1",
        "reviewed_at": "2026-10-03T12:31:45Z",
        "review_rows_sha256": manifest.review.sha256,
        "review_decision": "approved",
        "review_reason": None,
        "review_manifest_fingerprint": manifest.review_manifest_fingerprint,
        "immutable_review_fingerprint": manifest.immutable_review_fingerprint,
    }
    assert manifest.output.rows == 1
    assert (
        manifest.output.sha256
        == hashlib.sha256((output / "examples.jsonl").read_bytes()).hexdigest()
    )
    validate_dataset(
        output / "examples.jsonl", load_route_registry(tmp_path / "routes.yaml")
    )
    report = json.loads((output / "report.json").read_text())
    assert report["review_quality"]["rejection_rate"] == 0.2
    assert report["review_quality"]["routes"][0]["locally_invalid"] == 1
    assert report["gold_quality"]["routes"] == [
        {
            "route": "billing",
            "examples": 1,
            "valid_decisions": 1,
            "invalid_decisions": 0,
            "no_decision": 0,
            "route_correct": 1,
            "exact_decision_correct": 1,
            "invalid_decision_rate": 0.0,
        },
        {
            "route": "technical",
            "examples": 1,
            "valid_decisions": 0,
            "invalid_decisions": 0,
            "no_decision": 1,
            "route_correct": 0,
            "exact_decision_correct": 0,
            "invalid_decision_rate": 1.0,
        },
    ]


def test_acceptance_uses_deterministic_per_route_quotas(tmp_path: Path) -> None:
    _write_pipeline(tmp_path)
    _review_decisions(tmp_path, {1: "approved", 2: "approved", 3: "approved"})
    manifest = accept_approved_labels(
        _acceptance_config(
            tmp_path,
            quotas={"seed": 17, "per_route": {"billing": 1, "technical": 0}},
        )
    )

    selected = [
        json.loads(line)["id"]
        for line in (tmp_path / "accepted" / "examples.jsonl").read_text().splitlines()
    ]
    expected = min(
        ("ticket-1", "ticket-2"),
        key=lambda source_id: hashlib.sha256(
            f"equiroute-stage9-acceptance\0{17}\0billing\0{source_id}".encode()
        ).digest(),
    )
    assert selected == [expected]
    assert manifest.quotas is not None
    report = json.loads((tmp_path / "accepted" / "report.json").read_text())
    assert report["review_quality"]["routes"][0]["quota_shortfall"] == 0
    assert report["review_quality"]["routes"][1]["accepted"] == 0


def test_acceptance_refuses_overwrite_and_cleans_failed_stage1_output(
    tmp_path: Path,
) -> None:
    _write_pipeline(tmp_path / "occupied")
    _review_decisions(tmp_path / "occupied", {1: "approved"})
    occupied_config = _acceptance_config(tmp_path / "occupied")
    output = tmp_path / "occupied" / "accepted"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("unchanged")
    with pytest.raises(AcceptanceConfigError, match="refusing to overwrite"):
        accept_approved_labels(occupied_config)
    assert sentinel.read_text() == "unchanged"

    _write_pipeline(tmp_path / "invalid", duplicate_inputs=True)
    _review_decisions(tmp_path / "invalid", {1: "approved", 2: "approved"})
    with pytest.raises(ExampleLoadError):
        accept_approved_labels(_acceptance_config(tmp_path / "invalid"))
    assert not (tmp_path / "invalid" / "accepted").exists()


def test_acceptance_rejects_edited_immutable_review_fields(tmp_path: Path) -> None:
    _write_pipeline(tmp_path)
    _review_decisions(tmp_path, {1: "approved"})
    review = tmp_path / "review" / "review.jsonl"
    rows = [json.loads(line) for line in review.read_text().splitlines()]
    rows[0]["candidate"]["decision"]["name"] = "technical"
    review.write_bytes(b"".join(_compact(row) + b"\n" for row in rows))

    with pytest.raises(ReviewRecordLoadError, match="immutable data"):
        accept_approved_labels(_acceptance_config(tmp_path))
    assert not (tmp_path / "accepted").exists()


def test_acceptance_rejects_timezone_less_reviewer_timestamp_without_output(
    tmp_path: Path,
) -> None:
    _write_pipeline(tmp_path)
    _review_decisions(tmp_path, {1: "approved"})
    review = tmp_path / "review" / "review.jsonl"
    rows = [json.loads(line) for line in review.read_text().splitlines()]
    rows[0]["review"]["reviewed_at"] = "2026-10-03T12:31:45"
    review.write_bytes(b"".join(_compact(row) + b"\n" for row in rows))

    with pytest.raises(ReviewRecordLoadError, match="review row"):
        accept_approved_labels(_acceptance_config(tmp_path))
    assert not (tmp_path / "accepted").exists()


def test_acceptance_cli_rejects_mismatched_review_cardinality_without_output(
    tmp_path: Path,
) -> None:
    _write_pipeline(tmp_path)
    manifest_path = tmp_path / "review" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rows"]["rows"] += 1
    manifest_path.write_bytes(_compact(manifest))

    result = CliRunner().invoke(
        app, ["accept-labels", str(_acceptance_config(tmp_path))]
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert (
        "review report row count does not match verified source artifacts"
        in result.stderr
    )
    assert not (tmp_path / "accepted").exists()
