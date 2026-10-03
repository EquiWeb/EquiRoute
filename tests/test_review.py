from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from equiroute.errors import ReviewConfigError
from equiroute.io import (
    load_labeling_handoff,
    load_review_handoff,
    load_review_manifest,
    load_route_registry,
    load_sanitized_handoff,
)
from equiroute.review import review_label_candidates


def _compact(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_artifacts(
    directory: Path,
    *,
    first_source_id: str = "ticket-1",
    first_input: str = "Sanitized input 1",
) -> Path:
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
            {
                "name": "general",
                "description": "General requests.",
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

    sanitized = directory / "sanitized"
    sanitized.mkdir()
    source_rows = [
        {
            "id": first_source_id if number == 1 else f"ticket-{number}",
            "input": first_input if number == 1 else f"Sanitized input {number}",
            "metadata": {"_equiroute": {"source_line": number}},
        }
        for number in range(1, 7)
    ]
    sanitized_bytes = b"".join(_compact(row) + b"\n" for row in source_rows)
    (sanitized / "rows.jsonl").write_bytes(sanitized_bytes)
    sanitized_manifest = {
        "schema_version": "2",
        "source": {"rows": 6, "sha256": "a" * 64},
        "output": {"rows": 6, "sha256": hashlib.sha256(sanitized_bytes).hexdigest()},
        "config_fingerprint": "b" * 64,
        "max_input_bytes": 100,
        "redaction_count": 1,
    }
    sanitized_manifest_bytes = _compact(sanitized_manifest)
    (sanitized / "manifest.json").write_bytes(sanitized_manifest_bytes)

    candidates = directory / "candidates"
    candidates.mkdir()

    def provenance(number: int, status: str = "succeeded") -> dict[str, object]:
        return {
            "source_id": first_source_id if number == 1 else f"ticket-{number}",
            "source_line": number,
            "policy_fingerprint": "c" * 64,
            "registry_fingerprint": registry_fingerprint,
            "provider_model": "test/model",
            "provider_endpoint": "https://example.test/v1",
            "request_status": status,
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
            "decision": {"name": "technical", "arguments": {}},
        },
        {
            "schema_version": "2",
            "provenance": provenance(3),
            "status": "labeled",
            "decision": {"name": "billing", "arguments": {"unexpected": "x"}},
        },
        {
            "schema_version": "2",
            "provenance": provenance(4),
            "status": "labeled",
            "decision": {"name": "not-a-route", "arguments": {}},
        },
        {
            "schema_version": "2",
            "provenance": provenance(5),
            "status": "rejected",
            "rejection_reason": "malformed_response",
        },
        {
            "schema_version": "2",
            "provenance": provenance(6, "refused"),
            "status": "rejected",
            "rejection_reason": "refusal",
        },
    ]
    candidate_bytes = b"".join(_compact(row) + b"\n" for row in candidate_rows)
    (candidates / "candidates.jsonl").write_bytes(candidate_bytes)
    candidate_manifest = {
        "schema_version": "2",
        "input": {"rows": 6, "sha256": hashlib.sha256(sanitized_bytes).hexdigest()},
        "input_manifest_fingerprint": hashlib.sha256(
            sanitized_manifest_bytes
        ).hexdigest(),
        "output": {"rows": 6, "sha256": hashlib.sha256(candidate_bytes).hexdigest()},
        "policy_fingerprint": "c" * 64,
        "registry_fingerprint": registry_fingerprint,
        "provider_model": "test/model",
        "provider_endpoint": "https://example.test/v1",
    }
    (candidates / "manifest.json").write_bytes(_compact(candidate_manifest))

    gold_rows = [
        {
            "id": "ticket-1",
            "input": "gold",
            "route": {"name": "billing", "arguments": {}},
        },
        {
            "id": "ticket-2",
            "input": "gold",
            "route": {"name": "billing", "arguments": {}},
        },
        {
            "id": "ticket-3",
            "input": "gold",
            "route": {"name": "billing", "arguments": {}},
        },
        {
            "id": "ticket-4",
            "input": "gold",
            "route": {"name": "general", "arguments": {}},
        },
        {
            "id": "ticket-5",
            "input": "gold",
            "route": {"name": "technical", "arguments": {}},
        },
        {
            "id": "ticket-6",
            "input": "gold",
            "route": {"name": "general", "arguments": {}},
        },
    ]
    (directory / "gold.jsonl").write_bytes(
        b"".join(_compact(row) + b"\n" for row in gold_rows)
    )
    return directory / "review.yaml"


def _write_config(path: Path, *, output: str = "review") -> None:
    path.write_bytes(
        _compact(
            {
                "schema_version": "2",
                "sanitized": "sanitized",
                "candidates": "candidates",
                "routes": "routes.yaml",
                "gold": "gold.jsonl",
                "output": {"directory": output},
                "sampling": {
                    "seed": 29,
                    "per_route": {"billing": 2, "technical": 1, "general": 1},
                },
            }
        )
    )


def test_review_report_is_deterministic_and_covers_all_candidate_outcomes(
    tmp_path: Path,
) -> None:
    config = _write_artifacts(tmp_path)
    _write_config(config)

    manifest = review_label_candidates(config)
    report_directory = tmp_path / "review"
    rows = [
        json.loads(line)
        for line in (report_directory / "review.jsonl").read_text().splitlines()
    ]
    quality = json.loads((report_directory / "report.json").read_text())

    assert manifest.rows.rows == 6
    registry = load_route_registry(tmp_path / "routes.yaml")
    sanitized = load_sanitized_handoff(tmp_path / "sanitized")
    candidates = load_labeling_handoff(
        tmp_path / "candidates", sanitized=sanitized, registry=registry
    )
    verified = load_review_handoff(
        report_directory / "review.jsonl",
        manifest=load_review_manifest(report_directory / "manifest.json"),
        manifest_path=report_directory / "manifest.json",
        sanitized=sanitized,
        candidates=candidates,
        registry=registry,
    )
    assert len(verified.rows) == 6

    assert [row["validation"] for row in rows] == [
        {"valid": True},
        {"valid": True},
        {"valid": False, "reason": "invalid_arguments"},
        {"valid": False, "reason": "unknown_route"},
        {"valid": False, "reason": "provider_rejected"},
        {"valid": False, "reason": "provider_rejected"},
    ]
    assert [row["selected_for_review"] for row in rows] == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert all(row["review"] == {"decision": "unreviewed"} for row in rows)

    review_routes = quality["review_quality"]["routes"]
    assert review_routes == [
        {
            "route": "billing",
            "candidate_rows": 2,
            "provider_rejected": 0,
            "locally_invalid": 1,
            "selected_for_review": 1,
            "unreviewed": 1,
            "reviewer_rejected": 0,
            "approved": 0,
            "quota": 2,
            "quota_shortfall": 1,
            "accepted": 0,
        },
        {
            "route": "technical",
            "candidate_rows": 1,
            "provider_rejected": 0,
            "locally_invalid": 0,
            "selected_for_review": 1,
            "unreviewed": 1,
            "reviewer_rejected": 0,
            "approved": 0,
            "quota": 1,
            "quota_shortfall": 0,
            "accepted": 0,
        },
        {
            "route": "general",
            "candidate_rows": 0,
            "provider_rejected": 0,
            "locally_invalid": 0,
            "selected_for_review": 0,
            "unreviewed": 0,
            "reviewer_rejected": 0,
            "approved": 0,
            "quota": 1,
            "quota_shortfall": 1,
            "accepted": 0,
        },
    ]
    assert quality["review_quality"]["rejection_rate"] == pytest.approx(2 / 6)
    assert quality["review_quality"]["imbalance_detected"] is True

    gold_routes = quality["gold_quality"]["routes"]
    assert gold_routes == [
        {
            "route": "billing",
            "examples": 3,
            "valid_decisions": 2,
            "invalid_decisions": 1,
            "no_decision": 0,
            "route_correct": 1,
            "exact_decision_correct": 1,
            "invalid_decision_rate": pytest.approx(1 / 3),
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
        {
            "route": "general",
            "examples": 2,
            "valid_decisions": 0,
            "invalid_decisions": 1,
            "no_decision": 1,
            "route_correct": 0,
            "exact_decision_correct": 0,
            "invalid_decision_rate": 1.0,
        },
    ]
    assert quality["gold_quality"]["invalid_decision_rate"] == pytest.approx(4 / 6)

    with (report_directory / "review.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert [
        (row["source_id"], row["input"], row["validation_reason"]) for row in csv_rows
    ] == [
        (row["source_id"], row["input"], row["validation"].get("reason", ""))
        for row in rows
    ]
    assert [row["candidate_route"] for row in csv_rows] == [
        "billing",
        "technical",
        "billing",
        "not-a-route",
        "",
        "",
    ]
    repeat = tmp_path / "repeat"
    repeat.mkdir()
    repeat_config = _write_artifacts(repeat)
    _write_config(repeat_config)
    review_label_candidates(repeat_config)
    for name in ("review.jsonl", "review.csv", "report.json", "manifest.json"):
        assert (report_directory / name).read_bytes() == (
            repeat / "review" / name
        ).read_bytes()


@pytest.mark.parametrize("prefix", ("=", "+", "-", "@", "\t", "\r", "\n"))
def test_review_csv_escapes_formula_cells_without_mutating_jsonl(
    tmp_path: Path, prefix: str
) -> None:
    source_id = prefix + 'HYPERLINK("https://attacker.test")'
    source_input = prefix + "SUM(1,1)"
    config = _write_artifacts(
        tmp_path, first_source_id=source_id, first_input=source_input
    )
    _write_config(config)
    document = json.loads(config.read_text())
    document.pop("gold")
    config.write_bytes(_compact(document))

    review_label_candidates(config)

    report_directory = tmp_path / "review"
    jsonl_row = json.loads(
        (report_directory / "review.jsonl").read_text().splitlines()[0]
    )
    with (report_directory / "review.csv").open(newline="") as stream:
        csv_row = next(csv.DictReader(stream))

    assert jsonl_row["source_id"] == source_id
    assert jsonl_row["input"] == source_input
    assert csv_row["source_id"] == "'" + source_id
    assert csv_row["input"] == "'" + source_input


def test_review_preflight_and_no_overwrite_leave_outputs_untouched(
    tmp_path: Path,
) -> None:
    config = _write_artifacts(tmp_path)
    _write_config(config, output="review")
    (tmp_path / "review").mkdir()
    sentinel = tmp_path / "review" / "sentinel"
    sentinel.write_text("preserve")

    with pytest.raises(ReviewConfigError, match="refusing to overwrite"):
        review_label_candidates(config)
    assert sentinel.read_text() == "preserve"

    (tmp_path / "review").rmdir() if not any((tmp_path / "review").iterdir()) else None
    sentinel.unlink()
    (tmp_path / "review").rmdir()
    invalid = json.loads(config.read_text())
    invalid["sampling"]["per_route"] = {"unknown": 1}
    config.write_bytes(_compact(invalid))

    with pytest.raises(ReviewConfigError, match="absent from the active registry"):
        review_label_candidates(config)
    assert not (tmp_path / "review").exists()
