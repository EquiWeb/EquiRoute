"""Stage-9 deterministic review artifacts and generated-label quality evidence.

This module deliberately stops at human review preparation.  It never turns a
candidate into training data, calls a provider, or mutates an existing report.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .dataset import _canonical_compact_json_bytes, _canonical_pretty_json_bytes
from .errors import ReviewConfigError
from .io import (
    LoadedExample,
    LoadedLabelCandidate,
    ReviewInputs,
    iter_examples,
    load_review_inputs,
    resolve_review_paths,
)
from .schemas import (
    GoldQualityReport,
    GoldRouteQuality,
    ReviewArtifact,
    ReviewManifest,
    ReviewQualityReport,
    ReviewRouteQuality,
    ReviewRow,
    ReviewerDecision,
    RouteRegistry,
)

_SCHEMA_VERSION = "2"


def review_label_candidates(config_path: str | Path) -> ReviewManifest:
    """Atomically create an editable review report from verified Stage-7/8 inputs.

    Every candidate has one JSONL row, whether it is valid, malformed, rejected,
    or names an unknown route.  Sampling only selects locally valid labeled
    candidates because only those can ever be approved by the separate acceptance
    stage.  ``review.csv`` is a deterministic view of the authoritative JSONL.
    """
    configuration_path = Path(config_path)
    inputs = load_review_inputs(configuration_path)
    _, _, _, output, gold_path = resolve_review_paths(configuration_path, inputs.config)
    _validate_sampling_routes(
        inputs.config.sampling.per_route if inputs.config.sampling else {},
        inputs.registry,
        configuration_path,
    )
    gold = _load_gold(gold_path, inputs) if gold_path else None
    _ensure_new_output(configuration_path, output)
    selected = _selected_candidate_lines(inputs)
    rows = _review_rows(inputs, selected)
    quality = _review_quality(inputs, rows)
    gold_quality = _gold_quality(inputs, gold, gold_path) if gold is not None else None

    temporary = _prepare_output(output)
    try:
        review_bytes = _review_jsonl(rows)
        _write_bytes(temporary / "review.jsonl", review_bytes)
        _write_bytes(temporary / "review.csv", _review_csv(rows))
        report = {
            "schema_version": _SCHEMA_VERSION,
            "review_quality": quality.model_dump(mode="json"),
        }
        if gold_quality is not None:
            report["gold_quality"] = gold_quality.model_dump(mode="json")
        _write_bytes(temporary / "report.json", _canonical_pretty_json_bytes(report))
        manifest = _review_manifest(inputs, rows, review_bytes)
        _write_bytes(
            temporary / "manifest.json",
            _canonical_pretty_json_bytes(manifest.model_dump(mode="json")),
        )
        if os.path.lexists(output):
            raise FileExistsError("refusing to overwrite an existing output directory")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _ensure_new_output(config_path: Path, output: Path) -> None:
    if os.path.lexists(output):
        raise ReviewConfigError(
            "refusing to overwrite an existing output directory",
            source=config_path,
            path="output.directory",
            correction="choose a new output directory or remove the existing one",
        )


def _prepare_output(output: Path) -> Path:
    return Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))


def _validate_sampling_routes(
    quotas: dict[str, int], registry: RouteRegistry, source: Path
) -> None:
    unknown = sorted(set(quotas) - {route.name for route in registry.routes})
    if unknown:
        raise ReviewConfigError(
            "sampling quotas name routes absent from the active registry: "
            + ", ".join(repr(route) for route in unknown),
            source=source,
            path="sampling.per_route",
            correction="specify quotas only for routes in the configured registry",
        )


def _load_gold(
    gold_path: Path | None, inputs: ReviewInputs
) -> tuple[LoadedExample, ...]:
    assert gold_path is not None
    gold = tuple(iter_examples(gold_path, inputs.registry))
    by_id: set[str] = set()
    candidate_ids = {
        loaded.candidate.provenance.source_id for loaded in inputs.candidates.rows
    }
    for loaded in gold:
        identifier = loaded.example.id
        if identifier is None:
            raise ReviewConfigError(
                "gold examples require an id bound to a candidate source_id",
                source=gold_path,
                line=loaded.line,
                path="id",
                correction="add the matching Stage-7 source id",
            )
        if identifier in by_id:
            raise ReviewConfigError(
                "gold examples must not repeat an id",
                source=gold_path,
                line=loaded.line,
                path="id",
                correction="retain one gold decision per candidate source id",
            )
        if identifier not in candidate_ids:
            raise ReviewConfigError(
                "gold example id is absent from the verified candidate artifact",
                source=gold_path,
                line=loaded.line,
                path="id",
                correction="use gold rows for this Stage-8 candidate artifact",
            )
        by_id.add(identifier)
    return gold


def _selected_candidate_lines(inputs: ReviewInputs) -> set[int]:
    sampling = inputs.config.sampling
    selectable: dict[str, list[LoadedLabelCandidate]] = {
        route.name: [] for route in inputs.registry.routes
    }
    for loaded in inputs.candidates.rows:
        candidate = loaded.candidate
        if not loaded.validation.valid or candidate.decision is None:
            continue
        selectable[candidate.decision.name].append(loaded)

    if sampling is None:
        return {loaded.line for group in selectable.values() for loaded in group}

    selected: set[int] = set()
    for route in inputs.registry.routes:
        quota = sampling.per_route.get(route.name, 0)
        choices = sorted(
            selectable[route.name],
            key=lambda loaded: _sampling_key(
                sampling.seed, route.name, loaded.candidate.provenance.source_id
            ),
        )
        selected.update(loaded.line for loaded in choices[:quota])
    return selected


def _sampling_key(seed: int, route: str, source_id: str) -> bytes:
    return hashlib.sha256(
        f"equiroute-stage9\0{seed}\0{route}\0{source_id}".encode("utf-8")
    ).digest()


def _review_rows(inputs: ReviewInputs, selected: set[int]) -> tuple[ReviewRow, ...]:
    return tuple(
        ReviewRow(
            schema_version=_SCHEMA_VERSION,
            source_id=sanitized.row.id,
            input=sanitized.row.input,
            candidate=candidate.candidate,
            validation=candidate.validation,
            selected_for_review=candidate.line in selected,
            review=ReviewerDecision(decision="unreviewed"),
        )
        for sanitized, candidate in zip(
            inputs.sanitized.rows, inputs.candidates.rows, strict=True
        )
    )


def _review_quality(
    inputs: ReviewInputs, rows: Iterable[ReviewRow]
) -> ReviewQualityReport:
    review_rows = tuple(rows)
    route_rows: dict[str, list[ReviewRow]] = defaultdict(list)
    provider_rejected = 0
    for row in review_rows:
        decision = row.candidate.decision
        if decision is None:
            provider_rejected += 1
            continue
        if inputs.registry.route_named(decision.name) is None:
            continue
        route_rows[decision.name].append(row)

    qualities: list[ReviewRouteQuality] = []
    selection_counts: list[int] = []
    for route in inputs.registry.routes:
        records = route_rows[route.name]
        selected = sum(record.selected_for_review for record in records)
        quota = (
            inputs.config.sampling.per_route.get(route.name, 0)
            if inputs.config.sampling is not None
            else None
        )
        qualities.append(
            ReviewRouteQuality(
                route=route.name,
                candidate_rows=len(records),
                provider_rejected=0,
                locally_invalid=sum(not record.validation.valid for record in records),
                selected_for_review=selected,
                unreviewed=selected,
                reviewer_rejected=0,
                approved=0,
                quota=quota,
                quota_shortfall=max((quota or 0) - selected, 0),
                accepted=0,
            )
        )
        selection_counts.append(selected)

    # Provider rejections are the artifact-level rejection rate. Invalid
    # candidate decisions remain separately visible in their local validation
    # records; no fallback route is invented for unknown-route decisions.
    candidate_total = len(review_rows)
    return ReviewQualityReport(
        schema_version=_SCHEMA_VERSION,
        candidates=ReviewArtifact(
            rows=inputs.candidates.manifest.output.rows,
            sha256=inputs.candidates.manifest.output.sha256,
        ),
        rejection_rate=provider_rejected / candidate_total if candidate_total else 0.0,
        imbalance_detected=_imbalance_detected(selection_counts),
        routes=qualities,
    )


def _imbalance_detected(counts: list[int]) -> bool:
    nonzero = [count for count in counts if count]
    return bool(nonzero) and (
        len(nonzero) != len(counts) or min(nonzero) != max(nonzero)
    )


def _gold_quality(
    inputs: ReviewInputs, gold: tuple[LoadedExample, ...], gold_path: Path | None
) -> GoldQualityReport:
    assert gold_path is not None
    candidates = {
        loaded.candidate.provenance.source_id: loaded
        for loaded in inputs.candidates.rows
    }
    grouped: dict[str, list[tuple[LoadedExample, LoadedLabelCandidate]]] = {
        route.name: [] for route in inputs.registry.routes
    }
    for loaded_gold in gold:
        identifier = loaded_gold.example.id
        assert identifier is not None
        grouped[loaded_gold.example.route.name].append(
            (loaded_gold, candidates[identifier])
        )

    route_quality: list[GoldRouteQuality] = []
    total_invalid_or_missing = 0
    total_examples = 0
    for route in inputs.registry.routes:
        examples = grouped[route.name]
        valid = invalid = missing = route_correct = exact = 0
        for loaded_gold, candidate in examples:
            decision = candidate.candidate.decision
            if decision is None:
                missing += 1
            elif not candidate.validation.valid:
                invalid += 1
            else:
                valid += 1
                if decision.name == loaded_gold.example.route.name:
                    route_correct += 1
                    if decision.arguments == loaded_gold.example.route.arguments:
                        exact += 1
        total_examples += len(examples)
        total_invalid_or_missing += invalid + missing
        route_quality.append(
            GoldRouteQuality(
                route=route.name,
                examples=len(examples),
                valid_decisions=valid,
                invalid_decisions=invalid,
                no_decision=missing,
                route_correct=route_correct,
                exact_decision_correct=exact,
                invalid_decision_rate=(invalid + missing) / len(examples)
                if examples
                else None,
            )
        )

    gold_bytes = _read_bytes(gold_path)
    return GoldQualityReport(
        schema_version=_SCHEMA_VERSION,
        gold=ReviewArtifact(
            rows=len(gold), sha256=hashlib.sha256(gold_bytes).hexdigest()
        ),
        candidates=ReviewArtifact(
            rows=inputs.candidates.manifest.output.rows,
            sha256=inputs.candidates.manifest.output.sha256,
        ),
        invalid_decision_rate=(total_invalid_or_missing / total_examples)
        if total_examples
        else None,
        routes=route_quality,
    )


def _review_jsonl(rows: Iterable[ReviewRow]) -> bytes:
    return b"".join(
        _canonical_compact_json_bytes(row.model_dump(mode="json", exclude_none=True))
        + b"\n"
        for row in rows
    )


def _csv_cell(value: str) -> str:
    """Prevent spreadsheet applications from evaluating a review-view cell."""

    return (
        f"'{value}"
        if value.startswith(("=", "+", "-", "@", "\t", "\r", "\n"))
        else value
    )


def _review_csv(rows: Iterable[ReviewRow]) -> bytes:
    """Render a deterministic, deliberately non-authoritative flat review view."""
    fields = (
        "source_id",
        "input",
        "candidate_status",
        "candidate_route",
        "candidate_arguments",
        "candidate_rejection_reason",
        "validation_valid",
        "validation_reason",
        "selected_for_review",
        "provider_model",
        "provider_endpoint",
        "policy_fingerprint",
        "response_timestamp",
        "review_decision",
        "reviewer",
        "reviewed_at",
        "review_reason",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        candidate = row.candidate
        decision = candidate.decision
        writer.writerow(
            {
                "source_id": _csv_cell(row.source_id),
                "input": _csv_cell(row.input),
                "candidate_status": _csv_cell(candidate.status),
                "candidate_route": _csv_cell(decision.name) if decision else "",
                "candidate_arguments": _csv_cell(
                    _canonical_compact_json_bytes(decision.arguments).decode("utf-8")
                )
                if decision
                else "",
                "candidate_rejection_reason": _csv_cell(
                    candidate.rejection_reason or ""
                ),
                "validation_valid": str(row.validation.valid).lower(),
                "validation_reason": _csv_cell(row.validation.reason or ""),
                "selected_for_review": str(row.selected_for_review).lower(),
                "provider_model": _csv_cell(candidate.provenance.provider_model),
                "provider_endpoint": _csv_cell(candidate.provenance.provider_endpoint),
                "policy_fingerprint": _csv_cell(
                    candidate.provenance.policy_fingerprint
                ),
                "response_timestamp": _csv_cell(
                    candidate.provenance.response_timestamp
                ),
                "review_decision": _csv_cell(row.review.decision),
                "reviewer": _csv_cell(row.review.reviewer or ""),
                "reviewed_at": _csv_cell(row.review.reviewed_at or ""),
                "review_reason": _csv_cell(row.review.reason or ""),
            }
        )
    return stream.getvalue().encode("utf-8")


def _review_manifest(
    inputs: ReviewInputs, rows: Iterable[ReviewRow], review_bytes: bytes
) -> ReviewManifest:
    review_rows = tuple(rows)
    immutable = hashlib.sha256()
    for row in review_rows:
        document = row.model_dump(mode="json")
        document.pop("review")
        immutable.update(_canonical_compact_json_bytes(document))
        immutable.update(b"\n")
    return ReviewManifest(
        schema_version=_SCHEMA_VERSION,
        sanitized={
            "rows": inputs.sanitized.manifest.output.rows,
            "sha256": inputs.sanitized.manifest.output.sha256,
        },
        sanitized_manifest_fingerprint=_file_sha256(
            inputs.sanitized.directory / "manifest.json"
        ),
        candidates={
            "rows": inputs.candidates.manifest.output.rows,
            "sha256": inputs.candidates.manifest.output.sha256,
        },
        candidate_manifest_fingerprint=_file_sha256(
            inputs.candidates.directory / "manifest.json"
        ),
        registry_fingerprint=hashlib.sha256(
            _canonical_compact_json_bytes(inputs.registry.model_dump(mode="json"))
        ).hexdigest(),
        policy_fingerprint=inputs.candidates.manifest.policy_fingerprint,
        provider_model=inputs.candidates.manifest.provider_model,
        provider_endpoint=inputs.candidates.manifest.provider_endpoint,
        sampling=inputs.config.sampling,
        rows={
            "rows": len(review_rows),
            "sha256": hashlib.sha256(review_bytes).hexdigest(),
        },
        immutable_fingerprint=immutable.hexdigest(),
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(_read_bytes(path)).hexdigest()


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ReviewConfigError(
            "could not read gold examples",
            source=path,
            correction="ensure the configured gold JSONL exists and is readable",
        ) from error


def _write_bytes(path: Path, content: bytes) -> None:
    path.write_bytes(content)
