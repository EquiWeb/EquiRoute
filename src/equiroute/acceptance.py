"""Stage-9 atomic compilation of explicitly approved labels into canonical examples."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .dataset import (
    _canonical_compact_json_bytes,
    _canonical_pretty_json_bytes,
    validate_dataset,
)
from .errors import AcceptanceConfigError
from .io import (
    AcceptanceInputs,
    LoadedExample,
    LoadedReviewRow,
    iter_examples,
    load_acceptance_inputs,
    resolve_acceptance_paths,
)
from .schemas import (
    AcceptanceManifest,
    AcceptedLabelProvenance,
    Example,
    GoldQualityReport,
    GoldRouteQuality,
    ReviewArtifact,
    ReviewQualityReport,
    ReviewRouteQuality,
    RouteRegistry,
)

_SCHEMA_VERSION = "2"


def accept_approved_labels(config_path: str | Path) -> AcceptanceManifest:
    """Atomically emit Stage-1-valid examples from verified review approvals only."""
    configuration_path = Path(config_path)
    inputs = load_acceptance_inputs(configuration_path)
    _, _, _, _, review_manifest_path, output, gold_path = resolve_acceptance_paths(
        configuration_path, inputs.config
    )
    _validate_quota_routes(inputs, configuration_path)
    gold_quality = _gold_quality(inputs, gold_path) if gold_path is not None else None
    _ensure_new_output(configuration_path, output)

    accepted = _select_approved(inputs)
    examples = _accepted_examples(inputs, accepted, review_manifest_path)
    review_quality = _review_quality(inputs, accepted)
    temporary = _prepare_output(output)
    try:
        examples_bytes = _examples_jsonl(examples)
        examples_path = temporary / "examples.jsonl"
        examples_path.write_bytes(examples_bytes)
        # Re-read the emitted canonical bytes through the Stage-1 validator rather
        # than trusting in-memory construction.
        validate_dataset(examples_path, inputs.registry)

        report: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "review_quality": review_quality.model_dump(mode="json"),
        }
        if gold_quality is not None:
            report["gold_quality"] = gold_quality.model_dump(mode="json")
        (temporary / "report.json").write_bytes(_canonical_pretty_json_bytes(report))

        manifest = _acceptance_manifest(inputs, review_manifest_path, examples_bytes)
        (temporary / "manifest.json").write_bytes(
            _canonical_pretty_json_bytes(manifest.model_dump(mode="json"))
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
        raise AcceptanceConfigError(
            "refusing to overwrite an existing output directory",
            source=config_path,
            path="output.directory",
            correction="choose a new output directory or remove the existing one",
        )


def _prepare_output(output: Path) -> Path:
    return Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))


def _validate_quota_routes(inputs: AcceptanceInputs, source: Path) -> None:
    quotas = inputs.config.quotas
    if quotas is None:
        return
    known = {route.name for route in inputs.registry.routes}
    unknown = sorted(set(quotas.per_route) - known)
    if unknown:
        raise AcceptanceConfigError(
            "acceptance quotas name routes absent from the active registry: "
            + ", ".join(repr(route) for route in unknown),
            source=source,
            path="quotas.per_route",
            correction="specify quotas only for routes in the configured registry",
        )


def _select_approved(inputs: AcceptanceInputs) -> tuple[LoadedReviewRow, ...]:
    approved_by_route: dict[str, list[LoadedReviewRow]] = defaultdict(list)
    for loaded in inputs.approved:
        row = loaded.row
        decision = row.candidate.decision
        if (
            row.review.decision != "approved"
            or not row.selected_for_review
            or not row.validation.valid
            or row.candidate.status != "labeled"
            or decision is None
        ):
            # The review loader and schema make this unreachable for a valid
            # handoff, but keeping the acceptance gate explicit prevents a later
            # contract relaxation from silently widening training eligibility.
            continue
        approved_by_route[decision.name].append(loaded)

    quotas = inputs.config.quotas
    if quotas is None:
        return tuple(
            sorted(
                (
                    loaded
                    for route in inputs.registry.routes
                    for loaded in approved_by_route[route.name]
                ),
                key=lambda loaded: loaded.line,
            )
        )

    selected: list[LoadedReviewRow] = []
    for route in inputs.registry.routes:
        quota = quotas.per_route.get(route.name, 0)
        ranked = sorted(
            approved_by_route[route.name],
            key=lambda loaded: (
                _quota_key(quotas.seed, route.name, loaded.row.source_id),
                loaded.line,
            ),
        )
        selected.extend(ranked[:quota])
    return tuple(sorted(selected, key=lambda loaded: loaded.line))


def _quota_key(seed: int, route: str, source_id: str) -> bytes:
    return hashlib.sha256(
        f"equiroute-stage9-acceptance\0{seed}\0{route}\0{source_id}".encode("utf-8")
    ).digest()


def _accepted_examples(
    inputs: AcceptanceInputs,
    accepted: Iterable[LoadedReviewRow],
    review_manifest_path: Path,
) -> tuple[Example, ...]:
    examples: list[Example] = []
    review_manifest_fingerprint = _file_sha256(review_manifest_path)
    for loaded in accepted:
        candidate = inputs.candidates.rows[loaded.line - 1]
        sanitized = inputs.sanitized.rows[loaded.line - 1]
        decision = candidate.candidate.decision
        reviewer = loaded.row.review.reviewer
        reviewed_at = loaded.row.review.reviewed_at
        if decision is None or reviewer is None or reviewed_at is None:
            raise AcceptanceConfigError(
                "verified approved row lacks a decision or reviewer details",
                source=loaded.source,
                line=loaded.line,
                correction="use an unmodified verified review report",
            )
        provenance = AcceptedLabelProvenance(
            source_id=sanitized.row.id,
            source_line=candidate.candidate.provenance.source_line,
            candidate_sha256=candidate.sha256,
            candidate_artifact_sha256=inputs.candidates.manifest.output.sha256,
            handoff_rows_sha256=inputs.sanitized.manifest.output.sha256,
            handoff_manifest_fingerprint=_file_sha256(
                inputs.sanitized.directory / "manifest.json"
            ),
            registry_fingerprint=inputs.candidates.manifest.registry_fingerprint,
            policy_fingerprint=candidate.candidate.provenance.policy_fingerprint,
            provider_model=candidate.candidate.provenance.provider_model,
            provider_endpoint=candidate.candidate.provenance.provider_endpoint,
            reviewer=reviewer,
            reviewed_at=reviewed_at,
            review_rows_sha256=inputs.review.sha256,
        ).model_dump(mode="json")
        provenance.update(
            {
                "review_decision": "approved",
                "review_reason": loaded.row.review.reason,
                "review_manifest_fingerprint": review_manifest_fingerprint,
                "immutable_review_fingerprint": inputs.review.manifest.immutable_fingerprint,
            }
        )
        examples.append(
            Example(
                id=sanitized.row.id,
                input=sanitized.row.input,
                route=decision,
                metadata={"_equiroute": provenance},
            )
        )
    return tuple(examples)


def _examples_jsonl(examples: Iterable[Example]) -> bytes:
    return b"".join(
        _canonical_compact_json_bytes(
            example.model_dump(mode="json", exclude_none=True)
        )
        + b"\n"
        for example in examples
    )


def _review_quality(
    inputs: AcceptanceInputs, accepted: Iterable[LoadedReviewRow]
) -> ReviewQualityReport:
    accepted_lines = {loaded.line for loaded in accepted}
    route_rows: dict[str, list[LoadedReviewRow]] = defaultdict(list)
    provider_rejected = 0
    for loaded in inputs.review.rows:
        decision = loaded.row.candidate.decision
        if decision is None:
            provider_rejected += 1
        elif inputs.registry.route_named(decision.name) is not None:
            route_rows[decision.name].append(loaded)

    qualities: list[ReviewRouteQuality] = []
    accepted_counts: list[int] = []
    for route in inputs.registry.routes:
        records = route_rows[route.name]
        quota = (
            inputs.config.quotas.per_route.get(route.name, 0)
            if inputs.config.quotas is not None
            else None
        )
        approved = sum(record.row.review.decision == "approved" for record in records)
        accepted_count = sum(record.line in accepted_lines for record in records)
        qualities.append(
            ReviewRouteQuality(
                route=route.name,
                candidate_rows=len(records),
                provider_rejected=0,
                locally_invalid=sum(
                    not record.row.validation.valid for record in records
                ),
                selected_for_review=sum(
                    record.row.selected_for_review for record in records
                ),
                unreviewed=sum(
                    record.row.selected_for_review
                    and record.row.review.decision == "unreviewed"
                    for record in records
                ),
                reviewer_rejected=sum(
                    record.row.selected_for_review
                    and record.row.review.decision == "rejected"
                    for record in records
                ),
                approved=approved,
                quota=quota,
                quota_shortfall=max((quota or 0) - accepted_count, 0),
                accepted=accepted_count,
            )
        )
        accepted_counts.append(accepted_count)
    candidate_total = len(inputs.candidates.rows)
    return ReviewQualityReport(
        schema_version=_SCHEMA_VERSION,
        candidates=ReviewArtifact(
            rows=inputs.candidates.manifest.output.rows,
            sha256=inputs.candidates.manifest.output.sha256,
        ),
        rejection_rate=provider_rejected / candidate_total if candidate_total else 0.0,
        imbalance_detected=_imbalance_detected(accepted_counts),
        routes=qualities,
    )


def _imbalance_detected(counts: list[int]) -> bool:
    nonzero = [count for count in counts if count]
    return bool(nonzero) and (
        len(nonzero) != len(counts) or min(nonzero) != max(nonzero)
    )


def _gold_quality(inputs: AcceptanceInputs, gold_path: Path) -> GoldQualityReport:
    gold = tuple(iter_examples(gold_path, inputs.registry))
    candidates = {
        loaded.candidate.provenance.source_id: loaded
        for loaded in inputs.candidates.rows
    }
    seen: set[str] = set()
    grouped: dict[str, list[LoadedExample]] = defaultdict(list)
    for loaded in gold:
        identifier = loaded.example.id
        if identifier is None:
            raise AcceptanceConfigError(
                "gold examples require an id bound to a candidate source_id",
                source=gold_path,
                line=loaded.line,
                path="id",
                correction="add the matching Stage-7 source id",
            )
        if identifier in seen:
            raise AcceptanceConfigError(
                "gold examples must not repeat an id",
                source=gold_path,
                line=loaded.line,
                path="id",
                correction="retain one gold decision per candidate source id",
            )
        if identifier not in candidates:
            raise AcceptanceConfigError(
                "gold example id is absent from the verified candidate artifact",
                source=gold_path,
                line=loaded.line,
                path="id",
                correction="use gold rows for this Stage-8 candidate artifact",
            )
        seen.add(identifier)
        grouped[loaded.example.route.name].append(loaded)

    quality: list[GoldRouteQuality] = []
    total_examples = total_invalid_or_missing = 0
    for route in inputs.registry.routes:
        examples = grouped[route.name]
        valid = invalid = missing = route_correct = exact = 0
        for loaded in examples:
            candidate = candidates[loaded.example.id or ""]
            decision = candidate.candidate.decision
            if decision is None:
                missing += 1
            elif not candidate.validation.valid:
                invalid += 1
            else:
                valid += 1
                if decision.name == loaded.example.route.name:
                    route_correct += 1
                    if decision.arguments == loaded.example.route.arguments:
                        exact += 1
        total_examples += len(examples)
        total_invalid_or_missing += invalid + missing
        quality.append(
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
        routes=quality,
    )


def _acceptance_manifest(
    inputs: AcceptanceInputs, review_manifest_path: Path, examples: bytes
) -> AcceptanceManifest:
    return AcceptanceManifest(
        schema_version=_SCHEMA_VERSION,
        sanitized=ReviewArtifact(
            rows=inputs.sanitized.manifest.output.rows,
            sha256=inputs.sanitized.manifest.output.sha256,
        ),
        sanitized_manifest_fingerprint=_file_sha256(
            inputs.sanitized.directory / "manifest.json"
        ),
        candidates=ReviewArtifact(
            rows=inputs.candidates.manifest.output.rows,
            sha256=inputs.candidates.manifest.output.sha256,
        ),
        candidate_manifest_fingerprint=_file_sha256(
            inputs.candidates.directory / "manifest.json"
        ),
        review=ReviewArtifact(
            rows=len(inputs.review.rows), sha256=inputs.review.sha256
        ),
        review_manifest_fingerprint=_file_sha256(review_manifest_path),
        immutable_review_fingerprint=inputs.review.manifest.immutable_fingerprint,
        registry_fingerprint=inputs.candidates.manifest.registry_fingerprint,
        policy_fingerprint=inputs.candidates.manifest.policy_fingerprint,
        provider_model=inputs.candidates.manifest.provider_model,
        provider_endpoint=inputs.candidates.manifest.provider_endpoint,
        quotas=inputs.config.quotas,
        output=ReviewArtifact(
            rows=examples.count(b"\n"), sha256=hashlib.sha256(examples).hexdigest()
        ),
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(_read_bytes(path)).hexdigest()


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise AcceptanceConfigError(
            "could not read an acceptance input artifact",
            source=path,
            correction="ensure the configured artifact exists and is readable",
        ) from error
