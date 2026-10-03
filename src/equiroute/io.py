"""Load and validate EquiRoute YAML and JSONL contracts from local files."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from .decisions import DecisionValidationError, _argument_correction, validate_decision
from .errors import (
    AcceptanceConfigError,
    CandidateArtifactLoadError,
    ConfigLoadError,
    ExampleLoadError,
    GoldQualityConfigError,
    LabelingConfigError,
    RawIngestionConfigError,
    RawInputLoadError,
    RegistryLoadError,
    ReviewConfigError,
    ReviewRecordLoadError,
    SanitizedArtifactLoadError,
    SourceError,
)
from .migrations import (
    SchemaMigrationError,
    migrate_acceptance_config,
    migrate_gold_quality_config,
    migrate_labeling_config,
    migrate_labeling_manifest,
    migrate_raw_ingestion_config,
    migrate_raw_ingestion_manifest,
    migrate_review_config,
    migrate_review_manifest,
    migrate_training_config,
)

from .schemas import (
    AcceptanceConfig,
    CandidateValidation,
    Decision,
    Example,
    GoldQualityConfig,
    LabelCandidate,
    LabelingConfig,
    LabelingManifest,
    RawIngestionConfig,
    RawIngestionManifest,
    RawInputRow,
    ReviewConfig,
    ReviewManifest,
    ReviewRow,
    RouteRegistry,
    TrainingConfig,
    _json_pointer_tokens,
)

_Model = TypeVar("_Model", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class LoadedExample:
    """A validated example with its source location."""

    example: Example
    source: Path
    line: int


@dataclass(frozen=True, slots=True)
class LoadedRawInput:
    """A projected, redacted raw input with its source location."""

    raw_input: RawInputRow
    source: Path
    line: int

    @property
    def row(self) -> RawInputRow:
        """Return the canonical row under its concise pipeline name."""

        return self.raw_input


@dataclass(frozen=True, slots=True)
class LoadedSanitizedInput:
    """A verified Stage-7 row with its sanitized artifact location."""

    row: RawInputRow
    source: Path
    line: int


@dataclass(frozen=True, slots=True)
class SanitizedHandoff:
    """A fully verified Stage-7 handoff safe to pass to provider work."""

    directory: Path
    manifest: RawIngestionManifest
    rows: tuple[LoadedSanitizedInput, ...]


@dataclass(frozen=True, slots=True)
class LoadedLabelCandidate:
    """A verified Stage-8 candidate and its immutable JSONL row digest."""

    candidate: LabelCandidate
    source: Path
    line: int
    sha256: str
    validation: CandidateValidation


@dataclass(frozen=True, slots=True)
class LabelingHandoff:
    """A Stage-8 candidate artifact bound to its verified Stage-7 source."""

    directory: Path
    manifest: LabelingManifest
    rows: tuple[LoadedLabelCandidate, ...]


@dataclass(frozen=True, slots=True)
class LoadedReviewRow:
    """A verified review row whose immutable data matches the source artifacts."""

    row: ReviewRow
    source: Path
    line: int


@dataclass(frozen=True, slots=True)
class ReviewHandoff:
    """A review report whose non-review fields remain bound to original inputs."""

    manifest: ReviewManifest
    rows: tuple[LoadedReviewRow, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class ReviewInputs:
    """Verified immutable sources for a future review-report writer."""

    config: ReviewConfig
    sanitized: SanitizedHandoff
    candidates: LabelingHandoff
    registry: RouteRegistry


@dataclass(frozen=True, slots=True)
class AcceptanceInputs:
    """Verified sources plus only approved rows for a future acceptance writer."""

    config: AcceptanceConfig
    sanitized: SanitizedHandoff
    candidates: LabelingHandoff
    registry: RouteRegistry
    review: ReviewHandoff
    approved: tuple[LoadedReviewRow, ...]


def iter_raw_inputs(
    config_or_path: RawIngestionConfig | str | Path,
    config: RawIngestionConfig | None = None,
    *,
    content_hasher: Any | None = None,
) -> Iterator[LoadedRawInput]:
    """Stream configured raw JSONL as canonical, redacted rows.

    Pass a config alone to use its configured source.  The explicit
    ``path, config`` form is available to a future orchestrator that resolves
    a relative config source before streaming.
    """

    if isinstance(config_or_path, RawIngestionConfig):
        if config is not None:
            raise TypeError("iter_raw_inputs accepts either config or path plus config")
        ingestion_config = config_or_path
        source = Path(ingestion_config.source)
    else:
        if config is None:
            raise TypeError("iter_raw_inputs requires a RawIngestionConfig")
        ingestion_config = config
        source = Path(config_or_path)

    ids: dict[str, int] = {}
    try:
        with source.open("rb") as inputs_file:
            for line_number, raw_line in enumerate(inputs_file, start=1):
                if content_hasher is not None:
                    content_hasher.update(raw_line)
                document = _load_raw_json_object(source, line_number, raw_line)
                projected = _project_raw_input(
                    document, ingestion_config, source, line_number
                )
                _apply_raw_redactions(projected, ingestion_config, source, line_number)
                _validate_projected_raw_input_utf8(projected, source, line_number)
                raw_input = _validate_raw_input_row(projected, source, line_number)
                _validate_raw_input_size(
                    raw_input, ingestion_config, source, line_number
                )

                first_line = ids.get(raw_input.id)
                if first_line is not None:
                    raise RawInputLoadError(
                        f"duplicate raw input id; first declared on line {first_line}",
                        source=source,
                        line=line_number,
                        path="id",
                        correction="assign a unique id",
                    )
                ids[raw_input.id] = line_number
                yield LoadedRawInput(
                    raw_input=raw_input, source=source, line=line_number
                )
    except OSError as error:
        raise RawInputLoadError(
            f"could not read raw inputs: {error}",
            source=source,
            correction="ensure the file exists and is valid UTF-8 JSONL",
        ) from error


def iter_examples(
    path: str | Path, registry: RouteRegistry, *, content_hasher: Any | None = None
) -> Iterator[LoadedExample]:
    """Stream validated JSONL examples with their one-based source locations."""

    source = Path(path)
    try:
        with source.open("rb") as examples_file:
            for line_number, raw_line in enumerate(examples_file, start=1):
                if content_hasher is not None:
                    content_hasher.update(raw_line)
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ExampleLoadError(
                        f"could not decode UTF-8: {error}",
                        source=source,
                        line=line_number,
                        path="$",
                        correction="replace this line with valid UTF-8 JSON",
                    ) from error
                if not line.strip():
                    raise ExampleLoadError(
                        "expected a JSON object, got a blank line",
                        source=source,
                        line=line_number,
                        path="$",
                        correction="remove blank lines; each line must be a JSON object",
                    )
                try:
                    document = json.loads(
                        line, parse_constant=_reject_nonstandard_json_constant
                    )
                except (json.JSONDecodeError, ValueError) as error:
                    raise ExampleLoadError(
                        f"malformed JSON: {error}",
                        source=source,
                        line=line_number,
                        path="$",
                        correction="replace this line with a valid JSON object",
                    ) from error
                if not isinstance(document, dict):
                    raise ExampleLoadError(
                        "expected a JSON object",
                        source=source,
                        line=line_number,
                        path="$",
                        correction="replace this line with a JSON object",
                    )

                example = _validate_model(
                    document, Example, source, ExampleLoadError, "example", line_number
                )
                _validate_decision(example.route, registry, source, line_number)
                yield LoadedExample(example=example, source=source, line=line_number)
    except OSError as error:
        raise ExampleLoadError(
            f"could not read examples: {error}",
            source=source,
            correction="ensure the file exists and is valid UTF-8",
        ) from error


def load_route_registry(path: str | Path) -> RouteRegistry:
    """Load one strict route registry from a YAML document."""

    source = Path(path)
    document = _load_yaml_mapping(source, RegistryLoadError, "route registry")
    return _validate_model(
        document, RouteRegistry, source, RegistryLoadError, "route registry"
    )


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load one strict training configuration from a YAML document."""

    source = Path(path)
    document = _load_yaml_mapping(source, ConfigLoadError, "training configuration")
    document = _migrate_training_config(document, source)
    return _validate_model(
        document, TrainingConfig, source, ConfigLoadError, "training configuration"
    )


def load_raw_ingestion_config(path: str | Path) -> RawIngestionConfig:
    """Load one strict v2-only raw-input ingestion configuration."""

    source = Path(path)
    document = _load_yaml_mapping(
        source, RawIngestionConfigError, "raw ingestion configuration"
    )
    document = _migrate_raw_ingestion_config(document, source)
    return _validate_raw_ingestion_config(document, source)


def load_labeling_config(path: str | Path) -> LabelingConfig:
    """Load one strict v2-only, non-secret candidate-labeling configuration."""

    source = Path(path)
    document = _load_yaml_mapping(source, LabelingConfigError, "labeling configuration")
    document = _migrate_labeling_config(document, source)
    return _validate_labeling_config(document, source)


def load_review_config(path: str | Path) -> ReviewConfig:
    """Load one strict v2-only candidate-review configuration."""

    source = Path(path)
    document = _load_yaml_mapping(source, ReviewConfigError, "review configuration")
    document = _migrate_stage9_config(
        document,
        source,
        migrate_review_config,
        ReviewConfigError,
        "review configuration",
    )
    return _validate_model(
        document, ReviewConfig, source, ReviewConfigError, "review configuration"
    )


def load_acceptance_config(path: str | Path) -> AcceptanceConfig:
    """Load one strict v2-only label-acceptance configuration."""

    source = Path(path)
    document = _load_yaml_mapping(
        source, AcceptanceConfigError, "acceptance configuration"
    )
    document = _migrate_stage9_config(
        document,
        source,
        migrate_acceptance_config,
        AcceptanceConfigError,
        "acceptance configuration",
    )
    return _validate_model(
        document,
        AcceptanceConfig,
        source,
        AcceptanceConfigError,
        "acceptance configuration",
    )


def load_gold_quality_config(path: str | Path) -> GoldQualityConfig:
    """Load one strict v2-only generated-label gold-quality configuration."""

    source = Path(path)
    document = _load_yaml_mapping(
        source, GoldQualityConfigError, "gold quality configuration"
    )
    document = _migrate_stage9_config(
        document,
        source,
        migrate_gold_quality_config,
        GoldQualityConfigError,
        "gold quality configuration",
    )
    return _validate_model(
        document,
        GoldQualityConfig,
        source,
        GoldQualityConfigError,
        "gold quality configuration",
    )


def resolve_review_paths(
    config_path: str | Path, config: ReviewConfig
) -> tuple[Path, Path, Path, Path, Path | None]:
    """Resolve review inputs and optional gold data relative to its configuration."""

    directory = Path(config_path).parent
    return (
        directory / config.sanitized,
        directory / config.candidates,
        directory / config.routes,
        directory / config.output.directory,
        directory / config.gold if config.gold is not None else None,
    )


def resolve_acceptance_paths(
    config_path: str | Path, config: AcceptanceConfig
) -> tuple[Path, Path, Path, Path, Path, Path, Path | None]:
    """Resolve acceptance inputs and optional gold data relative to its configuration."""

    directory = Path(config_path).parent
    return (
        directory / config.sanitized,
        directory / config.candidates,
        directory / config.routes,
        directory / config.review,
        directory / config.review_manifest,
        directory / config.output.directory,
        directory / config.gold if config.gold is not None else None,
    )


def resolve_gold_quality_paths(
    config_path: str | Path, config: GoldQualityConfig
) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve gold comparison inputs and output relative to its configuration."""

    directory = Path(config_path).parent
    return (
        directory / config.sanitized,
        directory / config.candidates,
        directory / config.routes,
        directory / config.gold,
        directory / config.output.directory,
    )


def resolve_labeling_paths(
    config_path: str | Path, config: LabelingConfig
) -> tuple[Path, Path, Path]:
    """Resolve handoff, registry, and output paths relative to a config file."""

    directory = Path(config_path).parent
    return (
        directory / config.input.directory,
        directory / config.routes,
        directory / config.output.directory,
    )


def load_labeling_route_registry(
    config_path: str | Path, config: LabelingConfig
) -> RouteRegistry:
    """Load the route registry selected by a labeling configuration."""

    _, registry_path, _ = resolve_labeling_paths(config_path, config)
    return load_route_registry(registry_path)


def load_sanitized_handoff(path: str | Path) -> SanitizedHandoff:
    """Read only a Stage-7 handoff and verify its rows before provider work."""

    directory = Path(path)
    manifest_path = directory / "manifest.json"
    rows_path = directory / "rows.jsonl"
    manifest_document = _load_sanitized_manifest(manifest_path)
    try:
        migrated = migrate_raw_ingestion_manifest(manifest_document)
    except SchemaMigrationError as error:
        raise SanitizedArtifactLoadError(
            str(error),
            source=manifest_path,
            path="schema_version",
            correction='use the Stage-7 schema_version "2" manifest',
        ) from error
    assert isinstance(migrated, dict)
    manifest = _validate_model(
        migrated,
        RawIngestionManifest,
        manifest_path,
        SanitizedArtifactLoadError,
        "sanitized handoff manifest",
    )

    try:
        rows_bytes = rows_path.read_bytes()
    except OSError as error:
        raise SanitizedArtifactLoadError(
            "could not read sanitized rows",
            source=rows_path,
            correction="ensure the Stage-7 rows.jsonl artifact exists and is readable",
        ) from error

    if hashlib.sha256(rows_bytes).hexdigest() != manifest.output.sha256:
        raise SanitizedArtifactLoadError(
            "sanitized rows SHA-256 does not match the handoff manifest",
            source=rows_path,
            path="sha256",
            correction="use the unmodified rows.jsonl emitted with this manifest",
        )
    if rows_bytes and not rows_bytes.endswith(b"\n"):
        raise SanitizedArtifactLoadError(
            "sanitized rows must end every JSON object with an LF",
            source=rows_path,
            correction="use the rows.jsonl emitted by Stage 7",
        )

    row_count = rows_bytes.count(b"\n")
    if row_count != manifest.output.rows:
        raise SanitizedArtifactLoadError(
            "sanitized row count does not match the handoff manifest",
            source=rows_path,
            path="rows",
            correction="use the rows.jsonl emitted with this manifest",
        )

    rows: list[LoadedSanitizedInput] = []
    ids: dict[str, int] = {}
    for line_number, raw_line in enumerate(
        rows_bytes.splitlines(keepends=True), start=1
    ):
        document = _load_sanitized_json_object(rows_path, line_number, raw_line)
        row = _validate_model(
            document,
            RawInputRow,
            rows_path,
            SanitizedArtifactLoadError,
            "sanitized input row",
            line_number,
        )
        first_line = ids.get(row.id)
        if first_line is not None:
            raise SanitizedArtifactLoadError(
                f"duplicate sanitized input id; first declared on line {first_line}",
                source=rows_path,
                line=line_number,
                path="id",
                correction="use the unmodified Stage-7 handoff",
            )
        ids[row.id] = line_number
        rows.append(LoadedSanitizedInput(row=row, source=rows_path, line=line_number))

    return SanitizedHandoff(directory=directory, manifest=manifest, rows=tuple(rows))


def load_labeling_handoff(
    path: str | Path,
    *,
    sanitized: SanitizedHandoff,
    registry: RouteRegistry,
) -> LabelingHandoff:
    """Verify a Stage-8 artifact against its Stage-7 handoff and registry."""

    directory = Path(path)
    manifest_path = directory / "manifest.json"
    candidates_path = directory / "candidates.jsonl"
    manifest_document = _load_artifact_manifest(
        manifest_path, CandidateArtifactLoadError, "candidate artifact manifest"
    )
    try:
        migrated = migrate_labeling_manifest(manifest_document)
    except SchemaMigrationError as error:
        raise CandidateArtifactLoadError(
            str(error),
            source=manifest_path,
            path="schema_version",
            correction='use the Stage-8 schema_version "2" manifest',
        ) from error
    assert isinstance(migrated, dict)
    manifest = _validate_model(
        migrated,
        LabelingManifest,
        manifest_path,
        CandidateArtifactLoadError,
        "candidate artifact manifest",
    )
    candidates_bytes = _read_verified_artifact(
        candidates_path,
        manifest.output.rows,
        manifest.output.sha256,
        CandidateArtifactLoadError,
        "candidate rows",
    )
    _verify_labeling_handoff_bindings(
        manifest,
        manifest_path,
        sanitized=sanitized,
        registry=registry,
    )

    if len(sanitized.rows) != manifest.output.rows:
        raise CandidateArtifactLoadError(
            "candidate artifact does not cover every sanitized input",
            source=candidates_path,
            correction="use the unmodified Stage-8 artifact for this handoff",
        )

    rows: list[LoadedLabelCandidate] = []
    for line_number, raw_line in enumerate(
        candidates_bytes.splitlines(keepends=True), start=1
    ):
        document = _load_artifact_json_object(
            candidates_path,
            line_number,
            raw_line,
            CandidateArtifactLoadError,
            "candidate row",
        )
        candidate = _validate_model(
            document,
            LabelCandidate,
            candidates_path,
            CandidateArtifactLoadError,
            "candidate row",
            line_number,
        )
        expected = sanitized.rows[line_number - 1].row
        if (
            candidate.provenance.source_id != expected.id
            or candidate.provenance.source_line
            != expected.metadata["_equiroute"]["source_line"]
        ):
            raise CandidateArtifactLoadError(
                "candidate source binding does not match the sanitized handoff",
                source=candidates_path,
                line=line_number,
                correction="use the unmodified Stage-8 artifact for this handoff",
            )
        _verify_candidate_manifest_provenance(
            candidate, manifest, candidates_path, line_number
        )
        rows.append(
            LoadedLabelCandidate(
                candidate=candidate,
                source=candidates_path,
                line=line_number,
                sha256=hashlib.sha256(raw_line).hexdigest(),
                validation=_candidate_validation(candidate, registry),
            )
        )
    return LabelingHandoff(directory=directory, manifest=manifest, rows=tuple(rows))


def load_review_manifest(path: str | Path) -> ReviewManifest:
    """Load one strict v2-only review manifest without exposing report contents."""

    source = Path(path)
    document = _load_artifact_manifest(source, ReviewRecordLoadError, "review manifest")
    try:
        migrated = migrate_review_manifest(document)
    except SchemaMigrationError as error:
        raise ReviewRecordLoadError(
            str(error),
            source=source,
            path="schema_version",
            correction='use the Stage-9 schema_version "2" manifest',
        ) from error
    assert isinstance(migrated, dict)
    return _validate_model(
        migrated, ReviewManifest, source, ReviewRecordLoadError, "review manifest"
    )


def load_review_handoff(
    path: str | Path,
    *,
    manifest: ReviewManifest,
    manifest_path: str | Path,
    sanitized: SanitizedHandoff,
    candidates: LabelingHandoff,
    registry: RouteRegistry,
) -> ReviewHandoff:
    """Verify a review report while allowing only reviewer decision fields to change."""

    report_path = Path(path)
    source_manifest = Path(manifest_path)
    _verify_review_manifest_bindings(
        manifest,
        source_manifest,
        sanitized=sanitized,
        candidates=candidates,
        registry=registry,
    )
    expected_rows = len(sanitized.rows)
    if len(candidates.rows) != expected_rows or manifest.rows.rows != expected_rows:
        raise ReviewRecordLoadError(
            "review report row count does not match verified source artifacts",
            source=source_manifest,
            path="rows",
            correction="use review inputs from one verified Stage-7 and Stage-8 handoff",
        )

    review_bytes = _read_artifact_bytes(
        report_path, ReviewRecordLoadError, "review rows"
    )
    _verify_artifact_lf_and_count(
        review_bytes,
        report_path,
        manifest.rows.rows,
        ReviewRecordLoadError,
        "review rows",
    )
    rows: list[LoadedReviewRow] = []
    immutable = hashlib.sha256()
    for line_number, raw_line in enumerate(
        review_bytes.splitlines(keepends=True), start=1
    ):
        document = _load_artifact_json_object(
            report_path,
            line_number,
            raw_line,
            ReviewRecordLoadError,
            "review row",
        )
        row = _validate_model(
            document,
            ReviewRow,
            report_path,
            ReviewRecordLoadError,
            "review row",
            line_number,
        )
        expected_candidate = candidates.rows[line_number - 1]
        expected_input = sanitized.rows[line_number - 1].row
        if (
            row.source_id != expected_input.id
            or row.input != expected_input.input
            or row.candidate != expected_candidate.candidate
            or row.validation != expected_candidate.validation
        ):
            raise ReviewRecordLoadError(
                "review row immutable data does not match verified source artifacts",
                source=report_path,
                line=line_number,
                correction="edit only the review decision fields",
            )
        immutable.update(_immutable_review_row_bytes(row))
        immutable.update(b"\n")
        rows.append(LoadedReviewRow(row=row, source=report_path, line=line_number))
    if immutable.hexdigest() != manifest.immutable_fingerprint:
        raise ReviewRecordLoadError(
            "review report immutable fingerprint does not match its manifest",
            source=report_path,
            path="immutable_fingerprint",
            correction="edit only the review decision fields",
        )
    return ReviewHandoff(
        manifest=manifest,
        rows=tuple(rows),
        sha256=hashlib.sha256(review_bytes).hexdigest(),
    )


def approved_review_rows(handoff: ReviewHandoff) -> tuple[LoadedReviewRow, ...]:
    """Return only schema-validated approvals for future Stage-1 acceptance."""

    return tuple(
        loaded for loaded in handoff.rows if loaded.row.review.decision == "approved"
    )


def load_review_inputs(config_path: str | Path) -> ReviewInputs:
    """Load all immutable, verified inputs needed to generate a review report."""

    config = load_review_config(config_path)
    sanitized_path, candidates_path, registry_path, _, _ = resolve_review_paths(
        config_path, config
    )
    sanitized = load_sanitized_handoff(sanitized_path)
    registry = load_route_registry(registry_path)
    candidates = load_labeling_handoff(
        candidates_path, sanitized=sanitized, registry=registry
    )
    return ReviewInputs(
        config=config,
        sanitized=sanitized,
        candidates=candidates,
        registry=registry,
    )


def load_acceptance_inputs(config_path: str | Path) -> AcceptanceInputs:
    """Load only approved candidates after verifying every immutable input binding."""

    config = load_acceptance_config(config_path)
    (
        sanitized_path,
        candidates_path,
        registry_path,
        review_path,
        review_manifest_path,
        _,
        _,
    ) = resolve_acceptance_paths(config_path, config)
    sanitized = load_sanitized_handoff(sanitized_path)
    registry = load_route_registry(registry_path)
    candidates = load_labeling_handoff(
        candidates_path, sanitized=sanitized, registry=registry
    )
    review = load_review_handoff(
        review_path,
        manifest=load_review_manifest(review_manifest_path),
        manifest_path=review_manifest_path,
        sanitized=sanitized,
        candidates=candidates,
        registry=registry,
    )
    return AcceptanceInputs(
        config=config,
        sanitized=sanitized,
        candidates=candidates,
        registry=registry,
        review=review,
        approved=approved_review_rows(review),
    )


def load_examples(path: str | Path, registry: RouteRegistry) -> list[Example]:
    """Load every validated JSONL example into a list for existing callers."""

    examples: list[Example] = []
    ids: dict[str, int] = {}
    for loaded in iter_examples(path, registry):
        example = loaded.example
        if example.id is not None:
            first_line = ids.get(example.id)
            if first_line is not None:
                raise ExampleLoadError(
                    f"duplicate example id {example.id!r}; first declared on line {first_line}",
                    source=loaded.source,
                    line=loaded.line,
                    path="id",
                    correction="assign a unique id",
                )
            ids[example.id] = loaded.line
        examples.append(example)
    return examples


def _load_raw_json_object(
    source: Path, line_number: int, raw_line: bytes
) -> dict[str, Any]:
    try:
        line = raw_line.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RawInputLoadError(
            f"could not decode UTF-8: {error}",
            source=source,
            line=line_number,
            path="$",
            correction="replace this line with valid UTF-8 JSON",
        ) from error
    if not line.strip():
        raise RawInputLoadError(
            "expected a JSON object, got a blank line",
            source=source,
            line=line_number,
            path="$",
            correction="remove blank lines; each line must be a JSON object",
        )
    try:
        document = json.loads(line, parse_constant=_reject_nonstandard_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise RawInputLoadError(
            f"malformed JSON: {error}",
            source=source,
            line=line_number,
            path="$",
            correction="replace this line with a valid JSON object",
        ) from error
    if not isinstance(document, dict):
        raise RawInputLoadError(
            "expected a JSON object",
            source=source,
            line=line_number,
            path="$",
            correction="replace this line with a JSON object",
        )
    return document


def _load_sanitized_manifest(source: Path) -> dict[str, Any]:
    """Load a JSON object without exposing handoff contents in diagnostics."""

    try:
        raw_document = source.read_bytes()
    except OSError as error:
        raise SanitizedArtifactLoadError(
            "could not read sanitized handoff manifest",
            source=source,
            correction="ensure manifest.json exists in the Stage-7 handoff directory",
        ) from error
    try:
        text = raw_document.decode("utf-8")
        document = json.loads(text, parse_constant=_reject_nonstandard_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SanitizedArtifactLoadError(
            "sanitized handoff manifest must be valid UTF-8 JSON",
            source=source,
            correction="use the manifest.json emitted by Stage 7",
        ) from error
    if not isinstance(document, dict):
        raise SanitizedArtifactLoadError(
            "sanitized handoff manifest must be a JSON object",
            source=source,
            correction="use the manifest.json emitted by Stage 7",
        )
    return document


def _load_sanitized_json_object(
    source: Path, line_number: int, raw_line: bytes
) -> dict[str, Any]:
    """Parse one sanitized row without reflecting untrusted input content."""

    try:
        text = raw_line.decode("utf-8")
        document = json.loads(text, parse_constant=_reject_nonstandard_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SanitizedArtifactLoadError(
            "sanitized row must be valid UTF-8 JSON",
            source=source,
            line=line_number,
            path="$",
            correction="use the unmodified rows.jsonl emitted by Stage 7",
        ) from error
    if not isinstance(document, dict):
        raise SanitizedArtifactLoadError(
            "sanitized row must be a JSON object",
            source=source,
            line=line_number,
            path="$",
            correction="use the unmodified rows.jsonl emitted by Stage 7",
        )
    return document


def _load_artifact_manifest(
    source: Path, error_type: type[SourceError], document_name: str
) -> dict[str, Any]:
    """Load an artifact manifest without ever reflecting its contents."""

    raw_document = _read_artifact_bytes(source, error_type, document_name)
    try:
        document = json.loads(
            raw_document.decode("utf-8"),
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise error_type(
            f"{document_name} must be valid UTF-8 JSON",
            source=source,
            correction="use the unmodified artifact manifest",
        ) from error
    if not isinstance(document, dict):
        raise error_type(
            f"{document_name} must be a JSON object",
            source=source,
            correction="use the unmodified artifact manifest",
        )
    return document


def _load_artifact_json_object(
    source: Path,
    line_number: int,
    raw_line: bytes,
    error_type: type[SourceError],
    document_name: str,
) -> dict[str, Any]:
    """Parse untrusted artifact JSON without placing its content in diagnostics."""

    try:
        document = json.loads(
            raw_line.decode("utf-8"),
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise error_type(
            f"{document_name} must be valid UTF-8 JSON",
            source=source,
            line=line_number,
            path="$",
            correction="use the unmodified JSONL artifact",
        ) from error
    if not isinstance(document, dict):
        raise error_type(
            f"{document_name} must be a JSON object",
            source=source,
            line=line_number,
            path="$",
            correction="use the unmodified JSONL artifact",
        )
    return document


def _read_artifact_bytes(
    source: Path, error_type: type[SourceError], document_name: str
) -> bytes:
    try:
        return source.read_bytes()
    except OSError as error:
        raise error_type(
            f"could not read {document_name}",
            source=source,
            correction="ensure the artifact file exists and is readable",
        ) from error


def _verify_artifact_lf_and_count(
    contents: bytes,
    source: Path,
    expected_rows: int,
    error_type: type[SourceError],
    document_name: str,
) -> None:
    if contents and not contents.endswith(b"\n"):
        raise error_type(
            f"{document_name} must end every JSON object with an LF",
            source=source,
            correction="use the emitted JSONL artifact",
        )
    if contents.count(b"\n") != expected_rows:
        raise error_type(
            f"{document_name} row count does not match its manifest",
            source=source,
            path="rows",
            correction="use the complete emitted JSONL artifact",
        )


def _read_verified_artifact(
    source: Path,
    expected_rows: int,
    expected_sha256: str,
    error_type: type[SourceError],
    document_name: str,
) -> bytes:
    contents = _read_artifact_bytes(source, error_type, document_name)
    if hashlib.sha256(contents).hexdigest() != expected_sha256:
        raise error_type(
            f"{document_name} SHA-256 does not match its manifest",
            source=source,
            path="sha256",
            correction="use the unmodified emitted artifact",
        )
    _verify_artifact_lf_and_count(
        contents, source, expected_rows, error_type, document_name
    )
    return contents


def _file_sha256(
    source: Path, error_type: type[SourceError], document_name: str
) -> str:
    return hashlib.sha256(
        _read_artifact_bytes(source, error_type, document_name)
    ).hexdigest()


def _canonical_json_sha256(document: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _candidate_validation(
    candidate: LabelCandidate, registry: RouteRegistry
) -> CandidateValidation:
    if candidate.status == "rejected":
        return CandidateValidation(valid=False, reason="provider_rejected")
    assert candidate.decision is not None
    try:
        validate_decision(candidate.decision, registry)
    except DecisionValidationError as error:
        return CandidateValidation(valid=False, reason=error.category)
    return CandidateValidation(valid=True)


def _verify_labeling_handoff_bindings(
    manifest: LabelingManifest,
    manifest_path: Path,
    *,
    sanitized: SanitizedHandoff,
    registry: RouteRegistry,
) -> None:
    if (
        manifest.input.rows != sanitized.manifest.output.rows
        or manifest.input.sha256 != sanitized.manifest.output.sha256
        or manifest.input_manifest_fingerprint
        != _file_sha256(
            sanitized.directory / "manifest.json",
            CandidateArtifactLoadError,
            "sanitized handoff manifest",
        )
        or manifest.registry_fingerprint
        != _canonical_json_sha256(registry.model_dump(mode="json"))
    ):
        raise CandidateArtifactLoadError(
            "candidate manifest is not bound to the supplied sanitized handoff and registry",
            source=manifest_path,
            correction="use artifacts from the same Stage-7 and Stage-8 run",
        )


def _verify_candidate_manifest_provenance(
    candidate: LabelCandidate,
    manifest: LabelingManifest,
    source: Path,
    line_number: int,
) -> None:
    provenance = candidate.provenance
    if (
        provenance.policy_fingerprint != manifest.policy_fingerprint
        or provenance.registry_fingerprint != manifest.registry_fingerprint
        or provenance.provider_model != manifest.provider_model
        or provenance.provider_endpoint != manifest.provider_endpoint
    ):
        raise CandidateArtifactLoadError(
            "candidate provenance does not match its artifact manifest",
            source=source,
            line=line_number,
            correction="use the unmodified Stage-8 artifact",
        )


def _verify_review_manifest_bindings(
    manifest: ReviewManifest,
    manifest_path: Path,
    *,
    sanitized: SanitizedHandoff,
    candidates: LabelingHandoff,
    registry: RouteRegistry,
) -> None:
    if (
        manifest.sanitized.rows != sanitized.manifest.output.rows
        or manifest.sanitized.sha256 != sanitized.manifest.output.sha256
        or manifest.sanitized_manifest_fingerprint
        != _file_sha256(
            sanitized.directory / "manifest.json",
            ReviewRecordLoadError,
            "sanitized handoff manifest",
        )
        or manifest.candidates.rows != candidates.manifest.output.rows
        or manifest.candidates.sha256 != candidates.manifest.output.sha256
        or manifest.candidate_manifest_fingerprint
        != _file_sha256(
            candidates.directory / "manifest.json",
            ReviewRecordLoadError,
            "candidate artifact manifest",
        )
        or manifest.registry_fingerprint
        != _canonical_json_sha256(registry.model_dump(mode="json"))
        or manifest.policy_fingerprint != candidates.manifest.policy_fingerprint
        or manifest.provider_model != candidates.manifest.provider_model
        or manifest.provider_endpoint != candidates.manifest.provider_endpoint
    ):
        raise ReviewRecordLoadError(
            "review manifest is not bound to the supplied candidate, handoff, and registry artifacts",
            source=manifest_path,
            correction="use review inputs from one verified labeling run",
        )


def _immutable_review_row_bytes(row: ReviewRow) -> bytes:
    document = row.model_dump(mode="json")
    document.pop("review")
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _project_raw_input(
    document: dict[str, Any],
    config: RawIngestionConfig,
    source: Path,
    line_number: int,
) -> dict[str, Any]:
    metadata = {
        name: _resolve_raw_pointer(
            document,
            pointer,
            source,
            line_number,
            path=f"projection.metadata.{name}",
        )
        for name, pointer in config.projection.metadata.items()
    }
    return {
        "id": _resolve_raw_pointer(
            document,
            config.projection.id,
            source,
            line_number,
            path="projection.id",
        ),
        "input": _resolve_raw_pointer(
            document,
            config.projection.input,
            source,
            line_number,
            path="projection.input",
        ),
        "metadata": {
            "_equiroute": {"source_line": line_number},
            **metadata,
        },
    }


def _resolve_raw_pointer(
    document: dict[str, Any],
    pointer: str,
    source: Path,
    line_number: int,
    *,
    path: str,
) -> Any:
    value: Any = document
    for token in _json_pointer_tokens(pointer):
        if not isinstance(value, Mapping):
            raise RawInputLoadError(
                "JSON Pointer traversal requires mappings and cannot traverse arrays",
                source=source,
                line=line_number,
                path=path,
                correction="change the projection to traverse object fields only",
            )
        if token not in value:
            raise RawInputLoadError(
                "JSON Pointer does not resolve in this source row",
                source=source,
                line=line_number,
                path=path,
                correction="change the projection or provide the required object field",
            )
        value = value[token]
    return value


def _apply_raw_redactions(
    projected: dict[str, Any],
    config: RawIngestionConfig,
    source: Path,
    line_number: int,
) -> None:
    metadata = projected["metadata"]
    assert isinstance(metadata, dict)
    for rule in config.redactions:
        metadata_name = rule.metadata_name
        if metadata_name is None:
            target = projected
            field = "input"
            path = "input"
        else:
            target = metadata
            field = metadata_name
            path = f"metadata.{metadata_name}"
        value = target[field]
        if not isinstance(value, str):
            raise RawInputLoadError(
                "redaction target must resolve to a string",
                source=source,
                line=line_number,
                path=path,
                correction="change the projection or target a string metadata field",
            )
        target[field] = rule.apply(value)


def _validate_raw_input_row(
    projected: dict[str, Any], source: Path, line_number: int
) -> RawInputRow:
    try:
        return RawInputRow.model_validate(projected)
    except ValidationError as error:
        issues = error.errors()
        details = "; ".join(
            f"{_format_location(issue['loc'])}: {issue['msg']}" for issue in issues
        )
        raise RawInputLoadError(
            f"invalid raw input: {details}",
            source=source,
            line=line_number,
            path=_format_location(issues[0]["loc"]),
            correction=_validation_correction(error),
        ) from error


def _validate_projected_raw_input_utf8(
    projected: dict[str, Any], source: Path, line_number: int
) -> None:
    """Reject canonical values that JSON can render but UTF-8 cannot serialize."""

    for field in ("id", "input"):
        value = projected[field]
        if isinstance(value, str):
            _validate_utf8_string(value, source, line_number, path=field)
    _validate_metadata_utf8(projected["metadata"], source, line_number, path="metadata")


def _validate_metadata_utf8(
    value: Any, source: Path, line_number: int, *, path: str
) -> None:
    """Validate recursive metadata strings without exposing source values."""

    if isinstance(value, str):
        _validate_utf8_string(value, source, line_number, path=path)
    elif isinstance(value, Mapping):
        for key, nested_value in value.items():
            if not isinstance(key, str):
                continue
            _validate_utf8_string(key, source, line_number, path=path)
            _validate_metadata_utf8(
                nested_value,
                source,
                line_number,
                path=f"{path}.{key}",
            )
    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            _validate_metadata_utf8(
                nested_value,
                source,
                line_number,
                path=f"{path}[{index}]",
            )


def _validate_utf8_string(
    value: str, source: Path, line_number: int, *, path: str
) -> None:
    """Turn Unicode encoding failures into source-aware, redacted input errors."""

    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise RawInputLoadError(
            "canonical value cannot be encoded as UTF-8",
            source=source,
            line=line_number,
            path=path,
            correction="replace invalid Unicode with UTF-8 text",
        ) from error


def _validate_raw_input_size(
    raw_input: RawInputRow,
    config: RawIngestionConfig,
    source: Path,
    line_number: int,
) -> None:
    try:
        input_size = len(raw_input.input.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise RawInputLoadError(
            "input cannot be encoded as UTF-8",
            source=source,
            line=line_number,
            path="input",
            correction="replace invalid Unicode with UTF-8 text",
        ) from error
    if input_size > config.limits.max_input_bytes:
        raise RawInputLoadError(
            f"input exceeds the {config.limits.max_input_bytes}-byte limit after redaction",
            source=source,
            line=line_number,
            path="input",
            correction="shorten the input or add a redaction rule",
        )


def _load_yaml_mapping(
    source: Path, error_type: type[SourceError], document_name: str
) -> dict[str, Any]:
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise error_type(
            f"could not read {document_name}: {error}",
            source=source,
            correction="ensure the file exists and is valid UTF-8",
        ) from error

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        line = _yaml_error_line(error)
        raise error_type(
            f"malformed YAML: {getattr(error, 'problem', None) or error}",
            source=source,
            line=line,
            correction="repair the YAML syntax",
        ) from error

    if not isinstance(document, dict):
        raise error_type(
            f"expected a YAML mapping for {document_name}",
            source=source,
            correction="replace the document root with a YAML mapping",
        )
    return document


def _migrate_training_config(document: dict[str, Any], source: Path) -> dict[str, Any]:
    try:
        migrated = migrate_training_config(document)
    except SchemaMigrationError as error:
        raise ConfigLoadError(
            str(error),
            source=source,
            path="schema_version",
            correction='use supported schema_version "2" for new configurations',
        ) from error
    assert isinstance(migrated, dict)
    return migrated


def _migrate_raw_ingestion_config(
    document: dict[str, Any], source: Path
) -> dict[str, Any]:
    try:
        migrated = migrate_raw_ingestion_config(document)
    except SchemaMigrationError as error:
        raise RawIngestionConfigError(
            str(error),
            source=source,
            path="schema_version",
            correction='set schema_version to "2"',
        ) from error
    assert isinstance(migrated, dict)
    return migrated


def _migrate_labeling_config(document: dict[str, Any], source: Path) -> dict[str, Any]:
    try:
        migrated = migrate_labeling_config(document)
    except SchemaMigrationError as error:
        raise LabelingConfigError(
            str(error),
            source=source,
            path="schema_version",
            correction='set schema_version to "2"',
        ) from error
    assert isinstance(migrated, dict)
    return migrated


def _migrate_stage9_config(
    document: dict[str, Any],
    source: Path,
    migration: Any,
    error_type: type[SourceError],
    document_name: str,
) -> dict[str, Any]:
    try:
        migrated = migration(document)
    except SchemaMigrationError as error:
        raise error_type(
            str(error),
            source=source,
            path="schema_version",
            correction='set schema_version to "2"',
        ) from error
    assert isinstance(migrated, dict)
    return migrated


def _validate_raw_ingestion_config(
    document: dict[str, Any], source: Path
) -> RawIngestionConfig:
    try:
        return RawIngestionConfig.model_validate(document)
    except ValidationError as error:
        issues = error.errors()
        details = "; ".join(
            f"{_format_location(issue['loc'])}: {issue['msg']}" for issue in issues
        )
        raise RawIngestionConfigError(
            f"invalid raw ingestion configuration: {details}",
            source=source,
            path=_format_location(issues[0]["loc"]),
            correction=_validation_correction(error),
        ) from error


def _validate_labeling_config(document: dict[str, Any], source: Path) -> LabelingConfig:
    try:
        return LabelingConfig.model_validate(document)
    except ValidationError as error:
        issues = error.errors()
        details = "; ".join(
            f"{_format_location(issue['loc'])}: {issue['msg']}" for issue in issues
        )
        raise LabelingConfigError(
            f"invalid labeling configuration: {details}",
            source=source,
            path=_format_location(issues[0]["loc"]),
            correction=_validation_correction(error),
        ) from error


def _validate_model(
    document: dict[str, Any],
    model_type: type[_Model],
    source: Path,
    error_type: type[SourceError],
    document_name: str,
    line: int | None = None,
) -> _Model:
    try:
        return model_type.model_validate(document)
    except ValidationError as error:
        details = "; ".join(
            f"{_format_location(issue['loc'])}: {issue['msg']}"
            for issue in error.errors()
        )
        raise error_type(
            f"invalid {document_name}: {details}",
            source=source,
            line=line,
            correction=_validation_correction(error),
        ) from error


def _validate_decision(
    decision: Decision, registry: RouteRegistry, source: Path, line: int
) -> None:
    try:
        validate_decision(decision, registry)
    except DecisionValidationError as error:
        if error.category == "unknown_route":
            route_names = ", ".join(route.name for route in registry.routes)
            raise ExampleLoadError(
                error.detail,
                source=source,
                line=line,
                path="route.name",
                correction=f"choose one of: {route_names}",
            ) from error

        route = registry.route_named(decision.name)
        if route is None:
            raise AssertionError(
                "invalid arguments reported for an unknown route"
            ) from error
        raise ExampleLoadError(
            error.detail,
            source=source,
            line=line,
            path="route.arguments",
            correction=_argument_correction(decision.arguments, route),
        ) from error


def _validation_correction(error: ValidationError) -> str | None:
    issues = error.errors()
    types = {issue["type"] for issue in issues}
    locations = [_format_location(issue["loc"]) for issue in issues]

    if types == {"missing"}:
        return (
            "add required field"
            + ("s" if len(locations) > 1 else "")
            + ": "
            + ", ".join(locations)
        )
    if types == {"extra_forbidden"}:
        return (
            "remove unsupported field"
            + ("s" if len(locations) > 1 else "")
            + ": "
            + ", ".join(locations)
        )
    if types <= {"string_too_short"}:
        return "provide a non-empty string"
    if types <= {"string_type"}:
        return "provide a string"
    if types <= {"dict_type", "model_type"}:
        return "provide an object"
    if types <= {"list_type"}:
        return "provide a list"
    return (
        "correct invalid field"
        + ("s" if len(locations) > 1 else "")
        + ": "
        + ", ".join(locations)
    )


def _format_location(location: tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in location) or "root"


def _reject_nonstandard_json_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant {token!r}")


def _yaml_error_line(error: yaml.YAMLError) -> int | None:
    mark = getattr(error, "problem_mark", None)
    return mark.line + 1 if mark is not None else None
