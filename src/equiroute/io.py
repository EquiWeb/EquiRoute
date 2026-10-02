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
    ConfigLoadError,
    ExampleLoadError,
    LabelingConfigError,
    RawIngestionConfigError,
    RawInputLoadError,
    RegistryLoadError,
    SanitizedArtifactLoadError,
    SourceError,
)
from .migrations import (
    SchemaMigrationError,
    migrate_labeling_config,
    migrate_raw_ingestion_config,
    migrate_raw_ingestion_manifest,
    migrate_training_config,
)

from .schemas import (
    Decision,
    Example,
    LabelingConfig,
    RawIngestionConfig,
    RawIngestionManifest,
    RawInputRow,
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
