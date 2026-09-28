"""Load and validate EquiRoute YAML and JSONL contracts from local files."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from .decisions import DecisionValidationError, _argument_correction, validate_decision
from .errors import ConfigLoadError, ExampleLoadError, RegistryLoadError, SourceError
from .migrations import SchemaMigrationError, migrate_training_config

from .schemas import Decision, Example, RouteRegistry, TrainingConfig

_Model = TypeVar("_Model", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class LoadedExample:
    """A validated example with its source location."""

    example: Example
    source: Path
    line: int


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
