"""Load and validate EquiRoute YAML and JSONL contracts from local files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from .errors import ConfigLoadError, ExampleLoadError, RegistryLoadError, SourceError
from .schemas import (
    Decision,
    Example,
    ObjectArgumentSchema,
    Route,
    RouteRegistry,
    TrainingConfig,
)

_Model = TypeVar("_Model", bound=BaseModel)


def load_route_registry(path: str | Path) -> RouteRegistry:
    """Load one strict route registry from a YAML document."""

    source = Path(path)
    document = _load_yaml_mapping(source, RegistryLoadError, "route registry")
    return _validate_model(document, RouteRegistry, source, RegistryLoadError, "route registry")


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load one strict training configuration from a YAML document."""

    source = Path(path)
    document = _load_yaml_mapping(source, ConfigLoadError, "training configuration")
    return _validate_model(
        document, TrainingConfig, source, ConfigLoadError, "training configuration"
    )


def load_examples(path: str | Path, registry: RouteRegistry) -> list[Example]:
    """Load and validate every JSON object in a JSONL examples file."""

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ExampleLoadError(f"could not read examples: {error}", source=source) from error

    examples: list[Example] = []
    ids: dict[str, int] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ExampleLoadError("expected a JSON object, got a blank line", source=source, line=line_number)
        try:
            document = json.loads(line, parse_constant=_reject_nonstandard_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ExampleLoadError(
                f"malformed JSON: {error}", source=source, line=line_number
            ) from error
        if not isinstance(document, dict):
            raise ExampleLoadError(
                "expected a JSON object", source=source, line=line_number
            )

        example = _validate_model(
            document, Example, source, ExampleLoadError, "example", line_number
        )
        _validate_decision(example.route, registry, source, line_number)

        if example.id is not None:
            first_line = ids.get(example.id)
            if first_line is not None:
                raise ExampleLoadError(
                    f"duplicate example id {example.id!r}; first declared on line {first_line}",
                    source=source,
                    line=line_number,
                    path="id",
                )
            ids[example.id] = line_number
        examples.append(example)
    return examples


def _load_yaml_mapping(
    source: Path, error_type: type[SourceError], document_name: str
) -> dict[str, Any]:
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise error_type(f"could not read {document_name}: {error}", source=source) from error

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        line = _yaml_error_line(error)
        raise error_type(
            f"malformed YAML: {getattr(error, 'problem', None) or error}",
            source=source,
            line=line,
        ) from error

    if not isinstance(document, dict):
        raise error_type(f"expected a YAML mapping for {document_name}", source=source)
    return document


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
            f"{_format_location(issue['loc'])}: {issue['msg']}" for issue in error.errors()
        )
        raise error_type(
            f"invalid {document_name}: {details}", source=source, line=line
        ) from error


def _validate_decision(
    decision: Decision, registry: RouteRegistry, source: Path, line: int
) -> None:
    route = registry.route_named(decision.name)
    if route is None:
        raise ExampleLoadError(
            f"unknown route {decision.name!r}",
            source=source,
            line=line,
            path="route.name",
        )

    argument_errors = _argument_errors(decision.arguments, route)
    if argument_errors:
        raise ExampleLoadError(
            "; ".join(argument_errors),
            source=source,
            line=line,
            path="route.arguments",
        )


def _argument_errors(arguments: dict[str, Any], route: Route) -> list[str]:
    schema: ObjectArgumentSchema = route.parameters
    errors: list[str] = []

    for name in schema.required:
        if name not in arguments:
            errors.append(f"missing required argument {name!r}")

    for name in arguments:
        if name not in schema.properties:
            errors.append(f"unknown argument {name!r}")

    for name, value in arguments.items():
        property_schema = schema.properties.get(name)
        if property_schema is not None and not _matches_primitive(value, property_schema.type):
            errors.append(
                f"argument {name!r} must be {property_schema.type}, got {_json_type_name(value)}"
            )
    return errors


def _matches_primitive(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    return isinstance(value, bool)


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    return "object"


def _format_location(location: tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in location) or "root"


def _reject_nonstandard_json_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant {token!r}")


def _yaml_error_line(error: yaml.YAMLError) -> int | None:
    mark = getattr(error, "problem_mark", None)
    return mark.line + 1 if mark is not None else None
