"""Semantic validation for decisions against a route registry."""

from __future__ import annotations

import math
from typing import Any, Literal

from .schemas import Decision, ObjectArgumentSchema, Route, RouteRegistry


class DecisionValidationError(ValueError):
    """A decision that does not select or satisfy a registered route."""

    def __init__(
        self,
        category: Literal["unknown_route", "invalid_arguments"],
        detail: str,
    ) -> None:
        super().__init__(detail)
        self.category = category
        self.detail = detail


def validate_decision(decision: Decision, registry: RouteRegistry) -> None:
    """Require a decision to name a route and satisfy its argument schema."""

    route = registry.route_named(decision.name)
    if route is None:
        raise DecisionValidationError(
            "unknown_route", f"unknown route {decision.name!r}"
        )

    argument_errors = _argument_errors(decision.arguments, route)
    if argument_errors:
        raise DecisionValidationError("invalid_arguments", "; ".join(argument_errors))


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
        if property_schema is not None and not _matches_primitive(
            value, property_schema.type
        ):
            errors.append(
                f"argument {name!r} must be {property_schema.type}, got {_json_type_name(value)}"
            )
    return errors


def _argument_correction(arguments: dict[str, Any], route: Route) -> str:
    schema: ObjectArgumentSchema = route.parameters
    corrections: list[str] = []

    missing = [name for name in schema.required if name not in arguments]
    if missing:
        corrections.append(
            "add required argument"
            + ("s" if len(missing) > 1 else "")
            + ": "
            + ", ".join(repr(name) for name in missing)
        )

    unknown = [name for name in arguments if name not in schema.properties]
    if unknown:
        corrections.append(
            "remove unsupported argument"
            + ("s" if len(unknown) > 1 else "")
            + ": "
            + ", ".join(repr(name) for name in unknown)
        )

    for name, value in arguments.items():
        property_schema = schema.properties.get(name)
        if property_schema is not None and not _matches_primitive(
            value, property_schema.type
        ):
            corrections.append(f"set argument {name!r} to a {property_schema.type}")
    return "; ".join(corrections)


def _matches_primitive(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return isinstance(value, float) and math.isfinite(value)
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
