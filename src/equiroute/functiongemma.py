"""Pure-Python compiler for the pinned FunctionGemma native wire format."""

from __future__ import annotations

import math
import re
from typing import Any

from .decisions import validate_decision
from .schemas import Example, ObjectArgumentSchema, Route, RouteRegistry

_BOS = "<bos>"
_START_OF_TURN = "<start_of_turn>"
_END_OF_TURN = "<end_of_turn>"
_START_FUNCTION_DECLARATION = "<start_function_declaration>"
_END_FUNCTION_DECLARATION = "<end_function_declaration>"
_START_FUNCTION_CALL = "<start_function_call>"
_END_FUNCTION_CALL = "<end_function_call>"
_START_FUNCTION_RESPONSE = "<start_function_response>"
_ESCAPE = "<escape>"

_CONTROL_TOKENS = (
    _BOS,
    _START_OF_TURN,
    _END_OF_TURN,
    _START_FUNCTION_DECLARATION,
    _END_FUNCTION_DECLARATION,
    _START_FUNCTION_CALL,
    _END_FUNCTION_CALL,
    _START_FUNCTION_RESPONSE,
    "<end_function_response>",
    _ESCAPE,
)
_BARE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")


def render_functiongemma_prompt(input: str, registry: RouteRegistry) -> str:
    """Render the FunctionGemma inference prompt without a function call."""
    _require_safe_text(input, "example input")
    declarations = "".join(_compile_declaration(route) for route in registry.routes)
    return (
        f"{_BOS}{_START_OF_TURN}developer\n"
        f"{declarations}{_END_OF_TURN}\n"
        f"{_START_OF_TURN}user\n{input.strip()}{_END_OF_TURN}\n"
        f"{_START_OF_TURN}model\n"
    )


def compile_functiongemma(example: Example, registry: RouteRegistry) -> str:
    """Compile one validated example using FunctionGemma's pinned native template.

    This is a deliberately narrow equivalent of the pinned tokenizer chat
    template. It emits only EquiRoute's flat primitive argument schema and
    refuses raw values that would make the native literal syntax ambiguous.
    """
    validate_decision(example.route, registry)
    return (
        render_functiongemma_prompt(example.input, registry)
        + _compile_call(example.route.name, example.route.arguments)
        + _START_FUNCTION_RESPONSE
    )


def _compile_declaration(route: Route) -> str:
    _require_bare_name(route.name, "route name")
    _require_safe_text(route.description, "route description")
    parameters = _compile_parameters(route.parameters)
    return (
        f"{_START_FUNCTION_DECLARATION}declaration:{route.name}"
        f"{{description:{_ESCAPE}{route.description}{_ESCAPE},parameters:{parameters}}}"
        f"{_END_FUNCTION_DECLARATION}"
    )


def _compile_parameters(schema: ObjectArgumentSchema) -> str:
    properties = ""
    if schema.properties:
        members = ",".join(
            _compile_property(name, property_schema.type)
            for name, property_schema in sorted(schema.properties.items())
        )
        properties = f"properties:{{{members}}},"

    required = ""
    if schema.required:
        required = "required:[" + ",".join(
            f"{_ESCAPE}{name}{_ESCAPE}" for name in schema.required
        ) + "],"

    return f"{{{properties}{required}type:{_ESCAPE}OBJECT{_ESCAPE}}}"


def _compile_property(name: str, primitive_type: str) -> str:
    _require_bare_name(name, "argument name")
    return (
        f"{name}:{{description:{_ESCAPE}{_ESCAPE},"
        f"type:{_ESCAPE}{primitive_type.upper()}{_ESCAPE}}}"
    )


def _compile_call(name: str, arguments: dict[str, Any]) -> str:
    _require_bare_name(name, "route name")
    members = ",".join(
        f"{_safe_call_key(key)}:{_compile_literal(value)}"
        for key, value in sorted(arguments.items())
    )
    return f"{_START_FUNCTION_CALL}call:{name}{{{members}}}{_END_FUNCTION_CALL}"


def _safe_call_key(key: str) -> str:
    _require_bare_name(key, "argument name")
    return key


def _compile_literal(value: Any) -> str:
    if isinstance(value, str):
        _require_safe_text(value, "string argument")
        return f"{_ESCAPE}{value}{_ESCAPE}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("number argument must be finite")
        return str(value)
    raise ValueError(f"unsupported native literal value {value!r}")


def _require_bare_name(value: str, field: str) -> None:
    if not _BARE_NAME.fullmatch(value):
        raise ValueError(
            f"{field} {value!r} cannot be emitted unambiguously in FunctionGemma native syntax"
        )


def _require_safe_text(value: str, field: str) -> None:
    token = next((token for token in _CONTROL_TOKENS if token in value), None)
    if token is not None:
        raise ValueError(
            f"{field} contains native delimiter {token!r} and cannot be emitted unambiguously"
        )
