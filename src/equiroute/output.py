"""FunctionGemma completion parsing and reference output renderers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Literal

from .decisions import DecisionValidationError, validate_decision
from .schemas import Decision, RouteRegistry


_START_CALL = "<start_function_call>"
_END_CALL = "<end_function_call>"
_START_RESPONSE = "<start_function_response>"
_ESCAPE = "<escape>"
_JSON_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


@dataclass(frozen=True, slots=True)
class InvalidOutput:
    """One generated completion that cannot be used as a validated decision."""

    raw: str
    category: Literal[
        "missing_function_call",
        "malformed_function_call",
        "invalid_argument_syntax",
        "unknown_route",
        "invalid_arguments",
    ]
    detail: str


def parse_functiongemma_completion(
    raw: str, registry: RouteRegistry
) -> Decision | InvalidOutput:
    """Parse one complete, flat native FunctionGemma function call.

    The parser accepts only the native ``call:name{key:value}`` envelope.  It
    deliberately does not accept JSON, prose, nested values, or multiple calls.
    """

    call, error = _extract_call(raw)
    if error is not None:
        return error
    assert call is not None

    name, arguments, error = _parse_call(call, raw)
    if error is not None:
        return error
    assert name is not None and arguments is not None

    decision = Decision(name=name, arguments=arguments)
    try:
        validate_decision(decision, registry)
    except DecisionValidationError as validation_error:
        return InvalidOutput(
            raw=raw,
            category=validation_error.category,
            detail=validation_error.detail,
        )
    return decision


def render_canonical_json(decision: Decision) -> str:
    """Render a decision in the compact canonical JSON reference shape."""

    return _compact_json({"name": decision.name, "arguments": decision.arguments})


def render_openai_tool_call(decision: Decision) -> str:
    """Render a decision as a compact OpenAI-compatible ``tool_calls`` object."""

    arguments = _compact_json(decision.arguments)
    return _compact_json(
        {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": decision.name, "arguments": arguments},
                }
            ]
        }
    )


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _extract_call(raw: str) -> tuple[str | None, InvalidOutput | None]:
    text = raw.strip()
    if not text:
        return None, _invalid(
            raw, "missing_function_call", "no function call was generated"
        )

    if text.endswith(_START_RESPONSE):
        text = text[: -len(_START_RESPONSE)].rstrip()

    if _START_CALL not in text and _END_CALL not in text:
        return None, _invalid(
            raw,
            "missing_function_call",
            "no native function call envelope was generated",
        )
    if not text.startswith(_START_CALL) or not text.endswith(_END_CALL):
        return None, _invalid(
            raw,
            "malformed_function_call",
            "expected exactly one complete native function call envelope",
        )

    body = text[len(_START_CALL) : -len(_END_CALL)]
    if _START_CALL in body or _END_CALL in body:
        return None, _invalid(
            raw,
            "malformed_function_call",
            "multiple function call markers are not allowed",
        )
    return body, None


def _parse_call(
    call: str, raw: str
) -> tuple[str | None, dict[str, object] | None, InvalidOutput | None]:
    position = _skip_whitespace(call, 0)
    if not call.startswith("call:", position):
        return (
            None,
            None,
            _invalid(
                raw, "malformed_function_call", "native call must begin with 'call:'"
            ),
        )
    position += len("call:")

    name_start = position
    while position < len(call) and call[position] not in "{\t\n\r ":
        if call[position] in "},:":
            return (
                None,
                None,
                _invalid(
                    raw,
                    "malformed_function_call",
                    "function name contains a native delimiter",
                ),
            )
        position += 1
    name = call[name_start:position]
    if not name:
        return (
            None,
            None,
            _invalid(raw, "malformed_function_call", "function name must not be empty"),
        )

    position = _skip_whitespace(call, position)
    if position == len(call) or call[position] != "{":
        return (
            None,
            None,
            _invalid(
                raw,
                "malformed_function_call",
                "function name must be followed by an argument object",
            ),
        )
    position += 1

    arguments: dict[str, object] = {}
    position = _skip_whitespace(call, position)
    if position < len(call) and call[position] == "}":
        position += 1
    else:
        while True:
            key, position, error = _parse_key(call, position, raw)
            if error is not None:
                return None, None, error
            assert key is not None
            if key in arguments:
                return (
                    None,
                    None,
                    _invalid(
                        raw, "malformed_function_call", f"duplicate argument {key!r}"
                    ),
                )

            position = _skip_whitespace(call, position)
            if position == len(call) or call[position] != ":":
                return (
                    None,
                    None,
                    _invalid(
                        raw,
                        "malformed_function_call",
                        f"argument {key!r} must be followed by ':'",
                    ),
                )
            position = _skip_whitespace(call, position + 1)

            value, position, error = _parse_primitive(call, position, raw)
            if error is not None:
                return None, None, error
            arguments[key] = value

            position = _skip_whitespace(call, position)
            if position == len(call):
                return (
                    None,
                    None,
                    _invalid(
                        raw, "malformed_function_call", "argument object is not closed"
                    ),
                )
            if call[position] == "}":
                position += 1
                break
            if call[position] != ",":
                return (
                    None,
                    None,
                    _invalid(
                        raw,
                        "malformed_function_call",
                        "arguments must be separated by commas",
                    ),
                )
            position = _skip_whitespace(call, position + 1)
            if position == len(call) or call[position] == "}":
                return (
                    None,
                    None,
                    _invalid(
                        raw, "malformed_function_call", "trailing comma in arguments"
                    ),
                )

    if call[position:].strip():
        return (
            None,
            None,
            _invalid(
                raw,
                "malformed_function_call",
                "unexpected content after argument object",
            ),
        )
    return name, arguments, None


def _parse_key(
    call: str, position: int, raw: str
) -> tuple[str | None, int, InvalidOutput | None]:
    if position == len(call) or call[position] in "{}[],:":
        return (
            None,
            position,
            _invalid(raw, "malformed_function_call", "argument name must not be empty"),
        )

    start = position
    while position < len(call) and call[position] not in ":\t\n\r ":
        if call[position] in "{}[],":
            return (
                None,
                position,
                _invalid(
                    raw,
                    "malformed_function_call",
                    "argument name contains a native delimiter",
                ),
            )
        position += 1
    key = call[start:position]
    if not key:
        return (
            None,
            position,
            _invalid(raw, "malformed_function_call", "argument name must not be empty"),
        )
    return key, position, None


def _parse_primitive(
    call: str, position: int, raw: str
) -> tuple[object | None, int, InvalidOutput | None]:
    if position == len(call):
        return (
            None,
            position,
            _invalid(raw, "invalid_argument_syntax", "argument value is missing"),
        )

    if call.startswith(_ESCAPE, position):
        start = position + len(_ESCAPE)
        end = call.find(_ESCAPE, start)
        if end < 0:
            return (
                None,
                position,
                _invalid(raw, "invalid_argument_syntax", "unterminated escaped string"),
            )
        escaped_value = call[start:end]
        position = end + len(_ESCAPE)
        if call.startswith(_ESCAPE, position):
            return (
                None,
                position,
                _invalid(
                    raw,
                    "invalid_argument_syntax",
                    "nested escape delimiters are not supported",
                ),
            )
        return escaped_value, position, None

    if call[position] in "{[\"'":
        return (
            None,
            position,
            _invalid(
                raw,
                "invalid_argument_syntax",
                "arguments must use flat native primitive literals",
            ),
        )

    end = position
    while end < len(call) and call[end] not in ",}\t\n\r ":
        end += 1
    token = call[position:end]
    if token == "true":
        return True, end, None
    if token == "false":
        return False, end, None
    if _JSON_NUMBER.fullmatch(token):
        number: int | float = (
            int(token) if all(marker not in token for marker in ".eE") else float(token)
        )
        if math.isfinite(number):
            return number, end, None

    return (
        None,
        position,
        _invalid(raw, "invalid_argument_syntax", f"invalid native literal {token!r}"),
    )


def _skip_whitespace(value: str, position: int) -> int:
    while position < len(value) and value[position].isspace():
        position += 1
    return position


def _invalid(
    raw: str,
    category: Literal[
        "missing_function_call", "malformed_function_call", "invalid_argument_syntax"
    ],
    detail: str,
) -> InvalidOutput:
    return InvalidOutput(raw=raw, category=category, detail=detail)
