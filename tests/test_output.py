from __future__ import annotations

import json
from pathlib import Path

import pytest

from equiroute.io import load_route_registry
from equiroute.output import (
    InvalidOutput,
    parse_functiongemma_completion,
    render_canonical_json,
    render_openai_tool_call,
)
from equiroute.schemas import Decision


FIXTURES = Path(__file__).parent / "fixtures"


def _registry():
    return load_route_registry(FIXTURES / "stage2-routes.yaml")


def test_parses_native_completion_despite_member_order_and_harmless_whitespace():
    raw = """\
      <start_function_call>
      call:submit_ticket { subject: <escape>Cannot sign in<escape>,
      enabled: true, ratio: 7.5e-1, priority: 2 }
      <end_function_call><start_function_response>
    """

    parsed = parse_functiongemma_completion(raw, _registry())

    assert parsed == Decision(
        name="submit_ticket",
        arguments={
            "subject": "Cannot sign in",
            "priority": 2,
            "ratio": 0.75,
            "enabled": True,
        },
    )


@pytest.mark.parametrize(
    ("raw", "category"),
    [
        ("model response", "missing_function_call"),
        ("<start_function_call>call:submit_ticket{}", "malformed_function_call"),
        (
            '<start_function_call>call:submit_ticket{subject:"Cannot sign in"}'
            "<end_function_call>",
            "invalid_argument_syntax",
        ),
        (
            "<start_function_call>call:not_registered{}<end_function_call>",
            "unknown_route",
        ),
        (
            "<start_function_call>call:submit_ticket{priority:2,ratio:0.75,enabled:true}"
            "<end_function_call>",
            "invalid_arguments",
        ),
    ],
)
def test_rejects_invalid_native_completions_with_designated_categories(raw, category):
    parsed = parse_functiongemma_completion(raw, _registry())

    assert isinstance(parsed, InvalidOutput)
    assert parsed.raw == raw
    assert parsed.category == category
    assert parsed.detail


def test_reference_renderers_preserve_decision_semantics():
    decision = Decision(
        name="submit_ticket",
        arguments={
            "subject": "Cannot sign in",
            "priority": 2,
            "ratio": 0.75,
            "enabled": True,
        },
    )

    canonical = json.loads(render_canonical_json(decision))
    openai = json.loads(render_openai_tool_call(decision))

    assert canonical == decision.model_dump(mode="json")
    assert len(openai["tool_calls"]) == 1
    tool_call = openai["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == decision.name
    assert (
        json.loads(openai["tool_calls"][0]["function"]["arguments"])
        == decision.arguments
    )
