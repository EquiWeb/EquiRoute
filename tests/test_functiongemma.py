from __future__ import annotations

from pathlib import Path

import pytest

from equiroute.decisions import DecisionValidationError
from equiroute.functiongemma import compile_functiongemma, render_functiongemma_prompt
from equiroute.io import load_examples, load_route_registry
from equiroute.schemas import Decision, Example

FIXTURES = Path(__file__).parent / "fixtures"


def _registry_and_example():
    registry = load_route_registry(FIXTURES / "stage2-routes.yaml")
    example = load_examples(FIXTURES / "stage2-example.jsonl", registry)[0]
    return registry, example


def test_compiles_the_pinned_functiongemma_conversation_exactly() -> None:
    registry, example = _registry_and_example()

    compiled = compile_functiongemma(example, registry)

    assert compiled == (FIXTURES / "stage2-conversation.txt").read_text(encoding="utf-8")


def test_renders_the_input_only_prompt_as_the_exact_golden_prefix() -> None:
    registry, example = _registry_and_example()

    prompt = render_functiongemma_prompt(example.input, registry)
    golden = (FIXTURES / "stage2-conversation.txt").read_text(encoding="utf-8")

    assert prompt == golden.split("<start_function_call>", maxsplit=1)[0]


@pytest.mark.parametrize(
    "value",
    ["contains<escape>delimiter", "contains<end_function_call>delimiter"],
)
def test_rejects_ambiguous_native_string_delimiters(value: str) -> None:
    registry, example = _registry_and_example()
    ambiguous = example.model_copy(
        update={
            "route": example.route.model_copy(
                update={"arguments": {**example.route.arguments, "subject": value}}
            )
        }
    )

    with pytest.raises(ValueError, match="cannot be emitted unambiguously"):
        compile_functiongemma(ambiguous, registry)


@pytest.mark.parametrize(
    "decision, category",
    [
        (Decision(name="unknown_route"), "unknown_route"),
        (
            Decision(
                name="submit_ticket",
                arguments={
                    "subject": "Cannot sign in",
                    "priority": 2,
                    "ratio": 0.75,
                },
            ),
            "invalid_arguments",
        ),
    ],
)
def test_rejects_semantically_invalid_in_memory_examples(
    decision: Decision, category: str
) -> None:
    registry, example = _registry_and_example()
    invalid = Example(input=example.input, route=decision)

    with pytest.raises(DecisionValidationError) as raised:
        compile_functiongemma(invalid, registry)

    assert raised.value.category == category
