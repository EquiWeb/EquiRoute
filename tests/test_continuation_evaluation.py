from __future__ import annotations

from pathlib import Path

import pytest

import equiroute.evaluation as evaluation
from equiroute.io import load_examples, load_route_registry
from equiroute.schemas import (
    ContinuationConfig,
    DatasetArtifact,
    EvaluationConfig,
    Example,
    Route,
    RouteRegistry,
)


FIXTURES = Path(__file__).parent / "fixtures" / "stage4"


def _old_inputs() -> tuple[RouteRegistry, list[Example], DatasetArtifact]:
    registry = load_route_registry(FIXTURES / "routes.yaml")
    examples = load_examples(FIXTURES / "examples.jsonl", registry)[:2]
    data = DatasetArtifact(examples=len(examples), fingerprint="a" * 64)
    return registry, examples, data


def _child_registry(parent: RouteRegistry) -> RouteRegistry:
    return parent.model_copy(
        update={
            "routes": [
                *parent.routes,
                Route.model_validate(
                    {
                        "name": "new_route",
                        "description": "A route added by continuation.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                            "additionalProperties": False,
                        },
                    }
                ),
            ]
        }
    )


def _correct_completions() -> list[str]:
    return [
        "<start_function_call>call:lookup_balance{account_id:<escape>A-001<escape>,include_pending:true}<end_function_call>",
        "<start_function_call>call:lookup_balance{account_id:<escape>B-002<escape>,include_pending:false}<end_function_call>",
    ]


def test_loaded_continuation_evaluation_uses_identical_parent_prompts_and_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_registry, examples, data = _old_inputs()
    child_registry = _child_registry(parent_registry)
    prompts: list[list[str]] = []

    def fake_generate(
        model: object,
        tokenizer: object,
        torch: object,
        device: str,
        rendered_prompts: list[str],
        config: EvaluationConfig,
    ) -> list[str]:
        prompts.append(rendered_prompts)
        assert tokenizer is None
        assert torch is None
        assert device == "cpu"
        assert config == EvaluationConfig()
        return _correct_completions()

    monkeypatch.setattr(evaluation, "_generate_loaded_completions", fake_generate)

    parent = evaluation.evaluate_loaded_artifact(
        object(),
        None,
        torch=None,
        device="cpu",
        scoring_registry=parent_registry,
        prompt_registry=parent_registry,
        examples=examples,
        data=data,
        config=EvaluationConfig(),
        artifact="parent",
    )
    child = evaluation.evaluate_loaded_artifact(
        object(),
        None,
        torch=None,
        device="cpu",
        scoring_registry=child_registry,
        prompt_registry=parent_registry,
        examples=examples,
        data=data,
        config=EvaluationConfig(),
        artifact="child",
    )

    comparison = evaluation.compare_continuation_evaluations(
        parent,
        child,
        ContinuationConfig(
            regression="regression.jsonl",
            max_route_accuracy_drop=0.0,
            max_argument_accuracy_drop=0.0,
        ),
    )

    assert prompts[0] == prompts[1]
    assert "new_route" not in prompts[1][0]
    assert parent.data == child.data == data
    assert [route.name for route in child.routes] == [
        *(route.name for route in parent_registry.routes),
        "new_route",
    ]
    assert comparison.route_accuracy_drop == 0.0
    assert comparison.argument_accuracy_drop == 0.0
    assert comparison.passed is True


def test_continuation_comparison_fails_when_accuracy_drop_exceeds_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_registry, examples, data = _old_inputs()
    child_registry = _child_registry(parent_registry)

    def fake_generate(
        model: str,
        tokenizer: object,
        torch: object,
        device: str,
        prompts: list[str],
        config: EvaluationConfig,
    ) -> list[str]:
        if model == "parent":
            return _correct_completions()
        return [
            "<start_function_call>call:replace_card{}<end_function_call>",
            _correct_completions()[1],
        ]

    monkeypatch.setattr(evaluation, "_generate_loaded_completions", fake_generate)

    parent = evaluation.evaluate_loaded_artifact(
        "parent",
        None,
        torch=None,
        device="cpu",
        scoring_registry=parent_registry,
        prompt_registry=parent_registry,
        examples=examples,
        data=data,
        config=EvaluationConfig(),
        artifact="parent",
    )
    child = evaluation.evaluate_loaded_artifact(
        "child",
        None,
        torch=None,
        device="cpu",
        scoring_registry=child_registry,
        prompt_registry=parent_registry,
        examples=examples,
        data=data,
        config=EvaluationConfig(),
        artifact="child",
    )

    comparison = evaluation.compare_continuation_evaluations(
        parent,
        child,
        ContinuationConfig(regression="regression.jsonl"),
    )

    assert comparison.route_accuracy_drop == 0.5
    assert comparison.argument_accuracy_drop == 0.5
    assert parent.passed is True
    assert child.passed is True
    assert comparison.passed is False
