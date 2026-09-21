from __future__ import annotations

import pytest

from equiroute.decisions import DecisionValidationError, validate_decision
from equiroute.schemas import Decision, RouteRegistry


@pytest.fixture
def registry() -> RouteRegistry:
    return RouteRegistry.model_validate(
        {
            "routes": [
                {
                    "name": "typed_route",
                    "description": "Accepts every supported primitive argument.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "count": {"type": "integer"},
                            "score": {"type": "number"},
                            "enabled": {"type": "boolean"},
                        },
                        "required": ["text", "count", "score", "enabled"],
                        "additionalProperties": False,
                    },
                }
            ]
        }
    )


def test_accepts_registered_decision_with_flat_primitive_arguments(registry):
    decision = Decision(
        name="typed_route",
        arguments={"text": "hello", "count": 3, "score": 2.5, "enabled": True},
    )

    assert validate_decision(decision, registry) is None


def test_rejects_unknown_route_with_typed_error(registry):
    with pytest.raises(DecisionValidationError) as raised:
        validate_decision(Decision(name="missing"), registry)

    assert raised.value.category == "unknown_route"
    assert raised.value.detail == "unknown route 'missing'"
    assert str(raised.value) == raised.value.detail


def test_rejects_missing_required_argument(registry):
    with pytest.raises(DecisionValidationError) as raised:
        validate_decision(
            Decision(name="typed_route", arguments={"text": "hello"}), registry
        )

    assert raised.value.category == "invalid_arguments"
    assert raised.value.detail == (
        "missing required argument 'count'; missing required argument 'score'; "
        "missing required argument 'enabled'"
    )


def test_rejects_extra_argument(registry):
    with pytest.raises(DecisionValidationError) as raised:
        validate_decision(
            Decision(
                name="typed_route",
                arguments={
                    "text": "hello",
                    "count": 3,
                    "score": 2.5,
                    "enabled": True,
                    "extra": None,
                },
            ),
            registry,
        )

    assert raised.value.category == "invalid_arguments"
    assert raised.value.detail == "unknown argument 'extra'"


@pytest.mark.parametrize(
    ("name", "value", "expected_detail"),
    [
        ("text", 1, "argument 'text' must be string, got integer"),
        ("count", True, "argument 'count' must be integer, got boolean"),
        ("score", True, "argument 'score' must be number, got boolean"),
        ("enabled", 1, "argument 'enabled' must be boolean, got integer"),
    ],
)
def test_rejects_wrong_primitive_argument_type(
    registry, name, value, expected_detail
):
    arguments = {"text": "hello", "count": 3, "score": 2.5, "enabled": True}
    arguments[name] = value

    with pytest.raises(DecisionValidationError) as raised:
        validate_decision(Decision(name="typed_route", arguments=arguments), registry)

    assert raised.value.category == "invalid_arguments"
    assert raised.value.detail == expected_detail
