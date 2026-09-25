from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from equiroute.cli import app


@dataclass(frozen=True)
class _Report:
    payload: dict[str, Any]
    passed: bool

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return self.payload


def test_evaluate_prints_a_passing_report(monkeypatch) -> None:
    report = _Report(
        payload={
            "metrics": {"route_accuracy": 1.0},
            "passed": True,
            "schema_version": "1",
        },
        passed=True,
    )
    monkeypatch.setattr("equiroute.cli.evaluate_artifact", lambda artifact, data: report)

    result = CliRunner().invoke(
        app,
        ["evaluate", str(Path("artifact")), "--data", str(Path("examples.jsonl"))],
    )

    assert result.exit_code == 0
    assert result.output == '{"metrics":{"route_accuracy":1.0},"passed":true,"schema_version":"1"}\n'


def test_evaluate_prints_report_before_failing_quality_gate(monkeypatch) -> None:
    report = _Report(
        payload={
            "metrics": {"route_accuracy": 0.5},
            "passed": False,
            "schema_version": "1",
        },
        passed=False,
    )
    monkeypatch.setattr("equiroute.cli.evaluate_artifact", lambda artifact, data: report)

    result = CliRunner().invoke(
        app,
        ["evaluate", str(Path("artifact")), "--data", str(Path("examples.jsonl"))],
    )

    assert result.exit_code == 1
    assert result.output == '{"metrics":{"route_accuracy":0.5},"passed":false,"schema_version":"1"}\n'
