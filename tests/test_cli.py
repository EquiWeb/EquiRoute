import json

from pathlib import Path

from typer.testing import CliRunner

from equiroute.cli import app


FIXTURES = Path(__file__).parent / "fixtures"


def test_help_lists_the_stage_zero_command_surface() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("init", "validate", "split", "train", "evaluate", "continue", "export"):
        assert command in result.output


def test_validate_reports_checked_in_disjoint_fixture_contract() -> None:
    result = CliRunner().invoke(app, ["validate", str(FIXTURES / "config.yaml")])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "example_count": 9,
        "route_distribution": [
            {
                "name": "billing_support",
                "test": 1,
                "total": 3,
                "train": 1,
                "validation": 1,
            },
            {
                "name": "technical_support",
                "test": 1,
                "total": 3,
                "train": 1,
                "validation": 1,
            },
            {
                "name": "account_support",
                "test": 1,
                "total": 3,
                "train": 1,
                "validation": 1,
            },
        ],
        "schema_version": "1",
    }
