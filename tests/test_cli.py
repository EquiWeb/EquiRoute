from pathlib import Path

from typer.testing import CliRunner

from equiroute.cli import app


FIXTURES = Path(__file__).parent / "fixtures"


def test_help_lists_the_stage_zero_command_surface() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("init", "validate", "split", "train", "evaluate", "continue", "export"):
        assert command in result.output


def test_validate_loads_checked_in_fixture_contract() -> None:
    result = CliRunner().invoke(app, ["validate", str(FIXTURES / "config.yaml")])

    assert result.exit_code == 0
    assert "Validated" in result.output
