import json

from pathlib import Path

from typer.testing import CliRunner

from equiroute.cli import app
from equiroute.init import create_starter_project


FIXTURES = Path(__file__).parent / "fixtures"


def test_help_lists_the_command_surface() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "init",
        "validate",
        "split",
        "train",
        "evaluate",
        "continue",
        "export",
    ):
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
        "schema_version": "2",
    }


def test_validate_checks_generated_continuation_regression_locally(
    tmp_path: Path,
) -> None:
    target = create_starter_project(tmp_path / "starter")
    config = target / "config" / "add-shipping.yaml"

    result = CliRunner().invoke(app, ["validate", str(config)])

    assert result.exit_code == 0, result.output
    assert "Continuation regression data was validated locally" in result.output
    assert "`equiroute continue --from`" in result.output


def test_validate_rejects_continuation_regression_alias_and_content_overlap(
    tmp_path: Path,
) -> None:
    target = create_starter_project(tmp_path / "starter")
    config = target / "config" / "add-shipping.yaml"
    original_config = config.read_text(encoding="utf-8")
    regression = target / "data" / "add-shipping" / "regression.jsonl"
    train = target / "data" / "add-shipping" / "train.jsonl"

    config.write_text(
        original_config.replace(
            "../data/add-shipping/regression.jsonl", "../data/add-shipping/train.jsonl"
        ),
        encoding="utf-8",
    )
    aliased = CliRunner().invoke(app, ["validate", str(config)])

    assert aliased.exit_code == 1
    assert "must not alias a child partition" in aliased.output

    config.write_text(original_config, encoding="utf-8")
    regression.write_text(train.read_text(encoding="utf-8"), encoding="utf-8")
    overlapping = CliRunner().invoke(app, ["validate", str(config)])

    assert overlapping.exit_code == 1
    assert "reuses example id" in overlapping.output
