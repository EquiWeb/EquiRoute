import hashlib
import json
import shutil
import string
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
        "ingest",
        "train",
        "evaluate",
        "continue",
        "export",
    ):
        assert command in result.output
    ingest_help = CliRunner().invoke(app, ["ingest", "--help"])
    assert ingest_help.exit_code == 0
    assert "--debug" in ingest_help.output


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


def _raw_ingestion_config(tmp_path: Path, name: str) -> Path:
    project = tmp_path / name
    shutil.copytree(FIXTURES / "stage7", project)
    return project / "raw-ingestion.yaml"


def test_ingest_emits_only_the_canonical_manifest_summary(
    tmp_path: Path,
) -> None:
    config = _raw_ingestion_config(tmp_path, "normal")

    result = CliRunner().invoke(app, ["ingest", str(config)])

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    summary = json.loads(result.stdout)
    assert (
        result.stdout
        == json.dumps(
            summary,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    assert set(summary) == {
        "config_fingerprint",
        "max_input_bytes",
        "output",
        "redaction_count",
        "schema_version",
        "source",
    }
    assert len(summary["config_fingerprint"]) == 64
    assert set(summary["config_fingerprint"]) <= set(string.hexdigits.lower())
    assert summary["max_input_bytes"] == 512
    assert summary["output"] == {
        "rows": 2,
        "sha256": hashlib.sha256(
            config.parent.joinpath("sanitized", "rows.jsonl").read_bytes()
        ).hexdigest(),
    }
    assert summary["redaction_count"] == 2
    assert summary["schema_version"] == "2"
    assert summary["source"] == {
        "rows": 2,
        "sha256": hashlib.sha256(
            config.parent.joinpath("raw-inputs.jsonl").read_bytes()
        ).hexdigest(),
    }
    for sensitive_value in (
        "alice@example.com",
        "4111-1111-1111-1111",
        "Please email",
        "[email]",
        "[card]",
    ):
        assert sensitive_value not in result.output


def test_ingest_debug_preserves_stdout_and_adds_stderr_provenance(
    tmp_path: Path,
) -> None:
    normal = CliRunner().invoke(
        app, ["ingest", str(_raw_ingestion_config(tmp_path, "normal"))]
    )
    debug = CliRunner().invoke(
        app, ["ingest", str(_raw_ingestion_config(tmp_path, "debug")), "--debug"]
    )

    assert normal.exit_code == debug.exit_code == 0
    assert debug.stdout == normal.stdout
    assert "Ingestion debug:" in debug.stderr
    assert "config=" in debug.stderr
    assert "source_sha256=" in debug.stderr
    assert "output_sha256=" in debug.stderr
    for sensitive_value in ("alice@example.com", "4111-1111-1111-1111", "[email]"):
        assert sensitive_value not in debug.stderr


def test_ingest_reports_safe_source_line_and_reason_without_raw_input(
    tmp_path: Path,
) -> None:
    config = _raw_ingestion_config(tmp_path, "invalid")
    config.parent.joinpath("raw-inputs.jsonl").write_text(
        '{"ticket":{"id":"ticket-101","channel":"secret@example.com"}}\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["ingest", str(config)])

    assert result.exit_code == 1
    assert "Ingestion failed:" in result.stderr
    assert "raw-inputs.jsonl:1: projection.input" in result.stderr
    assert "JSON Pointer does not resolve in this source row" in result.stderr
    assert "correction:" in result.stderr
    assert "secret@example.com" not in result.output
