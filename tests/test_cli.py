import hashlib
import json
import os
import shutil
import string
import subprocess
import sys
from pathlib import Path

import pytest

from typer.testing import CliRunner

from equiroute.cli import app
from equiroute.errors import (
    AcceptanceConfigError,
    LabelingConfigError,
    ReviewConfigError,
)
from equiroute.init import create_starter_project
from equiroute.schemas import AcceptanceManifest, LabelingManifest, ReviewManifest

FIXTURES = Path(__file__).parent / "fixtures"


def test_help_lists_the_command_surface() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "init",
        "validate",
        "split",
        "ingest",
        "label",
        "review-labels",
        "accept-labels",
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


def _labeling_manifest() -> LabelingManifest:
    return LabelingManifest.model_validate(
        {
            "schema_version": "2",
            "input": {"rows": 2, "sha256": "a" * 64},
            "input_manifest_fingerprint": "b" * 64,
            "output": {"rows": 2, "sha256": "c" * 64},
            "policy_fingerprint": "d" * 64,
            "registry_fingerprint": "e" * 64,
            "provider_model": "openai/test-model",
            "provider_endpoint": "https://openrouter.ai/api/v1",
        }
    )


def test_label_emits_only_the_canonical_safe_manifest_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _labeling_manifest()
    monkeypatch.setattr("equiroute.cli.label_sanitized_inputs", lambda config: manifest)

    result = CliRunner().invoke(app, ["label", "labeling.yaml"])

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert result.stdout == (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def test_label_reports_setup_failures_without_policy_contents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "equiroute.cli.label_sanitized_inputs",
        lambda config: (_ for _ in ()).throw(
            LabelingConfigError(
                "invalid labeling configuration: policy-secret",
                source="labeling.yaml",
                path="policy_prompt",
                correction="repair the configuration",
            )
        ),
    )

    result = CliRunner().invoke(app, ["label", "labeling.yaml"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Labeling failed:" in result.stderr
    assert (
        "labeling.yaml: policy_prompt: invalid labeling configuration" in result.stderr
    )
    assert "policy-secret" not in result.stderr


def _review_manifest() -> ReviewManifest:
    return ReviewManifest.model_validate(
        {
            "schema_version": "2",
            "sanitized": {"rows": 2, "sha256": "a" * 64},
            "sanitized_manifest_fingerprint": "b" * 64,
            "candidates": {"rows": 2, "sha256": "c" * 64},
            "candidate_manifest_fingerprint": "d" * 64,
            "registry_fingerprint": "e" * 64,
            "policy_fingerprint": "f" * 64,
            "provider_model": "openai/test-model",
            "provider_endpoint": "https://openrouter.ai/api/v1",
            "rows": {"rows": 2, "sha256": "1" * 64},
            "immutable_fingerprint": "2" * 64,
        }
    )


def _acceptance_manifest() -> AcceptanceManifest:
    return AcceptanceManifest.model_validate(
        {
            "schema_version": "2",
            "sanitized": {"rows": 2, "sha256": "a" * 64},
            "sanitized_manifest_fingerprint": "b" * 64,
            "candidates": {"rows": 2, "sha256": "c" * 64},
            "candidate_manifest_fingerprint": "d" * 64,
            "review": {"rows": 2, "sha256": "e" * 64},
            "review_manifest_fingerprint": "f" * 64,
            "immutable_review_fingerprint": "1" * 64,
            "registry_fingerprint": "2" * 64,
            "policy_fingerprint": "3" * 64,
            "provider_model": "openai/test-model",
            "provider_endpoint": "https://openrouter.ai/api/v1",
            "output": {"rows": 1, "sha256": "4" * 64},
        }
    )


@pytest.mark.parametrize(
    ("command", "function_name", "manifest"),
    [
        ("review-labels", "review_label_candidates", _review_manifest()),
        ("accept-labels", "accept_approved_labels", _acceptance_manifest()),
    ],
)
def test_stage9_commands_forward_config_and_emit_only_canonical_manifest_summary(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    function_name: str,
    manifest: ReviewManifest | AcceptanceManifest,
) -> None:
    received: list[Path] = []
    monkeypatch.setattr(
        "equiroute.cli." + function_name,
        lambda config: (received.append(config), manifest)[1],
    )

    result = CliRunner().invoke(app, [command, "stage9.yaml"])

    assert result.exit_code == 0, result.output
    assert received == [Path("stage9.yaml")]
    assert result.stderr == ""
    assert result.stdout == (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


@pytest.mark.parametrize(
    ("command", "function_name", "error", "prefix"),
    [
        (
            "review-labels",
            "review_label_candidates",
            ReviewConfigError(
                "invalid review configuration: raw-source-secret",
                source="review.yaml",
                path="sampling",
                correction="repair the configuration",
            ),
            "Review failed:",
        ),
        (
            "accept-labels",
            "accept_approved_labels",
            AcceptanceConfigError(
                "invalid acceptance configuration: provider-credential-secret",
                source="acceptance.yaml",
                path="quotas",
                correction="repair the configuration",
            ),
            "Acceptance failed:",
        ),
    ],
)
def test_stage9_commands_render_safe_errors(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    function_name: str,
    error: Exception,
    prefix: str,
) -> None:
    monkeypatch.setattr(
        "equiroute.cli." + function_name,
        lambda config: (_ for _ in ()).throw(error),
    )

    result = CliRunner().invoke(app, [command, "stage9.yaml"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert prefix in result.stderr
    assert "repair the configuration" in result.stderr
    assert "raw-source-secret" not in result.stderr
    assert "provider-credential-secret" not in result.stderr


def test_live_smoke_refuses_without_configured_credential(tmp_path: Path) -> None:
    config = tmp_path / "labeling-live-smoke.yaml"
    config.write_text(
        """schema_version: "2"
input:
  directory: sanitized
routes: routes.yaml
output:
  directory: candidates
provider:
  endpoint: https://openrouter.ai/api/v1
  model: openai/test-model
  credential_env_var: EQUIROUTE_LIVE_SMOKE_TEST_KEY
policy_prompt: Choose one route.
concurrency: 1
rate_limit_per_minute: 1
max_retries: 0
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("EQUIROUTE_LIVE_SMOKE_TEST_KEY", None)

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "openrouter_live_smoke.py"),
            str(config),
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "required credential environment variable EQUIROUTE_LIVE_SMOKE_TEST_KEY" in (
        result.stderr
    )
