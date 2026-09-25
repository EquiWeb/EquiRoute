from pathlib import Path

from typer.testing import CliRunner

import equiroute.cli as cli
from equiroute.training import TrainingError


def test_training_command_help_exposes_training_arguments() -> None:
    runner = CliRunner()

    train_help = runner.invoke(cli.app, ["train", "--help"])
    continue_help = runner.invoke(cli.app, ["continue", "--help"])
    export_help = runner.invoke(cli.app, ["export", "--help"])

    assert train_help.exit_code == 0
    assert "config" in train_help.output
    assert "--resume" in train_help.output
    assert continue_help.exit_code == 0
    assert "--from" in continue_help.output
    assert "--config" in continue_help.output
    assert export_help.exit_code == 0
    assert "artifact" in export_help.output
    assert "--format" in export_help.output


def test_train_forwards_config_and_resume_to_public_api(monkeypatch, tmp_path) -> None:
    config = tmp_path / "config.yaml"
    calls: list[tuple[Path, bool]] = []

    def fake_train_router(config_path: Path, *, resume: bool):
        calls.append((config_path, resume))
        return object()

    monkeypatch.setattr(cli, "train_router", fake_train_router)

    result = CliRunner().invoke(cli.app, ["train", str(config), "--resume"])

    assert result.exit_code == 0
    assert calls == [(config, True)]


def test_train_reports_actionable_public_api_failures(monkeypatch, tmp_path) -> None:
    config = tmp_path / "config.yaml"

    def fake_train_router(config_path: Path, *, resume: bool):
        raise TrainingError("install the model extra with `uv sync --extra model`")

    monkeypatch.setattr(cli, "train_router", fake_train_router)

    result = CliRunner().invoke(cli.app, ["train", str(config)])

    assert result.exit_code == 1
    assert "Training failed:" in result.output
    assert "install the model extra with `uv sync --extra model`" in result.output


def test_export_forwards_huggingface_format_to_public_api(monkeypatch, tmp_path) -> None:
    artifact = tmp_path / "artifact"
    calls: list[Path] = []

    def fake_export_router(artifact_path: Path):
        calls.append(artifact_path)
        return object()

    monkeypatch.setattr(cli, "export_router", fake_export_router)

    result = CliRunner().invoke(
        cli.app, ["export", str(artifact), "--format", "huggingface"]
    )

    assert result.exit_code == 0
    assert calls == [artifact]


def test_export_rejects_formats_other_than_huggingface(monkeypatch, tmp_path) -> None:
    artifact = tmp_path / "artifact"

    def unexpected_export_router(artifact_path: Path):
        raise AssertionError("export_router must not receive an unsupported format")

    monkeypatch.setattr(cli, "export_router", unexpected_export_router)

    result = CliRunner().invoke(cli.app, ["export", str(artifact), "--format", "archive"])

    assert result.exit_code == 2
    assert "only --format huggingface is supported" in result.output


def test_export_reports_actionable_public_api_failures(monkeypatch, tmp_path) -> None:
    artifact = tmp_path / "artifact"

    def fake_export_router(artifact_path: Path):
        raise TrainingError("artifact has no retained adapter")

    monkeypatch.setattr(cli, "export_router", fake_export_router)

    result = CliRunner().invoke(cli.app, ["export", str(artifact)])

    assert result.exit_code == 1
    assert "Export failed:" in result.output
    assert "artifact has no retained adapter" in result.output


def test_continue_forwards_parent_artifact_and_config_to_public_api(monkeypatch, tmp_path) -> None:
    parent = tmp_path / "parent"
    config = tmp_path / "config.yaml"
    calls: list[tuple[Path, Path]] = []

    def fake_continue_router(from_artifact: Path, config_path: Path):
        calls.append((from_artifact, config_path))
        return object()

    monkeypatch.setattr(cli, "continue_router", fake_continue_router)

    result = CliRunner().invoke(
        cli.app, ["continue", "--from", str(parent), "--config", str(config)]
    )

    assert result.exit_code == 0
    assert calls == [(parent, config)]


def test_continue_reports_actionable_public_api_failures(monkeypatch, tmp_path) -> None:
    parent = tmp_path / "parent"
    config = tmp_path / "config.yaml"

    def fake_continue_router(from_artifact: Path, config_path: Path):
        raise TrainingError("parent artifact has no retained adapter")

    monkeypatch.setattr(cli, "continue_router", fake_continue_router)

    result = CliRunner().invoke(
        cli.app, ["continue", "--from", str(parent), "--config", str(config)]
    )

    assert result.exit_code == 1
    assert "Continuation failed:" in result.output
    assert "parent artifact has no retained adapter" in result.output
