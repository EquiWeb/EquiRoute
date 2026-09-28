from __future__ import annotations

import json
import tomllib

from pathlib import Path

import pytest
from typer.testing import CliRunner

import equiroute.init as starter
from equiroute.cli import app
from equiroute.init import InitError, create_starter_project
from equiroute.io import load_route_registry, load_training_config


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_init_creates_a_validatable_parent_and_continuation_tutorial(
    tmp_path: Path,
) -> None:
    target = tmp_path / "starter-router"

    result = CliRunner().invoke(app, ["init", str(target)])

    assert result.exit_code == 0
    assert result.output == (
        "Next commands:\n"
        f"  cd {target}\n"
        "  uv sync\n"
        "  See README.md for the complete workflow.\n"
    )
    readme = target / "README.md"
    assert readme.is_file()
    metadata = tomllib.loads((target / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"] == {
        "name": "equiroute-starter-router",
        "version": "0.1.0",
        "requires-python": ">=3.11",
        "dependencies": ["equiroute[model]"],
    }
    readme_text = readme.read_text(encoding="utf-8")
    parent_config_path = target / "config" / "parent.yaml"
    child_config_path = target / "config" / "add-shipping.yaml"
    parent_config = load_training_config(parent_config_path)
    child_config = load_training_config(child_config_path)
    parent_artifact = (
        (parent_config_path.parent / parent_config.output.directory)
        .resolve()
        .relative_to(target.resolve())
        .as_posix()
    )
    parent_test_data = (
        (parent_config_path.parent / parent_config.data.test)
        .resolve()
        .relative_to(target.resolve())
        .as_posix()
    )
    child_artifact = (
        (child_config_path.parent / child_config.output.directory)
        .resolve()
        .relative_to(target.resolve())
        .as_posix()
    )
    child_test_data = (
        (child_config_path.parent / child_config.data.test)
        .resolve()
        .relative_to(target.resolve())
        .as_posix()
    )
    model_load = (
        "uv run python -c 'from transformers import "
        f'AutoModelForCausalLM, AutoTokenizer; path = "{parent_artifact}/model"; '
        "tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True); "
        "model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)'"
    )
    workflow = [
        "uv sync",
        "uv run equiroute validate config/parent.yaml",
        "uv run equiroute validate config/add-shipping.yaml",
        "uv run equiroute train config/parent.yaml",
        f"uv run equiroute evaluate {parent_artifact} --data {parent_test_data}",
        model_load,
        "uv run equiroute continue "
        f"--from {parent_artifact} --config config/add-shipping.yaml",
        f"uv run equiroute evaluate {child_artifact} --data {child_test_data}",
        f"uv run equiroute export {child_artifact}",
    ]
    positions = [readme_text.index(f"$ {command}") for command in workflow]
    assert positions == sorted(positions)
    assert "$ uv sync --extra model" not in readme_text
    assert 'schema_version: "2"' in (target / "config" / "parent.yaml").read_text(
        encoding="utf-8"
    )
    assert 'schema_version: "2"' in (target / "config" / "add-shipping.yaml").read_text(
        encoding="utf-8"
    )

    parent = load_route_registry(target / "routes" / "parent.yaml")
    child = load_route_registry(target / "routes" / "add-shipping.yaml")
    assert [route.model_dump() for route in child.routes[: len(parent.routes)]] == [
        route.model_dump() for route in parent.routes
    ]
    assert [route.name for route in child.routes] == [
        "track_order",
        "cancel_order",
        "return_order",
        "add_shipping_address",
    ]

    for config in ("parent.yaml", "add-shipping.yaml"):
        validation = CliRunner().invoke(
            app, ["validate", str(target / "config" / config)]
        )
        assert validation.exit_code == 0, validation.output


def test_add_shipping_data_replays_parent_routes_and_seals_regression(
    tmp_path: Path,
) -> None:
    target = create_starter_project(tmp_path / "starter-router")
    data_directory = target / "data" / "add-shipping"
    partitions = {
        name: _records(data_directory / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    regression = _records(data_directory / "regression.jsonl")

    parent_routes = {"track_order", "cancel_order", "return_order"}
    child_routes = parent_routes | {"add_shipping_address"}
    for records in partitions.values():
        assert {record["route"]["name"] for record in records} == child_routes

    assert {record["route"]["name"] for record in regression} == parent_routes
    partition_ids = {
        record["id"] for records in partitions.values() for record in records
    }
    regression_ids = {record["id"] for record in regression}
    assert partition_ids.isdisjoint(regression_ids)


def test_init_refuses_to_overwrite_an_existing_destination(tmp_path: Path) -> None:
    target = tmp_path / "starter-router"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("do not overwrite", encoding="utf-8")

    result = CliRunner().invoke(app, ["init", str(target)])

    assert result.exit_code == 1
    assert "already exists; refusing to overwrite it" in result.output
    assert marker.read_text(encoding="utf-8") == "do not overwrite"


def test_init_removes_a_partial_destination_when_copying_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "starter-router"

    def fail_copy(_: Path, destination: Path, *, dirs_exist_ok: bool) -> None:
        assert dirs_exist_ok
        (destination / "partial.txt").write_text("partial", encoding="utf-8")
        raise OSError("simulated copy failure")

    monkeypatch.setattr(starter.shutil, "copytree", fail_copy)

    with pytest.raises(InitError, match="simulated copy failure"):
        create_starter_project(target)

    assert not target.exists()
