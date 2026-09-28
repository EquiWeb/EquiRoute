import json
from pathlib import Path

from typer.testing import CliRunner

from equiroute.cli import app


FIXTURES = Path(__file__).parent / "fixtures"
ROUTE_NAMES = ("billing_support", "technical_support", "account_support")


def _write_route_sorted_source(path: Path) -> None:
    examples = [
        {
            "id": f"{route_name}-{number:02d}",
            "input": f"{route_name} example request {number}",
            "route": {"name": route_name},
        }
        for route_name in ROUTE_NAMES
        for number in range(10)
    ]
    path.write_text(
        "".join(
            json.dumps(example, separators=(",", ":")) + "\n" for example in examples
        ),
        encoding="utf-8",
    )


def _read_route_names(path: Path) -> set[str]:
    return {
        json.loads(line)["route"]["name"]
        for line in path.read_text(encoding="utf-8").splitlines()
    }


def _write_leaky_config(directory: Path) -> Path:
    partition_paths = {
        partition: directory / f"{partition}.jsonl"
        for partition in ("train", "validation", "test")
    }
    for partition, path in partition_paths.items():
        examples = []
        for route_name in ROUTE_NAMES:
            input_text = f"{partition} {route_name} request"
            if partition == "train" and route_name == "billing_support":
                input_text = "Reset my password"
            elif partition == "test" and route_name == "billing_support":
                input_text = "  reset   MY password  "
            examples.append(
                {
                    "id": f"{partition}-{route_name}",
                    "input": input_text,
                    "route": {"name": route_name},
                }
            )
        path.write_text(
            "".join(json.dumps(example) + "\n" for example in examples),
            encoding="utf-8",
        )

    config = directory / "leaky-config.yaml"
    config.write_text(
        f"""model:
  base_model: google/functiongemma-270m-it
  revision: 39eccb091651513a5dfb56892d3714c1b5b8276c
routes: {json.dumps(str(FIXTURES / "routes.yaml"))}
data:
  train: {json.dumps(str(partition_paths["train"]))}
  validation: {json.dumps(str(partition_paths["validation"]))}
  test: {json.dumps(str(partition_paths["test"]))}
training:
  seed: 42
  epochs: 4
  learning_rate: 0.0002
  batch_size: 4
  gradient_accumulation_steps: 8
  lora_rank: 16
  lora_alpha: 32
  max_sequence_length: 1024
output:
  directory: runs/support-router-v1
  export: merged_huggingface
""",
        encoding="utf-8",
    )
    return config


def test_split_stratifies_a_route_sorted_source_and_announces_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "route-sorted.jsonl"
    output_directory = tmp_path / "published-splits"
    _write_route_sorted_source(source)

    result = CliRunner().invoke(
        app,
        [
            "split",
            str(source),
            "--routes",
            str(FIXTURES / "routes.yaml"),
            "--seed",
            "42",
            "--output-dir",
            str(output_directory),
        ],
    )

    assert result.exit_code == 0
    assert str(output_directory) in result.output
    assert {path.name for path in output_directory.iterdir()} == {
        "manifest.json",
        "report.json",
        "test.jsonl",
        "train.jsonl",
        "validation.jsonl",
    }
    assert {
        partition: _read_route_names(output_directory / f"{partition}.jsonl")
        for partition in ("train", "validation", "test")
    } == {partition: set(ROUTE_NAMES) for partition in ("train", "validation", "test")}


def test_split_refuses_to_overwrite_its_default_output_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    _write_route_sorted_source(source)
    command = ["split", str(source), "--routes", str(FIXTURES / "routes.yaml")]

    initial = CliRunner().invoke(app, command)
    assert initial.exit_code == 0

    output_directory = tmp_path / "source-splits"
    original_files = {
        path.name: path.read_bytes() for path in output_directory.iterdir()
    }
    repeated = CliRunner().invoke(app, command)

    assert repeated.exit_code == 1
    assert "Split failed:" in repeated.output
    assert {
        path.name: path.read_bytes() for path in output_directory.iterdir()
    } == original_files


def test_validate_rejects_normalized_input_leakage_between_configured_partitions(
    tmp_path: Path,
) -> None:
    config = _write_leaky_config(tmp_path)

    result = CliRunner().invoke(app, ["validate", str(config)])

    assert result.exit_code == 1
    assert "Validation failed:" in result.output
    assert "correction:" in result.output
