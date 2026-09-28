from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from equiroute.dataset import (
    split_dataset,
    validate_dataset,
    validate_partitions,
    write_split,
)
from equiroute.errors import ExampleLoadError
from equiroute.io import load_route_registry


FIXTURES = Path(__file__).parent / "fixtures"


def _registry():
    return load_route_registry(FIXTURES / "routes.yaml")


def _record(identifier: str | None, text: str, route: str) -> str:
    document = {"input": text, "route": {"name": route}}
    if identifier is not None:
        document["id"] = identifier
    return json.dumps(document)


def _write_records(path: Path, records: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(records) + "\n", encoding="utf-8")
    return path


def _partition_records(prefix: str, routes: tuple[str, ...]) -> list[str]:
    return [
        _record(f"{prefix}-{route}", f"{prefix} question for {route}", route)
        for route in routes
    ]


def _partition_map(tmp_path: Path) -> dict[str, Path]:
    routes = tuple(route.name for route in _registry().routes)
    return {
        partition: _write_records(
            tmp_path / f"{partition}.jsonl", _partition_records(partition, routes)
        )
        for partition in ("train", "validation", "test")
    }


def test_validate_dataset_requires_non_empty_stable_ids(tmp_path):
    registry = _registry()
    for identifier in (None, "", "  "):
        source = _write_records(
            tmp_path / f"{identifier!r}.jsonl",
            [_record(identifier, "A billing question", "billing_support")],
        )

        with pytest.raises(ExampleLoadError) as raised:
            validate_dataset(source, registry)

        assert f"{source}:1: id:" in str(raised.value)
        assert "non-empty stable identifier" in str(raised.value)
        assert raised.value.correction == "add a unique, non-empty id"


def test_validate_dataset_rejects_duplicate_ids_with_both_locations(tmp_path):
    source = _write_records(
        tmp_path / "duplicates.jsonl",
        [
            _record("reused", "First billing question", "billing_support"),
            _record("reused", "Different billing question", "billing_support"),
        ],
    )

    with pytest.raises(ExampleLoadError) as raised:
        validate_dataset(source, _registry())

    assert f"{source}:2: id: duplicate example id 'reused'" in str(raised.value)
    assert f"first declared at {source}:1" in str(raised.value)
    assert raised.value.correction == "assign a unique id"


def test_validate_dataset_rejects_nfkc_casefold_whitespace_input_leakage(tmp_path):
    source = _write_records(
        tmp_path / "normalized-inputs.jsonl",
        [
            _record("first", "Straße\u00a0payment status", "billing_support"),
            _record("second", "  STRASSE payment\tstatus ", "billing_support"),
        ],
    )

    with pytest.raises(ExampleLoadError) as raised:
        validate_dataset(source, _registry())

    assert f"{source}:2: input:" in str(raised.value)
    assert f"first declared at {source}:1" in str(raised.value)
    assert (
        raised.value.correction == "change the input so its normalized text is unique"
    )


def test_validate_partitions_reports_registry_order_and_curated_coverage(tmp_path):
    report = validate_partitions(_partition_map(tmp_path), _registry())

    assert [distribution.name for distribution in report.route_distribution] == [
        "billing_support",
        "technical_support",
        "account_support",
    ]
    assert report.example_count == 9
    assert [
        (
            distribution.total,
            distribution.train,
            distribution.validation,
            distribution.test,
        )
        for distribution in report.route_distribution
    ] == [(3, 1, 1, 1)] * 3


def test_validate_partitions_rejects_cross_file_identity_and_input_leakage(tmp_path):
    paths = _partition_map(tmp_path)
    routes = tuple(route.name for route in _registry().routes)
    _write_records(
        paths["validation"],
        [
            _record("train-billing_support", "validation question", "billing_support"),
            *_partition_records("different-validation", routes[1:]),
        ],
    )

    with pytest.raises(ExampleLoadError) as identity_error:
        validate_partitions(paths, _registry())

    assert f"{paths['validation']}:1: id:" in str(identity_error.value)
    assert f"first declared at {paths['train']}:1" in str(identity_error.value)

    paths = _partition_map(tmp_path / "input")
    _write_records(
        paths["validation"],
        [
            _record(
                "unique-validation-billing",
                " TRAIN question for billing_support ",
                "billing_support",
            ),
            *_partition_records(
                "unique-validation", ("technical_support", "account_support")
            ),
        ],
    )

    with pytest.raises(ExampleLoadError) as input_error:
        validate_partitions(paths, _registry())

    assert f"{paths['validation']}:1: input:" in str(input_error.value)
    assert f"first declared at {paths['train']}:1" in str(input_error.value)


def test_validate_partitions_rejects_aliased_paths_and_coverage_gaps(tmp_path):
    paths = _partition_map(tmp_path)
    paths["validation"] = paths["train"]

    with pytest.raises(ExampleLoadError, match="aliases") as alias_error:
        validate_partitions(paths, _registry())

    assert (
        alias_error.value.correction == "use a distinct file for each curated partition"
    )

    paths = _partition_map(tmp_path / "coverage")
    _write_records(
        paths["test"],
        _partition_records("test", ("billing_support", "technical_support")),
    )

    with pytest.raises(ExampleLoadError) as coverage_error:
        validate_partitions(paths, _registry())

    assert f"{paths['test']}: route.name:" in str(coverage_error.value)
    assert "missing coverage for route 'account_support'" in str(coverage_error.value)
    assert "add at least one 'account_support' example" in str(coverage_error.value)


def test_split_emits_route_covered_canonical_artifacts_with_exact_fingerprints(
    tmp_path,
):
    result = split_dataset(FIXTURES / "split-source.jsonl", _registry(), seed=42)
    output = tmp_path / "splits"
    manifest = write_split(result, output)

    assert [distribution.name for distribution in result.report.route_distribution] == [
        "billing_support",
        "technical_support",
        "account_support",
    ]
    assert [
        (
            distribution.total,
            distribution.train,
            distribution.validation,
            distribution.test,
        )
        for distribution in result.report.route_distribution
    ] == [(5, 3, 1, 1)] * 3
    assert (output / "report.json").read_bytes() == (
        json.dumps(
            result.report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )

    for partition in ("train", "validation", "test"):
        content = (output / f"{partition}.jsonl").read_bytes()
        records = [json.loads(line) for line in content.decode("utf-8").splitlines()]
        assert {record["route"]["name"] for record in records} == {
            "billing_support",
            "technical_support",
            "account_support",
        }
        assert manifest.datasets[partition].examples == len(records)
        assert (
            manifest.datasets[partition].fingerprint
            == hashlib.sha256(content).hexdigest()
        )
        assert all(
            line
            == json.dumps(
                json.loads(line),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for line in content.decode("utf-8").splitlines()
        )

    assert (
        manifest.source_fingerprint
        == hashlib.sha256((FIXTURES / "split-source.jsonl").read_bytes()).hexdigest()
    )
    assert json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    ) == manifest.model_dump(mode="json")


def test_split_same_seed_is_byte_identical_and_alternate_seed_moves_a_record(tmp_path):
    first = split_dataset(FIXTURES / "split-source.jsonl", _registry(), seed=42)
    second = split_dataset(FIXTURES / "split-source.jsonl", _registry(), seed=42)
    alternate = split_dataset(FIXTURES / "split-source.jsonl", _registry(), seed=43)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    write_split(first, first_output)
    write_split(second, second_output)

    for name in (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "report.json",
        "manifest.json",
    ):
        assert (first_output / name).read_bytes() == (second_output / name).read_bytes()

    first_assignments = {
        example.id: partition
        for partition, records in first.splits.items()
        for example in records
    }
    alternate_assignments = {
        example.id: partition
        for partition, records in alternate.splits.items()
        for example in records
    }
    assert any(
        first_assignments[identifier] != alternate_assignments[identifier]
        for identifier in first_assignments
    )


def test_split_requires_at_least_three_examples_for_every_route(tmp_path):
    source = _write_records(
        tmp_path / "too-small.jsonl",
        [
            _record("billing-1", "Billing one", "billing_support"),
            _record("billing-2", "Billing two", "billing_support"),
            _record("billing-3", "Billing three", "billing_support"),
            _record("technical-1", "Technical one", "technical_support"),
            _record("technical-2", "Technical two", "technical_support"),
            _record("technical-3", "Technical three", "technical_support"),
            _record("account-1", "Account one", "account_support"),
            _record("account-2", "Account two", "account_support"),
        ],
    )

    with pytest.raises(ExampleLoadError) as raised:
        split_dataset(source, _registry(), seed=42)

    assert f"{source}: route.name:" in str(raised.value)
    assert "at least 3 are required to populate every split" in str(raised.value)


def test_write_split_refuses_existing_output_and_cleans_temporary_files_on_error(
    tmp_path, monkeypatch
):
    result = split_dataset(FIXTURES / "split-source.jsonl", _registry(), seed=42)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ExampleLoadError, match="refusing to overwrite"):
        write_split(result, output)

    assert sentinel.read_text(encoding="utf-8") == "keep"

    failing_output = tmp_path / "failing"

    def fail_write(_self: Path, _content: bytes) -> int:
        raise OSError("simulated write failure")

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    with pytest.raises(OSError, match="simulated write failure"):
        write_split(result, failing_output)

    assert not failing_output.exists()
    assert not list(tmp_path.glob(".failing.tmp-*"))
