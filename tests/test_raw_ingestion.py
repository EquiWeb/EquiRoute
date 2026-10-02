from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from equiroute.errors import RawIngestionConfigError, RawInputLoadError
from equiroute.io import iter_raw_inputs, load_raw_ingestion_config
from equiroute.migrations import (
    SchemaMigrationError,
    migrate_raw_ingestion_config,
    migrate_raw_ingestion_manifest,
)
from equiroute.schemas import RawIngestionManifest, RawInputRow


def _config_document(source: Path) -> dict[str, object]:
    return {
        "schema_version": "2",
        "source": str(source),
        "output": {"directory": "sanitized-rows"},
        "projection": {
            "id": "/ticket/id",
            "input": "/ticket/text",
            "metadata": {"channel": "/ticket/channel"},
        },
        "redactions": [],
        "limits": {"max_input_bytes": 128},
    }


def _write_config(tmp_path: Path, document: dict[str, object]) -> Path:
    config = tmp_path / "raw-ingestion.yaml"
    config.write_text(json.dumps(document), encoding="utf-8")
    return config


def test_loads_and_streams_projected_redacted_canonical_raw_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "ticket": {
                    "id": "ticket-1",
                    "text": "Call SECRET 1234",
                    "channel": "inbound",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    document = _config_document(source)
    document["redactions"] = [
        {"target": "/input", "pattern": "SECRET", "replacement": "[removed]"},
        {"target": "/input", "pattern": r"\d+", "replacement": "[number]"},
        {
            "target": "/metadata/channel",
            "pattern": "in",
            "replacement": "X",
        },
    ]
    config = load_raw_ingestion_config(_write_config(tmp_path, document))

    loaded = next(iter_raw_inputs(config))

    assert loaded.source == source
    assert loaded.line == 1
    assert loaded.row is loaded.raw_input
    assert loaded.raw_input.model_dump() == {
        "id": "ticket-1",
        "input": "Call [removed] [number]",
        "metadata": {
            "_equiroute": {"source_line": 1},
            "channel": "Xbound",
        },
    }


def test_stream_supports_an_orchestrator_resolved_source_path(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"ticket":{"id":"ticket-1","text":"hello","channel":"chat"}}\n',
        encoding="utf-8",
    )
    config = load_raw_ingestion_config(
        _write_config(tmp_path, _config_document(source))
    )

    loaded = next(iter_raw_inputs(source, config))

    assert loaded.raw_input.id == "ticket-1"


def test_config_requires_v2_and_reports_invalid_regex_at_its_field(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    document = _config_document(source)
    document["schema_version"] = "1"

    with pytest.raises(RawIngestionConfigError) as version_error:
        load_raw_ingestion_config(_write_config(tmp_path, document))

    assert version_error.value.path == "schema_version"
    assert version_error.value.correction == 'set schema_version to "2"'

    document = _config_document(source)
    document["redactions"] = [{"target": "/input", "pattern": "["}]
    config_path = _write_config(tmp_path, document)
    with pytest.raises(RawIngestionConfigError) as regex_error:
        load_raw_ingestion_config(config_path)

    assert regex_error.value.path == "redactions.0.pattern"
    assert regex_error.value.correction == "correct invalid field: redactions.0.pattern"


def test_config_reserves_the_provenance_metadata_name(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    document = _config_document(source)
    projection = document["projection"]
    assert isinstance(projection, dict)
    metadata = projection["metadata"]
    assert isinstance(metadata, dict)
    metadata["_equiroute"] = "/ticket/export-provenance"

    with pytest.raises(RawIngestionConfigError) as raised:
        load_raw_ingestion_config(_write_config(tmp_path, document))

    assert raised.value.path == "projection.metadata"
    assert "reserved" in raised.value.message


@pytest.mark.parametrize(
    ("version", "declared"),
    [("1", True), ("3", True), (2, True), (None, True), (None, False)],
)
def test_raw_migrations_reject_every_non_v2_version_without_mutation(
    version: object, declared: bool
) -> None:
    document: dict[str, object] = {
        "source": "unused.jsonl",
        "output": {"directory": "sanitized"},
        "projection": {"id": "/id", "input": "/text"},
        "limits": {"max_input_bytes": 1},
    }
    if declared:
        document["schema_version"] = version

    original = deepcopy(document)
    with pytest.raises(SchemaMigrationError):
        migrate_raw_ingestion_config(document)

    assert document == original


def test_raw_manifest_migration_is_v2_only_and_idempotent() -> None:
    document = {
        "schema_version": "2",
        "source": {"rows": 1, "sha256": "a" * 64},
        "output": {"rows": 1, "sha256": "b" * 64},
        "config_fingerprint": "c" * 64,
        "max_input_bytes": 32,
        "redaction_count": 2,
    }

    migrated = migrate_raw_ingestion_manifest(document)

    assert migrated is document
    assert RawIngestionManifest.model_validate(migrated).schema_version == "2"
    with pytest.raises(SchemaMigrationError):
        migrate_raw_ingestion_manifest({})


def test_pointer_rejects_array_traversal_at_the_source_line_without_content(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"ticket":{"events":[{"id":"ticket-1"}],"text":"raw-secret","channel":"chat"}}\n',
        encoding="utf-8",
    )
    document = _config_document(source)
    projection = document["projection"]
    assert isinstance(projection, dict)
    projection["id"] = "/ticket/events/0/id"
    config = load_raw_ingestion_config(_write_config(tmp_path, document))

    with pytest.raises(RawInputLoadError) as raised:
        next(iter_raw_inputs(config))

    assert raised.value.line == 1
    assert raised.value.path == "projection.id"
    assert "arrays" in raised.value.message
    assert "raw-secret" not in str(raised.value)


def test_rejects_duplicate_ids_with_both_source_lines(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            [
                '{"ticket":{"id":"duplicate","text":"first","channel":"chat"}}',
                '{"ticket":{"id":"duplicate","text":"second","channel":"chat"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_raw_ingestion_config(
        _write_config(tmp_path, _config_document(source))
    )
    iterator = iter_raw_inputs(config)
    next(iterator)

    with pytest.raises(RawInputLoadError) as raised:
        next(iterator)

    assert raised.value.line == 2
    assert raised.value.path == "id"
    assert "first declared on line 1" in raised.value.message


def test_redaction_precedes_utf8_byte_limit_and_failures_do_not_reveal_input(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"ticket":{"id":"one","text":"xxSECRET","channel":"chat"}}\n',
        encoding="utf-8",
    )
    document = _config_document(source)
    document["redactions"] = [
        {"target": "/input", "pattern": "SECRET", "replacement": ""}
    ]
    document["limits"] = {"max_input_bytes": 2}
    config = load_raw_ingestion_config(_write_config(tmp_path, document))

    assert next(iter_raw_inputs(config)).raw_input.input == "xx"

    document["limits"] = {"max_input_bytes": 1}
    config = load_raw_ingestion_config(_write_config(tmp_path, document))
    with pytest.raises(RawInputLoadError) as raised:
        next(iter_raw_inputs(config))

    assert raised.value.path == "input"
    assert "after redaction" in raised.value.message
    assert "xxSECRET" not in str(raised.value)


def test_raw_row_requires_reserved_provenance_and_exact_contract() -> None:
    with pytest.raises(ValidationError, match="_equiroute"):
        RawInputRow.model_validate({"id": "one", "input": "hello", "metadata": {}})

    row = RawInputRow.model_validate(
        {
            "id": "one",
            "input": "hello",
            "metadata": {"_equiroute": {"source_line": 1}},
        }
    )
    assert row.model_dump() == {
        "id": "one",
        "input": "hello",
        "metadata": {"_equiroute": {"source_line": 1}},
    }
    with pytest.raises(ValidationError):
        RawInputRow.model_validate(
            {
                "id": "one",
                "input": "hello",
                "metadata": {"_equiroute": {"source_line": 1}},
                "route": {"name": "must-not-train"},
            }
        )


@pytest.mark.parametrize(
    ("raw_row", "path"),
    [
        (
            b'{"ticket":{"id":"\\ud800","text":"safe","channel":"chat"}}\n',
            "id",
        ),
        (
            b'{"ticket":{"id":"safe","text":"\\ud800","channel":"chat"}}\n',
            "input",
        ),
        (
            b'{"ticket":{"id":"safe","text":"safe","channel":{"nested":{"message":"\\ud800"}}}}\n',
            "metadata.channel.nested.message",
        ),
        (
            b'{"ticket":{"id":"safe","text":"safe","channel":{"nested":{"\\ud800":"value"}}}}\n',
            "metadata.channel.nested",
        ),
    ],
)
def test_rejects_lone_surrogates_in_canonical_projected_values(
    tmp_path: Path, raw_row: bytes, path: str
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(raw_row)
    config = load_raw_ingestion_config(
        _write_config(tmp_path, _config_document(source))
    )

    with pytest.raises(RawInputLoadError) as raised:
        next(iter_raw_inputs(config))

    assert raised.value.line == 1
    assert raised.value.path == path
    assert raised.value.message == "canonical value cannot be encoded as UTF-8"
    assert raised.value.correction == "replace invalid Unicode with UTF-8 text"
    assert r"\ud800" not in str(raised.value)


def test_preserves_utf8_encodable_canonical_projected_values(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(
        '{"ticket":{"id":"tëst","text":"Привет","channel":{"nested":{"café":"東京"}}}}\n'.encode(
            "utf-8"
        )
    )
    config = load_raw_ingestion_config(
        _write_config(tmp_path, _config_document(source))
    )

    loaded = next(iter_raw_inputs(config))

    assert loaded.raw_input.model_dump() == {
        "id": "tëst",
        "input": "Привет",
        "metadata": {
            "_equiroute": {"source_line": 1},
            "channel": {"nested": {"café": "東京"}},
        },
    }
