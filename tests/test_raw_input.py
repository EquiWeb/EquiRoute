from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from equiroute.errors import RawIngestionConfigError, RawInputLoadError
from equiroute.io import load_raw_ingestion_config
from equiroute.raw_input import ingest_raw_inputs


FIXTURES = Path(__file__).parent / "fixtures" / "stage7"


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "raw-project"
    shutil.copytree(FIXTURES, project)
    return project / "raw-ingestion.yaml"


def _compact_json_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_ingests_relative_source_to_sanitized_canonical_rows_and_manifest(
    tmp_path: Path,
) -> None:
    config_path = _project(tmp_path)
    source = config_path.parent / "raw-inputs.jsonl"
    source_before = source.read_bytes()

    manifest = ingest_raw_inputs(config_path)

    output = config_path.parent / "sanitized"
    rows = output.joinpath("rows.jsonl").read_bytes()
    assert rows == (
        b'{"id":"ticket-101","input":"Please email [email] about card [card].",'
        b'"metadata":{"_equiroute":{"source_line":1},"channel":"email"}}\n'
        b'{"id":"ticket-102","input":"Need help updating my profile.",'
        b'"metadata":{"_equiroute":{"source_line":2},"channel":"chat"}}\n'
    )
    assert source.read_bytes() == source_before

    persisted_rows = [json.loads(line) for line in rows.splitlines()]
    assert persisted_rows[0]["input"] == "Please email [email] about card [card]."
    assert persisted_rows[0]["metadata"]["_equiroute"] == {"source_line": 1}

    persisted_manifest = output.joinpath("manifest.json").read_bytes()
    manifest_document = json.loads(persisted_manifest)
    assert manifest_document == manifest.model_dump(mode="json")
    assert persisted_manifest == (
        json.dumps(
            manifest_document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    assert manifest.source.rows == 2
    assert manifest.source.sha256 == hashlib.sha256(source_before).hexdigest()
    assert manifest.output.rows == 2
    assert manifest.output.sha256 == hashlib.sha256(rows).hexdigest()
    config = load_raw_ingestion_config(config_path)
    assert (
        manifest.config_fingerprint
        == hashlib.sha256(
            _compact_json_bytes(config.model_dump(mode="json"))
        ).hexdigest()
    )

    manifest_text = persisted_manifest.decode("utf-8")
    for excluded in (
        "alice@example.com",
        "4111-1111-1111-1111",
        "ticket-101",
        "channel",
        "raw-inputs.jsonl",
        "[A-Za-z0-9._%+-]+",
    ):
        assert excluded not in manifest_text


def test_refuses_to_overwrite_an_existing_raw_ingestion_output(tmp_path: Path) -> None:
    config_path = _project(tmp_path)
    ingest_raw_inputs(config_path)
    output = config_path.parent / "sanitized"
    original = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(RawIngestionConfigError) as raised:
        ingest_raw_inputs(config_path)

    assert raised.value.path == "output.directory"
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original


def test_late_source_failure_leaves_no_partial_raw_ingestion_output(
    tmp_path: Path,
) -> None:
    config_path = _project(tmp_path)
    source = config_path.parent / "raw-inputs.jsonl"
    source.write_text(
        "\n".join(
            [
                '{"ticket":{"id":"first","text":"secret@example.com","channel":"email"}}',
                '{"ticket":{"id":"second","channel":"chat"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RawInputLoadError) as raised:
        ingest_raw_inputs(config_path)

    assert raised.value.line == 2
    assert "secret@example.com" not in str(raised.value)
    assert not (config_path.parent / "sanitized").exists()
    assert not list(config_path.parent.glob(".sanitized.tmp-*"))


def test_late_lone_surrogate_failure_leaves_no_partial_raw_ingestion_output(
    tmp_path: Path,
) -> None:
    config_path = _project(tmp_path)
    source = config_path.parent / "raw-inputs.jsonl"
    source.write_bytes(
        b'{"ticket":{"id":"first","text":"safe","channel":"email"}}\n'
        b'{"ticket":{"id":"second","text":"\\ud800","channel":"chat"}}\n'
    )

    with pytest.raises(RawInputLoadError) as raised:
        ingest_raw_inputs(config_path)

    assert raised.value.line == 2
    assert raised.value.path == "input"
    assert r"\ud800" not in str(raised.value)
    assert not (config_path.parent / "sanitized").exists()
    assert not list(config_path.parent.glob(".sanitized.tmp-*"))
