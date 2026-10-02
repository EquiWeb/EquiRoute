"""Atomically materialize sanitized canonical raw-input rows."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from .dataset import (
    _canonical_compact_json_bytes,
    _canonical_pretty_json_bytes,
    _fingerprint,
)
from .errors import RawIngestionConfigError
from .io import iter_raw_inputs, load_raw_ingestion_config
from .schemas import RawIngestionConfig, RawIngestionManifest


def ingest_raw_inputs(config_path: str | Path) -> RawIngestionManifest:
    """Stream one configured source into an atomic, sanitized JSONL artifact.

    Source and output paths in the configuration are resolved relative to the
    configuration file.  The output directory is newly created only after all
    source rows have been projected, redacted, and serialized successfully.
    """

    configuration_path = Path(config_path)
    config = load_raw_ingestion_config(configuration_path)
    source = _resolve_config_path(configuration_path, config.source)
    output = _resolve_config_path(configuration_path, config.output.directory)
    resolved_config = config.model_copy(update={"source": str(source)})

    if os.path.lexists(output):
        raise RawIngestionConfigError(
            "refusing to overwrite an existing output directory",
            source=configuration_path,
            path="output.directory",
            correction="choose a new output directory or remove the existing one",
        )

    temporary: Path | None = None
    try:
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
        )
        source_hasher = hashlib.sha256()
        output_hasher = hashlib.sha256()
        row_count = 0

        with (temporary / "rows.jsonl").open("xb") as rows_file:
            for loaded in iter_raw_inputs(
                resolved_config, content_hasher=source_hasher
            ):
                row = _canonical_compact_json_bytes(loaded.row.model_dump(mode="json"))
                rows_file.write(row)
                rows_file.write(b"\n")
                output_hasher.update(row)
                output_hasher.update(b"\n")
                row_count += 1

        manifest = RawIngestionManifest(
            schema_version="2",
            source={"rows": row_count, "sha256": source_hasher.hexdigest()},
            output={"rows": row_count, "sha256": output_hasher.hexdigest()},
            config_fingerprint=_config_fingerprint(config),
            max_input_bytes=config.limits.max_input_bytes,
            redaction_count=len(config.redactions),
        )
        (temporary / "manifest.json").write_bytes(
            _canonical_pretty_json_bytes(manifest.model_dump(mode="json"))
        )
        os.replace(temporary, output)
    except BaseException:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        raise

    return manifest


def _resolve_config_path(config_path: Path, value: str) -> Path:
    """Resolve a configured filesystem path relative to its configuration."""

    return config_path.parent / value


def _config_fingerprint(config: RawIngestionConfig) -> str:
    """Fingerprint canonical configuration data without retaining it in output."""

    return _fingerprint(_canonical_compact_json_bytes(config.model_dump(mode="json")))
