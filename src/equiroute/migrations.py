"""Explicit in-memory migrations for versioned EquiRoute documents.

Migrations transform raw mappings before Pydantic validates their current contract.
They never write to disk, so reading a legacy document cannot rewrite its bytes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CURRENT_SCHEMA_VERSION = "2"
_LEGACY_SCHEMA_VERSION = "1"
_SUPPORTED_VERSIONS = f'"{_LEGACY_SCHEMA_VERSION}" and "{CURRENT_SCHEMA_VERSION}"'


class SchemaMigrationError(ValueError):
    """A document declares a schema version this installation cannot load."""


def migrate_training_config(document: Any) -> Any:
    """Migrate an unversioned or v1 training configuration to the v2 raw form."""

    return _migrate_version(
        document,
        document_name="training configuration",
        allow_unversioned_v1=True,
    )


def migrate_raw_ingestion_config(document: Any) -> Any:
    """Require the only supported raw-ingestion configuration schema."""

    return _require_current_version(
        document, document_name="raw ingestion configuration"
    )


def migrate_raw_ingestion_manifest(document: Any) -> Any:
    """Require the only supported raw-ingestion manifest schema."""

    return _require_current_version(document, document_name="raw ingestion manifest")


def migrate_labeling_config(document: Any) -> Any:
    """Require the only supported candidate-labeling configuration schema."""

    return _require_current_version(document, document_name="labeling configuration")


def migrate_label_candidate(document: Any) -> Any:
    """Require the only supported candidate-label row schema."""

    return _require_current_version(document, document_name="label candidate")


def migrate_labeling_manifest(document: Any) -> Any:
    """Require the only supported candidate-label artifact manifest schema."""

    return _require_current_version(document, document_name="labeling manifest")


def migrate_review_config(document: Any) -> Any:
    """Require the only supported candidate-review configuration schema."""

    return _require_current_version(document, document_name="review configuration")


def migrate_acceptance_config(document: Any) -> Any:
    """Require the only supported label-acceptance configuration schema."""

    return _require_current_version(document, document_name="acceptance configuration")


def migrate_gold_quality_config(document: Any) -> Any:
    """Require the only supported gold-quality configuration schema."""

    return _require_current_version(
        document, document_name="gold quality configuration"
    )


def migrate_review_manifest(document: Any) -> Any:
    """Require the only supported candidate-review manifest schema."""

    return _require_current_version(document, document_name="review manifest")


def migrate_review_row(document: Any) -> Any:
    """Require the only supported candidate-review row schema."""

    return _require_current_version(document, document_name="review row")


def migrate_acceptance_manifest(document: Any) -> Any:
    """Require the only supported label-acceptance manifest schema."""

    return _require_current_version(document, document_name="acceptance manifest")


def migrate_review_quality_report(document: Any) -> Any:
    """Require the only supported candidate-review quality report schema."""

    return _require_current_version(document, document_name="review quality report")


def migrate_gold_quality_report(document: Any) -> Any:
    """Require the only supported generated-label gold-quality report schema."""

    return _require_current_version(document, document_name="gold quality report")


def migrate_dataset_report(document: Any) -> Any:
    """Migrate a versioned dataset report to the current raw form."""

    return _migrate_version(document, document_name="dataset report")


def migrate_dataset_manifest(document: Any) -> Any:
    """Migrate a versioned dataset manifest to the current raw form."""

    return _migrate_version(document, document_name="dataset manifest")


def migrate_evaluation_report(document: Any) -> Any:
    """Migrate a versioned semantic evaluation report to the current raw form."""

    return _migrate_version(document, document_name="evaluation report")


def migrate_training_evaluation(document: Any) -> Any:
    """Migrate legacy loss-evaluation evidence to the current raw form."""

    return _migrate_version(
        document,
        document_name="training evaluation",
        allow_unversioned_v1=True,
    )


def migrate_comparative_evaluation(document: Any) -> Any:
    """Migrate continuation evidence and its nested semantic reports."""

    migrated = _migrate_version(
        document,
        document_name="continuation evaluation",
        allow_unversioned_v1=True,
    )
    if not isinstance(migrated, Mapping):
        return migrated

    result: dict[Any, Any] | None = None
    for field in ("parent", "child"):
        nested = migrated.get(field)
        migrated_nested = migrate_evaluation_report(nested)
        if migrated_nested is not nested:
            if result is None:
                result = dict(migrated)
            result[field] = migrated_nested
    return migrated if result is None else result


def migrate_training_manifest(document: Any) -> Any:
    """Migrate a manifest and its nested comparative evaluation evidence."""

    migrated = _migrate_version(document, document_name="training manifest")
    if not isinstance(migrated, Mapping):
        return migrated

    result: dict[Any, Any] | None = None
    evaluation = migrated.get("evaluation")
    migrated_evaluation = migrate_training_evaluation(evaluation)
    if migrated_evaluation is not evaluation:
        result = dict(migrated)
        result["evaluation"] = migrated_evaluation

    comparison = migrated.get("comparative_evaluation")
    migrated_comparison = migrate_comparative_evaluation(comparison)
    if migrated_comparison is not comparison:
        if result is None:
            result = dict(migrated)
        result["comparative_evaluation"] = migrated_comparison
    return migrated if result is None else result


def _migrate_version(
    document: Any,
    *,
    document_name: str,
    allow_unversioned_v1: bool = False,
) -> Any:
    """Return a v2 raw mapping without mutating a caller-owned document."""

    if not isinstance(document, Mapping):
        return document

    version = document.get("schema_version")
    if version is None and "schema_version" not in document:
        if allow_unversioned_v1:
            migrated = dict(document)
            migrated["schema_version"] = CURRENT_SCHEMA_VERSION
            return migrated
        return document
    if not isinstance(version, str):
        raise SchemaMigrationError(
            f"{document_name} schema_version must be a string; use one of "
            f"{_SUPPORTED_VERSIONS}."
        )
    if version == CURRENT_SCHEMA_VERSION:
        return document
    if version == _LEGACY_SCHEMA_VERSION:
        migrated = dict(document)
        migrated["schema_version"] = CURRENT_SCHEMA_VERSION
        return migrated
    raise SchemaMigrationError(
        f"Unsupported {document_name} schema_version {version!r}; this EquiRoute "
        f"version supports {_SUPPORTED_VERSIONS}."
    )


def _require_current_version(document: Any, *, document_name: str) -> Any:
    """Return a raw v2 mapping only when it explicitly declares schema v2."""

    if not isinstance(document, Mapping):
        return document

    if "schema_version" not in document:
        raise SchemaMigrationError(
            f"{document_name} requires schema_version {CURRENT_SCHEMA_VERSION!r}; "
            "there is no supported legacy schema."
        )
    version = document["schema_version"]
    if not isinstance(version, str):
        raise SchemaMigrationError(
            f"{document_name} schema_version must be a string; use "
            f"{CURRENT_SCHEMA_VERSION!r}."
        )
    if version != CURRENT_SCHEMA_VERSION:
        raise SchemaMigrationError(
            f"Unsupported {document_name} schema_version {version!r}; this "
            f"EquiRoute version requires {CURRENT_SCHEMA_VERSION!r}."
        )
    return document
