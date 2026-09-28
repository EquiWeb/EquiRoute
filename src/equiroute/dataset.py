"""Stage 1 dataset validation, deterministic splitting, and artifact emission."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .errors import ExampleLoadError
from .io import LoadedExample, iter_examples
from .schemas import (
    DatasetArtifact,
    DatasetManifest,
    DatasetReport,
    Example,
    RouteDistribution,
    RouteRegistry,
)

_SPLIT_NAMES = ("train", "validation", "test")
_SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


@dataclass(frozen=True, slots=True)
class _SourceLocation:
    source: Path
    line: int


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Validated, deterministic split records ready for atomic emission."""

    splits: dict[str, tuple[Example, ...]]
    report: DatasetReport
    registry_fingerprint: str
    source_fingerprint: str
    seed: int


def validate_dataset(path: str | Path, registry: RouteRegistry) -> DatasetReport:
    """Stream and validate one Stage 1 source dataset."""

    source = Path(path)
    counts = _empty_partition_counts(registry)
    identities: dict[str, _SourceLocation] = {}
    inputs: dict[str, _SourceLocation] = {}

    for loaded in iter_examples(source, registry):
        _validate_dataset_identity(loaded, identities, inputs)
        counts["train"][loaded.example.route.name] += 1

    return _build_report(counts, registry)


def validate_partitions(
    paths: Mapping[str, Path], registry: RouteRegistry
) -> DatasetReport:
    """Validate curated train, validation, and test partitions together."""

    sources = _curated_partition_sources(paths)
    _reject_aliased_partitions(sources)

    counts = _empty_partition_counts(registry)
    identities: dict[str, _SourceLocation] = {}
    inputs: dict[str, _SourceLocation] = {}
    for partition in _SPLIT_NAMES:
        for loaded in iter_examples(sources[partition], registry):
            _validate_dataset_identity(loaded, identities, inputs)
            counts[partition][loaded.example.route.name] += 1

    for partition in _SPLIT_NAMES:
        for route in registry.routes:
            if counts[partition][route.name] == 0:
                raise ExampleLoadError(
                    f"missing coverage for route {route.name!r} in the {partition!r} partition",
                    source=sources[partition],
                    path="route.name",
                    correction=f"add at least one {route.name!r} example to the {partition!r} partition",
                )

    return _build_report(counts, registry)


def validate_continuation_regression(
    regression: str | Path,
    child_paths: Mapping[str, Path],
    child_registry: RouteRegistry,
) -> DatasetReport:
    """Validate local regression data without asserting an unknown parent artifact."""

    source = Path(regression)
    child_sources = _curated_partition_sources(child_paths)
    _reject_aliased_partitions(child_sources)
    if any(
        source.resolve(strict=False) == partition_source.resolve(strict=False)
        for partition_source in child_sources.values()
    ):
        raise ExampleLoadError(
            "continuation regression dataset must not alias a child partition",
            source=source,
            path="path",
            correction="use a separate regression JSONL file",
        )

    report = validate_dataset(source, child_registry)
    regression_ids: set[str] = set()
    regression_inputs: set[str] = set()
    for loaded in iter_examples(source, child_registry):
        regression_ids.add(_required_id(loaded.example))
        regression_inputs.add(_normalize_input(loaded.example.input))
    for partition in _SPLIT_NAMES:
        partition_source = child_sources[partition]
        for loaded in iter_examples(partition_source, child_registry):
            example = loaded.example
            if example.id in regression_ids:
                raise ExampleLoadError(
                    f"continuation regression dataset reuses example id {example.id!r} "
                    f"from child {partition} partition",
                    source=source,
                    path="id",
                    correction="use an example id distinct from every child partition",
                )
            if _normalize_input(example.input) in regression_inputs:
                raise ExampleLoadError(
                    "continuation regression dataset reuses an input from child "
                    f"{partition} partition",
                    source=source,
                    path="input",
                    correction="use input text distinct from every child partition",
                )
    return report


def split_dataset(
    data: str | Path, registry: RouteRegistry, *, seed: int
) -> SplitResult:
    """Validate and deterministically split one source at fixed Stage 1 ratios."""

    source = Path(data)
    groups: dict[str, list[Example]] = {route.name: [] for route in registry.routes}
    identities: dict[str, _SourceLocation] = {}
    inputs: dict[str, _SourceLocation] = {}
    source_hasher = hashlib.sha256()
    for loaded in iter_examples(source, registry, content_hasher=source_hasher):
        _validate_dataset_identity(loaded, identities, inputs)
        groups[loaded.example.route.name].append(loaded.example)
    source_fingerprint = source_hasher.hexdigest()

    for route in registry.routes:
        if len(groups[route.name]) < len(_SPLIT_NAMES):
            raise ExampleLoadError(
                f"route {route.name!r} has {len(groups[route.name])} examples; at least 3 are required to populate every split",
                source=source,
                path="route.name",
                correction=f"add examples for route {route.name!r} until it has at least 3 unique records",
            )

    split_records: dict[str, list[Example]] = {name: [] for name in _SPLIT_NAMES}
    counts = _empty_partition_counts(registry)
    for route in registry.routes:
        name = route.name
        shuffled = sorted(
            groups[name],
            key=lambda example: _shuffle_key(seed, name, _required_id(example)),
        )
        allocations = _split_counts(len(shuffled))
        start = 0
        for partition in _SPLIT_NAMES:
            stop = start + allocations[partition]
            split_records[partition].extend(shuffled[start:stop])
            counts[partition][name] = allocations[partition]
            start = stop

    return SplitResult(
        splits={name: tuple(split_records[name]) for name in _SPLIT_NAMES},
        report=_build_report(counts, registry),
        registry_fingerprint=_registry_fingerprint(registry),
        source_fingerprint=source_fingerprint,
        seed=seed,
    )


def write_split(result: SplitResult, output_directory: str | Path) -> DatasetManifest:
    """Atomically emit a validated split without overwriting an existing target."""

    output = Path(output_directory)
    if os.path.lexists(output):
        raise ExampleLoadError(
            "refusing to overwrite an existing output directory",
            source=output,
            correction="choose a new output directory or remove the existing one",
        )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        artifacts: dict[str, DatasetArtifact] = {}
        for partition in _SPLIT_NAMES:
            content = b"".join(
                _canonical_jsonl_bytes(example) for example in result.splits[partition]
            )
            _write_bytes(temporary / f"{partition}.jsonl", content)
            artifacts[partition] = DatasetArtifact(
                examples=len(result.splits[partition]),
                fingerprint=_fingerprint(content),
            )

        _write_bytes(
            temporary / "report.json",
            _canonical_pretty_json_bytes(result.report.model_dump(mode="json")),
        )
        manifest = DatasetManifest(
            schema_version="2",
            registry_fingerprint=result.registry_fingerprint,
            source_fingerprint=result.source_fingerprint,
            split_seed=result.seed,
            split_ratios=_SPLIT_RATIOS,
            datasets=artifacts,
        )
        _write_bytes(
            temporary / "manifest.json",
            _canonical_pretty_json_bytes(manifest.model_dump(mode="json")),
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return manifest


def _curated_partition_sources(paths: Mapping[str, Path]) -> dict[str, Path]:
    supplied = set(paths)
    required = set(_SPLIT_NAMES)
    if supplied != required:
        missing = ", ".join(name for name in _SPLIT_NAMES if name not in supplied)
        unexpected = ", ".join(
            sorted(name for name in supplied if name not in required)
        )
        details: list[str] = []
        if missing:
            details.append(f"missing partitions: {missing}")
        if unexpected:
            details.append(f"unexpected partitions: {unexpected}")
        raise ExampleLoadError(
            "; ".join(details),
            source="partitions",
            path="data",
            correction="provide exactly train, validation, and test dataset paths",
        )
    return {name: Path(paths[name]) for name in _SPLIT_NAMES}


def _reject_aliased_partitions(sources: Mapping[str, Path]) -> None:
    seen: dict[Path, tuple[str, Path]] = {}
    for partition in _SPLIT_NAMES:
        source = sources[partition]
        resolved = source.resolve(strict=False)
        first = seen.get(resolved)
        if first is not None:
            first_partition, first_source = first
            raise ExampleLoadError(
                f"partition path aliases {first_partition!r}: {first_source}",
                source=source,
                path="path",
                correction="use a distinct file for each curated partition",
            )
        seen[resolved] = (partition, source)


def _validate_dataset_identity(
    loaded: LoadedExample,
    identities: dict[str, _SourceLocation],
    inputs: dict[str, _SourceLocation],
) -> None:
    identifier = loaded.example.id
    if identifier is None or not identifier.strip():
        raise ExampleLoadError(
            "example id must be a non-empty stable identifier",
            source=loaded.source,
            line=loaded.line,
            path="id",
            correction="add a unique, non-empty id",
        )

    first_identity = identities.get(identifier)
    if first_identity is not None:
        raise ExampleLoadError(
            f"duplicate example id {identifier!r}; first declared at {_location_text(first_identity)}",
            source=loaded.source,
            line=loaded.line,
            path="id",
            correction="assign a unique id",
        )

    normalized_input = _normalize_input(loaded.example.input)
    first_input = inputs.get(normalized_input)
    if first_input is not None:
        raise ExampleLoadError(
            "normalized input duplicates an earlier example; "
            f"first declared at {_location_text(first_input)}",
            source=loaded.source,
            line=loaded.line,
            path="input",
            correction="change the input so its normalized text is unique",
        )

    location = _SourceLocation(source=loaded.source, line=loaded.line)
    identities[identifier] = location
    inputs[normalized_input] = location


def _empty_partition_counts(registry: RouteRegistry) -> dict[str, dict[str, int]]:
    return {
        partition: {route.name: 0 for route in registry.routes}
        for partition in _SPLIT_NAMES
    }


def _build_report(
    counts: Mapping[str, Mapping[str, int]], registry: RouteRegistry
) -> DatasetReport:
    distributions = [
        RouteDistribution(
            name=route.name,
            total=sum(counts[partition][route.name] for partition in _SPLIT_NAMES),
            train=counts["train"][route.name],
            validation=counts["validation"][route.name],
            test=counts["test"][route.name],
        )
        for route in registry.routes
    ]
    return DatasetReport(
        schema_version="2",
        example_count=sum(distribution.total for distribution in distributions),
        route_distribution=distributions,
    )


def _normalize_input(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _split_counts(total: int) -> dict[str, int]:
    """Allocate a largest-remainder 80/10/10 split with one record per split."""

    quotas = {name: total * _SPLIT_RATIOS[name] for name in _SPLIT_NAMES}
    counts = {name: int(quotas[name]) for name in _SPLIT_NAMES}
    remaining = total - sum(counts.values())
    for partition in sorted(
        _SPLIT_NAMES,
        key=lambda name: (-(quotas[name] - counts[name]), _SPLIT_NAMES.index(name)),
    )[:remaining]:
        counts[partition] += 1

    for partition in _SPLIT_NAMES:
        if counts[partition] != 0:
            continue
        donor = next(
            name
            for name in sorted(
                _SPLIT_NAMES,
                key=lambda name: (-counts[name], _SPLIT_NAMES.index(name)),
            )
            if counts[name] > 1
        )
        counts[donor] -= 1
        counts[partition] += 1
    return counts


def _shuffle_key(seed: int, route: str, identifier: str) -> bytes:
    return hashlib.sha256(
        f"equiroute-stage1\0{seed}\0{route}\0{identifier}".encode("utf-8")
    ).digest()


def _required_id(example: Example) -> str:
    """Return an ID after the Stage 1 source validator has enforced it."""

    assert example.id is not None
    return example.id


def _location_text(location: _SourceLocation) -> str:
    return f"{location.source}:{location.line}"


def _file_fingerprint(source: Path, document_name: str) -> str:
    try:
        return _fingerprint(source.read_bytes())
    except OSError as error:
        raise ExampleLoadError(
            f"could not read {document_name}: {error}",
            source=source,
            correction="ensure the file exists and is readable",
        ) from error


def _registry_fingerprint(registry: RouteRegistry) -> str:
    return _fingerprint(_canonical_compact_json_bytes(registry.model_dump(mode="json")))


def _canonical_jsonl_bytes(example: Example) -> bytes:
    return (
        _canonical_compact_json_bytes(
            example.model_dump(mode="json", exclude_none=True)
        )
        + b"\n"
    )


def _canonical_compact_json_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_pretty_json_bytes(document: object) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )


def _fingerprint(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_bytes(path: Path, content: bytes) -> None:
    path.write_bytes(content)
