"""Stage-8 orchestration for atomic, reviewable provider label candidates."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from time import monotonic as default_monotonic
from time import sleep as default_sleep
from typing import Any, Protocol

from pydantic import ValidationError

from .dataset import (
    _canonical_compact_json_bytes,
    _canonical_pretty_json_bytes,
    _fingerprint,
)
from .decisions import DecisionValidationError, validate_decision
from .errors import LabelingConfigError
from .io import (
    LoadedSanitizedInput,
    load_labeling_config,
    load_route_registry,
    load_sanitized_handoff,
    resolve_labeling_paths,
)
from .openrouter_client import OpenRouterClient, ProviderResponse
from .schemas import Decision, LabelCandidate, LabelingManifest, RouteRegistry


class CandidateLabelClient(Protocol):
    """The narrow provider operation needed by the labeling runner."""

    def request(
        self,
        *,
        model: str,
        policy_prompt: str,
        sanitized_input: str,
        source_id: str,
    ) -> ProviderResponse: ...


def label_sanitized_inputs(
    config_path: str | Path,
    *,
    client: CandidateLabelClient | None = None,
    sleep: Callable[[float], None] = default_sleep,
    monotonic: Callable[[], float] = default_monotonic,
) -> LabelingManifest:
    """Label a verified Stage-7 handoff into an atomic Stage-8 candidate artifact.

    The local configuration, handoff, and registry are completely validated before
    the output directory is reserved or a provider client can inspect credentials.
    Provider outcomes never become training examples: every source row produces a
    schema-valid candidate, with invalid outputs represented as safe rejections.
    """
    configuration_path = Path(config_path)
    config = load_labeling_config(configuration_path)
    handoff_path, registry_path, output = resolve_labeling_paths(
        configuration_path, config
    )
    handoff = load_sanitized_handoff(handoff_path)
    registry = load_route_registry(registry_path)
    _ensure_new_output(configuration_path, output)

    policy_fingerprint = _fingerprint(config.policy_prompt.encode("utf-8"))
    registry_fingerprint = _fingerprint(
        _canonical_compact_json_bytes(registry.model_dump(mode="json"))
    )
    input_manifest_fingerprint = _fingerprint(
        (handoff.directory / "manifest.json").read_bytes()
    )

    temporary = _prepare_output(output)
    try:
        provider: CandidateLabelClient
        if client is None:
            provider = OpenRouterClient(
                endpoint=config.provider.endpoint,
                credential_env_var=config.provider.credential_env_var,
                max_retries=config.max_retries,
                rate_limit_per_minute=config.rate_limit_per_minute,
                sleep=sleep,
                monotonic=monotonic,
            )
        else:
            provider = client

        candidates = _request_candidates(
            handoff.rows,
            registry=registry,
            client=provider,
            model=config.provider.model,
            endpoint=config.provider.endpoint,
            policy_prompt=config.policy_prompt,
            policy_fingerprint=policy_fingerprint,
            registry_fingerprint=registry_fingerprint,
            concurrency=config.concurrency,
        )
        return _emit_candidates(
            output,
            temporary,
            candidates,
            input_rows=handoff.manifest.output.rows,
            input_sha256=handoff.manifest.output.sha256,
            input_manifest_fingerprint=input_manifest_fingerprint,
            policy_fingerprint=policy_fingerprint,
            registry_fingerprint=registry_fingerprint,
            provider_model=config.provider.model,
            provider_endpoint=config.provider.endpoint,
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _ensure_new_output(config_path: Path, output: Path) -> None:
    if os.path.lexists(output):
        raise LabelingConfigError(
            "refusing to overwrite an existing output directory",
            source=config_path,
            path="output.directory",
            correction="choose a new output directory or remove the existing one",
        )


def _prepare_output(output: Path) -> Path:
    """Reserve only a sibling staging directory during local preflight."""
    return Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))


def _request_candidates(
    rows: tuple[LoadedSanitizedInput, ...],
    *,
    registry: RouteRegistry,
    client: CandidateLabelClient,
    model: str,
    endpoint: str,
    policy_prompt: str,
    policy_fingerprint: str,
    registry_fingerprint: str,
    concurrency: int,
) -> list[LabelCandidate]:
    """Bound provider work while retaining stable source order in the artifact."""
    if not rows:
        return []

    futures: list[Future[LabelCandidate]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for row in rows:
            futures.append(
                executor.submit(
                    _request_candidate,
                    row,
                    registry=registry,
                    client=client,
                    model=model,
                    endpoint=endpoint,
                    policy_prompt=policy_prompt,
                    policy_fingerprint=policy_fingerprint,
                    registry_fingerprint=registry_fingerprint,
                )
            )

        # Future collection deliberately follows source order rather than completion
        # order, so concurrent provider latency cannot reorder reviewable JSONL.
        return [future.result() for future in futures]


def _request_candidate(
    loaded: LoadedSanitizedInput,
    *,
    registry: RouteRegistry,
    client: CandidateLabelClient,
    model: str,
    endpoint: str,
    policy_prompt: str,
    policy_fingerprint: str,
    registry_fingerprint: str,
) -> LabelCandidate:
    response = client.request(
        model=model,
        policy_prompt=policy_prompt,
        sanitized_input=loaded.row.input,
        source_id=loaded.row.id,
    )
    source_line = loaded.row.metadata["_equiroute"]["source_line"]
    assert isinstance(source_line, int)
    provenance = {
        "source_id": loaded.row.id,
        "source_line": source_line,
        "policy_fingerprint": policy_fingerprint,
        "registry_fingerprint": registry_fingerprint,
        "provider_model": model,
        "provider_endpoint": endpoint,
        "request_status": response.status,
        "response_timestamp": response.timestamp,
        "attempts": response.attempts,
    }

    rejection = _provider_rejection(response)
    if rejection is not None:
        return LabelCandidate.model_validate(
            {
                "schema_version": "2",
                "provenance": provenance,
                "status": "rejected",
                "rejection_reason": rejection,
            }
        )

    decision = _parse_decision(response.content)
    if decision is None:
        return _rejected_candidate(provenance, "malformed_response")
    try:
        validate_decision(decision, registry)
    except DecisionValidationError:
        return _rejected_candidate(provenance, "invalid_decision")
    if not _decision_strings_are_utf8_encodable(decision):
        return _rejected_candidate(provenance, "invalid_decision")

    return LabelCandidate.model_validate(
        {
            "schema_version": "2",
            "provenance": provenance,
            "status": "labeled",
            "decision": decision.model_dump(mode="json"),
        }
    )


def _provider_rejection(response: ProviderResponse) -> str | None:
    return {
        "refused": "refusal",
        "timed_out": "timeout",
        "rate_limited": "rate_limited",
        "transport_failed": "transport_failure",
    }.get(response.status)


def _rejected_candidate(provenance: dict[str, Any], reason: str) -> LabelCandidate:
    return LabelCandidate.model_validate(
        {
            "schema_version": "2",
            "provenance": provenance,
            "status": "rejected",
            "rejection_reason": reason,
        }
    )


def _decision_strings_are_utf8_encodable(decision: Decision) -> bool:
    return _string_is_utf8_encodable(
        decision.name
    ) and _argument_strings_are_utf8_encodable(decision.arguments)


def _argument_strings_are_utf8_encodable(value: Any) -> bool:
    if isinstance(value, str):
        return _string_is_utf8_encodable(value)
    if isinstance(value, list):
        return all(_argument_strings_are_utf8_encodable(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str)
            and _string_is_utf8_encodable(key)
            and _argument_strings_are_utf8_encodable(item)
            for key, item in value.items()
        )
    return True


def _string_is_utf8_encodable(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _parse_decision(content: str | None) -> Decision | None:
    if content is None:
        return None
    try:
        document = json.loads(
            content,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        return Decision.model_validate(document)
    except (
        json.JSONDecodeError,
        TypeError,
        ValidationError,
        ValueError,
        RecursionError,
    ):
        return None


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("duplicate JSON object key")
        document[key] = value
    return document


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON constant {token!r}")


def _emit_candidates(
    output: Path,
    temporary: Path,
    candidates: list[LabelCandidate],
    *,
    input_rows: int,
    input_sha256: str,
    input_manifest_fingerprint: str,
    policy_fingerprint: str,
    registry_fingerprint: str,
    provider_model: str,
    provider_endpoint: str,
) -> LabelingManifest:
    """Write both Stage-8 files into a staging directory before publication."""
    candidates_path = temporary / "candidates.jsonl"
    digest = hashlib.sha256()
    with candidates_path.open("xb") as stream:
        for candidate in candidates:
            row = _canonical_compact_json_bytes(
                candidate.model_dump(mode="json", exclude_none=True)
            )
            stream.write(row)
            stream.write(b"\n")
            digest.update(row)
            digest.update(b"\n")

    manifest = LabelingManifest.model_validate(
        {
            "schema_version": "2",
            "input": {"rows": input_rows, "sha256": input_sha256},
            "output": {"rows": len(candidates), "sha256": digest.hexdigest()},
            "input_manifest_fingerprint": input_manifest_fingerprint,
            "policy_fingerprint": policy_fingerprint,
            "registry_fingerprint": registry_fingerprint,
            "provider_model": provider_model,
            "provider_endpoint": provider_endpoint,
        }
    )
    (temporary / "manifest.json").write_bytes(
        _canonical_pretty_json_bytes(manifest.model_dump(mode="json"))
    )
    if os.path.lexists(output):
        raise FileExistsError("refusing to overwrite an existing output directory")
    os.replace(temporary, output)
    return manifest
