from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from equiroute.errors import LabelingConfigError

from equiroute.labeling import label_sanitized_inputs
from equiroute.openrouter_client import ProviderResponse


FIXTURES = Path(__file__).parent / "fixtures" / "stage8" / "openrouter"
TIMESTAMP = (
    datetime(2026, 10, 2, 12, 30, 45, tzinfo=UTC).isoformat().replace("+00:00", "Z")
)


class FakeAdapter:
    def __init__(self, responses: dict[str, ProviderResponse]) -> None:
        self._responses = responses
        self.calls: list[dict[str, str]] = []

    def request(
        self,
        *,
        model: str,
        policy_prompt: str,
        sanitized_input: str,
        source_id: str,
    ) -> ProviderResponse:
        self.calls.append(
            {
                "model": model,
                "policy_prompt": policy_prompt,
                "sanitized_input": sanitized_input,
                "source_id": source_id,
            }
        )
        return self._responses[source_id]


class ConcurrentAdapter(FakeAdapter):
    def __init__(self, responses: dict[str, ProviderResponse]) -> None:
        super().__init__(responses)
        self._lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0

    def request(self, **kwargs: str) -> ProviderResponse:
        with self._lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            if kwargs["source_id"] == "row-1":
                time.sleep(0.03)
            return super().request(**kwargs)
        finally:
            with self._lock:
                self.active -= 1


def _response(
    source_id: str,
    *,
    status: str = "succeeded",
    content: str | None = '{"name":"billing","arguments":{}}',
) -> ProviderResponse:
    return ProviderResponse(
        source_id=source_id,
        status=status,  # type: ignore[arg-type]
        timestamp=TIMESTAMP,
        attempts=1,
        content=content,
    )


def _write_run_inputs(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    handoff = tmp_path / "sanitized"
    handoff.mkdir()
    rows_bytes = b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
    (handoff / "rows.jsonl").write_bytes(rows_bytes)
    stage7_manifest = {
        "schema_version": "2",
        "source": {"rows": len(rows), "sha256": "0" * 64},
        "output": {"rows": len(rows), "sha256": hashlib.sha256(rows_bytes).hexdigest()},
        "config_fingerprint": "1" * 64,
        "max_input_bytes": 1024,
        "redaction_count": 1,
    }
    (handoff / "manifest.json").write_text(
        json.dumps(stage7_manifest, sort_keys=True), encoding="utf-8"
    )
    (tmp_path / "routes.yaml").write_text(
        """routes:
  - name: billing
    description: Billing support.
    parameters:
      type: object
      properties: {}
      additionalProperties: false
  - name: billing_support
    description: Subscription billing support.
    parameters:
      type: object
      properties: {}
      additionalProperties: false
""",
        encoding="utf-8",
    )
    config = {
        "schema_version": "2",
        "input": {"directory": "sanitized"},
        "routes": "routes.yaml",
        "output": {"directory": "candidates"},
        "provider": {
            "endpoint": "https://openrouter.ai/api/v1",
            "model": "openai/test-model",
            "credential_env_var": "EQUIROUTE_TEST_SECRET",
        },
        "policy_prompt": "POLICY_SECRET: choose one route.",
        "concurrency": 2,
        "rate_limit_per_minute": 10_000,
        "max_retries": 0,
    }
    config_path = tmp_path / "labeling.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _row(identifier: str, line: int, value: str = "sanitized value") -> dict[str, Any]:
    return {
        "id": identifier,
        "input": value,
        "metadata": {"_equiroute": {"source_line": line}},
    }


def _candidates(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (directory / "candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_labels_recorded_provider_cases_as_safe_review_candidates(
    tmp_path: Path,
) -> None:
    fixture_content = {
        name: json.loads((FIXTURES / name).read_text(encoding="utf-8"))["choices"][0][
            "message"
        ]["content"]
        for name in (
            "valid_response.json",
            "malformed_response.json",
            "unknown_route_response.json",
        )
    }
    rows = [_row(f"row-{index}", index) for index in range(1, 7)]
    config_path = _write_run_inputs(tmp_path, rows)
    adapter = FakeAdapter(
        {
            "row-1": _response("row-1", content=fixture_content["valid_response.json"]),
            "row-2": _response(
                "row-2", content=fixture_content["malformed_response.json"]
            ),
            "row-3": _response(
                "row-3", content=fixture_content["unknown_route_response.json"]
            ),
            "row-4": _response("row-4", status="refused", content=None),
            "row-5": _response("row-5", status="timed_out", content=None),
            "row-6": _response("row-6", status="rate_limited", content=None),
        }
    )

    manifest = label_sanitized_inputs(config_path, client=adapter)

    candidates = _candidates(tmp_path / "candidates")
    assert [candidate["status"] for candidate in candidates] == [
        "labeled",
        "rejected",
        "rejected",
        "rejected",
        "rejected",
        "rejected",
    ]
    assert [candidate.get("rejection_reason") for candidate in candidates] == [
        None,
        "malformed_response",
        "invalid_decision",
        "refusal",
        "timeout",
        "rate_limited",
    ]
    assert [candidate["provenance"]["source_line"] for candidate in candidates] == list(
        range(1, 7)
    )
    assert candidates[0]["decision"] == {"arguments": {}, "name": "billing_support"}
    assert manifest.output.rows == 6


def test_emits_large_number_and_rejects_unencodable_decision_strings(
    tmp_path: Path,
) -> None:
    config_path = _write_run_inputs(tmp_path, [_row("row-1", 1), _row("row-2", 2)])
    (tmp_path / "routes.yaml").write_text(
        """routes:
  - name: billing
    description: Billing support.
    parameters:
      type: object
      properties:
        amount:
          type: number
        note:
          type: string
      additionalProperties: false
""",
        encoding="utf-8",
    )
    large_integer = "1" + "0" * 400
    adapter = FakeAdapter(
        {
            "row-1": _response(
                "row-1",
                content=f'{{"name":"billing","arguments":{{"amount":{large_integer}}}}}',
            ),
            "row-2": _response(
                "row-2",
                content=r'{"name":"billing","arguments":{"note":"\ud800"}}',
            ),
        }
    )

    manifest = label_sanitized_inputs(config_path, client=adapter)

    candidates = _candidates(tmp_path / "candidates")
    assert candidates[0]["decision"] == {
        "arguments": {"amount": 10**400},
        "name": "billing",
    }
    assert candidates[1]["status"] == "rejected"
    assert candidates[1]["rejection_reason"] == "invalid_decision"
    assert r"\ud800" not in (tmp_path / "candidates" / "candidates.jsonl").read_text(
        encoding="utf-8"
    )
    assert manifest.output.rows == 2


def test_keeps_source_order_with_bounded_concurrency_without_limiting_local_work(
    tmp_path: Path,
) -> None:
    config_path = _write_run_inputs(
        tmp_path, [_row("row-1", 11), _row("row-2", 12), _row("row-3", 13)]
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["rate_limit_per_minute"] = 60
    config_path.write_text(json.dumps(config), encoding="utf-8")
    adapter = ConcurrentAdapter(
        {f"row-{index}": _response(f"row-{index}") for index in range(1, 4)}
    )
    clock = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    label_sanitized_inputs(
        config_path,
        client=adapter,
        sleep=sleep,
        monotonic=lambda: clock[0],
    )

    candidates = _candidates(tmp_path / "candidates")
    assert [candidate["provenance"]["source_id"] for candidate in candidates] == [
        "row-1",
        "row-2",
        "row-3",
    ]
    assert adapter.maximum_active == 2
    assert sleeps == []


def test_artifacts_exclude_policy_input_provider_content_and_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_secret = "RAW_INPUT_NEVER_EMITTED"
    provider_secret = "PROVIDER_RESPONSE_NEVER_EMITTED"
    credential = "CREDENTIAL_NEVER_EMITTED"
    monkeypatch.setenv("EQUIROUTE_TEST_SECRET", credential)
    config_path = _write_run_inputs(tmp_path, [_row("row-1", 42, "sanitized text")])
    adapter = FakeAdapter({"row-1": _response("row-1", content=provider_secret)})

    label_sanitized_inputs(config_path, client=adapter)

    artifact = "".join(
        file.read_text(encoding="utf-8") for file in (tmp_path / "candidates").iterdir()
    )
    assert raw_secret not in artifact
    assert provider_secret not in artifact
    assert credential not in artifact
    assert "POLICY_SECRET" not in artifact
    assert "sanitized text" not in artifact
    assert adapter.calls[0]["sanitized_input"] == "sanitized text"


def test_fails_preflight_without_calling_provider_or_creating_output(
    tmp_path: Path,
) -> None:
    config_path = _write_run_inputs(tmp_path, [_row("row-1", 1)])
    (tmp_path / "candidates").mkdir()
    adapter = FakeAdapter({"row-1": _response("row-1")})

    with pytest.raises(LabelingConfigError, match="refusing to overwrite"):
        label_sanitized_inputs(config_path, client=adapter)

    assert adapter.calls == []


def test_fails_output_preflight_before_calling_provider(tmp_path: Path) -> None:
    config_path = _write_run_inputs(tmp_path, [_row("row-1", 1)])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["output"] = {"directory": "missing/candidates"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    adapter = FakeAdapter({"row-1": _response("row-1")})

    with pytest.raises(FileNotFoundError):
        label_sanitized_inputs(config_path, client=adapter)

    assert adapter.calls == []
