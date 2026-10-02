from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


SECRET = "provider-response-never-emitted"


def _live_smoke_module() -> ModuleType:
    script = Path(__file__).parents[1] / "scripts" / "openrouter_live_smoke.py"
    specification = importlib.util.spec_from_file_location(
        "live_smoke_test_module", script
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_live_smoke_fails_when_the_only_candidate_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    smoke = _live_smoke_module()
    credential_name = "EQUIROUTE_LIVE_SMOKE_TEST_KEY"
    output = tmp_path / "candidates"
    config = SimpleNamespace(
        provider=SimpleNamespace(credential_env_var=credential_name)
    )
    handoff = SimpleNamespace(manifest=SimpleNamespace(output=SimpleNamespace(rows=1)))
    monkeypatch.setenv(credential_name, "credential")
    monkeypatch.setattr(smoke, "load_labeling_config", lambda _: config)
    monkeypatch.setattr(
        smoke,
        "resolve_labeling_paths",
        lambda *_: (tmp_path / "sanitized", tmp_path / "routes.yaml", output),
    )
    monkeypatch.setattr(smoke, "load_sanitized_handoff", lambda _: handoff)

    def label(_: Path) -> SimpleNamespace:
        output.mkdir()
        (output / "candidates.jsonl").write_text(
            '{"status":"rejected","provider_response":"' + SECRET + '"}\n',
            encoding="utf-8",
        )
        return SimpleNamespace(model_dump=lambda **_: {"output": {"rows": 1}})

    monkeypatch.setattr(smoke, "label_sanitized_inputs", label)

    assert smoke.main(["labeling.yaml"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        captured.err
        == "Live smoke failed: the provider did not return a locally valid "
        "candidate label.\n"
    )
    assert SECRET not in captured.err
