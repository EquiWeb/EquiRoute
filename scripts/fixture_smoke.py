"""Exercise the installed CLI's model-free starter-project workflow."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

import yaml


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=False,
        text=True,
        capture_output=True,
    )


def _require_success(result: subprocess.CompletedProcess[str]) -> str:
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {result.args}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def _tree_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _generated_configs(directory: Path) -> dict[str, Path]:
    configs: dict[str, Path] = {}
    for candidate in sorted(directory.rglob("*.yaml")):
        document = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            continue
        required = {"model", "routes", "data", "training", "evaluation", "output"}
        if not required.issubset(document):
            continue
        role = "child" if "continuation" in document else "parent"
        if role in configs:
            raise RuntimeError(f"starter project has multiple {role} configurations")
        configs[role] = candidate

    if set(configs) != {"parent", "child"}:
        found = ", ".join(str(path.relative_to(directory)) for path in configs.values())
        raise RuntimeError(
            "starter project must contain one parent and one continuation child configuration; "
            f"found: {found or 'none'}"
        )
    return configs


def _validate_deterministically(executable: str, config: Path) -> None:
    first = _require_success(_run(executable, "validate", str(config)))
    second = _require_success(_run(executable, "validate", str(config)))
    if first != second:
        raise RuntimeError(f"validation output is not deterministic for {config}")

    try:
        report = json.loads(first)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"validation output is not JSON for {config}: {first!r}"
        ) from error
    canonical = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if first != f"{canonical}\n":
        raise RuntimeError(f"validation output is not canonical JSON for {config}")


def main() -> None:
    executable = shutil.which("equiroute")
    if executable is None:
        raise RuntimeError("the installed equiroute executable is not on PATH")

    with tempfile.TemporaryDirectory(
        prefix="equiroute-fixture-"
    ) as temporary_directory:
        target = Path(temporary_directory) / "starter"
        _require_success(_run(executable, "init", str(target)))
        before_refusal = _tree_digest(target)

        refusal = _run(executable, "init", str(target))
        if refusal.returncode == 0:
            raise RuntimeError("init accepted an existing destination")
        if "exist" not in f"{refusal.stdout}\n{refusal.stderr}".lower():
            raise RuntimeError(
                "init refusal did not explain that the destination already exists"
            )
        if _tree_digest(target) != before_refusal:
            raise RuntimeError("a refused init modified the existing destination")

        for config in _generated_configs(target).values():
            _validate_deterministically(executable, config)


if __name__ == "__main__":
    main()
