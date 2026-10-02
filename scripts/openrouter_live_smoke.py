"""Manually label one reviewed sanitized handoff through the OpenRouter SDK.

This intentionally does not run in CI. Use a dedicated one-row Stage-7 artifact and
an output directory that does not already exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from equiroute.errors import EquiRouteError
from equiroute.io import (
    load_labeling_config,
    load_sanitized_handoff,
    resolve_labeling_paths,
)
from equiroute.labeling import label_sanitized_inputs


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a one-off sanitized OpenRouter candidate-labeling smoke."
    )
    parser.add_argument("config", type=Path, metavar="CONFIG")
    parsed = parser.parse_args(arguments)

    try:
        config = load_labeling_config(parsed.config)
    except (EquiRouteError, OSError):
        print(
            "Live smoke refused: could not load a valid labeling configuration.",
            file=sys.stderr,
        )
        return 2

    if not os.environ.get(config.provider.credential_env_var):
        print(
            "Live smoke refused: required credential environment variable "
            f"{config.provider.credential_env_var} is not set.",
            file=sys.stderr,
        )
        return 2

    try:
        handoff_directory, _, output_directory = resolve_labeling_paths(
            parsed.config, config
        )
        handoff = load_sanitized_handoff(handoff_directory)
    except (EquiRouteError, OSError):
        print(
            "Live smoke refused: verify the Stage-7 sanitized handoff before "
            "running a live provider request.",
            file=sys.stderr,
        )
        return 2
    if handoff.manifest.output.rows != 1:
        print(
            "Live smoke refused: use a dedicated Stage-7 handoff with exactly "
            "one sanitized row.",
            file=sys.stderr,
        )
        return 2

    try:
        manifest = label_sanitized_inputs(parsed.config)
        candidate_lines = (
            (output_directory / "candidates.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        if len(candidate_lines) != 1:
            raise ValueError("live smoke did not produce exactly one candidate")
        candidate = json.loads(candidate_lines[0])
        if not isinstance(candidate, dict) or candidate.get("status") != "labeled":
            raise ValueError("live smoke candidate was not labeled")
    except (EquiRouteError, OSError, ValueError, json.JSONDecodeError):
        print(
            "Live smoke failed: the provider did not return a locally valid "
            "candidate label.",
            file=sys.stderr,
        )
        return 1

    print(_canonical_json(manifest.model_dump(mode="json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
