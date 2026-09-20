"""Actionable errors raised while reading local EquiRoute inputs."""

from __future__ import annotations

from pathlib import Path


class EquiRouteError(Exception):
    """Base error for invalid EquiRoute inputs."""


class SourceError(EquiRouteError):
    """An input error annotated with its local source and optional line or path."""

    def __init__(
        self,
        message: str,
        *,
        source: str | Path,
        line: int | None = None,
        path: str | None = None,
    ) -> None:
        self.message = message
        self.source = str(source)
        self.line = line
        self.path = path

        location = self.source
        if line is not None:
            location += f":{line}"
        if path:
            location += f": {path}"
        super().__init__(f"{location}: {message}")


class RegistryLoadError(SourceError):
    """A route registry could not be read or validated."""


class ConfigLoadError(SourceError):
    """A training configuration could not be read or validated."""


class ExampleLoadError(SourceError):
    """A JSONL example could not be read or validated."""
