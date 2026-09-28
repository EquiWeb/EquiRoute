"""Starter-project creation from packaged EquiRoute templates."""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path


class InitError(Exception):
    """Raised when a starter project cannot be created safely."""


def create_starter_project(destination: str | Path) -> Path:
    """Copy the packaged starter project to a new, previously absent directory."""
    target = Path(destination)
    try:
        target.mkdir()
    except FileExistsError as error:
        raise InitError(
            f"Destination {target} already exists; refusing to overwrite it."
        ) from error
    except OSError as error:
        raise InitError(f"Could not create destination {target}: {error}") from error

    template = resources.files("equiroute").joinpath("templates", "starter")
    try:
        with resources.as_file(template) as source:
            shutil.copytree(source, target, dirs_exist_ok=True)
    except Exception as error:
        try:
            shutil.rmtree(target)
        except OSError as cleanup_error:
            raise InitError(
                f"Could not copy starter project to {target}: {error}; "
                f"also could not remove the partial destination: {cleanup_error}"
            ) from error
        raise InitError(
            f"Could not copy starter project to {target}: {error}"
        ) from error

    return target
