"""Command-line interface for EquiRoute's local contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .errors import EquiRouteError
from .io import load_examples, load_route_registry, load_training_config

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def init() -> None:
    """Describe the Stage 0 project initialization boundary."""
    typer.echo("Stage 0 does not generate project files; author a registry, JSONL data, and config.")


@app.command()
def validate(config: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Validate a training configuration, route registry, and declared JSONL data."""
    try:
        training_config = load_training_config(config)
        base_directory = config.parent
        registry = load_route_registry(base_directory / training_config.routes)
        for dataset in (
            training_config.data.train,
            training_config.data.validation,
            training_config.data.test,
        ):
            load_examples(base_directory / dataset, registry)
    except EquiRouteError as error:
        typer.echo(f"Validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Validated {config}")


def _unavailable(stage: str) -> None:
    typer.echo(f"This command is not available until {stage}.", err=True)
    raise typer.Exit(code=2)


@app.command()
def split(
    data: Annotated[Path, typer.Argument()],
    routes: Annotated[Path, typer.Option()],
    seed: Annotated[int, typer.Option()] = 42,
) -> None:
    """Create deterministic data splits (available in Stage 1)."""
    _unavailable("Stage 1")


@app.command()
def train(config: Annotated[Path, typer.Argument()]) -> None:
    """Train a router (available in Stage 3)."""
    _unavailable("Stage 3")


@app.command()
def evaluate(
    artifact: Annotated[Path, typer.Argument()],
    data: Annotated[Path, typer.Option()],
) -> None:
    """Evaluate a trained router (available in Stage 4)."""
    _unavailable("Stage 4")


@app.command(name="continue")
def continue_training(
    from_artifact: Annotated[Path, typer.Option("--from")],
    config: Annotated[Path, typer.Option()],
) -> None:
    """Continue adapter training (available in Stage 5)."""
    _unavailable("Stage 5")


@app.command()
def export(
    artifact: Annotated[Path, typer.Argument()],
    format: Annotated[str, typer.Option()] = "huggingface",
) -> None:
    """Export a trained router (available in Stage 3)."""
    _unavailable("Stage 3")
