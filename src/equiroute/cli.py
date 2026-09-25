"""Command-line interface for EquiRoute's local contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .dataset import split_dataset, validate_partitions, write_split
from .evaluation import EvaluationError, evaluate_artifact
from .errors import EquiRouteError
from .io import load_route_registry, load_training_config
from .training import TrainingError, export_router, train_router

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
        report = validate_partitions(
            {
                "train": base_directory / training_config.data.train,
                "validation": base_directory / training_config.data.validation,
                "test": base_directory / training_config.data.test,
            },
            registry,
        )
    except EquiRouteError as error:
        typer.echo(f"Validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(_canonical_json(report.model_dump(mode="json")))


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _unavailable(stage: str) -> None:
    typer.echo(f"This command is not available until {stage}.", err=True)
    raise typer.Exit(code=2)


@app.command()
def split(
    data: Annotated[Path, typer.Argument(exists=True, readable=True)],
    routes: Annotated[Path, typer.Option(exists=True, readable=True)],
    seed: Annotated[int, typer.Option()] = 42,
    output_directory: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Create deterministic data splits."""
    output_directory = output_directory or data.with_name(f"{data.stem}-splits")
    try:
        registry = load_route_registry(routes)
        result = split_dataset(data, registry, seed=seed)
        write_split(result, output_directory)
    except EquiRouteError as error:
        typer.echo(f"Split failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Wrote splits to {output_directory}")


@app.command()
def train(
    config: Annotated[Path, typer.Argument()],
    resume: Annotated[bool, typer.Option()] = False,
) -> None:
    """Train a FunctionGemma router."""
    try:
        train_router(config, resume=resume)
    except TrainingError as error:
        typer.echo(f"Training failed: {error}", err=True)
        raise typer.Exit(code=1) from error


@app.command()
def evaluate(
    artifact: Annotated[Path, typer.Argument()],
    data: Annotated[Path, typer.Option()],
) -> None:
    """Evaluate a trained router against semantic quality gates."""
    try:
        report = evaluate_artifact(artifact, data)
    except EvaluationError as error:
        typer.echo(f"Evaluation failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(_canonical_json(report.model_dump(mode="json")))
    if not report.passed:
        raise typer.Exit(code=1)


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
    """Export a trained router as Hugging Face files."""
    if format != "huggingface":
        typer.echo("Export failed: only --format huggingface is supported.", err=True)
        raise typer.Exit(code=2)
    try:
        export_router(artifact)
    except TrainingError as error:
        typer.echo(f"Export failed: {error}", err=True)
        raise typer.Exit(code=1) from error
