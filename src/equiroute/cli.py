"""Command-line interface for EquiRoute's local contracts."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Annotated

import typer

from .dataset import (
    split_dataset,
    validate_continuation_regression,
    validate_partitions,
    write_split,
)
from .errors import (
    EquiRouteError,
    RawIngestionConfigError,
    RawInputLoadError,
    SourceError,
)
from .acceptance import accept_approved_labels
from .evaluation import EvaluationError, evaluate_artifact
from .init import InitError, create_starter_project
from .io import load_route_registry, load_training_config
from .labeling import label_sanitized_inputs
from .raw_input import ingest_raw_inputs
from .review import review_label_candidates
from .training import TrainingError, continue_router, export_router, train_router

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def init(target: Annotated[Path, typer.Argument()]) -> None:
    """Create a self-contained local router tutorial."""
    try:
        project = create_starter_project(target)
    except InitError as error:
        typer.echo(f"Initialization failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo("Next commands:")
    typer.echo(f"  cd {shlex.quote(str(project))}")
    typer.echo("  uv sync")
    typer.echo("  See README.md for the complete workflow.")


@app.command()
def validate(
    config: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Validate local training inputs and continuation replay separation.

    Parent artifact identity and preserved old-route coverage are verified only
    by ``equiroute continue --from``.
    """
    try:
        training_config = load_training_config(config)
        base_directory = config.parent
        registry = load_route_registry(base_directory / training_config.routes)
        partitions = {
            "train": base_directory / training_config.data.train,
            "validation": base_directory / training_config.data.validation,
            "test": base_directory / training_config.data.test,
        }
        report = validate_partitions(partitions, registry)
        if training_config.continuation is not None:
            regression = base_directory / training_config.continuation.regression
            validate_continuation_regression(regression, partitions, registry)
    except EquiRouteError as error:
        typer.echo(f"Validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(_canonical_json(report.model_dump(mode="json")))
    if training_config.continuation is not None:
        typer.echo(
            "Continuation regression data was validated locally; parent artifact "
            "identity and preserved-route coverage are checked by "
            "`equiroute continue --from`.",
            err=True,
        )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _safe_ingestion_error(error: SourceError) -> str:
    """Render structured ingestion failures without exception payloads."""

    reason = error.message
    for unsafe_prefix, safe_reason in (
        ("could not decode UTF-8:", "could not decode UTF-8"),
        ("malformed JSON:", "malformed JSON"),
        ("could not read raw inputs:", "could not read raw inputs"),
        (
            "could not read raw ingestion configuration:",
            "could not read raw ingestion configuration",
        ),
        ("malformed YAML:", "malformed YAML"),
    ):
        if reason.startswith(unsafe_prefix):
            reason = safe_reason
            break

    location = error.source
    if error.line is not None:
        location += f":{error.line}"
    if error.path:
        location += f": {error.path}"
    rendered = f"{location}: {reason}"
    if error.correction:
        rendered += f"; correction: {error.correction}"
    return rendered


def _safe_labeling_error(error: SourceError) -> str:
    """Render candidate-label setup failures without configuration contents."""

    reason = error.message
    for unsafe_prefix, safe_reason in (
        ("could not decode UTF-8:", "could not decode UTF-8"),
        ("malformed YAML:", "malformed YAML"),
        ("invalid labeling configuration:", "invalid labeling configuration"),
    ):
        if reason.startswith(unsafe_prefix):
            reason = safe_reason
            break

    location = error.source
    if error.line is not None:
        location += f":{error.line}"
    if error.path:
        location += f": {error.path}"
    rendered = f"{location}: {reason}"
    if error.correction:
        rendered += f"; correction: {error.correction}"
    return rendered


def _safe_stage9_error(error: EquiRouteError) -> str:
    """Render review/acceptance errors without artifact or configuration contents."""

    if not isinstance(error, SourceError):
        return "local contract validation failed"

    reason = error.message
    for unsafe_prefix, safe_reason in (
        ("could not decode UTF-8:", "could not decode UTF-8"),
        ("malformed JSON:", "malformed JSON"),
        ("malformed YAML:", "malformed YAML"),
        ("could not read", "could not read the local artifact"),
        ("invalid review configuration:", "invalid review configuration"),
        ("invalid acceptance configuration:", "invalid acceptance configuration"),
    ):
        if reason.startswith(unsafe_prefix):
            reason = safe_reason
            break

    location = error.source
    if error.line is not None:
        location += f":{error.line}"
    if error.path:
        location += f": {error.path}"
    rendered = f"{location}: {reason}"
    if error.correction:
        rendered += f"; correction: {error.correction}"
    return rendered


@app.command(name="review-labels")
def review_labels(
    config: Annotated[Path, typer.Argument()],
) -> None:
    """Create an editable Stage-9 review report from verified candidates."""

    try:
        manifest = review_label_candidates(config)
    except EquiRouteError as error:
        typer.echo(f"Review failed: {_safe_stage9_error(error)}", err=True)
        raise typer.Exit(code=1) from error
    except OSError as error:
        typer.echo(
            "Review failed: could not create the review output; "
            "correction: ensure the output parent exists and is writable",
            err=True,
        )
        raise typer.Exit(code=1) from error

    typer.echo(_canonical_json(manifest.model_dump(mode="json")))


@app.command(name="accept-labels")
def accept_labels(
    config: Annotated[Path, typer.Argument()],
) -> None:
    """Compile explicitly approved Stage-9 candidates into training JSONL."""

    try:
        manifest = accept_approved_labels(config)
    except EquiRouteError as error:
        typer.echo(f"Acceptance failed: {_safe_stage9_error(error)}", err=True)
        raise typer.Exit(code=1) from error
    except OSError as error:
        typer.echo(
            "Acceptance failed: could not create the acceptance output; "
            "correction: ensure the output parent exists and is writable",
            err=True,
        )
        raise typer.Exit(code=1) from error

    typer.echo(_canonical_json(manifest.model_dump(mode="json")))


@app.command()
def label(
    config: Annotated[Path, typer.Argument()],
) -> None:
    """Create review-only candidates from a verified sanitized Stage-7 handoff."""

    try:
        manifest = label_sanitized_inputs(config)
    except SourceError as error:
        typer.echo(f"Labeling failed: {_safe_labeling_error(error)}", err=True)
        raise typer.Exit(code=1) from error
    except OSError as error:
        typer.echo(
            "Labeling failed: could not create the candidate output; "
            "correction: ensure the output parent exists and is writable",
            err=True,
        )
        raise typer.Exit(code=1) from error

    typer.echo(_canonical_json(manifest.model_dump(mode="json")))


@app.command()
def ingest(
    config: Annotated[Path, typer.Argument()],
    debug: Annotated[
        bool,
        typer.Option(
            help="Write local provenance counts and hashes to standard error."
        ),
    ] = False,
) -> None:
    """Project and redact local raw JSONL into canonical unlabeled rows."""

    try:
        manifest = ingest_raw_inputs(config)
    except (RawIngestionConfigError, RawInputLoadError) as error:
        typer.echo(f"Ingestion failed: {_safe_ingestion_error(error)}", err=True)
        raise typer.Exit(code=1) from error
    except OSError as error:
        typer.echo(
            "Ingestion failed: "
            f"{config}: output.directory: could not write canonical output; "
            "correction: ensure the output parent exists and is writable",
            err=True,
        )
        raise typer.Exit(code=1) from error

    typer.echo(_canonical_json(manifest.model_dump(mode="json")))
    if debug:
        typer.echo(
            f"Ingestion debug: config={config} "
            f"source_rows={manifest.source.rows} "
            f"source_sha256={manifest.source.sha256} "
            f"output_rows={manifest.output.rows} "
            f"output_sha256={manifest.output.sha256} "
            f"redaction_count={manifest.redaction_count} "
            f"max_input_bytes={manifest.max_input_bytes}",
            err=True,
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
    """Continue adapter training from a completed artifact."""
    try:
        continue_router(from_artifact, config)
    except TrainingError as error:
        typer.echo(f"Continuation failed: {error}", err=True)
        raise typer.Exit(code=1) from error


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
