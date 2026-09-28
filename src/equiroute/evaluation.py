"""Semantic evaluation and promotion gates for completed FunctionGemma artifacts.

The scorer is deliberately model-free: it consumes raw completions, parses them
through the Stage-2 contract, and compares only validated decision semantics.
Model loading is isolated to :func:`evaluate_artifact` and remains lazy.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .dataset import _file_fingerprint, _registry_fingerprint
from .decisions import DecisionValidationError, validate_decision
from .errors import EquiRouteError
from .functiongemma import render_functiongemma_prompt
from .hardware import select_training_capability
from .io import load_examples, load_route_registry, load_training_config
from .output import InvalidOutput, parse_functiongemma_completion
from .schemas import (
    ComparativeEvaluation,
    ConfusionRow,
    ContinuationConfig,
    DatasetArtifact,
    Example,
    EvaluationConfig,
    EvaluationMetrics,
    EvaluationReport,
    InvalidOutputCount,
    RepresentativeError,
    RouteRegistry,
    RouteMetrics,
    ThresholdResult,
    TrainingArtifact,
    TrainingConfig,
    TrainingManifest,
)
from .training import (
    TrainingError,
    _read_manifest,
    _verify_existing_export,
    _write_json,
)

_SCHEMA_VERSION = "2"
_MODEL_DIRECTORY = "model"
_REPORT_PATH = Path("equiroute") / "semantic-evaluation.json"
_INVALID_CATEGORIES = (
    "missing_function_call",
    "malformed_function_call",
    "invalid_argument_syntax",
    "unknown_route",
    "invalid_arguments",
)


class EvaluationError(EquiRouteError):
    """An actionable failure while preparing or running semantic evaluation."""


def score_completions(
    examples: Iterable[Example],
    raw_completions: Iterable[str],
    registry: RouteRegistry,
    config: EvaluationConfig,
    *,
    artifact: str,
    data: DatasetArtifact,
) -> EvaluationReport:
    """Score raw model completions against validated gold decisions.

    Every metric uses the number of gold examples as its denominator.  A valid
    completion must select the gold route before its arguments can count as
    correct, so malformed and unknown-route outputs contribute zero to every
    correctness metric.
    """

    gold_examples = list(examples)
    completions = list(raw_completions)
    if not gold_examples:
        raise EvaluationError(
            "Evaluation data is empty; provide at least one validated example."
        )
    if len(gold_examples) != len(completions):
        raise EvaluationError(
            "Evaluation example and raw completion counts differ: "
            f"{len(gold_examples)} examples, {len(completions)} completions."
        )
    if data.examples != len(gold_examples):
        raise EvaluationError(
            "Evaluation data provenance does not match scored examples: "
            f"report records {data.examples}, received {len(gold_examples)}."
        )
    if not isinstance(artifact, str) or not artifact:
        raise EvaluationError(
            "Evaluation artifact provenance must be a non-empty string."
        )

    for index, example in enumerate(gold_examples, start=1):
        try:
            validate_decision(example.route, registry)
        except (AttributeError, DecisionValidationError) as error:
            raise EvaluationError(
                f"Gold evaluation example {index} does not contain a decision valid for the registry: {error}"
            ) from error
    for index, raw in enumerate(completions, start=1):
        if not isinstance(raw, str):
            raise EvaluationError(f"Raw completion {index} must be a string.")

    route_names = [route.name for route in registry.routes]
    support = {name: 0 for name in route_names}
    predictions = {name: 0 for name in route_names}
    true_positives = {name: 0 for name in route_names}
    confusion = {
        name: {predicted: 0 for predicted in route_names} for name in route_names
    }
    invalid_counts = {category: 0 for category in _INVALID_CATEGORIES}
    representatives: list[RepresentativeError] = []
    seen_failures: set[tuple[str, str | None, str | None]] = set()

    valid_decisions = 0
    route_correct = 0
    argument_correct = 0

    for example, raw in zip(gold_examples, completions, strict=True):
        gold = example.route
        support[gold.name] += 1
        parsed = parse_functiongemma_completion(raw, registry)

        if isinstance(parsed, InvalidOutput):
            invalid_counts[parsed.category] += 1
            _append_representative(
                representatives,
                seen_failures,
                example=example,
                raw=raw,
                predicted_route=None,
                invalid=parsed,
                redact=config.redact,
            )
            continue

        valid_decisions += 1
        predictions[parsed.name] += 1
        confusion[gold.name][parsed.name] += 1
        if parsed.name == gold.name:
            route_correct += 1
            true_positives[gold.name] += 1
            if parsed.arguments == gold.arguments:
                argument_correct += 1
                continue

        _append_representative(
            representatives,
            seen_failures,
            example=example,
            raw=raw,
            predicted_route=parsed.name,
            invalid=None,
            redact=config.redact,
        )

    total = len(gold_examples)
    metrics = EvaluationMetrics(
        examples=total,
        valid_decisions=valid_decisions,
        valid_decision_rate=valid_decisions / total,
        route_correct=route_correct,
        route_accuracy=route_correct / total,
        argument_correct=argument_correct,
        argument_accuracy=argument_correct / total,
    )
    route_metrics = [
        RouteMetrics(
            name=name,
            support=support[name],
            predictions=predictions[name],
            true_positives=true_positives[name],
            precision=(true_positives[name] / predictions[name])
            if predictions[name]
            else None,
            recall=(true_positives[name] / support[name]) if support[name] else None,
        )
        for name in route_names
    ]
    confusion_rows = [
        ConfusionRow(
            expected=expected,
            predicted={
                predicted: count
                for predicted, count in confusion[expected].items()
                if count
            },
        )
        for expected in route_names
    ]
    invalid_outputs = [
        InvalidOutputCount(category=category, count=invalid_counts[category])
        for category in _INVALID_CATEGORIES
    ]
    thresholds = _threshold_results(config, metrics)

    return EvaluationReport(
        schema_version=_SCHEMA_VERSION,
        artifact=artifact,
        model=_MODEL_DIRECTORY,
        registry_fingerprint=_registry_fingerprint(registry),
        data=data,
        config=config,
        metrics=metrics,
        routes=route_metrics,
        confusion_matrix=confusion_rows,
        invalid_outputs=invalid_outputs,
        representative_errors=representatives,
        thresholds=thresholds,
        passed=all(result.passed for result in thresholds),
    )


def evaluate_artifact(
    artifact: TrainingArtifact | str | Path, data: str | Path
) -> EvaluationReport:
    """Generate and semantically score one dataset with a completed artifact.

    Only the artifact's copied registry and frozen run configuration are used.
    The optional torch/transformers imports happen after all local provenance
    checks and input validation have completed.
    """

    directory = _artifact_directory(artifact)
    manifest = _completed_manifest(directory)
    registry = _artifact_registry(directory)
    _artifact_config(directory)
    _verify_registry_fingerprint(directory, manifest, registry)
    source = _evaluation_data_path(data)
    examples = _load_evaluation_examples(source, registry)
    try:
        provenance = DatasetArtifact(
            examples=len(examples),
            fingerprint=_file_fingerprint(source, "evaluation data"),
        )
    except EquiRouteError as error:
        raise EvaluationError(
            f"Could not fingerprint validated evaluation data {source}: {error}"
        ) from error
    model_directory = directory / _MODEL_DIRECTORY
    if not model_directory.is_dir():
        raise EvaluationError(
            f"Completed artifact {directory} has no exported model directory {model_directory}."
        )
    try:
        _verify_existing_export(model_directory, manifest)
    except TrainingError as error:
        raise EvaluationError(
            f"Exported model {model_directory} does not match artifact provenance: {error}"
        ) from error

    return _evaluate_artifact_inputs(
        model_directory,
        registry=registry,
        examples=examples,
        data=provenance,
        config=manifest.resolved_config.evaluation,
        artifact=str(directory),
        persist_path=directory / _REPORT_PATH,
    )


def _evaluate_artifact_inputs(
    model_directory: Path,
    *,
    registry: RouteRegistry,
    examples: Sequence[Example],
    data: DatasetArtifact,
    config: EvaluationConfig,
    artifact: str,
    prompt_registry: RouteRegistry | None = None,
    persist_path: Path | None = None,
) -> EvaluationReport:
    """Evaluate one verified export with prevalidated input evidence.

    ``registry`` governs completion parsing and report metrics.  A continuation
    evaluates its child against the larger child registry but renders every
    regression prompt from the parent registry via ``prompt_registry``.
    """

    prompt_registry = registry if prompt_registry is None else prompt_registry
    try:
        prompts = [
            render_functiongemma_prompt(example.input, prompt_registry)
            for example in examples
        ]
    except Exception as error:
        raise EvaluationError(
            f"Could not render evaluation prompts: {error}"
        ) from error
    raw_completions = _generate_completions(model_directory, prompts, config)
    report = score_completions(
        examples,
        raw_completions,
        registry,
        config,
        artifact=artifact,
        data=data,
    )
    if persist_path is not None:
        _write_report(persist_path, report)
    return report


def evaluate_loaded_artifact(
    model: Any,
    tokenizer: Any,
    *,
    torch: Any,
    device: str,
    scoring_registry: RouteRegistry,
    prompt_registry: RouteRegistry,
    examples: Sequence[Example],
    data: DatasetArtifact,
    config: EvaluationConfig,
    artifact: str,
    persist_path: Path | None = None,
) -> EvaluationReport:
    """Evaluate a continuation model already loaded by the training stack.

    Callers validate artifact lineage and hashes before loading the model.
    ``prompt_registry`` stays fixed to the parent registry for fair regression
    replay, while ``scoring_registry`` records the evaluated artifact's full
    route inventory.
    """

    try:
        prompts = [
            render_functiongemma_prompt(example.input, prompt_registry)
            for example in examples
        ]
    except Exception as error:
        raise EvaluationError(
            f"Could not render evaluation prompts: {error}"
        ) from error
    raw_completions = _generate_loaded_completions(
        model, tokenizer, torch, device, prompts, config
    )
    report = score_completions(
        examples,
        raw_completions,
        scoring_registry,
        config,
        artifact=artifact,
        data=data,
    )
    if persist_path is not None:
        _write_report(persist_path, report)
    return report


def compare_continuation_evaluations(
    parent: EvaluationReport,
    child: EvaluationReport,
    config: ContinuationConfig,
) -> ComparativeEvaluation:
    """Create strict regression evidence from two identically sourced reports."""

    route_accuracy_drop = parent.metrics.route_accuracy - child.metrics.route_accuracy
    argument_accuracy_drop = (
        parent.metrics.argument_accuracy - child.metrics.argument_accuracy
    )
    return ComparativeEvaluation(
        schema_version="2",
        regression_data=parent.data,
        parent=parent,
        child=child,
        old_route_names=[route.name for route in parent.routes],
        max_route_accuracy_drop=config.max_route_accuracy_drop,
        route_accuracy_drop=route_accuracy_drop,
        max_argument_accuracy_drop=config.max_argument_accuracy_drop,
        argument_accuracy_drop=argument_accuracy_drop,
        passed=(
            parent.passed
            and child.passed
            and route_accuracy_drop <= config.max_route_accuracy_drop
            and argument_accuracy_drop <= config.max_argument_accuracy_drop
        ),
    )


def _append_representative(
    representatives: list[RepresentativeError],
    seen_failures: set[tuple[str, str | None, str | None]],
    *,
    example: Example,
    raw: str,
    predicted_route: str | None,
    invalid: InvalidOutput | None,
    redact: bool,
) -> None:
    """Keep the first representative for each semantic failure signature."""

    category = invalid.category if invalid is not None else None
    signature = (example.route.name, predicted_route, category)
    if signature in seen_failures:
        return
    seen_failures.add(signature)
    representatives.append(
        RepresentativeError(
            expected_route=example.route.name,
            predicted_route=predicted_route,
            invalid_category=category,
            input=None if redact else example.input,
            raw_output=None if redact else raw,
            detail=None if redact or invalid is None else invalid.detail,
        )
    )


def _threshold_results(
    config: EvaluationConfig, metrics: EvaluationMetrics
) -> list[ThresholdResult]:
    return [
        ThresholdResult(
            name=name,
            minimum=minimum,
            actual=getattr(metrics, name),
            passed=getattr(metrics, name) >= minimum,
        )
        for name, minimum in (
            ("valid_decision_rate", config.thresholds.valid_decision_rate),
            ("route_accuracy", config.thresholds.route_accuracy),
            ("argument_accuracy", config.thresholds.argument_accuracy),
        )
    ]


def _artifact_directory(artifact: TrainingArtifact | str | Path) -> Path:
    if isinstance(artifact, TrainingArtifact):
        directory = Path(artifact.directory)
    elif isinstance(artifact, (str, Path)):
        directory = Path(artifact)
    else:
        raise EvaluationError(
            "Evaluation requires the TrainingArtifact returned by train_router or its artifact directory."
        )
    if not directory.is_dir():
        raise EvaluationError(
            f"Artifact directory {directory} does not exist or is not a directory."
        )
    return directory.resolve()


def _completed_manifest(directory: Path) -> TrainingManifest:
    manifest_path = directory / "equiroute" / "manifest.json"
    try:
        manifest = _read_manifest(manifest_path)
    except TrainingError as error:
        raise EvaluationError(
            f"Could not read completed artifact manifest {manifest_path}: {error}"
        ) from error
    if manifest.status != "completed":
        raise EvaluationError(
            f"Artifact {directory} is still running and cannot be evaluated until training completes."
        )
    return manifest


def _artifact_registry(directory: Path) -> RouteRegistry:
    source = directory / "equiroute" / "routes.yaml"
    try:
        return load_route_registry(source)
    except EquiRouteError as error:
        raise EvaluationError(
            f"Could not load artifact route registry {source}: {error}"
        ) from error


def _artifact_config(directory: Path) -> TrainingConfig:
    source = directory / "equiroute" / "run-config.yaml"
    try:
        return load_training_config(source)
    except EquiRouteError as error:
        raise EvaluationError(
            f"Could not load artifact run configuration {source}: {error}"
        ) from error


def _verify_registry_fingerprint(
    directory: Path, manifest: TrainingManifest, registry: RouteRegistry
) -> None:
    actual = _registry_fingerprint(registry)
    expected = manifest.inputs.route_registry_fingerprint
    if actual != expected:
        raise EvaluationError(
            f"Artifact {directory} route registry fingerprint {actual} does not match completed "
            f"manifest fingerprint {expected}."
        )


def _evaluation_data_path(data: str | Path) -> Path:
    try:
        source = Path(data)
    except TypeError as error:
        raise EvaluationError(
            "Evaluation data path must be a string or pathlib.Path."
        ) from error
    if not source.is_file():
        raise EvaluationError(
            f"Evaluation data file {source} does not exist or is not a file."
        )
    return source.resolve()


def _load_evaluation_examples(source: Path, registry: RouteRegistry) -> list[Example]:
    try:
        examples = load_examples(source, registry)
    except EquiRouteError as error:
        raise EvaluationError(
            f"Could not read validated evaluation data {source}: {error}"
        ) from error
    if not examples:
        raise EvaluationError(
            f"Evaluation data {source} is empty; provide at least one validated example."
        )
    return examples


def _generate_completions(
    model_directory: Path, prompts: Sequence[str], config: EvaluationConfig
) -> list[str]:
    """Generate greedy completions from only the exported artifact ``model/``.

    This intentionally private seam lets tests replace model execution without
    importing optional model packages or downloading model weights.
    """

    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_directory), local_files_only=True
        )
        capability = select_training_capability(torch)
        dtype = getattr(torch, capability.dtype)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_directory), torch_dtype=dtype, local_files_only=True
        )
        model.to(capability.device)
        model.eval()
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as error:
        raise EvaluationError(
            "Could not load exported model for semantic evaluation. Install EquiRoute with "
            f"the model extra and ensure {model_directory} is a valid local Transformers export: {error}"
        ) from error

    return _generate_loaded_completions(
        model, tokenizer, torch, capability.device, prompts, config
    )


def _generate_loaded_completions(
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: str,
    prompts: Sequence[str],
    config: EvaluationConfig,
) -> list[str]:
    """Generate deterministic completions from a model prepared on ``device``."""

    completions: list[str] = []
    try:
        model.eval()
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            encoded = _move_encoded_inputs(encoded, device)
            prompt_length = _sequence_length(encoded["input_ids"])
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=config.max_new_tokens,
                )
            completions.append(
                tokenizer.decode(
                    generated[0][prompt_length:], skip_special_tokens=False
                )
            )
    except EvaluationError:
        raise
    except Exception as error:
        raise EvaluationError(
            f"Could not generate evaluation completions: {error}"
        ) from error
    return completions


def _move_encoded_inputs(encoded: Mapping[str, Any], device: str) -> Mapping[str, Any]:
    move = getattr(encoded, "to", None)
    if callable(move):
        return move(device)
    return {
        name: value.to(device) if callable(getattr(value, "to", None)) else value
        for name, value in encoded.items()
    }


def _sequence_length(input_ids: Any) -> int:
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    return len(input_ids[0])


def _write_report(path: Path, report: EvaluationReport) -> None:
    """Persist the latest report using the Stage-3 atomic canonical JSON writer."""

    try:
        _write_json(path, report.model_dump(mode="json"))
    except TrainingError as error:
        raise EvaluationError(
            f"Could not write semantic evaluation report {path}: {error}"
        ) from error
