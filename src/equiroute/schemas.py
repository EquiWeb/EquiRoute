"""Strict data contracts for EquiRoute's local input and artifact formats."""

from __future__ import annotations

from typing import Any, Literal

from .model import (
    FUNCTIONGEMMA_LORA_BIAS,
    FUNCTIONGEMMA_LORA_DROPOUT,
    FUNCTIONGEMMA_LORA_TARGET_MODULES,
    FUNCTIONGEMMA_MODEL_ID,
    FUNCTIONGEMMA_REVISION,
    FUNCTIONGEMMA_TEMPLATE_ID,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that rejects coercion and undeclared fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


class PrimitiveArgumentSchema(StrictModel):
    type: Literal["string", "integer", "number", "boolean"]


class ObjectArgumentSchema(StrictModel):
    type: Literal["object"]
    properties: dict[str, PrimitiveArgumentSchema] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    additionalProperties: Literal[False]

    @model_validator(mode="after")
    def validate_members(self) -> ObjectArgumentSchema:
        empty_names = [name for name in self.properties if not name]
        if empty_names:
            raise ValueError("properties must not contain an empty name")

        seen_required: set[str] = set()
        duplicate_required: set[str] = set()
        for name in self.required:
            if name in seen_required:
                duplicate_required.add(name)
            seen_required.add(name)
        if duplicate_required:
            raise ValueError(
                "required contains duplicate names: "
                + ", ".join(sorted(duplicate_required))
            )

        missing_properties = sorted(set(self.required) - self.properties.keys())
        if missing_properties:
            raise ValueError(
                "required names must be declared in properties: "
                + ", ".join(missing_properties)
            )
        return self


class Route(StrictModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: ObjectArgumentSchema


class RouteRegistry(StrictModel):
    routes: list[Route] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_route_names(self) -> RouteRegistry:
        seen_names: set[str] = set()
        duplicate_names: set[str] = set()
        for route in self.routes:
            if route.name in seen_names:
                duplicate_names.add(route.name)
            seen_names.add(route.name)
        if duplicate_names:
            raise ValueError(
                "route names must be unique: " + ", ".join(sorted(duplicate_names))
            )
        return self

    def route_named(self, name: str) -> Route | None:
        return next((route for route in self.routes if route.name == name), None)


class Decision(StrictModel):
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class Example(StrictModel):
    id: str | None = None
    input: str = Field(min_length=1)
    route: Decision
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(StrictModel):
    base_model: Literal[FUNCTIONGEMMA_MODEL_ID]
    revision: Literal[FUNCTIONGEMMA_REVISION]


class DataConfig(StrictModel):
    train: str = Field(min_length=1)
    validation: str = Field(min_length=1)
    test: str = Field(min_length=1)


class TrainingOptions(StrictModel):
    seed: int
    epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    lora_rank: int = Field(ge=1)
    lora_alpha: int = Field(ge=1)
    max_sequence_length: int = Field(ge=1)


class OutputConfig(StrictModel):
    directory: str = Field(min_length=1)
    export: Literal["merged_huggingface"]


class TrainingConfig(StrictModel):
    model: ModelConfig
    routes: str = Field(min_length=1)
    data: DataConfig
    training: TrainingOptions
    output: OutputConfig


class RouteDistribution(StrictModel):
    """Counts for one registry route across Stage 1 dataset partitions."""

    name: str = Field(min_length=1)
    total: int = Field(ge=0)
    train: int = Field(ge=0)
    validation: int = Field(ge=0)
    test: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> RouteDistribution:
        if self.total != self.train + self.validation + self.test:
            raise ValueError("total must equal train + validation + test")
        return self


class DatasetArtifact(StrictModel):
    """One validated or emitted dataset file."""

    examples: int = Field(ge=0)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetReport(StrictModel):
    """Stage 1 dataset size and route-distribution report."""

    schema_version: Literal["1"]
    example_count: int = Field(ge=0)
    route_distribution: list[RouteDistribution] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_example_count(self) -> DatasetReport:
        if self.example_count != sum(
            distribution.total for distribution in self.route_distribution
        ):
            raise ValueError("example_count must equal the route distribution total")
        return self


class DatasetManifest(StrictModel):
    """Stage 1 dataset provenance, separate from the future training manifest."""

    schema_version: Literal["1"]
    registry_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_fingerprint: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    split_seed: int | None = None
    split_ratios: dict[str, float] | None = None
    datasets: dict[str, DatasetArtifact] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_split_metadata(self) -> DatasetManifest:
        if (self.split_seed is None) != (self.split_ratios is None):
            raise ValueError("split_seed and split_ratios must be provided together")
        if self.split_ratios is not None and self.split_ratios != {
            "train": 0.8,
            "validation": 0.1,
            "test": 0.1,
        }:
            raise ValueError("split_ratios must be the fixed Stage 1 80/10/10 ratios")
        if any(not name for name in self.datasets):
            raise ValueError("datasets must not contain an empty name")
        return self


class Manifest(StrictModel):
    """Vocabulary for a future reproducibility manifest; it has no behavior."""

    schema_version: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    base_model_revision: str = Field(min_length=1)
    tokenizer_template: str = Field(min_length=1)
    route_registry_fingerprint: str = Field(min_length=1)
    dataset_fingerprints: dict[str, str]
    resolved_config: dict[str, Any]
    software: dict[str, Any]
    hardware: dict[str, Any]
    evaluation: dict[str, Any]
    parent_artifact: str | None = None


class EvaluationReport(StrictModel):
    """Vocabulary for a future evaluation report; it has no behavior."""

    schema_version: str = Field(min_length=1)
    validation: dict[str, Any] | None = None
    test: dict[str, Any] | None = None


class TrainingInput(StrictModel):
    """Fingerprints for one raw dataset partition and its compiled strings."""

    examples: int = Field(ge=0)
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiled_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrainingInputProvenance(StrictModel):
    """The exact Stage-1 inputs consumed by a training run."""

    route_registry_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    train: TrainingInput
    validation: TrainingInput
    test: TrainingInput


class ResolvedLoRAConfig(StrictModel):
    """The non-configurable and resolved LoRA settings for FunctionGemma."""

    rank: int = Field(ge=1)
    alpha: int = Field(ge=1)
    target_modules: list[str] = Field(
        default_factory=lambda: list(FUNCTIONGEMMA_LORA_TARGET_MODULES)
    )
    dropout: float = Field(
        default=FUNCTIONGEMMA_LORA_DROPOUT, ge=0, le=1
    )
    bias: Literal["none"] = FUNCTIONGEMMA_LORA_BIAS

    @model_validator(mode="after")
    def validate_functiongemma_targets(self) -> ResolvedLoRAConfig:
        if self.target_modules != list(FUNCTIONGEMMA_LORA_TARGET_MODULES):
            raise ValueError("target_modules must be the FunctionGemma q_proj/v_proj pair")
        if self.dropout != FUNCTIONGEMMA_LORA_DROPOUT:
            raise ValueError("dropout must be the fixed FunctionGemma LoRA value")
        return self


class CheckpointPolicy(StrictModel):
    """Fixed checkpoint cadence and selection policy for Stage 3."""

    evaluation_strategy: Literal["epoch"]
    save_strategy: Literal["epoch"]
    metric_for_best_model: Literal["eval_loss"]
    greater_is_better: Literal[False]
    load_best_model_at_end: Literal[True]


class ResolvedTrainingConfig(StrictModel):
    """Complete model, compiler, optimizer, LoRA, and checkpoint inputs."""

    model: ModelConfig
    template_id: Literal[FUNCTIONGEMMA_TEMPLATE_ID]
    template_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    training: TrainingOptions
    lora: ResolvedLoRAConfig
    checkpoints: CheckpointPolicy

    @model_validator(mode="after")
    def validate_lora_matches_training(self) -> ResolvedTrainingConfig:
        if self.lora.rank != self.training.lora_rank:
            raise ValueError("LoRA rank must match the resolved training configuration")
        if self.lora.alpha != self.training.lora_alpha:
            raise ValueError("LoRA alpha must match the resolved training configuration")
        return self


class TrainingHardware(StrictModel):
    """Device and precision actually selected for a particular run."""

    device: Literal["cuda", "mps", "cpu"]
    dtype: Literal["bfloat16", "float16", "float32"]
    mixed_precision: Literal["bf16", "fp16", "no"]

    @model_validator(mode="after")
    def validate_precision_policy(self) -> TrainingHardware:
        expected = {
            "cuda": {
                ("bfloat16", "bf16"),
                ("float16", "fp16"),
            },
            "mps": {("float32", "no")},
            "cpu": {("float32", "no")},
        }
        if (self.dtype, self.mixed_precision) not in expected[self.device]:
            raise ValueError("dtype and mixed_precision do not support the selected device")
        return self


class CheckpointSelection(StrictModel):
    """Best validation checkpoint selected after epoch evaluation."""

    metric: Literal["eval_loss"]
    value: float
    path: str = Field(min_length=1)
    global_step: int = Field(ge=0)
    epoch: float = Field(ge=0)


class ArtifactFile(StrictModel):
    """One content-addressed file retained in a completed artifact."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArtifactHashes(StrictModel):
    """Hashes for the portable model and retained continuation adapter."""

    merged_model: list[ArtifactFile] = Field(min_length=1)
    adapter: list[ArtifactFile] = Field(min_length=1)


class PartitionEvaluation(StrictModel):
    """Evaluation metadata recorded for one fixed dataset partition."""

    examples: int = Field(ge=0)
    loss: float


class TrainingEvaluation(StrictModel):
    """Validation selects; sealed test evaluation is recorded afterwards."""

    validation: PartitionEvaluation
    test: PartitionEvaluation | None = None
    test_used_for_selection: Literal[False] = False


class TrainingManifest(StrictModel):
    """Strict Stage-3 provenance for a resumable FunctionGemma training run."""

    schema_version: Literal["1"]
    status: Literal["running", "completed"]
    inputs: TrainingInputProvenance
    resolved_config: ResolvedTrainingConfig
    hardware: TrainingHardware
    checkpoint_selection: CheckpointSelection | None = None
    evaluation: TrainingEvaluation | None = None
    artifacts: ArtifactHashes | None = None

    @model_validator(mode="after")
    def validate_completion_details(self) -> TrainingManifest:
        if self.status == "completed" and (
            self.checkpoint_selection is None
            or self.evaluation is None
            or self.artifacts is None
        ):
            raise ValueError(
                "completed training manifests require checkpoint, evaluation, and hashes"
            )
        return self


class TrainingArtifact(StrictModel):
    """JSON-friendly handle returned by a completed or resumable training run."""

    directory: str = Field(min_length=1)
    status: Literal["running", "completed"]
    manifest_path: str = Field(min_length=1)
