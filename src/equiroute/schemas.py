"""Strict data contracts for EquiRoute's local input and artifact formats."""

from __future__ import annotations

from typing import Any, Literal

from .model import FUNCTIONGEMMA_MODEL_ID, FUNCTIONGEMMA_REVISION

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
