"""Local, resumable FunctionGemma LoRA training and portable export.

Optional ML packages are imported only when a train or export operation actually
needs them.  Importing this module remains safe for configuration tooling and
normal test environments.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import _file_fingerprint, _registry_fingerprint, validate_partitions
from .errors import EquiRouteError
from .functiongemma import compile_functiongemma
from .hardware import TrainingCapability, select_training_capability
from .io import iter_examples, load_route_registry, load_training_config
from .model import (
    FUNCTIONGEMMA_LORA_BIAS,
    FUNCTIONGEMMA_LORA_DROPOUT,
    FUNCTIONGEMMA_LORA_TARGET_MODULES,
    FUNCTIONGEMMA_MODEL_ID,
    FUNCTIONGEMMA_REVISION,
    FUNCTIONGEMMA_TEMPLATE_ID,
)
from .schemas import (
    ArtifactFile,
    ArtifactHashes,
    CheckpointPolicy,
    CheckpointSelection,
    PartitionEvaluation,
    ResolvedLoRAConfig,
    ResolvedTrainingConfig,
    TrainingArtifact,
    TrainingEvaluation,
    TrainingHardware,
    TrainingInput,
    TrainingInputProvenance,
    TrainingManifest,
)


class TrainingError(RuntimeError):
    """An actionable failure while preparing, training, resuming, or exporting."""


@dataclass(frozen=True, slots=True)
class _TrainingStack:
    torch: Any
    transformers: Any
    peft: Any
    accelerate: Any


@dataclass(frozen=True, slots=True)
class _PreparedPartition:
    name: str
    source: Path
    source_fingerprint: str
    compiled_fingerprint: str
    records: list[dict[str, list[int]]]


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    partitions: Mapping[str, _PreparedPartition]
    manifest: TrainingManifest


_MARKER = "<start_function_call>"
_IGNORED_LABEL = -100
_SCHEMA_VERSION = "1"


def train_router(config_path: str | Path, *, resume: bool = False) -> TrainingArtifact:
    """Train and export the configured FunctionGemma router.

    A new run refuses to reuse an existing output directory.  ``resume=True``
    accepts only an interrupted EquiRoute run with a retained epoch checkpoint
    and exactly matching inputs, configuration, and hardware policy.
    """

    config_source = _as_path(config_path, "training configuration")
    config, registry, sources, output_directory = _load_local_inputs(config_source)

    if resume:
        _require_resume_directory(output_directory)
    elif output_directory.exists():
        raise TrainingError(
            f"Output directory {output_directory} already exists. "
            "Choose a new output.directory or resume its interrupted run with --resume."
        )

    stack = _load_training_stack()
    stack.transformers.set_seed(config.training.seed)
    capability = select_training_capability(stack.torch)
    tokenizer = _load_tokenizer(stack)
    prepared = _prepare_run(config, registry, sources, tokenizer, capability)

    if resume:
        previous = _read_manifest(_manifest_path(output_directory))
        _verify_resume_manifest(previous, prepared.manifest)
        checkpoint = _find_resume_checkpoint(stack, _trainer_state_directory(output_directory))
    else:
        checkpoint = None

    model = _load_base_model(stack, capability)
    adapter_model = _apply_lora(stack, model, config)

    if not resume:
        _initialize_artifact(output_directory, config_source, sources["routes"], prepared.manifest)

    try:
        trainer = _build_trainer(
            stack,
            adapter_model,
            tokenizer,
            prepared.partitions,
            config,
            capability,
            output_directory,
        )
        trainer.train(resume_from_checkpoint=checkpoint)

        validation_metrics = trainer.evaluate(
            eval_dataset=_dataset_for(stack.torch, prepared.partitions["validation"].records),
            metric_key_prefix="eval",
        )
        test_metrics = trainer.evaluate(
            eval_dataset=_dataset_for(stack.torch, prepared.partitions["test"].records),
            metric_key_prefix="test",
        )
        trainer.save_state()

        adapter_directory = _adapter_directory(output_directory)
        adapter_directory.mkdir(parents=True, exist_ok=True)
        adapter_model.save_pretrained(adapter_directory, safe_serialization=True)

        selection = _checkpoint_selection(trainer, output_directory)
        evaluation = TrainingEvaluation(
            validation=PartitionEvaluation(
                examples=len(prepared.partitions["validation"].records),
                loss=_required_loss(validation_metrics, "eval_loss"),
            ),
            test=PartitionEvaluation(
                examples=len(prepared.partitions["test"].records),
                loss=_required_loss(test_metrics, "test_loss"),
            ),
        )

        _write_json(_evaluation_path(output_directory), evaluation.model_dump(mode="json"))
        _save_merged_model(adapter_model, tokenizer, output_directory)
        completed = prepared.manifest.model_copy(
            update={
                "status": "completed",
                "checkpoint_selection": selection,
                "evaluation": evaluation,
                "artifacts": _artifact_hashes(output_directory),
            }
        )
        _write_manifest(_manifest_path(output_directory), completed)
    except TrainingError:
        raise
    except Exception as error:
        raise TrainingError(
            "Training did not complete. Its retained epoch checkpoints can be resumed "
            f"with `equiroute train {config_source} --resume` after fixing the error: {error}"
        ) from error

    return _artifact_for(output_directory, completed)


def export_router(artifact: TrainingArtifact | str | Path) -> TrainingArtifact:
    """Ensure a completed artifact has its merged standard Hugging Face model.

    Training normally performs the configured export before returning.  This
    entry point is intentionally idempotent for that completed artifact and can
    rebuild a missing ``model/`` directory from the retained adapter.
    """

    output_directory = _artifact_directory(artifact)
    manifest_path = _manifest_path(output_directory)
    manifest = _read_manifest(manifest_path)
    if manifest.status != "completed":
        raise TrainingError(
            f"Artifact {output_directory} is still running and cannot be exported. "
            "Resume or complete its training run first."
        )

    model_directory = _model_directory(output_directory)
    if model_directory.exists():
        _verify_existing_export(model_directory, manifest)
        return _artifact_for(output_directory, manifest)

    adapter_directory = _adapter_directory(output_directory)
    if not adapter_directory.is_dir():
        raise TrainingError(
            f"Artifact {output_directory} has no retained adapter at {adapter_directory}. "
            "It cannot be exported."
        )

    _verify_retained_adapter(adapter_directory, manifest)

    stack = _load_training_stack()
    capability = select_training_capability(stack.torch)
    tokenizer = _load_tokenizer(stack)
    model = _load_base_model(stack, capability)
    try:
        adapter_model = stack.peft.PeftModel.from_pretrained(model, adapter_directory)
        _save_merged_model(adapter_model, tokenizer, output_directory)
    except TrainingError:
        raise
    except Exception as error:
        raise TrainingError(
            f"Could not merge retained adapter {adapter_directory} into {FUNCTIONGEMMA_MODEL_ID}. "
            f"Confirm the adapter was produced by this pinned model: {error}"
        ) from error

    completed = manifest.model_copy(
        update={
            "artifacts": ArtifactHashes(
                merged_model=_hash_tree(model_directory, output_directory),
                adapter=manifest.artifacts.adapter,
            )
        }
    )
    _write_manifest(manifest_path, completed)
    return _artifact_for(output_directory, completed)


def _load_local_inputs(
    config_source: Path,
) -> tuple[Any, Any, dict[str, Path], Path]:
    try:
        config = load_training_config(config_source)
        base_directory = config_source.parent
        sources = {
            "routes": _resolve_config_path(base_directory, config.routes),
            "train": _resolve_config_path(base_directory, config.data.train),
            "validation": _resolve_config_path(base_directory, config.data.validation),
            "test": _resolve_config_path(base_directory, config.data.test),
        }
        registry = load_route_registry(sources["routes"])
        validate_partitions(
            {name: sources[name] for name in ("train", "validation", "test")}, registry
        )
    except EquiRouteError as error:
        raise TrainingError(f"Cannot prepare validated training inputs: {error}") from error

    output_directory = _resolve_config_path(base_directory, config.output.directory)
    return config, registry, sources, output_directory


def _prepare_run(
    config: Any,
    registry: Any,
    sources: Mapping[str, Path],
    tokenizer: Any,
    capability: TrainingCapability,
) -> _PreparedRun:
    partitions = {
        name: _prepare_partition(
            name,
            sources[name],
            registry,
            tokenizer,
            config.training.max_sequence_length,
        )
        for name in ("train", "validation", "test")
    }
    manifest = TrainingManifest(
        schema_version=_SCHEMA_VERSION,
        status="running",
        inputs=TrainingInputProvenance(
            route_registry_fingerprint=_registry_fingerprint(registry),
            train=_training_input(partitions["train"]),
            validation=_training_input(partitions["validation"]),
            test=_training_input(partitions["test"]),
        ),
        resolved_config=ResolvedTrainingConfig(
            model=config.model,
            template_id=FUNCTIONGEMMA_TEMPLATE_ID,
            template_fingerprint=_template_fingerprint(),
            training=config.training,
            evaluation=config.evaluation,
            lora=ResolvedLoRAConfig(
                rank=config.training.lora_rank,
                alpha=config.training.lora_alpha,
                target_modules=list(FUNCTIONGEMMA_LORA_TARGET_MODULES),
                dropout=FUNCTIONGEMMA_LORA_DROPOUT,
                bias=FUNCTIONGEMMA_LORA_BIAS,
            ),
            checkpoints=CheckpointPolicy(
                evaluation_strategy="epoch",
                save_strategy="epoch",
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                load_best_model_at_end=True,
            ),
        ),
        hardware=TrainingHardware(
            device=capability.device,
            dtype=capability.dtype,
            mixed_precision=capability.mixed_precision,
        ),
    )
    return _PreparedRun(partitions=partitions, manifest=manifest)


def _prepare_partition(
    name: str,
    source: Path,
    registry: Any,
    tokenizer: Any,
    max_sequence_length: int,
) -> _PreparedPartition:
    records: list[dict[str, list[int]]] = []
    compiled_hasher = hashlib.sha256()
    try:
        source_fingerprint = _file_fingerprint(source, f"{name} dataset")
        for loaded in iter_examples(source, registry):
            try:
                compiled = compile_functiongemma(loaded.example, registry)
            except Exception as error:
                raise TrainingError(
                    f"Could not compile {name} example at {loaded.source}:{loaded.line}: {error}"
                ) from error

            input_ids, prefix_length = _completion_only_ids(
                tokenizer,
                compiled,
                max_sequence_length,
                partition=name,
                source=loaded.source,
                line=loaded.line,
            )
            labels = [_IGNORED_LABEL] * prefix_length + input_ids[prefix_length:]
            records.append({"input_ids": input_ids, "labels": labels})
            _update_compiled_fingerprint(compiled_hasher, compiled)
    except EquiRouteError as error:
        raise TrainingError(f"Cannot read {name} dataset: {error}") from error

    if not records:
        raise TrainingError(f"The validated {name} dataset {source} is empty.")
    return _PreparedPartition(
        name=name,
        source=source,
        source_fingerprint=source_fingerprint,
        compiled_fingerprint=compiled_hasher.hexdigest(),
        records=records,
    )


def _completion_only_ids(
    tokenizer: Any,
    compiled: str,
    max_sequence_length: int,
    *,
    partition: str,
    source: Path,
    line: int,
) -> tuple[list[int], int]:
    marker_count = compiled.count(_MARKER)
    if marker_count != 1:
        raise TrainingError(
            f"Compiled {partition} example at {source}:{line} has {marker_count} completion "
            "boundaries; expected exactly one FunctionGemma function call."
        )
    prefix = compiled[: compiled.index(_MARKER)]
    try:
        input_ids = _token_ids(tokenizer, compiled)
        prefix_ids = _token_ids(tokenizer, prefix)
    except Exception as error:
        if isinstance(error, TrainingError):
            raise
        raise TrainingError(
            f"Could not tokenize {partition} example at {source}:{line}: {error}"
        ) from error

    prefix_length = len(prefix_ids)
    if not prefix_ids or prefix_length >= len(input_ids):
        raise TrainingError(
            f"Tokenizer preflight failed for {partition} example at {source}:{line}: "
            "the FunctionGemma completion has no tokens after its prompt."
        )
    if input_ids[:prefix_length] != prefix_ids:
        raise TrainingError(
            f"Tokenizer preflight failed for {partition} example at {source}:{line}: "
            "separate prompt tokenization is not a prefix of the compiled conversation."
        )
    if len(input_ids) > max_sequence_length:
        raise TrainingError(
            f"Compiled {partition} example at {source}:{line} tokenizes to {len(input_ids)} "
            f"tokens, exceeding training.max_sequence_length={max_sequence_length}. "
            "Shorten the example or increase the configured sequence limit."
        )
    return input_ids, prefix_length


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None,
    )
    try:
        token_ids = encoded["input_ids"]
    except (KeyError, TypeError) as error:
        raise TrainingError("The FunctionGemma tokenizer did not return input_ids.") from error
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if not isinstance(token_ids, list) or any(
        not isinstance(token, int) or isinstance(token, bool) for token in token_ids
    ):
        raise TrainingError("The FunctionGemma tokenizer returned non-integer input_ids.")
    return token_ids


def _update_compiled_fingerprint(hasher: Any, compiled: str) -> None:
    encoded = compiled.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)


def _training_input(partition: _PreparedPartition) -> TrainingInput:
    return TrainingInput(
        examples=len(partition.records),
        source_fingerprint=partition.source_fingerprint,
        compiled_fingerprint=partition.compiled_fingerprint,
    )


def _load_training_stack() -> _TrainingStack:
    try:
        return _TrainingStack(
            torch=importlib.import_module("torch"),
            transformers=importlib.import_module("transformers"),
            peft=importlib.import_module("peft"),
            accelerate=importlib.import_module("accelerate"),
        )
    except ImportError as error:
        raise TrainingError(
            "Training and export require the optional torch, transformers, peft, and "
            "accelerate dependencies. Install them with `uv sync --extra model`."
        ) from error


def _load_tokenizer(stack: _TrainingStack) -> Any:
    try:
        tokenizer = stack.transformers.AutoTokenizer.from_pretrained(
            FUNCTIONGEMMA_MODEL_ID,
            revision=FUNCTIONGEMMA_REVISION,
        )
    except Exception as error:
        raise TrainingError(
            f"Could not download or load {FUNCTIONGEMMA_MODEL_ID} at revision "
            f"{FUNCTIONGEMMA_REVISION}. Confirm network access, Hugging Face credentials, "
            "and acceptance of the model's gated license."
        ) from error

    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise TrainingError(
                f"{FUNCTIONGEMMA_MODEL_ID} tokenizer has neither a pad token nor an EOS token."
            )
        tokenizer.pad_token = eos_token
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise TrainingError(
            f"{FUNCTIONGEMMA_MODEL_ID} tokenizer did not expose a usable pad token ID."
        )
    tokenizer.padding_side = "right"
    return tokenizer


def _load_base_model(stack: _TrainingStack, capability: TrainingCapability) -> Any:
    try:
        dtype = getattr(stack.torch, capability.dtype)
        model = stack.transformers.AutoModelForCausalLM.from_pretrained(
            FUNCTIONGEMMA_MODEL_ID,
            revision=FUNCTIONGEMMA_REVISION,
            torch_dtype=dtype,
        )
        model.to(capability.device)
    except Exception as error:
        raise TrainingError(
            f"Could not download or load {FUNCTIONGEMMA_MODEL_ID} at revision "
            f"{FUNCTIONGEMMA_REVISION} on {capability.device}. Confirm network access, "
            "Hugging Face credentials, acceptance of the model's gated license, and a "
            "PyTorch build that supports the selected device."
        ) from error

    model_config = getattr(model, "config", None)
    if model_config is not None:
        model_config.use_cache = False
    return model


def _apply_lora(stack: _TrainingStack, model: Any, config: Any) -> Any:
    try:
        task_type = stack.peft.TaskType.CAUSAL_LM
        lora_config = stack.peft.LoraConfig(
            r=config.training.lora_rank,
            lora_alpha=config.training.lora_alpha,
            target_modules=list(FUNCTIONGEMMA_LORA_TARGET_MODULES),
            lora_dropout=FUNCTIONGEMMA_LORA_DROPOUT,
            bias=FUNCTIONGEMMA_LORA_BIAS,
            task_type=task_type,
            revision=FUNCTIONGEMMA_REVISION,
            inference_mode=False,
        )
        return stack.peft.get_peft_model(model, lora_config)
    except Exception as error:
        raise TrainingError(
            "Could not configure the FunctionGemma q_proj/v_proj LoRA adapter: "
            f"{error}"
        ) from error


def _build_trainer(
    stack: _TrainingStack,
    model: Any,
    tokenizer: Any,
    partitions: Mapping[str, _PreparedPartition],
    config: Any,
    capability: TrainingCapability,
    output_directory: Path,
) -> Any:
    state_directory = _trainer_state_directory(output_directory)
    model_config = getattr(model, "config", None)
    if model_config is not None:
        model_config.pad_token_id = tokenizer.pad_token_id

    arguments = {
        "output_dir": str(state_directory),
        "num_train_epochs": config.training.epochs,
        "learning_rate": config.training.learning_rate,
        "per_device_train_batch_size": config.training.batch_size,
        "per_device_eval_batch_size": config.training.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 1,
        "seed": config.training.seed,
        "data_seed": config.training.seed,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "report_to": [],
        "fp16": capability.mixed_precision == "fp16",
        "bf16": capability.mixed_precision == "bf16",
    }
    if capability.device == "cpu":
        arguments["use_cpu"] = True

    try:
        training_arguments = stack.transformers.TrainingArguments(**arguments)
        return stack.transformers.Trainer(
            model=model,
            args=training_arguments,
            train_dataset=_dataset_for(stack.torch, partitions["train"].records),
            eval_dataset=_dataset_for(stack.torch, partitions["validation"].records),
            data_collator=_collator_for(stack.torch, tokenizer.pad_token_id),
        )
    except Exception as error:
        raise TrainingError(
            "Could not initialize the deterministic epoch-based LoRA trainer: " f"{error}"
        ) from error


def _dataset_for(torch_module: Any, records: Sequence[dict[str, list[int]]]) -> Any:
    dataset_base = getattr(getattr(getattr(torch_module, "utils", None), "data", None), "Dataset", None)
    if dataset_base is None:
        raise TrainingError("The installed torch package does not provide torch.utils.data.Dataset.")

    class CompiledCompletionDataset(dataset_base):
        def __init__(self, entries: Sequence[dict[str, list[int]]]) -> None:
            self.records = entries

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return self.records[index]

    return CompiledCompletionDataset(records)


def _collator_for(torch_module: Any, pad_token_id: int) -> Callable[[list[dict[str, list[int]]]], dict[str, Any]]:
    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        if not features:
            raise TrainingError("The trainer requested a batch with no training examples.")
        width = max(len(feature["input_ids"]) for feature in features)
        batch_size = len(features)
        input_ids = torch_module.full(
            (batch_size, width), pad_token_id, dtype=torch_module.long
        )
        attention_mask = torch_module.zeros(
            (batch_size, width), dtype=torch_module.long
        )
        labels = torch_module.full(
            (batch_size, width), _IGNORED_LABEL, dtype=torch_module.long
        )
        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch_module.tensor(
                feature["input_ids"], dtype=torch_module.long
            )
            attention_mask[row, :length] = 1
            labels[row, :length] = torch_module.tensor(feature["labels"], dtype=torch_module.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate


def _checkpoint_selection(trainer: Any, output_directory: Path) -> CheckpointSelection:
    state = getattr(trainer, "state", None)
    checkpoint = getattr(state, "best_model_checkpoint", None)
    metric = getattr(state, "best_metric", None)
    if not isinstance(checkpoint, str) or not checkpoint:
        raise TrainingError(
            "Epoch evaluation did not record a best checkpoint by eval_loss; no portable "
            "artifact will be emitted."
        )
    try:
        value = float(metric)
        global_step = int(getattr(state, "global_step"))
        epoch = float(getattr(state, "epoch"))
    except (TypeError, ValueError, AttributeError) as error:
        raise TrainingError(
            "The best checkpoint is missing its eval_loss, global step, or epoch metadata."
        ) from error

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = _trainer_state_directory(output_directory) / checkpoint_path
    try:
        relative_path = checkpoint_path.resolve().relative_to(output_directory.resolve())
    except ValueError as error:
        raise TrainingError(
            f"Trainer selected checkpoint {checkpoint_path}, which is outside {output_directory}."
        ) from error
    return CheckpointSelection(
        metric="eval_loss",
        value=value,
        path=relative_path.as_posix(),
        global_step=global_step,
        epoch=epoch,
    )


def _required_loss(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise TrainingError(
            f"Trainer evaluation did not produce required {key}; received {dict(metrics)}."
        ) from error


def _initialize_artifact(
    output_directory: Path,
    config_source: Path,
    routes_source: Path,
    manifest: TrainingManifest,
) -> None:
    try:
        output_directory.mkdir(parents=True, exist_ok=False)
        provenance = _provenance_directory(output_directory)
        provenance.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(config_source, provenance / "run-config.yaml")
        shutil.copyfile(routes_source, provenance / "routes.yaml")
        _write_manifest(_manifest_path(output_directory), manifest)
    except OSError as error:
        raise TrainingError(
            f"Could not create training artifact directory {output_directory}: {error}"
        ) from error


def _require_resume_directory(output_directory: Path) -> None:
    if not output_directory.is_dir():
        raise TrainingError(
            f"Cannot resume because output directory {output_directory} does not exist. "
            "Start a new run without --resume."
        )
    if not _manifest_path(output_directory).is_file():
        raise TrainingError(
            f"Cannot resume {output_directory}: its equiroute/manifest.json is missing."
        )


def _verify_resume_manifest(previous: TrainingManifest, expected: TrainingManifest) -> None:
    if previous.status != "running":
        raise TrainingError(
            "Cannot resume a completed artifact. Choose a new output.directory for another "
            "training run."
        )
    if previous.inputs != expected.inputs:
        raise TrainingError(
            "Cannot resume because the route registry or one of the validated dataset "
            "partitions has changed since the interrupted run."
        )
    if previous.resolved_config != expected.resolved_config:
        raise TrainingError(
            "Cannot resume because the resolved training configuration differs from the "
            "interrupted run. Restore its original configuration."
        )
    if previous.hardware != expected.hardware:
        raise TrainingError(
            "Cannot resume because the selected device or precision differs from the "
            "interrupted run. Use the original hardware policy."
        )


def _find_resume_checkpoint(stack: _TrainingStack, state_directory: Path) -> str:
    try:
        utility = getattr(stack.transformers, "trainer_utils")
        checkpoint = utility.get_last_checkpoint(str(state_directory))
    except Exception as error:
        raise TrainingError(
            f"Could not inspect resumable checkpoints in {state_directory}: {error}"
        ) from error
    if not checkpoint:
        raise TrainingError(
            f"Cannot resume {state_directory}: no epoch checkpoint was retained. "
            "Start a new run with a new output.directory."
        )
    return str(checkpoint)


def _save_merged_model(model: Any, tokenizer: Any, output_directory: Path) -> None:
    target = _model_directory(output_directory)
    if target.exists():
        raise TrainingError(
            f"Refusing to overwrite existing merged model directory {target}."
        )
    temporary = Path(
        tempfile.mkdtemp(prefix=".model.tmp-", dir=output_directory)
    )
    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        temporary.replace(target)
    except Exception as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise TrainingError(
            f"Could not export merged FunctionGemma model to {target}: {error}"
        ) from error


def _artifact_hashes(output_directory: Path) -> ArtifactHashes:
    return ArtifactHashes(
        merged_model=_hash_tree(_model_directory(output_directory), output_directory),
        adapter=_hash_tree(_adapter_directory(output_directory), output_directory),
    )


def _hash_tree(directory: Path, root: Path) -> list[ArtifactFile]:
    if not directory.is_dir():
        raise TrainingError(f"Expected retained artifact directory {directory} was not written.")
    files = [path for path in sorted(directory.rglob("*")) if path.is_file()]
    if not files:
        raise TrainingError(f"Expected retained artifact directory {directory} contains no files.")
    return [
        ArtifactFile(
            path=path.relative_to(root).as_posix(),
            sha256=_sha256_file(path),
        )
        for path in files
    ]


def _verify_existing_export(model_directory: Path, manifest: TrainingManifest) -> None:
    expected = manifest.artifacts
    if expected is None:
        raise TrainingError(
            f"Artifact {model_directory.parent} has a completed manifest without model hashes."
        )
    actual = _hash_tree(model_directory, model_directory.parent)
    if actual != expected.merged_model:
        raise TrainingError(
            f"Existing merged model directory {model_directory} does not match its artifact "
            "manifest. Refusing to overwrite it."
        )


def _verify_retained_adapter(adapter_directory: Path, manifest: TrainingManifest) -> None:
    expected = manifest.artifacts
    if expected is None:
        raise TrainingError(
            f"Artifact {adapter_directory.parent.parent} has a completed manifest without adapter hashes."
        )
    actual = _hash_tree(adapter_directory, adapter_directory.parent.parent)
    if actual != expected.adapter:
        raise TrainingError(
            f"Retained adapter directory {adapter_directory} does not match its artifact "
            "manifest. Refusing to export it."
        )

def _read_manifest(path: Path) -> TrainingManifest:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        return TrainingManifest.model_validate(document)
    except (OSError, ValueError, TypeError) as error:
        raise TrainingError(
            f"Could not read valid EquiRoute training manifest {path}: {error}"
        ) from error


def _write_manifest(path: Path, manifest: TrainingManifest) -> None:
    _write_json(path, manifest.model_dump(mode="json"))


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary:
            json.dump(
                document,
                temporary,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    except (OSError, TypeError, ValueError) as error:
        raise TrainingError(f"Could not write artifact provenance {path}: {error}") from error


def _artifact_for(output_directory: Path, manifest: TrainingManifest) -> TrainingArtifact:
    return TrainingArtifact(
        directory=str(output_directory),
        status=manifest.status,
        manifest_path=str(_manifest_path(output_directory)),
    )


def _artifact_directory(artifact: TrainingArtifact | str | Path) -> Path:
    if isinstance(artifact, TrainingArtifact):
        directory = Path(artifact.directory)
    elif isinstance(artifact, (str, Path)):
        directory = Path(artifact)
    else:
        raise TrainingError(
            "Export requires the TrainingArtifact returned by train_router or its artifact directory."
        )
    if not directory.is_dir():
        raise TrainingError(f"Artifact directory {directory} does not exist.")
    return directory.resolve()


def _as_path(value: str | Path, name: str) -> Path:
    try:
        path = Path(value)
    except TypeError as error:
        raise TrainingError(f"{name.capitalize()} path must be a string or pathlib.Path.") from error
    if not path.is_file():
        raise TrainingError(f"{name.capitalize()} file {path} does not exist or is not a file.")
    return path.resolve()


def _resolve_config_path(base_directory: Path, configured: str) -> Path:
    path = Path(configured)
    return path.resolve() if path.is_absolute() else (base_directory / path).resolve()


def _provenance_directory(output_directory: Path) -> Path:
    return output_directory / "equiroute"


def _manifest_path(output_directory: Path) -> Path:
    return _provenance_directory(output_directory) / "manifest.json"


def _evaluation_path(output_directory: Path) -> Path:
    return _provenance_directory(output_directory) / "evaluation.json"


def _trainer_state_directory(output_directory: Path) -> Path:
    return output_directory / "continuation" / "trainer-state"


def _adapter_directory(output_directory: Path) -> Path:
    return output_directory / "continuation" / "adapter"


def _model_directory(output_directory: Path) -> Path:
    return output_directory / "model"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _template_fingerprint() -> str:
    return hashlib.sha256(FUNCTIONGEMMA_TEMPLATE_ID.encode("utf-8")).hexdigest()
