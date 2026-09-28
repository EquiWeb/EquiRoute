from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import equiroute.training as training
from equiroute.training import TrainingError, export_router, train_router


class _Matrix:
    def __init__(self, rows: int, columns: int, fill: int) -> None:
        self.values = [[fill for _ in range(columns)] for _ in range(rows)]

    def __setitem__(self, key: tuple[int, slice], value: Any) -> None:
        row, column = key
        values = value.values if isinstance(value, _Vector) else value
        if isinstance(column, slice) and isinstance(values, int):
            start, stop, step = column.indices(len(self.values[row]))
            self.values[row][column] = [values] * len(range(start, stop, step))
        else:
            self.values[row][column] = values


class _Vector:
    def __init__(self, values: list[int]) -> None:
        self.values = values


class _FakeTorch:
    long = "long"
    float32 = "float32"
    bfloat16 = "bfloat16"
    float16 = "float16"
    cuda = SimpleNamespace(
        is_available=lambda: False,
        is_bf16_supported=lambda: False,
    )
    backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    utils = SimpleNamespace(data=SimpleNamespace(Dataset=object))

    @staticmethod
    def full(shape: tuple[int, int], value: int, *, dtype: str) -> _Matrix:
        return _Matrix(shape[0], shape[1], value)

    @staticmethod
    def zeros(shape: tuple[int, int], *, dtype: str) -> _Matrix:
        return _Matrix(shape[0], shape[1], 0)

    @staticmethod
    def tensor(values: list[int], *, dtype: str) -> _Vector:
        return _Vector(values)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.padding_side: str | None = None

    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        return {"input_ids": list(range(1, len(text) + 1))}

    def save_pretrained(self, directory: str | Path) -> None:
        Path(directory, "tokenizer.json").write_text("tokenizer", encoding="utf-8")


class _FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=True)

    def to(self, device: str) -> _FakeModel:
        return self

    def save_pretrained(self, directory: str | Path, **_: Any) -> None:
        Path(directory, "config.json").write_text("model", encoding="utf-8")

    def merge_and_unload(self) -> _FakeModel:
        return self


class _FakeTrainingArguments:
    def __init__(self, **values: Any) -> None:
        self.values = values


class _FakeTrainer:
    interrupted = False
    instances: list[_FakeTrainer] = []

    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)
        state_directory = Path(self.args.values["output_dir"])
        self.state = SimpleNamespace(
            best_model_checkpoint=str(state_directory / "checkpoint-1"),
            best_metric=0.125,
            global_step=1,
            epoch=1.0,
        )
        self.resume_from_checkpoint: str | None = None
        self.batch: dict[str, _Matrix] | None = None
        self.__class__.instances.append(self)

    def train(self, *, resume_from_checkpoint: str | None = None) -> None:
        self.resume_from_checkpoint = resume_from_checkpoint
        self.batch = self.data_collator([self.train_dataset[0], self.train_dataset[1]])
        checkpoint = Path(self.args.values["output_dir"], "checkpoint-1")
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "trainer_state.json").write_text("state", encoding="utf-8")
        if self.__class__.interrupted:
            raise RuntimeError("deliberate interruption")

    def evaluate(
        self, *, eval_dataset: Any, metric_key_prefix: str
    ) -> dict[str, float]:
        return {f"{metric_key_prefix}_loss": 0.125}

    def save_state(self) -> None:
        Path(self.args.values["output_dir"]).mkdir(parents=True, exist_ok=True)


class _FakePeft:
    class TaskType:
        CAUSAL_LM = "causal_lm"

    class LoraConfig:
        def __init__(self, **values: Any) -> None:
            self.values = values

    class PeftModel:
        @staticmethod
        def from_pretrained(model: _FakeModel, directory: str | Path) -> _FakeModel:
            return model

    @staticmethod
    def get_peft_model(model: _FakeModel, config: Any) -> _FakeModel:
        return model


def _fake_stack() -> Any:
    def tokenizer_from_pretrained(*_: Any, **__: Any) -> _FakeTokenizer:
        return _FakeTokenizer()

    def model_from_pretrained(*_: Any, **__: Any) -> _FakeModel:
        return _FakeModel()

    def last_checkpoint(directory: str) -> str | None:
        checkpoints = sorted(Path(directory).glob("checkpoint-*"))
        return str(checkpoints[-1]) if checkpoints else None

    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_from_pretrained),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_from_pretrained),
        TrainingArguments=_FakeTrainingArguments,
        Trainer=_FakeTrainer,
        set_seed=lambda _: None,
        trainer_utils=SimpleNamespace(get_last_checkpoint=last_checkpoint),
    )
    return training._TrainingStack(
        torch=_FakeTorch(),
        transformers=transformers,
        peft=_FakePeft(),
        accelerate=object(),
    )


def _write_training_project(tmp_path: Path) -> Path:
    (tmp_path / "routes.yaml").write_text(
        """routes:
  - name: balance_lookup
    description: Look up an account balance.
    parameters:
      type: object
      additionalProperties: false
  - name: card_replace
    description: Replace a payment card.
    parameters:
      type: object
      additionalProperties: false
  - name: transfer_status
    description: Check a bank transfer status.
    parameters:
      type: object
      additionalProperties: false
""",
        encoding="utf-8",
    )
    for partition in ("train", "validation", "test"):
        records = [
            {
                "id": f"{partition}-{route}",
                "input": f"{partition} request for {route.replace('_', ' ')}",
                "route": {"name": route},
            }
            for route in ("balance_lookup", "card_replace", "transfer_status")
        ]
        (tmp_path / f"{partition}.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
    config = tmp_path / "config.yaml"
    config.write_text(
        """model:
  base_model: google/functiongemma-270m-it
  revision: 39eccb091651513a5dfb56892d3714c1b5b8276c
routes: routes.yaml
data:
  train: train.jsonl
  validation: validation.jsonl
  test: test.jsonl
training:
  seed: 7
  epochs: 1
  learning_rate: 0.0002
  batch_size: 2
  gradient_accumulation_steps: 1
  lora_rank: 4
  lora_alpha: 8
  max_sequence_length: 4096
output:
  directory: artifact
  export: merged_huggingface
""",
        encoding="utf-8",
    )
    return config


def _install_fake_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeTrainer.instances.clear()
    _FakeTrainer.interrupted = False
    monkeypatch.setattr(training, "_load_training_stack", _fake_stack)


def test_train_prepares_completion_only_labels_and_writes_complete_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_training_project(tmp_path)
    _install_fake_stack(monkeypatch)

    artifact = train_router(config)

    assert artifact.status == "completed"
    assert Path(artifact.manifest_path).is_file()
    assert (Path(artifact.directory) / "equiroute" / "routes.yaml").is_file()
    assert (Path(artifact.directory) / "equiroute" / "run-config.yaml").is_file()
    assert (
        Path(artifact.directory) / "continuation" / "adapter" / "config.json"
    ).is_file()
    assert (Path(artifact.directory) / "model" / "config.json").is_file()
    persisted_evaluation = json.loads(
        (Path(artifact.directory) / "equiroute" / "evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted_evaluation["schema_version"] == "2"

    trainer = _FakeTrainer.instances[-1]
    records = trainer.train_dataset.records
    assert all(record["labels"] != record["input_ids"] for record in records)
    for record in records:
        first_completion = next(
            index for index, label in enumerate(record["labels"]) if label != -100
        )
        assert record["labels"][:first_completion] == [-100] * first_completion
        assert (
            record["labels"][first_completion:]
            == record["input_ids"][first_completion:]
        )
    assert trainer.batch is not None
    assert len(trainer.batch["input_ids"].values[0]) == max(
        len(record["input_ids"]) for record in records[:2]
    )

    assert export_router(artifact) == artifact


def test_interrupted_run_resumes_from_retained_epoch_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_training_project(tmp_path)
    _install_fake_stack(monkeypatch)
    _FakeTrainer.interrupted = True

    with pytest.raises(TrainingError, match="Training did not complete"):
        train_router(config)

    checkpoint = (
        tmp_path / "artifact" / "continuation" / "trainer-state" / "checkpoint-1"
    )
    assert checkpoint.is_dir()

    _FakeTrainer.interrupted = False
    artifact = train_router(config, resume=True)

    assert artifact.status == "completed"
    assert _FakeTrainer.instances[-1].resume_from_checkpoint == str(checkpoint)


def test_train_rejects_overflow_during_tokenizer_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_training_project(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "max_sequence_length: 4096", "max_sequence_length: 1"
        ),
        encoding="utf-8",
    )
    _install_fake_stack(monkeypatch)

    with pytest.raises(TrainingError, match="exceeding training.max_sequence_length"):
        train_router(config)

    assert not (tmp_path / "artifact").exists()


def test_new_run_refuses_existing_output_before_loading_optional_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_training_project(tmp_path)
    (tmp_path / "artifact").mkdir()

    def unexpected_stack_load() -> Any:
        raise AssertionError("optional stack should not be loaded")

    monkeypatch.setattr(training, "_load_training_stack", unexpected_stack_load)

    with pytest.raises(TrainingError, match="already exists"):
        train_router(config)


def test_train_seeds_before_initializing_the_pinned_lora_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_training_project(tmp_path)
    stack = _fake_stack()
    events: list[tuple[str, object]] = []
    stack.transformers.set_seed = lambda seed: events.append(("seed", seed))
    stack.peft.get_peft_model = lambda model, lora_config: (
        events.append(("lora", lora_config.values["revision"])),
        model,
    )[1]
    monkeypatch.setattr(training, "_load_training_stack", lambda: stack)

    train_router(config)

    assert events[:2] == [
        ("seed", 7),
        ("lora", "39eccb091651513a5dfb56892d3714c1b5b8276c"),
    ]
