from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import equiroute.evaluation as evaluation
import equiroute.training as training
from equiroute.evaluation import score_completions
from equiroute.training import TrainingError, continue_router, train_router


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
    cuda = SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)
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

    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        return {"input_ids": list(range(1, len(text) + 1))}

    def save_pretrained(self, directory: str | Path) -> None:
        Path(directory, "tokenizer.json").write_text("tokenizer", encoding="utf-8")


class _FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=True)

    def to(self, _: str) -> _FakeModel:
        return self

    def save_pretrained(self, directory: str | Path, **_: Any) -> None:
        Path(directory, "config.json").write_text("model", encoding="utf-8")

    def merge_and_unload(self) -> _FakeModel:
        return self


class _FakeTrainingArguments:
    def __init__(self, **values: Any) -> None:
        self.values = values


class _FakeTrainer:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)
        state_directory = Path(self.args.values["output_dir"])
        self.state = SimpleNamespace(
            best_model_checkpoint=str(state_directory / "checkpoint-1"),
            best_metric=0.125,
            global_step=1,
            epoch=1.0,
        )

    def train(self, *, resume_from_checkpoint: str | None = None) -> None:
        del resume_from_checkpoint
        self.data_collator([self.train_dataset[0], self.train_dataset[1]])
        checkpoint = Path(self.args.values["output_dir"], "checkpoint-1")
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "trainer_state.json").write_text("state", encoding="utf-8")

    def evaluate(self, *, eval_dataset: Any, metric_key_prefix: str) -> dict[str, float]:
        del eval_dataset
        return {f"{metric_key_prefix}_loss": 0.125}

    def save_state(self) -> None:
        Path(self.args.values["output_dir"]).mkdir(parents=True, exist_ok=True)


class _FakePeft:
    class TaskType:
        CAUSAL_LM = "causal_lm"

    class LoraConfig:
        def __init__(self, **values: Any) -> None:
            self.values = values

    adapter_loads: list[tuple[_FakeModel, Path, bool]] = []

    class PeftModel:
        @staticmethod
        def from_pretrained(
            model: _FakeModel,
            directory: str | Path,
            *,
            is_trainable: bool = False,
        ) -> _FakeModel:
            _FakePeft.adapter_loads.append((model, Path(directory), is_trainable))
            return model

    @staticmethod
    def get_peft_model(model: _FakeModel, _: Any) -> _FakeModel:
        return model


def _fake_stack() -> Any:
    def last_checkpoint(directory: str) -> str | None:
        checkpoints = sorted(Path(directory).glob("checkpoint-*"))
        return str(checkpoints[-1]) if checkpoints else None

    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: _FakeTokenizer()),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: _FakeModel()
        ),
        TrainingArguments=_FakeTrainingArguments,
        Trainer=_FakeTrainer,
        set_seed=lambda _: None,
        trainer_utils=SimpleNamespace(get_last_checkpoint=last_checkpoint),
    )
    return training._TrainingStack(
        torch=_FakeTorch(), transformers=transformers, peft=_FakePeft(), accelerate=object()
    )


def _write_registry(path: Path, names: list[str]) -> None:
    routes = "\n".join(
        f"""  - name: {name}
    description: Handle {name.replace('_', ' ')} requests.
    parameters:
      type: object
      additionalProperties: false"""
        for name in names
    )
    path.write_text(f"routes:\n{routes}\n", encoding="utf-8")


def _write_records(path: Path, names: list[str], prefix: str) -> None:
    records = [
        {
            "id": f"{prefix}-{name}",
            "input": f"{prefix} request for {name.replace('_', ' ')}",
            "route": {"name": name},
        }
        for name in names
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _write_project(
    directory: Path,
    names: list[str],
    *,
    output: str,
    continuation: bool,
    regression: str = "regression.jsonl",
    evaluation: str = "",
) -> Path:
    directory.mkdir()
    _write_registry(directory / "routes.yaml", names)
    for partition in ("train", "validation", "test"):
        _write_records(directory / f"{partition}.jsonl", names, f"{directory.name}-{partition}")
    if continuation:
        _write_records(directory / regression, names[:-1], f"{directory.name}-regression")
    continuation_yaml = (
        f"""continuation:
  regression: {regression}
  max_route_accuracy_drop: 0.0
  max_argument_accuracy_drop: 0.0
"""
        if continuation
        else ""
    )
    config = directory / "config.yaml"
    config.write_text(
        f"""model:
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
  directory: {output}
  export: merged_huggingface
{evaluation}{continuation_yaml}""",
        encoding="utf-8",
    )
    return config




def _parent_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    parent_config = _write_project(
        tmp_path / "parent-source", ["alpha", "beta"], output="parent-artifact", continuation=False
    )
    monkeypatch.setattr(training, "_load_training_stack", _fake_stack)
    return Path(train_router(parent_config).directory)


def test_continue_trains_from_verified_snapshot_with_child_evaluation_policy_without_parent_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakePeft.adapter_loads.clear()
    parent = _parent_artifact(tmp_path, monkeypatch)
    parent_manifest = parent / "equiroute" / "manifest.json"
    parent_manifest_bytes = parent_manifest.read_bytes()
    parent_adapter = parent / "continuation" / "adapter"
    parent_adapter_bytes = (parent_adapter / "config.json").read_bytes()
    shutil.rmtree(parent / "model")
    child_config = _write_project(
        tmp_path / "child-source",
        ["alpha", "beta", "gamma"],
        output="child-artifact",
        continuation=True,
        evaluation="""evaluation:
  redact: true
  max_new_tokens: 17
""",
    )
    calls: list[dict[str, Any]] = []

    def fake_loaded_evaluator(_: Any, __: Any, **values: Any) -> Any:
        calls.append(values)
        completions = (
            [
                f"<start_function_call>call:{example.route.name}{{}}<end_function_call>"
                for example in values["examples"]
            ]
            if values.get("persist_path") is not None
            else ["not a function call" for _ in values["examples"]]
        )
        report = score_completions(
            values["examples"],
            completions,
            values["scoring_registry"],
            values["config"],
            artifact=values["artifact"],
            data=values["data"],
        )
        if (persist_path := values.get("persist_path")) is not None:
            persist_path.write_text(
                json.dumps(report.model_dump(mode="json")), encoding="utf-8"
            )
        return report

    monkeypatch.setattr(evaluation, "evaluate_loaded_artifact", fake_loaded_evaluator)

    artifact = continue_router(parent, child_config)

    child = Path(artifact.directory)
    manifest = training._read_manifest(child / "equiroute" / "manifest.json")
    assert artifact.status == "completed"
    snapshot = _FakePeft.adapter_loads[0][1]
    assert [is_trainable for _, _, is_trainable in _FakePeft.adapter_loads] == [True, False]
    assert snapshot != parent_adapter
    assert _FakePeft.adapter_loads[1][1] == snapshot
    assert snapshot.parts[-2:] == ("continuation", "adapter")
    assert not snapshot.exists()
    assert len(calls) == 3
    assert [route.name for route in calls[0]["scoring_registry"].routes] == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert [route.name for route in calls[0]["prompt_registry"].routes] == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert [route.name for route in calls[1]["scoring_registry"].routes] == ["alpha", "beta"]
    assert [route.name for route in calls[1]["prompt_registry"].routes] == ["alpha", "beta"]
    assert [route.name for route in calls[2]["scoring_registry"].routes] == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert [route.name for route in calls[2]["prompt_registry"].routes] == ["alpha", "beta"]
    assert calls[1]["data"] == calls[2]["data"]
    assert calls[0]["config"] == calls[1]["config"] == calls[2]["config"]
    assert calls[0]["config"].redact is True
    assert calls[0]["config"].max_new_tokens == 17
    assert not (parent / "model").exists()
    assert parent_manifest.read_bytes() == parent_manifest_bytes
    assert (parent_adapter / "config.json").read_bytes() == parent_adapter_bytes
    assert manifest.parent is not None
    assert manifest.parent.manifest_sha256 == hashlib.sha256(parent_manifest_bytes).hexdigest()
    assert manifest.registry_change is not None
    assert manifest.registry_change.retained_route_names == ["alpha", "beta"]
    assert [route.name for route in manifest.registry_change.added_routes] == ["gamma"]
    comparison = manifest.comparative_evaluation
    assert comparison is not None
    assert comparison.parent.config == calls[1]["config"]
    assert comparison.child.config == calls[1]["config"]
    for report in (comparison.parent, comparison.child):
        assert report.representative_errors
        assert all(
            representative.input is None
            and representative.raw_output is None
            and representative.detail is None
            for representative in report.representative_errors
        )
    persisted = child / "equiroute" / "continuation-evaluation.json"
    assert json.loads(persisted.read_text(encoding="utf-8")) == comparison.model_dump(mode="json")
    semantic = json.loads(
        (child / "equiroute" / "semantic-evaluation.json").read_text(encoding="utf-8")
    )
    assert semantic["data"]["examples"] == 3
    assert semantic["routes"][-1] == {
        "name": "gamma",
        "support": 1,
        "predictions": 1,
        "true_positives": 1,
        "precision": 1.0,
        "recall": 1.0,
    }


def test_continuation_does_not_complete_when_held_out_added_route_recall_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _parent_artifact(tmp_path, monkeypatch)
    child_config = _write_project(
        tmp_path / "child-source",
        ["alpha", "beta", "gamma"],
        output="child-artifact",
        continuation=True,
    )

    def fake_loaded_evaluator(_: Any, __: Any, **values: Any) -> Any:
        report = score_completions(
            values["examples"],
            ["not a function call" for _ in values["examples"]],
            values["scoring_registry"],
            values["config"],
            artifact=values["artifact"],
            data=values["data"],
        )
        values["persist_path"].write_text(
            json.dumps(report.model_dump(mode="json")), encoding="utf-8"
        )
        return report

    monkeypatch.setattr(evaluation, "evaluate_loaded_artifact", fake_loaded_evaluator)

    with pytest.raises(TrainingError, match="added-route semantic"):
        continue_router(parent, child_config)

    child = child_config.parent / "child-artifact"
    manifest = training._read_manifest(child / "equiroute" / "manifest.json")
    assert manifest.status == "running"
    assert not (child / "equiroute" / "continuation-evaluation.json").exists()
    semantic = json.loads(
        (child / "equiroute" / "semantic-evaluation.json").read_text(encoding="utf-8")
    )
    assert semantic["routes"][-1]["name"] == "gamma"
    assert semantic["routes"][-1]["recall"] == 0.0



def test_continuation_rejects_tampered_adapter_snapshot_before_ml_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _parent_artifact(tmp_path, monkeypatch)
    child_config = _write_project(
        tmp_path / "child-source", ["alpha", "beta", "gamma"], output="child-artifact", continuation=True
    )
    copytree = training.shutil.copytree

    def corrupt_snapshot(source: Any, destination: Any, *args: Any, **kwargs: Any) -> Any:
        copied = copytree(source, destination, *args, **kwargs)
        Path(destination, "config.json").write_text("tampered", encoding="utf-8")
        return copied

    def unexpected_stack_load() -> Any:
        raise AssertionError("model stack must not load before the adapter snapshot is verified")

    monkeypatch.setattr(training.shutil, "copytree", corrupt_snapshot)
    monkeypatch.setattr(training, "_load_training_stack", unexpected_stack_load)
    with pytest.raises(TrainingError, match="does not match"):
        continue_router(parent, child_config)

    assert not (tmp_path / "child-source" / "child-artifact").exists()


@pytest.mark.parametrize(
    ("partition", "overlap"),
    [
        ("train", "id"),
        ("validation", "id"),
        ("test", "id"),
        ("train", "normalized_input"),
        ("validation", "normalized_input"),
        ("test", "normalized_input"),
    ],
)
def test_continuation_preflight_rejects_regression_content_overlap_from_each_child_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, partition: str, overlap: str
) -> None:
    parent = _parent_artifact(tmp_path, monkeypatch)
    child_config = _write_project(
        tmp_path / "child-source", ["alpha", "beta", "gamma"], output="child-artifact", continuation=True
    )
    child_record = json.loads(
        (child_config.parent / f"{partition}.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    regression_beta = json.loads(
        (child_config.parent / "regression.jsonl").read_text(encoding="utf-8").splitlines()[1]
    )
    regression_alpha = {
        "id": f"regression-{partition}-{overlap}",
        "input": f"independent regression request for alpha from {partition}",
        "route": {"name": "alpha"},
    }
    if overlap == "id":
        regression_alpha["id"] = child_record["id"]
        expected = "reuses example id"
    else:
        regression_alpha["input"] = (
            "\u00a0\uff23" + child_record["input"][1:].upper().replace(" ", "\t") + "  "
        )
        expected = "reuses an input"
    (child_config.parent / "regression.jsonl").write_text(
        "\n".join(json.dumps(record) for record in (regression_alpha, regression_beta)) + "\n",
        encoding="utf-8",
    )

    def unexpected_stack_load() -> Any:
        raise AssertionError("optional model stack must not load during failed preflight")

    monkeypatch.setattr(training, "_load_training_stack", unexpected_stack_load)
    with pytest.raises(TrainingError, match=expected):
        continue_router(parent, child_config)

    assert not (tmp_path / "child-source" / "child-artifact").exists()


@pytest.mark.parametrize(
    "failure",
    ["tampered_adapter", "reordered_routes", "partition_replay", "copied_replay"],
)
def test_continuation_preflight_rejects_invalid_parent_registry_or_replay_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    parent = _parent_artifact(tmp_path, monkeypatch)
    child_config = _write_project(
        tmp_path / "child-source", ["alpha", "beta", "gamma"], output="child-artifact", continuation=True
    )
    if failure == "tampered_adapter":
        (parent / "continuation" / "adapter" / "config.json").write_text("tampered", encoding="utf-8")
        expected = "does not match"
    elif failure == "reordered_routes":
        _write_registry(child_config.parent / "routes.yaml", ["beta", "alpha", "gamma"])
        expected = "exact ordered prefix"
    elif failure == "partition_replay":
        child_config.write_text(
            child_config.read_text(encoding="utf-8").replace(
                "regression: regression.jsonl", "regression: train.jsonl"
            ),
            encoding="utf-8",
        )
        expected = "must not alias"
    else:
        train_alpha = (child_config.parent / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        regression_beta = (child_config.parent / "regression.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()[1]
        (child_config.parent / "copied-regression.jsonl").write_text(
            train_alpha + "\n" + regression_beta + "\n",
            encoding="utf-8",
        )
        child_config.write_text(
            child_config.read_text(encoding="utf-8").replace(
                "regression: regression.jsonl",
                "regression: copied-regression.jsonl",
            ),
            encoding="utf-8",
        )
        expected = "reuses"

    def unexpected_stack_load() -> Any:
        raise AssertionError("optional model stack must not load during failed preflight")

    monkeypatch.setattr(training, "_load_training_stack", unexpected_stack_load)
    with pytest.raises(TrainingError, match=expected):
        continue_router(parent, child_config)

    assert not (tmp_path / "child-source" / "child-artifact").exists()


def test_train_rejects_continuation_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_project(
        tmp_path / "child-source", ["alpha", "beta"], output="child-artifact", continuation=True
    )

    def unexpected_stack_load() -> Any:
        raise AssertionError("ordinary train must reject continuation before optional imports")

    monkeypatch.setattr(training, "_load_training_stack", unexpected_stack_load)
    with pytest.raises(TrainingError, match="must use.*continue"):
        train_router(config)

    assert not (tmp_path / "child-source" / "child-artifact").exists()
