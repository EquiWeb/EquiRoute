from __future__ import annotations

from pathlib import Path

import pytest

from equiroute.errors import ConfigLoadError, ExampleLoadError, RegistryLoadError
from equiroute.io import iter_examples, load_examples, load_route_registry, load_training_config


def _registry_file(tmp_path):
    path = tmp_path / "routes.yaml"
    path.write_text(
        """\
routes:
  - name: billing_support
    description: Billing questions.
    parameters:
      type: object
      properties:
        account_id:
          type: string
      required:
        - account_id
      additionalProperties: false
  - name: technical_support
    description: Technical questions.
    parameters:
      type: object
      properties: {}
      required: []
      additionalProperties: false
""",
        encoding="utf-8",
    )
    return path


def test_loads_valid_registry_and_examples_with_omitted_arguments(tmp_path):
    registry = load_route_registry(_registry_file(tmp_path))
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        '{"id":"one","input":"The app fails","route":{"name":"technical_support"}}\n',
        encoding="utf-8",
    )

    loaded = load_examples(examples, registry)

    assert loaded[0].route.arguments == {}


def test_rejects_unknown_selected_route_with_line_context(tmp_path):
    registry = load_route_registry(_registry_file(tmp_path))
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        '{"input":"Where is my invoice?","route":{"name":"unknown"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ExampleLoadError, match=r"examples\.jsonl:1: route\.name: unknown route 'unknown'"):
        load_examples(examples, registry)


def test_rejects_arguments_outside_the_route_schema(tmp_path):
    registry = load_route_registry(_registry_file(tmp_path))
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        '{"input":"Where is my invoice?","route":{"name":"billing_support","arguments":{"account_id":7,"extra":true}}}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ExampleLoadError,
        match=r"unknown argument 'extra'.*account_id.*must be string",
    ) as raised:
        load_examples(examples, registry)

    assert raised.value.path == "route.arguments"
    assert raised.value.correction == (
        "remove unsupported argument: 'extra'; set argument 'account_id' to a string"
    )



def test_rejects_duplicate_example_ids_with_both_line_numbers(tmp_path):
    registry = load_route_registry(_registry_file(tmp_path))
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        "\n".join(
            [
                '{"id":"same","input":"First","route":{"name":"technical_support"}}',
                '{"id":"same","input":"Second","route":{"name":"technical_support"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ExampleLoadError,
        match=r"examples\.jsonl:2: id: duplicate example id 'same'; first declared on line 1",
    ):
        load_examples(examples, registry)

def test_iter_examples_yields_locations_before_later_row_validation(tmp_path):
    registry = load_route_registry(_registry_file(tmp_path))
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        "\n".join(
            [
                '{"id":"one","input":"First","route":{"name":"technical_support"}}',
                "{not json}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    iterator = iter_examples(examples, registry)
    loaded = next(iterator)

    assert loaded.example.id == "one"
    assert loaded.source == examples
    assert loaded.line == 1
    with pytest.raises(ExampleLoadError, match=r"examples\.jsonl:2: \$: malformed JSON"):
        next(iterator)


def test_rejects_malformed_jsonl_with_line_context(tmp_path):
    registry = load_route_registry(_registry_file(tmp_path))
    examples = tmp_path / "examples.jsonl"
    examples.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(
        ExampleLoadError,
        match=r"examples\.jsonl:1: \$: malformed JSON.*; correction: replace this line with a valid JSON object",
    ) as raised:
        next(iter_examples(examples, registry))

    assert raised.value.correction == "replace this line with a valid JSON object"


def test_rejects_invalid_utf8_with_source_context(tmp_path):
    registry = load_route_registry(_registry_file(tmp_path))
    examples = tmp_path / "examples.jsonl"
    examples.write_bytes(b"\xff")

    with pytest.raises(ExampleLoadError, match=r"examples\.jsonl:1: \$: could not decode UTF-8"):
        load_examples(examples, registry)


def test_rejects_unknown_training_config_fields(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """\
model:
  base_model: google/functiongemma-270m-it
  revision: pinned
routes: routes.yaml
data:
  train: data/train.jsonl
  validation: data/validation.jsonl
  test: data/test.jsonl
training:
  seed: 42
  epochs: 4
  learning_rate: 0.0002
  batch_size: 4
  gradient_accumulation_steps: 8
  lora_rank: 16
  lora_alpha: 32
  max_sequence_length: 1024
output:
  directory: runs/support-router-v1
  export: merged_huggingface
unexpected: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError, match=r"unexpected: Extra inputs are not permitted"):
        load_training_config(config)

def test_rejects_unpinned_or_unsupported_base_model(tmp_path):
    config = tmp_path / "config.yaml"
    fixture = Path(__file__).parent / "fixtures" / "config.yaml"
    config.write_text(
        fixture.read_text(encoding="utf-8").replace(
            "google/functiongemma-270m-it", "other/model"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError, match=r"model\.base_model: Input should be"):
        load_training_config(config)


def test_rejects_duplicate_route_names(tmp_path):
    registry = tmp_path / "routes.yaml"
    registry.write_text(
        """\
routes:
  - name: duplicate
    description: First route.
    parameters:
      type: object
      properties: {}
      required: []
      additionalProperties: false
  - name: duplicate
    description: Second route.
    parameters:
      type: object
      properties: {}
      required: []
      additionalProperties: false
""",
        encoding="utf-8",
    )

    with pytest.raises(
        RegistryLoadError, match=r"route names must be unique: duplicate"
    ):
        load_route_registry(registry)
