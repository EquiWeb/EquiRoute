# Schemas, versions, and migrations

EquiRoute treats configuration, datasets, reports, and manifests as contracts. Loaders parse YAML or JSONL locally, migrate a supported older document in memory, then run strict Pydantic validation. They reject unknown fields and type coercion. A successful read never edits the source file.

## Version policy

New training configuration and persisted EquiRoute artifacts use:

```yaml
schema_version: "2"
```

Version 1 configuration and artifacts remain readable through explicit v1-to-v2 migrations. Migration happens before model validation and only for recognized v1 shapes. The legacy unversioned training configuration shape is also treated as v1 for compatibility. It is not an automatic file rewrite: a v1 source stays v1 unless its owner deliberately replaces it. A non-string, future, or otherwise unsupported declared version fails with an actionable migration/version error; missing versions on versioned artifacts and unknown fields after migration fail strict validation.

Versioning is not a promise that arbitrary historical or hand-edited documents are accepted. Keep original artifact files with their manifests and hashes; create a new run or consciously persist a new document when changing its provenance.

## Training configuration

A v2 configuration declares the pinned model, local relative paths, dataset partitions, training options, semantic-evaluation policy, and output directory.

```yaml
schema_version: "2"
model:
  base_model: google/functiongemma-270m-it
  revision: 39eccb091651513a5dfb56892d3714c1b5b8276c
routes: routes/parent.yaml
data:
  train: data/parent/train.jsonl
  validation: data/parent/validation.jsonl
  test: data/parent/test.jsonl
training:
  seed: 42
  epochs: 4
  learning_rate: 0.0002
  batch_size: 4
  gradient_accumulation_steps: 8
  lora_rank: 16
  lora_alpha: 32
  max_sequence_length: 1024
evaluation:
  arguments: exact
  redact: true
  max_new_tokens: 128
  thresholds:
    valid_decision_rate: 0.8
    route_accuracy: 0.8
    argument_accuracy: 0.8
output:
  directory: artifacts/parent-router
  export: merged_huggingface
```

Paths are resolved relative to the configuration file unless already absolute. `base_model`, `revision`, `arguments`, and `export` are fixed values, not extension points. Training values have positive/range validation. Evaluation thresholds are inclusive values in `[0, 1]`; all three default to `0.0` if omitted. `redact` defaults to `false`, so set it explicitly when reports must not keep representative failure contents.

A continuation configuration additionally declares a sealed replay set and optional allowed drops, both defaulting to no loss when present:

```yaml
continuation:
  regression: data/add-shipping/regression.jsonl
  max_route_accuracy_drop: 0.0
  max_argument_accuracy_drop: 0.0
```

Use it only with `uv run equiroute continue --from PARENT --config CONFIG`, not `uv run equiroute train`.

## Route registry and examples

A registry is a YAML mapping with a non-empty ordered `routes` list. Every route has a unique non-empty `name`, non-empty `description`, and a constrained object parameter schema. Properties use only the primitive types `string`, `integer`, `number`, and `boolean`; required names must be declared properties, and `additionalProperties` must be `false`. A route decision names one registered route and supplies arguments valid for that route.

Each JSONL record is one example:

```json
{"id":"billing-001","input":"Why was I charged twice?","route":{"name":"billing_support","arguments":{}},"metadata":{}}
```

`input` is non-empty. `id` and `metadata` are optional at the basic document level, but partition validation requires stable IDs and rejects duplicate normalized inputs across the train, validation, and test partitions. `validate` checks the registry and all declared partitions before training. `split` validates one source and emits a deterministic, route-stratified 80/10/10 split using the supplied seed.

## Persisted evidence

Reports and training manifests are versioned documents too. They are emitted as UTF-8 JSON with sorted keys, no NaN, and a trailing newline; training manifests and semantic reports use compact separators, while split reports/manifests use readable indentation. A current persisted document carries `schema_version: "2"`; a v1 persisted document is read through the explicit migration path.

Validation does more than shape-checking. Dataset reports reconcile route totals; semantic reports reconcile metrics, confusion rows, invalid-output counts, thresholds, and redaction; completed manifests require checkpoint, evaluation, and artifact-hash evidence. Invalid evidence is rejected instead of being silently normalized.
