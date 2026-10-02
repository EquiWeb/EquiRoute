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

## Raw-ingestion configuration and artifact

`equiroute ingest CONFIG` accepts only a strict v2 YAML configuration and
creates an unlabeled, sanitized artifact locally. It is independent of the
training configuration and is not a training input.

```yaml
schema_version: "2"
source: exports/support.jsonl
output:
  directory: prepared/support
projection:
  id: /ticket/id
  input: /ticket/message
  metadata:
    channel: /ticket/channel
redactions:
  - target: /input
    pattern: '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    replacement: "[email]"
limits:
  max_input_bytes: 4096
```

`source` and `output.directory` resolve relative to `CONFIG`. The source is
one JSON object per non-blank JSONL line. Projection values are RFC 6901 JSON
Pointers: they must resolve through object mappings (not arrays) in every
source row. `id` and `input` are required; `metadata` is optional. The
canonical output appends `metadata._equiroute.source_line`, a one-based source
line number, for local provenance.

Each ordered redaction targets `/input` or one projected
`/metadata/<name>` string. Redactions run after projection and before row
validation, hashing the sanitized output, or writing anything. The resulting
`input` must be non-empty UTF-8 and no larger than `max_input_bytes` after
redaction. A source-row failure reports its source path, line, projection or
canonical field path, rejection reason, and correction; it never prints the
row or its projected values.

The output directory must not already exist. On success it is atomically
created with:

```text
prepared/support/
├── rows.jsonl
└── manifest.json
```

`rows.jsonl` is compact, sorted-key UTF-8 JSONL with only:

```json
{"id":"chat-0104","input":"I cannot log in after changing my phone number.","metadata":{"_equiroute":{"source_line":1},"channel":"support-export"}}
```

`manifest.json` is a pretty, sorted-key v2 document. It records only source
and output row counts and SHA-256 hashes, a fingerprint of the validated
configuration, the input-size limit, and redaction-rule count. It deliberately
does not retain source rows, canonical row values, source paths, projection
pointers, redaction patterns, or replacement text. The source hash identifies
the original bytes, while the output hash identifies the canonical sanitized
rows.

## Candidate-labeling configuration and artifact

`equiroute label CONFIG` is the explicit Stage-8 handoff from a verified,
sanitized Stage-7 artifact to untrusted provider candidates. It is not called
by `ingest`, `validate`, `train`, or `continue`; it has no acceptance or
training path. Install its optional SDK with `uv sync --extra labeling`, and
set the credential in the environment variable named by the configuration
rather than storing a credential value in the file:

```yaml
schema_version: "2"
input:
  directory: prepared/support
routes: routes/parent.yaml
output:
  directory: candidates/support-review
provider:
  endpoint: https://openrouter.ai/api/v1
  model: openai/gpt-4o-mini
  credential_env_var: OPENROUTER_API_KEY
policy_prompt: >-
  Choose one registered route. Return a JSON object with name and arguments.
concurrency: 2
rate_limit_per_minute: 30
max_retries: 2
```

All paths resolve relative to `CONFIG`. `input.directory` must be an
unmodified Stage-7 directory containing its canonical `rows.jsonl` and
`manifest.json`; both row count and SHA-256 are verified before any provider
request. `output.directory` must not already exist. `concurrency` is 1–64,
the rate limit is 1–10,000 requests per minute, and `max_retries` is 0–8.

The immutable `policy_prompt` is a system message. Each sanitized input is
separately JSON-quoted as untrusted data and accompanied by an instruction not
to execute instructions it contains. A response is only a candidate: provider
refusal, timeout, rate limit, transport failure, malformed JSON, unknown
route, or invalid arguments emits a schema-valid rejected candidate. Nothing
in Stage 8 makes a candidate a training example. That review/acceptance
decision is the Stage-9 boundary and is intentionally not implemented here.

The atomically created candidate directory contains:

```text
candidates/support-review/
├── candidates.jsonl
└── manifest.json
```

The JSONL records a candidate decision or rejection reason plus source ID/line,
request status/attempt count/timestamp, and non-secret provider and fingerprint
provenance. The manifest records the input/output rows and SHA-256 values,
input-manifest, policy, and registry fingerprints, plus provider model and
endpoint. Neither artifact retains sanitized input text, policy text, raw
provider response content, or credentials. Successful CLI stdout is only that
canonical manifest summary; normal errors and output do not log secrets or
those contents.

## Persisted evidence

Reports and training manifests are versioned documents too. They are emitted as UTF-8 JSON with sorted keys, no NaN, and a trailing newline; training manifests and semantic reports use compact separators, while split reports/manifests use readable indentation. A current persisted document carries `schema_version: "2"`; a v1 persisted document is read through the explicit migration path.

Validation does more than shape-checking. Dataset reports reconcile route totals; semantic reports reconcile metrics, confusion rows, invalid-output counts, thresholds, and redaction; completed manifests require checkpoint, evaluation, and artifact-hash evidence. Invalid evidence is rejected instead of being silently normalized.
