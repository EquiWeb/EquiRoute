# EquiRoute

EquiRoute fine-tunes the pinned FunctionGemma router model for one local, structured-output task: map an input to one route and validated arguments. It is deliberately not a serving system, a labeling service, or a general multi-model training framework.

## Quick start

EquiRoute uses Python 3.11 or newer. The normal development and validation path is local and model-free:

```bash
uv sync
uv run equiroute init support-router
cd support-router
uv sync
uv run equiroute validate config/parent.yaml
```

`init` creates a self-contained three-route parent router and a child router that adds shipping support. It refuses an existing target directory. The generated project contains its own README and uses only relative paths:

```text
support-router/
├── pyproject.toml
├── config/
│   ├── parent.yaml
│   └── add-shipping.yaml
├── routes/
│   ├── parent.yaml
│   └── add-shipping.yaml
└── data/
    ├── parent/{train,validation,test}.jsonl
    └── add-shipping/{train,validation,test,regression}.jsonl
```

After reviewing and replacing the example data, follow the generated README's complete command sequence:

```bash
uv sync
uv run equiroute validate config/parent.yaml
uv run equiroute validate config/add-shipping.yaml
uv run equiroute train config/parent.yaml
uv run equiroute evaluate artifacts/parent-router --data data/parent/test.jsonl
uv run python -c 'from transformers import AutoModelForCausalLM, AutoTokenizer; path = "artifacts/parent-router/model"; tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True); model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)'
uv run equiroute continue --from artifacts/parent-router --config config/add-shipping.yaml
uv run equiroute evaluate artifacts/add-shipping-router --data data/add-shipping/test.jsonl
```

The Python command independently loads the parent export with standard Transformers APIs and does not contact the network. The starter's `pyproject.toml` declares `equiroute[model]`, so its one-time `uv sync` installs the CLI and optional local model runtime. Training, semantic evaluation, and continuation still require model access.

Optionally verify the completed child export:

```bash
uv run equiroute export artifacts/add-shipping-router
```

Testing `uv sync` inside a starter from an unreleased source checkout requires an installable released package or a built wheel. The generated project deliberately contains no source-path dependency fallback. These model-loading commands can load the pinned base model or a local artifact. Rebuilding a missing `model/` export is a manual workflow, not part of the ordinary test or CI workload. No normal validation command, unit test, or fixture smoke test downloads or loads model weights.

## Prepare raw inputs locally

`ingest` is an optional, model-free preparation step for unlabeled JSONL. It
projects only configured fields, applies ordered redactions before writing, and
creates a new canonical artifact directory:

```bash
uv run equiroute ingest raw-ingestion.yaml
```

The configuration names a local source, a new output directory, JSON Pointer
projections for `id`, `input`, and optional metadata, redaction rules, and a
post-redaction input-size limit. Its source and output paths are relative to
the configuration file. The command writes `rows.jsonl` containing only
canonical sanitized rows and `manifest.json` containing row counts,
fingerprints, the configured size limit, and the redaction-rule count.
`ingest` does not label, train on, send, or otherwise use those rows; they are
only the possible local handoff for an explicit Stage-8 candidate-labeling step.

On success, standard output is exactly the compact canonical manifest summary.
It contains no source rows, projected values, or replacement text. Normal
errors identify only a source path, optional line and field path, reason, and
correction. For local provenance diagnosis, `--debug` adds the configuration
location, counts, and SHA-256 fingerprints to standard error without changing
standard output:
```bash
uv run equiroute ingest raw-ingestion.yaml --debug
```

Debug output intentionally never prints raw, projected, or redacted values,
but its counts and fingerprints can still be sensitive provenance. Treat it
as unsafe for routine shared logs.

## Create review-only provider candidates

Stage 8 can turn a reviewed Stage-7 sanitized handoff into **untrusted**
candidate labels. It is explicit: it never runs as part of `ingest`, validation,
training, or continuation, and it does not accept candidates or train on them.
Install the optional official OpenRouter SDK only when this workflow is needed:

```bash
uv sync --extra labeling
export OPENROUTER_API_KEY
uv run equiroute label labeling.yaml
```

The key is read from the environment variable named by
`credential_env_var`; never put a credential value in YAML. `label` requires
an intact Stage-7 artifact (`rows.jsonl` and `manifest.json`) whose sanitized
rows and manifest hash verify before any provider work. Its configuration uses
only relative paths and non-secret provider settings:

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
  Choose exactly one registered route and return JSON with name and arguments.
concurrency: 2
rate_limit_per_minute: 30
max_retries: 2
```

The policy is sent as the system message. Each sanitized row is sent separately
as JSON-quoted untrusted data with an instruction not to follow instructions
inside that data. The command uses the configured bounded concurrency and
per-minute rate limit; SDK retries are limited by `max_retries` (0–8).
Refusals, timeouts, rate limits, malformed responses, and invalid route
decisions become rejected candidates rather than examples.

On success stdout is exactly a compact, canonical manifest summary. The new
output directory contains `candidates.jsonl` and `manifest.json`; it records
only candidate decisions or rejection reasons and provenance fingerprints,
counts, hashes, provider model/endpoint, and policy/registry fingerprints.
It does not retain input text, the policy text, provider response text, or
credentials. Normal command output and errors never log secrets, policy text,
credentials. Review candidates in the explicit Stage-9 workflow before any
later use; `label` does not make a candidate a training row.

For a deliberately live SDK check, create a dedicated configuration for one
non-sensitive Stage-7 sanitized row and a new output directory, then run this
manual-only command. It is excluded from CI:

```bash
uv sync --extra labeling
export OPENROUTER_API_KEY
uv run python scripts/openrouter_live_smoke.py labeling-live-smoke.yaml
```

The smoke script refuses to run when the credential environment variable named
by its configuration is absent. It prints only the same safe canonical manifest
summary; do not use ordinary or sensitive production rows for this live check.

## Review and accept generated labels

Stage 9 is local and model-free. It verifies the immutable Stage-7 sanitized
handoff and Stage-8 candidate artifact before creating a new review directory:

```bash
uv run equiroute review-labels review.yaml
```

The resulting `review.jsonl` is the authoritative reviewer-edit input.
Reviewers may change only each row's `review` object: leave it
`{"decision":"unreviewed"}`, or record `approved` or `rejected` with a
non-empty reviewer and RFC 3339 `reviewed_at` timestamp. Rejections also need
one of `incorrect_route`, `incorrect_arguments`, `insufficient_context`, or
`other`. Approval is valid only for a row that Stage 9 selected for review,
whose candidate is locally valid, and whose candidate status is `labeled`;
reviewers do not edit the candidate route decision, arguments, validation, or
provenance.

`review.csv` is a read-only spreadsheet view, not an acceptance input. The
review directory also includes `report.json` and `manifest.json`. Both Stage-9
commands print only their compact, canonical manifest summary on success;
normal errors omit provider credentials and raw-source contents. Optional
deterministic review sampling sets per-route quotas; its report makes provider
rejections, locally invalid rows, unreviewed/rejected/approved counts, quota
shortfalls, and route imbalance visible. An optional hand-labeled gold JSONL
compares candidate decisions by route and reports valid, invalid, missing,
route-correct, exact-decision-correct, and invalid-decision-rate counts.

After editing only `review.jsonl`, explicitly compile the approved subset:

```bash
uv run equiroute accept-labels acceptance.yaml
```

Acceptance re-verifies the Stage-7, Stage-8, registry, and review-manifest
bindings, including an immutable fingerprint of every non-review field. It
accepts only explicit approvals and runs the resulting `examples.jsonl`
through the same Stage-1 dataset validation as hand-authored data. Optional
deterministic per-route acceptance quotas can select a bounded approved
subset; zero is a valid quota. The resulting `report.json` repeats review,
quota/imbalance, and optional gold-quality evidence. It does not invoke
`train`, `continue`, or any model/provider command: add accepted examples to
curated partitions and run normal validation/training deliberately.

Every accepted row stores `_equiroute` provenance binding its source ID and
line, candidate and handoff hashes, handoff and review-manifest fingerprints,
registry and policy fingerprints, provider model/endpoint, reviewer,
timestamp, and approved decision. This is traceability, not automatic
trust or automatic training.

## What to read next

- [Schemas and migrations](docs/schemas.md): strict YAML/JSONL contracts, v1-to-v2 reading, and persistence rules.
- [Artifacts and workflows](docs/artifacts-and-workflows.md): training, resumption, evaluation, continuation, hashes, and export.
- [Operations and boundaries](docs/operations.md): deployment, devices, privacy, licensing, and test discipline.

## Scope boundaries

The sole supported base model is `google/functiongemma-270m-it` at revision `39eccb091651513a5dfb56892d3714c1b5b8276c`. EquiRoute fixes the native FunctionGemma template and uses LoRA on its `q_proj` and `v_proj` modules. It does not accept an alternative model, model revision, or user device override.

EquiRoute emits a merged local Hugging Face model directory for deployment, but it does not run an endpoint, choose a production fallback, or implement a runtime proxy. Its optional Stage-8 integration creates review-only provider candidates from an explicitly selected sanitized handoff; it does not manage credentials, accept candidates, or train on them.
