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
only the possible local handoff for a future labeling step.

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

## What to read next

- [Schemas and migrations](docs/schemas.md): strict YAML/JSONL contracts, v1-to-v2 reading, and persistence rules.
- [Artifacts and workflows](docs/artifacts-and-workflows.md): training, resumption, evaluation, continuation, hashes, and export.
- [Operations and boundaries](docs/operations.md): deployment, devices, privacy, licensing, and test discipline.

## Scope boundaries

The sole supported base model is `google/functiongemma-270m-it` at revision `39eccb091651513a5dfb56892d3714c1b5b8276c`. EquiRoute fixes the native FunctionGemma template and uses LoRA on its `q_proj` and `v_proj` modules. It does not accept an alternative model, model revision, or user device override.

EquiRoute emits a merged local Hugging Face model directory for deployment, but it does not run an endpoint, choose a production fallback, or implement a runtime proxy. It validates human-authored local data; it does not provide assisted labeling, provider integration, or credential management.
