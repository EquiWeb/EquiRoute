# Operations and boundaries

## Normal checks versus model-extra work

The ordinary engineering path is deliberately model-free. It uses the Python 3.11 project environment and exercises formatting, static typing, unit tests, and the deterministic CLI fixture smoke without downloading weights or importing the model stack:

```bash
uv run ruff format --check
uv run mypy src
uv run pytest
uv run python scripts/fixture_smoke.py
```

The fixture smoke initializes a disposable starter project and validates its parent configuration. It is not a training or inference check. Keep model loading, network access, long-running training, and hardware-dependent investigation out of ordinary tests and CI.

Model-extra commands are manual operational work:

```bash
uv sync --extra model
uv run equiroute train CONFIG
uv run equiroute evaluate ARTIFACT --data DATA
uv run equiroute continue --from PARENT --config CONFIG
uv run equiroute export ARTIFACT
```

They require compatible local PyTorch/Transformers/PEFT/Accelerate installation and may obtain or load FunctionGemma files. Run them only where that is intended and where the data/artifact access is appropriate.

## Device policy and diagnosis

EquiRoute selects the best available device in fixed order: CUDA, then Apple MPS, then CPU. There is no configuration or CLI device override. The selected capability is written into the training manifest and must match when resuming.

| Device | EquiRoute precision policy | First checks when it fails |
| --- | --- | --- |
| CUDA | `bfloat16` with Accelerate `bf16` when supported; otherwise `float16` with `fp16` | Confirm the installed PyTorch build can see the CUDA runtime/GPU, that driver/runtime compatibility is correct, and that memory is sufficient for the configured sequence length and batch accumulation. |
| MPS | `float32`, no mixed precision | Confirm a macOS PyTorch build reports MPS available. The core path intentionally does not enable mixed precision on MPS; use smaller batch/sequence settings if memory pressure occurs. |
| CPU | `float32`, no mixed precision | CPU is supported for the same local contract but can be slow. Confirm optional model dependencies are installed and use a realistic manual run; do not mistake model-free tests for a CPU training benchmark. |

If the model extra is missing, install it. If a manual model command fails before model load, first run `uv run equiroute validate CONFIG` to isolate configuration/data errors. If resumption fails after a device change, use the original hardware policy or start a new output directory; EquiRoute intentionally does not silently change precision or device for a resumed run.

## Data and privacy limits

Route descriptions, examples, copied configuration/registry provenance,
semantic reports, checkpoints, adapters, and merged model weights can contain
information derived from your routing task. Treat local paths, backups, logs,
artifact directories, and any downstream deployment as part of your
data-handling boundary.

Raw ingestion is local-only. `equiroute ingest` reads a configured local JSONL
source and writes only its projected, redacted canonical rows plus a manifest;
it has no provider, request, credential, telemetry, training, or automatic
labeling integration. Projection and redaction are controls for the resulting
artifact, not a deletion operation: the original source file, backups, shell
history, and filesystem access remain the operator's responsibility. Review
the artifact before any later use, and keep the output directory under the
same access and retention controls as the source.

Successful normal ingest output is a manifest summary only. Standard errors
for expected failures contain safe source location, line, field path, reason,
and correction rather than source or projected values. `--debug` is explicit
and stderr-only; it adds the configuration location, row counts, and SHA-256
provenance fingerprints without printing raw, projected, replacement, or
sanitized values. Those identifiers can nevertheless reveal or correlate
local datasets, so debug output is unsafe for routine shared logs and must
remain within the local data boundary.

`evaluation.redact: true` writes `null` for representative input, raw
completion, and parse-detail values in semantic-evaluation reports, so those
values are not retained there. It is not a general privacy mechanism, a
deletion operation, a guarantee that training data cannot be memorized, or a
promise of compliance with any law, policy, or organization requirement.
EquiRoute has no hosted labeling provider, API credential store, telemetry
pipeline, or server in this release; users remain responsible for the data
they place in local files and for downstream controls.

## FunctionGemma and Gemma terms

EquiRoute supports only Google's FunctionGemma model at the pinned revision named in its configuration. Access to the model and use of its weights are governed by the applicable Gemma terms, not by EquiRoute. Review Google's official [Gemma terms of use](https://ai.google.dev/gemma/terms) and the official [FunctionGemma model page on Hugging Face](https://huggingface.co/google/functiongemma-270m-it) before downloading, fine-tuning, redistributing, or deploying it. The Hugging Face page exposes the model's access and licensing information for that specific repository.

These links are references, not legal advice and not a statement that a particular use is permitted, compliant, safe, or suitable. You must determine which terms, notices, restrictions, and obligations apply to your use and to any distribution of an artifact.

## Deployment responsibility

The exported `model/` directory uses the standard local Transformers loading interface. EquiRoute stops there. The deployment operator selects a compatible runtime and owns access control, isolation, input/output handling, observability, retention, rate limits, fallback behavior, and all production safety decisions. In particular, invalid structured output is an evaluation failure in EquiRoute; this project does not prescribe a deployment fallback route.
