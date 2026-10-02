# Artifacts and workflows

## Train and resume

Run a new router from a validated v2 configuration:

```bash
uv run equiroute validate config/parent.yaml
uv run equiroute train config/parent.yaml
```

A new run refuses to reuse `output.directory`. Training validates all local configuration, registry, and partition inputs before importing the optional ML stack. It uses the configured seed, native FunctionGemma compiler, fixed LoRA target modules (`q_proj`, `v_proj`), epoch validation/checkpoints, and best validation loss selection. The held-out test partition is recorded after selection and is never used to choose the checkpoint.

If a run is interrupted after retaining an epoch checkpoint, resume the exact run:

```bash
uv run equiroute train config/parent.yaml --resume
```

Resumption requires a running manifest, a retained checkpoint, and identical route registry/data fingerprints, resolved configuration, selected device, and precision policy. A completed artifact cannot be resumed. Fix the underlying problem and choose a new output directory if those inputs have changed.

## Artifact layout and integrity

A completed run has this material layout (some checkpoint-state content exists only while training or for continuation):

```text
artifacts/parent-router/
├── model/                              # merged standard Transformers export
├── continuation/
│   ├── adapter/                        # retained adapter for continuation
│   └── trainer-state/                  # epoch checkpoints/state
└── equiroute/
    ├── manifest.json                   # versioned completed provenance
    ├── run-config.yaml                 # copied input configuration
    ├── routes.yaml                     # copied input registry
    ├── evaluation.json                 # validation/test loss evidence
    ├── semantic-evaluation.json        # explicit semantic evaluation, when run
    └── continuation-evaluation.json    # child comparison, for a continuation
```

The manifest records the pinned base-model identity, native-template fingerprint, resolved LoRA/checkpoint/evaluation settings, input fingerprints, actual hardware policy, selected checkpoint, evaluation evidence, lineage when applicable, and SHA-256 trees for `model/` and `continuation/adapter/`. File paths and hashes in the manifest are integrity evidence, not advisory metadata. Evaluation and export re-hash the relevant files and refuse mismatches.

The copied configuration and registry make an artifact self-describing. They are provenance snapshots, not live references to the source project. Do not replace files beneath a completed artifact: that breaks its recorded hashes or reproducibility evidence.

## Semantic evaluation

```bash
uv run equiroute evaluate artifacts/parent-router --data data/parent/test.jsonl
```

Evaluation loads only the artifact's copied registry and frozen run configuration, checks its manifest and model hashes, then greedily generates from the local `model/` directory. Every gold example is the denominator for all metrics. A completion must parse as one valid FunctionGemma decision; malformed or unknown-route outputs contribute zero. Argument accuracy requires both the correct route and exactly equal arguments.

The report contains aggregate metrics, per-route precision/recall/counts, an ordered confusion matrix, fixed invalid-output categories, first representative failures by signature, and the three inclusive threshold outcomes. With `evaluation.redact: true`, representative input, raw output, and parser-detail values are `null`, so that report does not retain those contents. Redaction limits what the report retains; it does not erase source datasets, console output, checkpoints, model weights, or external copies.

## Continue a router safely

A child registry must retain the parent routes and parameter schemas in the same order, then append new routes. Removed old routes, modified old schemas, an incompatible pinned model identity, missing old-route replay coverage, aliased/overlapping replay data, or missing new-route held-out coverage fail before training.

```bash
uv run equiroute continue --from artifacts/parent-router --config config/add-shipping.yaml
```

Continuation verifies the parent completed manifest and retained adapter hashes, snapshots that adapter into the child artifact, trains the child, and evaluates it in two ways:

1. The child is semantically evaluated on its held-out child test set; every newly added route must have full recall there.
2. Parent and child are evaluated on the same sealed parent-registry replay data. Prompts use the parent registry for a fair replay, while child output is scored against its full registry.

The child manifest binds the parent manifest digest and adapter hashes, parent/child registry fingerprints, ordered retained routes, added route definitions, replay data fingerprint, both reports, configured accuracy-drop limits, computed drops, and the comparison outcome. It is the lineage evidence for the child, not a claim that a continuation is automatically suitable for production.

## Review-only candidate labeling

Stage 8 is a separate, explicit workflow from a verified Stage-7 sanitized
handoff; it does not label during ingestion or make candidates available to
training:

```bash
uv sync --extra labeling
export OPENROUTER_API_KEY
uv run equiroute label labeling.yaml
```

The YAML stores the environment-variable name in
`provider.credential_env_var`, never a credential value. Before opening the
provider client, labeling verifies the Stage-7 `rows.jsonl` against its
`manifest.json`, validates the route registry, and refuses an existing output
directory. The configured system policy is kept separate from each
JSON-quoted sanitized input, which is treated as untrusted data rather than
instructions. Concurrency, rate-limit, and retry bounds come from the
configuration.

The result is a new candidate artifact, not a training artifact:

```text
candidates/support-review/
├── candidates.jsonl
└── manifest.json
```

Each row is either a route decision candidate or a rejection. Refusal, timeout,
rate-limit, transport, malformed-response, and invalid-decision outcomes are
recorded as rejected candidates. The manifest binds input/output counts and
SHA-256 values, the source manifest, policy and registry fingerprints, and the
provider model/endpoint. Candidate rows retain non-secret request provenance
only; neither artifact stores sanitized input text, policy text, provider
response content, or credentials. Normal stdout is the canonical safe manifest
summary, and normal errors do not echo those values. Independent human review
and any acceptance/training conversion are reserved for the Stage-9 boundary;
they are not included here.

## Export and deployment boundary

Training produces the merged export. `uv run equiroute export ARTIFACT` verifies a completed artifact's retained adapter and recorded hashes, then ensures its standard `model/` directory is present and matches the manifest. It never overwrites a mismatched model directory.

`model/` is intended for the user's normal local Hugging Face workflow, without EquiRoute at runtime:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "artifacts/parent-router/model"
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
```

EquiRoute does not provide an HTTP server, batch-serving policy, authentication, monitoring, fallback route, or deployment conversion. A downstream runtime must support the exported FunctionGemma model and must own production policy, including how to handle an invalid model completion.
