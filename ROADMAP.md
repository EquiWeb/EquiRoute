# EquiRoute roadmap

## Product boundary

EquiRoute is a **local-first training tool** for making small language models into routing specialists.

```text
labeled route examples → EquiRoute training → portable trained model → user-selected inference stack → application dispatch
```

EquiRoute is **not** in the runtime request path. It does not host models, proxy requests, call user endpoints, or choose fallbacks at production inference time. A user trains or continues a router, receives a standard model artifact, and deploys it with their existing stack: Hugging Face Transformers, vLLM, llama.cpp, Ollama, MLX, LiteRT, or another compatible runtime.

The framework's job ends with a reproducible, evaluated, portable model export plus the metadata required to continue training it safely.

## Product goal

Enable a developer to teach a sub-1B parameter model to turn a natural-language query into a structured route decision, such as:

```json
{
  "name": "billing_support",
  "arguments": {}
}
```

The deployed application owns the operational flow:

```text
request → deployer's inference server → generated tool/route call → deployer's application code → chosen endpoint/model
```

EquiRoute supplies neither the server nor the dispatch code.

## Initial technical decisions

| Concern | Decision | Why |
| --- | --- | --- |
| Language | Python 3.11+ | It is the native ecosystem for model loading, fine-tuning, evaluation, and export. |
| Product shape | Typed library and Typer CLI | A developer needs scripts and artifacts, not a hosted control plane. |
| Configuration | Pydantic v2 models with YAML files | Strict validation and readable training recipes. |
| Training stack | PyTorch, Transformers, TRL, PEFT, Accelerate | Established Hugging Face components; supports CUDA, MPS, and CPU. |
| Dataset format | Canonical JSONL | Easy to author, inspect, version, stream, and generate from other tools. |
| Initial model | FunctionGemma 270M | It is already trained for function calling and intended for further specialization. |
| Adaptation method | LoRA supervised fine-tuning | Small artifacts and practical training on user-owned hardware. |
| First model family | FunctionGemma only | Prove one well-supported path before introducing compatibility abstractions. |
| Testing | pytest with deterministic fixtures | Core schema and evaluation tests must not require a network or model download. |

FunctionGemma is a Gemma 3 270M variant specialized for function calling. Google describes it as a base for custom local agents and routing/traffic-control scenarios, and documents fine-tuning with the Hugging Face ecosystem. See [the announcement](https://blog.google/innovation-and-ai/technology/developers-tools/functiongemma/) and [the fine-tuning guide](https://developers.googleblog.com/a-guide-to-fine-tuning-functiongemma/). PEFT is designed to adapt a pretrained model by training a small set of additional parameters rather than the complete model; see the [PEFT documentation](https://huggingface.co/docs/peft/en/index).

### Explicit non-goals for the initial MVP

- A hosted training service, accounts, telemetry, database, or web UI.
- An inference server, gateway, proxy, or endpoint dispatcher.
- Training a transformer from scratch.
- A custom model architecture.
- Mandatory cloud services or API keys.
- Multiple base-model families before the first training pipeline is proven.
- Multi-route execution plans. One input produces one route decision in V1.

## Why fine-tune rather than train from scratch

A small scratch-trained model still needs substantial pretraining before it can robustly understand paraphrase, intent, tool schemas, and structured generation. User-provided routing examples are far more valuable when spent adapting a model that already understands natural language and function-call conventions.

FunctionGemma is the correct first baseline because it is small, specialized for structured tool calls, and viable for local and edge inference. EquiRoute will fine-tune it to learn the user's routing policy and domain vocabulary.

The framework should retain the following extension seam, but not build it prematurely:

```text
Model adapter = load + format training examples + configure LoRA + export
```

A later adapter can support another sub-1B function-calling base model without changing EquiRoute's dataset, evaluation, artifact, or continuation contracts.

## Canonical routing contract

EquiRoute needs one semantic internal representation:

```json
{
  "name": "billing_support",
  "arguments": {
    "account_id": "A-1042"
  }
}
```

The base model is trained using its native function-call/chat template. This preserves its function-calling prior instead of teaching every user-defined punctuation convention from scratch.

EquiRoute does **not** need to sit between deployed requests to translate the result. Developers may either:

1. retain FunctionGemma's native generated function-call format and configure their standard inference layer accordingly; or
2. use their own thin parser/renderer in the deploying application after inference.

EquiRoute will ship reference parsers and rendering examples for local evaluation and integration testing, but they are training/evaluation utilities, not a production runtime dependency.

### Output-format policy

A user may configure the expected tool schema, route names, argument JSON Schema, and a target deployment convention. The primary stable contract is semantic correctness: the selected route and validated arguments.

Supported reference renderers can include:

| Renderer | Example |
| --- | --- |
| `canonical_json` | `{"name":"billing_support","arguments":{}}` |
| `route_name` | `billing_support` |
| `openai_tool_call` | An OpenAI-compatible `tool_calls` object |
| `json_template` | A constrained user-supplied JSON shape |

The exported model remains a standard model artifact. EquiRoute is never required at serving time.

## User-facing data contract

### Route registry

Every training run supplies a route registry. It gives the model the allowed route names, their descriptions, and the valid parameter structure.

```yaml
routes:
  - name: billing_support
    description: Billing, invoices, refunds, payment failures, and account charges.
    parameters:
      type: object
      properties:
        account_id:
          type: string
      additionalProperties: false

  - name: technical_support
    description: Product errors, configuration issues, and integration failures.
    parameters:
      type: object
      properties: {}
      additionalProperties: false
```

Rules:

- Every route name is unique and immutable within a model lineage.
- Route descriptions are model context, so they must be versioned with the artifact.
- Parameters use a constrained JSON Schema subset.
- A labeled decision selects exactly one registered route.
- Arguments must validate against that route's parameter schema.
- V1 does not model route chains, confidence arbitration, or runtime endpoint execution.

### Labeled examples

The canonical training format is one JSON object per line:

```json
{
  "id": "support-00017",
  "input": "My card was charged twice for the Pro subscription.",
  "route": {
    "name": "billing_support",
    "arguments": {}
  },
  "metadata": {
    "source": "hand_labeled",
    "domain": "subscriptions"
  }
}
```

`input` and `route.name` are required. `id`, `route.arguments`, and `metadata` are optional. An omitted `arguments` value is equivalent to `{}`.

CSV and provider-specific imports can eventually be supplied as converters, but they must compile to this JSONL representation before validation or training.

### Training configuration

```yaml
model:
  base_model: google/functiongemma-270m-it
  revision: "<pinned revision>"

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
```

### Dataset split policy

- Preferred mode: users provide separately curated `train`, `validation`, and sealed `test` sets.
- Convenience mode: EquiRoute makes deterministic, seeded, stratified splits by route.
- Validation selects checkpoints and hyperparameters.
- The test set is evaluated only after model selection; it must not be used for tuning.
- EquiRoute rejects splits where a route is absent from training or too rare for the requested split.
- Dataset manifest fingerprints and the split seed are part of the artifact provenance.

## Artifact and portability contract

A successful training run produces two related outputs:

```text
support-router-v1/
├── model/                         # deployment artifact
│   ├── model.safetensors           # merged FunctionGemma + LoRA weights
│   ├── config.json
│   ├── generation_config.json
│   └── tokenizer files
├── continuation/                  # retained only to continue training
│   ├── adapter/                   # LoRA weights and PEFT configuration
│   └── trainer-state/             # resumable checkpoint where requested
├── equiroute/
│   ├── routes.yaml                # exact registry used for training
│   ├── run-config.yaml            # resolved configuration
│   ├── manifest.json              # lineage and reproducibility data
│   └── evaluation.json            # validation/test reports
└── README.md                      # deployment and provenance summary
```

`model/` is a normal Hugging Face-compatible model directory. A user can move it to a standard inference workflow without installing EquiRoute:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("runs/support-router-v1/model")
tokenizer = AutoTokenizer.from_pretrained("runs/support-router-v1/model")
```

It can then be converted or served through the user's usual deployment chain, subject to that runtime's FunctionGemma compatibility:

```text
Hugging Face Transformers → vLLM / TGI
Hugging Face model → llama.cpp conversion → GGUF
Hugging Face model → MLX conversion
Hugging Face model → platform-specific quantization/export
```

The `continuation/` directory is not needed for serving. It is required to continue adapter training safely and should therefore be retained by the model owner.

`manifest.json` records the EquiRoute schema version, base model identifier and pinned revision, tokenizer/template identity, route-registry fingerprint, dataset fingerprints, resolved configuration, software/hardware metadata, evaluation results, and parent artifact when applicable.

## CLI shape

The initial product should be useful through a small command surface:

```bash
equiroute init
equiroute validate config.yaml
equiroute split data.jsonl --routes routes.yaml --seed 42
equiroute train config.yaml
equiroute evaluate runs/support-router-v1 --data data/test.jsonl
equiroute continue --from runs/support-router-v1 --config support-router-v2.yaml
equiroute export runs/support-router-v1 --format huggingface
```

`equiroute infer` may exist as a local smoke-test and evaluation utility. It is not a server and is not part of deployment architecture.

# Staged implementation plan

Each stage has a hard completion gate. Do not proceed until its gate passes.

## Stage 0 — Foundation and model-loading spike

### Deliver

- Python package structure, dependency policy, CLI skeleton, and error conventions.
- Pydantic schemas for routes, examples, configuration, manifests, and evaluation reports.
- A compact, checked-in three-route fixture dataset and registry.
- Model-loading smoke path against the pinned FunctionGemma revision.
- Hardware capability detection for CUDA, MPS, and CPU.

### Exclusions

- No actual training.
- No server, runtime proxy, web interface, or deployment integration.

### Completion gate

1. `equiroute --help` lists the intended commands and their inputs.
2. The sample registry and JSONL load into canonical typed models.
3. Invalid route names, invalid arguments, duplicate IDs, malformed JSONL, and unknown configuration fields produce actionable errors.
4. A smoke script loads the pinned FunctionGemma model and generates at least one token on a supported local device.
5. Unit tests run without downloading a model or calling any external service.

**MVP standard:** the repository has one stable vocabulary for data, configuration, artifacts, and errors.

## Stage 1 — Data validation and deterministic splitting

### Deliver

```bash
equiroute validate config.yaml
equiroute split data.jsonl --routes routes.yaml --seed 42
```

- Streaming JSONL validation.
- Route and argument validation against the route registry.
- Duplicate-ID detection with source line numbers.
- Dataset-size and route-distribution report.
- Deterministic, stratified train/validation/test splitting.
- Dataset manifests and fingerprints.
- Early rejection of insufficient per-route coverage and leakage-prone inputs.

### Completion gate

1. The fixture dataset produces byte-identical splits for the same seed.
2. A changed seed changes at least one eligible row assignment.
3. Every emitted split preserves required route coverage.
4. Invalid rows report file, line, JSON path, and correction.
5. A deliberately route-sorted input still results in stratified splits.

**MVP standard:** training cannot begin with an invalid, ambiguous, or leaky dataset.

## Stage 2 — Training-target compilation and semantic evaluation primitives

### Deliver

- Compilation of canonical examples and route registry into FunctionGemma's supported function-call conversation template.
- Parsing of generated function calls into canonical `RouteDecision` values for evaluation.
- Local argument-schema validation.
- Reference output renderers for development tests and integration examples.
- A typed invalid-output result: raw completion plus parse/validation category.

### Completion gate

1. Golden fixture examples compile to the exact expected FunctionGemma training conversation.
2. Valid completions round-trip through `RouteDecision` without semantic change.
3. Unknown routes, malformed call syntax, missing required arguments, extra arguments, and invalid JSON are rejected.
4. Every reference renderer preserves the same semantic route and arguments.
5. Tests do not pin incidental implementation details such as non-semantic whitespace.

**MVP standard:** model targets, local evaluation, and application integration share a stable semantic contract.

## Stage 3 — End-to-end LoRA training and portable export

### Deliver

```bash
equiroute train config.yaml
```

- Pinned base-model loading and LoRA configuration.
- Training on compiled canonical conversations.
- Support for CUDA, MPS, and CPU with explicit unsupported-option errors.
- Checkpoints, resolved configuration, and provenance capture.
- Best-checkpoint selection by validation metric.
- Interrupted-run resumption.
- Adapter preservation for future continuation.
- Merged standard Hugging Face model export to `model/`.

The default training path should remain small and predictable. CUDA-only low-bit options may be offered later, but the core path must not depend on them.

### Completion gate

Use a compact fixture with at least three separable routes and paraphrased held-out inputs.

1. Training completes locally on a supported device without a hosted service.
2. The exported `model/` loads in a fresh Python process using standard Transformers APIs and no EquiRoute imports.
3. The exported model's held-out semantic route accuracy materially exceeds the base model on the fixture; the fixture threshold is explicit and versioned.
4. Re-running with the same seed/environment reproduces the same split and equivalent fixed-fixture predictions.
5. An interrupted run resumes and emits a valid artifact.
6. The artifact identifies the exact base revision, dataset fingerprints, template, and training configuration.

**MVP standard:** a developer can produce a deployable model from labeled routing data on their own hardware.

## Stage 4 — Evaluation and quality gates

### Deliver

```bash
equiroute evaluate runs/support-router-v1 --data data/test.jsonl
```

Metrics:

- Exact route accuracy.
- Exact argument match when arguments are in scope.
- Valid-decision rate.
- Per-route precision, recall, and support.
- Confusion matrix.
- Invalid-output count by failure category.
- Representative error report with optional redaction.
- Configurable non-zero exit on failed quality thresholds.

Evaluation is semantic. It does not compare arbitrary output strings.

### Completion gate

1. A fixed prediction fixture yields exact expected metric values.
2. Semantically equivalent JSON argument order does not count as a failure.
3. Unknown routes and malformed output lower valid-decision rate and never count as correct.
4. Routes with zero predictions remain visible in reports.
5. A configured threshold failure returns a non-zero process exit.

**MVP standard:** the user can make a defensible promotion decision from routing-specific quality evidence, rather than loss alone.

## Stage 5 — Continuation training and route expansion

### Deliver

```bash
equiroute continue \
  --from runs/support-router-v1 \
  --config support-router-v2.yaml
```

Rules:

- Continuation starts from an EquiRoute adapter artifact, not an arbitrary merged model directory.
- The new route registry must be compatible with and normally a superset of the prior registry.
- Existing route names and parameter schemas cannot silently change.
- Adding a route requires examples for the new route and replay examples for old routes.
- By default, continuation fails if the supplied data lacks coverage for an existing route.
- The resulting artifact records parent identity, registry changes, data changes, and comparative evaluation.

The retained adapter exists only to support this workflow. The merged model export remains the artifact intended for deployment.

### Completion gate

1. Train a three-route model, then continue it with a fourth route.
2. The continued model correctly selects the fourth route on unseen examples.
3. It maintains a defined minimum performance on the original routes' sealed regression set.
4. Continuation with a removed route, changed old schema, incompatible base revision, or missing replay coverage fails before training.
5. The child artifact records a verifiable parent lineage.

**MVP standard:** users can expand a router without restarting from scratch or silently forgetting established routes.

## Stage 6 — Developer experience and release readiness

### Deliver

- `equiroute init` generates an intentionally minimal project: configuration, route registry, sample data, and command sequence.
- Documentation for schemas, configuration, artifacts, training, evaluation, continuation, deployment export, base-model licensing, and privacy boundaries.
- Versioned configuration and artifact schemas with explicit migrations.
- Clear device-specific troubleshooting for CUDA, MPS, and CPU.
- Continuous integration for formatting, typing, unit tests, and deterministic fixture smoke tests.

### Completion gate

A developer unfamiliar with the repository can:

1. run `equiroute init`;
2. validate the generated project;
3. train the tiny fixture router;
4. evaluate the exported artifact;
5. load `model/` with standard Transformers in a separate process; and
6. continue the router with an additional route.

They must accomplish this without modifying generated files or reading EquiRoute source code.

**MVP release line:** Stages 0–6 deliver EquiRoute's initial product: local, reproducible training and continuation of portable small routing models.

# Post-MVP: optional labeling harness

The labeling system follows the supervised pipeline. It must never be allowed to generate data before EquiRoute has a strong contract for validating and auditing labels.

## Stage 7 — Raw-input ingestion

### Deliver

Canonical unlabeled input JSONL:

```json
{
  "id": "chat-0104",
  "input": "I cannot log in after changing my phone number.",
  "metadata": {
    "source": "support-export"
  }
}
```

- Normalization into a canonical raw-row format.
- Opt-in field projection and redaction before data leaves the user's environment.
- Input-size validation and provenance preservation.
- No automatic training use.

### Completion gate

1. A fixture source normalizes deterministically.
2. Redaction rules remove matching sensitive fixture text before request construction.
3. Raw content is absent from standard logs unless explicit debug logging is enabled.
4. Invalid rows identify their source line and rejection reason.

## Stage 8 — OpenRouter-compatible candidate labeling

### Deliver

```bash
equiroute label labeling.yaml
```

- User-selected provider endpoint, model, policy prompt, credential environment variable, concurrency, and rate limits.
- Structured candidate labels using the same canonical route schema as human-authored examples.
- Local validation of every response.
- Prompt-injection-aware input handling: raw input is quoted data, not policy or schema.
- Recorded provenance: source ID, provider model, policy fingerprint, route-registry fingerprint, request status, and response timestamp.
- Bounded retry for transport and parse failures only.
- No API key persistence in configs, logs, exceptions, or artifacts.

Candidate labels are not trusted training rows.

### Completion gate

1. Recorded provider-response fixtures cover valid labels, malformed JSON, unknown route, refusal, timeout, and rate limiting without calling a real provider.
2. Every emitted candidate validates against the active route registry.
3. Invalid provider output produces a rejected candidate with a reason and cannot enter training.
4. Credential values never appear in logs or artifact files.
5. A manually invoked, sanitized live integration smoke test labels a small fixture set through a user-selected model.

## Stage 9 — Review, acceptance, and generated-data quality checks

### Deliver

- Reviewable JSONL/CSV report: input, candidate decision, validation result, provenance, and reviewer decision.
- `equiroute accept-labels` compiles only approved candidates into canonical training JSONL.
- Sampling, per-route quotas, and imbalance detection.
- Comparison of generated labels against a hand-labeled gold set.

### Completion gate

1. Rejected and unreviewed candidates cannot enter a training command.
2. Approved candidates pass the same Stage 1 validation as hand-authored labels.
3. Gold-set labeling quality is reported by route, including invalid-decision rate.
4. Route imbalance and candidate rejection rates are clear in the report.
5. Every accepted synthetic training row traces to source ID, policy version, provider model, and review decision.

# Non-negotiable quality rules

1. **EquiRoute is not a serving layer.** Its output is a model artifact that users deploy with standard inference tooling.
2. **Semantic evaluation only.** Measure structured route correctness, never only text similarity.
3. **No implicit fallback route.** Invalid model output is an evaluation failure; deployment fallback policy belongs to the user.
4. **No test-set tuning.** Validation selects; test certifies.
5. **No silent forgetting.** Continuation requires old-route replay coverage and regression evaluation.
6. **No accidental disclosure.** Provider labeling is opt-in; secrets and sensitive raw data require explicit controls.
7. **Reproducibility is part of the result.** A weights file without its pinned base revision, route registry, template, and evaluation report is incomplete.
8. **Keep the core small.** New model families, output adapters, exporters, or label providers must preserve the contracts above and solve a demonstrated user need.
