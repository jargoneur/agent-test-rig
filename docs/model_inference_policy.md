# Model Inference Policy

**Status:** protocol decision v1.0, frozen on 2026-08-05  
**Scope:** all evaluated model calls in the ContextBench scaffold study

## Primary rule

Each model is evaluated with the inference settings recommended by its own official model card, release documentation, or maintainer-provided reference configuration for coding, agentic tool use, or instruction following.

The study does **not** force one universal temperature or sampling profile across unrelated model families when that would contradict the model maintainer's recommended use.

## Fairness rule

For a given model artifact, the exact same locked inference profile is used across:

- all benchmark tasks;
- all five conditions, including no_scaffold;
- all repeats;
- all worker machines and inference backends.

Therefore scaffold effects are compared within a model under identical sampling conditions. Cross-model results describe **recommended-configured model systems**, not model architecture isolated from decoding configuration.

## Source priority

Settings are selected in this order:

1. official model card for the exact checkpoint;
2. official repository documentation pinned to a commit;
3. official inference example for the exact model family;
4. maintainer-provided serving template or generation configuration bundled with the checkpoint.

Community presets, benchmark-specific unofficial recipes, and locally tuned values are not accepted for the main experiment.

When an official source provides several profiles, the coding/agent/tool-use profile is selected. If no such profile exists, the general instruction-following profile is used. The choice and source quotation are recorded before any benchmark run.

## Locked fields

Every model profile must record, where applicable:

- exact model/checkpoint identifier and revision;
- model artifact hash;
- quantization method and quantized artifact hash;
- tokenizer revision and hash;
- chat template revision and hash;
- system-prompt handling;
- thinking/reasoning mode and its control tokens;
- `temperature`;
- `top_p`;
- `top_k`;
- `min_p`;
- repetition or presence/frequency penalties;
- maximum generated tokens;
- stop strings and stop token IDs;
- context length and any RoPE scaling configuration;
- seed support and seed value derivation;
- backend name, version, and relevant kernel/runtime options.

A field omitted by the official recommendation remains at the documented upstream/server default and is still logged. We do not invent a value merely to fill the schema.

## Auxiliary scaffold calls

Localization and summarization calls are part of their upstream scaffold implementations.

- When the upstream scaffold explicitly fixes decoding settings for an auxiliary call, those settings are retained and logged.
- Otherwise the same locked model profile used by the main action loop is used.
- No scaffold may receive a locally tuned temperature or sampling advantage.

## Reproducibility

Each run record must contain the full resolved inference profile and its profile hash. A worker must reject a job when its local model artifact, tokenizer, chat template, backend, or resolved inference settings do not match the manifest.

Paired runs use the same derived seed across scaffold conditions whenever the backend supports seeded generation. Backend nondeterminism is recorded rather than silently treated as deterministic behavior.

## Sensitivity analysis

The primary study uses recommended per-model profiles. A smaller preregistered sensitivity subset will additionally run a harmonized decoding profile across models where technically valid. This checks whether major model-size conclusions depend on model-specific recommendations.

The harmonized subset is secondary and does not replace the recommended-settings main analysis.

## Interpretation

Primary scaffold claims are causal within each model because decoding settings are held constant across scaffolds. Cross-model claims are about the performance and scaffold response of each model under its recommended deployment configuration.
