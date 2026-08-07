# Model Selection for the Context-Management Study

**Status:** exact revisions and official metadata frozen; artifacts and measured deployment validation pending
**Date:** 2026-08-07

## Binding study design

The base experiment compares two complete selected instruction-tuned model families:

- all six Qwen3.5 checkpoints in the selected official ladder;
- all five official Gemma 4 instruction-tuned sizes;
- eleven models in total.

The earlier ten-model plan and the attempt to choose four Gemma checkpoints are superseded. A model is not removed merely to preserve the old run count.

## Qwen3.5 ladder

1. `Qwen/Qwen3.5-0.8B`
2. `Qwen/Qwen3.5-2B`
3. `Qwen/Qwen3.5-4B`
4. `Qwen/Qwen3.5-9B`
5. `Qwen/Qwen3.5-27B`
6. `Qwen/Qwen3.5-35B-A3B`

## Gemma 4 ladder

1. `google/gemma-4-E2B-it`
2. `google/gemma-4-E4B-it`
3. `google/gemma-4-12B-it`
4. `google/gemma-4-26B-A4B-it`
5. `google/gemma-4-31B-it`

These five official instruction checkpoints define the Gemma 4 family used by the core experiment. The earlier supervisor wording describing four sizes and three MoE models does not match the official family and is no longer used as a selection rule.

## Core size

```text
11 models × 150 tasks × 5 conditions × 3 repeats = 24,750 runs
```

Cost is not a research factor. Runtime and resource use are operational
telemetry and do not determine model, task, or condition inclusion.

## Hardware and deployment constraint

The frozen deployment policy is:

- one GPU with 32 GB VRAM per worker;
- one exact Q8_0 GGUF artifact per model on every worker;
- each checkpoint's native advertised context length, not an arbitrary common
  context restriction;
- a frozen per-model KV-cache precision, currently Q8_0 for K and V;
- CPU offload and llama.cpp fit handling are allowed;
- the complete deployment profile is identical across all five conditions.

Every checkpoint must complete a full native-context prefill validation and
record measured peak VRAM plus its frozen fit safety target. If a model can run
under the frozen profile, it remains in the experiment; a terminal model or
context failure is retained as usable data rather than silently removing the
model.

## Distributed execution requirement

Every exact model artifact is content-addressed by SHA-256. A worker may advertise a model only when:

1. the artifact exists locally;
2. its hash matches the registry;
3. the local runtime binary is available;
4. the worker's declared resource class is sufficient.

The same exact artifact and frozen inference profile are used across all tasks, scaffolds, repeats and workers for a model.

## Inference settings

For each checkpoint, the following remain to be resolved and frozen from authoritative sources:

- exact source revision;
- recommended instruction/coding/agentic generation profile;
- tokenizer revision and hash;
- chat template and hash;
- thinking mode where applicable;
- quantized artifact hash;
- measured peak VRAM including prefill;
- validated native-context deployment profile.

## Freeze gate

The final 24,750-run manifest may be generated only after:

1. every source revision resolves to an exact immutable revision;
2. official generation profiles are recorded;
3. every source, tokenizer and template hash is recorded;
4. every int8 artifact is built and hashed;
5. the frozen per-model KV-cache profile is validated;
6. full native-context prefill and measured peak VRAM are recorded;
7. every worker advertises only locally verified artifacts;
8. current-head distributed recovery, stop/resume, backup/restore, and
   provenance-mismatch tests pass.

## Plato operating permission

Prof. Alexander Eck has stated that computation may run on Plato until he contacts the user. There is no user-visible scheduler or interest-management system. Plato therefore uses `manual_operator` mode and must not be described as automatically preemptible. The operator remains responsible for observing the machine and obeying a direct stop request.
