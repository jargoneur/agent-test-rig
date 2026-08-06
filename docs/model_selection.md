# Model Selection for the Context-Management Study

**Status:** family membership fixed; exact revisions and deployment profiles pending  
**Date:** 2026-08-06

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
11 models × 150 tasks × 4 scaffolds × 3 repeats = 19,800 runs
```

Approximately 50 GPU-hours per model, including prefill, remains a planning estimate. Applied linearly, the eleven-model core is approximately 550 GPU-hours; this is not a measured runtime claim.

## Hardware and deployment constraint

The intended deployment envelope remains:

- one GPU with 32 GB VRAM per worker;
- 8-bit weight quantization target;
- 16-bit KV cache target;
- common active context limit determined after measured memory profiling;
- model, KV cache, runtime buffers and safety margin must fit without CPU offload for the one-GPU condition.

This envelope must be measured for every checkpoint. If a complete-family member does not fit under the intended profile, that result is documented as an explicit deployment constraint and resolved through a protocol decision. The model must not be silently removed.

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
- final common context limit.

## Freeze gate

The final 19,800-run manifest may be generated only after:

1. every source revision resolves to an exact immutable revision;
2. official generation profiles are recorded;
3. every source, tokenizer and template hash is recorded;
4. every int8 artifact is built and hashed;
5. FP16 KV-cache behavior is validated;
6. measured peak VRAM including prefill is recorded with safety margin;
7. every worker advertises only locally verified artifacts;
8. current-head distributed smoke tests pass.

## Plato operating permission

Prof. Alexander Eck has stated that computation may run on Plato until he contacts the user. There is no user-visible scheduler or interest-management system. Plato therefore uses `manual_operator` mode and must not be described as automatically preemptible. The operator remains responsible for observing the machine and obeying a direct stop request.
