# Model Selection for the Context-Management Study

**Status:** family-level plan restored; exact Gemma 4 subset requires clarification  
**Date:** 2026-08-05

## Authoritative design source

The model plan is the recommendation from Prof. Alexander Eck:

- six Qwen3.5 sizes, ranging from the 0.8B dense model to the 35B MoE model;
- four Gemma 4 sizes;
- ten models in total;
- every selected deployment must fit, including KV cache, within 32 GB VRAM;
- 8-bit weight quantization and 16-bit KV cache are the recommended provisioning target;
- roughly 50 GPU-hours per model, including prefill, for approximately 500 GPU-hours overall.

The previous Qwen2.5-Coder ladder plus four unrelated coding anchors was not part of this plan and has been withdrawn.

## Minimum, not ceiling

The ten-model programme is the **minimum successful research result** for this project. A pilot, a partial family, or a reduced model ladder is a technical validation stage and does not replace the minimum.

The project may be expanded during execution when time and compute remain available. Expansion is append-only: additional models, tasks, repetitions, scaffold variants, or diagnostics receive a new wave and protocol amendment. They may not narrow, replace, or endanger completion of the ten-model core. The binding scope and expansion policy are recorded in `docs/research_scope_and_resource_policy.md` and `experiments/research_scope.yml`.

## Confirmed Qwen3.5 ladder

The current official post-trained Qwen3.5 checkpoints matching the six-size description are:

1. `Qwen/Qwen3.5-0.8B`
2. `Qwen/Qwen3.5-2B`
3. `Qwen/Qwen3.5-4B`
4. `Qwen/Qwen3.5-9B`
5. `Qwen/Qwen3.5-27B`
6. `Qwen/Qwen3.5-35B-A3B`

The first five are dense checkpoints; the final checkpoint is a 35B-total, approximately 3B-active mixture-of-experts model.

## Gemma 4 discrepancy to resolve

The supervisor email describes **four Gemma 4 sizes, three of them MoE**. The currently published official Gemma 4 family does not match that description exactly. Official sources currently list five instruction-tuned sizes:

- `google/gemma-4-E2B-it`
- `google/gemma-4-E4B-it`
- `google/gemma-4-12B-it`
- `google/gemma-4-26B-A4B-it`
- `google/gemma-4-31B-it`

Current official architecture documentation classifies E2B, E4B, 12B and 31B as dense/PLE variants and 26B-A4B as the MoE variant. Therefore the exact four-model Gemma subset must not be guessed or frozen silently.

Likely explanations include a preliminary family description, a typo in the email, or an intended four-model subset constrained by 32 GB VRAM. This must be clarified before checkpoint IDs are locked.

## Hardware constraint

The scientific deployment condition is provisionally:

- one GPU with 32 GB VRAM;
- 8-bit weight quantization;
- 16-bit KV cache;
- common active context limit determined after measured memory profiling;
- model plus KV cache, runtime buffers and safety margin must fit without CPU offload.

The earlier Q4_K_M/llama.cpp assumption is no longer treated as the primary plan because it conflicts with the supervisor's explicit 8-bit recommendation. Backend and exact quantization format remain pending hardware and model-support validation.

## Shared-resource constraint

All shared compute is opportunistic. Other users have priority over this project. Production workers on shared machines are blocked until the external priority or preemption mechanism has been verified and documented. A pause or yield request releases the current scheduler lease at the next safe checkpoint, keeps completed runs, and returns the incomplete block to the queue.

## Inference settings

For each final checkpoint, official recommended sampling, thinking mode, chat template and tool-use configuration will be frozen. The same resolved profile is used across every scaffold, task, repeat and worker for that model.

## Freeze gate

No benchmark manifest may be generated until:

1. the four Gemma 4 checkpoints are confirmed;
2. every checkpoint and revision is pinned;
3. recommended inference settings are extracted from official sources;
4. an 8-bit artifact and 16-bit KV-cache configuration are validated;
5. measured peak VRAM, including prefill, remains below 32 GB with a safety margin;
6. artifact, tokenizer, template and profile hashes are recorded;
7. shared-resource priority is verified before any shared production worker is enabled.
