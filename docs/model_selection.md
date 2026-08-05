# Model Selection for the Context-Management Study

**Status:** selected, pending artifact and backend validation  
**Date:** 2026-08-05

## Selection objective

The model set must support two analyses at once:

1. a clean within-family estimate of how scaffold usefulness changes with model size;
2. a cross-family check that the observed effects are not unique to one architecture or training pipeline.

All selected models are instruction-tuned code models with openly downloadable checkpoints. The main experiment uses each checkpoint's official generation configuration while holding that configuration constant across all scaffold conditions.

## Selected set

### Controlled size ladder: Qwen2.5-Coder Instruct

- Qwen2.5-Coder-0.5B-Instruct
- Qwen2.5-Coder-1.5B-Instruct
- Qwen2.5-Coder-3B-Instruct
- Qwen2.5-Coder-7B-Instruct
- Qwen2.5-Coder-14B-Instruct
- Qwen2.5-Coder-32B-Instruct

These six checkpoints belong to one published code-model family and span roughly two orders of magnitude in parameter count. This makes them the primary basis for estimating model-size thresholds, diminishing scaffold returns, floor regions, and ceiling regions without changing the family at every size.

### External coding-model anchors

- bigcode/starcoder2-15b-instruct-v0.1
- deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- mistralai/Devstral-Small-2-24B-Instruct-2512
- Qwen/Qwen3-Coder-30B-A3B-Instruct

The anchors add different training pipelines and both dense and mixture-of-experts architectures. Devstral and Qwen3-Coder are explicitly aimed at agentic software-engineering use. DeepSeek-Coder-V2-Lite supplies a low-active-parameter MoE comparison, while StarCoder2-Instruct supplies an independently developed dense code model with a transparent self-alignment pipeline.

## Why the set is not ten unrelated models

Ten unrelated checkpoints would maximize vendor diversity but make the central scaling question hard to identify. A model-size effect would be inseparable from architecture, data, tokenizer, instruction tuning, and release generation.

The six-member Qwen2.5-Coder ladder supplies the controlled axis. The four anchors test external validity around that axis.

## Experimental interpretation

Within Qwen2.5-Coder, size-response curves are interpreted as within-family scaling evidence.

Across all ten models, comparisons are descriptive and model-conditioned. They test whether the location and magnitude of scaffold gains transfer across model families; they are not treated as a pure causal estimate of parameter count.

For MoE models, both total and active parameter counts are retained. Analyses will not place total and active parameter counts on one undifferentiated scalar axis without reporting the distinction.

## Inference and context controls

- Sampling follows the exact official checkpoint configuration recorded in `experiments/model_profiles.lock.yml`.
- Each model keeps one profile across tasks, scaffolds, repeats, workers, and backends.
- The common active input budget is 16,384 tokens.
- The common action-call output cap is 2,048 tokens.
- Larger native context windows and larger recommended maximum outputs are recorded but not granted as extra experimental budget.
- A secondary harmonized-decoding sensitivity sample is retained in the protocol.

## Deployment artifact policy

The canonical sources are the pinned official Hugging Face checkpoints. For cross-vendor NVIDIA/AMD execution, the planned common runtime is a pinned `llama.cpp` OpenAI-compatible server and one identical hashed GGUF artifact per model.

The target quantization is Q4_K_M, but the experiment remains blocked until conversion and numerical smoke tests succeed for every architecture. Unsupported or materially broken conversions trigger a protocol amendment; they are not replaced silently with community quantizations or a custom converter.

## Exclusions

- Base models are excluded because the action loop requires instruction following and structured tool actions.
- Closed API models are excluded because exact artifacts, tokenizers, templates, and backend behavior cannot be frozen locally.
- Deprecated short-context code families are excluded from the main set because they would introduce context-window truncation as an additional model-family confound.
- Reasoning models that require a separate thinking budget are excluded from the primary ladder. Qwen3-Coder-30B-A3B-Instruct is non-thinking-only and therefore fits the common action protocol.

## Freeze status

Checkpoint identities and decoding profiles are selected. The lock is not yet fully operational because model, tokenizer, chat-template, converted-artifact, and final profile hashes still require acquisition and validation. The validation gate in `experiments/model_profiles.lock.yml` prevents benchmark execution before those fields are resolved.
