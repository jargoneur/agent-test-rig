# Research Scope and Shared-Resource Policy

**Status:** binding project constraint  
**Date:** 2026-08-06

## Minimum successful research result

The base experiment compares two complete model families rather than a hand-picked ten-model subset:

- all six selected Qwen3.5 instruction checkpoints;
- all five official Gemma 4 instruction-tuned sizes;
- 150 frozen benchmark tasks;
- four frozen main scaffold conditions;
- three paired repeats;
- identical task/repeat seeds across scaffold conditions where supported.

This yields:

```text
11 models × 150 tasks × 4 scaffolds × 3 repeats = 19,800 runs
```

The earlier ten-model/18,000-run plan is superseded. Approximately 50 GPU-hours per model, including prefill, remains a planning estimate rather than a measured result; applying it linearly gives approximately 550 GPU-hours for the eleven-model core.

Technical smoke tests, the pilot, a partial model ladder, one model family, fewer tasks, or fewer scaffold conditions do not replace this minimum. They are validation stages used to make the minimum reliable.

## Model-family rule

The core contains the complete selected instruction-tuned ladders of both families at protocol freeze:

### Qwen3.5

- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.5-27B`
- `Qwen/Qwen3.5-35B-A3B`

### Gemma 4

- `google/gemma-4-E2B-it`
- `google/gemma-4-E4B-it`
- `google/gemma-4-12B-it`
- `google/gemma-4-26B-A4B-it`
- `google/gemma-4-31B-it`

Exact source revisions, inference profiles, quantized artifacts, tokenizer/template hashes and measured memory envelopes remain freeze-gate items. A model is not removed merely to preserve the obsolete ten-model count. If a model cannot fit the intended one-GPU envelope under the frozen deployment policy, that is documented as a deployment constraint and resolved explicitly rather than silently changing the family.

## Staged expansion plan

The project is built as a complete minimum core followed by append-only expansion waves. The eventual research objective is a multidimensional capability landscape over model size/capability, scaffold mechanism, task complexity, and computational cost.

### Wave 0: validation

Validate benchmark execution, upstream scaffold integration, official model profiles, VRAM including prefill and KV cache, result recovery, distributed execution, cooperative pause, and analysis code. This wave is not the research result.

### Wave 1: minimum core

Complete all 19,800 frozen core runs. This is the minimum successful result.

### Later expansion waves

Only after the minimum is protected and time/compute remain available, expand outward through separately identified waves. Candidate additions include:

- lower-difficulty real tasks and easy anchors;
- HANDBOOK-style delayed constraints, distractors and prohibited-action checks;
- ECBD-style validity checks for constructed or corrected variants;
- oracle-context diagnostics;
- matched-noise controls;
- task-difficulty metadata and scaffold ablations;
- additional models, tasks or repeats through a documented amendment.

Expansion never replaces, narrows, or silently changes the minimum core. Completed runs are immutable and reusable. Every extension receives a new wave identifier, an explicit rationale, and its own manifest hash.

## Plato operating permission

Prof. Alexander Eck has explicitly stated that project computation may run on Plato until he contacts the user with a different instruction. No scheduler, queue, reservation or automatic interest-management system is available to the user.

This establishes permission to compute; it does **not** create automatic preemption. Plato workers therefore use the explicit `manual_operator` policy:

- the operator checks current GPU use before starting;
- starts only the intended number of workers;
- keeps the run observable;
- obeys a direct stop or release request immediately;
- can centrally pause new claims and cooperatively release active blocks at safe checkpoints;
- preserves completed runs and returns incomplete paired blocks to the queue.

The verification reference recorded in worker configurations should state the professor's permission and its date, for example `prof_eck_permission_compute_until_contact_2026-08-05`.

## Other shared or additional computers

Additional PCs may be local-exclusive or shared. Every worker must declare its actual operating mode:

- `local_exclusive` for hardware controlled exclusively by the user;
- `scheduler_preemptible` when an external scheduler owns allocation and release;
- `external_yield_signal` when a verified external availability signal exists;
- `manual_operator` when explicit permission exists but no automatic management system is available and the worker is manually supervised.

A worker must advertise only model artifacts that exist locally and pass the recorded SHA-256 check. A block is eligible only when both the concrete model ID and any declared resource class match the worker.

## What the harness can and cannot guarantee

The harness can record the resolved resource policy, stop issuing new claims when centrally paused, check cooperative pause/yield signals, release scheduler leases, preserve completed run files, and recover expired blocks.

The harness cannot interrupt an inference request or tool process already executing. Its maximum cooperative stop latency is therefore the currently active model call or tool action. It also cannot infer that another person wants a GPU when no scheduler or external signal exists; manual supervision remains an operating responsibility.

## Operational evidence

Each run records the resolved resource-policy metadata. Records must retain the worker configuration or its hash, operating permission/reference, resource mode, release behavior, worker identity, model artifact hash and yield events. This keeps the actual execution conditions auditable without claiming an automatic priority mechanism that does not exist.
