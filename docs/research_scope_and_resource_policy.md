# Research Scope and Shared-Resource Policy

**Status:** binding project constraint  
**Date:** 2026-08-05

## Minimum successful research result

Prof. Alexander Eck's ten-model proposal is the **minimum successful research result**, not an optional stretch goal. The complete minimum is:

- six Qwen3.5 model sizes;
- four Gemma 4 model sizes;
- 150 frozen benchmark tasks;
- four frozen main scaffold conditions;
- three paired repeats;
- identical task/repeat seeds across scaffold conditions where supported;
- approximately 50 GPU-hours per model, including prefill;
- approximately 500 GPU-hours for the ten-model core.

This yields:

```text
10 models × 150 tasks × 4 scaffolds × 3 repeats = 18,000 runs
```

Technical smoke tests, the pilot, a partial model ladder, one model family, fewer tasks, or fewer scaffold conditions do not replace this minimum. They are validation stages used to make the minimum reliable.

## Staged expansion plan from the prior project design

The project is built as a complete minimum core followed by append-only expansion waves. The eventual research objective is a multidimensional capability landscape over model size/capability, scaffold mechanism, task complexity, and computational cost.

### Wave 0: validation

Validate benchmark execution, upstream scaffold integration, official model profiles, VRAM including prefill and KV cache, result recovery, resource yielding, and analysis code. This wave is not the research result.

### Wave 1: minimum core

Complete all 18,000 frozen core runs. This is the minimum successful result.

### Later expansion waves

Only after the minimum is protected and time/compute remain available, expand outward through separately identified waves. The prior design calls for:

- retaining the hard ContextBench tasks while adding lower-difficulty coverage and easy anchors;
- adding easier real repository tasks so model failure is observed gradually rather than only through a floor effect;
- adding HANDBOOK-style task variants that expose delayed constraints, distractors, and prohibited-action checks;
- using ECBD-style validity checks when constructing or correcting benchmark variants;
- adding oracle-context diagnostics to estimate the model ceiling when localization is solved;
- adding matched-noise controls so improvements are not confused with merely receiving more context;
- recording task-difficulty metadata and scaffold-taxonomy/ablation information;
- classifying regions as baseline-solvable, scaffold-sensitive, scaffold-enabled, oracle-only, or capability-limited;
- adding further models, repetitions, tasks, or scaffold mechanisms only through a documented protocol amendment.

Expansion never replaces, narrows, or silently changes the minimum core. Completed runs are immutable and reusable. Every extension receives a new wave identifier, an explicit rationale, and its own manifest hash.

## Priority of other users

On every shared GPU or shared compute system, other users have priority over this project. Project jobs are opportunistic and must be scheduler-preemptible or cooperatively yieldable.

The following rules are mandatory across every compute resource and worker entry point:

- every worker configuration must explicitly declare whether the resource is local-exclusive or shared;
- no production worker may start on a shared resource until the external priority mechanism has been verified;
- verification must have a written reference, such as administrator confirmation, queue/QoS documentation, or an agreed operating procedure;
- shared workers must use `scheduler_preemptible` or `external_yield_signal` mode;
- shared workers must declare how the model server and GPU allocation are released, either through an explicit release command or an externally managed scheduler allocation;
- a pause, scheduler signal, or external availability check must prevent new work immediately;
- during a run, the worker checks for yield before prompt construction, before and after each model call, and before tool execution;
- completed runs are retained, while an incomplete paired block is returned to the queue;
- optional expansion jobs have no stronger resource claim than minimum-core jobs and must never reduce other users' access.

## What the harness can and cannot guarantee

The harness now fails closed when a shared resource is not explicitly declared and verified. It records the resolved resource policy, checks pause/availability signals throughout execution, releases scheduler leases on yield, preserves completed runs, and requires a release mechanism for shared allocations.

The harness cannot independently prove that a university cluster or manually shared machine really grants other users higher priority. That guarantee must come from the resource owner or scheduler configuration. The shared Plato example is therefore deliberately blocked with `priority_mechanism_verified: false` until the real queue/QoS or operating procedure has been confirmed.

Cooperative yielding cannot interrupt an inference request or test process already executing. Its maximum voluntary latency is therefore the currently active model call or tool action. Scheduler-level preemption is the stronger guarantee and should be used wherever the infrastructure supports it.

## Operational evidence

Each run records the resolved resource-policy metadata. Production records must retain the worker configuration or its hash, the verification reference, priority mode, release mode, scheduler/job identifiers where available, and yield events. This makes it auditable whether every shared-resource run was executed under the agreed lower-priority policy.
