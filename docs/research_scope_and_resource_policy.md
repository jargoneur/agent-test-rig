# Research Scope and Shared-Resource Policy

**Status:** binding project constraint  
**Date:** 2026-08-05

## Minimum successful research result

The ten-model programme proposed by Prof. Alexander Eck is the **minimum successful research result**, not an optional stretch goal:

- six Qwen3.5 model sizes;
- four Gemma 4 model sizes;
- the frozen main benchmark;
- every frozen main scaffold condition;
- the frozen repeat protocol;
- approximately 50 GPU-hours per model, or approximately 500 GPU-hours for the ten-model core.

Technical smoke tests, the pilot, partial model ladders, or results from only one family do not replace this minimum. They are validation stages used to make the minimum reliable.

## Expansion plan

The project is designed as a small complete core followed by append-only expansion waves.

1. **Validation wave:** prove that benchmark execution, upstream scaffolds, model profiles, result recovery, and memory limits work. This wave is not the research result.
2. **Minimum core wave:** complete the ten-model programme under one frozen protocol.
3. **Optional expansion waves:** add tasks, repetitions, models, scaffold variants, or diagnostic conditions only when available time and compute make them feasible without endangering completion of the minimum core.

Expansion never replaces, narrows, or silently changes the minimum core. Every expansion is appended with a new wave identifier and a protocol amendment. Existing completed runs remain immutable and reusable.

## Priority of other users

On every shared GPU or shared compute system, other users have priority over this project. Project jobs are opportunistic and must be preemptible or cooperatively yieldable.

The following rules are mandatory:

- no production worker may start on a shared resource until the external priority mechanism has been verified;
- the verification must have a written reference, such as an administrator confirmation, queue/QoS documentation, or an agreed operating procedure;
- shared workers must run in either `scheduler_preemptible` mode or `external_yield_signal` mode;
- a pause or external yield request must stop new work immediately and release the current scheduler lease at the next safe checkpoint;
- already completed runs are retained, while incomplete paired blocks return to the queue;
- model servers and GPU allocations must be released when the worker yields;
- expansion jobs have no stronger resource claim than minimum-core jobs and must never reduce access for other users.

## What the harness can and cannot guarantee

The harness can fail closed when priority has not been verified, check pause and availability signals before claiming work and between scaffold runs, preserve completed results, and release a partially completed block for later resumption.

The harness cannot independently prove that a university cluster or manually shared machine actually gives other users higher priority. That requires confirmation from the resource owner or administrator. For this reason, the example shared-worker configuration remains deliberately blocked until `priority_mechanism_verified: true` and a concrete `verification_reference` are supplied.

With cooperative signalling, the current maximum voluntary yield latency is one active scaffold run. A scheduler-managed preemption may interrupt sooner. A later implementation may add safe checkpoints inside a scaffold run, but the production policy does not rely on that future improvement.

## Operational evidence

Each shared worker records the resolved resource-policy metadata in its startup log. Production results must retain the worker configuration or its hash so that the priority mode and verification reference are auditable.
