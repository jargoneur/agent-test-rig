# Agent Test Rig

Distributed, resumable ContextBench execution for comparing context-management
scaffolds across local language models.

## Implemented stack

- central SQLite-backed pull scheduler with leases and heartbeats;
- four-GPU Plato launcher plus an optional local laptop worker;
- managed llama.cpp servers with one GPU per worker;
- exact GGUF artifact hashes and model registry;
- frozen 150-task ContextBench subset and pinned evaluator;
- real pinned scaffold implementations:
  - SWE-agent `LastNObservations(n=5)`;
  - Aider `RepoMap`;
  - Agentless file and symbol localization;
  - Aider `ChatSummary`;
- per-run prompts, actions, context trajectories, patches, provenance, and results;
- independent ContextBench evaluation for every model × scaffold × repeat condition.

## Plato quick start

```bash
git checkout distributed-workers
git pull --ff-only
bash scripts/bootstrap.sh --require-cuda --prepare-contextbench
```

Then provision one exact model revision, run `scripts/preflight.py`, generate
`experiments/contextbench_plato_smoke.yml`, and start the four workers with
`scripts/start_plato.sh`.

The complete commands, stop procedure, local-worker setup, and result evaluation
are in [`docs/operations.md`](docs/operations.md).

## Scientific protocol status

The benchmark and four scaffold conditions are implemented and pinned. The full
18,000-run scientific manifest remains blocked only on the unresolved four Gemma
4 checkpoint IDs and final per-checkpoint model profiles. The six-Qwen manifest
is explicitly marked provisional validation rather than silently presented as a
frozen research run.
