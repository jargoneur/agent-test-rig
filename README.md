# Agent Test Rig

Distributed, resumable ContextBench execution for comparing context-management
scaffolds across local language models.

## Implemented stack

- central SQLite-backed pull scheduler with leases and heartbeats;
- selectable-GPU Plato launcher plus additional distributed workers;
- managed llama.cpp servers with one GPU per worker;
- exact GGUF artifact hashes and model registry;
- frozen 150-task ContextBench subset and pinned evaluator;
- no-scaffold comparison line plus real pinned scaffold implementations:
  - no scaffold (raw conversation history only);
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
`experiments/contextbench_plato_smoke.yml`, and start the selected workers with
`scripts/start_plato.sh`.

The complete commands, stop procedure, local-worker setup, and result evaluation
are in [`docs/operations.md`](docs/operations.md).

## Scientific protocol status

The 150-task benchmark and exact five conditions are frozen. The core contains
11 models × 150 tasks × 5 conditions × 3 paired repeats = 24,750 runs (4,950
whole paired blocks). Exact model revisions and official metadata are frozen;
the launch gate remains closed until every GGUF artifact, native-context
deployment profile, and distributed stop/recovery test is validated. Terminal
model and scaffold failures are retained as usable results; infrastructure
failures alone are retried, at most twice.
