# Agent Test Rig Operations

This document is the operator procedure for the distributed ContextBench study.
The operator starts and stops the system. The workers pull compatible paired
blocks from one central scheduler; tasks, repeats, and scaffold conditions are
not assigned manually.

## Runtime layout

- Main worker environment: `.venv`, Python 3.12.
- ContextBench evaluator environment: `.venv-contextbench`, Python 3.11.
- Frozen upstream sources: `.upstreams/`.
- Model artifacts and registry: `model_artifacts/`.
- Shared repository mirror cache: `.cache/repos/` by default.
- Scheduler databases: `scheduler/*.sqlite3`.
- Central completed records: inside the selected scheduler database.
- Worker-local logs and resumable run files: `distributed_results/`.

The separate evaluator environment is intentional. Frozen Aider and frozen
ContextBench require incompatible tree-sitter versions.

## 1. Checkout and bootstrap on Plato

```bash
cd ~

git clone \
  --branch distributed-workers \
  --single-branch \
  https://github.com/jargoneur/agent-test-rig.git

cd ~/agent-test-rig
bash scripts/bootstrap.sh --require-cuda --prepare-contextbench
```

For an existing checkout:

```bash
cd ~/agent-test-rig
git checkout distributed-workers
git pull --ff-only
bash scripts/bootstrap.sh --require-cuda --prepare-contextbench
```

The bootstrap performs all of the following:

1. installs user-local Python 3.12 and 3.11 through `uv`;
2. creates the worker and evaluator environments;
3. installs the runtime dependencies;
4. checks out every frozen upstream at its exact commit;
5. installs the frozen Aider dependency lock;
6. compiles the pinned llama.cpp `llama-server` and `llama-quantize`;
7. creates the frozen 150-task ContextBench cache and gold parquet;
8. runs the compile and unit-test suite.

A CUDA build requires an installed CUDA toolkit containing `nvcc`. The version
reported by `nvidia-smi` describes driver compatibility and does not prove that
the compiler is installed.

## 2. Provision a smoke-test model

Resolve and review the exact Hugging Face revision:

```bash
cd ~/agent-test-rig
source .venv/bin/activate

python scripts/resolve_model_revision.py Qwen/Qwen3.5-0.8B
```

Copy the printed 40-character `resolved_revision` into the next command:

```bash
python scripts/provision_gguf.py \
  --profile-key qwen3_5_0_8b \
  --model-id Qwen/Qwen3.5-0.8B \
  --revision REPLACE_WITH_RESOLVED_40_CHARACTER_SHA \
  --context-size 32768
```

This downloads the exact checkpoint, converts it with the pinned llama.cpp,
quantizes it to Q8_0, hashes the GGUF, and writes
`model_artifacts/registry.yml` plus a provenance record.

## 3. Full preflight

Select one free Plato GPU UUID from `nvidia-smi -L`, then run:

```bash
python scripts/preflight.py \
  --registry model_artifacts/registry.yml \
  --require-model qwen3_5_0_8b \
  --start-model qwen3_5_0_8b \
  --gpu GPU-REPLACE-WITH-UUID
```

The preflight verifies:

- every upstream checkout and commit;
- all four real scaffold imports;
- the isolated ContextBench evaluator;
- exactly 150 frozen benchmark tasks;
- model artifact existence and SHA-256;
- the llama-server binary;
- actual model loading and clean shutdown on the selected GPU.

## 4. Generate the four-GPU smoke manifest

```bash
python plan_experiment.py \
  --config experiments/contextbench_plato_smoke.yml \
  --output jobs/contextbench_plato_smoke.jsonl
```

Expected dimensions:

```text
4 ContextBench tasks × 1 model × 4 real scaffolds × 1 repeat = 16 runs
```

The scheduler groups these into four paired blocks, so all four Plato workers
receive real work. Every block contains all four conditions:

- `sweagent_last5`
- `aider_repomap`
- `agentless_localization`
- `aider_chat_summary`

## 5. Start Plato

Use a separate database for the final smoke test:

```bash
export SCHEDULER_DB=scheduler/contextbench_plato_smoke.sqlite3
bash scripts/start_plato.sh jobs/contextbench_plato_smoke.jsonl
```

This starts:

- one central SQLite-backed HTTP scheduler;
- four independent scheduled workers;
- one managed llama-server port per worker;
- one fixed V100 UUID per worker;
- automatic model loading, reuse, and block assignment.

No `--max-blocks` limit is used. Workers remain available for compatible work.
The operator-started shared-resource mode is recorded explicitly in every run.

Status:

```bash
bash scripts/status_plato.sh
```

Logs:

```bash
tail -f logs/plato/scheduler.log
tail -f logs/plato/worker-gpu*.log
tail -f logs/plato/llama-gpu*.log
```

Graceful stop:

```bash
bash scripts/stop_plato.sh
```

The stop procedure creates `PAUSE`, lets workers release their current blocks at
safe checkpoints, stops their model servers, and then stops the scheduler.
Completed runs remain in the scheduler database. Interrupted blocks return to
the queue.

## 6. Attach the local laptop to the same queue

On the laptop, a CPU llama.cpp build is sufficient for the 0.8B smoke model:

```bash
cd ~/agent-test-rig
git checkout distributed-workers
git pull --ff-only
bash scripts/bootstrap.sh --prepare-contextbench
```

Copy the smoke GGUF and `model_artifacts/registry.yml` from Plato, preserving the
same relative artifact path and SHA-256, or provision the exact same revision
locally.

Open and keep an SSH tunnel running:

```bash
ssh -N -L 8877:127.0.0.1:8787 jaron@10.50.200.80
```

In another local terminal:

```bash
cd ~/agent-test-rig
source .venv/bin/activate

export SCHEDULER_TOKEN="$(
  ssh jaron@10.50.200.80 \
    'cat ~/agent-test-rig/run/plato/scheduler.token'
)"
export SCHEDULER_URL=http://127.0.0.1:8877

bash scripts/start_local_worker.sh
```

The local registry determines which model IDs the laptop advertises. The
scheduler sends it only compatible paired blocks.

## 7. Export and evaluate completed results

After stopping the smoke system:

```bash
source .venv/bin/activate
rm -rf distributed_results/scheduler_export

python scheduler_server.py \
  --database scheduler/contextbench_plato_smoke.sqlite3 \
  --export-results distributed_results/scheduler_export

python scripts/evaluate_contextbench.py \
  distributed_results/scheduler_export \
  --gold benchmarks/contextbench/gold.parquet \
  --output-dir results/contextbench_evaluation
```

Evaluation is performed independently for every model × scaffold × repeat
condition by the pinned ContextBench evaluator.

## 8. Generate the provisional six-Qwen validation queue

After all six exact model artifacts and registry entries are provisioned:

```bash
python plan_experiment.py \
  --config experiments/contextbench_qwen_core.yml \
  --output jobs/contextbench_qwen3_5_provisional_validation.jsonl
```

This queue contains:

```text
6 models × 150 tasks × 4 scaffolds × 3 repeats = 10,800 runs
```

It is deliberately marked `provisional_validation_only`. It validates the full
operational scale but is not the frozen scientific core until exact checkpoint
revisions, artifact hashes, chat/thinking modes, and official per-checkpoint
generation profiles are locked. The full ten-model 18,000-run manifest also
requires confirmation of the four Gemma 4 checkpoint IDs.
