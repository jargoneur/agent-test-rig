# Agent Test Rig Operations

This is the operator procedure for the distributed ContextBench study. Workers
pull compatible paired blocks from one central scheduler; tasks, repeats and
scaffold conditions are not assigned manually.

## Runtime layout

- Main worker environment: `.venv`, Python 3.12.
- ContextBench evaluator environment: `.venv-contextbench`, Python 3.11.
- Frozen upstream sources: `.upstreams/`.
- Model artifacts and registry: `model_artifacts/`.
- Shared repository mirror cache: `.cache/repos/` by default.
- Authoritative scheduler database: `scheduler/*.sqlite3`.
- Online scheduler backups: `scheduler/backups/` by default.
- Central completed records: inside the scheduler database and its backups.
- Worker-local logs and resumable run files: `distributed_results/`.

The separate evaluator environment is intentional. Frozen Aider and frozen
ContextBench require incompatible tree-sitter versions.

## Resource rule for Plato

Prof. Alexander Eck has stated that project computation may run on Plato until
he contacts the user with a different instruction. No scheduler, reservation or
automatic interest-management system is available. Plato therefore runs in
`manual_operator` mode: the operator checks use before starting, keeps the run
observable and obeys a direct stop request. The harness does not claim automatic
preemption.

## Plato single-model storage rule

Only one model artifact is stored on Plato at a time. This is a Plato disk-space
policy, not a global distributed-execution restriction.

Consequences:

- `MODEL_IDS` on Plato must contain exactly one model ID;
- the same selected GGUF may be loaded by one or more Plato GPUs concurrently;
- other computers may store or execute different models at the same time;
- the scheduler database may contain pending blocks for all eleven models;
- after the selected model's Plato blocks are complete and the scheduler is
  stopped/backed up, its GGUF may be removed and replaced by the next model;
- registry and provenance records are preserved even when an old GGUF is
  removed.

The Plato launcher validates this rule and refuses to start if another
registered GGUF is still present.

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

The bootstrap installs user-local Python environments, fetches frozen upstreams,
compiles pinned llama.cpp, prepares the frozen 150-task ContextBench cache and
runs the compile and unit-test suite.

A CUDA build requires an installed toolkit containing `nvcc`. The version shown
by `nvidia-smi` describes driver compatibility and does not prove that the
compiler is installed.

## 2. Provision and validate a locked model

All model IDs, revisions, Q8_0 deployment fields, generation settings and native
context sizes come from experiments/model_profiles.lock.yml. Do not override
them manually. Provision one model into the Plato single-artifact slot:

    source .venv/bin/activate
    python scripts/provision_locked_model.py qwen3_5_0_8b

This downloads the exact frozen checkpoint, converts it with pinned llama.cpp,
quantizes it to Q8_0, records the immutable source-weight manifest and writes a
SHA-256 registry entry. The source snapshot and F16 intermediate are removed
after successful provisioning unless their keep flags were explicitly used.

With Plato otherwise stopped, select an idle GPU by exact UUID and run the
full native-context prefill validation:

    nvidia-smi -L
    python scripts/validate_model_profile.py qwen3_5_0_8b --gpu-uuid GPU-REPLACE-WITH-UUID

    python scripts/finalize_model_profile.py qwen3_5_0_8b --validation-evidence model_artifacts/qwen3_5_0_8b/validation-evidence.json

Validation refuses a busy GPU, validates the chat template, evaluates
native_context_size - 1 tokens without truncation, samples peak VRAM and
requires the frozen safety margin. Failed evidence is retained as a usable
deployment-boundary result, but it does not open the launch gate. Repeat this
process for all eleven profile keys, copying validated artifacts and their
registry and provenance records between hosts when necessary.

Before starting Plato, validate the single-model slot:

```bash
python scripts/plato_model_slot.py \
  --registry model_artifacts/registry.yml \
  --keep-model-id qwen3_5_0_8b
```

If another registered GGUF is present, the command reports it and exits without
deleting anything. Only after completed results and scheduler backups have been
confirmed, explicitly remove other registered artifacts:

```bash
python scripts/plato_model_slot.py \
  --registry model_artifacts/registry.yml \
  --keep-model-id qwen3_5_0_8b \
  --remove-other-artifacts
```

## 3. Full preflight

Select one free GPU UUID from `nvidia-smi -L`:

```bash
python scripts/preflight.py \
  --registry model_artifacts/registry.yml \
  --require-model qwen3_5_0_8b \
  --start-model qwen3_5_0_8b \
  --gpu GPU-REPLACE-WITH-UUID
```

Preflight verifies upstream commits, all five condition imports, the isolated
evaluator, exactly 150 cached tasks, artifact SHA-256, llama-server startup and
clean model shutdown on the selected GPU.

## 4. Generate the distributed smoke manifest

```bash
python plan_experiment.py \
  --config experiments/contextbench_plato_smoke.yml \
  --output jobs/contextbench_distributed_smoke.jsonl
```

Expected dimensions:

```text
8 tasks × 1 model × 5 conditions × 1 repeat = 40 runs
```

The scheduler groups these into eight paired blocks. Every block contains:

- no_scaffold
- `sweagent_last5`
- `aider_repomap`
- `agentless_localization`
- `aider_chat_summary`

## 5. Start Plato

Use a separate scheduler database for the smoke:

```bash
export SCHEDULER_DB=scheduler/contextbench_distributed_smoke.sqlite3
export MODEL_IDS=qwen3_5_0_8b
export PLATO_WORKERS=1
export PLATO_GPU_UUIDS=GPU-REPLACE-WITH-UUID
bash scripts/start_plato.sh jobs/contextbench_distributed_smoke.jsonl
```

`MODEL_IDS` must be exactly one model ID on Plato. `PLATO_WORKERS` defaults to
1. PLATO_GPU_UUIDS must contain the same number of exact, idle GPU UUIDs.
Increase the worker count only after the one-GPU smoke has passed and the operator
deliberately chooses to use more GPUs. Multiple Plato workers may load the same
selected GGUF on different GPUs; they still satisfy the one-artifact storage
rule.

The scheduler:

- binds to `127.0.0.1` and uses a bearer token;
- stores leases, workers and results in SQLite WAL mode;
- creates a validated startup backup;
- creates online backups every 15 minutes by default;
- retains the latest 96 backups by default;
- accepts exact duplicate completion uploads idempotently;
- refuses a conflicting result for an already completed block;
- persists central pause state across restarts.

Status:

```bash
bash scripts/status_plato.sh
```

The status output includes per-model pending, leased, completed and failed block
counts. Do not retire a Plato model while it has leased blocks.

Explicit scheduler controls:

```bash
TOKEN="$(cat run/plato/scheduler.token)"

.venv/bin/python scripts/scheduler_control.py status --token "$TOKEN"
.venv/bin/python scripts/scheduler_control.py pause --token "$TOKEN" \
  --reason "operator pause"
.venv/bin/python scripts/scheduler_control.py resume --token "$TOKEN"
.venv/bin/python scripts/scheduler_control.py backup --token "$TOKEN" \
  --label manual-checkpoint
```

A central pause prevents new claims. Heartbeats propagate the pause into each
worker's local pause file, so resource-policy checkpoints release an incomplete
paired block at the next safe model/tool checkpoint. A model call or tool
process already executing cannot be interrupted cooperatively.

Logs:

```bash
tail -f logs/plato/scheduler.log
tail -f logs/plato/worker-gpu*.log
tail -f logs/workers/plato-v100-gpu*/llama-server.log
```

Graceful stop:

```bash
bash scripts/stop_plato.sh
```

The stop procedure centrally pauses the queue, signals workers, requests a
backup, then stops the scheduler. The scheduler also attempts a final shutdown
backup. Completed runs remain stored; incomplete leases return to the queue.

## 6. Switch Plato to the next model

After the current model's desired blocks are complete:

1. inspect per-model status;
2. stop Plato gracefully;
3. confirm the pre-stop/shutdown backup exists;
4. provision or copy the next model;
5. remove the previous registered GGUF explicitly while keeping the next one;
6. run preflight for the next model;
7. restart with `MODEL_IDS` set to the next model ID.

Example after `qwen3_5_2b` has been provisioned:

```bash
python scripts/plato_model_slot.py \
  --registry model_artifacts/registry.yml \
  --keep-model-id qwen3_5_2b \
  --remove-other-artifacts

export MODEL_IDS=qwen3_5_2b
export PLATO_GPU_UUIDS=GPU-REPLACE-WITH-UUID
export SCHEDULER_DB=scheduler/contextbench_core.sqlite3
bash scripts/start_plato.sh jobs/contextbench_scaffold_boundaries_v1.jsonl
```

The same scheduler database can continue across model changes. Completed blocks
remain immutable; workers advertising the new single model claim only compatible
pending blocks.

## 7. Recover the scheduler database

The scheduler performs a full integrity check at startup and refuses to run a
corrupt database. Restore only while the scheduler is stopped.

Preserve the damaged/current file first, then restore a chosen backup:

```bash
mv scheduler/contextbench_distributed_smoke.sqlite3 \
   scheduler/contextbench_distributed_smoke.sqlite3.preserved

python scripts/restore_scheduler_backup.py \
  --backup scheduler/backups/SCHEDULER-BACKUP.sqlite3 \
  --database scheduler/contextbench_distributed_smoke.sqlite3
```

To replace an existing destination deliberately, add `--replace-existing`. The
restore command validates the backup and prints the restored queue status.

SQLite is sufficient for the current queue size. The protection required here
is consistent backup and restart recovery, not replacement with a larger
database system.

## 8. Attach an additional PC

Keep the central scheduler local to Plato and open one SSH tunnel per remote
machine:

```bash
ssh -N -L 8877:127.0.0.1:8787 jaron@10.50.200.80
```

A remote machine is not subject to Plato's one-artifact storage policy. It may
store and advertise any locally verified model set allowed by its own storage
and hardware envelope.

Synchronize exact model artifacts with resumable `rsync` and SHA-256 validation:

```bash
export ARTIFACT_SOURCE="jaron@10.50.200.80:~/agent-test-rig/model_artifacts/"
export MODEL_IDS=qwen3_5_0_8b
bash scripts/sync_model_artifacts.sh
```

Read the scheduler token through SSH and start one generic worker:

```bash
export SCHEDULER_TOKEN="$(
  ssh jaron@10.50.200.80 \
    'cat ~/agent-test-rig/run/plato/scheduler.token'
)"
export SCHEDULER_URL=http://127.0.0.1:8877
export MODEL_REGISTRY=model_artifacts/registry.yml
export MODEL_IDS=qwen3_5_0_8b
export WORKER_ID="$(hostname)-gpu0"
export WORKER_GPU=0
export LLAMA_PORT=8080
export RESOURCE_MODE=local_exclusive
export SHARED_RESOURCE=false

bash scripts/start_worker.sh
```

For another shared machine, declare its real mode and permission reference
instead of using `local_exclusive`. A worker advertises only artifacts that
exist locally and match the recorded SHA-256. Scheduler eligibility requires
both the concrete model ID and any declared resource class.

## 9. Export and evaluate results

After stopping the smoke system:

```bash
source .venv/bin/activate
rm -rf distributed_results/scheduler_export

python scheduler_server.py \
  --database scheduler/contextbench_distributed_smoke.sqlite3 \
  --export-results distributed_results/scheduler_export

python scripts/evaluate_contextbench.py \
  distributed_results/scheduler_export \
  --gold benchmarks/contextbench/gold.parquet \
  --output-dir results/contextbench_evaluation
```

Evaluation is performed independently for every model × scaffold × repeat
condition by the pinned ContextBench evaluator.

## 10. Scientific core

The binding base experiment contains two complete selected instruction-tuned
families:

```text
6 Qwen3.5 + 5 Gemma 4 = 11 models
11 × 150 tasks × 5 conditions × 3 repeats = 24,750 runs
```

The conditions are the no-scaffold line plus the four pinned scaffold
implementations. The complete manifest remains blocked until all eleven
artifact hashes, native-context deployment validations, and distributed
recovery evidence satisfy scripts/launch_gate.py. These scientific freeze gates
do not block a separately labelled infrastructure smoke.

After all eleven profiles have been finalized, create a commit so the harness is
clean and exact. Then generate machine-checked recovery evidence, open the
profile gate, and commit the resulting lock update:

    git status --short
    python scripts/run_distributed_recovery_validation.py
    python scripts/approve_model_profiles.py --distributed-recovery-evidence model_artifacts/validation/distributed-recovery.json
    git add experiments/model_profiles.lock.yml
    git commit -m Open_ContextBench_experiment_launch_gate

Only after that approval commit should the final manifest be generated and
checked. The manifest is deliberately generated, ignored state tied to the
current clean commit:

    python plan_experiment.py --config experiments/contextbench_core.yml
    python scripts/launch_gate.py --config experiments/contextbench_core.yml

Expected output is 24,750 jobs in 4,950 five-condition paired blocks and a gate
report with ok: true. A manual stop, resource yield or pause does not spend the
two-attempt infrastructure retry allowance; an interrupted block discards
all partial condition records and restarts all five conditions together.

The scientific launch remains an explicit operator action. Select exactly one
validated artifact and one or more idle GPU UUIDs, then start the core manifest:

    export MODEL_IDS=qwen3_5_0_8b
    export PLATO_WORKERS=1
    export PLATO_GPU_UUIDS=GPU-REPLACE-WITH-UUID
    export SCHEDULER_DB=scheduler/contextbench_core.sqlite3
    bash scripts/start_plato.sh jobs/contextbench_scaffold_boundaries_v1.jsonl

Do not run that command until the operator has reviewed the green launch-gate
report and explicitly decided to begin the large experiment.
