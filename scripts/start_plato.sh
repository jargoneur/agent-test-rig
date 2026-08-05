#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MANIFEST="${1:-jobs/contextbench_smoke.jsonl}"
REGISTRY="${MODEL_REGISTRY:-model_artifacts/registry.yml}"
RUNTIME_DIR="$ROOT/run/plato"
LOG_DIR="$ROOT/logs/plato"
RESULTS_ROOT="$ROOT/distributed_results"
TOKEN_FILE="$RUNTIME_DIR/scheduler.token"
SCHEDULER_DB="${SCHEDULER_DB:-scheduler/contextbench.sqlite3}"

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv. Run: bash scripts/bootstrap.sh --require-cuda --prepare-contextbench" >&2
    exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing manifest: $MANIFEST" >&2
    exit 1
fi
if [[ ! -f "$REGISTRY" ]]; then
    echo "Missing model registry: $REGISTRY" >&2
    exit 1
fi
if [[ ! -x .upstreams/llama.cpp/build/bin/llama-server ]]; then
    echo "Missing CUDA llama-server build. Run: bash scripts/bootstrap.sh --require-cuda" >&2
    exit 1
fi

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
if [[ "${#GPU_UUIDS[@]}" -lt 4 ]]; then
    echo "Expected at least four NVIDIA GPUs; found ${#GPU_UUIDS[@]}" >&2
    exit 1
fi

mkdir -p "$RUNTIME_DIR" "$LOG_DIR" "$RESULTS_ROOT" scheduler
rm -f PAUSE

if [[ -f "$RUNTIME_DIR/scheduler.pid" ]] && kill -0 "$(cat "$RUNTIME_DIR/scheduler.pid")" 2>/dev/null; then
    echo "Plato scheduler already appears to be running." >&2
    exit 1
fi

if [[ ! -f "$TOKEN_FILE" ]]; then
    .venv/bin/python - <<'PY' > "$TOKEN_FILE"
import secrets
print(secrets.token_hex(32))
PY
    chmod 600 "$TOKEN_FILE"
fi
TOKEN="$(cat "$TOKEN_FILE")"

nohup .venv/bin/python scheduler_server.py \
    --database "$SCHEDULER_DB" \
    --manifest "$MANIFEST" \
    --host 127.0.0.1 \
    --port 8787 \
    --token "$TOKEN" \
    > "$LOG_DIR/scheduler.log" 2>&1 &
echo "$!" > "$RUNTIME_DIR/scheduler.pid"

for attempt in $(seq 1 60); do
    if curl -fsS -H "Authorization: Bearer $TOKEN" \
        http://127.0.0.1:8787/health >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$(cat "$RUNTIME_DIR/scheduler.pid")" 2>/dev/null; then
        echo "Scheduler exited during startup. See $LOG_DIR/scheduler.log" >&2
        exit 1
    fi
    sleep 1
done

for index in 0 1 2 3; do
    gpu="${GPU_UUIDS[$index]}"
    port="$((8080 + index))"
    config="$RUNTIME_DIR/worker-gpu${index}.yml"

    cat > "$config" <<YAML
worker_id: plato-v100-gpu${index}
scheduler_url: http://127.0.0.1:8787
results_root: distributed_results/plato-v100-gpu${index}
lease_seconds: 14400
heartbeat_seconds: 30
idle_seconds: 10
stop_when_idle: false
retry_failed_blocks: true

resource_policy:
  shared_resource: true
  other_users_priority: true
  mode: manual_operator
  operator_acknowledgement: user_started_manually_after_observing_resources_available
  release_managed_externally: true
  pause_file: $ROOT/PAUSE

model_runtime:
  type: llama_cpp
  registry: $REGISTRY
  binary: .upstreams/llama.cpp/build/bin/llama-server
  host: 127.0.0.1
  port: $port
  gpu: $gpu
  startup_timeout_seconds: 1800
  shutdown_timeout_seconds: 30
  log_path: logs/plato/llama-gpu${index}.log

capabilities: {}
YAML

    nohup env SCHEDULER_TOKEN="$TOKEN" \
        .venv/bin/python run_scheduled_worker.py --config "$config" \
        > "$LOG_DIR/worker-gpu${index}.log" 2>&1 &
    echo "$!" > "$RUNTIME_DIR/worker-gpu${index}.pid"
done

sleep 3

echo "Plato system started."
echo "Manifest: $MANIFEST"
echo "Scheduler database: $SCHEDULER_DB"
echo "Token file: $TOKEN_FILE"
echo "Worker logs: $LOG_DIR/worker-gpu*.log"
echo
curl -fsS -H "Authorization: Bearer $TOKEN" \
    http://127.0.0.1:8787/status | .venv/bin/python -m json.tool
echo
nvidia-smi
