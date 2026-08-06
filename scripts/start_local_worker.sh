#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${SCHEDULER_TOKEN:?Set SCHEDULER_TOKEN to the Plato scheduler token}"
SCHEDULER_URL="${SCHEDULER_URL:-http://127.0.0.1:8877}"
REGISTRY="${MODEL_REGISTRY:-model_artifacts/registry.yml}"
GPU="${LOCAL_GPU:-}"
PORT="${LOCAL_LLAMA_PORT:-8080}"
CONFIG="$ROOT/run/local-worker.yml"

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv. Run: bash scripts/bootstrap.sh --prepare-contextbench" >&2
    exit 1
fi
if [[ ! -f "$REGISTRY" ]]; then
    echo "Missing local model registry: $REGISTRY" >&2
    exit 1
fi
mkdir -p run logs/local
rm -f PAUSE

cat > "$CONFIG" <<YAML
worker_id: local-$(hostname)
scheduler_url: $SCHEDULER_URL
results_root: distributed_results/local-$(hostname)
lease_seconds: 14400
heartbeat_seconds: 30
idle_seconds: 10
stop_when_idle: false
retry_failed_blocks: true

resource_policy:
  shared_resource: false
  mode: local_exclusive
  pause_file: $ROOT/PAUSE

model_runtime:
  type: llama_cpp
  registry: $REGISTRY
  binary: .upstreams/llama.cpp/build/bin/llama-server
  host: 127.0.0.1
  port: $PORT
  gpu: "$GPU"
  startup_timeout_seconds: 1800
  shutdown_timeout_seconds: 30
  log_path: logs/local/llama-server.log

capabilities: {}
YAML

exec env SCHEDULER_TOKEN="$SCHEDULER_TOKEN" \
    .venv/bin/python run_scheduled_worker.py --config "$CONFIG"
