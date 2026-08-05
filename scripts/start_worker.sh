#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${SCHEDULER_URL:?Set SCHEDULER_URL to the central scheduler URL}"
: "${SCHEDULER_TOKEN:?Set SCHEDULER_TOKEN to the scheduler token}"

SOURCE_REGISTRY="${MODEL_REGISTRY:-model_artifacts/registry.yml}"
WORKER_ID="${WORKER_ID:-$(hostname)-gpu${WORKER_GPU:-none}}"
WORKER_GPU="${WORKER_GPU:-}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
RESOURCE_MODE="${RESOURCE_MODE:-local_exclusive}"
SHARED_RESOURCE="${SHARED_RESOURCE:-false}"
OPERATOR_ACKNOWLEDGEMENT="${OPERATOR_ACKNOWLEDGEMENT:-}"
VERIFICATION_REFERENCE="${VERIFICATION_REFERENCE:-}"
RESOURCE_CLASSES="${RESOURCE_CLASSES:-}"
WAVES="${WAVES:-}"
MODEL_IDS="${MODEL_IDS:-}"
RUN_DIR="$ROOT/run/workers/$WORKER_ID"
CONFIG="$RUN_DIR/worker.yml"
FILTERED_REGISTRY="$RUN_DIR/verified-registry.yml"
PAUSE_FILE="$RUN_DIR/PAUSE"

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv. Run scripts/bootstrap.sh first." >&2
    exit 1
fi
if [[ ! -x .upstreams/llama.cpp/build/bin/llama-server ]]; then
    echo "Missing llama-server. Run scripts/bootstrap.sh first." >&2
    exit 1
fi
if [[ ! -f "$SOURCE_REGISTRY" ]]; then
    echo "Missing model registry: $SOURCE_REGISTRY" >&2
    exit 1
fi

mkdir -p "$RUN_DIR" "logs/workers/$WORKER_ID"
rm -f "$PAUSE_FILE"

VALIDATE_ARGS=(
    --registry "$SOURCE_REGISTRY"
    --output "$FILTERED_REGISTRY"
)
IFS=',' read -r -a REQUESTED_MODELS <<< "$MODEL_IDS"
for model_id in "${REQUESTED_MODELS[@]}"; do
    model_id="${model_id#${model_id%%[![:space:]]*}}"
    model_id="${model_id%${model_id##*[![:space:]]}}"
    if [[ -n "$model_id" ]]; then
        VALIDATE_ARGS+=(--model-id "$model_id")
    fi
done
.venv/bin/python scripts/prepare_worker_registry.py "${VALIDATE_ARGS[@]}"

MODEL_LIST="$($ROOT/.venv/bin/python - "$FILTERED_REGISTRY" <<'PY'
import sys, yaml
value = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
for model_id in sorted(value['models']):
    print(model_id)
PY
)"

cat > "$CONFIG" <<YAML
worker_id: $WORKER_ID
scheduler_url: $SCHEDULER_URL
results_root: distributed_results/$WORKER_ID
lease_seconds: 14400
heartbeat_seconds: 30
idle_seconds: 10
stop_when_idle: false
retry_failed_blocks: true
scheduler_timeout_seconds: 60
completion_timeout_seconds: 300
completion_retries: 8

resource_policy:
  shared_resource: $SHARED_RESOURCE
  mode: $RESOURCE_MODE
  pause_file: $PAUSE_FILE
  operator_acknowledgement: "$OPERATOR_ACKNOWLEDGEMENT"
  verification_reference: "$VERIFICATION_REFERENCE"
  release_managed_externally: true

model_runtime:
  type: llama_cpp
  registry: $FILTERED_REGISTRY
  binary: .upstreams/llama.cpp/build/bin/llama-server
  host: 127.0.0.1
  port: $LLAMA_PORT
  gpu: "$WORKER_GPU"
  startup_timeout_seconds: 1800
  shutdown_timeout_seconds: 30
  log_path: logs/workers/$WORKER_ID/llama-server.log

capabilities:
  model_ids:
YAML
while IFS= read -r model_id; do
    [[ -n "$model_id" ]] && printf '    - %s\n' "$model_id" >> "$CONFIG"
done <<< "$MODEL_LIST"

if [[ -n "$RESOURCE_CLASSES" ]]; then
    printf '  resource_classes:\n' >> "$CONFIG"
    IFS=',' read -r -a CLASSES <<< "$RESOURCE_CLASSES"
    for value in "${CLASSES[@]}"; do
        [[ -n "$value" ]] && printf '    - %s\n' "$value" >> "$CONFIG"
    done
fi
if [[ -n "$WAVES" ]]; then
    printf '  waves:\n' >> "$CONFIG"
    IFS=',' read -r -a WAVE_VALUES <<< "$WAVES"
    for value in "${WAVE_VALUES[@]}"; do
        [[ -n "$value" ]] && printf '    - %s\n' "$value" >> "$CONFIG"
    done
fi

echo "Starting worker $WORKER_ID"
echo "Config: $CONFIG"
exec env SCHEDULER_TOKEN="$SCHEDULER_TOKEN" \
    .venv/bin/python run_scheduled_worker.py \
    --config "$CONFIG" \
    --pause-file "$PAUSE_FILE"
