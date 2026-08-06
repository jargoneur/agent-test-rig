#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MANIFEST="${1:-jobs/contextbench_smoke.jsonl}"
REGISTRY="${MODEL_REGISTRY:-model_artifacts/registry.yml}"
RUNTIME_DIR="$ROOT/run/plato"
LOG_DIR="$ROOT/logs/plato"
TOKEN_FILE="$RUNTIME_DIR/scheduler.token"
SCHEDULER_DB="${SCHEDULER_DB:-scheduler/contextbench.sqlite3}"
BACKUP_DIR="${SCHEDULER_BACKUP_DIR:-scheduler/backups}"
WORKER_COUNT="${PLATO_WORKERS:-1}"
MODEL_IDS="${MODEL_IDS:-}"
RESOURCE_CLASSES="${RESOURCE_CLASSES:-}"
WAVES="${WAVES:-}"
SELECTED_GPU_UUIDS_RAW="${PLATO_GPU_UUIDS:-}"
STORAGE_SOFT_LIMIT_GIB="${PLATO_STORAGE_SOFT_LIMIT_GIB:-80}"
STORAGE_RESERVE_GIB="${PLATO_STORAGE_RESERVE_GIB:-5}"

trim() {
    local value="$1"
    value="${value#${value%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"
    printf '%s' "$value"
}

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
if ! [[ "$WORKER_COUNT" =~ ^[0-9]+$ ]] || [[ "$WORKER_COUNT" -lt 1 ]]; then
    echo "PLATO_WORKERS must be a positive integer." >&2
    exit 1
fi
if [[ -z "$MODEL_IDS" ]]; then
    echo "Set MODEL_IDS to exactly one model ID for the active Plato model slot." >&2
    exit 1
fi
if [[ "$MODEL_IDS" == *,* ]]; then
    echo "Plato accepts exactly one active MODEL_IDS value, not a comma-separated list." >&2
    exit 1
fi
ACTIVE_MODEL_ID="$(trim "$MODEL_IDS")"
if [[ -z "$ACTIVE_MODEL_ID" ]]; then
    echo "MODEL_IDS resolved to an empty model ID." >&2
    exit 1
fi

if [[ -z "$SELECTED_GPU_UUIDS_RAW" ]]; then
    echo "Set PLATO_GPU_UUIDS explicitly after checking nvidia-smi." >&2
    echo "Example: export PLATO_GPU_UUIDS=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" >&2
    exit 1
fi
IFS=',' read -r -a RAW_SELECTED_GPU_UUIDS <<< "$SELECTED_GPU_UUIDS_RAW"
SELECTED_GPU_UUIDS=()
for raw_uuid in "${RAW_SELECTED_GPU_UUIDS[@]}"; do
    gpu_uuid="$(trim "$raw_uuid")"
    [[ -n "$gpu_uuid" ]] && SELECTED_GPU_UUIDS+=("$gpu_uuid")
done
if [[ "${#SELECTED_GPU_UUIDS[@]}" -ne "$WORKER_COUNT" ]]; then
    echo "PLATO_GPU_UUIDS contains ${#SELECTED_GPU_UUIDS[@]} UUIDs but PLATO_WORKERS=$WORKER_COUNT." >&2
    exit 1
fi

mapfile -t AVAILABLE_GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
declare -A AVAILABLE_GPU_SET=()
declare -A SELECTED_GPU_SET=()
for gpu_uuid in "${AVAILABLE_GPU_UUIDS[@]}"; do
    AVAILABLE_GPU_SET["$(trim "$gpu_uuid")"]=1
done
for gpu_uuid in "${SELECTED_GPU_UUIDS[@]}"; do
    if [[ -z "${AVAILABLE_GPU_SET[$gpu_uuid]:-}" ]]; then
        echo "Selected GPU UUID is not present on Plato: $gpu_uuid" >&2
        exit 1
    fi
    if [[ -n "${SELECTED_GPU_SET[$gpu_uuid]:-}" ]]; then
        echo "Selected GPU UUID is duplicated: $gpu_uuid" >&2
        exit 1
    fi
    SELECTED_GPU_SET["$gpu_uuid"]=1
done

ACTIVE_COMPUTE_UUIDS="$({
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null || true
} | sed '/^[[:space:]]*$/d')"
for gpu_uuid in "${SELECTED_GPU_UUIDS[@]}"; do
    if grep -Fxq "$gpu_uuid" <<< "$ACTIVE_COMPUTE_UUIDS"; then
        echo "Selected GPU already has an active compute process: $gpu_uuid" >&2
        echo "Choose another GPU after reviewing nvidia-smi." >&2
        exit 1
    fi
done

.venv/bin/python scripts/plato_model_slot.py \
    --registry "$REGISTRY" \
    --keep-model-id "$ACTIVE_MODEL_ID"

.venv/bin/python scripts/storage_preflight.py \
    --soft-limit-gib "$STORAGE_SOFT_LIMIT_GIB" \
    --minimum-filesystem-free-gib "$STORAGE_RESERVE_GIB"

mkdir -p "$RUNTIME_DIR" "$LOG_DIR" scheduler "$BACKUP_DIR"
rm -f "$RUNTIME_DIR"/PAUSE "$RUNTIME_DIR"/worker-*.pid

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
    --backup-directory "$BACKUP_DIR" \
    --backup-interval-seconds "${SCHEDULER_BACKUP_INTERVAL_SECONDS:-900}" \
    --backup-retain "${SCHEDULER_BACKUP_RETAIN:-96}" \
    > "$LOG_DIR/scheduler.log" 2>&1 &
echo "$!" > "$RUNTIME_DIR/scheduler.pid"

ready=0
for _ in $(seq 1 120); do
    if curl -fsS -H "Authorization: Bearer $TOKEN" \
        http://127.0.0.1:8787/health >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$(cat "$RUNTIME_DIR/scheduler.pid")" 2>/dev/null; then
        echo "Scheduler exited during startup. See $LOG_DIR/scheduler.log" >&2
        exit 1
    fi
    sleep 1
done
if [[ "$ready" -ne 1 ]]; then
    echo "Scheduler did not become healthy. See $LOG_DIR/scheduler.log" >&2
    exit 1
fi

# A clean stop persists the pause state. Starting a new supervised session is
# the explicit operator decision to resume claims.
.venv/bin/python scripts/scheduler_control.py resume \
    --scheduler-url http://127.0.0.1:8787 \
    --token "$TOKEN" >/dev/null

for index in $(seq 0 $((WORKER_COUNT - 1))); do
    gpu="${SELECTED_GPU_UUIDS[$index]}"
    port="$((8080 + index))"
    worker_id="plato-v100-gpu${index}"

    nohup env \
        SCHEDULER_URL="http://127.0.0.1:8787" \
        SCHEDULER_TOKEN="$TOKEN" \
        MODEL_REGISTRY="$REGISTRY" \
        WORKER_ID="$worker_id" \
        WORKER_GPU="$gpu" \
        LLAMA_PORT="$port" \
        RESOURCE_MODE="manual_operator" \
        SHARED_RESOURCE="true" \
        OTHER_USERS_PRIORITY="true" \
        OPERATOR_ACKNOWLEDGEMENT="user_started_manually_under_prof_eck_permission" \
        VERIFICATION_REFERENCE="prof_eck_permission_compute_until_contact_2026-08-05" \
        MODEL_IDS="$ACTIVE_MODEL_ID" \
        RESOURCE_CLASSES="$RESOURCE_CLASSES" \
        WAVES="$WAVES" \
        bash scripts/start_worker.sh \
        > "$LOG_DIR/worker-gpu${index}.log" 2>&1 &
    echo "$!" > "$RUNTIME_DIR/worker-gpu${index}.pid"
done

sleep 3

echo "Plato system started."
echo "Manifest: $MANIFEST"
echo "Active Plato model: $ACTIVE_MODEL_ID"
echo "Selected GPU UUIDs: ${SELECTED_GPU_UUIDS[*]}"
echo "Scheduler database: $SCHEDULER_DB"
echo "Backup directory: $BACKUP_DIR"
echo "Workers started: $WORKER_COUNT"
echo "Storage soft limit: $STORAGE_SOFT_LIMIT_GIB GiB"
echo "Token file: $TOKEN_FILE"
echo "Worker logs: $LOG_DIR/worker-gpu*.log"
echo
curl -fsS -H "Authorization: Bearer $TOKEN" \
    http://127.0.0.1:8787/status | .venv/bin/python -m json.tool
echo
nvidia-smi
