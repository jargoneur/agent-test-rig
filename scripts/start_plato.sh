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
STORAGE_SOFT_LIMIT_GIB="${PLATO_STORAGE_SOFT_LIMIT_GIB:-80}"
STORAGE_RESERVE_GIB="${PLATO_STORAGE_RESERVE_GIB:-5}"

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

.venv/bin/python scripts/storage_preflight.py \
    --soft-limit-gib "$STORAGE_SOFT_LIMIT_GIB" \
    --minimum-filesystem-free-gib "$STORAGE_RESERVE_GIB"

mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
if [[ "$WORKER_COUNT" -gt "${#GPU_UUIDS[@]}" ]]; then
    echo "Requested $WORKER_COUNT workers but found ${#GPU_UUIDS[@]} NVIDIA GPUs." >&2
    exit 1
fi

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
    gpu="${GPU_UUIDS[$index]}"
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
        OPERATOR_ACKNOWLEDGEMENT="user_started_manually_under_prof_eck_permission" \
        VERIFICATION_REFERENCE="prof_eck_permission_compute_until_contact_2026-08-05" \
        MODEL_IDS="$MODEL_IDS" \
        RESOURCE_CLASSES="$RESOURCE_CLASSES" \
        WAVES="$WAVES" \
        bash scripts/start_worker.sh \
        > "$LOG_DIR/worker-gpu${index}.log" 2>&1 &
    echo "$!" > "$RUNTIME_DIR/worker-gpu${index}.pid"
done

sleep 3

echo "Plato system started."
echo "Manifest: $MANIFEST"
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
