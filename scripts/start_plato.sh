#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MANIFEST="${1:-jobs/contextbench_smoke.jsonl}"
REGISTRY="${MODEL_REGISTRY:-model_artifacts/registry.yml}"
CORE_CONFIG="$ROOT/experiments/contextbench_core.yml"
CORE_MANIFEST="$ROOT/jobs/contextbench_scaffold_boundaries_v1.jsonl"
RUNTIME_DIR="$ROOT/run/plato"
LOG_DIR="${PLATO_LOG_DIR:-$ROOT/logs/plato}"
TOKEN_FILE="$RUNTIME_DIR/scheduler.token"
SCHEDULER_DB="${SCHEDULER_DB:-scheduler/contextbench.sqlite3}"
BACKUP_DIR="${SCHEDULER_BACKUP_DIR:-scheduler/backups}"
WORKER_COUNT="${PLATO_WORKERS:-1}"
RESULTS_BASE="${PLATO_RESULTS_ROOT:-$ROOT/distributed_results}"
LEASE_SECONDS="${PLATO_LEASE_SECONDS:-28800}"
MODEL_IDS="${MODEL_IDS:-}"
RESOURCE_CLASSES="${RESOURCE_CLASSES:-}"
WAVES="${WAVES:-}"
SELECTED_GPU_UUIDS_RAW="${PLATO_GPU_UUIDS:-}"
SELECTED_GPU_GROUPS_RAW="${PLATO_GPU_GROUPS:-}"
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
if [[ "$(readlink -f "$MANIFEST")" == "$CORE_MANIFEST" ]]; then
    .venv/bin/python scripts/launch_gate.py \
        --config "$CORE_CONFIG"
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

if [[ -n "$SELECTED_GPU_UUIDS_RAW" && -n "$SELECTED_GPU_GROUPS_RAW" ]]; then
    echo "Set PLATO_GPU_UUIDS or PLATO_GPU_GROUPS, not both." >&2
    exit 1
fi
if [[ -z "$SELECTED_GPU_UUIDS_RAW" && -z "$SELECTED_GPU_GROUPS_RAW" ]]; then
    echo "Set exact idle GPUs with PLATO_GPU_UUIDS or PLATO_GPU_GROUPS." >&2
    exit 1
fi

SELECTED_GPU_GROUPS=()
SELECTED_GPU_UUIDS=()
if [[ -n "$SELECTED_GPU_GROUPS_RAW" ]]; then
    IFS=';' read -r -a RAW_GROUPS <<< "$SELECTED_GPU_GROUPS_RAW"
    for raw_group in "${RAW_GROUPS[@]}"; do
        group="$(trim "$raw_group")"
        [[ -n "$group" ]] || continue
        IFS=',' read -r -a RAW_GROUP_UUIDS <<< "$group"
        NORMALIZED_GROUP=()
        for raw_uuid in "${RAW_GROUP_UUIDS[@]}"; do
            gpu_uuid="$(trim "$raw_uuid")"
            [[ -n "$gpu_uuid" ]] && NORMALIZED_GROUP+=("$gpu_uuid")
        done
        if [[ "${#NORMALIZED_GROUP[@]}" -eq 0 ]]; then
            echo "PLATO_GPU_GROUPS contains an empty group." >&2
            exit 1
        fi
        normalized="$(IFS=,; echo "${NORMALIZED_GROUP[*]}")"
        SELECTED_GPU_GROUPS+=("$normalized")
        SELECTED_GPU_UUIDS+=("${NORMALIZED_GROUP[@]}")
    done
else
    IFS=',' read -r -a RAW_SELECTED_GPU_UUIDS <<< "$SELECTED_GPU_UUIDS_RAW"
    for raw_uuid in "${RAW_SELECTED_GPU_UUIDS[@]}"; do
        gpu_uuid="$(trim "$raw_uuid")"
        if [[ -n "$gpu_uuid" ]]; then
            SELECTED_GPU_GROUPS+=("$gpu_uuid")
            SELECTED_GPU_UUIDS+=("$gpu_uuid")
        fi
    done
fi
if [[ "${#SELECTED_GPU_GROUPS[@]}" -ne "$WORKER_COUNT" ]]; then
    echo "Selected GPU groups (${#SELECTED_GPU_GROUPS[@]}) do not match PLATO_WORKERS=$WORKER_COUNT." >&2
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

MODEL_GPU_COUNT="$(
    .venv/bin/python - "$REGISTRY" "$ACTIVE_MODEL_ID" <<'PY'
import pathlib
import sys
import yaml

registry = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
entry = registry["models"][sys.argv[2]]
print(int(entry.get("gpu_count") or 1))
PY
)"
for gpu_group in "${SELECTED_GPU_GROUPS[@]}"; do
    IFS=',' read -r -a GROUP_UUIDS <<< "$gpu_group"
    if [[ "${#GROUP_UUIDS[@]}" -ne "$MODEL_GPU_COUNT" ]]; then
        echo "Model $ACTIVE_MODEL_ID requires $MODEL_GPU_COUNT GPUs per worker; group $gpu_group has ${#GROUP_UUIDS[@]}." >&2
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
printf '%s\n' "${SELECTED_GPU_UUIDS[@]}" > "$RUNTIME_DIR/selected-gpus.txt.tmp"
mv -f "$RUNTIME_DIR/selected-gpus.txt.tmp" "$RUNTIME_DIR/selected-gpus.txt"
: > "$RUNTIME_DIR/worker-ports.txt.tmp"
for index in $(seq 0 $((WORKER_COUNT - 1))); do
    echo "$((8080 + index))" >> "$RUNTIME_DIR/worker-ports.txt.tmp"
done
mv -f "$RUNTIME_DIR/worker-ports.txt.tmp" "$RUNTIME_DIR/worker-ports.txt"

if [[ -f "$RUNTIME_DIR/scheduler.pid" ]] && kill -0 "$(cat "$RUNTIME_DIR/scheduler.pid")" 2>/dev/null; then
    echo "Plato scheduler already appears to be running." >&2
    exit 1
fi

umask 077
.venv/bin/python - <<'PY' > "$TOKEN_FILE.tmp"
import secrets
print(secrets.token_hex(32))
PY
chmod 600 "$TOKEN_FILE.tmp"
mv -f "$TOKEN_FILE.tmp" "$TOKEN_FILE"

STARTUP_COMPLETE=0
cleanup_failed_start() {
    if [[ "$STARTUP_COMPLETE" -ne 1 ]]; then
        echo "Plato startup failed; stopping any partially started services." >&2
        bash scripts/stop_plato.sh || true
    fi
}
trap cleanup_failed_start EXIT

nohup .venv/bin/python scheduler_server.py \
    --database "$SCHEDULER_DB" \
    --manifest "$MANIFEST" \
    --host 127.0.0.1 \
    --port 8787 \
    --token-file "$TOKEN_FILE" \
    --backup-directory "$BACKUP_DIR" \
    --backup-interval-seconds "${SCHEDULER_BACKUP_INTERVAL_SECONDS:-900}" \
    --backup-retain "${SCHEDULER_BACKUP_RETAIN:-96}" \
    > "$LOG_DIR/scheduler.log" 2>&1 &
echo "$!" > "$RUNTIME_DIR/scheduler.pid"

ready=0
for _ in $(seq 1 120); do
    if .venv/bin/python scripts/scheduler_control.py status \
        --scheduler-url http://127.0.0.1:8787 \
        --token-file "$TOKEN_FILE" >/dev/null 2>&1; then
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
    --token-file "$TOKEN_FILE" >/dev/null

for index in $(seq 0 $((WORKER_COUNT - 1))); do
    gpu_group="${SELECTED_GPU_GROUPS[$index]}"
    port="$((8080 + index))"
    worker_id="plato-v100-group${index}"

    nohup env \
        SCHEDULER_URL="http://127.0.0.1:8787" \
        SCHEDULER_TOKEN_FILE="$TOKEN_FILE" \
        MODEL_REGISTRY="$REGISTRY" \
        WORKER_ID="$worker_id" \
        WORKER_GPU="$gpu_group" \
        LLAMA_PORT="$port" \
        RESOURCE_MODE="manual_operator" \
        SHARED_RESOURCE="true" \
        OTHER_USERS_PRIORITY="true" \
        OPERATOR_ACKNOWLEDGEMENT="user_started_manually_under_prof_eck_permission" \
        VERIFICATION_REFERENCE="prof_eck_permission_compute_until_contact_2026-08-05" \
        MODEL_IDS="$ACTIVE_MODEL_ID" \
        RESOURCE_CLASSES="$RESOURCE_CLASSES" \
        WAVES="$WAVES" \
        RESULTS_ROOT="$RESULTS_BASE/$worker_id" \
        WORKER_LOG_DIR="$LOG_DIR/model-servers/$worker_id" \
        LEASE_SECONDS="$LEASE_SECONDS" \
        bash scripts/start_worker.sh \
        > "$LOG_DIR/worker-gpu${index}.log" 2>&1 &
    echo "$!" > "$RUNTIME_DIR/worker-gpu${index}.pid"
done

sleep 3
startup_failed=0
for index in $(seq 0 $((WORKER_COUNT - 1))); do
    pidfile="$RUNTIME_DIR/worker-gpu${index}.pid"
    pid="$(cat "$pidfile")"
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "Worker gpu${index} exited during startup." >&2
        tail -n 80 "$LOG_DIR/worker-gpu${index}.log" >&2 || true
        startup_failed=1
    fi
done
if [[ "$startup_failed" -ne 0 ]]; then
    exit 1
fi

echo "Plato system started."
echo "Manifest: $MANIFEST"
echo "Active Plato model: $ACTIVE_MODEL_ID"
echo "Selected GPU groups: ${SELECTED_GPU_GROUPS[*]}"
echo "Selected GPU UUIDs: ${SELECTED_GPU_UUIDS[*]}"
echo "Scheduler database: $SCHEDULER_DB"
echo "Backup directory: $BACKUP_DIR"
echo "Workers started: $WORKER_COUNT"
echo "Storage soft limit: $STORAGE_SOFT_LIMIT_GIB GiB"
echo "Token file: $TOKEN_FILE"
echo "Worker logs: $LOG_DIR/worker-gpu*.log"
echo
.venv/bin/python scripts/scheduler_control.py status \
    --scheduler-url http://127.0.0.1:8787 \
    --token-file "$TOKEN_FILE"
echo
nvidia-smi

STARTUP_COMPLETE=1
trap - EXIT
