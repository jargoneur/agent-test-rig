#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT/run/plato"
TOKEN_FILE="$RUNTIME_DIR/scheduler.token"
cd "$ROOT"

pid_matches() {
    local pid="$1"
    local needle="$2"
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "$needle"
}

signal_managed_pid() {
    local pid="$1"
    local signal_name="$2"
    local needle="$3"
    if ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    if ! pid_matches "$pid" "$needle"; then
        echo "Refusing to signal reused or unexpected PID $pid (expected $needle)." >&2
        return 1
    fi
    kill "-$signal_name" "$pid" 2>/dev/null || true
}

workers_alive() {
    local pidfile pid
    for pidfile in "$RUNTIME_DIR"/worker-gpu*.pid; do
        [[ -e "$pidfile" ]] || continue
        pid="$(cat "$pidfile")"
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

wait_for_workers() {
    local seconds="$1"
    local _
    for _ in $(seq 1 "$seconds"); do
        workers_alive || return 0
        sleep 1
    done
    workers_alive && return 1
    return 0
}

scheduler_alive=0
scheduler_pid=""
if [[ -f "$RUNTIME_DIR/scheduler.pid" ]]; then
    scheduler_pid="$(cat "$RUNTIME_DIR/scheduler.pid")"
    kill -0 "$scheduler_pid" 2>/dev/null && scheduler_alive=1
fi

if [[ "$scheduler_alive" -eq 1 && -f "$TOKEN_FILE" ]]; then
    .venv/bin/python scripts/scheduler_control.py pause \
        --scheduler-url http://127.0.0.1:8787 \
        --token-file "$TOKEN_FILE" \
        --reason "Plato operator shutdown" || true
fi

for pidfile in "$RUNTIME_DIR"/worker-gpu*.pid; do
    [[ -e "$pidfile" ]] || continue
    pid="$(cat "$pidfile")"
    signal_managed_pid "$pid" INT "run_scheduled_worker.py" || true
done

if ! wait_for_workers 20; then
    echo "Workers did not exit after SIGINT; escalating to SIGTERM." >&2
    for pidfile in "$RUNTIME_DIR"/worker-gpu*.pid; do
        [[ -e "$pidfile" ]] || continue
        pid="$(cat "$pidfile")"
        signal_managed_pid "$pid" TERM "run_scheduled_worker.py" || true
    done
    wait_for_workers 10 || true
fi

if workers_alive; then
    echo "Workers did not exit after SIGTERM; escalating exact managed PIDs to SIGKILL." >&2
    for pidfile in "$RUNTIME_DIR"/worker-gpu*.pid; do
        [[ -e "$pidfile" ]] || continue
        pid="$(cat "$pidfile")"
        signal_managed_pid "$pid" KILL "run_scheduled_worker.py" || true
    done
fi

mapfile -t model_pids < <(
    pgrep -f "$ROOT/.upstreams/llama.cpp/build/bin/llama-server" || true
)
if [[ "${#model_pids[@]}" -gt 0 ]]; then
    echo "Stopping remaining managed llama-server processes: ${model_pids[*]}" >&2
    for pid in "${model_pids[@]}"; do
        pid_matches "$pid" "$ROOT/.upstreams/llama.cpp/build/bin/llama-server" &&
            kill -TERM "$pid" 2>/dev/null || true
    done
    for _ in $(seq 1 10); do
        remaining=0
        for pid in "${model_pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && remaining=1
        done
        [[ "$remaining" -eq 0 ]] && break
        sleep 1
    done
    for pid in "${model_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null &&
            pid_matches "$pid" "$ROOT/.upstreams/llama.cpp/build/bin/llama-server"; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
fi

if [[ "$scheduler_alive" -eq 1 && -f "$TOKEN_FILE" ]]; then
    .venv/bin/python scripts/scheduler_control.py backup \
        --scheduler-url http://127.0.0.1:8787 \
        --token-file "$TOKEN_FILE" \
        --label before-stop || true
fi

if [[ "$scheduler_alive" -eq 1 ]]; then
    signal_managed_pid "$scheduler_pid" INT "scheduler_server.py" || true
    for _ in $(seq 1 20); do
        kill -0 "$scheduler_pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$scheduler_pid" 2>/dev/null; then
        signal_managed_pid "$scheduler_pid" TERM "scheduler_server.py" || true
        for _ in $(seq 1 5); do
            kill -0 "$scheduler_pid" 2>/dev/null || break
            sleep 1
        done
    fi
    if kill -0 "$scheduler_pid" 2>/dev/null; then
        signal_managed_pid "$scheduler_pid" KILL "scheduler_server.py" || true
    fi
fi

sleep 1
errors=()
for pidfile in "$RUNTIME_DIR"/worker-gpu*.pid "$RUNTIME_DIR/scheduler.pid"; do
    [[ -e "$pidfile" ]] || continue
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
        errors+=("PID $pid from $pidfile is still alive")
    fi
done

for pattern in \
    "run_scheduled_worker.py" \
    "scheduler_server.py" \
    "$ROOT/.upstreams/llama.cpp/build/bin/llama-server"; do
    mapfile -t matches < <(pgrep -f "$pattern" || true)
    if [[ "${#matches[@]}" -gt 0 ]]; then
        errors+=("managed process pattern remains: $pattern (PIDs ${matches[*]})")
    fi
done

ports=(8787)
if [[ -f "$RUNTIME_DIR/worker-ports.txt" ]]; then
    while IFS= read -r port; do
        [[ -n "$port" ]] && ports+=("$port")
    done < "$RUNTIME_DIR/worker-ports.txt"
fi
if command -v ss >/dev/null 2>&1; then
    for port in "${ports[@]}"; do
        if ss -ltnH "sport = :$port" 2>/dev/null | grep -q .; then
            errors+=("TCP port $port is still listening")
        fi
    done
fi

if [[ "${#errors[@]}" -gt 0 ]]; then
    printf 'Plato shutdown verification failed:\n' >&2
    printf '  - %s\n' "${errors[@]}" >&2
    exit 1
fi

rm -f "$RUNTIME_DIR"/worker-gpu*.pid "$RUNTIME_DIR/scheduler.pid"
echo "Plato workers, model servers, and scheduler are stopped."
echo "No managed process or recorded service port remains active."

if [[ -f "$RUNTIME_DIR/selected-gpus.txt" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "Current compute processes on the previously selected GPUs (may belong to others):"
    while IFS= read -r gpu_uuid; do
        [[ -n "$gpu_uuid" ]] || continue
        nvidia-smi \
            --query-compute-apps=gpu_uuid,pid,process_name \
            --format=csv,noheader 2>/dev/null |
            grep -F "$gpu_uuid" || true
    done < "$RUNTIME_DIR/selected-gpus.txt"
fi
