#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT/run/plato"
TOKEN_FILE="$RUNTIME_DIR/scheduler.token"

scheduler_running=0
if [[ -f "$RUNTIME_DIR/scheduler.pid" ]]; then
    scheduler_pid="$(cat "$RUNTIME_DIR/scheduler.pid")"
    kill -0 "$scheduler_pid" 2>/dev/null && scheduler_running=1
fi

if [[ "$scheduler_running" -eq 1 && -f "$TOKEN_FILE" ]]; then
    "$ROOT/.venv/bin/python" "$ROOT/scripts/scheduler_control.py" status \
        --scheduler-url http://127.0.0.1:8787 \
        --token-file "$TOKEN_FILE"
else
    echo "Scheduler: stopped"
fi

echo
echo "Recorded worker processes:"
found_worker=0
for pidfile in "$RUNTIME_DIR"/worker-gpu*.pid; do
    [[ -e "$pidfile" ]] || continue
    found_worker=1
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
        echo "  $(basename "$pidfile"): running (PID $pid)"
    else
        echo "  $(basename "$pidfile"): stopped (stale PID $pid)"
    fi
done
[[ "$found_worker" -eq 1 ]] || echo "  none"

echo
echo "Managed repository processes:"
managed="$(
    pgrep -af "(run_scheduled_worker.py|scheduler_server.py|$ROOT/.upstreams/llama.cpp/build/bin/llama-server)" ||
        true
)"
if [[ -n "$managed" ]]; then
    printf '%s\n' "$managed"
else
    echo "  none"
fi

echo
echo "Recorded service ports:"
ports=(8787)
if [[ -f "$RUNTIME_DIR/worker-ports.txt" ]]; then
    while IFS= read -r port; do
        [[ -n "$port" ]] && ports+=("$port")
    done < "$RUNTIME_DIR/worker-ports.txt"
fi
if command -v ss >/dev/null 2>&1; then
    for port in "${ports[@]}"; do
        listener="$(ss -ltnH "sport = :$port" 2>/dev/null || true)"
        if [[ -n "$listener" ]]; then
            echo "  $port: listening"
        else
            echo "  $port: closed"
        fi
    done
else
    echo "  ss unavailable"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    nvidia-smi
fi
