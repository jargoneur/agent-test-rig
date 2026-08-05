#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT/run/plato"
TOKEN_FILE="$RUNTIME_DIR/scheduler.token"
cd "$ROOT"

TOKEN=""
[[ -f "$TOKEN_FILE" ]] && TOKEN="$(cat "$TOKEN_FILE")"

scheduler_alive=0
if [[ -f "$RUNTIME_DIR/scheduler.pid" ]]; then
    scheduler_pid="$(cat "$RUNTIME_DIR/scheduler.pid")"
    kill -0 "$scheduler_pid" 2>/dev/null && scheduler_alive=1
fi

if [[ "$scheduler_alive" -eq 1 && -n "$TOKEN" ]]; then
    .venv/bin/python scripts/scheduler_control.py pause \
        --scheduler-url http://127.0.0.1:8787 \
        --token "$TOKEN" \
        --reason "Plato operator shutdown" || true
fi

for pidfile in "$RUNTIME_DIR"/worker-gpu*.pid; do
    [[ -e "$pidfile" ]] || continue
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
        kill -INT "$pid" 2>/dev/null || true
    fi
done

for _ in $(seq 1 120); do
    alive=0
    for pidfile in "$RUNTIME_DIR"/worker-gpu*.pid; do
        [[ -e "$pidfile" ]] || continue
        pid="$(cat "$pidfile")"
        kill -0 "$pid" 2>/dev/null && alive=1
    done
    [[ "$alive" -eq 0 ]] && break
    sleep 1
done

for pidfile in "$RUNTIME_DIR"/worker-gpu*.pid; do
    [[ -e "$pidfile" ]] || continue
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
done

if [[ "$scheduler_alive" -eq 1 && -n "$TOKEN" ]]; then
    .venv/bin/python scripts/scheduler_control.py backup \
        --scheduler-url http://127.0.0.1:8787 \
        --token "$TOKEN" \
        --label before-stop || true
fi

if [[ -f "$RUNTIME_DIR/scheduler.pid" ]]; then
    pid="$(cat "$RUNTIME_DIR/scheduler.pid")"
    if kill -0 "$pid" 2>/dev/null; then
        kill -INT "$pid" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
    fi
    rm -f "$RUNTIME_DIR/scheduler.pid"
fi

echo "Plato workers and scheduler stopped."
echo "The scheduler was centrally paused and backed up before shutdown."
