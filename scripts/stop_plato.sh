#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT/run/plato"
cd "$ROOT"

touch PAUSE

for pidfile in "$RUNTIME_DIR"/worker-gpu*.pid; do
    [[ -e "$pidfile" ]] || continue
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
        kill -INT "$pid" 2>/dev/null || true
    fi
done

for _ in $(seq 1 60); do
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

if [[ -f "$RUNTIME_DIR/scheduler.pid" ]]; then
    pid="$(cat "$RUNTIME_DIR/scheduler.pid")"
    if kill -0 "$pid" 2>/dev/null; then
        kill -INT "$pid" 2>/dev/null || true
    fi
    rm -f "$RUNTIME_DIR/scheduler.pid"
fi

echo "Plato workers and scheduler stopped."
echo "PAUSE remains present; remove it before the next start."
