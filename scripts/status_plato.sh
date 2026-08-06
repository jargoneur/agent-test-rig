#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKEN_FILE="$ROOT/run/plato/scheduler.token"

if [[ ! -f "$TOKEN_FILE" ]]; then
    echo "No scheduler token found: $TOKEN_FILE" >&2
    exit 1
fi
TOKEN="$(cat "$TOKEN_FILE")"

curl -fsS -H "Authorization: Bearer $TOKEN" \
    http://127.0.0.1:8787/status | "$ROOT/.venv/bin/python" -m json.tool

echo
nvidia-smi
