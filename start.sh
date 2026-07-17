#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

fail() {
    printf '\nERROR: %s\n' "$1" >&2
    exit 1
}

command -v curl >/dev/null 2>&1 || fail "curl is not installed in WSL."
command -v python3 >/dev/null 2>&1 || fail "python3 is not installed in WSL."
command -v powershell.exe >/dev/null 2>&1 || fail "powershell.exe is not reachable from WSL."

WINDOWS_HOST="$(ip route | awk '/default/ {print $3; exit}')"
[[ -n "$WINDOWS_HOST" ]] || fail "Could not determine the Windows host address."

export OLLAMA_HOST="${OLLAMA_HOST:-http://${WINDOWS_HOST}:11434}"
export OLLAMA_HOST="${OLLAMA_HOST%/}"

ollama_ready() {
    curl --silent --show-error --fail --max-time 2 \
        "$OLLAMA_HOST/api/version" >/dev/null 2>&1
}

if ! ollama_ready; then
    echo "Starting Ollama on Windows ..."

    powershell.exe -NoProfile -NonInteractive -Command \
        "Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden" \
        >/dev/null 2>&1 || true

    for _ in $(seq 1 30); do
        if ollama_ready; then
            break
        fi
        sleep 1
    done
fi

if ! ollama_ready; then
    cat >&2 <<EOF

ERROR: Ollama did not become reachable at:
  $OLLAMA_HOST

Open PowerShell once and check:
  ollama serve

Then retry:
  bash start.sh
EOF
    exit 1
fi

OLLAMA_VERSION="$(
    curl --silent --show-error --fail "$OLLAMA_HOST/api/version" |
        python3 -c 'import json, sys; print(json.load(sys.stdin).get("version", "unknown"))'
)"

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
    echo "Creating Python virtual environment ..."
    python3 -m venv "$ROOT_DIR/.venv"
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"

MISSING_PACKAGES="$(python - <<'PY'
import importlib.util

required = {
    "requests": "requests",
    "yaml": "PyYAML",
    "pytest": "pytest",
}

print(" ".join(
    package
    for module, package in required.items()
    if importlib.util.find_spec(module) is None
))
PY
)"

if [[ -n "$MISSING_PACKAGES" ]]; then
    cat >&2 <<EOF

ERROR: The virtual environment is missing required packages:
  $MISSING_PACKAGES

Install them once with:
  .venv/bin/python -m pip install $MISSING_PACKAGES

Then retry:
  bash start.sh
EOF
    exit 1
fi

AVAILABLE_MODELS="$(
    curl --silent --show-error --fail "$OLLAMA_HOST/api/tags" |
        python -c 'import json, sys; print(" ".join(sorted(model.get("name", "") for model in json.load(sys.stdin).get("models", []))))'
)"

cat <<EOF

Agent test rig is ready.

Project:       $ROOT_DIR
Python:        $(python --version 2>&1)
Virtual env:   $VIRTUAL_ENV
Ollama host:   $OLLAMA_HOST
Ollama:        $OLLAMA_VERSION
Models:        ${AVAILABLE_MODELS:-none found}

Examples:
  python run_experiment.py --help
  python run_batch.py --help

Type 'exit' to leave this prepared shell.
EOF

if [[ $# -gt 0 ]]; then
    exec "$@"
fi

export PS1='(.venv) agent-test-rig:\w\$ '
exec bash --noprofile --norc -i
