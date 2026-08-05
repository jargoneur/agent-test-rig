#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_VERSION="${AGENT_RIG_PYTHON:-3.12}"
REQUIRE_CUDA=0
PREPARE_CONTEXTBENCH=0
SKIP_LLAMA_BUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --require-cuda)
            REQUIRE_CUDA=1
            ;;
        --prepare-contextbench)
            PREPARE_CONTEXTBENCH=1
            ;;
        --skip-llama-build)
            SKIP_LLAMA_BUILD=1
            ;;
        *)
            echo "Unknown bootstrap argument: $1" >&2
            exit 2
            ;;
    esac
    shift
done

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

uv python install "$PYTHON_VERSION"
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python "$PYTHON_VERSION" .venv
fi

uv pip install --python .venv/bin/python -r requirements/core.txt

bash scripts/fetch_upstreams.sh

# Install the pinned packages without editable mode so their verified source
# checkouts stay clean. The harness imports the exact checkouts at runtime.
uv pip install --python .venv/bin/python \
    .upstreams/aider \
    .upstreams/swe-agent

if [[ "$SKIP_LLAMA_BUILD" -eq 0 ]]; then
    CUDA_FLAG=OFF
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_FLAG=ON
    elif [[ "$REQUIRE_CUDA" -eq 1 ]]; then
        echo "CUDA build requested, but nvcc was not found." >&2
        echo "The NVIDIA driver alone is insufficient; a CUDA toolkit is required." >&2
        exit 1
    fi

    CMAKE_BIN="$ROOT/.venv/bin/cmake"
    NINJA_BIN="$ROOT/.venv/bin/ninja"
    "$CMAKE_BIN" \
        -S .upstreams/llama.cpp \
        -B .upstreams/llama.cpp/build \
        -G Ninja \
        -DGGML_CUDA="$CUDA_FLAG" \
        -DLLAMA_CURL=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_MAKE_PROGRAM="$NINJA_BIN"
    "$CMAKE_BIN" --build .upstreams/llama.cpp/build --target llama-server llama-quantize -j
fi

if [[ "$PREPARE_CONTEXTBENCH" -eq 1 ]]; then
    .venv/bin/python scripts/prepare_contextbench.py --limit 150
fi

.venv/bin/python - <<'PY'
from harness.upstreams import upstream_path

for name in (
    "contextbench",
    "sweagent_last5",
    "aider_repomap",
    "agentless_localization",
    "aider_chat_summary",
    "llama_cpp_runtime",
):
    print(name, upstream_path(name))
PY

.venv/bin/python -m pytest \
    test_experiment_jobs.py \
    test_scheduler_store.py \
    test_model_adapter.py \
    test_model_profiles.py \
    test_resource_policy.py \
    test_worker_configs.py

echo
echo "Bootstrap complete."
echo "Python: $ROOT/.venv/bin/python"
echo "llama-server: $ROOT/.upstreams/llama.cpp/build/bin/llama-server"
