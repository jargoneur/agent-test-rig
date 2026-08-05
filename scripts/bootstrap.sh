#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_VERSION="${AGENT_RIG_PYTHON:-3.12}"
EVAL_PYTHON_VERSION="${CONTEXTBENCH_PYTHON:-3.11}"
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

if [[ -n "${AGENT_RIG_STORAGE_SOFT_LIMIT_GIB:-}" ]]; then
    python3 scripts/storage_preflight.py \
        --soft-limit-gib "$AGENT_RIG_STORAGE_SOFT_LIMIT_GIB" \
        --minimum-filesystem-free-gib "${AGENT_RIG_STORAGE_RESERVE_GIB:-5}"
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

uv python install "$PYTHON_VERSION" "$EVAL_PYTHON_VERSION"

ensure_venv() {
    local path="$1"
    local requested="$2"
    local current=""
    if [[ -x "$path/bin/python" ]]; then
        current="$($path/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    fi
    if [[ "$current" != "$requested" ]]; then
        if [[ -d "$path" ]]; then
            echo "Recreating $path: Python ${current:-unknown} -> $requested"
            rm -rf "$path"
        fi
        uv venv --python "$requested" "$path"
    fi
}

ensure_venv .venv "$PYTHON_VERSION"
ensure_venv .venv-contextbench "$EVAL_PYTHON_VERSION"

uv pip install --python .venv/bin/python -r requirements/core.txt
uv pip install --python .venv-contextbench/bin/python -r requirements/contextbench-eval.txt

bash scripts/fetch_upstreams.sh

# Install dependency locks shipped by the frozen upstreams. Their source trees
# remain untouched and are imported from the verified commits.
uv pip install --python .venv/bin/python -r .upstreams/aider/requirements.txt
uv pip install --python .venv/bin/python \
    -r .upstreams/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

if [[ "$SKIP_LLAMA_BUILD" -eq 0 ]]; then
    CUDA_FLAG=OFF
    if command -v nvcc >/dev/null 2>&1; then
        export CUDACXX="$(command -v nvcc)"
        CUDA_FLAG=ON
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        export CUDACXX=/usr/local/cuda/bin/nvcc
        export PATH="/usr/local/cuda/bin:$PATH"
        CUDA_FLAG=ON
    elif [[ "$REQUIRE_CUDA" -eq 1 ]]; then
        echo "CUDA build requested, but nvcc was not found in PATH or /usr/local/cuda/bin." >&2
        echo "nvidia-smi reports driver capability only; a CUDA toolkit is also required." >&2
        exit 1
    fi

    CMAKE_BIN="$ROOT/.venv/bin/cmake"
    NINJA_BIN="$ROOT/.venv/bin/ninja"
    CMAKE_ARGS=(
        -S .upstreams/llama.cpp
        -B .upstreams/llama.cpp/build
        -G Ninja
        -DGGML_CUDA="$CUDA_FLAG"
        -DLLAMA_CURL=OFF
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_MAKE_PROGRAM="$NINJA_BIN"
    )
    if [[ "$CUDA_FLAG" == ON ]]; then
        CMAKE_ARGS+=( -DCMAKE_CUDA_COMPILER="$CUDACXX" )
    fi
    "$CMAKE_BIN" "${CMAKE_ARGS[@]}"
    "$CMAKE_BIN" --build .upstreams/llama.cpp/build \
        --target llama-server llama-quantize -j
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

CONTEXTBENCH_ROOT="$ROOT/.upstreams/contextbench"
PYTHONPATH="$CONTEXTBENCH_ROOT" .venv-contextbench/bin/python - <<'PY'
from contextbench.extractors.treesitter import available

if not available():
    raise SystemExit("ContextBench tree-sitter parser is not available")
print("ContextBench evaluator tree-sitter: available")
PY

.venv/bin/python -m compileall -q \
    agents harness scaffolds scripts \
    plan_experiment.py run_worker.py run_scheduled_worker.py \
    scheduler_server.py merge_results.py

.venv/bin/python -m pytest \
    test_experiment_jobs.py \
    test_scheduler_store.py \
    test_distributed_scheduler_store.py \
    test_model_adapter.py \
    test_model_profiles.py \
    test_resource_policy.py \
    test_worker_configs.py \
    test_contextbench_runtime.py \
    test_scaffold_runtime.py

echo
echo "Bootstrap complete."
echo "Worker Python: $ROOT/.venv/bin/python"
echo "Evaluator Python: $ROOT/.venv-contextbench/bin/python"
echo "llama-server: $ROOT/.upstreams/llama.cpp/build/bin/llama-server"
