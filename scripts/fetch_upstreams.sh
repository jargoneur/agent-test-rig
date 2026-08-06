#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${UPSTREAM_ROOT:-$ROOT/.upstreams}"
mkdir -p "$DEST"

checkout_locked() {
    local name="$1"
    local url="$2"
    local commit="$3"
    local dir="$DEST/$name"

    if [[ ! -d "$dir/.git" ]]; then
        git clone --filter=blob:none --no-checkout "$url" "$dir"
    else
        local actual_url
        actual_url="$(git -C "$dir" remote get-url origin)"
        if [[ "$actual_url" != "$url" ]]; then
            echo "Refusing $dir: origin is $actual_url, expected $url" >&2
            exit 1
        fi
        if [[ -n "$(git -C "$dir" status --porcelain)" ]]; then
            echo "Refusing $dir: upstream checkout has local changes" >&2
            exit 1
        fi
    fi

    git -C "$dir" fetch --depth 1 origin "$commit"
    git -C "$dir" checkout --detach --force "$commit"

    local actual
    actual="$(git -C "$dir" rev-parse HEAD)"
    if [[ "$actual" != "$commit" ]]; then
        echo "Revision mismatch for $name: $actual != $commit" >&2
        exit 1
    fi
    if [[ -n "$(git -C "$dir" status --porcelain)" ]]; then
        echo "Checkout is not clean for $name" >&2
        exit 1
    fi

    printf '%-14s %s\n' "$name" "$actual"
}

checkout_locked contextbench \
    https://github.com/EuniAI/ContextBench.git \
    1436c28a8eb95496da4ea69ad458b9f8a8eb7d61

checkout_locked swe-agent \
    https://github.com/SWE-agent/SWE-agent.git \
    3ea751c087f32b16e039a2233dd6eefecef325d5

checkout_locked aider \
    https://github.com/Aider-AI/aider.git \
    5dc9490bb35f9729ef2c95d00a19ccd30c26339c

checkout_locked agentless \
    https://github.com/OpenAutoCoder/Agentless.git \
    5ce5888b9f149beaace393957a55ea8ee46c9f71

checkout_locked llama.cpp \
    https://github.com/ggml-org/llama.cpp.git \
    6ea215d171fd31df943bf1ac8227129f2b963160

echo "Locked upstream sources are available under: $DEST"
