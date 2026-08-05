#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${ARTIFACT_SOURCE:?Set ARTIFACT_SOURCE, for example user@host:/path/agent-test-rig/model_artifacts/}"
DESTINATION="${MODEL_ARTIFACT_DESTINATION:-$ROOT/model_artifacts}"
MODEL_IDS="${MODEL_IDS:-}"
VALIDATION_OUTPUT="${VALIDATION_OUTPUT:-$ROOT/run/artifact-sync/verified-registry.yml}"

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required for resumable artifact synchronization." >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv. Run scripts/bootstrap.sh first." >&2
    exit 1
fi

mkdir -p "$DESTINATION" "$(dirname "$VALIDATION_OUTPUT")"
source_path="${ARTIFACT_SOURCE%/}/"

rsync \
    --archive \
    --partial \
    --append-verify \
    --human-readable \
    --info=progress2 \
    "$source_path" \
    "$DESTINATION/"

registry="$DESTINATION/registry.yml"
if [[ ! -f "$registry" ]]; then
    echo "Synchronized directory does not contain registry.yml: $registry" >&2
    exit 1
fi

args=(--registry "$registry" --output "$VALIDATION_OUTPUT")
IFS=',' read -r -a requested <<< "$MODEL_IDS"
for model_id in "${requested[@]}"; do
    model_id="${model_id#${model_id%%[![:space:]]*}}"
    model_id="${model_id%${model_id##*[![:space:]]}}"
    [[ -n "$model_id" ]] && args+=(--model-id "$model_id")
done

.venv/bin/python scripts/prepare_worker_registry.py "${args[@]}"
echo "Artifact synchronization and SHA-256 validation completed."
echo "Validated registry: $VALIDATION_OUTPUT"
