#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
LBI_DATA_ROOT="${LBI_DATA_ROOT:-data}"
TASKS="${TASKS:-hellaswag,piqa,arc_easy}"
SPLIT="${SPLIT:-validation}"
LIMIT="${LIMIT:-0}"
SEED="${SEED:-12345}"
OUTPUT_DIR="${OUTPUT_DIR:-${LBI_DATA_ROOT%/}/downstream}"
CACHE_DIR="${CACHE_DIR:-${LBI_DATA_ROOT%/}/hf_cache}"
SHUFFLE="${SHUFFLE:-0}"

ARGS=(
  -m scripts.prepare_downstream_jsonl
  --tasks "${TASKS}"
  --split "${SPLIT}"
  --limit "${LIMIT}"
  --seed "${SEED}"
  --output-dir "${OUTPUT_DIR}"
  --cache-dir "${CACHE_DIR}"
)

if [[ "${SHUFFLE}" == "1" || "${SHUFFLE}" == "true" || "${SHUFFLE}" == "yes" ]]; then
  ARGS+=(--shuffle)
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${ARGS[@]}" "$@"
