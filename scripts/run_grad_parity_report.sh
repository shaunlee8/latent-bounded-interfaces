#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/srv/u2/shauncl/conda/envs/RSCAN/bin/python}"
PRESET="${PRESET:-report}"
OUTPUT_DIR="${OUTPUT_DIR:-out/grad_parity/${PRESET}}"
PLOT="${PLOT:-1}"
CASES="${CASES:-}"
SEEDS="${SEEDS:-}"
BATCHES_PER_SEED="${BATCHES_PER_SEED:-}"
INTERFACE_JACOBIAN_MODE="${INTERFACE_JACOBIAN_MODE:-recompute}"
JACOBIAN_BASIS_CHUNK="${JACOBIAN_BASIS_CHUNK:-32}"

if [[ "${INTERFACE_JACOBIAN_MODE}" != "graph" && "${INTERFACE_JACOBIAN_MODE}" != "recompute" ]]; then
  echo "INTERFACE_JACOBIAN_MODE must be graph or recompute, got '${INTERFACE_JACOBIAN_MODE}'" >&2
  exit 1
fi

if ! [[ "${JACOBIAN_BASIS_CHUNK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JACOBIAN_BASIS_CHUNK must be a positive integer, got '${JACOBIAN_BASIS_CHUNK}'" >&2
  exit 1
fi

ARGS=(
  -m scripts.generate_grad_parity_report
  --preset "${PRESET}"
  --output-dir "${OUTPUT_DIR}"
  --interface-jacobian-mode "${INTERFACE_JACOBIAN_MODE}"
  --jacobian-basis-chunk "${JACOBIAN_BASIS_CHUNK}"
)

if [[ "${PLOT}" != "0" ]]; then
  ARGS+=(--plot)
fi

if [[ -n "${CASES}" ]]; then
  read -r -a CASE_VALUES <<< "${CASES}"
  ARGS+=(--cases "${CASE_VALUES[@]}")
fi

if [[ -n "${SEEDS}" ]]; then
  read -r -a SEED_VALUES <<< "${SEEDS}"
  ARGS+=(--seeds "${SEED_VALUES[@]}")
fi

if [[ -n "${BATCHES_PER_SEED}" ]]; then
  ARGS+=(--batches-per-seed "${BATCHES_PER_SEED}")
fi

echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Preset: ${PRESET}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Plot: ${PLOT}"
echo "Cases: ${CASES:-<all ${PRESET} cases>}"
echo "Seeds: ${SEEDS:-<script default>}"
echo "Batches/seed: ${BATCHES_PER_SEED:-<script default>}"
echo "Interface Jacobian mode: ${INTERFACE_JACOBIAN_MODE}"
echo "Jacobian basis chunk: ${JACOBIAN_BASIS_CHUNK}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${ARGS[@]}" "$@"
