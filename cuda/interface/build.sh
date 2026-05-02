#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/build_logs"
mkdir -p "${LOG_DIR}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "error: could not find python or python3 on PATH" >&2
  exit 1
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/build_${STAMP}.log"

echo "repo_root=${REPO_ROOT}"
echo "python=${PYTHON_BIN}"
echo "log=${LOG_PATH}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" cuda/interface/setup.py build_ext --inplace 2>&1 | tee "${LOG_PATH}"
