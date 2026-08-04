#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/build_logs"
mkdir -p "${LOG_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
STAMP="$(date +%Y%m%d_%H%M%S)"
cd "${SCRIPT_DIR}"
"${PYTHON_BIN}" setup.py build_ext --inplace 2>&1 | tee "${LOG_DIR}/build_${STAMP}.log"
