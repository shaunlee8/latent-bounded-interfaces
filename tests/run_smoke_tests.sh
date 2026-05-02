#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/out/smoke"
mkdir -p "${OUT_DIR}"

LOG_PATH="${OUT_DIR}/pytest_smoke.log"
XML_PATH="${OUT_DIR}/pytest_smoke.xml"

cd "${REPO_ROOT}"
python -m pytest -vv -s tests/test_training_workflow_smoke.py \
  --junitxml="${XML_PATH}" 2>&1 | tee "${LOG_PATH}"
