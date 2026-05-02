#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-out/region_interface}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-out/evals/region_interface}"
BACKBONE="${BACKBONE:-mamba3}"
MODEL_SCALE="${MODEL_SCALE:-canonical}"
TARGET_STEPS="${TARGET_STEPS:-20000}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-llama}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-true}"
FAMILY_SEQ_LEN="${FAMILY_SEQ_LEN:-1024}"
CHECKPOINT="${CHECKPOINT:-best}"
EVAL_BATCHES="${EVAL_BATCHES:-512}"
EVAL_SEED="${EVAL_SEED:-12345}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-}"
SEQ_LEN="${SEQ_LEN:-}"

step_tag() {
  local steps="$1"
  if (( steps % 1000000 == 0 )); then
    printf "%dmsteps" "$((steps / 1000000))"
  elif (( steps % 1000 == 0 )); then
    printf "%dksteps" "$((steps / 1000))"
  else
    printf "%dsteps" "${steps}"
  fi
}

STEP_TAG="$(step_tag "${TARGET_STEPS}")"
if [[ "${TOKENIZER_TYPE}" == "llama" && "${VOCAB_SIZE}" == "32000" ]]; then
  TOKENIZER_TAG="llama32k"
elif [[ "${TOKENIZER_TYPE}" == "llama31" ]]; then
  TOKENIZER_TAG="llama31"
else
  TOKENIZER_TAG="${TOKENIZER_TYPE}${VOCAB_SIZE}"
fi
if [[ "${TIE_EMBEDDINGS}" == "true" || "${TIE_EMBEDDINGS}" == "1" || "${TIE_EMBEDDINGS}" == "yes" ]]; then
  TIE_TAG="tied"
else
  TIE_TAG="untied"
fi

if [[ "${MODEL_SCALE}" != "canonical" && "${MODEL_SCALE}" != "large" ]]; then
  echo "MODEL_SCALE must be canonical or large, got '${MODEL_SCALE}'" >&2
  exit 1
fi

case "${MODEL_SCALE}:${BACKBONE}" in
  canonical:mamba2|canonical:mamba3)
    ARCH_TAG="14l_768d"
    ;;
  canonical:transformer)
    ARCH_TAG="12l_512d"
    ;;
  canonical:hybrid)
    ARCH_TAG="12l_768d"
    ;;
  large:mamba2|large:mamba3)
    ARCH_TAG="28l_768d"
    ;;
  large:transformer)
    ARCH_TAG="12l_768d"
    ;;
  large:hybrid)
    ARCH_TAG="20l_768d"
    ;;
  *)
    echo "Unsupported BACKBONE='${BACKBONE}'. Expected one of: mamba2, mamba3, transformer, hybrid" >&2
    exit 1
    ;;
esac

FAMILY_NAME_DEFAULT="paper_${BACKBONE}_${ARCH_TAG}_${TOKENIZER_TAG}_${TIE_TAG}_seq${FAMILY_SEQ_LEN}_${STEP_TAG}"
FAMILY_NAME="${FAMILY_NAME:-${FAMILY_NAME_DEFAULT}}"
FAMILY_DIR="${FAMILY_DIR:-${OUTPUT_ROOT%/}/${FAMILY_NAME}}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${EVAL_OUTPUT_ROOT%/}/${FAMILY_NAME}}"

ARGS=(
  -m scripts.evaluate_paper_checkpoints
  --family "${FAMILY_DIR}"
  --output-dir "${EVAL_OUTPUT_DIR}"
  --checkpoint "${CHECKPOINT}"
  --eval-batches "${EVAL_BATCHES}"
  --eval-seed "${EVAL_SEED}"
  --device "${DEVICE}"
)

if [[ -n "${BATCH_SIZE}" ]]; then
  ARGS+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${SEQ_LEN}" ]]; then
  ARGS+=(--seq-len "${SEQ_LEN}")
fi
if [[ -n "${TOKENIZER_PATH}" ]]; then
  ARGS+=(--tokenizer-path "${TOKENIZER_PATH}")
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${ARGS[@]}" "$@"
