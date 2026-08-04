#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${OUTPUT_ROOT+x}" ]]; then
  if [[ $# -gt 0 && "${1}" != --* ]]; then
    OUTPUT_ROOT="${1}"
    shift
  else
    OUTPUT_ROOT="out/region_interface"
  fi
fi

if [[ -z "${BACKBONE+x}" ]]; then
  if [[ $# -gt 0 && "${1}" != --* ]]; then
    BACKBONE="${1}"
    shift
  else
    BACKBONE="mamba3"
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_SCALE="${MODEL_SCALE:-canonical}"
SEQ_LEN="${SEQ_LEN:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TARGET_STEPS="${TARGET_STEPS:-5000}"
SEED="${SEED:-7}"
EVAL_EVERY="${EVAL_EVERY:-200}"
EVAL_BATCHES="${EVAL_BATCHES:-4}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-0}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LRS="${LRS:-1e-4 3e-4 6e-4}"
WEIGHT_DECAYS="${WEIGHT_DECAYS:-0.01 0.1}"
WARMUP_STEPS_LIST="${WARMUP_STEPS_LIST:-500 1000}"

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

sanitize_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//+}"
  value="${value//-neg}"
  printf "%s" "${value}"
}

if [[ "${BATCH_SIZE}" -le 0 ]]; then
  echo "BATCH_SIZE must be > 0" >&2
  exit 1
fi
if [[ "${SEQ_LEN}" -le 0 ]]; then
  echo "SEQ_LEN must be > 0" >&2
  exit 1
fi
if [[ "${TARGET_STEPS}" -le 0 ]]; then
  echo "TARGET_STEPS must be > 0" >&2
  exit 1
fi
if [[ "${MODEL_SCALE}" != "canonical" && "${MODEL_SCALE}" != "large" ]]; then
  echo "MODEL_SCALE must be canonical or large, got '${MODEL_SCALE}'" >&2
  exit 1
fi

STEP_TAG="$(step_tag "${TARGET_STEPS}")"
if [[ "${TOKENIZER_TYPE:-llama}" == "llama" && "${VOCAB_SIZE:-32000}" == "32000" ]]; then
  TOKENIZER_TAG="llama32k"
elif [[ "${TOKENIZER_TYPE:-llama}" == "llama31" ]]; then
  TOKENIZER_TAG="llama31"
else
  TOKENIZER_TAG="${TOKENIZER_TYPE:-llama}${VOCAB_SIZE:-32000}"
fi
if [[ "${TIE_EMBEDDINGS:-true}" == "true" || "${TIE_EMBEDDINGS:-true}" == "1" || "${TIE_EMBEDDINGS:-true}" == "yes" ]]; then
  TIE_TAG="tied"
else
  TIE_TAG="untied"
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
SWEEP_ROOT="${SWEEP_ROOT:-${OUTPUT_ROOT%/}/sweeps}"
FAMILY_NAME="${FAMILY_NAME:-dense_tuning_${BACKBONE}_${ARCH_TAG}_seq${SEQ_LEN}_${TOKENIZER_TAG}_${TIE_TAG}_${STEP_TAG}}"

read -r -a LR_VALUES <<< "${LRS}"
read -r -a WD_VALUES <<< "${WEIGHT_DECAYS}"
read -r -a WARMUP_VALUES <<< "${WARMUP_STEPS_LIST}"

echo "Sweep root: ${SWEEP_ROOT}"
echo "Backbone: ${BACKBONE}"
echo "Model scale: ${MODEL_SCALE}"
echo "Architecture tag: ${ARCH_TAG}"
echo "Family name: ${FAMILY_NAME}"
echo "Seed: ${SEED}"
echo "Target steps: ${TARGET_STEPS}"
echo "LRs: ${LRS}"
echo "Weight decays: ${WEIGHT_DECAYS}"
echo "Warmup steps: ${WARMUP_STEPS_LIST}"
echo "Tie embeddings: ${TIE_EMBEDDINGS:-true}"
echo "Tokenizer: ${TOKENIZER_TYPE:-llama}"
echo "Vocab size: ${VOCAB_SIZE:-32000}"
echo "Checkpoints: disabled"

for lr in "${LR_VALUES[@]}"; do
  for wd in "${WD_VALUES[@]}"; do
    for warmup in "${WARMUP_VALUES[@]}"; do
      lr_tag="$(sanitize_value "${lr}")"
      wd_tag="$(sanitize_value "${wd}")"
      variant="lr${lr_tag}_wd${wd_tag}_warm${warmup}_seed${SEED}"
      echo
      echo "Running sweep candidate: ${variant}"
      PYTHON_BIN="${PYTHON_BIN}" \
      OUTPUT_ROOT="${SWEEP_ROOT}" \
      BACKBONE="${BACKBONE}" \
      MODEL_SCALE="${MODEL_SCALE}" \
      FAMILY_NAME="${FAMILY_NAME}" \
      VARIANT_NAME="${variant}" \
      RUN_NAME="${variant}" \
      SEQ_LEN="${SEQ_LEN}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      TARGET_STEPS="${TARGET_STEPS}" \
      SEED="${SEED}" \
      EVAL_EVERY="${EVAL_EVERY}" \
      EVAL_BATCHES="${EVAL_BATCHES}" \
      LOG_EVERY="${LOG_EVERY}" \
      SAVE_EVERY="${SAVE_EVERY}" \
      LR_MODEL="${lr}" \
      LR_SCHEDULE="${LR_SCHEDULE}" \
      WARMUP_STEPS="${warmup}" \
      MIN_LR_RATIO="${MIN_LR_RATIO}" \
      WEIGHT_DECAY="${wd}" \
      GRAD_CLIP="${GRAD_CLIP}" \
      TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-true}" \
      TOKENIZER_TYPE="${TOKENIZER_TYPE:-llama}" \
      VOCAB_SIZE="${VOCAB_SIZE:-32000}" \
      "${SCRIPT_DIR}/train_dense_paper.sh" --no-save-checkpoints "$@"
    done
  done
done

echo
echo "Sweep complete. Summaries are under: ${SWEEP_ROOT}/${FAMILY_NAME}"
