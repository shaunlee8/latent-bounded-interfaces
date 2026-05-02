#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
MODEL_SCALE="${MODEL_SCALE:-canonical}"
SEQ_LEN="${SEQ_LEN:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TARGET_STEPS="${TARGET_STEPS:-20000}"
SEED="${SEED:-7}"
EVAL_EVERY="${EVAL_EVERY:-200}"
EVAL_BATCHES="${EVAL_BATCHES:-4}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-true}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-llama}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
REGION_SIZE="${REGION_SIZE:-2}"
MESSAGE_DIM="${MESSAGE_DIM:-64}"
MESSAGE_HIDDEN_DIM="${MESSAGE_HIDDEN_DIM:-0}"
MESSAGE_SCALE_INIT="${MESSAGE_SCALE_INIT:-0.5}"
if [[ -z "${LR_MODEL+x}" ]]; then
  case "${MODEL_SCALE}:${BACKBONE}" in
    canonical:mamba2)
      LR_MODEL="8e-4"
      ;;
    canonical:mamba3)
      LR_MODEL="3e-4"
      ;;
    canonical:hybrid)
      if [[ "${MESSAGE_DIM}" == "32" || "${MESSAGE_DIM}" == "64" ]]; then
        LR_MODEL="3e-4"
      else
        LR_MODEL="6e-4"
      fi
      ;;
    *:mamba2|*:transformer)
      LR_MODEL="6e-4"
      ;;
    *)
      LR_MODEL="3e-4"
      ;;
  esac
fi
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
if [[ -z "${WARMUP_STEPS+x}" ]]; then
  case "${MODEL_SCALE}:${BACKBONE}" in
    canonical:mamba3|canonical:hybrid)
      WARMUP_STEPS="500"
      ;;
    *)
      WARMUP_STEPS="1000"
      ;;
  esac
fi
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
if [[ -z "${WEIGHT_DECAY+x}" ]]; then
  case "${MODEL_SCALE}:${BACKBONE}" in
    canonical:mamba2|canonical:mamba3|canonical:hybrid)
      WEIGHT_DECAY="0.03"
      ;;
    *:mamba3)
      WEIGHT_DECAY="0.1"
      ;;
    *)
      WEIGHT_DECAY="0.01"
      ;;
  esac
fi
GRAD_CLIP="${GRAD_CLIP:-1.0}"
INTERFACE_JACOBIAN_MODE="${INTERFACE_JACOBIAN_MODE:-recompute}"
JACOBIAN_BASIS_CHUNK="${JACOBIAN_BASIS_CHUNK:-32}"

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
if [[ "${INTERFACE_JACOBIAN_MODE}" != "graph" && "${INTERFACE_JACOBIAN_MODE}" != "recompute" ]]; then
  echo "INTERFACE_JACOBIAN_MODE must be graph or recompute, got '${INTERFACE_JACOBIAN_MODE}'" >&2
  exit 1
fi
if [[ "${JACOBIAN_BASIS_CHUNK}" -le 0 ]]; then
  echo "JACOBIAN_BASIS_CHUNK must be > 0" >&2
  exit 1
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

FAMILY_NAME_DEFAULT="paper_${BACKBONE}_${ARCH_TAG}_${TOKENIZER_TAG}_${TIE_TAG}_seq${SEQ_LEN}_${STEP_TAG}"
FAMILY_NAME="${FAMILY_NAME:-${FAMILY_NAME_DEFAULT}}"
VARIANT_NAME="${VARIANT_NAME:-lbi_r${MESSAGE_DIM}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT%/}/${FAMILY_NAME}}"
RUN_NAME="${RUN_NAME:-${VARIANT_NAME}}"

TOKENS_PER_STEP=$((BATCH_SIZE * SEQ_LEN))
TOTAL_TOKENS=$((TARGET_STEPS * TOKENS_PER_STEP))

COMMON_ARGS=(
  -m train.train_region_interface
  --regime native_region_interface
  --backbone "${BACKBONE}"
  --output-dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
  --seed "${SEED}"
  --data-mode text_bpe_sharded
  --text-corpus fineweb_edu
  --tokenizer-type "${TOKENIZER_TYPE}"
  --vocab-size "${VOCAB_SIZE}"
  --seq-len "${SEQ_LEN}"
  --batch-size "${BATCH_SIZE}"
  --steps "${TARGET_STEPS}"
  --eval-every "${EVAL_EVERY}"
  --eval-batches "${EVAL_BATCHES}"
  --log-every "${LOG_EVERY}"
  --save-every "${SAVE_EVERY}"
  --lr-model "${LR_MODEL}"
  --lr-schedule "${LR_SCHEDULE}"
  --warmup-steps "${WARMUP_STEPS}"
  --min-lr-ratio "${MIN_LR_RATIO}"
  --weight-decay "${WEIGHT_DECAY}"
  --grad-clip "${GRAD_CLIP}"
  --region-size "${REGION_SIZE}"
  --message-dim "${MESSAGE_DIM}"
  --message-hidden-dim "${MESSAGE_HIDDEN_DIM}"
  --message-scale-init "${MESSAGE_SCALE_INIT}"
  --interface-jacobian-mode "${INTERFACE_JACOBIAN_MODE}"
  --jacobian-basis-chunk "${JACOBIAN_BASIS_CHUNK}"
)
if [[ "${TIE_EMBEDDINGS}" == "true" || "${TIE_EMBEDDINGS}" == "1" || "${TIE_EMBEDDINGS}" == "yes" ]]; then
  COMMON_ARGS+=(--tie-embeddings)
elif [[ "${TIE_EMBEDDINGS}" == "false" || "${TIE_EMBEDDINGS}" == "0" || "${TIE_EMBEDDINGS}" == "no" ]]; then
  COMMON_ARGS+=(--no-tie-embeddings)
else
  echo "TIE_EMBEDDINGS must be true/false, got '${TIE_EMBEDDINGS}'" >&2
  exit 1
fi
if [[ -n "${CHECKPOINT_ROOT}" ]]; then
  COMMON_ARGS+=(--checkpoint-root "${CHECKPOINT_ROOT}")
fi
if [[ -n "${TOKENIZER_PATH}" ]]; then
  COMMON_ARGS+=(--tokenizer-path "${TOKENIZER_PATH}")
fi

case "${MODEL_SCALE}:${BACKBONE}" in
  canonical:mamba2)
    BACKBONE_ARGS=(
      --layers 14
      --dim 768
      --d-state 64
      --d-conv 4
      --expand 2
      --headdim 64
      --ngroups 1
      --chunk-size 128
    )
    ;;
  canonical:mamba3)
    BACKBONE_ARGS=(
      --device cuda
      --dtype bfloat16
      --layers 14
      --dim 768
      --d-state 128
      --expand 2
      --headdim 64
      --ngroups 1
      --chunk-size 64
    )
    ;;
  canonical:transformer)
    BACKBONE_ARGS=(
      --layers 12
      --dim 512
      --n-heads 8
      --n-kv-heads 4
      --attn-head-dim 64
      --d-intermediate 2048
    )
    ;;
  canonical:hybrid)
    BACKBONE_ARGS=(
      --device cuda
      --dtype bfloat16
      --layers 12
      --layer-types "mamba3,mamba3,mamba3,transformer,mamba3,mamba3,mamba3,transformer,mamba3,mamba3,mamba3,transformer"
      --dim 768
      --d-state 128
      --d-conv 4
      --expand 2
      --headdim 64
      --ngroups 1
      --chunk-size 64
      --n-heads 12
      --n-kv-heads 6
      --attn-head-dim 64
      --d-intermediate 3072
    )
    ;;
  large:mamba2)
    BACKBONE_ARGS=(
      --layers 28
      --dim 768
      --d-state 64
      --d-conv 4
      --expand 2
      --headdim 64
      --ngroups 1
      --chunk-size 128
    )
    ;;
  large:mamba3)
    BACKBONE_ARGS=(
      --device cuda
      --dtype bfloat16
      --layers 28
      --dim 768
      --d-state 128
      --expand 2
      --headdim 64
      --ngroups 1
      --chunk-size 64
    )
    ;;
  large:transformer)
    BACKBONE_ARGS=(
      --layers 12
      --dim 768
      --n-heads 12
      --n-kv-heads 6
      --attn-head-dim 64
      --d-intermediate 3072
    )
    ;;
  large:hybrid)
    BACKBONE_ARGS=(
      --device cuda
      --dtype bfloat16
      --layers 20
      --layer-types "mamba3,mamba3,mamba3,transformer,mamba3,mamba3,mamba3,transformer,mamba3,mamba3,mamba3,transformer,mamba3,mamba3,mamba3,transformer,mamba3,mamba3,mamba3,transformer"
      --dim 768
      --d-state 128
      --d-conv 4
      --expand 2
      --headdim 64
      --ngroups 1
      --chunk-size 64
      --n-heads 12
      --n-kv-heads 6
      --attn-head-dim 64
      --d-intermediate 3072
    )
    ;;
  *)
    echo "Unsupported BACKBONE='${BACKBONE}'. Expected one of: mamba2, mamba3, transformer, hybrid" >&2
    exit 1
    ;;
esac

echo "Repo root: ${REPO_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Checkpoint root: ${CHECKPOINT_ROOT:-<under output dir>}"
echo "Backbone: ${BACKBONE}"
echo "Model scale: ${MODEL_SCALE}"
echo "Architecture tag: ${ARCH_TAG}"
echo "Family name: ${FAMILY_NAME}"
echo "Variant name: ${VARIANT_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Run name: ${RUN_NAME}"
echo "Target steps: ${TARGET_STEPS}"
echo "Total tokens: ${TOTAL_TOKENS}"
echo "Tokens/step: ${TOKENS_PER_STEP}"
echo "Step tag: ${STEP_TAG}"
echo "Tokenizer tag: ${TOKENIZER_TAG}"
echo "Tie tag: ${TIE_TAG}"
echo "Peak LR: ${LR_MODEL}"
echo "LR schedule: ${LR_SCHEDULE}"
echo "Warmup steps: ${WARMUP_STEPS}"
echo "Min LR ratio: ${MIN_LR_RATIO}"
echo "Weight decay: ${WEIGHT_DECAY}"
echo "Tie embeddings: ${TIE_EMBEDDINGS}"
echo "Tokenizer type: ${TOKENIZER_TYPE}"
echo "Tokenizer path: ${TOKENIZER_PATH:-<default>}"
echo "Requested vocab size: ${VOCAB_SIZE}"
echo "Region size: ${REGION_SIZE}"
echo "Message dim: ${MESSAGE_DIM}"
echo "Interface Jacobian mode: ${INTERFACE_JACOBIAN_MODE}"
echo "Jacobian basis chunk: ${JACOBIAN_BASIS_CHUNK}"
if [[ "${MESSAGE_HIDDEN_DIM}" -eq 0 ]]; then
  echo "Message hidden dim: infer from model dim"
else
  echo "Message hidden dim: ${MESSAGE_HIDDEN_DIM}"
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${COMMON_ARGS[@]}" "${BACKBONE_ARGS[@]}" "$@"
