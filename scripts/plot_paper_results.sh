#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-out/region_interface}"
PLOT_OUTPUT_ROOT="${PLOT_OUTPUT_ROOT:-}"
BACKBONE="${BACKBONE:-mamba3}"
MODEL_SCALE="${MODEL_SCALE:-canonical}"
TARGET_STEPS="${TARGET_STEPS:-20000}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-llama}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-true}"
FAMILY_SEQ_LEN="${FAMILY_SEQ_LEN:-1024}"
DPI="${DPI:-150}"
LABEL_MODE="${LABEL_MODE:-auto}"
REGIMES="${REGIMES:-}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-25}"
FONT_SCALE="${FONT_SCALE:-1.0}"
INCLUDE_VARIANT_REGEX="${INCLUDE_VARIANT_REGEX:-}"
EXCLUDE_VARIANT_REGEX="${EXCLUDE_VARIANT_REGEX:-}"
PLOT_SUBDIR="${PLOT_SUBDIR:-}"
RUN_ABLATIONS="${RUN_ABLATIONS:-1}"

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

if [[ $# -gt 0 ]]; then
  RUN_ROOTS=("$@")
else
  RUN_ROOTS=("${FAMILY_DIR}")
fi

if [[ -z "${PLOT_SUBDIR}" ]]; then
  if [[ "${LABEL_MODE}" == "auto" ]]; then
    PLOT_DIR_NAME="plots"
    ABLATION_PLOT_DIR_NAME="ablation_plots"
  else
    PLOT_DIR_NAME="plots_${LABEL_MODE}"
    ABLATION_PLOT_DIR_NAME="ablation_plots_${LABEL_MODE}"
  fi
else
  PLOT_DIR_NAME="plots_${PLOT_SUBDIR}"
  ABLATION_PLOT_DIR_NAME="ablation_plots_${PLOT_SUBDIR}"
fi

if [[ -z "${PLOT_OUTPUT_ROOT}" ]]; then
  MAIN_OUTPUT_DIR="${RUN_ROOTS[0]%/}/${PLOT_DIR_NAME}"
  ABLATION_OUTPUT_DIR="${RUN_ROOTS[0]%/}/${ABLATION_PLOT_DIR_NAME}"
else
  MAIN_OUTPUT_DIR="${PLOT_OUTPUT_ROOT%/}/${FAMILY_NAME}/${PLOT_DIR_NAME}"
  ABLATION_OUTPUT_DIR="${PLOT_OUTPUT_ROOT%/}/${FAMILY_NAME}/${ABLATION_PLOT_DIR_NAME}"
fi

MAIN_ARGS=(
  -m scripts.plot_training_curves
  "${RUN_ROOTS[@]}"
  --output-dir "${MAIN_OUTPUT_DIR}"
  --dpi "${DPI}"
  --label-mode "${LABEL_MODE}"
  --smooth-window "${SMOOTH_WINDOW}"
  --font-scale "${FONT_SCALE}"
)
if [[ -n "${REGIMES}" ]]; then
  MAIN_ARGS+=(--regimes "${REGIMES}")
fi
if [[ -n "${INCLUDE_VARIANT_REGEX}" ]]; then
  MAIN_ARGS+=(--include-variant-regex "${INCLUDE_VARIANT_REGEX}")
fi
if [[ -n "${EXCLUDE_VARIANT_REGEX}" ]]; then
  MAIN_ARGS+=(--exclude-variant-regex "${EXCLUDE_VARIANT_REGEX}")
fi

cd "${REPO_ROOT}"
MPLBACKEND="${MPLBACKEND:-Agg}" "${PYTHON_BIN}" "${MAIN_ARGS[@]}"

if [[ "${RUN_ABLATIONS}" == "1" || "${RUN_ABLATIONS}" == "true" || "${RUN_ABLATIONS}" == "yes" ]]; then
  ABLATION_ARGS=(
    -m scripts.plot_message_ablations
    "${RUN_ROOTS[@]}"
    --output-dir "${ABLATION_OUTPUT_DIR}"
    --dpi "${DPI}"
  )
  if [[ -n "${INCLUDE_VARIANT_REGEX}" ]]; then
    ABLATION_ARGS+=(--include-variant-regex "${INCLUDE_VARIANT_REGEX}")
  fi
  if [[ -n "${EXCLUDE_VARIANT_REGEX}" ]]; then
    ABLATION_ARGS+=(--exclude-variant-regex "${EXCLUDE_VARIANT_REGEX}")
  fi
  MPLBACKEND="${MPLBACKEND:-Agg}" "${PYTHON_BIN}" "${ABLATION_ARGS[@]}"
fi
