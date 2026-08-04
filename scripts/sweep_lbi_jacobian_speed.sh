#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SWEEP_ROOT="${SWEEP_ROOT:-out/region_interface/speed_sweeps}"
SWEEP_NAME="${SWEEP_NAME:-lbi_jacobian_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SWEEP_ROOT%/}/${SWEEP_NAME}}"
SUMMARY_DIR="${SUMMARY_DIR:-${OUTPUT_ROOT%/}/summary}"

BACKBONES="${BACKBONES:-mamba3}"
MODEL_SCALE="${MODEL_SCALE:-canonical}"
RANKS="${RANKS:-16 32 64}"
RECOMPUTE_CHUNKS="${RECOMPUTE_CHUNKS:-1 2 4 8}"
RUN_GRAPH="${RUN_GRAPH:-1}"
RUN_RECOMPUTE="${RUN_RECOMPUTE:-1}"
SKIP_CHUNK_GT_RANK="${SKIP_CHUNK_GT_RANK:-1}"

SEQ_LEN="${SEQ_LEN:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TARGET_STEPS="${TARGET_STEPS:-100}"
SEED="${SEED:-7}"
EVAL_EVERY="${EVAL_EVERY:-1000000}"
EVAL_BATCHES="${EVAL_BATCHES:-1}"
LOG_EVERY="${LOG_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-0}"
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-true}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-llama}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
SUMMARY_DROP_FRAC="${SUMMARY_DROP_FRAC:-0.2}"
SUMMARY_MIN_STEP="${SUMMARY_MIN_STEP:-10}"
EXTRA_ARGS=("$@")

normalize_list() {
  local value="$1"
  value="${value//,/ }"
  printf "%s" "${value}"
}

is_enabled() {
  case "$1" in
    1|true|yes|on) return 0 ;;
    0|false|no|off) return 1 ;;
    *)
      echo "Expected boolean value, got '$1'" >&2
      exit 1
      ;;
  esac
}

BACKBONES="$(normalize_list "${BACKBONES}")"
RANKS="$(normalize_list "${RANKS}")"
RECOMPUTE_CHUNKS="$(normalize_list "${RECOMPUTE_CHUNKS}")"

if [[ "${TARGET_STEPS}" -le 0 ]]; then
  echo "TARGET_STEPS must be > 0" >&2
  exit 1
fi
if [[ "${BATCH_SIZE}" -le 0 ]]; then
  echo "BATCH_SIZE must be > 0" >&2
  exit 1
fi
if [[ "${SEQ_LEN}" -le 0 ]]; then
  echo "SEQ_LEN must be > 0" >&2
  exit 1
fi
if [[ "${MODEL_SCALE}" != "canonical" && "${MODEL_SCALE}" != "large" ]]; then
  echo "MODEL_SCALE must be canonical or large, got '${MODEL_SCALE}'" >&2
  exit 1
fi
if [[ "${WARMUP_STEPS}" -lt 0 ]]; then
  echo "WARMUP_STEPS must be >= 0" >&2
  exit 1
fi
if [[ "${WARMUP_STEPS}" -gt "${TARGET_STEPS}" ]]; then
  echo "WARMUP_STEPS must be <= TARGET_STEPS for timing sweeps; got WARMUP_STEPS=${WARMUP_STEPS}, TARGET_STEPS=${TARGET_STEPS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${SUMMARY_DIR}"

echo "Repo root: ${REPO_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Summary dir: ${SUMMARY_DIR}"
echo "Backbones: ${BACKBONES}"
echo "Model scale: ${MODEL_SCALE}"
echo "Ranks: ${RANKS}"
echo "Recompute chunks: ${RECOMPUTE_CHUNKS}"
echo "Run graph: ${RUN_GRAPH}"
echo "Run recompute: ${RUN_RECOMPUTE}"
echo "Skip chunk > rank: ${SKIP_CHUNK_GT_RANK}"
echo "Target steps: ${TARGET_STEPS}"
echo "LR schedule: ${LR_SCHEDULE}"
echo "Warmup steps: ${WARMUP_STEPS}"
echo "Log every: ${LOG_EVERY}"
echo "Eval every: ${EVAL_EVERY}"
echo "Eval batches: ${EVAL_BATCHES}"
echo "Checkpoints: disabled"

run_candidate() {
  local backbone="$1"
  local rank="$2"
  local mode="$3"
  local chunk="$4"
  local variant
  if [[ "${mode}" == "graph" ]]; then
    variant="speed_${backbone}_graph_r${rank}_seed${SEED}"
  else
    variant="speed_${backbone}_recompute_c${chunk}_r${rank}_seed${SEED}"
  fi

  echo
  echo "Running ${variant}"
  PYTHON_BIN="${PYTHON_BIN}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  BACKBONE="${backbone}" \
  MODEL_SCALE="${MODEL_SCALE}" \
  SEQ_LEN="${SEQ_LEN}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  TARGET_STEPS="${TARGET_STEPS}" \
  SEED="${SEED}" \
  EVAL_EVERY="${EVAL_EVERY}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  LOG_EVERY="${LOG_EVERY}" \
  SAVE_EVERY="${SAVE_EVERY}" \
  LR_SCHEDULE="${LR_SCHEDULE}" \
  WARMUP_STEPS="${WARMUP_STEPS}" \
  MIN_LR_RATIO="${MIN_LR_RATIO}" \
  WEIGHT_DECAY="${WEIGHT_DECAY}" \
  GRAD_CLIP="${GRAD_CLIP}" \
  TIE_EMBEDDINGS="${TIE_EMBEDDINGS}" \
  TOKENIZER_TYPE="${TOKENIZER_TYPE}" \
  VOCAB_SIZE="${VOCAB_SIZE}" \
  MESSAGE_DIM="${rank}" \
  INTERFACE_JACOBIAN_MODE="${mode}" \
  JACOBIAN_BASIS_CHUNK="${chunk}" \
  VARIANT_NAME="${variant}" \
  RUN_NAME="${variant}" \
  bash "${SCRIPT_DIR}/train_lbi_paper.sh" --no-save-checkpoints "${EXTRA_ARGS[@]}"
}

read -r -a BACKBONE_VALUES <<< "${BACKBONES}"
read -r -a RANK_VALUES <<< "${RANKS}"
read -r -a CHUNK_VALUES <<< "${RECOMPUTE_CHUNKS}"

for backbone in "${BACKBONE_VALUES[@]}"; do
  for rank in "${RANK_VALUES[@]}"; do
    if is_enabled "${RUN_GRAPH}"; then
      run_candidate "${backbone}" "${rank}" "graph" "1"
    fi
    if is_enabled "${RUN_RECOMPUTE}"; then
      for chunk in "${CHUNK_VALUES[@]}"; do
        if is_enabled "${SKIP_CHUNK_GT_RANK}" && (( chunk > rank )); then
          echo
          echo "Skipping recompute chunk ${chunk} for rank ${rank} because SKIP_CHUNK_GT_RANK=${SKIP_CHUNK_GT_RANK}"
          continue
        fi
        run_candidate "${backbone}" "${rank}" "recompute" "${chunk}"
      done
    fi
  done
done

echo
echo "Aggregating speed summary..."
"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${SUMMARY_DIR}" "${SUMMARY_DROP_FRAC}" "${SUMMARY_MIN_STEP}" <<'PY'
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


output_root = Path(sys.argv[1])
summary_dir = Path(sys.argv[2])
drop_frac = float(sys.argv[3])
min_step = int(sys.argv[4])
summary_dir.mkdir(parents=True, exist_ok=True)

rows: list[dict[str, object]] = []
for metrics_path in sorted(output_root.rglob("native_region_interface/metrics.csv")):
    run_dir = metrics_path.parent
    variant_dir = run_dir.parent
    family_dir = variant_dir.parent
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        config = {}
    try:
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        run_summary = {}

    train_rows: list[dict[str, str]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") == "train":
                train_rows.append(row)
    if not train_rows:
        continue

    max_step = max(as_int(row.get("step")) for row in train_rows)
    drop_step = max(min_step, int(max_step * drop_frac))
    kept = [row for row in train_rows if as_int(row.get("step")) >= drop_step]
    if not kept:
        kept = train_rows

    token_rates = [as_float(row.get("tokens_per_s", "")) for row in kept]
    wall_times = [as_float(row.get("wall_time_s", "")) for row in kept]
    token_rates = [value for value in token_rates if value > 0.0]
    wall_times = [value for value in wall_times if value > 0.0]
    if not token_rates:
        continue

    rows.append(
        {
            "family": family_dir.name,
            "variant": variant_dir.name,
            "run_dir": str(run_dir),
            "backbone": config.get("backbone", ""),
            "layers": config.get("layers", ""),
            "dim": config.get("dim", ""),
            "model_scale_tag": family_dir.name,
            "message_dim": config.get("message_dim", ""),
            "region_size": config.get("region_size", ""),
            "interface_jacobian_mode": config.get("interface_jacobian_mode", ""),
            "jacobian_basis_chunk": config.get("jacobian_basis_chunk", ""),
            "steps": config.get("steps", run_summary.get("steps", "")),
            "log_rows_used": len(kept),
            "drop_step": drop_step,
            "mean_tokens_per_s": statistics.fmean(token_rates),
            "median_tokens_per_s": statistics.median(token_rates),
            "min_tokens_per_s": min(token_rates),
            "max_tokens_per_s": max(token_rates),
            "mean_wall_time_s": statistics.fmean(wall_times) if wall_times else "",
            "median_wall_time_s": statistics.median(wall_times) if wall_times else "",
        }
    )

rows.sort(key=lambda row: float(row["median_tokens_per_s"]), reverse=True)
csv_path = summary_dir / "speed_summary.csv"
json_path = summary_dir / "speed_summary.json"
fieldnames = [
    "family",
    "variant",
    "backbone",
    "layers",
    "dim",
    "message_dim",
    "region_size",
    "interface_jacobian_mode",
    "jacobian_basis_chunk",
    "steps",
    "log_rows_used",
    "drop_step",
    "mean_tokens_per_s",
    "median_tokens_per_s",
    "min_tokens_per_s",
    "max_tokens_per_s",
    "mean_wall_time_s",
    "median_wall_time_s",
    "run_dir",
]
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

print(f"Wrote {csv_path}")
print(f"Wrote {json_path}")
if rows:
    print()
    print("Top speed rows:")
    for row in rows[:10]:
        print(
            f"{row['median_tokens_per_s']:.2f} tok/s | "
            f"{row['backbone']} r={row['message_dim']} "
            f"{row['interface_jacobian_mode']} c={row['jacobian_basis_chunk']} | "
            f"{row['variant']}"
        )
else:
    print("No train metrics found.")
PY

echo
echo "Speed sweep complete."
echo "Output root: ${OUTPUT_ROOT}"
echo "Summary: ${SUMMARY_DIR}/speed_summary.csv"
