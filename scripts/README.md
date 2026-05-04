# Paper Scripts

Use these wrappers for paper reproduction instead of calling `train/train_region_interface.py` directly. The lower-level training script exposes additional research/debug flags; the wrappers encode the canonical paper recipes.

## Training

### Dense Baselines

```bash
PYTHON_BIN=${PYTHON_BIN:-python} \
OUTPUT_ROOT=out/region_interface \
BACKBONE=mamba3 \
MODEL_SCALE=canonical \
TARGET_STEPS=20000 \
SEED=7 \
VARIANT_NAME=dense_seed7 \
RUN_NAME=dense_seed7 \
TOKENIZER_PATH=${LBI_LLAMA_TOKENIZER_ROOT} \
./scripts/train_dense_paper.sh
```

### LBI Runs

```bash
PYTHON_BIN=${PYTHON_BIN:-python} \
OUTPUT_ROOT=out/region_interface \
BACKBONE=mamba3 \
MODEL_SCALE=canonical \
TARGET_STEPS=20000 \
SEED=7 \
MESSAGE_DIM=32 \
REGION_SIZE=2 \
VARIANT_NAME=lbi_r32_seed7 \
RUN_NAME=lbi_r32_seed7 \
TOKENIZER_PATH=${LBI_LLAMA_TOKENIZER_ROOT} \
./scripts/train_lbi_paper.sh
```

Canonical paper settings:

- backbones: `mamba2`, `mamba3`, `transformer`, `hybrid`
- seeds: `7,8,9`
- dense baseline: `dense_seed{seed}`
- LBI ranks: `MESSAGE_DIM=16,32,64`
- canonical region size: `REGION_SIZE=2`
- sequence length: `1024`
- steps: `20000`
- tokenizer: original 32k LLaMA tokenizer

Wrapper LR defaults:

- Mamba-2: dense/LBI `8e-4`
- Mamba-3: dense `6e-4`, LBI `3e-4`
- Transformer: dense/LBI `6e-4`
- Hybrid: dense `6e-4`, LBI r16 `6e-4`, LBI r32/r64 `3e-4`

## Hyperparameter Selection Details

Dense recipes were selected from short seed-7 sweeps over:

```text
lr_model
weight_decay
warmup_steps
```

Initial dense grid:

```text
lr_model: 1e-4, 3e-4, 6e-4
weight_decay: 0.01, 0.1
warmup_steps: 500, 1000
target_steps: 5000
seed: 7
```

Follow-up canonical-size refinements were run where needed. The selected dense settings used by `train_dense_paper.sh` are:

| Backbone | `lr_model` | `weight_decay` | `warmup_steps` |
|---|---:|---:|---:|
| Mamba-2 | `8e-4` | `0.03` | `1000` |
| Mamba-3 | `6e-4` | `0.03` | `500` |
| Transformer | `6e-4` | `0.01` | `1000` |
| Hybrid | `6e-4` | `0.03` | `500` |

LBI recipes reuse the corresponding dense non-LR optimizer settings and apply targeted LR diagnostics. The selected LBI settings used by `train_lbi_paper.sh` are:

| Backbone / Rank | `lr_model` | `weight_decay` | `warmup_steps` |
|---|---:|---:|---:|
| Mamba-2, all ranks | `8e-4` | `0.03` | `1000` |
| Mamba-3, all ranks | `3e-4` | `0.03` | `500` |
| Transformer, all ranks | `6e-4` | `0.01` | `1000` |
| Hybrid, r16 | `6e-4` | `0.03` | `500` |
| Hybrid, r32/r64 | `3e-4` | `0.03` | `500` |

Held fixed across the canonical runs:

```text
dataset: FineWeb-Edu
tokenizer: original 32k LLaMA tokenizer
seq_len: 1024
batch_size: 1
optimizer: AdamW
lr_schedule: cosine
min_lr_ratio: 0.1
grad_clip: 1.0
tie_embeddings: true
region_size: 2
message_hidden_dim: model_dim
```

Region-size sweeps and Jacobian diagnostics are separate appendix experiments and do not change the selected optimizer recipe.

## Post-Hoc Evaluation

Use post-hoc eval on `latest.pt` for reported CE:

```bash
PYTHON_BIN=${PYTHON_BIN:-python} \
OUTPUT_ROOT=out/region_interface \
EVAL_OUTPUT_ROOT=out/evals/region_interface \
BACKBONE=mamba3 \
MODEL_SCALE=canonical \
TARGET_STEPS=20000 \
CHECKPOINT=latest \
EVAL_BATCHES=512 \
TOKENIZER_PATH=${LBI_LLAMA_TOKENIZER_ROOT} \
./scripts/evaluate_paper_checkpoints.sh
```

Outputs:

```text
out/evals/region_interface/<family_name>/lm_eval_summary.csv
out/evals/region_interface/<family_name>/lm_eval_summary.json
```

Use `posthoc_val_ce_loss` and aggregate mean/std over seeds.

## Training Curves

Canonical rank plot:

```bash
PYTHON_BIN=${PYTHON_BIN:-python} \
OUTPUT_ROOT=out/region_interface \
BACKBONE=mamba3 \
MODEL_SCALE=canonical \
TARGET_STEPS=20000 \
REGIMES=backprop_ref,native_region_interface \
LABEL_MODE=rank_region \
RUN_ABLATIONS=0 \
FONT_SCALE=1.2 \
INCLUDE_VARIANT_REGEX='^(dense_seed[789]|lbi_r(16|32|64)_seed[789])$' \
./scripts/plot_paper_results.sh
```

`plot_paper_results.sh` calls `plot_training_curves.py`. If `RUN_ABLATIONS=1`, it also calls `plot_message_ablations.py`, but this is unused.

Important filters:

- canonical ranks: `^(dense_seed[789]|lbi_r(16|32|64)_seed[789])$`
- region sweep: `^lbi_r32_region[1-4]_seed[789]$`

## Region-Size Sweep

The appendix region-size sweep uses Mamba-3 and Transformer, rank `r=32`, seeds `7,8,9`, and region sizes `1,2,3,4`.

Example run:

```bash
PYTHON_BIN=${PYTHON_BIN:-python} \
OUTPUT_ROOT=out/region_interface \
BACKBONE=transformer \
MODEL_SCALE=canonical \
TARGET_STEPS=20000 \
SEED=7 \
MESSAGE_DIM=32 \
REGION_SIZE=4 \
VARIANT_NAME=lbi_r32_region4_seed7 \
RUN_NAME=lbi_r32_region4_seed7 \
TOKENIZER_PATH=${LBI_LLAMA_TOKENIZER_ROOT} \
./scripts/train_lbi_paper.sh
```

Plot:

```bash
PYTHON_BIN=${PYTHON_BIN:-python} \
OUTPUT_ROOT=out/region_interface \
BACKBONE=transformer \
MODEL_SCALE=canonical \
TARGET_STEPS=20000 \
REGIMES=native_region_interface \
LABEL_MODE=region_size \
RUN_ABLATIONS=0 \
FONT_SCALE=1.2 \
INCLUDE_VARIANT_REGEX='^lbi_r32_region[1-4]_seed[789]$' \
./scripts/plot_paper_results.sh
```

## Jacobian Diagnostics

Dedicated Mamba-3 Jacobian runs:

```bash
PYTHON_BIN=${PYTHON_BIN:-python} \
OUTPUT_ROOT=out/region_interface \
FAMILY_NAME=paper_mamba3_14l_768d_llama32k_tied_seq1024_20ksteps_jacobian \
BACKBONE=mamba3 \
MODEL_SCALE=canonical \
TARGET_STEPS=20000 \
SEED=7 \
MESSAGE_DIM=32 \
VARIANT_NAME=lbi_r32_jac_seed7 \
RUN_NAME=lbi_r32_jac_seed7 \
TOKENIZER_PATH=${LBI_LLAMA_TOKENIZER_ROOT} \
./scripts/train_lbi_paper.sh \
  --log-interface-jacobian-every 200 \
  --log-interface-jacobian-suffix
```

Run for ranks `16,32,64` and seeds `7,8,9`.

Generate figure/table:

```bash
${PYTHON_BIN:-python} -m scripts.plot_jacobian_diagnostics \
  out/region_interface/paper_mamba3_14l_768d_llama32k_tied_seq1024_20ksteps_jacobian \
  --output-dir out/region_interface/paper_mamba3_14l_768d_llama32k_tied_seq1024_20ksteps_jacobian/jacobian_plots
```

Outputs include:

- `jacobian_spec_mean_vs_tokens.png`
- `jacobian_spec_mean_vs_tokens.pdf`
- `jacobian_final_summary.csv`
- `jacobian_final_summary.tex`

## Gradient Parity Report

```bash
${PYTHON_BIN:-python} -m scripts.generate_grad_parity_report --preset test
```

The report cases cover Transformer, Mamba-2, Mamba-3, and Hybrid. Mamba-1 functionality has been removed.
