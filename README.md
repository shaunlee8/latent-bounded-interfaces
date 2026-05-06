# Bounded-Interface Language Model Experiments

This repository contains the code used for the bounded-interface/LBI language-model experiments. The supported paper backbones are:

- Mamba-2
- Mamba-3
- Transformer
- Hybrid Mamba-3/Transformer stacks

The central implementation is `models/native_region_interface.py`. The paper training entrypoint is `train/train_region_interface.py`, but users should normally use the wrapper scripts in `scripts/`.

## What's Included

- supported backbone implementations under `backbones/`
- bounded-interface model and native backward path under `models/`
- FineWeb-Edu export and pretokenization helpers under `data/`
- canonical training, post-hoc eval, and plotting scripts under `scripts/`
- CUDA suffix-scan source under `cuda/interface/`
- smoke, gradient parity, and scan tests under `tests/`

Not included:

- FineWeb-Edu raw text
- pretokenized token shards
- LLaMA tokenizer files
- checkpoints and generated `out/` results

## Third-Party Code

Portions of `backbones/mamba2/`, `backbones/mamba3/`, and `backbones/transformer/` are adapted from the official Mamba repository:

```text
https://github.com/state-spaces/mamba
```

The upstream Mamba code is licensed under Apache-2.0. See `THIRD_PARTY_NOTICES.md` and `third_party_licenses/mamba/LICENSE` for attribution and license details.

## License

Unless otherwise noted, this repository's code and documentation are released under the Apache License, Version 2.0; see `LICENSE`. Third-party code adapted from upstream projects remains subject to its original license and attribution notices; see `THIRD_PARTY_NOTICES.md`.

## Environment

Set the Python executable used by the scripts:

```bash
conda env create -f environment.yml
conda activate lbi
export PYTHON_BIN=python
```

The environment installs `causal-conv1d`, which is required by the Mamba-2 path used in the Mamba-2 and Hybrid experiments. The pip section of `environment.yml` disables build isolation so this package builds against the PyTorch/CUDA stack installed by conda; this source build can take several minutes.

Build the interface CUDA extension from the activated environment before running CUDA-backed Mamba/LBI experiments:

```bash
PYTHON_BIN=${PYTHON_BIN} ./cuda/interface/build.sh
```

Keep the conda environment activated while building so tools such as `ninja` are on `PATH`. The code expects PyTorch, Triton, causal-conv1d, einops, transformers, sentencepiece, datasets, huggingface_hub, numpy, and matplotlib. CUDA is required for Mamba-3 and Hybrid paper-scale runs.

## Data And Tokenizer

Setting the dataset and tokenizer path:

```bash
export LBI_DATA_ROOT=/path/to/lbi_data
export LBI_LLAMA_TOKENIZER_ROOT=${LBI_DATA_ROOT}/tokenizers/llama
```

The paper uses the original 32k LLaMA tokenizer. We do not redistribute tokenizer files. Users should request access to the official Meta Llama 2 Hugging Face repository:

https://huggingface.co/meta-llama/Llama-2-7b-hf

After access is granted:

```bash
huggingface-cli login

huggingface-cli download meta-llama/Llama-2-7b-hf \
  --include "config.json" \
  --include "tokenizer.*" \
  --include "special_tokens_map.json" \
  --include "tokenizer_config.json" \
  --local-dir ${LBI_LLAMA_TOKENIZER_ROOT}
```

Validate the local tokenizer directory before pretokenizing:

```bash
python - <<'PY'
import os
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    os.environ["LBI_LLAMA_TOKENIZER_ROOT"],
    use_fast=True,
    local_files_only=True,
)
print("vocab size:", len(tok))
assert len(tok) == 32000
PY
```

Export the bounded FineWeb-Edu text subset:

```bash
${PYTHON_BIN} -m data.export_fineweb_edu \
  --output-dir ${LBI_DATA_ROOT}/corpora/fineweb_edu \
  --cache-dir ${LBI_DATA_ROOT}/hf_cache \
  --config sample-10BT \
  --train-bytes 1000000000 \
  --val-bytes 100000000 \
  --force-exit-after-export
```

`--force-exit-after-export` exits immediately after the bounded train/validation text files are flushed. This avoids a known `datasets`/`pyarrow` interpreter-teardown crash observed on some hosts after the export has already completed.

Pretokenize once:

```bash
${PYTHON_BIN} -m data.pretokenize_corpus \
  --text-corpus fineweb_edu \
  --tokenizer-type llama \
  --vocab-size 32000 \
  --tokenizer-path ${LBI_LLAMA_TOKENIZER_ROOT} \
  --shard-tokens 5000000
```

Expected token shards:

```text
${LBI_DATA_ROOT}/corpora/fineweb_edu/tokens/fineweb_edu_llama/
  train_manifest.json
  val_manifest.json
```

See `data/README.md` for details.

## Canonical Training

Shared canonical settings:

- tokenizer: `TOKENIZER_TYPE=llama`, `VOCAB_SIZE=32000`
- tied embeddings: `TIE_EMBEDDINGS=true`
- sequence length: `SEQ_LEN=1024`
- steps: `TARGET_STEPS=20000`
- seeds: `7,8,9`
- LBI ranks: `MESSAGE_DIM in {16,32,64}`
- canonical region size: `REGION_SIZE=2`
- output root: `out/region_interface`

Run dense baselines:

```bash
for backbone in mamba2 mamba3 transformer hybrid; do
  for seed in 7 8 9; do
    PYTHON_BIN=${PYTHON_BIN} \
    OUTPUT_ROOT=out/region_interface \
    BACKBONE=${backbone} \
    MODEL_SCALE=canonical \
    TARGET_STEPS=20000 \
    SEED=${seed} \
    VARIANT_NAME=dense_seed${seed} \
    RUN_NAME=dense_seed${seed} \
    TOKENIZER_PATH=${LBI_LLAMA_TOKENIZER_ROOT} \
    ./scripts/train_dense_paper.sh
  done
done
```

Run LBI ranks:

```bash
for backbone in mamba2 mamba3 transformer hybrid; do
  for seed in 7 8 9; do
    for rank in 16 32 64; do
      PYTHON_BIN=${PYTHON_BIN} \
      OUTPUT_ROOT=out/region_interface \
      BACKBONE=${backbone} \
      MODEL_SCALE=canonical \
      TARGET_STEPS=20000 \
      SEED=${seed} \
      MESSAGE_DIM=${rank} \
      REGION_SIZE=2 \
      VARIANT_NAME=lbi_r${rank}_seed${seed} \
      RUN_NAME=lbi_r${rank}_seed${seed} \
      TOKENIZER_PATH=${LBI_LLAMA_TOKENIZER_ROOT} \
      ./scripts/train_lbi_paper.sh
    done
  done
done
```

The wrapper defaults encode the optimizer settings used for the reported runs:

- Mamba-2: dense and LBI use `lr=8e-4`
- Mamba-3: dense uses `lr=6e-4`; LBI uses `lr=3e-4`
- Transformer: dense and LBI use `lr=6e-4`
- Hybrid: dense uses `lr=6e-4`; LBI r16 uses `lr=6e-4`; LBI r32/r64 use `lr=3e-4`

## Hyperparameter Selection

The reported recipes use architecture-specific optimizer settings selected from short seed-7 tuning runs, then fixed for all reported seeds. Dense baselines were tuned over learning rate, weight decay, and warmup. LBI runs reused the corresponding dense non-LR optimizer settings and used targeted LR diagnostics for the interface models.

We did not tune over dataset, tokenizer, sequence length, batch size, LR schedule, minimum LR ratio, gradient clipping, model width/depth, tied embeddings, or interface hidden size for the reported canonical experiments. The region-size sweep and Jacobian diagnostics are separate appendix studies, not part of optimizer selection.

The final recipes are encoded in `scripts/train_dense_paper.sh` and `scripts/train_lbi_paper.sh`. See `scripts/README.md` for the tuning grids and selected values.

## Post-Hoc CE Evaluation

The paper CE table should use post-hoc evaluation on `latest.pt`, not the small online validation rows.

```bash
for backbone in mamba2 mamba3 transformer hybrid; do
  PYTHON_BIN=${PYTHON_BIN} \
  OUTPUT_ROOT=out/region_interface \
  EVAL_OUTPUT_ROOT=out/evals/region_interface \
  BACKBONE=${backbone} \
  MODEL_SCALE=canonical \
  TARGET_STEPS=20000 \
  CHECKPOINT=latest \
  EVAL_BATCHES=512 \
  TOKENIZER_PATH=${LBI_LLAMA_TOKENIZER_ROOT} \
  ./scripts/evaluate_paper_checkpoints.sh
done
```

Outputs:

```text
out/evals/region_interface/<family_name>/lm_eval_summary.csv
out/evals/region_interface/<family_name>/lm_eval_summary.json
```

Report mean and standard deviation over seeds for `posthoc_val_ce_loss`.

## Main Plots

Training curves are generated from the run directories. It currently uses variant filters so canonical runs are not mixed with LR diagnostics or sweeps. Filters are not needed when only running the canonical set.

```bash
PYTHON_BIN=${PYTHON_BIN} \
OUTPUT_ROOT=out/region_interface \
BACKBONE=transformer \
MODEL_SCALE=canonical \
TARGET_STEPS=20000 \
REGIMES=backprop_ref,native_region_interface \
LABEL_MODE=rank_region \
RUN_ABLATIONS=0 \
FONT_SCALE=1.2 \
INCLUDE_VARIANT_REGEX='^(dense_seed[789]|lbi_r(16|32|64)_seed[789])$' \
./scripts/plot_paper_results.sh
```

The main CE-vs-token plot emphasizes train CE and overlays online validation CE as a small-budget diagnostic. Final reported CE should still come from post-hoc eval.

## Region-Size Sweep

The appendix region-size sweep uses Mamba-3 and Transformer at fixed rank `r=32`, seeds `7,8,9`, and region sizes `1,2,3,4`.

Example command for one Mamba-3 run:

```bash
PYTHON_BIN=${PYTHON_BIN} \
OUTPUT_ROOT=out/region_interface \
BACKBONE=mamba3 \
MODEL_SCALE=canonical \
TARGET_STEPS=20000 \
SEED=7 \
MESSAGE_DIM=32 \
REGION_SIZE=3 \
VARIANT_NAME=lbi_r32_region3_seed7 \
RUN_NAME=lbi_r32_region3_seed7 \
TOKENIZER_PATH=${LBI_LLAMA_TOKENIZER_ROOT} \
./scripts/train_lbi_paper.sh
```

Plot the sweep:

```bash
PYTHON_BIN=${PYTHON_BIN} \
OUTPUT_ROOT=out/region_interface \
BACKBONE=mamba3 \
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

Jacobian diagnostics are separate from canonical runs. The Mamba-3 appendix run uses ranks `16,32,64`, seeds `7,8,9`, and logs local/suffix interface Jacobian statistics during training.

```bash
for seed in 7 8 9; do
  for rank in 16 32 64; do
    PYTHON_BIN=${PYTHON_BIN} \
    OUTPUT_ROOT=out/region_interface \
    FAMILY_NAME=paper_mamba3_14l_768d_llama32k_tied_seq1024_20ksteps_jacobian \
    BACKBONE=mamba3 \
    MODEL_SCALE=canonical \
    TARGET_STEPS=20000 \
    SEED=${seed} \
    MESSAGE_DIM=${rank} \
    VARIANT_NAME=lbi_r${rank}_jac_seed${seed} \
    RUN_NAME=lbi_r${rank}_jac_seed${seed} \
    TOKENIZER_PATH=${LBI_LLAMA_TOKENIZER_ROOT} \
    ./scripts/train_lbi_paper.sh \
      --log-interface-jacobian-every 200 \
      --log-interface-jacobian-suffix
  done
done
```

Generate the appendix figure and summary table:

```bash
${PYTHON_BIN} -m scripts.plot_jacobian_diagnostics \
  out/region_interface/paper_mamba3_14l_768d_llama32k_tied_seq1024_20ksteps_jacobian \
  --output-dir out/region_interface/paper_mamba3_14l_768d_llama32k_tied_seq1024_20ksteps_jacobian/jacobian_plots
```

## Tests

Functionality checks are provided:

```bash
${PYTHON_BIN} -m pytest \
  tests/test_token_shards.py \
  tests/test_interface_scan.py \
  tests/test_lbi_interface_components.py \
  tests/test_lbi_gradient_parity.py \
  tests/test_training_workflow_smoke.py \
  -q
```

The paper gradient parity table can be generated with:

```bash
${PYTHON_BIN} -m scripts.generate_grad_parity_report \
  --preset report \
  --plot \
  --output-dir out/grad_parity/recompute_report
```

For a faster functionality check:

```bash
${PYTHON_BIN} -m scripts.generate_grad_parity_report --preset test
```

The report defaults to `--interface-jacobian-mode recompute --jacobian-basis-chunk 32`, matching the LBI training launcher and the paper table. The graph-mode reference path remains available via explicit flags.
