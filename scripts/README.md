# Reproducing the paper's numbers

Every number in the paper is produced by one of the six commands below with the
environment values listed. Runs write under `CHECKPOINT_ROOT` (checkpoints,
logs) and `out/region_interface/<family>/<variant>` (config, metrics).

Common environment: `PYTHONPATH=<repo>`; `LBI_DATA_ROOT` and
`LBI_LLAMA_TOKENIZER_ROOT` as in `data/README.md`;
`TOKENIZER_PATH=$LBI_LLAMA_TOKENIZER_ROOT` (the launcher's and the evaluator's
`--tokenizer-path`); `RUN_ROOT=<checkpoint and log root>`. One H100 per
training run, four H100s of one node for the four-device rows.

## 1. Training (Table 1, Section 4.1, Appendix E)

Wrappers: `scripts/train_lbi_paper.sh` (bounded-interface) and
`scripts/train_dense_paper.sh` (dense). Fixed for every row: `SEQ_LEN=1024`,
`LR_SCHEDULE=cosine`, `MIN_LR_RATIO=0.1`, the vector MLP interface, the autograd
engine (`LBI_BACKWARD=autograd NATIVE_BACKWARD=false`, the launcher default), tied
embeddings, `SAVE_EVERY=TARGET_STEPS`, `EVAL_EVERY=200`. `MODEL_SCALE` selects
the architecture: `mid:mamba3` = 18L/1024d (120M), `canonical:mamba3` = 14L/768d
(54M), `large:mamba3` = 48L/1024d (321M), `mid:transformer` = 10L/1024d (157M),
`canonical:transformer` = 12L/512d (47M), `mid:hybrid` = 16L/1024d (143M),
`canonical:hybrid` = 12L/768d (61M).

| row | BACKBONE / MODEL_SCALE | BATCH_SIZE x TARGET_STEPS | LR_MODEL dense / LBI | WARMUP_STEPS | weight decay | seeds |
|---|---|---|---|---|---|---|
| Table 1 Mamba-3 | mamba3 / mid | 32 x 90000 | 6e-4 / 6e-4 | 500 (seed 7 LBI), 1000 (others) | 0.03 | 7, 8, 9 dense; 7, 8, 10 LBI |
| Table 1 Transformer | transformer / mid | 16 x 180000 | 6e-4 / 6e-4 | 1000 | 0.01 | 7, 8, 10 |
| Table 1 Hybrid | hybrid / mid | 32 x 90000 | 1.2e-3 / 6e-4 | 1000 | 0.01 | 7, 8, 10 |
| rank rows r = 8, 32 (App. E.1) | mamba3 / mid, `MESSAGE_DIM=8|32` | 32 x 90000 | 6e-4 | 1000 | 0.03 | 7, 8, 9 |
| window row (App. E.1) | mamba3 / mid, `CANVAS_GRAD_WINDOW=32 CANVAS_GRAD_WINDOW_LR_MULT=8` | 32 x 90000 | 6e-4 | 1000 | 0.03 | 7, 8 |
| truncated rank sweep, Mamba-3 (App. E.1) | mamba3 / mid, `MESSAGE_DIM=8|16|32|64 LR_SCHEDULE_STEPS=90000` | 16 x 15000 | 6e-4 | 500 | 0.03 | 7 |
| truncated rank sweep, Transformer (App. E.1) | transformer / mid, `MESSAGE_DIM=8|16|32|64 LR_SCHEDULE_STEPS=180000` | 16 x 15000 | 6e-4 | 1000 | 0.01 | 9, 10 |
| truncated rank sweep, Hybrid (App. E.1) | hybrid / mid, `MESSAGE_DIM=8|16|32|64 LR_SCHEDULE_STEPS=90000` | 32 x 15000 | 6e-4 | 1000 | 0.01 | 7, 10 |
| granularity (App. E.2) | mamba3 / mid, `REGION_SIZE=2|3|6|9` | 16 x 90000 | 3e-4 (dense reference 6e-4) | 500 | 0.03 | 7, 8, 9 |
| 54M pair (App. E.2) | mamba3 / canonical | 32 x 90000 | 8e-4 / 1.2e-3 | 500 | 0.03 | 7, 8, 9 |
| 321M pair (App. E.2) | mamba3 / large | 8 x 360000 | 6e-4 / 4.5e-4 | 500 | 0.03 | 7, 8 |
| 47M Transformer pair | transformer / canonical | 32 x 90000 | 6e-4 / 6e-4 | 1000 | 0.01 | 7 |
| 61M Hybrid pair | hybrid / canonical | 32 x 90000 | 1.2e-3 / 1.2e-3 | 1000 | 0.01 | 7, 8, 9 |
| parameter-matched dense (App. E.3) | mamba3 / matched (21L) and 16L/768d | 32 x 90000 | 8e-4, 1.2e-3 | 1000 | 0.03, 0.1 | 7 |

Example (Table 1, Mamba-3, bounded-interface, seed 8):

    CHECKPOINT_ROOT=$RUN_ROOT BACKBONE=mamba3 MODEL_SCALE=mid SEQ_LEN=1024 BATCH_SIZE=32 SEED=8 \
    TARGET_STEPS=90000 SAVE_EVERY=90000 LR_SCHEDULE=cosine MIN_LR_RATIO=0.1 EVAL_EVERY=200 \
    LR_MODEL=6e-4 WARMUP_STEPS=1000 MESSAGE_DIM=16 \
    VARIANT_NAME=m3m_mlp16_s8_90k bash scripts/train_lbi_paper.sh

Learning rates were picked per configuration on a 15k-step prefix of the 90k
schedule (`TARGET_STEPS=15000 LR_SCHEDULE_STEPS=90000`, batch 16) over a small
bracket; the picks are the values above.

## 2. Post-hoc evaluation (every printed cross-entropy)

    python scripts/evaluate_paper_checkpoints.py --family out/region_interface/<family> \
      --run-dir out/region_interface/<family>/<variant>/{lbi|dense} --checkpoint latest \
      --eval-batches 512 --batch-size 8 --tokenizer-path $TOKENIZER_PATH --output-dir $RUN_ROOT/evals/<variant>

512 x 8 x 1024 = 4,194,304 held-out tokens, evaluation seed 12345, final
checkpoint. `lm_eval_summary.json` holds `posthoc_val_ce_loss`.

## 3. Single-device cost (Section 4.2, Appendix C, D.1)

    python scripts/construction_profile.py --backbone mamba3 \
      --layers 18 --dim 1024 --rank 16 --seq-len 1024 --batch 32 --output profile_r16.json

gives the engine rows (step time, peak memory) and the construction constant
`c`; `--rank 8|32` the `c(r)` series; `--seq-len 2048 --batch 16` the context
check. Kernel breakdown: `nsys profile` of the same construction, then
`nsys stats --report cuda_gpu_kern_sum`; counters: `ncu` on the recurrence and
chunked-scan kernels.

## 4. Gradient parity (Appendix E)

    python scripts/appendix/gradient_error_vs_k.py --backbone mamba3|transformer|hybrid \
      --dim 512 --rank 16 --region-size 2 --regions 2 3 4 7 10 14 --dtypes float32 bfloat16 \
      --modes forward graph --seq-len 512 --batch 4 --seed <1..10> --output parity_<backbone>_s<seed>.json

One JSON row per (K, dtype, construction) with max-abs, per-parameter and
full-vector relative L2 error, cosine, and counts of missing or non-finite
gradients; unsupported combinations (Mamba-3 and Hybrid in float32, the
Transformer's forward-mode construction in float32, the Hybrid at K not
divisible by 2) are recorded as error rows.

## 5. Four-device rows (Tables 2 and 3, Appendix D)

Region-parallel step (`--rank 16|8|2`, `--backbone mamba3|transformer|hybrid`):

    python scripts/region_parallel_step.py --backbone mamba3 --regions 4 --world-size 4 \
      --layers-per-region 2 --dim 1024 --seq-len 2048 --batch 8 --rank 8 \
      [--canvas-accum N --warmup N --iters N --timing block]

`LBI_MEM_TRACE=1` prints the per-phase allocated and peak memory on rank 0 (the
memory table; `--direction-chunk 8|4|2` and `--microbatches 4` give its rows).

Pipeline (`--schedule gpipe|1f1b`, `--microbatches 4|8`, `--stages-per-device 2` for K=8):

    python scripts/pipeline_baseline.py --backbone mamba3 --regions 4 --world-size 4 \
      --layers-per-region 2 --dim 1024 --seq-len 2048 --batch 8 --microbatches 4

Emulated link (Table 2 "emul." columns and the full grid): the same commands with
`LBI_EMU_BW_MBPS=<Mbit/s> LBI_EMU_RTT_MS=5 LBI_EMU_MODE=sync LBI_EMU_LOG=1`
(`LBI_EMU_P2P_SCALE=0.5|0.25` for the compressed-pipeline proxies; RTT 0.1 for the
bandwidth-to-95% column); log names `netx_<scheme>_bw<Mbit>_rtt<ms>_<mode>_rep<i>.log`.

Shaped link (Table 2 "shaped" columns, the round-trip table of Appendix D): NCCL over TCP on
the loopback interface, shaped with Linux traffic control (root):

    tc qdisc add dev lo root handle 1: tbf rate <Mbit>mbit burst 16mb latency 400ms
    tc qdisc add dev lo parent 1:1 handle 10: netem delay <RTT/2>ms limit 200000

with `NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
NCCL_NET_SHARED_BUFFERS=0` and the two commands above (window rows with
`--timing block`); `tc qdisc del dev lo root` afterwards.

Table 2 from the logs: `LBI_RUN_ROOT=$RUN_ROOT python scripts/netx_analyze.py`
(writes `netx_table.md` / `netx_table.json`).

## 6. Figures

    python scripts/appendix/make_timeline_figure.py --db <nsys trace exported to sqlite> --out timeline_sharded_step.pdf

The timeline script reads the four-device nsys trace of the timed step.

## Optimizer settings

AdamW with the peak learning rates, warmup, and weight decay of the table
above; `grad_clip 1.0`, cosine decay to `0.1` of the peak, sequence length 1024,
tied embeddings, two blocks per region, and the interface map's hidden width
equal to the model width (`MESSAGE_HIDDEN_DIM=0`). The launcher defaults are
the Table 1 values (learning rate 6e-4, or 1.2e-3 for the dense Hybrid; warmup
1000; weight decay 0.03 on Mamba-3 and 0.01 otherwise); every row above that
differs sets its values explicitly.
