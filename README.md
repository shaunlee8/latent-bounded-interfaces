# Latent Bounded Interfaces

This branch contains the LBI-2 development codebase. It refactors the original LBI-1 experiment implementation into explicit component boundaries for interface maps, canvases, readouts, region backends, backward engines, and CUDA lowering.

For the LBI-1 paper reproduction workflow, use the `paper-release` branch. That branch contains the canonical training, evaluation, plotting, tokenizer, and dataset instructions for the reported LBI-1 experiments.

## Repository Layout

- `interfaces/`: interface map modules.
- `canvas/`: token/input canvas modules.
- `readouts/`: language-model readout heads.
- `backends/`: region backend contracts and Transformer backend lowering.
- `backward/`: bounded-interface backward engines and local VJP helpers.
- `models/`: dense and LBI language-model wrappers.
- `train/`: LBI-2 training construction, config, checkpointing, metrics, data, eval, and runners.
- `cuda/`: CUDA extensions for interface scans and backend-local pullbacks.
- `legacy/`: compatibility boundary for LBI-1 code paths retained during the refactor.
- `tests/`: unit and parity tests for the refactored interfaces, models, backends, backward engines, and CUDA kernels.
- `benchmarks/`: local kernel timing scripts.

## CUDA

CUDA extensions are documented under `cuda/README.md`.

Current CUDA extensions:

- `cuda/interface/`: suffix-scan composition for bounded-interface pullback matrices.
- `cuda/mamba3/`: Mamba-3 forward-mode kernels (lane-batched dual walk, dual-scan forward, chunk-parallel pass C).
- `cuda/transformer/`: flash-JVP attention tangent kernel (primary on Hopper; triton alternative in `backbones/transformer/ops/triton/`).

Build commands:

```bash
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/interface/build.sh
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/mamba3/build.sh
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/transformer/build.sh
```

## Tests

```bash
${PYTHON_BIN:-python} -m pytest tests/ -q
```

## Benchmarks

- `benchmarks/sharded_forward_region_parallel.py`: the production distributed training step — region-sharded forward chain plus bubble-free region-parallel backward (multi-GPU, NCCL).
- `benchmarks/nccl_region_parallel.py`: backward-only region-parallel row (mamba3 and transformer backbones).
- `benchmarks/pipeline_parallel_baseline.py`: GPipe-style pipeline baseline over the same stack.
- `benchmarks/dense_sequential_baseline.py`: single-GPU dense sequential backprop row (no interfaces).

## Third-Party Code

Portions of the backbone implementations are adapted from the official Mamba repository:

```text
https://github.com/state-spaces/mamba
```

The upstream Mamba code is licensed under Apache-2.0. See `THIRD_PARTY_NOTICES.md` and `third_party_licenses/mamba/LICENSE` for attribution and license details.

## License

Unless otherwise noted, this repository's code and documentation are released under the Apache License, Version 2.0; see `LICENSE`. Third-party code adapted from upstream projects remains subject to its original license and attribution notices; see `THIRD_PARTY_NOTICES.md`.
