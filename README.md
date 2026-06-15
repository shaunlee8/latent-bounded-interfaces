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
- `cuda/transformer/`: Transformer attention input-pullback kernels.

Build commands:

```bash
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/interface/build.sh
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/transformer/build.sh
```

The Transformer CUDA extension requires CUTLASS/CuTe. Initialize the submodule before building if needed:

```bash
git submodule update --init --recursive third_party/cutlass
```

## Tests

Core LBI-2 tests are not yet available.

CUDA tests:

```bash
${PYTHON_BIN:-python} -m pytest \
  tests/test_interface_scan.py \
  tests/test_cuda_attention_pullback.py \
  -q
```

Transformer attention pullback timing:

```bash
${PYTHON_BIN:-python} benchmarks/time_attention_pullback.py \
  --seq-lens 128 \
  --head-dims 64 \
  --basis-sizes 4 \
  --dtype float16 \
  --skip-torch-formula \
  --include-fa2-loop \
  --include-autograd \
  --check
```

## Third-Party Code

Portions of the backbone implementations are adapted from the official Mamba repository:

```text
https://github.com/state-spaces/mamba
```

The upstream Mamba code is licensed under Apache-2.0. See `THIRD_PARTY_NOTICES.md` and `third_party_licenses/mamba/LICENSE` for attribution and license details.

This branch also uses CUTLASS/CuTe for Transformer CUDA kernels via `third_party/cutlass`.

## License

Unless otherwise noted, this repository's code and documentation are released under the Apache License, Version 2.0; see `LICENSE`. Third-party code adapted from upstream projects remains subject to its original license and attribution notices; see `THIRD_PARTY_NOTICES.md`.
