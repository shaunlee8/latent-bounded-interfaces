# Bounded-Interface Backpropagation (LBI)

Code for the paper's experiments: bounded-interface language models (Mamba-3, Transformer, Hybrid backbones over a shared canvas with an $r$-dimensional MLP interface), the exact region-parallel backward (suffix scan over $r \times r$ interface Jacobians), the fused forward-mode Jacobian construction kernels, and the four-device region-parallel training step with its pipeline and sequential baselines.

## Paper Terms

| paper term | code term |
|---|---|
| bounded-interface model | `models/lbi_language_model.py` (`LBILanguageModel`) |
| interface (the MLP realization) | `interfaces/vector_mlp.py` |
| interface rank $r$ | `--message-dim` / `MESSAGE_DIM` in training; `--rank` in the four-device scripts |
| region (two consecutive blocks) | `--region-size` / `REGION_SIZE`; `--layers-per-region` in the four-device scripts |
| shared canvas | `canvas/token_embedding.py`; the shared region view in `canvas/region_views.py` |
| gradient engines (forward-mode construction, reverse-mode construction, end-to-end autograd) | `backward/scan.py` (`ScanADEngine`) with `--interface-jacobian-mode forward` / `graph` in training (the reverse-mode construction is the `native` mode of `scripts/construction_profile.py`), and `AutogradEngine`; `LBI_BACKWARD=scan\|autograd` |
| region-parallel step | `scripts/region_parallel_step.py` (`sharded_step`) |
| $N$-step window (canvas gradient accumulated over $N$ steps) | `--canvas-accum N` (`CanvasReducer`) in the step; `CANVAS_GRAD_WINDOW` in training |
| tuned pipeline / lowest-memory pipeline | `scripts/pipeline_baseline.py --schedule gpipe --microbatches 4` / `--schedule 1f1b --microbatches 8` |
| sequential reference | `scripts/sequential_reference.py` |
| emulated link | `scripts/link_emu.py` (`LBI_EMU_BW_MBPS`, `LBI_EMU_RTT_MS`) |
| construction constant $c$ | `scripts/construction_profile.py` |


## Repository Layout

- `models/`: the dense and bounded-interface language models.
- `canvas/`, `interfaces/`, `readouts/`: the shared canvas, the vector MLP interface with its Jacobian applications, and the language-model readout.
- `backbones/`: the Mamba-3 and Transformer blocks and their Triton and tilelang kernels.
- `backends/`: the region backends (Mamba-3, Transformer, Hybrid) with their native VJPs and forward-mode region JVPs.
- `backward/`: the scan engine (Jacobian construction, suffix scan, region-local backward) and the autograd engine.
- `train/`: configuration, data, training loops, checkpointing, evaluation, and metrics.
- `data/`: the FineWeb-Edu export, the LLaMA tokenizer wrapper, and the token-shard pretokenizer and sampler; see `data/README.md`.
- `cuda/`: the CUDA extensions (interface suffix scan, Mamba-3 recurrence kernel, Transformer flash-JVP kernel); see `cuda/README.md`.
- `scripts/`: the paper's experiment scripts (training launchers, post-hoc evaluation, the four-device step and its baselines, the construction profile) with every printed recipe in `scripts/README.md`; `scripts/appendix/` holds the appendix experiments.
- `tests/`: parity tests for the interfaces, models, backends, backward engines, and kernels.

## Build and test

```bash
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/interface/build.sh
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/mamba3/build.sh
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/transformer/build.sh
${PYTHON_BIN:-python} -m pytest tests/ -q
```

The extension builds and the tilelang JIT need the CUDA 12.8 toolkit (`nvcc`) on `PATH` and an H100-class (sm_90a) device. The tilelang kernel tests compile on first use and are opt-in with `LBI_TILELANG_TESTS=1`.

## Third-Party Code

Portions of the backbone implementations are adapted from the official Mamba repository:

```text
https://github.com/state-spaces/mamba
```

The upstream Mamba code is licensed under Apache-2.0. See `THIRD_PARTY_NOTICES.md` and `third_party_licenses/mamba/LICENSE` for attribution and license details.

## License

Unless otherwise noted, this repository's code and documentation are released under the Apache License, Version 2.0; see `LICENSE`. Third-party code adapted from upstream projects remains subject to its original license and attribution notices; see `THIRD_PARTY_NOTICES.md`.
