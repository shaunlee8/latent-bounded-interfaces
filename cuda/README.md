# CUDA Support

CUDA extensions are organized by LBI-2 component.

- `cuda/interface/`: backend-agnostic suffix-scan kernels for composing bounded-interface pullback matrices.
- `cuda/mamba3/`: Mamba-3 forward-mode kernels (lane-batched wgmma dual walk, dual-scan forward, chunk-parallel pass C). Uses tilelang's shipped headers; see `cuda/mamba3/README.md`.
- `cuda/transformer/`: transformer flash-JVP kernel (occupancy-2 lane-pair, flat epilogue) — the primary attention tangent kernel for forward-mode A_k on Hopper. The triton kernel in `backbones/transformer/ops/triton/region_jvp.py` remains the alternative for configurations outside the CUDA contract (hd != 64, L not a multiple of 64), non-Hopper devices, or `LBI_FWDMODE_CUDA_ATTN=0`. JIT-builds on first use against tilelang's shipped headers (no separate CUTLASS checkout); `build.sh` prewarms.

## Build

Build extensions from the repository root after activating the project environment:

```bash
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/interface/build.sh
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/mamba3/build.sh
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/transformer/build.sh
```

## Tests

```bash
${PYTHON_BIN:-python} -m pytest tests/test_interface_scan.py tests/test_tilelang_dualscan.py tests/test_transformer_forward_mode.py -q
```
