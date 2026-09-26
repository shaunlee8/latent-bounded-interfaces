# CUDA Support

CUDA extensions are organized by component.

- `cuda/interface/`: the suffix scan that composes the interface Jacobians `A_k` across regions (cuBLAS batched GEMMs; a Torch fallback runs without the extension).
- `cuda/mamba3/`: the lane-batched wgmma recurrence kernel of the chunked SSM scan; see `cuda/mamba3/README.md`.
- `cuda/transformer/`: the fused attention JVP kernel (all `r` tangent directions, flat epilogue) for the Transformer's forward-mode construction on Hopper; it builds on first use against tilelang's shipped headers, and `build.sh` prewarms that build. The Triton kernel in `backbones/transformer/ops/triton/region_jvp.py` serves configurations outside the CUDA contract (`hd != 64`, `L` not a multiple of 64) and non-Hopper devices.
- `cuda/common/`: the wgmma issue helpers shared by the two backbone kernels.

## Build

Build extensions from the repository root after activating the project environment:

```bash
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/interface/build.sh
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/mamba3/build.sh
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/transformer/build.sh
```

## Tests

```bash
${PYTHON_BIN:-python} -m pytest tests/test_interface_scan.py tests/test_forward_mode_region.py tests/test_transformer_forward_mode.py -q
LBI_TILELANG_TESTS=1 ${PYTHON_BIN:-python} -m pytest tests/test_tilelang_chunked_scan.py -q
```

The forward-mode provider selects the Mamba-3 extension when it is built; the tilelang file's kernel-level tests are opt-in because the JIT compile takes minutes.
