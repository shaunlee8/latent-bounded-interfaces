# Transformer CUDA Kernels

This directory contains CUDA kernels for the Transformer region backend.

The extension exposes attention input-pullback functions for the local attention contract:

```python
attention_input_pullback_basis(
    q,                      # [B, H, T, Dh]
    k,                      # [B, H, T, Dh]
    v,                      # [B, H, T, Dh]
    output_cotangent_basis, # [B, P, H, T, Dh]
    softmax_scale=...,
    causal=True,
) -> (dQ, dK, dV)           # each [B, P, H, T, Dh]
```

This computes the Jacobian-transpose action of attention outputs with respect to attention inputs. It does not compute projection-weight gradients or other parameter VJPs.

## Build

From the repository root:

```bash
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/transformer/build.sh
```

The extension module is `transformer_lbi_cuda`. The tensor-core kernels require CUTLASS/CuTe from `third_party/cutlass` by default:

```bash
git submodule update --init --recursive third_party/cutlass
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/transformer/build.sh
```

Use `LBI_CUTLASS_ROOT=/path/to/cutlass` to point at a different CUTLASS checkout. The build fails early if CUTLASS/CuTe headers are unavailable.

The Python wrapper is:

```python
from cuda.transformer import attention_input_pullback_basis
```

## Kernel Modes

The Python wrapper accepts `kernel_mode`:

- `simple`: scalar CUDA reference implementation for local correctness checks across the extension boundary.
- `fa2_loop`: SDPA/autograd reference path that calls PyTorch backward once per cotangent basis vector.
- `fa2_p1`: P=1 fused SDPA/FlashAttention reference path. For fp16/bf16 CUDA tensors it calls PyTorch's fused FlashAttention forward/backward ops directly and rejects `P > 1`; unsupported dtypes fall back to the SDPA autograd adapter.
- `fa2_mma_p1`: CUTLASS/CuTe P=1 implementation for fp16 `Dh=64`, causal attention with `T` a multiple of 16. Unsupported shapes fall back through the Python wrapper to `fa2_p1`.
- `fa2_mma_pblock4`: default CUTLASS/CuTe P-block implementation for fp16 `Dh=64`, causal attention with `1 <= P <= 4` and `T` a multiple of 16. Unsupported shapes fall back through the Python wrapper to `fa2_loop`.

## Test

```bash
${PYTHON_BIN:-python} -m pytest tests/test_cuda_attention_pullback.py -q
```

The test compares the Torch formula, PyTorch SDPA/autograd references, and CUDA kernels over the same local attention contract.

## Timing

Use the benchmark script for shape sweeps:

```bash
${PYTHON_BIN:-python} benchmarks/time_attention_pullback.py \
  --batches 1 \
  --heads 8 \
  --seq-lens 128,512 \
  --head-dims 64 \
  --basis-sizes 4 \
  --dtype float16 \
  --causal true
```

Reference comparisons:

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

P=1 comparison against PyTorch's fused SDPA/FlashAttention path:

```bash
${PYTHON_BIN:-python} benchmarks/time_attention_pullback.py \
  --seq-lens 128 \
  --head-dims 64 \
  --basis-sizes 1 \
  --dtype float16 \
  --no-default-kernel \
  --skip-torch-formula \
  --include-fa2-p1 \
  --include-fa2-mma-p1 \
  --include-autograd
```

## Files

- `setup.py`: extension build configuration.
- `build.sh`: repository-local build entrypoint.
- `binding.cpp`: pybind bindings.
- `attention_pullback.hpp`: scalar reference declaration.
- `attention_pullback_simple.cu`: scalar CUDA reference implementation.
- `attention_pullback_mma.hpp`: CUTLASS/CuTe implementation declarations.
- `attention_pullback_mma.cu`: CUTLASS/CuTe P=1 and P-block implementations.
