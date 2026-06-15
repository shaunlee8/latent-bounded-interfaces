# Interface CUDA Extension

This directory contains backend-agnostic CUDA kernels for scan composition of bounded-interface pullback matrices.

The LBI backward path materializes per-region interface pullbacks with shape:

```text
[batch, num_regions, rank, rank]
```

The suffix scan computes products and returns:

```text
[batch, num_regions + 1, rank, rank]
```

These suffix products propagate message adjoints across regions. Backend-local activation pullbacks live in backend-specific CUDA folders such as `cuda/transformer/`.

## Build

From the repository root, after activating the project conda environment:

```bash
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/interface/build.sh
```

The extension module is `interface_scan_cuda`. The Python wrapper is:

```python
from cuda.interface import suffix_scan_pullbacks
```

`suffix_scan_pullbacks(mats)` uses the CUDA extension for CUDA tensors when available and falls back to a Torch reference implementation otherwise.

## Files

- `setup.py`: extension build configuration.
- `build.sh`: repository-local build entrypoint.
- `binding.cpp`: pybind bindings.
- `suffix_scan.hpp`: C++ declarations.
- `suffix_scan.cu`: CUDA suffix-scan implementation.

## Test

```bash
${PYTHON_BIN:-python} -m pytest tests/test_interface_scan.py -q
```

The test checks CPU fallback correctness, CUDA correctness when the extension is available, low-precision input promotion, and scan order on structured matrices.
