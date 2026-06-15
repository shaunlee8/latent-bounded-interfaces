# CUDA Support

CUDA extensions are organized by LBI-2 component.

- `cuda/interface/`: backend-agnostic suffix-scan kernels for composing bounded-interface pullback matrices.
- `cuda/transformer/`: Transformer attention input-pullback kernels.
- `cuda/mamba3/`: Mamba-3 SISO CUDA namespace. No extension is implemented in this directory yet.

## Build

Build extensions from the repository root after activating the project environment:

```bash
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/interface/build.sh
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/transformer/build.sh
```

## Tests

```bash
${PYTHON_BIN:-python} -m pytest tests/test_interface_scan.py -q
${PYTHON_BIN:-python} -m pytest tests/test_cuda_attention_pullback.py -q
```

`tests/test_cuda_attention_pullback.py` compares the basis-aware CUDA attention pullback against PyTorch reference implementations for the local contract
`q,k,v,dO_basis -> dQ_basis,dK_basis,dV_basis`.
