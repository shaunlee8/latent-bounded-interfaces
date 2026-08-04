# Mamba-3 CUDA Kernels

Hand-CUDA kernels for the Mamba-3 SISO backend, built as one torch extension
(`mamba3_lbi_cuda`) via `build.sh` / `setup.py` (sm_90a).

- `gwalk_dual.cu`: the lane-batched wgmma dual walk (forward-mode JVP scan),
  with a pooled-mean and a raw OUT/DOUT variant.
- `fwd_dualscan_simple.cu` / `fwd_dualscan_opt.cu`: forward dual-scan passes;
  the simple file is the readable correctness scaffold, the opt file the
  wmma rewrite.
- `chunkparallel_pass_c_simple.cu` / `chunkparallel_pass_c_mma.cu`: pass C of
  the chunk-parallel P-batched backward, scaffold and tensor-core rewrite.
- `wmma_gemm.cuh`: shared wmma GEMM helpers.

All kernels are parity-gated against the tilelang oracles
(`tests/test_tilelang_dualscan.py`, `tests/test_tilelang_pbatched.py`).
The import is optional; callers fall back to the tilelang path when the
extension is not built.
