# Mamba-3 CUDA Kernel

The lane-batched wgmma recurrence kernel of the chunked SSM scan, built as the
torch extension `mamba3_lbi_cuda` via `build.sh` / `setup.py` (sm_90a, against
tilelang's shipped wgmma headers).

- `recurrence_jvp.cu`: the recurrence kernel; `recurrence_full` emits the raw
  OUT/DOUT fields and `recurrence_full_fin` the finalized fields.
- `binding.cpp`: the torch binding.

The forward-mode region tests (`tests/test_forward_mode_region.py`) exercise
the kernel against the torch.func reference when the extension is built.
The import is optional; callers fall back to the tilelang recurrence when the
extension is not built.
