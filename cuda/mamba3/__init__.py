from __future__ import annotations


def state_mixer_input_pullback_basis(*args, **kwargs):
    raise NotImplementedError(
        "Mamba-3 CUDA pullback kernels are not implemented yet. "
        "This package is reserved for SISO state-mixer activation pullbacks used by LBI-2."
    )


__all__ = ["state_mixer_input_pullback_basis"]
