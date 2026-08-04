"""Transformer flash-JVP CUDA extension (Hopper sm_90a).

`flash_jvp.cu` holds the lane-pair flash-JVP kernel family; the entry point
is `flash_jvp_occ2_full` (occupancy-2, full-r loop, flat [B, L, H*hd]
epilogue), wrapped as the custom op `lbi::flash_jvp` so compiled callers
trace it as one graph node. JIT-builds on first use against tilelang's
shipped wgmma headers; `build.sh` prewarms the build.
"""
from __future__ import annotations

import functools
from pathlib import Path

import torch

_KERNEL_CONTRACT = "hd64; L % 64 == 0; bf16; sm_90a"


@functools.cache
def _module():
    from torch.utils.cpp_extension import load

    import tilelang

    tl_dir = Path(tilelang.__file__).resolve().parent
    return load(
        name="lbi_transformer_flash_jvp",
        sources=[str(Path(__file__).resolve().parent / "flash_jvp.cu")],
        extra_cuda_cflags=[
            "-O3", "-arch=sm_90a", "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        extra_include_paths=[str(tl_dir / "src"),
                             str(tl_dir / "3rdparty" / "cutlass" / "include")],
        verbose=False,
    )


def flash_jvp_available() -> bool:
    """True when the kernel can run here: Hopper GPU and a buildable
    extension (the build itself is deferred to first use)."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (9, 0)


@torch.library.custom_op("lbi::flash_jvp", mutates_args=())
def _flash_jvp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor,
               scale: float) -> list[torch.Tensor]:
    o, do = _module().flash_jvp_occ2_full(q, k, v, dq, dk, dv, scale)
    return [o, do]


@_flash_jvp.register_fake
def _(q, k, v, dq, dk, dv, scale):
    B, H, L, HD = q.shape
    r = dq.shape[0]
    return [q.new_empty(B, L, H * HD), dq.new_empty(r, B, L, H * HD)]


def flash_attention_jvp_cuda(q, k, v, dq, dk, dv, scale):
    """q/k/v [B, H, L, hd] bf16, dq/dk/dv [r, B, H, L, hd]; contract:
    hd == 64, L % 64 == 0. Returns (o [B, L, H*hd], do [r, B, L, H*hd]) --
    the epilogue writes the out_proj-ready flat layout directly. Odd r is
    padded to a lane pair internally."""
    r = dq.shape[0]
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()
    if r % 2:
        dq = torch.cat([dq, dq[-1:]], dim=0)
        dk = torch.cat([dk, dk[-1:]], dim=0)
        dv = torch.cat([dv, dv[-1:]], dim=0)
    o, do = _flash_jvp(q, k, v, dq, dk, dv, float(scale))
    return o, do[:r]
