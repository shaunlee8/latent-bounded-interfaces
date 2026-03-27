from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from backbones.mamba2.mamba2_simple import Mamba2Simple
from backbones.mamba2.ops.triton.layernorm_gated import RMSNorm


class Mamba2Block(nn.Module):
    """Minimal upstream-style Add -> LN -> Mixer block for Mamba-2.

    This mirrors the non-fused `Block` semantics used upstream without pulling in
    the full Triton layer-norm stack. It is the correct training-time wrapper for
    `Mamba2Simple`; stacking bare mixers materially changes the architecture.
    """

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 128,
        ngroups: int = 1,
        chunk_size: int = 256,
        use_mem_eff_path: bool = True,
        bias: bool = False,
        conv_bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim, eps=1e-5, device=device, dtype=dtype)
        self.mixer = Mamba2Simple(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            use_mem_eff_path=use_mem_eff_path,
            bias=bias,
            conv_bias=conv_bias,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        seq_idx=None,
    ) -> tuple[Tensor, Tensor]:
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        hidden_states = self.mixer(hidden_states, seq_idx=seq_idx)
        return hidden_states, residual
