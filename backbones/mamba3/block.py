from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from backbones.mamba3.mamba3 import Mamba3
from backbones.mamba3.ops.triton.layernorm_gated import RMSNorm


class Mamba3Block(nn.Module):
    """Minimal Add -> RMSNorm -> Mixer block for the LBI Mamba-3 SISO port."""

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 64,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim, eps=1e-5, device=device, dtype=dtype)
        self.mixer = Mamba3(
            d_model=dim,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            is_mimo=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, **kwargs) -> tuple[Tensor, Tensor]:
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        hidden_states = self.mixer(hidden_states, **kwargs)
        return hidden_states, residual
