from __future__ import annotations

import torch
from torch import nn

from backbones.transformer.attention import CausalSelfAttention
from backbones.transformer.mlp import GatedMLP


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x_float * rms
        return out.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        d_intermediate: int = 0,
        mlp_ratio: float = 4.0,
        rope_base: float = 10000.0,
        rope_interleaved: bool = False,
        softmax_scale: float | None = None,
        d_conv: int = 0,
        use_flash_attn: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = RMSNorm(dim, eps=1e-5)
        self.attn = CausalSelfAttention(
            dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rope_base=rope_base,
            rope_interleaved=rope_interleaved,
            softmax_scale=softmax_scale,
            d_conv=d_conv,
            use_flash_attn=use_flash_attn,
            bias=bias,
        )
        self.norm2 = RMSNorm(dim, eps=1e-5)
        hidden_features = d_intermediate if d_intermediate > 0 else None
        if hidden_features is None and mlp_ratio > 0.0:
            hidden_features = max(1, int(dim * mlp_ratio))
        self.mlp = GatedMLP(dim, hidden_features=hidden_features, out_features=dim, bias=bias)

    def _prenorm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        norm: RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = (hidden_states + residual) if residual is not None else hidden_states
        normed = norm(residual.to(dtype=norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        return normed, residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self._prenorm(hidden_states, residual, self.norm)
        hidden_states = self.attn(hidden_states)
        hidden_states, residual = self._prenorm(hidden_states, residual, self.norm2)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
