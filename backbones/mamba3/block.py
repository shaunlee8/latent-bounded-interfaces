from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from backbones.mamba3.mamba3 import Mamba3, Mamba3ForwardCache
from backbones.mamba3.ops.triton.layernorm_gated import RMSNorm


@dataclass
class Mamba3BlockForwardCache:
    hidden_input: Tensor
    residual_input: Optional[Tensor]
    residual_output: Tensor
    norm_input: Tensor
    mixer_cache: Mamba3ForwardCache
    hidden_output: Tensor


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

    def _module_input_pullback_matrix_autograd(
        self,
        module: nn.Module,
        x: Tensor,
        g_out: Tensor,
    ) -> Tensor:
        if x.dim() != 3:
            raise ValueError("block pullback expects x shaped [B, L, D].")
        if g_out.dim() != 4:
            raise ValueError("block pullback expects g_out shaped [B, P, L, D].")
        bsz, basis, seqlen, dim = g_out.shape
        x_rep = (
            x.detach()
            .unsqueeze(1)
            .expand(bsz, basis, seqlen, dim)
            .reshape(bsz * basis, seqlen, dim)
            .requires_grad_(True)
        )
        y_rep = module(x_rep)
        g_rep = g_out.to(device=y_rep.device, dtype=y_rep.dtype).reshape_as(y_rep)
        g_in = torch.autograd.grad(
            y_rep,
            x_rep,
            grad_outputs=g_rep,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        return g_in.reshape(bsz, basis, seqlen, dim).to(device=g_out.device, dtype=g_out.dtype)

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, **kwargs) -> tuple[Tensor, Tensor]:
        hidden_states, residual, _ = self.forward_with_cache(hidden_states, residual=residual, **kwargs)
        return hidden_states, residual

    def forward_with_cache(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor, Mamba3BlockForwardCache]:
        residual_out = (hidden_states + residual) if residual is not None else hidden_states
        norm_input = self.norm(residual_out.to(dtype=self.norm.weight.dtype))
        hidden_out, mixer_cache = self.mixer.forward_with_cache(norm_input, **kwargs)
        return hidden_out, residual_out, Mamba3BlockForwardCache(
            hidden_input=hidden_states,
            residual_input=residual,
            residual_output=residual_out,
            norm_input=norm_input,
            mixer_cache=mixer_cache,
            hidden_output=hidden_out,
        )

    def input_pullback_matrix(
        self,
        cache: Mamba3BlockForwardCache,
        g_hidden_out: Tensor,
        g_residual_out: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        if g_hidden_out.dim() != 4:
            raise ValueError("g_hidden_out must have shape [B, P, L, D].")
        g_norm_input = self.mixer.input_pullback_matrix(cache.mixer_cache, g_hidden_out)
        g_residual_from_norm = self._module_input_pullback_matrix_autograd(self.norm, cache.residual_output, g_norm_input)
        if g_residual_out is None:
            g_residual_total = g_residual_from_norm
        else:
            g_residual_total = g_residual_from_norm + g_residual_out.to(
                device=g_residual_from_norm.device, dtype=g_residual_from_norm.dtype
            )
        g_hidden_in = g_residual_total.to(device=g_hidden_out.device, dtype=g_hidden_out.dtype)
        if cache.residual_input is None:
            return g_hidden_in, None
        return g_hidden_in, g_hidden_in.clone()
