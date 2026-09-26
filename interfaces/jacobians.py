"""Interface Jacobian helpers: the autograd-backed input Jacobian transpose
of a module, and the closed-form LayerNorm input JVP."""

from __future__ import annotations

import torch
import torch.nn as nn


def _module_input_jacobian_t_apply_autograd(
    module: nn.Module,
    x: torch.Tensor,
    g_out: torch.Tensor,
) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("module input Jacobian-transpose apply expects x shaped [B, D].")
    if g_out.dim() != 3:
        raise ValueError("module input Jacobian-transpose apply expects g_out shaped [B, P, D_out].")
    bsz, basis = g_out.shape[:2]
    # The helper builds its own local graph, so it also runs under a caller's no_grad.
    with torch.enable_grad():
        x_rep = (
            x.detach()
            .unsqueeze(1)
            .expand(bsz, basis, x.shape[-1])
            .reshape(bsz * basis, x.shape[-1])
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
    return g_in.reshape(bsz, basis, x.shape[-1]).to(device=g_out.device, dtype=g_out.dtype)

def _layernorm_input_jacobian_apply(
    norm: nn.LayerNorm,
    x: torch.Tensor,
    d_in: torch.Tensor,
) -> torch.Tensor:
    """Forward-mode JVP of LayerNorm w.r.t. its input (affine bias drops out)."""
    if x.dim() != 2:
        raise ValueError("LayerNorm input Jacobian apply expects x shaped [B, D].")
    if d_in.dim() != 3:
        raise ValueError("LayerNorm input Jacobian apply expects d_in shaped [B, P, D].")
    # fp32 compute factored through the normalized quantities (x_hat, g), so
    # every intermediate stays O(1) at any pre-norm magnitude.
    compute_dtype = torch.promote_types(torch.promote_types(x.dtype, d_in.dtype), torch.float32)
    x_c = x.to(device=d_in.device, dtype=compute_dtype)
    d_c = d_in.to(dtype=compute_dtype)
    eps = float(norm.eps)
    mean = x_c.mean(-1, keepdim=True)
    x_centered = x_c - mean                                     # [B, D]
    var = (x_centered * x_centered).mean(-1, keepdim=True)      # [B, 1]
    istd = torch.rsqrt(var + eps)                               # [B, 1]
    x_hat = x_centered * istd                                   # [B, D], O(1)
    d_mean = d_c.mean(-1, keepdim=True)                         # [B, P, 1]
    d_centered = d_c - d_mean                                   # [B, P, D]
    g = d_centered * istd.unsqueeze(1)                          # [B, P, D]
    d_norm = g - x_hat.unsqueeze(1) * (x_hat.unsqueeze(1) * g).mean(-1, keepdim=True)
    weight = norm.weight.to(device=d_in.device, dtype=compute_dtype) if norm.weight is not None else None
    if weight is not None:
        d_norm = d_norm * weight.view(1, 1, -1)
    return d_norm.to(device=d_in.device, dtype=d_in.dtype)
