"""Differentiable pure-PyTorch reference for the Mamba-3 SISO scan forward.

This is the autodiff ORACLE for the forward-mode (JVP) interface-Jacobian work.
The registered Triton scan (`mamba3_siso_combined` / `mamba3_siso_fwd`) is a
custom autograd.Function that supports NEITHER functorch transforms nor
forward-mode dual numbers, so `torch.func.jvp` cannot push a message tangent
through it, and in bf16 finite differences are unusable. A faithful,
differentiable torch reproduction of the same scan is therefore the only exact
way to obtain the forward-mode scan JVP in-repo, and it is the parity oracle
the native forward-mode scan kernels gate against.

The math is ported verbatim from the upstream reference
(`~/src/mamba/tests/ops/triton/test_mamba3_siso.py::mamba3_siso_fwd_ref`), which
their own tests assert matches the Triton kernel. Restricted to the LBI region
case: batched (no varlen) and zero initial state (the cross-region state is the
bounded message `m_k`, NOT the SSM state -- each region's scan starts fresh).

Input conventions match the registered kernel exactly:
  Q, K   : [B, S, H_qk, Dqk]     (pre-rotary, pre-bias)
  V      : [B, S, H, Dv]
  ADT    : [B, H, S]             already the negative decay (-softplus(...)*dt)
  DT     : [B, H, S]             already positive (softplus(dd_dt + dt_bias))
  Trap   : [B, H, S]             raw (sigmoid applied here)
  Q_bias,
  K_bias : [H, Dqk]
  Angles : [B, S, H, Da]         raw (tanh*pi applied here)
  D      : [H] or None           skip
  Z      : [B, S, H, Dv] or None SiLU gate
Returns Out [B, S, H, Dv] with Z-gating applied (matches the kernel's `Out`).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import repeat

TWO_PI = 2.0 * math.pi


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Lower-triangular segment sum: out[..., t, s] = sum_{s < k <= t} x[..., k]."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def _rotary(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair rotary embedding (pairs (2i, 2i+1)), matching the kernel."""
    tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
    tensor_0 = tensor_reshaped[..., 0]
    tensor_1 = tensor_reshaped[..., 1]
    if cos.shape[-1] < tensor_0.shape[-1]:
        pad = tensor_0.shape[-1] - cos.shape[-1]
        cos = F.pad(cos, (0, pad), value=1.0)
        sin = F.pad(sin, (0, pad), value=0.0)
    rotated_0 = tensor_0 * cos - tensor_1 * sin
    rotated_1 = tensor_0 * sin + tensor_1 * cos
    return torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)


def mamba3_siso_out_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Gated SISO scan output `Out` [B, S, H, Dv], fully differentiable.

    Zero initial state, batched. `compute_dtype` is the working precision (use
    fp32 for the forward-mode oracle; the recurrence is exact linear algebra so
    fp32 gives a clean JVP even when the model runs bf16)."""
    batch, seqlen, nheads_qk, _ = Q.shape
    _, _, nheads, headdim_v = V.shape

    Q = Q.to(compute_dtype)
    K = K.to(compute_dtype)
    V = V.to(compute_dtype)
    ADT = ADT.to(torch.float32)
    DT = DT.to(torch.float32)
    Trap = Trap.to(compute_dtype)
    Q_bias = Q_bias.to(compute_dtype)
    K_bias = K_bias.to(compute_dtype)
    Angles = Angles.to(compute_dtype)
    if D is not None:
        D = D.to(compute_dtype)
    if Z is not None:
        Z = Z.to(compute_dtype)

    Angles = torch.tanh(Angles) * math.pi
    # GQA expand Q/K to full head count.
    if nheads_qk != nheads:
        Q = repeat(Q, "b s hbc d -> b s (hbc g) d", g=nheads // nheads_qk)
        K = repeat(K, "b s hbc d -> b s (hbc g) d", g=nheads // nheads_qk)

    out_zs = []
    for seq_idx in range(batch):
        Q_curr = Q[seq_idx]            # [S, H, Dqk]
        K_curr = K[seq_idx]
        V_curr = V[seq_idx]            # [S, H, Dv]
        ADT_curr = ADT[seq_idx]        # [H, S]
        DT_curr = DT[seq_idx]          # [H, S]
        Trap_curr = torch.sigmoid(Trap[seq_idx])   # [H, S]
        Angles_curr = Angles[seq_idx]  # [S, H, Da]
        Z_curr = Z[seq_idx] if Z is not None else None

        # Discretized cumulative rotary angle, wrapped to [0, 2pi).
        angles_scaled = Angles_curr.float() * DT_curr.transpose(0, 1).unsqueeze(-1)
        angles_cumsum = torch.cumsum(angles_scaled, dim=0)
        angles_cumsum = angles_cumsum - TWO_PI * torch.floor(angles_cumsum / TWO_PI)

        # Shifted trapezoidal gamma / scale factors.
        DT_shifted = F.pad(DT_curr[:, 1:], (0, 1))
        Trap_shifted = F.pad(Trap_curr[:, 1:], (0, 1))
        shifted_gamma = DT_shifted * (1 - Trap_shifted)
        scale = DT_curr * Trap_curr + DT_shifted * (1 - Trap_shifted)

        Q_curr = Q_curr + Q_bias.unsqueeze(0)
        K_curr = K_curr + K_bias.unsqueeze(0)

        # Local QK-dot skip term (uses pre-rotary Q/K).
        QK_dot = torch.sum(K_curr * Q_curr, dim=-1) * shifted_gamma.transpose(0, 1)

        cos_a = torch.cos(angles_cumsum).to(Q_curr.dtype)
        sin_a = torch.sin(angles_cumsum).to(Q_curr.dtype)
        Q_curr = _rotary(Q_curr, cos_a, sin_a)
        K_curr = _rotary(K_curr, cos_a, sin_a)

        K_scaled = K_curr * scale.transpose(0, 1).unsqueeze(-1).to(K_curr.dtype)

        # Quadratic (attention) form of the decayed linear scan.
        QK = torch.einsum("thd,shd->hts", Q_curr, K_scaled)
        QK_causal = torch.tril(QK)
        QK_causal = (QK_causal * torch.exp(_segsum(ADT_curr))).to(QK_causal.dtype)
        out = torch.einsum("hts,shd->thd", QK_causal, V_curr)

        if D is not None:
            out = out + D[None, :, None] * V_curr
        out = out - V_curr * QK_dot.unsqueeze(-1)

        if Z_curr is not None:
            out = out * Z_curr * torch.sigmoid(Z_curr)
        out_zs.append(out)

    return torch.stack(out_zs, dim=0)


def mamba3_siso_jvp(
    primals: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-mode JVP of the SISO scan: push scan-input tangents to `dOut`.

    `primals` / `tangents` are ordered (Q, K, V, ADT, DT, Trap, Q_bias, K_bias,
    Angles[, D][, Z]) -- the differentiable scan inputs. Returns (Out, dOut).
    This is the forward-linearized scan (the compute-bound regime); the native
    forward-mode kernel implements exactly this map. Tangents for inputs that do
    not vary in a given direction should be zeros_like."""

    def f(*args: torch.Tensor) -> torch.Tensor:
        return mamba3_siso_out_ref(*args, compute_dtype=compute_dtype)

    return torch.func.jvp(f, primals, tangents)
