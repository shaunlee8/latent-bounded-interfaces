# Copyright (c) 2026, Dao AI Lab, Goombalab.

from dataclasses import dataclass
import math

from einops import rearrange

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from backbones.mamba3.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from backbones.mamba3.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
import backbones.mamba3.ops.triton.mamba3.mamba3_siso_combined as _siso_combined_module


@dataclass
class Mamba3ForwardCache:
    input_u: Tensor
    in_proj: Tensor
    z: Tensor
    x: Tensor
    B: Tensor
    C: Tensor
    dd_dt: Tensor
    dd_A: Tensor
    trap: Tensor
    angles: Tensor
    ADT: Tensor
    DT: Tensor
    B_normed: Tensor
    C_normed: Tensor
    y_inner: Tensor
    output: Tensor
    # Optional forward-cache field reuse: the scan kernel's materialized
    # intermediates (q_rot, theta_cs, k_scaled, scale_s) for the construction.
    q_rot: "Tensor | None" = None
    theta_cs: "Tensor | None" = None
    k_scaled: "Tensor | None" = None
    scale_s: "Tensor | None" = None


class Mamba3(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        # ----------------------------------------
        # Mamba-3 configs
        rope_fraction=0.5,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_floor=1e-4,
        chunk_size=64,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.A_floor = A_floor

        self.d_inner = int(self.expand * self.d_model)
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = ngroups
        
        # RoPE flags
        assert rope_fraction in [0.5, 1.0]
        self.rotary_dim_divisor = int(2/rope_fraction)
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        # Order: [z, x, B, C, dd_dt, dd_A, trap, angle]
        d_in_proj = 2 * self.d_inner + 2 * self.d_state * self.num_bc_heads + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        # dt_bias parameterization        
        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True
        
        # B and C biases; the unit middle axis is the checkpoint layout.
        self.B_bias = nn.Parameter(1+torch.zeros((self.nheads, 1, self.d_state), dtype=torch.float32, device=device), requires_grad=True)
        self.C_bias = nn.Parameter(1+torch.zeros((self.nheads, 1, self.d_state), dtype=torch.float32, device=device), requires_grad=True)
                                                       
        # RMS Norm for B and C
        assert RMSNormGated is not None
        self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)


    def _forward_impl(self, u, *, return_cache: bool = False):
        """Full-sequence mixer forward; u is (batch, seqlen, hidden_dim) and the output has the same shape."""
        # Apply in_proj
        zxBCdtAtrap = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner, self.d_inner, 
                self.d_state * self.num_bc_heads,
                self.d_state * self.num_bc_heads,
                self.nheads, self.nheads, self.nheads, 
                self.num_rope_angles
            ],
            dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=1, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=1, g=self.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        # Compute ADT, DT
        _A = -F.softplus(dd_A.to(torch.float32)) # (B, L, N)
        _A = torch.clamp(_A, max=-self.A_floor)            
        DT = F.softplus(dd_dt + self.dt_bias) # (B, L, N)
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        # Compute angle
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1) # (B, L, N, S)

        # Apply RMS Norm on B and C
        B = self.B_norm(B)
        C = self.C_norm(C)

        # Apply Mamba-3 kernel
        # Under return_cache, capture the scan kernel's materialized
        # intermediates for the construction (see LBI_CAPTURE_SINK).
        cap = {} if return_cache else None
        if cap is not None:
            _siso_combined_module.LBI_CAPTURE_SINK = cap
        try:
            y = mamba3_siso_combined(
                Q=C.squeeze(2),
                K=B.squeeze(2),
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=trap,
                Q_bias=self.C_bias.squeeze(1),
                K_bias=self.B_bias.squeeze(1),
                Angles=angles,
                D=self.D,
                Z=z,
                chunk_size=self.chunk_size,
                Input_States=None,
                return_final_states=False,
            )
        finally:
            if cap is not None:
                _siso_combined_module.LBI_CAPTURE_SINK = None
        y = rearrange(y, "b l h p -> b l (h p)")

        out = self.out_proj(y.to(x.dtype))
        if not return_cache:
            return out
        return out, Mamba3ForwardCache(
            input_u=u,
            in_proj=zxBCdtAtrap,
            z=z,
            x=x,
            B=B,
            C=C,
            dd_dt=dd_dt,
            dd_A=dd_A,
            trap=trap,
            angles=angles,
            ADT=ADT,
            DT=DT,
            B_normed=B,
            C_normed=C,
            y_inner=y,
            output=out,
            q_rot=cap.get("q_rot") if cap else None,
            theta_cs=cap.get("theta_cs") if cap else None,
            k_scaled=cap.get("k_scaled") if cap else None,
            scale_s=cap.get("scale_s") if cap else None,
        )

    def forward(self, u):
        return self._forward_impl(u, return_cache=False)

    def forward_with_cache(self, u) -> tuple[Tensor, Mamba3ForwardCache]:
        out, cache = self._forward_impl(u, return_cache=True)
        return out, cache

    def input_pullback_matrix(self, cache: Mamba3ForwardCache, g_out: Tensor) -> Tensor:
        if g_out.dim() != 4:
            raise ValueError("g_out must have shape [B, P, L, D].")
        bsz, basis, seqlen, d_model = g_out.shape
        if cache.output.shape != (bsz, seqlen, d_model):
            raise ValueError("g_out shape does not match cached mixer output.")
        cols = []
        for pidx in range(basis):
            grad_out = g_out[:, pidx, :, :].to(device=cache.output.device, dtype=cache.output.dtype)
            grad_in = torch.autograd.grad(
                cache.output,
                cache.input_u,
                grad_outputs=grad_out,
                retain_graph=pidx + 1 < basis,
                create_graph=False,
                allow_unused=False,
            )[0]
            cols.append(grad_in.unsqueeze(1))
        return torch.cat(cols, dim=1).to(device=g_out.device, dtype=g_out.dtype)
