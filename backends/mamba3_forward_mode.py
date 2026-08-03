"""Forward-mode (JVP) region output map for the Mamba-3 SISO backend.

`region_output_jvp` pushes an r-wide region-input tangent basis through the
region to a region-output tangent basis. This is the forward-mode factor of
the interface Jacobian A_k, dual to the reverse-mode `input_pullback_basis`.

Two implementations share one map: a pure-torch reference (`torch.func.jvp`
over the differentiable scan reference `mamba3_siso_out_ref`, norms via
`rms_norm_ref`) and the kernel-backed path (`*_kernel` entry points) running
the fused tilelang/CUDA dual-scan kernels. The reference validates the
kernels; both are gated against each other and reverse mode.

SISO only (`is_mimo=False`, `is_outproj_norm=False`), the LBI backend config.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from backbones.mamba3.ops.triton.mamba3.mamba3_siso_ref import mamba3_siso_out_ref
from backbones.mamba3.ops.triton.mamba3.mamba3_siso_dualscan_ref import _rotary_fwd
from backbones.mamba3.ops.triton.layernorm_gated import rms_norm_ref


def _rmsnorm_ref(x: torch.Tensor, norm: Any) -> torch.Tensor:
    bias = getattr(norm, "bias", None)
    return rms_norm_ref(x, norm.weight, bias, eps=float(norm.eps))


def _gwalk_available() -> bool:
    """Is the lane-batched wgmma dual walk enabled and built? (Shape
    eligibility is checked separately in `_gwalk_mod`.)"""
    if os.environ.get("LBI_FWDMODE_GWALK", "1") != "1":
        return False
    try:
        import cuda.mamba3 as _cm
        return _cm.mamba3_lbi_cuda is not None and hasattr(
            _cm.mamba3_lbi_cuda, "gwalk_full")
    except Exception:
        return False


def _gwalk_mod(N: int, P: int, Da: int, Sp: int):
    """The gwalk extension when enabled AND the shape matches the kernel's
    compile-time design (N=128, P=64, cs=32 walk), else None.
    IMPORTANT: gwalk requires the fields laid out at cs=32 (L/DL are
    chunk-local cumsums), which `mamba3_mixer_jvp_kernel` forces when
    `_gwalk_available()`."""
    if not _gwalk_available():
        return None
    if not (N == 128 and P == 64 and Sp % 32 == 0 and Da <= 64 and Da % 4 == 0):
        return None
    import cuda.mamba3 as _cm
    return _cm.mamba3_lbi_cuda


def mamba3_mixer_forward_ref(mixer: Any, u: torch.Tensor, *, compute_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Differentiable pure-torch reproduction of the SISO mixer forward.

    Matches `Mamba3._forward_impl` (is_mimo=False, is_outproj_norm=False) with
    the registered scan swapped for `mamba3_siso_out_ref` and the B/C norms for
    `rms_norm_ref`. Validated against the real mixer at bf16 tolerance."""
    if getattr(mixer, "is_mimo", False):
        raise NotImplementedError("forward-mode region JVP supports SISO (is_mimo=False) only.")
    if getattr(mixer, "is_outproj_norm", False):
        raise NotImplementedError("forward-mode region JVP supports is_outproj_norm=False only.")

    di, ds = mixer.d_inner, mixer.d_state
    g, r, nh = mixer.num_bc_heads, mixer.mimo_rank, mixer.nheads
    nra = mixer.num_rope_angles

    proj = u.to(mixer.in_proj.weight.dtype) @ mixer.in_proj.weight.t()
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        proj, [di, di, ds * g * r, ds * g * r, nh, nh, nh, nra], dim=-1
    )
    z = rearrange(z, "b l (h p) -> b l h p", p=mixer.headdim)
    x = rearrange(x, "b l (h p) -> b l h p", p=mixer.headdim)
    B = rearrange(B, "b l (r g n) -> b l r g n", r=r, g=g)
    C = rearrange(C, "b l (r g n) -> b l r g n", r=r, g=g)
    trap = rearrange(trap, "b l h -> b h l")

    _A = -F.softplus(dd_A.float())
    _A = torch.clamp(_A, max=-mixer.A_floor)
    DT = F.softplus(dd_dt + mixer.dt_bias)
    ADT = _A * DT
    DT = rearrange(DT, "b l n -> b n l")
    ADT = rearrange(ADT, "b l n -> b n l")

    angles = angles.unsqueeze(-2).expand(-1, -1, nh, -1)

    B = _rmsnorm_ref(B, mixer.B_norm)
    C = _rmsnorm_ref(C, mixer.C_norm)

    y = mamba3_siso_out_ref(
        C.squeeze(2), B.squeeze(2), x, ADT, DT, trap,
        mixer.C_bias.squeeze(1), mixer.B_bias.squeeze(1), angles,
        mixer.D, z, compute_dtype=compute_dtype,
    )
    y = rearrange(y, "b l h p -> b l (h p)")
    return y.to(mixer.out_proj.weight.dtype) @ mixer.out_proj.weight.t()


def _mixer_preprocess(mixer: Any, u: torch.Tensor, cd: torch.dtype) -> tuple:
    """SISO mixer preprocess: region-input `u` -> the 11 scan inputs
    (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z), matching the args of
    `mamba3_siso_out_ref`. Pure torch (functorch-safe) so `torch.func.jvp` can
    push a tangent through it. This is the decode/preprocess half of the
    region JVP."""
    di, ds = mixer.d_inner, mixer.d_state
    g, rk, nh = mixer.num_bc_heads, mixer.mimo_rank, mixer.nheads
    nra = mixer.num_rope_angles
    proj = u.to(cd) @ mixer.in_proj.weight.t().to(cd)
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        proj, [di, di, ds * g * rk, ds * g * rk, nh, nh, nh, nra], dim=-1)
    z = rearrange(z, "b l (h p) -> b l h p", p=mixer.headdim)
    x = rearrange(x, "b l (h p) -> b l h p", p=mixer.headdim)
    B = rearrange(B, "b l (r g n) -> b l r g n", r=rk, g=g)
    C = rearrange(C, "b l (r g n) -> b l r g n", r=rk, g=g)
    trap = rearrange(trap, "b l h -> b h l")
    _A = -F.softplus(dd_A.float())
    _A = torch.clamp(_A, max=-mixer.A_floor)
    DT = F.softplus(dd_dt + mixer.dt_bias.to(dd_dt.dtype))
    ADT = _A * DT
    DT = rearrange(DT, "b l n -> b n l")
    ADT = rearrange(ADT, "b l n -> b n l")
    angles = angles.unsqueeze(-2).expand(-1, -1, nh, -1)
    B = _rmsnorm_ref(B, mixer.B_norm)
    C = _rmsnorm_ref(C, mixer.C_norm)
    return (C.squeeze(2), B.squeeze(2), x, ADT, DT, trap,
            mixer.C_bias.squeeze(1), mixer.B_bias.squeeze(1), angles, mixer.D, z)


def _rmsnorm_fwd_jvp(x: torch.Tensor, dx: torch.Tensor, norm: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm forward + batched JVP over the last dim. x [.,n], dx [r,.,n].
    Bias (if present) shifts the forward only (zero JVP)."""
    w = norm.weight.float()
    eps = float(norm.eps)
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    xn = xf * rstd * w
    bias = getattr(norm, "bias", None)
    if bias is not None:
        xn = xn + bias.float()
    dvar = 2.0 * (xf.unsqueeze(0) * dx.float()).mean(-1, keepdim=True)     # [r,.,1]
    drstd = -0.5 * rstd.unsqueeze(0).pow(3) * dvar                          # [r,.,1]
    dxn = dx.float() * rstd.unsqueeze(0) * w + (xf * w).unsqueeze(0) * drstd
    return xn.to(x.dtype), dxn.to(x.dtype)


def _mixer_preprocess_fwd_jvp(mixer: Any, u: torch.Tensor, du: torch.Tensor, cd: torch.dtype):
    """Explicit (no torch.func) SISO mixer preprocess FORWARD + batched JVP.
    `du` is [r, B, L, D] (r message directions). Returns the 11 primal scan
    inputs and the r-batched scan-input tangents (dQ,dK,dV,dADT,dDT,dTrap,None,
    None,dAng,None,dZ); biases and D carry no tangent. Analytic dual of
    `_mixer_preprocess`; validated bit-close vs torch.func.jvp."""
    W = mixer.in_proj.weight.t().to(cd)
    # Primal projection in bf16: the real model's in_proj runs bf16, so bf16
    # here is both faster and more faithful than fp32.
    proj = (u.to(torch.bfloat16) @ W.to(torch.bfloat16)).to(cd)    # [B,L,dproj]
    # bf16 tangent projection: the largest tensor in the path (~1.3 GB fp32
    # at dim1024/r8); bf16 halves its traffic and every downstream pass.
    dproj = (du.to(torch.bfloat16) @ W.to(torch.bfloat16))  # [r,B,L,dproj] bf16
    return _preproc_tail(mixer, proj, dproj, cd)


def _mixer_preprocess_bcast_fwd_jvp(mixer: Any, norm: Any, u: torch.Tensor,
                                    du_cond: torch.Tensor, cd: torch.dtype):
    """Fused block-norm JVP + tangent projection for L-broadcast region
    tangents (`du(l) = du_cond` at every position). `u` is the pre-norm block
    input [B, L, D]; `du_cond` is [r, B, D]. Exact rank-2-in-L algebra:

        dproj[r,b,l] = rstd[b,l] * G[r,b] + coef[r,b,l] * proj_nb[b,l],
        G = (du_cond * w) @ W_in    (the L-collapsed GEMM),
        coef = -0.5 * rstd^2 * dvar,  dvar = 2*mean_D(x * du_cond),

    so neither the [r,B,L,D] norm tangent nor the [r*B*L,D] projection GEMM
    is ever formed. Returns (norm_in, (scan_inputs, tangents))."""
    w = norm.weight.float()
    eps = float(norm.eps)
    xf = u.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)                                  # [B,L,1]
    xn = xf * rstd * w
    bias = getattr(norm, "bias", None)
    if bias is not None:
        xn = xn + bias.float()
    W = mixer.in_proj.weight.t()
    proj_b = (xn.to(torch.bfloat16) @ W.to(torch.bfloat16))        # [B,L,dproj]
    proj = proj_b.to(cd)
    G = ((du_cond.float() * w).to(torch.bfloat16) @ W.to(torch.bfloat16))  # [r,B,dproj]
    dvar = 2.0 * torch.einsum("bld,rbd->rbl", xf, du_cond.float()) / xf.shape[-1]
    coef = (-0.5 * rstd.squeeze(-1).pow(2)).unsqueeze(0) * dvar    # [r,B,L]
    proj_nb = proj_b.float()
    if bias is not None:
        proj_nb = proj_nb - (bias.to(W.dtype) @ W).float()
    dproj = (rstd.squeeze(-1).unsqueeze(0).unsqueeze(-1) * G.unsqueeze(2).float()
             + coef.unsqueeze(-1) * proj_nb.unsqueeze(0)).to(torch.bfloat16)
    return xn.to(u.dtype), _preproc_tail(mixer, proj, dproj, cd)


def _native_qkcs(q_rot, theta_cs, Kg, Kb, Sp):
    """QR/KR/COS/SIN assembled from the forward-captured intermediates in
    kernel-native layouts: `q_rot` [B,S,H,N] is used as QR directly,
    `theta_cs` [B,S,H,Da] gives COS/SIN, and K_r is recomputed from group-K +
    bias with the same captured angles. The construction linearizes at the
    true forward rotations."""
    B, S, H, Da = theta_cs.shape
    two_pi = 2.0 * math.pi
    t = theta_cs.float()
    t = t - two_pi * torch.floor(t / two_pi)
    cos_s, sin_s = torch.cos(t), torch.sin(t)                 # [B,S,H,Da] fp32
    G = Kg.shape[2]
    Kh = Kg if G == H else Kg.repeat_interleave(H // G, dim=2)
    Kfull = Kh.float() + Kb.view(1, 1, H, -1).float()
    KRn = _rotary_fwd(Kfull, cos_s, sin_s)                    # [B,S,H,N]
    bf16 = torch.bfloat16
    if Sp > S:
        QR = F.pad(q_rot, (0, 0, 0, 0, 0, Sp - S))
        KR = F.pad(KRn.to(bf16), (0, 0, 0, 0, 0, Sp - S)).contiguous()
        COS = F.pad(cos_s.permute(0, 2, 1, 3), (0, 0, 0, Sp - S)).to(bf16).contiguous()
        SIN = F.pad(sin_s.permute(0, 2, 1, 3), (0, 0, 0, Sp - S)).to(bf16).contiguous()
    else:
        QR = q_rot
        KR = KRn.to(bf16).contiguous()
        COS = cos_s.permute(0, 2, 1, 3).to(bf16).contiguous()
        SIN = sin_s.permute(0, 2, 1, 3).to(bf16).contiguous()
    return QR, KR, COS, SIN


def _mixer_preprocess_cached_fwd_jvp(mixer: Any, du: torch.Tensor, cd: torch.dtype,
                                     proj_c: torch.Tensor):
    """Preprocess JVP reading the primal projection from the banked forward
    cache (`Mamba3ForwardCache.in_proj`) instead of recomputing it; only the
    r-fold tangent GEMM runs, and the JVP linearizes at the true forward
    values."""
    W = mixer.in_proj.weight.t()
    dproj = (du.to(torch.bfloat16) @ W.to(torch.bfloat16))         # [r,B,L,dproj]
    return _preproc_tail(mixer, proj_c.to(cd), dproj, cd)


def _mixer_preprocess_bcast_cached_fwd_jvp(mixer: Any, norm: Any, u: torch.Tensor,
                                           du_cond: torch.Tensor, cd: torch.dtype,
                                           proj_c: torch.Tensor, xn_c: torch.Tensor):
    """Broadcast variant of the cached preprocess JVP: primal norm and
    projection come from the banked cache; the tangent algebra keeps only the
    cheap rstd/dvar reductions over the pre-norm input."""
    w = norm.weight.float()
    eps = float(norm.eps)
    xf = u.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)                                  # [B,L,1]
    bias = getattr(norm, "bias", None)
    W = mixer.in_proj.weight.t()
    G = ((du_cond.float() * w).to(torch.bfloat16) @ W.to(torch.bfloat16))  # [r,B,dproj]
    dvar = 2.0 * torch.einsum("bld,rbd->rbl", xf, du_cond.float()) / xf.shape[-1]
    coef = (-0.5 * rstd.squeeze(-1).pow(2)).unsqueeze(0) * dvar    # [r,B,L]
    proj_nb = proj_c.float()
    if bias is not None:
        proj_nb = proj_nb - (bias.to(W.dtype) @ W).float()
    dproj = (rstd.squeeze(-1).unsqueeze(0).unsqueeze(-1) * G.unsqueeze(2).float()
             + coef.unsqueeze(-1) * proj_nb.unsqueeze(0)).to(torch.bfloat16)
    return xn_c, _preproc_tail(mixer, proj_c.to(cd), dproj, cd)


def _preproc_tail(mixer: Any, proj: torch.Tensor, dproj: torch.Tensor, cd: torch.dtype):
    """Shared tail of the preprocess fwd+JVP: split proj/dproj and apply the
    analytic elementwise/norm JVPs down to the 11 scan inputs + tangents."""
    di, ds = mixer.d_inner, mixer.d_state
    g, rk, nh = mixer.num_bc_heads, mixer.mimo_rank, mixer.nheads
    nra = mixer.num_rope_angles
    sizes = [di, di, ds * g * rk, ds * g * rk, nh, nh, nh, nra]
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(proj, sizes, dim=-1)
    dz, dx, dB, dC, ddd_dt, ddd_A, dtrap, dangles = torch.split(dproj, sizes, dim=-1)

    def rr(t, pat, **kw):
        return rearrange(t, pat, **kw)
    z = rr(z, "b l (h p) -> b l h p", p=mixer.headdim); dz = rr(dz, "r b l (h p) -> r b l h p", p=mixer.headdim)
    x = rr(x, "b l (h p) -> b l h p", p=mixer.headdim); dx = rr(dx, "r b l (h p) -> r b l h p", p=mixer.headdim)
    B = rr(B, "b l (r g n) -> b l r g n", r=rk, g=g); dB = rr(dB, "z b l (r g n) -> z b l r g n", r=rk, g=g)
    C = rr(C, "b l (r g n) -> b l r g n", r=rk, g=g); dC = rr(dC, "z b l (r g n) -> z b l r g n", r=rk, g=g)
    trap = rr(trap, "b l h -> b h l"); dtrap = rr(dtrap, "z b l h -> z b h l")

    # softplus / clamp for A, softplus for DT (analytic JVP).
    sp_A = F.softplus(dd_A.float())
    _A_pre = -sp_A
    mask = (_A_pre < -mixer.A_floor).float()               # 1 where not clamped
    _A = torch.clamp(_A_pre, max=-mixer.A_floor)
    d_A = (-torch.sigmoid(dd_A.float()) * ddd_A.float()) * mask.unsqueeze(0)   # [r,B,L,nh]
    pre_dt = dd_dt + mixer.dt_bias.to(dd_dt.dtype)
    DT = F.softplus(pre_dt)
    dDT = torch.sigmoid(pre_dt.float()).unsqueeze(0) * ddd_dt.float()          # [r,B,L,nh]
    ADT = _A * DT.float()
    dADT = d_A * DT.float().unsqueeze(0) + _A.unsqueeze(0) * dDT
    DT = rr(DT, "b l n -> b n l"); ADT = rr(ADT, "b l n -> b n l")
    dDT = rr(dDT, "z b l n -> z b n l"); dADT = rr(dADT, "z b l n -> z b n l")

    angles = angles.unsqueeze(-2).expand(-1, -1, nh, -1)
    dangles = dangles.unsqueeze(-2).expand(-1, -1, -1, nh, -1)

    Bn, dBn = _rmsnorm_fwd_jvp(B, dB, mixer.B_norm)
    Cn, dCn = _rmsnorm_fwd_jvp(C, dC, mixer.C_norm)

    scan_inputs = (Cn.squeeze(2), Bn.squeeze(2), x, ADT, DT, trap,
                   mixer.C_bias.squeeze(1), mixer.B_bias.squeeze(1), angles, mixer.D, z)
    tangents = (dCn.squeeze(3), dBn.squeeze(3), dx, dADT, dDT, dtrap,
                None, None, dangles, None, dz)
    return scan_inputs, tangents


def _mixer_postprocess(mixer: Any, y_scan: torch.Tensor) -> torch.Tensor:
    """SISO mixer postprocess: gated scan output [B, L, H, Dv] -> mixer output
    [B, L, D] via out_proj. The map is linear, so its JVP is out_proj on the
    tangent."""
    y = rearrange(y_scan, "b l h p -> b l (h p)")
    return y.to(mixer.out_proj.weight.dtype) @ mixer.out_proj.weight.t()


def _finalize_postprocess_fused(out_quad, dout_quad, tf, w_out_t):
    """Finalize (D/QK-dot/Z-gate) + out_proj postprocess in one compiled
    unit: [.,S,BH,Dv] view-reshapes to [.,S,B,H*Dv] with no copy, so the
    matmul runs straight off the fused elementwise chain; only the D-wide
    result pays one transpose. `w_out_t` is out_proj.weight.t()."""
    B, S, H, Dv = tf["B"], tf["S"], tf["H"], tf["Dv"]
    r = tf["r"]
    Vf, dVf, qkdot, dqkdot = tf["Vf"], tf["dVf"], tf["qkdot"], tf["dqkdot"]
    out = out_quad.to(tf["cd"])                                   # [S,BH,Dv]
    dout = dout_quad.to(tf["cd"])                                 # [r,S,BH,Dv]
    if tf["Df"] is not None:
        out = out + tf["Df"].view(1, -1, 1) * Vf
        dout = dout + tf["Df"].view(1, 1, -1, 1) * dVf
    out = out - Vf * qkdot.unsqueeze(-1)
    dout = dout - (dVf * qkdot.unsqueeze(-1) + Vf.unsqueeze(0) * dqkdot.unsqueeze(-1))
    if tf["Zf"] is not None:
        Zf, dZf = tf["Zf"], tf["dZf"]
        sig = torch.sigmoid(Zf)
        gate = Zf * sig
        dgate = (sig * (1 + Zf * (1 - sig))) * dZf
        dout = dout * gate.unsqueeze(0) + out.unsqueeze(0) * dgate
        out = out * gate
    wd = w_out_t.dtype
    y = out.reshape(S, B, H * Dv).to(wd) @ w_out_t                # [S,B,D]
    dy = dout.reshape(r, S, B, H * Dv).to(wd) @ w_out_t           # [r,S,B,D]
    return y.transpose(0, 1).contiguous(), dy.permute(0, 2, 1, 3).contiguous()


def _fp8_sim(t, name):
    """fp8-storage precision gate: rowwise-scaled e4m3 round-trip of a tangent
    stream (env LBI_FWDMODE_FP8SIM = "1"/"all"/comma-list); no-op unset."""
    spec = os.environ.get("LBI_FWDMODE_FP8SIM", "")
    if not spec or t is None:
        return t
    if spec not in ("1", "all") and name not in spec.split(","):
        return t
    tf32 = t.float()
    s = tf32.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 448.0
    q = (tf32 / s).to(torch.float8_e4m3fn)
    return (q.to(torch.float32) * s).to(t.dtype)


def _finalize_postprocess_native(OUTn, DOUTn, dVfn, dZfn, tf, w_out_t):
    """Finalize consuming kernel-native layouts: OUTn [B,S,H,Dv] and DOUTn
    [r,B,S,H,Dv] are the walk outputs as S-sliced views, dVfn/dZfn are
    tb[2]/tb[10] read in place, and primal fields arrive as strided views.
    dy leaves the out_proj GEMM already in [r,B,S,D], the stacked d_mix
    the region loop consumes."""
    B, S, H, Dv = tf["B"], tf["S"], tf["H"], tf["Dv"]
    r = tf["r"]
    Vfn = tf["Vf"].view(S, B, H, Dv).permute(1, 0, 2, 3)          # [B,S,H,Dv]
    qkn = tf["qkdot"].view(S, B, H).permute(1, 0, 2).unsqueeze(-1)
    dqkn = tf["dqkdot"].view(r, S, B, H).permute(0, 2, 1, 3).unsqueeze(-1)
    out = OUTn.to(tf["cd"])                                       # [B,S,H,Dv]
    dout = DOUTn.to(tf["cd"])                                     # [r,B,S,H,Dv]
    if tf["Df"] is not None:
        Dfn = tf["Df"].view(B, 1, H, 1)
        out = out + Dfn * Vfn
        dout = dout + Dfn.unsqueeze(0) * dVfn
    out = out - Vfn * qkn
    dout = dout - (dVfn * qkn + Vfn.unsqueeze(0) * dqkn)
    if tf["Zf"] is not None:
        Zfn = tf["Zf"].view(S, B, H, Dv).permute(1, 0, 2, 3)
        sig = torch.sigmoid(Zfn)
        gate = Zfn * sig
        dgate = (sig * (1 + Zfn * (1 - sig))) * dZfn
        dout = dout * gate.unsqueeze(0) + out.unsqueeze(0) * dgate
        out = out * gate
    wd = w_out_t.dtype
    y = out.reshape(B, S, H * Dv).to(wd) @ w_out_t                # [B,S,D]
    dy = dout.reshape(r, B, S, H * Dv).to(wd) @ w_out_t           # [r,B,S,D]
    return y, dy


def _gen_fields_layout(pf: dict, tb: tuple, Sp: int, cs: int, mean_mode: bool = False,
                       skip_raw: bool = False, skip_fin: bool = False,
                       skip_qkcs: bool = False):
    """Torch-side prep for the generation kernel. Emits the small kernel
    inputs: raw group-level tangents (no GQA expand, no rotary), primal
    K_r/cos/sin/scale fields, cross-chunk tangent pieces (dtheta cumsum,
    dscale, dL), and the finalize fields. The [r,B,S,H,N] tangent fields are
    never built; the kernel generates them on-chip."""
    S, B, H = pf["S"], pf["B"], pf["H"]
    r = tb[0].shape[0]
    bf16 = torch.bfloat16

    def lay_fwd(f):  # [S, BH, X] -> [B, Sp, H, X] bf16
        X = f.shape[-1]
        t = f.reshape(S, B, H, X).permute(1, 0, 2, 3)
        if Sp > S:
            t = F.pad(t, (0, 0, 0, 0, 0, Sp - S))
        return t.to(bf16).contiguous()

    def lay_bhs(f):  # [S, BH] -> [B, H, Sp] fp32
        t = f.reshape(S, B, H).permute(1, 2, 0)
        if Sp > S:
            t = F.pad(t, (0, Sp - S))
        return t.float().contiguous()

    def lay_bhsd(f):  # [S, BH, Da] -> [B, H, Sp, Da] bf16
        X = f.shape[-1]
        t = f.reshape(S, B, H, X).permute(1, 2, 0, 3)
        if Sp > S:
            t = F.pad(t, (0, 0, 0, Sp - S))
        return t.to(bf16).contiguous()

    if skip_qkcs:
        # QR/KR/COS/SIN come from the forward-captured intermediates,
        # assembled natively at the call site (_native_qkcs).
        QR = KR = COS = SIN = None
        Vv = lay_fwd(pf["Vf"])
    else:
        QR = lay_fwd(pf["Q_r"]); KR = lay_fwd(pf["K_r"]); Vv = lay_fwd(pf["Vf"])
        COS = lay_bhsd(pf["cos_t"]); SIN = lay_bhsd(pf["sin_t"])
    SCALE = lay_bhs(pf["scale"])
    ADTf_p = F.pad(pf["ADTf"], (0, 0, 0, Sp - S)) if Sp > S else pf["ADTf"]
    L_bhs = torch.cumsum(ADTf_p.reshape(Sp // cs, cs, B * H), dim=1)
    L_bhs = L_bhs.reshape(Sp, B, H).permute(1, 2, 0).contiguous()

    # raw group-level tangents [r,B,S,G,N] -> pad + cast only; skip_raw
    # callers consume tb[0..2] unpadded and skip the materialization.
    def lay_raw(f):
        t = f
        if Sp > S:
            t = F.pad(t, (0, 0, 0, 0, 0, Sp - S))
        return t.to(bf16).contiguous()

    if skip_raw:
        DQRAW = DKRAW = DV = None
    else:
        DQRAW = lay_raw(tb[0]); DKRAW = lay_raw(tb[1])
        DV = lay_raw(tb[2])                              # [r,B,Sp,H,P]

    # per-lane scalar/small fields (tb[3..5] are [r,B,H,S] per the preprocess).
    dDTf = tb[4].permute(0, 3, 1, 2).reshape(r, S, B * H)
    dTrapf = tb[5].permute(0, 3, 1, 2).reshape(r, S, B * H)
    dADTf = tb[3].permute(0, 3, 1, 2).reshape(r, S, B * H)
    trap_sig, DTf = pf["trap_sig"], pf["DTf"]
    DT_sh, trap_sh = pf["DT_sh"], pf["trap_sh"]
    dtrap_sig = trap_sig * (1 - trap_sig) * dTrapf
    _shr = (0, 0, 0, 1)
    dDT_sh = F.pad(dDTf[:, 1:], _shr)
    dtrap_sh = F.pad(dtrap_sig[:, 1:], _shr)
    dshifted_gamma = dDT_sh * (1 - trap_sh) + DT_sh * (-dtrap_sh)
    dscale = dDTf * trap_sig + DTf * dtrap_sig + dshifted_gamma

    def lay_lane_bhs(f):  # [r, S, BH] -> [r, B, H, Sp] fp32
        t = f.reshape(r, S, B, H).permute(0, 2, 3, 1)
        if Sp > S:
            t = F.pad(t, (0, Sp - S))
        return t.float().contiguous()

    DSCALE = lay_lane_bhs(dscale)
    dADT_p = F.pad(dADTf, (0, 0, 0, Sp - S)) if Sp > S else dADTf
    # per-chunk cumsum over the last dim via the permute trick; inductor's
    # dim!=last split-scan codegen fails at some shapes.
    DL = torch.cumsum(dADT_p.reshape(r, Sp // cs, cs, B * H).permute(0, 1, 3, 2), dim=-1)
    DL = DL.permute(0, 1, 3, 2).reshape(r, Sp, B, H).permute(0, 2, 3, 1).contiguous()

    # dtheta = cumsum_s(pi(1-tanh^2)dAng*DT + tanh*pi*dDT), computed as
    # per-chunk cumsum + chunk-boundary offsets (exact identity, fuses).
    tanh_ang = pf["tanh_ang"]                                   # [S, BH, Da]
    Da = tanh_ang.shape[-1]
    dAngf = tb[8].permute(0, 2, 1, 3, 4).reshape(r, S, B * H, -1)
    dang_scaled = (
        (math.pi * (1 - tanh_ang.pow(2)) * dAngf.float()) * DTf.unsqueeze(-1)
        + tanh_ang * math.pi * dDTf.unsqueeze(-1)
    )
    if Sp > S:
        dang_scaled = F.pad(dang_scaled, (0, 0, 0, 0, 0, Sp - S))
    nch = Sp // cs
    dang_t = (dang_scaled.reshape(r, nch, cs, B * H, Da)
              .permute(0, 1, 3, 4, 2).contiguous())       # [r,nc,BH,Da,cs]
    pc = torch.cumsum(dang_t, dim=-1)
    csum = pc[..., -1]                                    # chunk sums [r,nc,BH,Da]
    off = torch.cumsum(csum, dim=1) - csum                # exclusive over chunks
    DTHETA = ((pc + off.unsqueeze(-1))
              .view(r, nch, B, H, Da, cs)
              .permute(0, 2, 3, 1, 5, 4)                  # [r,B,H,nc,cs,Da]
              .to(bf16).contiguous()
              .view(r, B, H, Sp, Da))

    # dqkdot for the finalize (pre-rotary dot; group-level contraction, no expand).
    G = tb[0].shape[3]
    Qb_full = pf["Qb_full"].reshape(S, B, G, H // G, -1)         # [S,B,G,HG,N]
    Kb_full = pf["Kb_full"].reshape(S, B, G, H // G, -1)
    dQg = tb[0].to(torch.float32)                                # [r,B,S,G,N]
    dKg = tb[1].to(torch.float32)
    sgamma = pf["shifted_gamma"]                                 # [S, BH]
    term = (
        torch.einsum("rbsgn,sbghn->rsbgh", dKg, Qb_full.float())
        + torch.einsum("rbsgn,sbghn->rsbgh", dQg, Kb_full.float())
    ).reshape(r, S, B * H)
    kq = (pf["Kb_full"] * pf["Qb_full"]).sum(-1)                 # [S, BH]
    dqkdot = term * sgamma + kq * dshifted_gamma

    fields = dict(pf)
    if mean_mode or skip_fin:
        # mean path: dVf is unused and dZf is superseded by DZk built from
        # tb[10], so both [r,S,BH,P] round-trips are skipped.
        fields.update({"dVf": None, "dZf": None,
                       "dqkdot": dqkdot, "dADTf": dADTf, "r": r})
    else:
        fields.update({
            "dVf": tb[2].permute(0, 2, 1, 3, 4).reshape(r, S, B * H, -1),
            "dZf": tb[10].permute(0, 2, 1, 3, 4).reshape(r, S, B * H, -1) if tb[10] is not None else None,
            "dqkdot": dqkdot, "dADTf": dADTf, "r": r,
        })
    return fields, QR, KR, Vv, COS, SIN, SCALE, L_bhs, DQRAW, DKRAW, DV, DTHETA, DSCALE, DL


def _gen_mean_extra_layout(pf: dict, fields: dict, Sp: int, tb: "tuple | None" = None,
                           skip_dz: bool = False):
    """Extra kernel inputs for the mean-epilogue variant: the Z-gate pair,
    the QK-dot skip pair, and the D weights in kernel layout. When `tb` is
    given, DZk is built directly from tb[10] [r,B,S,H,P] in one pad+cast pass
    (no [r,S,BH,P] intermediate)."""
    S, B, H = pf["S"], pf["B"], pf["H"]
    r = fields["r"]
    bf16 = torch.bfloat16

    def lay_fwd(f):  # [S, BH, X] -> [B, Sp, H, X] bf16
        X = f.shape[-1]
        t = f.reshape(S, B, H, X).permute(1, 0, 2, 3)
        if Sp > S:
            t = F.pad(t, (0, 0, 0, 0, 0, Sp - S))
        return t.to(bf16).contiguous()

    def lay_lane(f):  # [r, S, BH, X] -> [r, B, Sp, H, X] bf16
        X = f.shape[-1]
        t = f.reshape(r, S, B, H, X).permute(0, 2, 1, 3, 4)
        if Sp > S:
            t = F.pad(t, (0, 0, 0, 0, 0, Sp - S))
        return t.to(bf16).contiguous()

    def lay_bhs(f):  # [S, BH] -> [B, H, Sp] fp32
        t = f.reshape(S, B, H).permute(1, 2, 0)
        if Sp > S:
            t = F.pad(t, (0, Sp - S))
        return t.float().contiguous()

    def lay_lane_bhs(f):  # [r, S, BH] -> [r, B, H, Sp] fp32
        t = f.reshape(r, S, B, H).permute(0, 2, 3, 1)
        if Sp > S:
            t = F.pad(t, (0, Sp - S))
        return t.float().contiguous()

    Zk = lay_fwd(pf["Zf"])
    if skip_dz:
        DZk = None            # native fold: kernel reads tb[10] fp32 directly
    elif tb is not None:
        dz = tb[10]                                       # [r, B, S, H, P]
        if Sp > S:
            dz = F.pad(dz, (0, 0, 0, 0, 0, Sp - S))
        DZk = dz.to(bf16).contiguous()
    else:
        DZk = lay_lane(fields["dZf"])
    QKD = lay_bhs(pf["qkdot"])
    DQKD = lay_lane_bhs(fields["dqkdot"])
    DSK = pf["Df"][:H].float().contiguous()          # Df is [B*H] = D repeated over B
    return Zk, DZk, QKD, DQKD, DSK


def _mean_postprocess(accs: torch.Tensor, w_out_t: torch.Tensor, s_true: int) -> torch.Tensor:
    """Pooled epilogue: chunk-slot partial sums [r, B, cs, H, P] -> sequence
    mean -> out_proj at the pooled level (the mean commutes with the linear
    out_proj) -> mean_L(d_mixer_out) [r, B, D]."""
    m = accs.sum(dim=2) / float(s_true)              # [r, B, H, P]
    r, B, H, P = m.shape
    return m.reshape(r, B, H * P).to(w_out_t.dtype) @ w_out_t


def mamba3_mixer_jvp_mean_kernel(
    mixer: Any,
    u: torch.Tensor,
    du_lanes: "list[torch.Tensor]",
    *,
    compute_dtype: torch.dtype = torch.float32,
    chunk_size: "int | None" = None,
    operand_dtype: str = "bfloat16",
    proj_cached: "torch.Tensor | None" = None,
    fwd_caps: "tuple | None" = None,
) -> torch.Tensor:
    """Mixer JVP returning only mean_L(d_mixer_out) [r, B, D], the interface
    meanpool input. The finalize and the sequence row-sum run in the gen
    kernel's epilogue, so the [r,B,S,H,P] DOUT never reaches HBM and the
    full-sequence out_proj collapses to one pooled matmul. Requires the bf16
    gen path and a mixer with D-skip and Z-gate; otherwise falls back to the
    full path + torch mean (the same map). `du_lanes` may be the stacked
    [r, B, L, D] tensor directly."""
    import os

    use_gen = (
        os.environ.get("LBI_FWDMODE_GEN", "1") == "1"
        and operand_dtype == "bfloat16"
    )
    if not use_gen:
        du_list = (list(du_lanes) if torch.is_tensor(du_lanes) else du_lanes)
        _, douts = mamba3_mixer_jvp_kernel(
            mixer, u, du_list, compute_dtype=compute_dtype,
            chunk_size=chunk_size, operand_dtype=operand_dtype)
        return douts.mean(dim=2)

    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_dualscan import (
        _get_gen_mean_kernel, _maybe_compile,
    )
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_dualscan_ref import _prepare_primal_fields

    if chunk_size is None:
        # gen_mean fits cs=64 at d_state<=128, halving the chunk chain.
        chunk_size = 64 if mixer.d_state <= 128 else 32
    cs = int(chunk_size)
    du_stack = (du_lanes.to(compute_dtype) if torch.is_tensor(du_lanes)
                else torch.stack([d.to(compute_dtype) for d in du_lanes], dim=0))
    r = du_stack.shape[0]
    if proj_cached is not None:
        scan_inputs, tb = _maybe_compile(
            _mixer_preprocess_cached_fwd_jvp, "mixer_pre_jvp_c")(
            mixer, du_stack, compute_dtype, proj_cached)
    else:
        scan_inputs, tb = _maybe_compile(_mixer_preprocess_fwd_jvp, "mixer_pre_jvp")(
            mixer, u, du_stack, compute_dtype)
    use_caps = (fwd_caps is not None
                and all(c is not None for c in fwd_caps))
    pf = _maybe_compile(_prepare_primal_fields, "primal")(
        scan_inputs, compute_dtype, use_caps)
    if pf["Zf"] is None or pf["Df"] is None:
        # Not the real-mixer configuration; take the full path + torch mean.
        _, douts = mamba3_mixer_jvp_kernel(
            mixer, u, list(du_stack), compute_dtype=compute_dtype,
            chunk_size=chunk_size, operand_dtype=operand_dtype)
        return douts.mean(dim=2)
    S, B, H = pf["S"], pf["B"], pf["H"]
    Sp = ((S + cs - 1) // cs) * cs
    (fields, QR, KR, Vv, COS, SIN, SCALE, L_bhs,
     DQRAW, DKRAW, DV, DTHETA, DSCALE, DL) = _maybe_compile(
        _gen_fields_layout, "gen_fields_mean")(
        pf, tb, Sp, cs, True, False, False, use_caps)
    if use_caps:
        # QR/KR/COS/SIN from the forward-captured intermediates.
        QR, KR, COS, SIN = _maybe_compile(_native_qkcs, "native_qkcs")(
            fwd_caps[0], fwd_caps[1], scan_inputs[1], scan_inputs[7], Sp)
    Zk, DZk, QKD, DQKD, DSK = _maybe_compile(
        _gen_mean_extra_layout, "gen_mean_extra")(pf, fields, Sp, tb, False)
    if os.environ.get("LBI_FWDMODE_FP8SIM"):
        # precision gate: simulate fp8 storage of the per-lane tangent streams.
        DQRAW, DKRAW, DV = (_fp8_sim(DQRAW, "dqraw"),
                            _fp8_sim(DKRAW, "dkraw"), _fp8_sim(DV, "dv"))
        DTHETA, DSCALE, DL = (_fp8_sim(DTHETA, "dtheta"),
                              _fp8_sim(DSCALE, "dscale"), _fp8_sim(DL, "dl"))
        DZk, DQKD = _fp8_sim(DZk, "dz"), _fp8_sim(DQKD, "dqkd")
    G = DQRAW.shape[3]
    N = QR.shape[-1]
    P = Vv.shape[-1]
    Da = COS.shape[-1]
    kern = _get_gen_mean_kernel(B, Sp, H, G, N, Da, P, cs, S, dtype="bfloat16")
    accs = [
        kern(QR, KR, Vv, DQRAW[l], DKRAW[l], DV[l], DTHETA[l], COS, SIN,
             SCALE, DSCALE[l], L_bhs, DL[l], Zk, DZk[l], QKD, DQKD[l], DSK)
        for l in range(r)
    ]
    acc_stack = torch.stack(accs, dim=0)                    # [r, B, cs, H, P]
    return _maybe_compile(_mean_postprocess, "mean_post")(
        acc_stack, mixer.out_proj.weight.t(), S)


def _mixer_jvp_fields_fused(mixer: Any, u: torch.Tensor, du: torch.Tensor,
                            cd: torch.dtype, cs: int, bf16_out: bool):
    """One torch.compile region from raw (u, du[r]) to the kernel-ready
    inputs (preprocess fwd+JVP -> primal fields -> tangent fields -> layout
    emission), fusing every intermediate."""
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_dualscan_ref import (
        _prepare_primal_fields, _tangent_fields_core,
    )

    scan_inputs, tb = _mixer_preprocess_fwd_jvp(mixer, u, du, cd)
    pf = _prepare_primal_fields(scan_inputs, cd)
    S, B, H = pf["S"], pf["B"], pf["H"]
    Sp = ((S + cs - 1) // cs) * cs
    tf = _tangent_fields_core(pf, tb[0], tb[1], tb[2], tb[3], tb[4], tb[5], tb[8], tb[10])
    r = tf["r"]
    out_dt = torch.bfloat16 if bf16_out else torch.float32

    def lay_lane(f):
        X = f.shape[-1]
        t = f.reshape(r, S, B, H, X).permute(0, 2, 1, 3, 4)
        if Sp > S:
            t = F.pad(t, (0, 0, 0, 0, 0, Sp - S))
        return t.to(out_dt).contiguous()

    def lay_fwd(f):
        X = f.shape[-1]
        t = f.reshape(S, B, H, X).permute(1, 0, 2, 3)
        if Sp > S:
            t = F.pad(t, (0, 0, 0, 0, 0, Sp - S))
        return t.to(out_dt).contiguous()

    dQr, dKr, dVr = lay_lane(tf["dQ_r"]), lay_lane(tf["dK_sc"]), lay_lane(tf["dVf"])
    QR, KSC, Vv = lay_fwd(pf["Q_r"]), lay_fwd(pf["K_sc"]), lay_fwd(pf["Vf"])
    ADTf_p = F.pad(pf["ADTf"], (0, 0, 0, Sp - S)) if Sp > S else pf["ADTf"]
    L_bhs = torch.cumsum(ADTf_p.reshape(Sp // cs, cs, B * H), dim=1)
    L_bhs = L_bhs.reshape(Sp, B, H).permute(1, 2, 0).contiguous()
    dADTf_p = F.pad(tf["dADTf"], (0, 0, 0, Sp - S)) if Sp > S else tf["dADTf"]
    dLr = torch.cumsum(dADTf_p.reshape(r, Sp // cs, cs, B * H), dim=2)
    dLr = dLr.reshape(r, Sp, B, H).permute(0, 2, 3, 1).contiguous()
    return tf, QR, KSC, Vv, L_bhs, dQr, dKr, dVr, dLr


def mamba3_mixer_jvp_kernel(
    mixer: Any,
    u: torch.Tensor,
    du_lanes: "list[torch.Tensor] | None",
    *,
    compute_dtype: torch.dtype = torch.float32,
    chunk_size: "int | None" = None,
    operand_dtype: str = "bfloat16",
    bcast_norm: Any = None,
    du_bcast: "torch.Tensor | None" = None,
    proj_cached: "torch.Tensor | None" = None,
    xn_cached: "torch.Tensor | None" = None,
    fwd_caps: "tuple | None" = None,
) -> "tuple[torch.Tensor, list[torch.Tensor]]":
    """Kernel-backed SISO mixer forward+JVP over r lanes: explicit preprocess
    and linear postprocess around the fused scan kernel, which returns the
    forward and all r JVP tangents together. Returns (mixer_out [B,L,D],
    d_mixer_out stacked [r,B,L,D]); `du_lanes` may be a list of [B,L,D] lanes
    or the stacked tensor.

    Broadcast fast path: pass `bcast_norm` (the block RMSNorm) and `du_bcast`
    [r, B, D] with `u` = the pre-norm block input and `du_lanes=None`; for
    L-broadcast tangents the norm-JVP + tangent projection collapse exactly
    (see `_mixer_preprocess_bcast_fwd_jvp`)."""
    import os
    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_dualscan import (
        mamba3_siso_dualscan_tilelang_rwide,
        _maybe_compile,
    )

    if chunk_size is None:
        # gen path is bf16-only; gen kernels fit cs=64 at d_state<=128
        # (fp32 fallback below stays at cs=32).
        gen_bf16 = os.environ.get("LBI_FWDMODE_GEN", "1") == "1" and operand_dtype == "bfloat16"
        chunk_size = 64 if (gen_bf16 and mixer.d_state <= 128) else (32 if mixer.d_state > 64 else 64)
        if gen_bf16 and _gwalk_available():
            # gwalk consumes cs=32-local L/DL cumsums and wins from lane
            # batching only at r >= 4 (r1 regresses, r2 is a wash).
            _r_est = (du_bcast.shape[0] if du_bcast is not None
                      else (du_lanes.shape[0] if torch.is_tensor(du_lanes)
                            else len(du_lanes)))
            if _r_est >= 4:
                chunk_size = 32

    # explicit batched preprocess forward+JVP, compiled as three separate
    # units (a single fused region is slower).
    if du_bcast is not None:
        # broadcast fast path: fused norm-JVP + L-collapsed projection
        # (exact rank-2-in-L); the [B,r,L,D] basis is never materialized.
        r = du_bcast.shape[0]
        if proj_cached is not None and xn_cached is not None:
            _, (scan_inputs, tb) = _maybe_compile(
                _mixer_preprocess_bcast_cached_fwd_jvp, "mixer_pre_bcast_c")(
                mixer, bcast_norm, u, du_bcast, compute_dtype,
                proj_cached, xn_cached)
        else:
            _, (scan_inputs, tb) = _maybe_compile(
                _mixer_preprocess_bcast_fwd_jvp, "mixer_pre_bcast")(
                mixer, bcast_norm, u, du_bcast, compute_dtype)
    else:
        # accept the stacked [r, B, L, D] tensor directly (skips the
        # list -> stack copy at every inter-block boundary).
        du_stack = (du_lanes.to(compute_dtype) if torch.is_tensor(du_lanes)
                    else torch.stack([d.to(compute_dtype) for d in du_lanes], dim=0))
        r = du_stack.shape[0]
        if proj_cached is not None:
            scan_inputs, tb = _maybe_compile(
                _mixer_preprocess_cached_fwd_jvp, "mixer_pre_jvp_c")(
                mixer, du_stack, compute_dtype, proj_cached)
        else:
            scan_inputs, tb = _maybe_compile(_mixer_preprocess_fwd_jvp, "mixer_pre_jvp")(
                mixer, u, du_stack, compute_dtype)

    use_gen = os.environ.get("LBI_FWDMODE_GEN", "1") == "1" and operand_dtype == "bfloat16"
    if use_gen:
        # on-chip tangent generation: the kernel builds dQ_r/dK_sc from raw
        # group-level tangents; [r,B,S,H,N] fields never materialize.
        from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_dualscan import _get_gen_kernel
        from backbones.mamba3.ops.triton.mamba3.mamba3_siso_dualscan_ref import _prepare_primal_fields

        cs = int(chunk_size)
        use_caps = (fwd_caps is not None
                    and all(c is not None for c in fwd_caps))
        pf = _maybe_compile(_prepare_primal_fields, "primal")(
            scan_inputs, compute_dtype, use_caps)
        S, B, H = pf["S"], pf["B"], pf["H"]
        Sp = ((S + cs - 1) // cs) * cs
        (fields, QR, KR, Vv, COS, SIN, SCALE, L_bhs,
         DQRAW, DKRAW, DV, DTHETA, DSCALE, DL) = _maybe_compile(
            _gen_fields_layout, "gen_fields")(pf, tb, Sp, cs, False, False, True,
                                              use_caps)
        if use_caps:
            # assemble QR/KR/COS/SIN from the forward-captured
            # intermediates in kernel-native layout.
            QR, KR, COS, SIN = _maybe_compile(_native_qkcs, "native_qkcs")(
                fwd_caps[0], fwd_caps[1], scan_inputs[1], scan_inputs[7], Sp)
        fin_dv, fin_dz = tb[2], tb[10]
        if os.environ.get("LBI_FWDMODE_FP8SIM"):
            # precision gate: simulate fp8 storage of the per-lane tangent streams.
            DQRAW, DKRAW, DV = (_fp8_sim(DQRAW, "dqraw"),
                                _fp8_sim(DKRAW, "dkraw"), _fp8_sim(DV, "dv"))
            DTHETA, DSCALE, DL = (_fp8_sim(DTHETA, "dtheta"),
                                  _fp8_sim(DSCALE, "dscale"), _fp8_sim(DL, "dl"))
            fin_dv, fin_dz = _fp8_sim(fin_dv, "dv"), _fp8_sim(fin_dz, "dz")
            fields = {**fields, "dqkdot": _fp8_sim(fields["dqkdot"], "dqkdot")}
        G = DQRAW.shape[3]
        N = QR.shape[-1]
        P = Vv.shape[-1]
        Da = COS.shape[-1]
        if cs == 32 and r >= 4 and _gwalk_mod(N, P, Da, Sp) is not None:
            # Lane-batched wgmma dual walk: one launch for all r lanes.
            # cs==32 guards the L/DL chunk-locality contract of the walk.
            GW = _gwalk_mod(N, P, Da, Sp)
            OUT, DOUT = GW.gwalk_full(
                QR.contiguous(), KR.contiguous(), Vv.contiguous(),
                DQRAW.contiguous(), DKRAW.contiguous(), DV.contiguous(),
                DTHETA.contiguous(), COS.contiguous(), SIN.contiguous(),
                SCALE.contiguous(), DSCALE.contiguous(),
                L_bhs.contiguous(), DL.contiguous())
            OUTn = OUT[:, :S]                                # [B,S,H,P]
            DOUTn = DOUT[:, :, :S]                           # [r,B,S,H,P]
        else:
            kern = _get_gen_kernel(B, Sp, H, G, N, Da, P, cs, dtype="bfloat16")
            OUTn = None
            dout_slices = []
            for l in range(r):
                OUT, DOUT = kern(QR, KR, Vv, DQRAW[l], DKRAW[l], DV[l], DTHETA[l],
                                 COS, SIN, SCALE, DSCALE[l], L_bhs, DL[l])
                if OUTn is None:
                    OUTn = OUT[:, :S]
                dout_slices.append(DOUT[:, :S])
            # one native stack [r,B,S,H,P] (contiguous cat when Sp == S)
            # replaces per-lane permute-reshape clones.
            DOUTn = torch.stack(dout_slices, dim=0)
        fwd_out, douts = _maybe_compile(_finalize_postprocess_native, "finalize_nat")(
            OUTn, DOUTn, fin_dv, fin_dz, fields, mixer.out_proj.weight.t())
        return fwd_out, douts

    tangent_lanes = [
        (tb[0][l], tb[1][l], tb[2][l], tb[3][l], tb[4][l], tb[5][l],
         None, None, tb[8][l], None, tb[10][l])
        for l in range(r)
    ]
    # scan via the fused kernel; pre-finalize quads returned so the finalize and
    # the out_proj postprocess fuse into ONE compiled epilogue (lever c).
    out_quad, dout_quad, tf = mamba3_siso_dualscan_tilelang_rwide(
        scan_inputs, tangent_lanes, chunk_size=chunk_size,
        compute_dtype=compute_dtype, operand_dtype=operand_dtype, return_quad=True)
    fwd_out, douts = _maybe_compile(_finalize_postprocess_fused, "finalize_post")(
        out_quad, dout_quad, tf, mixer.out_proj.weight.t())
    return fwd_out, douts


def mamba3_region_forward_ref(
    backend: Any,
    region_index: int,
    region_input: torch.Tensor,
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Differentiable pure-torch reproduction of `_run_region` (pre-norm residual
    stack of SISO blocks). Returns the region output [B, L, D]."""
    start, end = backend._region_range(region_index)
    hidden = region_input
    residual: torch.Tensor | None = None
    for layer_index in range(start, end):
        block = backend.backbone.blocks[layer_index]
        residual_out = (hidden + residual) if residual is not None else hidden
        norm_input = _rmsnorm_ref(residual_out, block.norm)
        hidden = mamba3_mixer_forward_ref(block.mixer, norm_input, compute_dtype=compute_dtype)
        residual = residual_out
    return (hidden + residual) if residual is not None else hidden


def mamba3_region_output_jvp(
    backend: Any,
    *,
    cache: Any,
    region_input_tangent_basis: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Forward-mode region JVP: region-input tangent basis [B, P, L, D] ->
    region-output tangent basis [B, P, L, D], at the frozen operating point
    (`cache.region_input`). r independent JVPs sharing one forward
    linearization, exactly the r-wide forward scan the native kernel batches."""
    if region_input_tangent_basis.dim() != 4:
        raise ValueError("region_input_tangent_basis must have shape [B, P, L, D].")
    region_index = cache.region_index
    u0 = cache.region_input.detach().to(compute_dtype)

    def f(u: torch.Tensor) -> torch.Tensor:
        return mamba3_region_forward_ref(backend, region_index, u, compute_dtype=compute_dtype)

    # r JVPs share one forward linearization via vmap over the tangent
    # batch; the region forward reference is pure torch (vmap-safe).
    tangents = region_input_tangent_basis.to(compute_dtype).transpose(0, 1)  # [P, B, L, D]

    def push(dv: torch.Tensor) -> torch.Tensor:
        return torch.func.jvp(f, (u0,), (dv,))[1]

    douts = torch.vmap(push)(tangents)  # [P, B, L, D]
    return douts.transpose(0, 1).contiguous().to(region_input_tangent_basis.dtype)


def mamba3_region_output_jvp_kernel(
    backend: Any,
    *,
    cache: Any,
    region_input_tangent_basis: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
    chunk_size: "int | None" = None,
    operand_dtype: str = "bfloat16",
    pooled: bool = False,
) -> torch.Tensor:
    """Kernel-backed region JVP: same map as `mamba3_region_output_jvp` with
    the mixer scan JVP running through the fused dual-scan kernel; the
    pre-norm residual stack is threaded in forward mode. Returns the
    region-output tangent basis [B, r, L, D].

    `pooled`: the region tangent's only consumer in the A_k path is the
    interface meanpool, so the last block runs the mean-epilogue kernel and
    neither the [r,B,S,H,P] DOUT nor the full [B,r,L,D] region tangent exist
    for that block. Returns [B, r, 1, D] (mean over L, keepdim)."""
    if region_input_tangent_basis.dim() != 4:
        raise ValueError("region_input_tangent_basis must have shape [B, r, L, D].")
    import os
    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_dualscan import _maybe_compile

    start, end = backend._region_range(cache.region_index)
    r = region_input_tangent_basis.shape[1]
    hidden = cache.region_input.detach().to(compute_dtype)
    # the A_k decode basis is L-constant (stride 0 or L=1): keep it
    # collapsed [B,r,1,D] so the first block takes the broadcast fast path.
    basis = region_input_tangent_basis
    is_bcast = (
        os.environ.get("LBI_FWDMODE_BCAST", "1") == "1"
        and (basis.shape[2] == 1 or basis.stride(2) == 0)
    )
    # loop runs lane-major [r,B,L,D] (the kernels' native layout); one
    # transpose at the return, none per block.
    d_hidden = (basis[:, :, :1] if is_bcast else basis).to(compute_dtype).transpose(0, 1)
    residual = None
    d_residual = None
    # per-block forward caches carry the primal in_proj + normed input;
    # the preprocess reads them instead of recomputing.
    lcs = getattr(cache, "layer_caches", None)
    use_cachepre = (
        os.environ.get("LBI_FWDMODE_CACHEPRE", "1") == "1"
        and lcs is not None
    )
    def _mc(layer_index):
        if not use_cachepre:
            return None
        for lc in lcs:
            if getattr(lc, "layer_index", None) == layer_index:
                return lc.block.mixer_cache
        return None
    use_capfields = os.environ.get("LBI_FWDMODE_CAPFIELDS", "1") == "1"
    def _caps(mc):
        # the forward kernel's captured intermediates (q_rot + absolute
        # angle cumsum) for the native QR/KR/COS/SIN assembly.
        if not use_capfields or mc is None:
            return None
        if getattr(mc, "q_rot", None) is None or getattr(mc, "theta_cs", None) is None:
            return None
        return (mc.q_rot, mc.theta_cs)
    for layer_index in range(start, end):
        block = backend.backbone.blocks[layer_index]
        res_out = (hidden + residual) if residual is not None else hidden
        d_res_out = d_hidden if d_residual is None else (d_hidden + d_residual)
        first_bcast = is_bcast and layer_index == start
        if first_bcast:
            mc = _mc(layer_index)
            # fused norm-JVP + L-collapsed tangent projection (exact).
            hidden, d_mix = mamba3_mixer_jvp_kernel(
                block.mixer, res_out, None,
                compute_dtype=compute_dtype, chunk_size=chunk_size,
                operand_dtype=operand_dtype, bcast_norm=block.norm,
                du_bcast=d_res_out[:, :, 0].contiguous(),
                proj_cached=None if mc is None else mc.in_proj,
                xn_cached=None if mc is None else mc.input_u,
                fwd_caps=_caps(mc))
        else:
            # analytic fused RMSNorm forward + batched JVP (compiled).
            norm_in, d_norm_stack = _maybe_compile(_rmsnorm_fwd_jvp, "rmsnorm_jvp")(
                res_out, d_res_out, block.norm)                     # [B,L,D], [r,B,L,D]
            if pooled and layer_index == end - 1:
                # last block: mean-epilogue kernel; the mean distributes over
                # d_mix + d_res_out, and the last primal mixer is not needed.
                mc = _mc(layer_index)
                d_mix_mean = mamba3_mixer_jvp_mean_kernel(
                    block.mixer, norm_in, d_norm_stack,
                    compute_dtype=compute_dtype, chunk_size=chunk_size,
                    operand_dtype=operand_dtype,
                    proj_cached=None if mc is None else mc.in_proj,
                    fwd_caps=_caps(mc))                              # [r, B, D]
                out_pool = d_res_out.mean(dim=2) + d_mix_mean        # [r, B, D]
                return (out_pool.transpose(0, 1).unsqueeze(2).contiguous()
                        .to(region_input_tangent_basis.dtype))
            # The kernel returns the forward mixer output too, so `hidden`
            # needs no dense O(L^2) recompute.
            mc2 = _mc(layer_index)
            hidden, d_mix = mamba3_mixer_jvp_kernel(
                block.mixer, norm_in, d_norm_stack,
                compute_dtype=compute_dtype, chunk_size=chunk_size,
                operand_dtype=operand_dtype,
                proj_cached=None if mc2 is None else mc2.in_proj,
                fwd_caps=_caps(mc2))
        d_hidden = d_mix                                             # [r, B, L, D]
        residual = res_out
        d_residual = d_res_out
    out = (d_hidden + d_residual) if d_residual is not None else d_hidden
    if pooled:
        # Reached when the last block took the bcast fast path (e.g.
        # single-block regions); pool here instead of the mean kernel.
        out = out.mean(dim=2, keepdim=True)
    return out.transpose(0, 1).contiguous().to(region_input_tangent_basis.dtype)
