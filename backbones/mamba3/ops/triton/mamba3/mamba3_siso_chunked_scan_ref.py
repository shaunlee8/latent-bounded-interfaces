"""Recurrent reference for the Mamba-3 SISO forward-mode JVP. The (S, dS)
recurrence is written out in state-space form, carrying the forward state S
and the tangent state dS through the same decay and rotary, so the fused
tilelang and CUDA kernels can be checked against it; it matches
`mamba3_siso_jvp` (torch.func on the quadratic reference). Batched, zero
initial state, input conventions as in `mamba3_siso_out_ref`.

Recurrence (per batch and head; a[t] = exp(ADT[t]), S_t = [N_qk, Dv]):
    S_t  = a[t] S_{t-1} + K_sc[t] (x) V[t]
    dS_t = a[t] dS_{t-1} + (a[t] dADT[t]) S_{t-1}
                        + dK_sc[t] (x) V[t] + K_sc[t] (x) dV[t]
    out_quad[t]  = Q_r[t] . S_t ;  d(out_quad)[t] = dQ_r[t] . S_t + Q_r[t] . dS_t
with the local skip (D, QK-dot) and Z-gate tangents added per position."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import repeat

TWO_PI = 2.0 * math.pi


def _rotary_fwd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair rotary (forward only)."""
    xr = x.view(*x.shape[:-1], -1, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    npair = x0.shape[-1]
    if cos.shape[-1] < npair:
        pad = npair - cos.shape[-1]
        cos = F.pad(cos, (0, pad), value=1.0)
        sin = F.pad(sin, (0, pad), value=0.0)
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos
    return torch.stack([r0, r1], dim=-1).view_as(x)


def _rotary_jvp(x_r: torch.Tensor, dx: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, dtheta: torch.Tensor) -> torch.Tensor:
    """JVP of the interleaved-pair rotary given the already-rotated primal x_r:
    dx_r = rotary(dx, theta) + dtheta (x) rot90(x_r) (deterministic tail padding
    matches `_rotary_fwd`)."""
    rr = x_r.view(*x_r.shape[:-1], -1, 2)
    r0, r1 = rr[..., 0], rr[..., 1]
    dxr = dx.view(*dx.shape[:-1], -1, 2)
    dx0, dx1 = dxr[..., 0], dxr[..., 1]
    npair = r0.shape[-1]
    if cos.shape[-1] < npair:
        pad = npair - cos.shape[-1]
        cos = F.pad(cos, (0, pad), value=1.0)
        sin = F.pad(sin, (0, pad), value=0.0)
    if dtheta.shape[-1] < npair:
        dtheta = F.pad(dtheta, (0, npair - dtheta.shape[-1]), value=0.0)
    dr0 = (dx0 * cos - dx1 * sin) + dtheta * (-r1)
    dr1 = (dx0 * sin + dx1 * cos) + dtheta * (r0)
    return torch.stack([dr0, dr1], dim=-1).view_as(dx)


def _prepare_primal_fields(primals: tuple[torch.Tensor, ...], compute_dtype: torch.dtype,
                           skip_qkcs: bool = False) -> dict:
    """FORWARD-HALF field prep -- depends only on `primals`, so it is computed
    ONCE and reused across every tangent direction (stage 3: the forward half is
    free). Returns the primal scan fields (Q_r, K_sc, Vf, ADTf, qkdot, gate
    inputs) plus the intermediates the tangent half needs (cos/sin, K_r, scale,
    the discretization fields), all [S, BH, ...]."""
    Q, K, V, ADT, DT, Trap, Qb, Kb, Ang, D, Z = primals
    cd = compute_dtype
    Q, K, V = Q.to(cd), K.to(cd), V.to(cd)
    ADT, DT = ADT.to(torch.float32), DT.to(torch.float32)
    Trap, Qb, Kb, Ang = Trap.to(cd), Qb.to(cd), Kb.to(cd), Ang.to(cd)
    D = D.to(cd) if D is not None else None
    Z = Z.to(cd) if Z is not None else None

    B, S, nqk, Dqk = Q.shape
    _, _, H, Dv = V.shape
    if nqk != H:
        g = H // nqk
        Q = repeat(Q, "b s h d -> b s (h g) d", g=g)
        K = repeat(K, "b s h d -> b s (h g) d", g=g)

    def bh(t):
        return t.permute(1, 0, 2, 3).reshape(S, B * H, -1)

    Qf, Kf, Vf = bh(Q), bh(K), bh(V)
    Angf = bh(Ang)
    ADTf = ADT.permute(2, 0, 1).reshape(S, B * H)
    DTf = DT.permute(2, 0, 1).reshape(S, B * H)
    Trapf = Trap.permute(2, 0, 1).reshape(S, B * H)
    Qbf = Qb.repeat(B, 1).view(B, H, Dqk).reshape(B * H, Dqk)
    Kbf = Kb.repeat(B, 1).view(B, H, Dqk).reshape(B * H, Dqk)
    Df = D.repeat(B).view(B, H).reshape(B * H) if D is not None else None
    Zf = bh(Z) if Z is not None else None

    trap_sig = torch.sigmoid(Trapf)
    _sh = (0, 0, 0, 1)
    DT_sh = F.pad(DTf[1:], _sh)
    trap_sh = F.pad(trap_sig[1:], _sh)
    shifted_gamma = DT_sh * (1 - trap_sh)
    scale = DTf * trap_sig + DT_sh * (1 - trap_sh)

    tanh_ang = torch.tanh(Angf).float()
    if skip_qkcs:
        # theta/cos/sin and rotated Q/K come from forward-captured
        # intermediates; only tanh_ang survives for the tangent dtheta chain.
        Qb_full = Qf + Qbf.unsqueeze(0)
        Kb_full = Kf + Kbf.unsqueeze(0)
        qkdot_s = (Kb_full * Qb_full).sum(-1) * shifted_gamma
        return {
            "Q_r": None, "K_sc": None, "K_r": None,
            "cos_t": None, "sin_t": None,
            "Vf": Vf, "ADTf": ADTf, "qkdot": qkdot_s, "Df": Df, "Zf": Zf,
            "scale": scale, "DTf": DTf, "trap_sig": trap_sig,
            "DT_sh": DT_sh, "trap_sh": trap_sh,
            "shifted_gamma": shifted_gamma, "tanh_ang": tanh_ang,
            "Qb_full": Qb_full, "Kb_full": Kb_full,
            "B": B, "S": S, "H": H, "nqk": nqk, "Dqk": Dqk, "Dv": Dv,
            "cd": cd, "device": Q.device,
        }
    ang_scaled = tanh_ang * math.pi * DTf.unsqueeze(-1)
    # Full-sequence cumsum computed as per-block cumsum + boundary offsets
    # (exact identity; contiguous rows scan in-block, the offset add fuses).
    CB = 64
    Da_ = ang_scaled.shape[-1]
    Sp2 = ((S + CB - 1) // CB) * CB
    a_p = ang_scaled if Sp2 == S else F.pad(ang_scaled, (0, 0, 0, 0, 0, Sp2 - S))
    a_t = (a_p.reshape(Sp2 // CB, CB, B * H, Da_)
           .permute(0, 2, 3, 1).contiguous())               # [nb,BH,Da,CB]
    pc = torch.cumsum(a_t, dim=-1)
    off = torch.cumsum(pc[..., -1], dim=0) - pc[..., -1]    # exclusive blocks
    theta = ((pc + off.unsqueeze(-1))
             .permute(0, 3, 1, 2).reshape(Sp2, B * H, Da_))
    if Sp2 != S:
        theta = theta[:S]
    theta = theta - TWO_PI * torch.floor(theta / TWO_PI)
    cos_t, sin_t = torch.cos(theta).to(cd), torch.sin(theta).to(cd)

    Qb_full = Qf + Qbf.unsqueeze(0)
    Kb_full = Kf + Kbf.unsqueeze(0)
    qkdot = (Kb_full * Qb_full).sum(-1) * shifted_gamma

    Q_r = _rotary_fwd(Qb_full, cos_t, sin_t)
    K_r = _rotary_fwd(Kb_full, cos_t, sin_t)
    K_sc = K_r * scale.unsqueeze(-1)

    return {
        # primal scan fields
        "Q_r": Q_r, "K_sc": K_sc, "Vf": Vf, "ADTf": ADTf, "qkdot": qkdot,
        "Df": Df, "Zf": Zf,
        # intermediates for the tangent half
        "cos_t": cos_t, "sin_t": sin_t, "K_r": K_r, "scale": scale,
        "DTf": DTf, "trap_sig": trap_sig, "DT_sh": DT_sh, "trap_sh": trap_sh,
        "shifted_gamma": shifted_gamma, "tanh_ang": tanh_ang,
        "Qb_full": Qb_full, "Kb_full": Kb_full,
        # shapes / meta
        "B": B, "S": S, "H": H, "nqk": nqk, "Dqk": Dqk, "Dv": Dv, "cd": cd, "device": Q.device,
    }


def _prepare_tangent_fields(pf: dict, tangents: tuple[torch.Tensor, ...]) -> dict:
    """TANGENT-HALF field prep -- the per-direction new work, given the shared primal
    cache `pf`. Biases/D carry zero tangent (parameters). Returns the tangent
    scan fields merged onto a copy of `pf` (a full field dict for the scans)."""
    dQ, dK, dV, dADT, dDT, dTrap = tangents[0], tangents[1], tangents[2], tangents[3], tangents[4], tangents[5]
    dAng, dZ = tangents[8], tangents[10]
    cd = pf["cd"]; B, S, H, nqk, Dqk = pf["B"], pf["S"], pf["H"], pf["nqk"], pf["Dqk"]
    dQ, dK, dV = dQ.to(cd), dK.to(cd), dV.to(cd)
    dADT, dDT = dADT.to(torch.float32), dDT.to(torch.float32)
    dTrap, dAng = dTrap.to(cd), dAng.to(cd)
    dZ = dZ.to(cd) if dZ is not None else None
    if nqk != H:
        g = H // nqk
        dQ = repeat(dQ, "b s h d -> b s (h g) d", g=g)
        dK = repeat(dK, "b s h d -> b s (h g) d", g=g)

    def bh(t):
        return t.permute(1, 0, 2, 3).reshape(S, B * H, -1)

    dQf, dKf, dVf = bh(dQ), bh(dK), bh(dV)
    dAngf = bh(dAng)
    dADTf = dADT.permute(2, 0, 1).reshape(S, B * H)
    dDTf = dDT.permute(2, 0, 1).reshape(S, B * H)
    dTrapf = dTrap.permute(2, 0, 1).reshape(S, B * H)
    dZf = bh(dZ) if dZ is not None else None

    trap_sig = pf["trap_sig"]; DTf = pf["DTf"]; DT_sh = pf["DT_sh"]; trap_sh = pf["trap_sh"]
    shifted_gamma = pf["shifted_gamma"]; tanh_ang = pf["tanh_ang"]
    cos_t, sin_t = pf["cos_t"], pf["sin_t"]; Q_r, K_r = pf["Q_r"], pf["K_r"]; scale = pf["scale"]
    Qb_full, Kb_full = pf["Qb_full"], pf["Kb_full"]

    dtrap_sig = trap_sig * (1 - trap_sig) * dTrapf
    _sh = (0, 0, 0, 1)
    dDT_sh = F.pad(dDTf[1:], _sh)
    dtrap_sh = F.pad(dtrap_sig[1:], _sh)
    dshifted_gamma = dDT_sh * (1 - trap_sh) + DT_sh * (-dtrap_sh)
    dscale = dDTf * trap_sig + DTf * dtrap_sig + dshifted_gamma

    dang_scaled = (
        (math.pi * (1 - tanh_ang.pow(2)) * dAngf.float()) * DTf.unsqueeze(-1)
        + tanh_ang * math.pi * dDTf.unsqueeze(-1)
    )
    dtheta = torch.cumsum(dang_scaled, dim=0)

    dqkdot = (
        (dKf * Qb_full + Kb_full * dQf).sum(-1) * shifted_gamma
        + (Kb_full * Qb_full).sum(-1) * dshifted_gamma
    )
    dQ_r = _rotary_jvp(Q_r, dQf, cos_t, sin_t, dtheta.to(cd))
    dK_r = _rotary_jvp(K_r, dKf, cos_t, sin_t, dtheta.to(cd))
    dK_sc = dK_r * scale.unsqueeze(-1) + K_r * dscale.unsqueeze(-1)

    out = dict(pf)
    out.update({"dQ_r": dQ_r, "dK_sc": dK_sc, "dVf": dVf, "dADTf": dADTf, "dqkdot": dqkdot, "dZf": dZf})
    return out


def _prepare_tangent_fields_batched(pf: dict, directions: "list[tuple[torch.Tensor, ...]]") -> dict:
    """Lane-batched tangent-half prep: compute the tangent fields for all
    r directions in ONE set of broadcast torch ops (r as a leading batch dim) instead
    of a python loop -- the per-direction tangent prep is the r-wide A_k bottleneck.
    Returns the tangent fields shaped [r, S, BH, ...] (numerically identical to
    looping `_prepare_tangent_fields`)."""
    r = len(directions)

    def stack_idx(i):
        return torch.stack([directions[l][i] for l in range(r)], dim=0)  # [r, B, ...]

    dZ = stack_idx(10) if directions[0][10] is not None else None
    return _tangent_fields_core(
        pf, stack_idx(0), stack_idx(1), stack_idx(2), stack_idx(3), stack_idx(4),
        stack_idx(5), stack_idx(8), dZ,
    )


def _tangent_fields_core(pf: dict, dQ, dK, dV, dADT, dDT, dTrap, dAng, dZ) -> dict:
    """The tangent-field math on ALREADY-BATCHED raw tangents [r, B, ...] (the
    post-stack body of `_prepare_tangent_fields_batched`; also called directly by
    the fused producer, skipping the stack roundtrip)."""
    cd = pf["cd"]; B, S, H, nqk = pf["B"], pf["S"], pf["H"], pf["nqk"]
    r = dQ.shape[0]
    dQ, dK, dV = dQ.to(cd), dK.to(cd), dV.to(cd)
    dADT = dADT.to(torch.float32); dDT = dDT.to(torch.float32)
    dTrap = dTrap.to(cd); dAng = dAng.to(cd)
    dZ = dZ.to(cd) if dZ is not None else None
    if nqk != H:
        g = H // nqk
        dQ = repeat(dQ, "r b s h d -> r b s (h g) d", g=g)
        dK = repeat(dK, "r b s h d -> r b s (h g) d", g=g)

    def bh_r(t):  # [r, B, S, H, X] -> [r, S, BH, X]
        return t.permute(0, 2, 1, 3, 4).reshape(r, S, B * H, -1)

    dQf, dKf, dVf = bh_r(dQ), bh_r(dK), bh_r(dV)
    dAngf = bh_r(dAng)
    dADTf = dADT.permute(0, 3, 1, 2).reshape(r, S, B * H)   # [r,B,H,S] -> [r,S,BH]
    dDTf = dDT.permute(0, 3, 1, 2).reshape(r, S, B * H)
    dTrapf = dTrap.permute(0, 3, 1, 2).reshape(r, S, B * H)
    dZf = bh_r(dZ) if dZ is not None else None

    trap_sig = pf["trap_sig"]; DTf = pf["DTf"]; DT_sh = pf["DT_sh"]; trap_sh = pf["trap_sh"]
    shifted_gamma = pf["shifted_gamma"]; tanh_ang = pf["tanh_ang"]
    cos_t, sin_t = pf["cos_t"], pf["sin_t"]; Q_r, K_r = pf["Q_r"], pf["K_r"]; scale = pf["scale"]
    Qb_full, Kb_full = pf["Qb_full"], pf["Kb_full"]

    dtrap_sig = trap_sig * (1 - trap_sig) * dTrapf
    _shr = (0, 0, 0, 1)   # shift along S (now dim 1), last row 0
    dDT_sh = F.pad(dDTf[:, 1:], _shr)
    dtrap_sh = F.pad(dtrap_sig[:, 1:], _shr)
    dshifted_gamma = dDT_sh * (1 - trap_sh) + DT_sh * (-dtrap_sh)
    dscale = dDTf * trap_sig + DTf * dtrap_sig + dshifted_gamma

    dang_scaled = (
        (math.pi * (1 - tanh_ang.pow(2)) * dAngf.float()) * DTf.unsqueeze(-1)
        + tanh_ang * math.pi * dDTf.unsqueeze(-1)
    )
    dtheta = torch.cumsum(dang_scaled, dim=1)   # cumsum over S (dim 1)

    dqkdot = (
        (dKf * Qb_full + Kb_full * dQf).sum(-1) * shifted_gamma
        + (Kb_full * Qb_full).sum(-1) * dshifted_gamma
    )
    # _rotary_jvp broadcasts the shared primal x_r/cos/sin against the r-batched dx.
    dQ_r = _rotary_jvp(Q_r, dQf, cos_t, sin_t, dtheta.to(cd))
    dK_r = _rotary_jvp(K_r, dKf, cos_t, sin_t, dtheta.to(cd))
    dK_sc = dK_r * scale.unsqueeze(-1) + K_r * dscale.unsqueeze(-1)

    out = dict(pf)
    out.update({"dQ_r": dQ_r, "dK_sc": dK_sc, "dVf": dVf, "dADTf": dADTf,
                "dqkdot": dqkdot, "dZf": dZf, "r": r})
    return out


def _finalize_batched(out_quad: torch.Tensor, dout_quad: torch.Tensor, f: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Lane-batched finalize: out_quad [S, BH, Dv] (direction-free), dout_quad
    [r, S, BH, Dv]. Applies the D/QK-dot/Z-gate tangents for all r directions at once.
    Returns out [B, S, H, Dv] and douts [r, B, S, H, Dv]."""
    B, S, H, Dv = f["B"], f["S"], f["H"], f["Dv"]
    r = f["r"]
    Vf, dVf, qkdot, dqkdot = f["Vf"], f["dVf"], f["qkdot"], f["dqkdot"]
    out = out_quad.to(f["cd"]).clone()
    dout = dout_quad.to(f["cd"]).clone()
    if f["Df"] is not None:
        out = out + f["Df"].view(1, -1, 1) * Vf
        dout = dout + f["Df"].view(1, 1, -1, 1) * dVf
    out = out - Vf * qkdot.unsqueeze(-1)
    dout = dout - (dVf * qkdot.unsqueeze(-1) + Vf.unsqueeze(0) * dqkdot.unsqueeze(-1))
    if f["Zf"] is not None:
        Zf, dZf = f["Zf"], f["dZf"]
        sig = torch.sigmoid(Zf)
        gate = Zf * sig
        dgate = (sig * (1 + Zf * (1 - sig))) * dZf
        dout = dout * gate.unsqueeze(0) + out.unsqueeze(0) * dgate
        out = out * gate

    def unbh(t):  # [S, BH, Dv] -> [B, S, H, Dv]
        return t.reshape(S, B, H, Dv).permute(1, 0, 2, 3).contiguous()

    def unbh_r(t):  # [r, S, BH, Dv] -> [r, B, S, H, Dv]
        return t.reshape(r, S, B, H, Dv).permute(0, 2, 1, 3, 4).contiguous()

    return unbh(out), unbh_r(dout)


def _prepare_dual_fields(
    primals: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    compute_dtype: torch.dtype,
) -> dict:
    """Shared field prep (primal forward-half + tangent-half), for the recurrent
    and chunked scans. Composes `_prepare_primal_fields` + `_prepare_tangent_
    fields`; identical numerics to the pre-split single pass."""
    pf = _prepare_primal_fields(primals, compute_dtype)
    return _prepare_tangent_fields(pf, tangents)


def _finalize(out_quad: torch.Tensor, dout_quad: torch.Tensor, f: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the local D-skip, QK-dot skip, and Z-gate tangents to the scan
    quadratic output, then unfold [S, BH, Dv] -> [B, S, H, Dv]."""
    B, S, H, Dv = f["B"], f["S"], f["H"], f["Dv"]
    Vf, dVf, qkdot, dqkdot = f["Vf"], f["dVf"], f["qkdot"], f["dqkdot"]
    out = out_quad.to(f["cd"]).clone()
    dout = dout_quad.to(f["cd"]).clone()
    if f["Df"] is not None:
        out = out + f["Df"].view(1, -1, 1) * Vf
        dout = dout + f["Df"].view(1, -1, 1) * dVf
    out = out - Vf * qkdot.unsqueeze(-1)
    dout = dout - (dVf * qkdot.unsqueeze(-1) + Vf * dqkdot.unsqueeze(-1))
    if f["Zf"] is not None:
        Zf, dZf = f["Zf"], f["dZf"]
        sig = torch.sigmoid(Zf)
        gate = Zf * sig
        dgate = (sig * (1 + Zf * (1 - sig))) * dZf
        dout = dout * gate + out * dgate
        out = out * gate

    def unbh(t):
        return t.reshape(S, B, H, Dv).permute(1, 0, 2, 3).contiguous()

    return unbh(out), unbh(dout)


def mamba3_siso_chunked_scan(
    primals: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(Out, dOut) via the explicit recurrent (S, dS) scan. primals/tangents ordered
    (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z)."""
    f = _prepare_dual_fields(primals, tangents, compute_dtype)
    Q_r, dQ_r, K_sc, dK_sc, Vf, dVf = f["Q_r"], f["dQ_r"], f["K_sc"], f["dK_sc"], f["Vf"], f["dVf"]
    a = torch.exp(f["ADTf"])
    da = a * f["dADTf"]
    S, BH, Dqk, Dv = f["S"], f["B"] * f["H"], f["Dqk"], f["Dv"]

    Sst = torch.zeros(BH, Dqk, Dv, dtype=torch.float32, device=f["device"])
    dSst = torch.zeros_like(Sst)
    outs, douts = [], []
    for t in range(S):
        at = a[t].view(-1, 1, 1)
        dat = da[t].view(-1, 1, 1)
        kv = K_sc[t].unsqueeze(-1) * Vf[t].unsqueeze(1)
        dkv = dK_sc[t].unsqueeze(-1) * Vf[t].unsqueeze(1) + K_sc[t].unsqueeze(-1) * dVf[t].unsqueeze(1)
        Sst_prev = Sst
        Sst = at * Sst_prev + kv.float()
        dSst = at * dSst + dat * Sst_prev + dkv.float()
        outs.append(torch.einsum("bn,bnd->bd", Q_r[t].float(), Sst))
        douts.append(
            torch.einsum("bn,bnd->bd", dQ_r[t].float(), Sst)
            + torch.einsum("bn,bnd->bd", Q_r[t].float(), dSst)
        )
    out_quad = torch.stack(outs, 0)
    dout_quad = torch.stack(douts, 0)
    return _finalize(out_quad, dout_quad, f)


def mamba3_siso_chunked_scan_ref_blocked(
    primals: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    *,
    chunk_size: int = 64,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(Out, dOut) via the chunked (S, dS) scan -- the exact tiling the fused
    tilelang kernel implements: chunk-local quadratic (intra) + inter-chunk
    (S, dS) state passing (inter). Numerically identical to the recurrent scan;
    this is the algorithm the kernel transcribes, in torch for debuggability."""
    f = _prepare_dual_fields(primals, tangents, compute_dtype)
    S, B, H, Dqk, Dv = f["S"], f["B"], f["H"], f["Dqk"], f["Dv"]
    BH = B * H
    cs = int(chunk_size)
    nc = (S + cs - 1) // cs
    Sp = nc * cs
    dev = f["device"]

    def pad_seq(t):  # [S, BH, X] -> [nc, cs, BH, X], zero-padded on the sequence
        if t.shape[0] < Sp:
            t = F.pad(t, (0, 0, 0, 0, 0, Sp - t.shape[0]))
        return t.reshape(nc, cs, BH, -1)

    Qr = pad_seq(f["Q_r"]).float(); dQr = pad_seq(f["dQ_r"]).float()
    Ksc = pad_seq(f["K_sc"]).float(); dKsc = pad_seq(f["dK_sc"]).float()
    Vf = pad_seq(f["Vf"]).float(); dVf = pad_seq(f["dVf"]).float()
    ADT = pad_seq(f["ADTf"].unsqueeze(-1)).squeeze(-1)   # [nc, cs, BH]
    dADT = pad_seq(f["dADTf"].unsqueeze(-1)).squeeze(-1)

    Lc = torch.cumsum(ADT, dim=1)              # inclusive within-chunk log-decay [nc,cs,BH]
    dLc = torch.cumsum(dADT, dim=1)
    tri = torch.tril(torch.ones(cs, cs, dtype=torch.bool, device=dev))  # j<=i

    # ---- intra (chunk-local quadratic) ----
    QK = torch.einsum("cibn,cjbn->cijb", Qr, Ksc)
    dQK = torch.einsum("cibn,cjbn->cijb", dQr, Ksc) + torch.einsum("cibn,cjbn->cijb", Qr, dKsc)
    Ldiff = Lc.unsqueeze(2) - Lc.unsqueeze(1)          # [nc, i, j, BH]
    dLdiff = dLc.unsqueeze(2) - dLc.unsqueeze(1)
    W = torch.where(tri.view(1, cs, cs, 1), torch.exp(Ldiff), torch.zeros_like(Ldiff))
    dW = W * dLdiff
    intra = torch.einsum("cijb,cjbd->cibd", W * QK, Vf)
    d_intra = (
        torch.einsum("cijb,cjbd->cibd", dW * QK + W * dQK, Vf)
        + torch.einsum("cijb,cjbd->cibd", W * QK, dVf)
    )

    # ---- per-chunk state contributions (decay to chunk end) ----
    L_last = Lc[:, -1:, :]                             # [nc,1,BH]
    dL_last = dLc[:, -1:, :]
    wl = torch.exp(L_last - Lc)                        # [nc,cs,BH]
    dwl = wl * (dL_last - dLc)
    S_contrib = torch.einsum("cjb,cjbn,cjbd->cbnd", wl, Ksc, Vf)
    dS_contrib = (
        torch.einsum("cjb,cjbn,cjbd->cbnd", dwl, Ksc, Vf)
        + torch.einsum("cjb,cjbn,cjbd->cbnd", wl, dKsc, Vf)
        + torch.einsum("cjb,cjbn,cjbd->cbnd", wl, Ksc, dVf)
    )
    chunk_decay = torch.exp(Lc[:, -1, :])             # [nc, BH]
    dchunk_decay = chunk_decay * dLc[:, -1, :]

    # ---- inter-chunk (S, dS) state scan (sequential over chunks) ----
    S_in = torch.zeros(nc, BH, Dqk, Dv, dtype=torch.float32, device=dev)
    dS_in = torch.zeros_like(S_in)
    for c in range(1, nc):
        S_in[c] = chunk_decay[c - 1].view(BH, 1, 1) * S_in[c - 1] + S_contrib[c - 1]
        dS_in[c] = (
            chunk_decay[c - 1].view(BH, 1, 1) * dS_in[c - 1]
            + dchunk_decay[c - 1].view(BH, 1, 1) * S_in[c - 1]
            + dS_contrib[c - 1]
        )

    # ---- inter output (entering state contribution to each position) ----
    expL = torch.exp(Lc)                              # [nc,cs,BH]
    qsin = torch.einsum("cibn,cbnd->cibd", Qr, S_in)
    dqsin = torch.einsum("cibn,cbnd->cibd", dQr, S_in)
    qdsin = torch.einsum("cibn,cbnd->cibd", Qr, dS_in)
    inter = expL.unsqueeze(-1) * qsin
    d_inter = expL.unsqueeze(-1) * (dLc.unsqueeze(-1) * qsin + dqsin + qdsin)

    out_quad = (intra + inter).reshape(Sp, BH, Dv)[:S]
    dout_quad = (d_intra + d_inter).reshape(Sp, BH, Dv)[:S]
    return _finalize(out_quad, dout_quad, f)
