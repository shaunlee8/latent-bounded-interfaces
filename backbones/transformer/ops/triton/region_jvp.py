"""Triton kernels for the r-batched transformer region JVP.

`_flash_attention_jvp` runs one flash pass computing the primal and up to
four tangent directions with shared Q/K/V tiles and softmax statistics;
`_fused_qkv_prep` fuses (tangent synthesis | load) -> causal conv taps ->
half-split rope -> head-layout stores for the primal and all directions;
`_rmsnorm_jvp_row` computes the RMSNorm primal + r-direction tangent one token
row per program, reading each row and direction once.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_lane_step(q, k, v, p, pb, causal, alpha, acc_do, acc_s,
                         DQp, DKp, DVp, base_d, rm_rel, rn, rd, QS, DL,
                         SCALE: tl.constexpr, HD: tl.constexpr):
    dq = tl.load(DQp + base_d + rm_rel[:, None] * HD + rd[None, :],
                 mask=rm_rel[:, None] < DL, other=0.0)
    rn_rel = rn - QS
    dkv_mask = (rn_rel[:, None] >= 0) & (rn_rel[:, None] < DL)
    dk = tl.load(DKp + base_d + rn_rel[:, None] * HD + rd[None, :],
                 mask=dkv_mask, other=0.0)
    dv = tl.load(DVp + base_d + rn_rel[:, None] * HD + rd[None, :],
                 mask=dkv_mask, other=0.0)
    ds = (tl.dot(dq, tl.trans(k)) + tl.dot(q, tl.trans(dk))) * SCALE
    ds = tl.where(causal, ds, 0.0)
    pds = p * ds
    acc_s = acc_s * alpha + tl.sum(pds, 1)
    acc_do = acc_do * alpha[:, None] + tl.dot(pds.to(tl.bfloat16), v) + tl.dot(pb, dv)
    return acc_do, acc_s


@triton.jit
def _flash_attention_jvp(Qp, Kp, Vp, DQp, DKp, DVp, Op, DOp,
                         Lctx, DL, QS, lane_str,
                         SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                         HD: tl.constexpr, R: tl.constexpr):
    """Causal flash attention with R tangent directions riding one KV pass:
        dO = (sum(p*dS) V + sum(p) dV - rowsum(p*dS) * O) / l with the online
        rescale corrections, the per-direction accumulators unrolled (R is
        constexpr, R <= 4). QS/DL select the query-suffix mode, where Q/K/V are
        full length while the tangents and outputs cover the DL = Lctx - QS suffix
        rows, exact when the tangent stream is zero before QS; QS = 0, DL = Lctx
        is the square causal case."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    base = pid_bh.to(tl.int64) * Lctx * HD
    base_d = pid_bh.to(tl.int64) * DL * HD
    rm_rel = pid_m * BM + tl.arange(0, BM)
    rm = QS + rm_rel
    rd = tl.arange(0, HD)
    q = tl.load(Qp + base + rm[:, None] * HD + rd[None, :],
                mask=rm_rel[:, None] < DL, other=0.0)

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HD], tl.float32)
    do0 = tl.zeros([BM, HD], tl.float32)
    s0 = tl.zeros([BM], tl.float32)
    do1 = tl.zeros([BM, HD], tl.float32)
    s1 = tl.zeros([BM], tl.float32)
    do2 = tl.zeros([BM, HD], tl.float32)
    s2 = tl.zeros([BM], tl.float32)
    do3 = tl.zeros([BM, HD], tl.float32)
    s3 = tl.zeros([BM], tl.float32)

    hi = QS + (pid_m + 1) * BM
    for start in range(0, hi, BN):
        rn = start + tl.arange(0, BN)
        kv_mask = rn[:, None] < Lctx
        k = tl.load(Kp + base + rn[:, None] * HD + rd[None, :], mask=kv_mask, other=0.0)
        v = tl.load(Vp + base + rn[:, None] * HD + rd[None, :], mask=kv_mask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * SCALE
        causal = rm[:, None] >= rn[None, :]
        s = tl.where(causal, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        pb = p.to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(pb, v)
        do0, s0 = _attention_lane_step(q, k, v, p, pb, causal, alpha, do0, s0,
                                       DQp + 0 * lane_str, DKp + 0 * lane_str,
                                       DVp + 0 * lane_str, base_d, rm_rel, rn, rd,
                                       QS, DL, SCALE, HD)
        if R >= 2:
            do1, s1 = _attention_lane_step(q, k, v, p, pb, causal, alpha, do1, s1,
                                           DQp + 1 * lane_str, DKp + 1 * lane_str,
                                           DVp + 1 * lane_str, base_d, rm_rel, rn, rd,
                                           QS, DL, SCALE, HD)
        if R >= 3:
            do2, s2 = _attention_lane_step(q, k, v, p, pb, causal, alpha, do2, s2,
                                           DQp + 2 * lane_str, DKp + 2 * lane_str,
                                           DVp + 2 * lane_str, base_d, rm_rel, rn, rd,
                                           QS, DL, SCALE, HD)
        if R >= 4:
            do3, s3 = _attention_lane_step(q, k, v, p, pb, causal, alpha, do3, s3,
                                           DQp + 3 * lane_str, DKp + 3 * lane_str,
                                           DVp + 3 * lane_str, base_d, rm_rel, rn, rd,
                                           QS, DL, SCALE, HD)
        m_i = m_new

    o = acc / l_i[:, None]
    st_mask = rm_rel[:, None] < DL
    tl.store(Op + base_d + rm_rel[:, None] * HD + rd[None, :], o.to(tl.bfloat16), mask=st_mask)
    tl.store(DOp + 0 * lane_str + base_d + rm_rel[:, None] * HD + rd[None, :],
             ((do0 - s0[:, None] * o) / l_i[:, None]).to(tl.bfloat16), mask=st_mask)
    if R >= 2:
        tl.store(DOp + 1 * lane_str + base_d + rm_rel[:, None] * HD + rd[None, :],
                 ((do1 - s1[:, None] * o) / l_i[:, None]).to(tl.bfloat16), mask=st_mask)
    if R >= 3:
        tl.store(DOp + 2 * lane_str + base_d + rm_rel[:, None] * HD + rd[None, :],
                 ((do2 - s2[:, None] * o) / l_i[:, None]).to(tl.bfloat16), mask=st_mask)
    if R >= 4:
        tl.store(DOp + 3 * lane_str + base_d + rm_rel[:, None] * HD + rd[None, :],
                 ((do3 - s3[:, None] * o) / l_i[:, None]).to(tl.bfloat16), mask=st_mask)


def flash_attention_jvp(q, k, v, dq, dk, dv, scale, query_start=0):
    """q/k/v [B,H,L,hd] bf16 contiguous; dq/dk/dv [r,B,H,DL,hd] with
    DL = L - query_start; returns (o [B,H,DL,hd], do [r,B,H,DL,hd]). With
    query_start > 0 only the suffix query rows are computed against the
    full-length keys/values — exact when the tangent stream is zero before
    query_start. Lanes run as pairs, the primal is recomputed per pair with
    identical results; config is keyed on (hd, L)."""
    B, H, L, HD = q.shape
    qs = int(query_start)
    DL = L - qs
    r = dq.shape[0]
    if dq.shape[3] != DL:
        raise ValueError("tangent length must equal L - query_start")
    if L >= 4096 and HD <= 64:
        BM, BN, warps, stages = 128, 64, 8, 3
    else:
        BM, BN, warps, stages = 64, 64, 4, 2
    o = torch.empty(B, H, DL, HD, device=q.device, dtype=q.dtype)
    do = torch.empty_like(dq)
    grid = (triton.cdiv(DL, BM), B * H)
    for j in range(0, r, 2):
        rt = min(2, r - j)
        doj = torch.empty(rt, B, H, DL, HD, device=q.device, dtype=q.dtype)
        _flash_attention_jvp[grid](q, k, v, dq[j:], dk[j:], dv[j:], o, doj,
                                   L, DL, qs, dq.stride(0),
                                   SCALE=scale, BM=BM, BN=BN, HD=HD, R=rt,
                                   num_warps=warps, num_stages=stages)
        do[j:j + rt] = doj
    return o, do


@triton.jit
def _fused_qkv_prep(
    QKV, DQKV, U, S, A, W, BIAS, COS, SIN, OUT, DOUT,
    B, L, D3, SOFF, lane_dq, lane_do,
    H: tl.constexpr, HD: tl.constexpr, KW: tl.constexpr,
    R: tl.constexpr, SYNTH: tl.constexpr, ROPE: tl.constexpr, BL: tl.constexpr,
    HASBIAS: tl.constexpr, EMIT_PRIMAL: tl.constexpr,
):
    """One (batch, stream, head, L-tile) program: conv taps, rope, and head
        split for the primal and all R directions, stored directly as [B,H,L,HD]
        and [r,B,H,L,HD]. Conv taps are per-tap masked loads shifted by one row so
        L1 absorbs the re-reads, with the rows below zero masked to the constant
        pad. SYNTH builds the directions in registers from per-token scalars
        (S, A) and a per-direction row U instead of loading DQKV."""
    HALF: tl.constexpr = HD // 2
    pid_l = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    c0 = SOFF + h * HD

    rows = pid_l * BL + tl.arange(0, BL)
    rmask = rows < L
    ce = tl.arange(0, HALF)
    co = HALF + tl.arange(0, HALF)
    qkv_b = QKV + b * L * D3 + c0

    if ROPE:
        cos = tl.load(COS + rows[:, None] * HALF + ce[None, :],
                      mask=rmask[:, None], other=0.0)
        sin = tl.load(SIN + rows[:, None] * HALF + ce[None, :],
                      mask=rmask[:, None], other=0.0)

    ob = b * (H * L * HD) + h * (L * HD)
    if EMIT_PRIMAL:
        ye = tl.zeros((BL, HALF), tl.float32)
        yo = tl.zeros((BL, HALF), tl.float32)
        for kk in tl.static_range(KW):
            hr = rows - (KW - 1) + kk
            hm = (hr >= 0)[:, None] & rmask[:, None]
            wke = tl.load(W + (c0 + ce) * KW + kk)
            wko = tl.load(W + (c0 + co) * KW + kk)
            xe = tl.load(qkv_b + hr[:, None] * D3 + ce[None, :],
                         mask=hm, other=0.0).to(tl.float32)
            xo = tl.load(qkv_b + hr[:, None] * D3 + co[None, :],
                         mask=hm, other=0.0).to(tl.float32)
            ye += wke[None, :] * xe
            yo += wko[None, :] * xo
        if HASBIAS:
            ye += tl.load(BIAS + c0 + ce)[None, :]
            yo += tl.load(BIAS + c0 + co)[None, :]
        if ROPE:
            pe = ye * cos - yo * sin
            po = ye * sin + yo * cos
        else:
            pe = ye
            po = yo
        tl.store(OUT + ob + rows[:, None] * HD + ce[None, :],
                 pe.to(OUT.dtype.element_ty), mask=rmask[:, None])
        tl.store(OUT + ob + rows[:, None] * HD + co[None, :],
                 po.to(OUT.dtype.element_ty), mask=rmask[:, None])

    for j in tl.static_range(R):
        if SYNTH:
            uje = tl.load(U + (j * B + b) * D3 + c0 + ce).to(tl.float32)
            ujo = tl.load(U + (j * B + b) * D3 + c0 + co).to(tl.float32)
        de = tl.zeros((BL, HALF), tl.float32)
        do = tl.zeros((BL, HALF), tl.float32)
        for kk in tl.static_range(KW):
            hr = rows - (KW - 1) + kk
            hmr = (hr >= 0) & rmask
            hm = hmr[:, None]
            wke = tl.load(W + (c0 + ce) * KW + kk)
            wko = tl.load(W + (c0 + co) * KW + kk)
            if SYNTH:
                sk = tl.load(S + b * L + hr, mask=hmr, other=0.0)
                ak = tl.load(A + (j * B + b) * L + hr, mask=hmr, other=0.0)
                xe = tl.load(qkv_b + hr[:, None] * D3 + ce[None, :],
                             mask=hm, other=0.0).to(tl.float32)
                xo = tl.load(qkv_b + hr[:, None] * D3 + co[None, :],
                             mask=hm, other=0.0).to(tl.float32)
                dxe = sk[:, None] * uje[None, :] - ak[:, None] * xe
                dxo = sk[:, None] * ujo[None, :] - ak[:, None] * xo
            else:
                dqkv_b = DQKV + j * lane_dq + b * L * D3 + c0
                dxe = tl.load(dqkv_b + hr[:, None] * D3 + ce[None, :],
                              mask=hm, other=0.0).to(tl.float32)
                dxo = tl.load(dqkv_b + hr[:, None] * D3 + co[None, :],
                              mask=hm, other=0.0).to(tl.float32)
            de += wke[None, :] * dxe
            do += wko[None, :] * dxo
        if ROPE:
            te = de * cos - do * sin
            to = de * sin + do * cos
        else:
            te = de
            to = do
        dob = j * lane_do + ob
        tl.store(DOUT + dob + rows[:, None] * HD + ce[None, :],
                 te.to(DOUT.dtype.element_ty), mask=rmask[:, None])
        tl.store(DOUT + dob + rows[:, None] * HD + co[None, :],
                 to.to(DOUT.dtype.element_ty), mask=rmask[:, None])


def fused_qkv_prep(qkv, dqkv, synth, conv_w, conv_b, cos, sin,
                   n_heads, n_kv_heads, hd, BL=64, warps=4, emit_primal=True):
    """Fused conv + rope + head split for the primal and all tangent directions.

    qkv   [B, L, D3] bf16 (post in_proj, pre conv)
    dqkv  [r, B, L, D3] bf16, or None with `synth`
    synth None, or (U [r,B,D3], s [B,L] fp32, a [r,B,L] fp32) for the
          in-register tangent synthesis dx = s*U_j - a_j*x
    Returns (q, k, v, dq, dk, dv): q [B,H,L,hd] contiguous, dq [r,B,H,L,hd].
    With emit_primal=False the primal side is skipped entirely (the caller
    holds cached q/k/v) and the q/k/v slots return None.
    """
    B, L, D3 = qkv.shape
    if synth is not None:
        # The kernel indexes these as flat contiguous buffers; einsum-produced
        # terms can arrive with permuted strides.
        U, s, a = (x.contiguous() for x in synth)
        r = U.shape[0]
        dq_src = qkv  # dead branch, pointer unused
    else:
        U = s = a = qkv
        r = dqkv.shape[0]
        dq_src = dqkv
    qd = n_heads * hd
    kd = n_kv_heads * hd
    mk = lambda hh: torch.empty(B, hh, L, hd, device=qkv.device, dtype=qkv.dtype)
    mkd = lambda hh: torch.empty(r, B, hh, L, hd, device=qkv.device, dtype=qkv.dtype)
    q, k, v = (mk(n_heads), mk(n_kv_heads), mk(n_kv_heads)) if emit_primal else (None, None, None)
    dq, dk, dv = mkd(n_heads), mkd(n_kv_heads), mkd(n_kv_heads)
    lane_dq = 0 if synth is not None else B * L * D3
    for soff, hh, out, dout, rope in (
        (0, n_heads, q, dq, True),
        (qd, n_kv_heads, k, dk, True),
        (qd + kd, n_kv_heads, v, dv, False),
    ):
        grid = (triton.cdiv(L, BL), B * hh)
        _fused_qkv_prep[grid](
            qkv, dq_src, U, s, a, conv_w,
            conv_w if conv_b is None else conv_b, cos, sin,
            dout if out is None else out, dout,
            B, L, D3, soff, lane_dq, B * hh * L * hd,
            H=hh, HD=hd, KW=conv_w.shape[-1], R=r,
            SYNTH=synth is not None, ROPE=rope, BL=BL,
            HASBIAS=conv_b is not None, EMIT_PRIMAL=emit_primal,
            num_warps=warps,
        )
    return q, k, v, dq, dk, dv


@triton.jit
def _rmsnorm_jvp_row(X, T, W, Y, DY, NROW, EPS,
                     D: tl.constexpr, R: tl.constexpr, EMIT_Y: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, D)
    x = tl.load(X + pid * D + offs).to(tl.float32)
    w = tl.load(W + offs).to(tl.float32)
    s = tl.rsqrt(tl.sum(x * x) / D + EPS)
    if EMIT_Y:
        tl.store(Y + pid * D + offs, (x * s * w).to(Y.dtype.element_ty))
    s3 = s * s * s
    for j in tl.static_range(R):
        t = tl.load(T + (j * NROW + pid) * D + offs).to(tl.float32)
        c = tl.sum(x * t) / D
        dy = (t * s - x * s3 * c) * w
        tl.store(DY + (j * NROW + pid) * D + offs, dy.to(DY.dtype.element_ty))


def rmsnorm_jvp(x, t, weight, eps, warps=8, emit_y=True):
    """RMSNorm primal + tangent; x [B,L,d] contiguous, t [r,B,L,d].
    Requires d to be a power of two; returns (y, dy) in x/t dtypes. With
    emit_y=False only the tangent is computed and y returns None."""
    B, L, d = x.shape
    r = t.shape[0]
    y = torch.empty_like(x) if emit_y else None
    dy = torch.empty_like(t)
    _rmsnorm_jvp_row[(B * L,)](x, t, weight, dy if y is None else y, dy,
                               B * L, eps, D=d, R=r, EMIT_Y=emit_y,
                               num_warps=warps)
    return y, dy


@triton.jit
def _swiglu_jvp_row(GATE, UP, ACT, DH1, OUT, NROW,
                    H: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    # One program per primal row: reads gate/up/act once, then every
    # direction's (dup, dgate) halves of the fc1 tangent row [.., 2H]
    # (up half first, gate half second, matching dh1.chunk(2, dim=-1)).
    pid = tl.program_id(0)
    for h0 in tl.static_range(0, H, BLOCK):
        offs = h0 + tl.arange(0, BLOCK)
        g = tl.load(GATE + pid * H + offs).to(tl.float32)
        u = tl.load(UP + pid * H + offs).to(tl.float32)
        a = tl.load(ACT + pid * H + offs).to(tl.float32)
        sg = tl.sigmoid(g)
        dsilu = sg * (1 + g * (1 - sg))
        for j in tl.static_range(R):
            row = (j * NROW + pid) * (2 * H)
            dup = tl.load(DH1 + row + offs).to(tl.float32)
            dgate = tl.load(DH1 + row + H + offs).to(tl.float32)
            out = dup * a + u * dsilu * dgate
            tl.store(OUT + (j * NROW + pid) * H + offs, out.to(OUT.dtype.element_ty))


def swiglu_jvp(gate, up, activation, dh1, warps=None):
    """Fused tangent of up * silu(gate): gate/up/activation [B,L,H] (the
    cached primal operating point, activation = silu(gate)), dh1 [r,B,L,2H]
    contiguous (the fc1 tangent; its two halves are dup, dgate). Returns
    dmid [r,B,L,H] in dh1's dtype; one pass over the tangent instead of the
    four of the unfused expression. H must be a multiple of 512."""
    B, L, H = gate.shape
    r = dh1.shape[0]
    assert dh1.shape[-1] == 2 * H and dh1.is_contiguous() and H % 512 == 0
    # Widest power-of-two block dividing H: one row per program at H=4096
    # with 16 warps runs ~2.4x faster than 512-wide chunks with 8 warps.
    block = next(b for b in (4096, 2048, 1024, 512) if H % b == 0)
    if warps is None:
        warps = 16 if block >= 2048 else 4
    out = torch.empty((r, B, L, H), device=dh1.device, dtype=dh1.dtype)
    _swiglu_jvp_row[(B * L,)](gate.contiguous(), up.contiguous(), activation.contiguous(),
                              dh1, out, B * L, H=H, R=r, BLOCK=block, num_warps=warps)
    return out
