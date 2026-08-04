"""Gated parity for the fused tilelang forward-mode dual-scan kernel.

The kernel transcribes the chunked (S, dS) dual-scan (`mamba3_siso_dualscan_
chunked`, the frozen reference) onto tensor cores. It must reproduce that
reference, and hence torch.func.jvp on the dense reference, at TF32
tensor-core tolerance.

Gated behind LBI_TILELANG_TESTS=1: the tilelang JIT compile takes minutes.
"""

from __future__ import annotations

import os

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_tilelang_opt_in = pytest.mark.skipif(
    os.environ.get("LBI_TILELANG_TESTS") != "1",
    reason="set LBI_TILELANG_TESTS=1 (tilelang JIT compile is slow)",
)


def _make_inputs(S, H=4, N=64, hd=64, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    B = 2
    Q = torch.randn(B, S, 1, N, device=dev)
    K = torch.randn(B, S, 1, N, device=dev)
    V = torch.randn(B, S, H, hd, device=dev)
    ADT = -torch.rand(B, H, S, device=dev)
    DT = torch.rand(B, H, S, device=dev)
    Trap = torch.randn(B, H, S, device=dev)
    Qb = torch.randn(H, N, device=dev)
    Kb = torch.randn(H, N, device=dev)
    Ang = torch.randn(B, S, 1, N // 2, device=dev).expand(-1, -1, H, -1).contiguous()
    D = torch.randn(H, device=dev)
    Z = torch.randn(B, S, H, hd, device=dev)
    p = (Q, K, V, ADT, DT, Trap, Qb, Kb, Ang, D, Z)
    t = [torch.randn_like(x) for x in p]
    t[6] = torch.zeros_like(p[6]); t[7] = torch.zeros_like(p[7]); t[9] = torch.zeros_like(p[9])
    return p, tuple(t)


def _relerr(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()
    return cos, rel


@requires_cuda
@requires_tilelang_opt_in
@pytest.mark.parametrize("S,cs", [(128, 64), (256, 64), (128, 32)])
def test_tilelang_dualscan_fp32_matches_chunked_reference(S: int, cs: int) -> None:
    """Tight correctness gate: fp32/TF32 operands reproduce the chunked reference
    to ~1e-3 (proves the kernel logic exact up to TF32)."""
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_dualscan_ref import mamba3_siso_dualscan_chunked
    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_dualscan import mamba3_siso_dualscan_tilelang

    p, t = _make_inputs(S, seed=S)
    o_ref, do_ref = mamba3_siso_dualscan_chunked(p, t, chunk_size=cs, compute_dtype=torch.float32)
    o_tl, do_tl = mamba3_siso_dualscan_tilelang(p, t, chunk_size=cs, compute_dtype=torch.float32, operand_dtype='float32')
    cos_f, rel_f = _relerr(o_tl, o_ref)
    cos_j, rel_j = _relerr(do_tl, do_ref)
    assert cos_f > 0.9999 and rel_f < 1e-2, f"tilelang(fp32) forward vs chunked ref (cos {cos_f:.6f}, rel {rel_f:.3e})"
    assert cos_j > 0.9999 and rel_j < 1e-2, f"tilelang(fp32) JVP vs chunked ref (cos {cos_j:.6f}, rel {rel_j:.3e})"


@requires_cuda
@requires_tilelang_opt_in
@pytest.mark.parametrize("S,cs", [(128, 64), (256, 64)])
def test_tilelang_dualscan_bf16_throughput_regime(S: int, cs: int) -> None:
    """Throughput gate: bf16 tensor-core operands (fp32 accum) reproduce the
    reference in the standard bf16 regime (error grows with sequence length via
    state accumulation; the model dtype, and end-to-end A_k meanpools this)."""
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_dualscan_ref import mamba3_siso_dualscan_chunked
    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_dualscan import mamba3_siso_dualscan_tilelang

    p, t = _make_inputs(S, seed=S)
    o_ref, do_ref = mamba3_siso_dualscan_chunked(p, t, chunk_size=cs, compute_dtype=torch.float32)
    o_tl, do_tl = mamba3_siso_dualscan_tilelang(p, t, chunk_size=cs, compute_dtype=torch.float32, operand_dtype='bfloat16')
    cos_f, rel_f = _relerr(o_tl, o_ref)
    cos_j, rel_j = _relerr(do_tl, do_ref)
    assert cos_f > 0.999 and rel_f < 5e-2, f"tilelang(bf16) forward vs chunked ref (cos {cos_f:.6f}, rel {rel_f:.3e})"
    assert cos_j > 0.999 and rel_j < 5e-2, f"tilelang(bf16) JVP vs chunked ref (cos {cos_j:.6f}, rel {rel_j:.3e})"


@requires_cuda
@requires_tilelang_opt_in
def test_tilelang_dualscan_rwide_forward_half_reuse_matches_reference() -> None:
    """Forming A_k over r tangent lanes with the forward half computed
    ONCE (`_rwide`) must produce, for each lane, the same dOut as the per-lane
    reference -- i.e. reuse is exact, not an approximation."""
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_dualscan_ref import mamba3_siso_dualscan_chunked
    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_dualscan import mamba3_siso_dualscan_tilelang_rwide

    S, cs, r = 256, 64, 6
    p, _ = _make_inputs(S, seed=11)
    lanes = []
    for i in range(r):
        _, ti = _make_inputs(S, seed=100 + i)
        lanes.append(ti)
    out, douts = mamba3_siso_dualscan_tilelang_rwide(p, lanes, chunk_size=cs, operand_dtype='float32')
    assert len(douts) == r
    worst = 0.0
    for i in range(r):
        _, do_ref = mamba3_siso_dualscan_chunked(p, lanes[i], chunk_size=cs, compute_dtype=torch.float32)
        _, rel = _relerr(douts[i], do_ref)
        worst = max(worst, rel)
    assert worst < 2e-2, f"rwide forward-half reuse diverges from per-lane reference (worst rel {worst:.3e})"


@requires_cuda
@requires_tilelang_opt_in
def test_tilelang_dualscan_encode_fusion_is_exact() -> None:
    """Encode fusion: folding the interface meanpool + linear
    encode into the kernel output boundary (`_rwide_encoded`) must equal the
    unfused path (materialize the full [r,B,S,H,Dv] tangent, then meanpool +
    encode) EXACTLY -- the fusion is a reordering of a linear reduction, not an
    approximation -- while never materializing the r-stacked region-output."""
    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_dualscan import (
        mamba3_siso_dualscan_tilelang_rwide,
        mamba3_siso_dualscan_tilelang_rwide_encoded,
    )

    S, cs, r, r_out = 256, 64, 6, 8
    p, _ = _make_inputs(S, seed=13)
    B, H, Dv = 2, 4, 64
    lanes = [_make_inputs(S, seed=200 + i)[1] for i in range(r)]
    Wr = torch.randn(r_out, H * Dv, device="cuda")

    delta_f = mamba3_siso_dualscan_tilelang_rwide_encoded(p, lanes, Wr, chunk_size=cs, operand_dtype='float32')
    _, douts = mamba3_siso_dualscan_tilelang_rwide(p, lanes, chunk_size=cs, operand_dtype='float32')
    delta_u = torch.stack([do.reshape(B, S, H * Dv).float().mean(1) @ Wr.t() for do in douts], 0)
    assert delta_f.shape == (r, B, r_out)
    _, rel = _relerr(delta_f, delta_u)
    # The reduction reordering itself is exact; the residual is compiled-vs-eager
    # float reassociation (the encoded path preps per-lane eager, the batched
    # path is torch.compile-fused), ~1e-4-level, far below the bf16 bars.
    assert rel < 2e-3, f"encode fusion diverges from unfused reduce (rel {rel:.3e})"


@requires_cuda
@requires_tilelang_opt_in
@pytest.mark.parametrize("cs,od", [(32, 'bfloat16'), (32, 'float32'), (64, 'bfloat16')])
def test_tilelang_dualscan_d_state_128(cs: int, od: str) -> None:
    """d_state=128 (the representative regime): the kernel must compile and hold
    parity at N=128. bf16 fits at cs=32 AND cs=64; the fp32 accuracy fallback
    needs cs=32 (fp32@cs=64 overflows smem)."""
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_dualscan_ref import mamba3_siso_dualscan_chunked
    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_dualscan import mamba3_siso_dualscan_tilelang

    p, t = _make_inputs(128, H=4, N=128, seed=21)
    o_ref, do_ref = mamba3_siso_dualscan_chunked(p, t, chunk_size=cs, compute_dtype=torch.float32)
    o_tl, do_tl = mamba3_siso_dualscan_tilelang(p, t, chunk_size=cs, compute_dtype=torch.float32, operand_dtype=od)
    cos_f, rel_f = _relerr(o_tl, o_ref)
    cos_j, rel_j = _relerr(do_tl, do_ref)
    bar = 1e-2 if od == 'float32' else 5e-2
    assert cos_f > 0.999 and rel_f < bar, f"N=128 forward (cs={cs},{od}): cos {cos_f:.6f} rel {rel_f:.3e}"
    assert cos_j > 0.999 and rel_j < bar, f"N=128 JVP (cs={cs},{od}): cos {cos_j:.6f} rel {rel_j:.3e}"


@requires_cuda
@requires_tilelang_opt_in
@pytest.mark.parametrize("N,cs", [(64, 64), (128, 32)])
def test_tilelang_dualscan_chunk_parallel_matches_serial(N: int, cs: int) -> None:
    """The three-pass chunk-parallel dualscan (pass A per-chunk contribs +
    tri-matmul pass B + pass C apply) must reproduce the serial kernel exactly
    (same math, re-gridded) and the torch chunked reference. It is auto-selected
    only for underfilled serial grids (B*H <= 32); at saturation the state
    round-trips lose."""
    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_dualscan import mamba3_siso_dualscan_tilelang_rwide
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_dualscan_ref import mamba3_siso_dualscan_chunked

    S, r = 256, 4
    p, _ = _make_inputs(S, H=4, N=N, seed=31)
    lanes = [_make_inputs(S, H=4, N=N, seed=300 + i)[1] for i in range(r)]
    o_s, d_s = mamba3_siso_dualscan_tilelang_rwide(p, lanes, chunk_size=cs, operand_dtype='float32', chunk_parallel=False)
    o_c, d_c = mamba3_siso_dualscan_tilelang_rwide(p, lanes, chunk_size=cs, operand_dtype='float32', chunk_parallel=True)
    _, rel_f = _relerr(o_c, o_s)
    worst_d = max(_relerr(a, b)[1] for a, b in zip(d_c, d_s))
    assert rel_f < 1e-3 and worst_d < 1e-3, f"chunk-parallel vs serial (fwd {rel_f:.3e}, dout {worst_d:.3e})"
    _, do_ref = mamba3_siso_dualscan_chunked(p, lanes[0], chunk_size=cs, compute_dtype=torch.float32)
    _, rel_ref = _relerr(d_c[0], do_ref)
    assert rel_ref < 1e-2, f"chunk-parallel vs torch reference (dout {rel_ref:.3e})"


@requires_cuda
@requires_tilelang_opt_in
def test_forward_mode_kernel_Ak_matches_reverse_on_real_mamba3() -> None:
    """The forward-mode interface Jacobian A_k built with the
    fused dual-scan KERNEL (wired through the real block preprocess/scan/
    postprocess JVPs + the interface assembly) must equal the reverse-mode A_k
    on a real Mamba-3 LBI model, at the bf16 A_k tolerance."""
    from train.config import LBITrainingConfig
    from train.model_builders import build_lbi_model
    from backward.pullbacks import (
        ForwardModeInterfacePullbackProvider,
        NativeInterfacePullbackProvider,
    )

    torch.manual_seed(0)
    cfg = LBITrainingConfig(
        vocab_size=64, backbone="mamba3", layers=4, dim=64, d_state=64, expand=2,
        headdim=64, ngroups=1, chunk_size=64, region_size=2, message_dim=8,
        message_hidden_dim=16, dtype="bfloat16")
    m = build_lbi_model(cfg).to("cuda", torch.bfloat16)
    ids = torch.randint(0, 64, (2, 128), device="cuda")
    _, cache = m.forward_with_cache(ids)

    m.region_backend.forward_mode_use_kernel = True
    fwd = ForwardModeInterfacePullbackProvider().materialize_state_jacobian_t(model=m, cache=cache)
    rev = NativeInterfacePullbackProvider().materialize_state_jacobian_t(model=m, cache=cache)
    assert len(fwd) == len(rev)
    worst = 0.0
    for a, b in zip(fwd, rev):
        assert a.shape == b.shape
        _, rel = _relerr(a, b)
        worst = max(worst, rel)
    assert worst < 3e-2, f"forward-mode-KERNEL A_k diverges from reverse (worst rel {worst:.3e})"


@pytest.mark.parametrize("L", [512, 500])  # 500: Sp > S exercises the s_true mask
def test_gen_mean_kernel_matches_full_path_mean(L: int) -> None:
    """Mean-epilogue gate: the kernel (finalize + row-sum on-chip, only
    [B,cs,H,P] partial sums reach HBM) must match the full gen path + torch
    finalize + sequence mean, at the bf16 tolerance (the Z-gate runs from bf16
    operands in-kernel vs fp32 in the torch finalize). B=2 with DISTINCT lanes
    (the lane-batched-reshape gate rule)."""
    from train.config import LBITrainingConfig
    from train.model_builders import build_lbi_model
    from backends.mamba3_forward_mode import (
        mamba3_mixer_jvp_kernel,
        mamba3_mixer_jvp_mean_kernel,
    )

    torch.manual_seed(0)
    cfg = LBITrainingConfig(
        vocab_size=64, backbone="mamba3", layers=2, dim=256, d_state=128, expand=2,
        headdim=64, ngroups=1, chunk_size=64, region_size=2, message_dim=8,
        message_hidden_dim=16, dtype="bfloat16")
    m = build_lbi_model(cfg).to("cuda", torch.bfloat16)
    mixer = m.region_backend.backbone.blocks[0].mixer
    B = 2
    u = torch.randn(B, L, cfg.dim, device="cuda", dtype=torch.float32)
    dus = [torch.randn(B, L, cfg.dim, device="cuda", dtype=torch.float32) * 0.1
           for _ in range(4)]
    _, douts = mamba3_mixer_jvp_kernel(mixer, u, dus, compute_dtype=torch.float32)
    ref_mean = douts.mean(dim=2)   # douts is stacked [r,B,L,D]
    got = mamba3_mixer_jvp_mean_kernel(mixer, u, dus, compute_dtype=torch.float32)
    assert got.shape == ref_mean.shape
    rel = ((got - ref_mean).abs().max() / (ref_mean.abs().max() + 1e-9)).item()
    assert rel < 2e-2, f"mean-epilogue kernel diverges from full path + mean (rel {rel:.3e})"


@pytest.mark.parametrize("L", [512, 500])  # 500 exercises the s_true mask under rb
def test_gen_mean_lane_blocked_matches_rb1(L: int) -> None:
    """Lane-blocking gate: the rb=2 lane-blocked mean kernel (primal forward-half
    shared across the lane-block, single scratch reused per lane) must match the
    rb=1 mean path at the bf16 tolerance; the shared-vs-recomputed primal
    GEMMs schedule differently, so agreement is to bf16 precision, not bit-exact.
    B=2 with DISTINCT lanes."""
    import os as _os
    from train.config import LBITrainingConfig
    from train.model_builders import build_lbi_model
    from backends.mamba3_forward_mode import mamba3_mixer_jvp_mean_kernel

    torch.manual_seed(0)
    cfg = LBITrainingConfig(
        vocab_size=64, backbone="mamba3", layers=2, dim=256, d_state=128, expand=2,
        headdim=64, ngroups=1, chunk_size=64, region_size=2, message_dim=8,
        message_hidden_dim=16, dtype="bfloat16")
    m = build_lbi_model(cfg).to("cuda", torch.bfloat16)
    mixer = m.region_backend.backbone.blocks[0].mixer
    B = 2
    u = torch.randn(B, L, cfg.dim, device="cuda", dtype=torch.float32)
    dus = [torch.randn(B, L, cfg.dim, device="cuda", dtype=torch.float32) * 0.1
           for _ in range(4)]
    prev = _os.environ.get("LBI_FWDMODE_RB")
    try:
        _os.environ["LBI_FWDMODE_RB"] = "1"
        g1 = mamba3_mixer_jvp_mean_kernel(mixer, u, dus, compute_dtype=torch.float32)
        _os.environ["LBI_FWDMODE_RB"] = "2"
        g2 = mamba3_mixer_jvp_mean_kernel(mixer, u, dus, compute_dtype=torch.float32)
    finally:
        if prev is None:
            _os.environ.pop("LBI_FWDMODE_RB", None)
        else:
            _os.environ["LBI_FWDMODE_RB"] = prev
    assert g2.shape == g1.shape
    rel = ((g2 - g1).abs().max() / (g1.abs().max() + 1e-9)).item()
    assert rel < 2e-2, f"lane-blocked rb=2 diverges from rb=1 (rel {rel:.3e})"


def test_lane_ilp_kernel_matches_base() -> None:
    """Lane-ILP gate: the independent-buffer rb=2 lane-ILP base kernel
    must MATCH the rb=1 base kernel per lane (it does identical per-lane GEMMs,
    just co-resident) -- expected bit-exact. B and 2 DISTINCT lanes."""
    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_dualscan import (
        _get_kernel, _get_lbi_kernel,
    )
    torch.manual_seed(0)
    B, S, H, N, P, cs = 2, 512, 4, 128, 64, 32
    g = lambda *s: torch.randn(*s, device="cuda", dtype=torch.bfloat16)
    QR, KSC, V = g(B, S, H, N), g(B, S, H, N), g(B, S, H, P)
    L = (-torch.rand(B, H, S, device="cuda")).float()
    lanes = [(g(B, S, H, N), g(B, S, H, N), g(B, S, H, P),
              (torch.randn(B, H, S, device="cuda") * 0.1)) for _ in range(2)]
    base = _get_kernel(B, S, H, N, P, cs)
    lbi = _get_lbi_kernel(B, S, H, N, P, cs, 2)
    DQR = torch.stack([l[0] for l in lanes], 0); DKSC = torch.stack([l[1] for l in lanes], 0)
    DV = torch.stack([l[2] for l in lanes], 0); DL = torch.stack([l[3] for l in lanes], 0)
    OUT_l, DOUT_l = lbi(QR, KSC, V, DQR, DKSC, DV, L, DL)
    worst = 0.0
    for b in range(2):
        _, do_b = base(QR, lanes[b][0], KSC, lanes[b][1], V, lanes[b][2], L, lanes[b][3])
        _, rel = _relerr(DOUT_l[b], do_b)
        worst = max(worst, rel)
    assert worst < 5e-3, f"lane-ILP rb=2 diverges from base per-lane (worst rel {worst:.3e})"


@pytest.mark.parametrize("opt", [False, True])
def test_cuda_fwd_passes_match_tilelang_oracle(opt: bool) -> None:
    """Hand-CUDA gate: the forward dual-scan kernels (simple scaffolds
    and the wmma/arena opt rewrites) must match the tilelang oracles on
    real-model inputs at the bf16 band. Skipped when the extension is not
    built (cuda/mamba3/build.sh)."""
    import sys as _sys
    _sys.path.insert(0, "/home/shauncl/LBI/cuda/mamba3")
    CU = pytest.importorskip("mamba3_lbi_cuda")
    from train.config import LBITrainingConfig
    from train.model_builders import build_lbi_model
    from backends.mamba3_forward_mode import (
        _mixer_preprocess_fwd_jvp, _gen_fields_layout, _gen_mean_extra_layout)
    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_dualscan import (
        _get_p1_kernels, _pass_b_tri_lanes)
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_dualscan_ref import (
        _prepare_primal_fields)

    torch.manual_seed(0)
    dim, L, B, r, cs = 256, 512, 2, 4, 64
    cfg = LBITrainingConfig(vocab_size=64, backbone="mamba3", layers=2, dim=dim,
        d_state=128, expand=2, headdim=64, ngroups=1, chunk_size=64, region_size=2,
        message_dim=8, message_hidden_dim=16, dtype="bfloat16")
    m = build_lbi_model(cfg).to("cuda", torch.bfloat16)
    mixer = m.region_backend.backbone.blocks[0].mixer
    u = torch.randn(B, L, dim, device="cuda")
    du = torch.stack([torch.randn(B, L, dim, device="cuda") * 0.1 for _ in range(r)], 0)
    scan_inputs, tb = _mixer_preprocess_fwd_jvp(mixer, u, du, torch.bfloat16)
    pf = _prepare_primal_fields(scan_inputs, torch.bfloat16)
    S, Bb, H = pf["S"], pf["B"], pf["H"]
    Sp = ((S + cs - 1) // cs) * cs
    (fields, QR, KR, Vv, COS, SIN, SCALE, L_bhs,
     DQRAW, DKRAW, DV, DTHETA, DSCALE, DL) = _gen_fields_layout(pf, tb, Sp, cs)
    Zk, DZk, QKD, DQKD, DSK = _gen_mean_extra_layout(pf, fields, Sp, tb)
    G, N, P, Da = DQRAW.shape[3], QR.shape[-1], Vv.shape[-1], COS.shape[-1]
    nc = Sp // cs
    kpa, kpat, kpc, kpcm = _get_p1_kernels(Bb, Sp, H, G, N, Da, P, cs, r, S,
                                           dtype="bfloat16")
    SC_t = kpa(KR, Vv, SCALE, L_bhs)
    DSC_t = kpat(KR, Vv, DKRAW, DV, DTHETA, COS, SIN, SCALE, DSCALE, L_bhs, DL)
    S_IN, DS_IN = _pass_b_tri_lanes(SC_t, DSC_t, L_bhs, DL, cs)
    OUT_t, DOUT_t = kpc(QR, KR, Vv, DQRAW, DKRAW, DV, DTHETA, COS, SIN, SCALE,
                        DSCALE, L_bhs, DL, S_IN, DS_IN)
    PART_t = kpcm(QR, KR, Vv, DQRAW, DKRAW, DV, DTHETA, COS, SIN, SCALE, DSCALE,
                  L_bhs, DL, Zk, DZk, QKD, DQKD, DSK, S_IN, DS_IN)
    # opt path: bf16 state stream (pass A emits bf16 SC/DSC, the CUDA
    # pass B consumes them and emits bf16 S_IN/DS_IN for pass C).
    sdt = torch.bfloat16 if opt else torch.float32
    SC = torch.empty(Bb, H, nc, N, P, device="cuda", dtype=sdt)
    DSC = torch.empty(r, Bb, H, nc, N, P, device="cuda", dtype=sdt)
    CU.fwd_passA(QR, KR, Vv, DQRAW, DKRAW, DV, DTHETA, COS, SIN, SCALE, DSCALE,
                 L_bhs, DL, SC, DSC, cs, opt=opt)
    if opt:
        Sst = torch.empty_like(SC)
        DSst = torch.empty_like(DSC)
        CU.fwd_passB(SC, DSC, L_bhs, DL, Sst, DSst, Sp, cs)
        _, rb1 = _relerr(Sst, S_IN)
        _, rb2 = _relerr(DSst, DS_IN)
        assert rb1 < 3e-2, f"CUDA passB S_IN diverges (rel {rb1:.3e})"
        assert rb2 < 3e-2, f"CUDA passB DS_IN diverges (rel {rb2:.3e})"
    else:
        Sst, DSst = S_IN, DS_IN
    OUT = torch.empty(Bb, Sp, H, P, device="cuda", dtype=torch.float32)
    DOUT = torch.empty(r, Bb, Sp, H, P, device="cuda", dtype=torch.float32)
    CU.fwd_passC(QR, KR, Vv, DQRAW, DKRAW, DV, DTHETA, COS, SIN, SCALE, DSCALE,
                 L_bhs, DL, Sst, DSst, OUT, DOUT, cs, opt=opt)
    PART = torch.empty(r, Bb, H, nc, P, device="cuda", dtype=torch.float32)
    CU.fwd_passC_mean(QR, KR, Vv, DQRAW, DKRAW, DV, DTHETA, COS, SIN, SCALE,
                      DSCALE, L_bhs, DL, Zk, DZk, QKD, DQKD, DSK, Sst, DSst,
                      PART, cs, S, opt=opt)
    for name, a, b in [("SC", SC, SC_t), ("DSC", DSC, DSC_t), ("OUT", OUT, OUT_t),
                       ("DOUT", DOUT, DOUT_t), ("PART", PART, PART_t)]:
        _, r_ = _relerr(a, b)
        assert r_ < 3e-2, f"CUDA {name} diverges from the tilelang oracle (rel {r_:.3e})"


def test_region_jvp_bcast_fast_path_matches_general() -> None:
    """Broadcast gate: for L-broadcast tangents (the A_k decode basis) the block-1
    fused norm+projection collapse (rank-2-in-L) must match the general path at
    the bf16 band, full and pooled."""
    import os as _os
    from train.config import LBITrainingConfig
    from train.model_builders import build_lbi_model

    torch.manual_seed(0)
    cfg = LBITrainingConfig(
        vocab_size=64, backbone="mamba3", layers=2, dim=256, d_state=128, expand=2,
        headdim=64, ngroups=1, chunk_size=64, region_size=2, message_dim=8,
        message_hidden_dim=16, dtype="bfloat16")
    m = build_lbi_model(cfg).to("cuda", torch.bfloat16)
    B, L, r = 2, 512, 4
    ids = torch.randint(0, 64, (B, L), device="cuda")
    _, cache = m.forward_with_cache(ids)
    be = m.region_backend
    be.forward_mode_use_kernel = True
    rc = cache["region_caches"][0].backend_cache
    dcond = torch.randn(B, r, cfg.dim, device="cuda", dtype=torch.float32) * 0.1
    basis = dcond.unsqueeze(2).expand(-1, -1, L, -1)
    prev = _os.environ.get("LBI_FWDMODE_BCAST")
    try:
        for pooled in (False, True):
            _os.environ["LBI_FWDMODE_BCAST"] = "0"
            a = be.region_output_jvp(cache=rc, region_input_tangent_basis=basis,
                                     compute_dtype=torch.float32, pooled=pooled)
            _os.environ["LBI_FWDMODE_BCAST"] = "1"
            b = be.region_output_jvp(cache=rc, region_input_tangent_basis=basis,
                                     compute_dtype=torch.float32, pooled=pooled)
            assert b.shape == a.shape
            rel = ((b - a).abs().max() / (a.abs().max() + 1e-9)).item()
            assert rel < 3e-2, f"bcast fast path diverges (pooled={pooled}, rel {rel:.3e})"
    finally:
        if prev is None:
            _os.environ.pop("LBI_FWDMODE_BCAST", None)
        else:
            _os.environ["LBI_FWDMODE_BCAST"] = prev


def test_region_jvp_pooled_matches_full_mean() -> None:
    """Pooled region gate: `region_output_jvp(pooled=True)` (last block through
    the mean-epilogue kernel) must equal the full region tangent's sequence
    mean."""
    from train.config import LBITrainingConfig
    from train.model_builders import build_lbi_model

    torch.manual_seed(0)
    cfg = LBITrainingConfig(
        vocab_size=64, backbone="mamba3", layers=2, dim=256, d_state=128, expand=2,
        headdim=64, ngroups=1, chunk_size=64, region_size=2, message_dim=8,
        message_hidden_dim=16, dtype="bfloat16")
    m = build_lbi_model(cfg).to("cuda", torch.bfloat16)
    B, L = 2, 512
    ids = torch.randint(0, 64, (B, L), device="cuda")
    _, cache = m.forward_with_cache(ids)
    be = m.region_backend
    be.forward_mode_use_kernel = True
    rc = cache["region_caches"][0].backend_cache
    basis = torch.randn(B, 4, L, cfg.dim, device="cuda", dtype=torch.float32) * 0.1
    full = be.region_output_jvp(cache=rc, region_input_tangent_basis=basis)
    pooled = be.region_output_jvp(cache=rc, region_input_tangent_basis=basis, pooled=True)
    ref = full.mean(dim=2, keepdim=True)
    assert pooled.shape == ref.shape
    rel = ((pooled - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
    assert rel < 2e-2, f"pooled region JVP diverges from full mean (rel {rel:.3e})"
