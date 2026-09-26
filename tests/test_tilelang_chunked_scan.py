"""Parity checks for the fused tilelang forward-mode chunked-scan kernels.

The kernels transcribe the chunked (S, dS) scan of
`mamba3_siso_chunked_scan_ref_blocked` onto tensor cores and must reproduce
that reference, and hence torch.func.jvp on the dense reference, at the
tensor-core tolerance. The kernel-level tests are opt-in behind
LBI_TILELANG_TESTS=1 because the tilelang JIT compile takes minutes; the
backend-level tests at the end run the compiled path on every CUDA run.
"""

from __future__ import annotations

import os

import pytest
import torch

from tests.helpers import cos_rel as _relerr

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_tilelang_opt_in = pytest.mark.skipif(
    os.environ.get("LBI_TILELANG_TESTS") != "1",
    reason="set LBI_TILELANG_TESTS=1 (tilelang JIT compile is slow)",
)


def _mamba3_256_model():
    """Two-block, 256-wide Mamba-3 bounded-interface model in bf16 on CUDA."""
    from train.config import LBITrainingConfig
    from train.model_builders import build_lbi_model

    cfg = LBITrainingConfig(
        vocab_size=64, backbone="mamba3", layers=2, dim=256, d_state=128, expand=2,
        headdim=64, ngroups=1, chunk_size=64, region_size=2, message_dim=8,
        message_hidden_dim=16, dtype="bfloat16")
    return build_lbi_model(cfg).to("cuda", torch.bfloat16), cfg


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


@requires_cuda
@requires_tilelang_opt_in
@pytest.mark.parametrize("S,cs", [(128, 64), (256, 64), (128, 32)])
def test_tilelang_chunked_scan_fp32_matches_chunked_reference(S: int, cs: int) -> None:
    """fp32/TF32 operands reproduce the chunked reference to ~1e-3."""
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_chunked_scan_ref import mamba3_siso_chunked_scan_ref_blocked
    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_chunked_scan import mamba3_siso_chunked_scan_tilelang

    p, t = _make_inputs(S, seed=S)
    o_ref, do_ref = mamba3_siso_chunked_scan_ref_blocked(p, t, chunk_size=cs, compute_dtype=torch.float32)
    o_tl, do_tl = mamba3_siso_chunked_scan_tilelang(p, t, chunk_size=cs, compute_dtype=torch.float32, operand_dtype='float32')
    cos_f, rel_f = _relerr(o_tl, o_ref)
    cos_j, rel_j = _relerr(do_tl, do_ref)
    assert cos_f > 0.9999 and rel_f < 1e-2, f"tilelang(fp32) forward vs chunked ref (cos {cos_f:.6f}, rel {rel_f:.3e})"
    assert cos_j > 0.9999 and rel_j < 1e-2, f"tilelang(fp32) JVP vs chunked ref (cos {cos_j:.6f}, rel {rel_j:.3e})"


@requires_cuda
@requires_tilelang_opt_in
@pytest.mark.parametrize("S,cs", [(128, 64), (256, 64)])
def test_tilelang_chunked_scan_bf16_throughput_regime(S: int, cs: int) -> None:
    """bf16 tensor-core operands (fp32 accumulation) reproduce the reference at
    the bf16 tolerance; the error grows with sequence length through the state."""
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_chunked_scan_ref import mamba3_siso_chunked_scan_ref_blocked
    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_chunked_scan import mamba3_siso_chunked_scan_tilelang

    p, t = _make_inputs(S, seed=S)
    o_ref, do_ref = mamba3_siso_chunked_scan_ref_blocked(p, t, chunk_size=cs, compute_dtype=torch.float32)
    o_tl, do_tl = mamba3_siso_chunked_scan_tilelang(p, t, chunk_size=cs, compute_dtype=torch.float32, operand_dtype='bfloat16')
    cos_f, rel_f = _relerr(o_tl, o_ref)
    cos_j, rel_j = _relerr(do_tl, do_ref)
    assert cos_f > 0.999 and rel_f < 5e-2, f"tilelang(bf16) forward vs chunked ref (cos {cos_f:.6f}, rel {rel_f:.3e})"
    assert cos_j > 0.999 and rel_j < 5e-2, f"tilelang(bf16) JVP vs chunked ref (cos {cos_j:.6f}, rel {rel_j:.3e})"


@requires_cuda
@requires_tilelang_opt_in
def test_tilelang_chunked_scan_rwide_forward_half_reuse_matches_reference() -> None:
    """Forming A_k over r tangent directions with the forward half computed
    once (`_rwide`) must give each direction the same dOut as the per-direction
    reference: the reuse is exact."""
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_chunked_scan_ref import mamba3_siso_chunked_scan_ref_blocked
    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_chunked_scan import mamba3_siso_chunked_scan_tilelang_rwide

    S, cs, r = 256, 64, 6
    p, _ = _make_inputs(S, seed=11)
    directions = []
    for i in range(r):
        _, ti = _make_inputs(S, seed=100 + i)
        directions.append(ti)
    out, douts = mamba3_siso_chunked_scan_tilelang_rwide(p, directions, chunk_size=cs, operand_dtype='float32')
    assert len(douts) == r
    worst = 0.0
    for i in range(r):
        _, do_ref = mamba3_siso_chunked_scan_ref_blocked(p, directions[i], chunk_size=cs, compute_dtype=torch.float32)
        _, rel = _relerr(douts[i], do_ref)
        worst = max(worst, rel)
    assert worst < 2e-2, f"rwide forward-half reuse diverges from per-direction reference (worst rel {worst:.3e})"


@requires_cuda
@requires_tilelang_opt_in
@pytest.mark.parametrize("cs,od", [(32, 'bfloat16'), (32, 'float32'), (64, 'bfloat16')])
def test_tilelang_chunked_scan_d_state_128(cs: int, od: str) -> None:
    """d_state=128 (the representative regime): the kernel must compile and hold
    parity at N=128. bf16 fits at cs=32 AND cs=64; the fp32 accuracy fallback
    needs cs=32 (fp32@cs=64 overflows smem)."""
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_chunked_scan_ref import mamba3_siso_chunked_scan_ref_blocked
    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_chunked_scan import mamba3_siso_chunked_scan_tilelang

    p, t = _make_inputs(128, H=4, N=128, seed=21)
    o_ref, do_ref = mamba3_siso_chunked_scan_ref_blocked(p, t, chunk_size=cs, compute_dtype=torch.float32)
    o_tl, do_tl = mamba3_siso_chunked_scan_tilelang(p, t, chunk_size=cs, compute_dtype=torch.float32, operand_dtype=od)
    cos_f, rel_f = _relerr(o_tl, o_ref)
    cos_j, rel_j = _relerr(do_tl, do_ref)
    bar = 1e-2 if od == 'float32' else 5e-2
    assert cos_f > 0.999 and rel_f < bar, f"N=128 forward (cs={cs},{od}): cos {cos_f:.6f} rel {rel_f:.3e}"
    assert cos_j > 0.999 and rel_j < bar, f"N=128 JVP (cs={cs},{od}): cos {cos_j:.6f} rel {rel_j:.3e}"


@requires_cuda
@requires_tilelang_opt_in
@pytest.mark.parametrize("N,cs", [(64, 64), (128, 32)])
def test_tilelang_chunked_scan_chunk_parallel_matches_serial(N: int, cs: int) -> None:
    """The three-pass chunk-parallel scan (per-chunk contributions, tri-matmul
    combine, apply) must reproduce the serial kernel and the torch chunked
    reference. It is selected only for underfilled serial grids."""
    from backbones.mamba3.ops.tilelang.mamba3.mamba3_siso_chunked_scan import mamba3_siso_chunked_scan_tilelang_rwide
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_chunked_scan_ref import mamba3_siso_chunked_scan_ref_blocked

    S, r = 256, 4
    p, _ = _make_inputs(S, H=4, N=N, seed=31)
    directions = [_make_inputs(S, H=4, N=N, seed=300 + i)[1] for i in range(r)]
    o_s, d_s = mamba3_siso_chunked_scan_tilelang_rwide(p, directions, chunk_size=cs, operand_dtype='float32', chunk_parallel=False)
    o_c, d_c = mamba3_siso_chunked_scan_tilelang_rwide(p, directions, chunk_size=cs, operand_dtype='float32', chunk_parallel=True)
    _, rel_f = _relerr(o_c, o_s)
    worst_d = max(_relerr(a, b)[1] for a, b in zip(d_c, d_s))
    assert rel_f < 1e-3 and worst_d < 1e-3, f"chunk-parallel vs serial (fwd {rel_f:.3e}, dout {worst_d:.3e})"
    _, do_ref = mamba3_siso_chunked_scan_ref_blocked(p, directions[0], chunk_size=cs, compute_dtype=torch.float32)
    _, rel_ref = _relerr(d_c[0], do_ref)
    assert rel_ref < 1e-2, f"chunk-parallel vs torch reference (dout {rel_ref:.3e})"


@requires_cuda
@requires_tilelang_opt_in
def test_forward_mode_kernel_Ak_matches_reverse_on_real_mamba3() -> None:
    """The forward-mode A_k built with the fused kernel (through the block's
    preprocess, scan, and postprocess JVPs and the interface assembly) must
    equal the reverse-mode A_k on a Mamba-3 LBI model at the bf16 tolerance."""
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


@requires_cuda
@pytest.mark.parametrize("L", [512, 500])  # 500: Sp > S exercises the s_true mask
def test_gen_mean_kernel_matches_full_path_mean(L: int) -> None:
    """The mean-epilogue kernel (finalize and row-sum on chip) must match the
    full path followed by the torch finalize and sequence mean, at the bf16
    tolerance; B=2 with distinct directions."""
    from backends.mamba3_forward_mode import (
        mamba3_mixer_jvp_kernel,
        mamba3_mixer_jvp_mean_kernel,
    )

    torch.manual_seed(0)
    m, cfg = _mamba3_256_model()
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

@requires_cuda
def test_region_jvp_bcast_fast_path_matches_general() -> None:
    """For L-broadcast tangents (the A_k decode basis) the first-block fused
    norm-and-projection collapse must match the general path at the bf16
    tolerance, full and pooled."""
    torch.manual_seed(0)
    m, cfg = _mamba3_256_model()
    B, L, r = 2, 512, 4
    ids = torch.randint(0, 64, (B, L), device="cuda")
    _, cache = m.forward_with_cache(ids)
    be = m.region_backend
    be.forward_mode_use_kernel = True
    rc = cache["region_caches"][0].backend_cache
    dcond = torch.randn(B, r, cfg.dim, device="cuda", dtype=torch.float32) * 0.1
    basis = dcond.unsqueeze(2).expand(-1, -1, L, -1)
    for pooled in (False, True):
        # A materialized L-constant basis takes the general path; the expanded
        # (stride-0) basis takes the broadcast fast path.
        a = be.region_output_jvp(cache=rc, region_input_tangent_basis=basis.contiguous(),
                                 compute_dtype=torch.float32, pooled=pooled)
        b = be.region_output_jvp(cache=rc, region_input_tangent_basis=basis,
                                 compute_dtype=torch.float32, pooled=pooled)
        assert b.shape == a.shape
        rel = ((b - a).abs().max() / (a.abs().max() + 1e-9)).item()
        assert rel < 3e-2, f"bcast fast path diverges (pooled={pooled}, rel {rel:.3e})"


@requires_cuda
def test_region_jvp_pooled_matches_full_mean() -> None:
    """`region_output_jvp(pooled=True)` (the last block through the mean-epilogue
    kernel) must equal the full region tangent's sequence mean."""
    torch.manual_seed(0)
    m, cfg = _mamba3_256_model()
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
