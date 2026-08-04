"""Parity gates for the forward-mode (JVP) SISO scan primitive.

The forward-mode interface-Jacobian path needs a differentiable oracle for
the scan, because the registered kernel
supports no functorch/dual transforms and bf16 finite differences are unusable.
`mamba3_siso_out_ref` is that oracle; these tests assert:

  1. the reference forward matches the registered `mamba3_siso_combined` kernel
     at the standard bf16 tolerance (it is a faithful reproduction, not a new
     model), and
  2. the forward-mode JVP built on it (`mamba3_siso_jvp`) is EXACT against
     central finite differences of the same reference (fp32).

Together these establish the forward-mode scan contract that the native
forward-mode scan kernel will implement and gate against.
"""

from __future__ import annotations

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_inputs(dtype: torch.dtype, *, B=2, S=128, H=4, hd=64, N=64, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    Q = torch.randn(B, S, 1, N, device=dev, dtype=dtype)
    K = torch.randn(B, S, 1, N, device=dev, dtype=dtype)
    V = torch.randn(B, S, H, hd, device=dev, dtype=dtype)
    ADT = -torch.rand(B, H, S, device=dev, dtype=torch.float32)
    DT = torch.rand(B, H, S, device=dev, dtype=torch.float32)
    Trap = torch.randn(B, H, S, device=dev, dtype=dtype)
    Qb = torch.randn(H, N, device=dev, dtype=torch.float32)
    Kb = torch.randn(H, N, device=dev, dtype=torch.float32)
    Ang = torch.randn(B, S, 1, N // 2, device=dev, dtype=dtype).expand(-1, -1, H, -1).contiguous()
    D = torch.randn(H, device=dev, dtype=torch.float32)
    Z = torch.randn(B, S, H, hd, device=dev, dtype=dtype)
    return Q, K, V, ADT, DT, Trap, Qb, Kb, Ang, D, Z


def _relerr(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a, b = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()
    return cos, rel


@requires_cuda
def test_reference_forward_matches_registered_kernel() -> None:
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_ref import mamba3_siso_out_ref

    Q, K, V, ADT, DT, Trap, Qb, Kb, Ang, D, Z = _make_inputs(torch.bfloat16)
    out_k = mamba3_siso_combined(
        Q=Q.contiguous(), K=K.contiguous(), V=V.contiguous(),
        ADT=ADT, DT=DT, Trap=Trap, Q_bias=Qb, K_bias=Kb,
        Angles=Ang, D=D, Z=Z, chunk_size=64,
    )
    out_r = mamba3_siso_out_ref(Q, K, V, ADT, DT, Trap, Qb, Kb, Ang, D, Z, compute_dtype=torch.float32)
    cos, rel = _relerr(out_r, out_k)
    assert cos > 0.9999 and rel < 2e-2, f"reference forward diverges from kernel (cos {cos:.5f}, rel {rel:.3e})"


@requires_cuda
def test_scan_jvp_matches_finite_differences() -> None:
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_ref import mamba3_siso_out_ref, mamba3_siso_jvp

    primals = _make_inputs(torch.float32)
    tangents = tuple(torch.randn_like(p) for p in primals)
    _, dout = mamba3_siso_jvp(primals, tangents, compute_dtype=torch.float32)

    eps = 1e-3
    pp = tuple(p + eps * t for p, t in zip(primals, tangents))
    pm = tuple(p - eps * t for p, t in zip(primals, tangents))
    fp = mamba3_siso_out_ref(*pp, compute_dtype=torch.float64)
    fm = mamba3_siso_out_ref(*pm, compute_dtype=torch.float64)
    fd = ((fp - fm) / (2 * eps)).float()

    cos, rel = _relerr(dout, fd)
    assert cos > 0.9999 and rel < 1e-2, f"scan JVP disagrees with finite differences (cos {cos:.6f}, rel {rel:.3e})"


@requires_cuda
@pytest.mark.parametrize("nheads_qk", [1, 4])
def test_dualscan_recurrent_forward_matches_reference(nheads_qk: int) -> None:
    """The explicit augmented dual-scan runs the scan in RECURRENT state-space
    form (the tiling the fused kernel chunks); its forward must match the dense
    quadratic reference -- gate that the recurrence is the same scan."""
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_ref import mamba3_siso_out_ref
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_dualscan_ref import mamba3_siso_dualscan

    p = _make_inputs(torch.float32, H=4, N=64, seed=3)
    if nheads_qk == 4:  # non-GQA: give Q/K a full head count
        B, S = p[0].shape[:2]
        dev = "cuda"
        p = (torch.randn(B, S, 4, 64, device=dev), torch.randn(B, S, 4, 64, device=dev)) + p[2:]
    tang = tuple(torch.zeros_like(x) for x in p)
    out_ds, _ = mamba3_siso_dualscan(p, tang, compute_dtype=torch.float32)
    out_ref = mamba3_siso_out_ref(*p, compute_dtype=torch.float32)
    cos, rel = _relerr(out_ds, out_ref)
    assert cos > 0.9999 and rel < 1e-4, f"recurrent dual-scan forward vs dense ref (cos {cos:.6f}, rel {rel:.3e})"


@requires_cuda
def test_dualscan_jvp_matches_func_jvp() -> None:
    """The frozen kernel semantics: the hand-derived (S, dS) dual recurrence must
    equal torch.func.jvp on the dense reference exactly. This is the parity
    oracle the fused forward-mode kernel transcribes."""
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_ref import mamba3_siso_jvp
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_dualscan_ref import mamba3_siso_dualscan

    p = _make_inputs(torch.float32, seed=5)
    tang = [torch.randn_like(x) for x in p]
    # biases (idx 6,7) and D (idx 9) are parameters -> zero tangent in the region JVP.
    tang[6] = torch.zeros_like(p[6]); tang[7] = torch.zeros_like(p[7]); tang[9] = torch.zeros_like(p[9])
    tang = tuple(tang)
    _, dout_ds = mamba3_siso_dualscan(p, tang, compute_dtype=torch.float32)
    _, dout_true = mamba3_siso_jvp(p, tang, compute_dtype=torch.float32)
    cos, rel = _relerr(dout_ds, dout_true)
    assert cos > 0.99999 and rel < 1e-4, f"dual-scan JVP vs torch.func.jvp (cos {cos:.6f}, rel {rel:.3e})"


@requires_cuda
@pytest.mark.parametrize("S,cs", [(128, 64), (96, 32), (130, 64), (50, 16)])
def test_chunked_dualscan_matches_recurrent_and_func_jvp(S: int, cs: int) -> None:
    """The CHUNKED (SSD) dual-scan -- chunk-local quadratic + inter-chunk (S, dS)
    state passing, the exact tiling the fused tilelang kernel implements -- must
    equal both the recurrent dual-scan and torch.func.jvp, at sequence lengths
    not divisible by the chunk size (padding path)."""
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_ref import mamba3_siso_jvp
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_dualscan_ref import (
        mamba3_siso_dualscan,
        mamba3_siso_dualscan_chunked,
    )

    p = _make_inputs(torch.float32, S=S, seed=7)
    tang = [torch.randn_like(x) for x in p]
    tang[6] = torch.zeros_like(p[6]); tang[7] = torch.zeros_like(p[7]); tang[9] = torch.zeros_like(p[9])
    tang = tuple(tang)

    o_rec, _ = mamba3_siso_dualscan(p, tang, compute_dtype=torch.float32)
    o_ch, do_ch = mamba3_siso_dualscan_chunked(p, tang, chunk_size=cs, compute_dtype=torch.float32)
    _, do_true = mamba3_siso_jvp(p, tang, compute_dtype=torch.float32)

    cos_f, rel_f = _relerr(o_ch, o_rec)
    cos_j, rel_j = _relerr(do_ch, do_true)
    assert cos_f > 0.99999 and rel_f < 1e-4, f"chunked forward vs recurrent (cos {cos_f:.6f}, rel {rel_f:.3e})"
    assert cos_j > 0.99999 and rel_j < 1e-4, f"chunked JVP vs torch.func.jvp (cos {cos_j:.6f}, rel {rel_j:.3e})"


@requires_cuda
def test_scan_jvp_is_r_wide_vmappable() -> None:
    """The message-basis push is r independent JVPs sharing one forward
    linearization -- confirm vmap over the tangent batch works (the r-wide
    forward scan the native kernel batches in-tile)."""
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_ref import mamba3_siso_jvp

    primals = _make_inputs(torch.float32)
    r = 8
    Vt = torch.randn(r, *primals[2].shape, device="cuda", dtype=torch.float32)
    zeros = [torch.zeros_like(p) for p in primals]

    def push(dv: torch.Tensor) -> torch.Tensor:
        tang = list(zeros)
        tang[2] = dv
        _, d = mamba3_siso_jvp(primals, tuple(tang), compute_dtype=torch.float32)
        return d

    douts = torch.vmap(push)(Vt)
    assert douts.shape == (r, *primals[2].shape)
    assert torch.isfinite(douts).all()
