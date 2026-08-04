"""Parity for the tilelang P-batched SISO pullback.

Compares the P-in-tile paths (MIMO backward with lanes as rank channels)
against the registered Triton backward -- the gradient oracle for the official
recurrence -- at the scan-input boundary: per-lane for Q/Z (which the
unmodified kernel lane-resolves) and lane-sums for K/V/ADT/DT/Trap/Angles
(gradients are linear in the cotangent); the lane-resolved kernels upgrade
the sums to per-lane.

Gated behind LBI_TILELANG_TESTS=1: the tilelang JIT compile takes minutes on
first run, which would dominate the default suite.
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


@requires_cuda
@requires_tilelang_opt_in
def test_tilelang_lanegrid_per_lane_matches_registered_backward() -> None:
    # Stage B3 (P-in-grid `lane_grid` kernel axis): per-lane adjoints for ALL EIGHT
    # scan inputs, zero input replication (forward tensors shared across lane CTAs),
    # single shared bwd_fwd pass, full tuned chunk size.
    pytest.importorskip("tilelang")
    from backbones.mamba3.mamba3 import Mamba3
    from backends.mamba3 import _mamba3_scan_inputs_from_cache
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    from backbones.mamba3.ops.tilelang.mamba3.siso_pbatched import (
        mamba3_siso_pbatched_pullback_lanegrid,
    )

    torch.manual_seed(3)
    dev = "cuda"
    bsz, seqlen, dim, num_lanes = 2, 256, 128, 8
    mixer = Mamba3(
        d_model=dim, d_state=64, expand=2, headdim=64, ngroups=1, chunk_size=64,
        device=dev, dtype=torch.bfloat16,
    )
    u = torch.randn(bsz, seqlen, dim, device=dev, dtype=torch.bfloat16)
    _, cache = mixer.forward_with_cache(u)
    scan = _mamba3_scan_inputs_from_cache(mixer, cache)
    heads, hd = mixer.nheads, mixer.headdim

    basis = torch.randn(bsz, num_lanes, seqlen, heads, hd, device=dev, dtype=torch.bfloat16)
    got = mamba3_siso_pbatched_pullback_lanegrid(
        mixer=mixer, cache=cache, output_cotangent_basis=basis
    )

    names = ("Q", "K", "V", "ADT", "DT", "Trap", "Angles", "Z")
    leaves = {n: scan[n].detach().clone().contiguous().requires_grad_(True) for n in names}
    out = mamba3_siso_combined(
        leaves["Q"], leaves["K"], leaves["V"], leaves["ADT"], leaves["DT"], leaves["Trap"],
        mixer.C_bias.squeeze(1), mixer.B_bias.squeeze(1), leaves["Angles"],
        D=mixer.D, Z=leaves["Z"], chunk_size=mixer.chunk_size,
    )
    for n in names:
        lanes = []
        for j in range(num_lanes):
            (g,) = torch.autograd.grad(
                out, leaves[n], grad_outputs=basis[:, j].to(out.dtype).reshape(out.shape),
                retain_graph=True,
            )
            lanes.append(g.unsqueeze(1))
        ref = torch.cat(lanes, dim=1).float()
        a, b = got[n].float().flatten(), ref.flatten()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        rel = ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()
        assert cos > 0.999 and rel < 5e-2, f"{n} per-lane: cos {cos:.4f} rel {rel:.3e}"


@requires_cuda
@requires_tilelang_opt_in
def test_tilelang_lanetile_per_lane_matches_registered_backward() -> None:
    # In-tile lane kernel (mamba_lanetile_bwd_bwd): LT lanes per CTA with shared
    # forward tiles. Kept as a validated artifact although the in-tile lever
    # is a net loss: ~95% of backward FLOPs are per-lane (each
    # lane carries its own dstates chain), so tile-sharing cannot approach MIMO's
    # "rank rides free" -- lane-grid remains the production kernel.
    pytest.importorskip("tilelang")
    from backbones.mamba3.mamba3 import Mamba3
    from backends.mamba3 import _mamba3_scan_inputs_from_cache
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    from backbones.mamba3.ops.tilelang.mamba3.siso_pbatched import (
        mamba3_siso_pbatched_pullback_lanegrid,
    )

    torch.manual_seed(3)
    dev = "cuda"
    bsz, seqlen, dim, num_lanes = 2, 256, 128, 8
    mixer = Mamba3(
        d_model=dim, d_state=64, expand=2, headdim=64, ngroups=1, chunk_size=64,
        device=dev, dtype=torch.bfloat16,
    )
    u = torch.randn(bsz, seqlen, dim, device=dev, dtype=torch.bfloat16)
    _, cache = mixer.forward_with_cache(u)
    scan = _mamba3_scan_inputs_from_cache(mixer, cache)
    heads, hd = mixer.nheads, mixer.headdim

    basis = torch.randn(bsz, num_lanes, seqlen, heads, hd, device=dev, dtype=torch.bfloat16)
    got = mamba3_siso_pbatched_pullback_lanegrid(
        mixer=mixer, cache=cache, output_cotangent_basis=basis, lane_tile=4
    )

    names = ("Q", "K", "V", "ADT", "DT", "Trap", "Angles", "Z")
    leaves = {n: scan[n].detach().clone().contiguous().requires_grad_(True) for n in names}
    out = mamba3_siso_combined(
        leaves["Q"], leaves["K"], leaves["V"], leaves["ADT"], leaves["DT"], leaves["Trap"],
        mixer.C_bias.squeeze(1), mixer.B_bias.squeeze(1), leaves["Angles"],
        D=mixer.D, Z=leaves["Z"], chunk_size=mixer.chunk_size,
    )
    for n in names:
        lanes = []
        for j in range(num_lanes):
            (g,) = torch.autograd.grad(
                out, leaves[n], grad_outputs=basis[:, j].to(out.dtype).reshape(out.shape),
                retain_graph=True,
            )
            lanes.append(g.unsqueeze(1))
        ref = torch.cat(lanes, dim=1).float()
        a, b = got[n].float().flatten(), ref.flatten()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        rel = ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()
        assert cos > 0.999 and rel < 5e-2, f"{n} per-lane (lanetile): cos {cos:.4f} rel {rel:.3e}"


@requires_cuda
@requires_tilelang_opt_in
def test_tilelang_chunkparallel_per_lane_matches_registered_backward() -> None:
    # Chunk-parallel three-pass backward (torch passes A/B + the
    # mamba_chunkparallel_bwd_bwd pass-C kernel). Parity-exact but an
    # end-to-end loss (0.6-0.7x vs the serial lane-grid kernel:
    # pass C hits the same per-CTA resource wall as the serial kernel, so
    # removing the serial chunk chain buys nothing). Kept validated as the
    # scaffold for a future body rewrite.
    pytest.importorskip("tilelang")
    from backbones.mamba3.mamba3 import Mamba3
    from backends.mamba3 import _mamba3_scan_inputs_from_cache
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    from backbones.mamba3.ops.tilelang.mamba3.siso_pbatched import (
        mamba3_siso_pbatched_pullback_lanegrid,
    )

    torch.manual_seed(3)
    dev = "cuda"
    bsz, seqlen, dim, num_lanes = 2, 512, 128, 8
    mixer = Mamba3(
        d_model=dim, d_state=64, expand=2, headdim=64, ngroups=1, chunk_size=64,
        device=dev, dtype=torch.bfloat16,
    )
    u = torch.randn(bsz, seqlen, dim, device=dev, dtype=torch.bfloat16)
    _, cache = mixer.forward_with_cache(u)
    scan = _mamba3_scan_inputs_from_cache(mixer, cache)
    heads, hd = mixer.nheads, mixer.headdim

    basis = torch.randn(bsz, num_lanes, seqlen, heads, hd, device=dev, dtype=torch.bfloat16)

    names = ("Q", "K", "V", "ADT", "DT", "Trap", "Angles", "Z")
    leaves = {n: scan[n].detach().clone().contiguous().requires_grad_(True) for n in names}
    out = mamba3_siso_combined(
        leaves["Q"], leaves["K"], leaves["V"], leaves["ADT"], leaves["DT"], leaves["Trap"],
        mixer.C_bias.squeeze(1), mixer.B_bias.squeeze(1), leaves["Angles"],
        D=mixer.D, Z=leaves["Z"], chunk_size=mixer.chunk_size,
    )
    # tcs=64 -> nch=8 (torch pass-B fallback); tcs=32 -> nch=16 (fused pass-B
    # kernel). Angles gets a slightly relaxed bar: it is the bf16-accumulation-
    # noisiest channel and the SERIAL path scores the same at these lengths
    # (serial 0.9990 vs chunk-parallel 0.9989 at L=1024).
    for tcs in (64, 32):
        got = mamba3_siso_pbatched_pullback_lanegrid(
            mixer=mixer, cache=cache, output_cotangent_basis=basis,
            tl_chunk_size=tcs, chunk_parallel=True,
        )
        for n in names:
            lanes = []
            for j in range(num_lanes):
                (g,) = torch.autograd.grad(
                    out, leaves[n], grad_outputs=basis[:, j].to(out.dtype).reshape(out.shape),
                    retain_graph=True,
                )
                lanes.append(g.unsqueeze(1))
            ref = torch.cat(lanes, dim=1).float()
            a, b = got[n].float().flatten(), ref.flatten()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            rel = ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()
            cos_bar, rel_bar = (0.998, 6e-2) if n == "Angles" else (0.999, 5e-2)
            assert cos > cos_bar and rel < rel_bar, (
                f"{n} per-lane (chunk-parallel tcs={tcs}): cos {cos:.4f} rel {rel:.3e}"
            )


@requires_cuda
@requires_tilelang_opt_in
def test_tilelang_stage_c_mixer_pullback_matches_autograd() -> None:
    # Stage C: mixer-level P-batched pullback (out_proj^T -> lane-grid scan
    # pullback -> fused lane-batched preprocess VJP) vs TRUE autograd through the
    # mixer. NOTE: this parity only holds with the layout fix in
    # `_Mamba3Function` (see tests/test_mamba3_backward_layout.py) -- the
    # pre-fix autograd reference was itself corrupted at cos ~0.88.
    pytest.importorskip("tilelang")
    from backbones.mamba3.mamba3 import Mamba3
    from backends.mamba3 import _autograd_mixer_input_pullback_basis
    from backbones.mamba3.ops.tilelang.mamba3.siso_pbatched import (
        mamba3_mixer_input_pullback_pbatched,
    )

    torch.manual_seed(3)
    dev = "cuda"
    bsz, seqlen, dim, num_lanes = 2, 256, 128, 8
    mixer = Mamba3(
        d_model=dim, d_state=64, expand=2, headdim=64, ngroups=1, chunk_size=64,
        device=dev, dtype=torch.bfloat16,
    )
    u = torch.randn(bsz, seqlen, dim, device=dev, dtype=torch.bfloat16)
    _, cache = mixer.forward_with_cache(u)
    basis = torch.randn(bsz, num_lanes, seqlen, dim, device=dev, dtype=torch.bfloat16)

    for fused in (True, False):
        got = mamba3_mixer_input_pullback_pbatched(
            mixer=mixer, cache=cache, output_cotangent_basis=basis, fused_epilogue=fused
        )
        ref = _autograd_mixer_input_pullback_basis(
            mixer=mixer, cache=cache, output_cotangent_basis=basis
        )
        a, b = got.float().flatten(), ref.float().flatten()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        rel = ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()
        assert cos > 0.999 and rel < 5e-2, (
            f"mixer pullback (fused={fused}): cos {cos:.4f} rel {rel:.3e}"
        )


@requires_cuda
@requires_tilelang_opt_in
def test_tilelang_pbatched_region_lowering_matches_native() -> None:
    # Region-level: the tilelang_pbatched lowering's P-batched cotangent walk
    # must match the per-lane native lowering across a multi-block region.
    pytest.importorskip("tilelang")
    from backbones.general import BackboneSpec
    from backends.mamba3 import (
        Mamba3RegionBackend,
        NativeMamba3Lowering,
        TileLangPBatchedMamba3Lowering,
    )

    torch.manual_seed(3)
    dev = "cuda"
    bsz, seqlen, dim, num_lanes = 2, 256, 128, 8
    spec = BackboneSpec(
        name="mamba3", dim=dim, layers=2, d_state=64, expand=2, headdim=64,
        ngroups=1, chunk_size=64,
    )
    backend = Mamba3RegionBackend(
        backbone_spec=spec, region_ranges=[(0, 2)], lowering=NativeMamba3Lowering()
    ).to(device=dev, dtype=torch.bfloat16)
    u = torch.randn(bsz, seqlen, dim, device=dev, dtype=torch.bfloat16)
    _, cache = backend.forward_region(region_input=u, region_index=0)
    basis = torch.randn(bsz, num_lanes, seqlen, dim, device=dev, dtype=torch.bfloat16)

    ref = backend.input_pullback_basis(cache=cache, output_cotangent_basis=basis)
    backend.lowering = TileLangPBatchedMamba3Lowering()
    got = backend.input_pullback_basis(cache=cache, output_cotangent_basis=basis)
    a, b = got.float().flatten(), ref.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()
    assert cos > 0.999 and rel < 5e-2, f"region lowering: cos {cos:.4f} rel {rel:.3e}"


@requires_cuda
@requires_tilelang_opt_in
def test_tilelang_pbatched_pullback_per_lane_matches_registered_backward() -> None:
    # Stage B1 (batch-folded lanes, R=1 per lane instance): PER-LANE adjoints for
    # ALL EIGHT scan inputs from one kernel launch pair must match the registered
    # backward lane by lane.
    pytest.importorskip("tilelang")
    from backbones.mamba3.mamba3 import Mamba3
    from backends.mamba3 import _mamba3_scan_inputs_from_cache
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    from backbones.mamba3.ops.tilelang.mamba3.siso_pbatched import mamba3_siso_pbatched_pullback

    torch.manual_seed(3)
    dev = "cuda"
    bsz, seqlen, dim, num_lanes = 2, 256, 128, 8
    mixer = Mamba3(
        d_model=dim, d_state=64, expand=2, headdim=64, ngroups=1, chunk_size=64,
        device=dev, dtype=torch.bfloat16,
    )
    u = torch.randn(bsz, seqlen, dim, device=dev, dtype=torch.bfloat16)
    _, cache = mixer.forward_with_cache(u)
    scan = _mamba3_scan_inputs_from_cache(mixer, cache)
    heads, hd = mixer.nheads, mixer.headdim

    basis = torch.randn(bsz, num_lanes, seqlen, heads, hd, device=dev, dtype=torch.bfloat16)
    got = mamba3_siso_pbatched_pullback(mixer=mixer, cache=cache, output_cotangent_basis=basis)

    names = ("Q", "K", "V", "ADT", "DT", "Trap", "Angles", "Z")
    leaves = {n: scan[n].detach().clone().contiguous().requires_grad_(True) for n in names}
    out = mamba3_siso_combined(
        leaves["Q"], leaves["K"], leaves["V"], leaves["ADT"], leaves["DT"], leaves["Trap"],
        mixer.C_bias.squeeze(1), mixer.B_bias.squeeze(1), leaves["Angles"],
        D=mixer.D, Z=leaves["Z"], chunk_size=mixer.chunk_size,
    )
    for n in names:
        lanes = []
        for j in range(num_lanes):
            (g,) = torch.autograd.grad(
                out, leaves[n], grad_outputs=basis[:, j].to(out.dtype).reshape(out.shape),
                retain_graph=True,
            )
            lanes.append(g.unsqueeze(1))
        ref = torch.cat(lanes, dim=1).float()
        a, b = got[n].float().flatten(), ref.flatten()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        rel = ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()
        assert cos > 0.999 and rel < 5e-2, f"{n} per-lane: cos {cos:.4f} rel {rel:.3e}"


@requires_cuda
@requires_tilelang_opt_in
def test_cuda_pass_c_matches_registered_backward() -> None:
    # The hand-CUDA pass-C scaffold must match the registered
    # backward through the same three-pass harness (chunk_parallel="cuda").
    # Skipped when the extension is not built (cuda/mamba3/build.sh).
    pytest.importorskip("tilelang")
    from cuda.mamba3 import has_cuda_pass_c

    if not has_cuda_pass_c():
        pytest.skip("mamba3_lbi_cuda extension not built")
    from backbones.mamba3.mamba3 import Mamba3
    from backends.mamba3 import _mamba3_scan_inputs_from_cache
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    from backbones.mamba3.ops.tilelang.mamba3.siso_pbatched import (
        mamba3_siso_pbatched_pullback_lanegrid,
    )

    torch.manual_seed(3)
    dev = "cuda"
    bsz, seqlen, dim, num_lanes = 2, 512, 128, 8
    mixer = Mamba3(
        d_model=dim, d_state=64, expand=2, headdim=64, ngroups=1, chunk_size=64,
        device=dev, dtype=torch.bfloat16,
    )
    u = torch.randn(bsz, seqlen, dim, device=dev, dtype=torch.bfloat16)
    _, cache = mixer.forward_with_cache(u)
    scan = _mamba3_scan_inputs_from_cache(mixer, cache)
    heads, hd = mixer.nheads, mixer.headdim
    basis = torch.randn(bsz, num_lanes, seqlen, heads, hd, device=dev, dtype=torch.bfloat16)
    names = ("Q", "K", "V", "ADT", "DT", "Trap", "Angles", "Z")
    leaves = {n: scan[n].detach().clone().contiguous().requires_grad_(True) for n in names}
    out = mamba3_siso_combined(
        leaves["Q"], leaves["K"], leaves["V"], leaves["ADT"], leaves["DT"], leaves["Trap"],
        mixer.C_bias.squeeze(1), mixer.B_bias.squeeze(1), leaves["Angles"],
        D=mixer.D, Z=leaves["Z"], chunk_size=mixer.chunk_size,
    )
    ref = {}
    for n in names:
        lanes = []
        for j in range(num_lanes):
            (g,) = torch.autograd.grad(
                out, leaves[n], grad_outputs=basis[:, j].to(out.dtype).reshape(out.shape),
                retain_graph=True,
            )
            lanes.append(g.unsqueeze(1))
        ref[n] = torch.cat(lanes, dim=1).float()
    # both the scalar scaffold and the tensor-core (wmma) kernel
    for mode in ("cuda", "cuda_mma"):
        got = mamba3_siso_pbatched_pullback_lanegrid(
            mixer=mixer, cache=cache, output_cotangent_basis=basis,
            tl_chunk_size=32, chunk_parallel=mode,
        )
        for n in names:
            a, b = got[n].float().flatten(), ref[n].flatten()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            rel = ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()
            cos_bar, rel_bar = (0.998, 6e-2) if n == "Angles" else (0.999, 5e-2)
            assert cos > cos_bar and rel < rel_bar, (
                f"{n} per-lane ({mode} pass C): cos {cos:.4f} rel {rel:.3e}"
            )
