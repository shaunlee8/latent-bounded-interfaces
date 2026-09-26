"""Forward-mode A_k checks on a Mamba-3 LBI model.

The forward-mode interface Jacobian (with the region JVP supplied by
`region_output_jvp`) must produce the same A_k as the reverse-mode path,
end to end on the Mamba-3 region (in_proj, B/C RMSNorm, SISO scan, out_proj,
pre-norm residual stack), against the autograd graph through the bf16 kernel
and the native reverse provider.
"""

from __future__ import annotations

import pytest
import torch

from tests.helpers import worst_cos_rel

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _build_model():
    from train.config import LBITrainingConfig
    from train.model_builders import build_lbi_model

    cfg = LBITrainingConfig(
        vocab_size=64, backbone="mamba3", layers=4, dim=64, d_state=64, expand=2,
        headdim=64, ngroups=1, chunk_size=16, region_size=2, message_dim=8,
        message_hidden_dim=16, dtype="bfloat16", interface_type="vector_mlp",
    )
    return build_lbi_model(cfg).to(device="cuda", dtype=torch.bfloat16)


@requires_cuda
def test_forward_mode_Ak_matches_reverse_on_real_mamba3() -> None:
    from backward.pullbacks import (
        ForwardModeInterfacePullbackProvider,
        NativeInterfacePullbackProvider,
        materialize_interface_state_jacobian_t_graph,
    )

    torch.manual_seed(0)
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 32), dtype=torch.long, device="cuda")
    _, cache = model.forward_with_cache(input_ids)
    assert len(cache["region_caches"]) == model.num_regions

    fwd = ForwardModeInterfacePullbackProvider().materialize_state_jacobian_t(model=model, cache=cache)
    graph = materialize_interface_state_jacobian_t_graph(model=model, cache=cache)
    native = NativeInterfacePullbackProvider().materialize_state_jacobian_t(model=model, cache=cache)

    assert len(fwd) == len(graph)
    for a, e in zip(fwd, graph):
        assert a.shape == e.shape

    # Forward-mode A_k == reverse-mode A_k at the bf16 tolerance the reverse
    # path holds vs the graph (~5e-3 expected: fp32 reference vs bf16 kernel).
    cos_g, rel_g = worst_cos_rel(fwd, graph)
    cos_n, rel_n = worst_cos_rel(fwd, native)
    assert cos_g > 0.999 and rel_g < 3e-2, f"forward-mode A_k vs graph: cos {cos_g:.5f} rel {rel_g:.3e}"
    assert cos_n > 0.999 and rel_n < 3e-2, f"forward-mode A_k vs native: cos {cos_n:.5f} rel {rel_n:.3e}"


@requires_cuda
def test_forward_mode_region_jvp_shapes_and_finiteness() -> None:
    torch.manual_seed(1)
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 32), dtype=torch.long, device="cuda")
    _, cache = model.forward_with_cache(input_ids)
    region_cache = cache["region_caches"][0].backend_cache

    B, L, D = region_cache.region_input.shape
    P = 5
    basis = torch.randn(B, P, L, D, device="cuda", dtype=torch.bfloat16)
    out = model.region_backend.region_output_jvp(cache=region_cache, region_input_tangent_basis=basis)
    assert out.shape == (B, P, L, D)
    assert torch.isfinite(out.float()).all()


@requires_cuda
def test_mixer_suffix_scan_matches_full() -> None:
    # The suffix route must reproduce the full scan's suffix rows for a
    # zero-prefix tangent stream.
    from backends.mamba3_forward_mode import mamba3_mixer_jvp_kernel
    from train.config import LBITrainingConfig
    from train.model_builders import build_lbi_model

    torch.manual_seed(11)
    cfg = LBITrainingConfig(
        vocab_size=64, backbone="mamba3", layers=2, dim=256, d_state=128, expand=2,
        headdim=64, ngroups=1, chunk_size=64, region_size=2, message_dim=8,
        message_hidden_dim=16, dtype="bfloat16")
    m = build_lbi_model(cfg).to("cuda", torch.bfloat16)
    mixer = m.region_backend.backbone.blocks[0].mixer
    B, L, r, s = 2, 512, 2, 128
    u = torch.randn(B, L, 256, device="cuda", dtype=torch.bfloat16)
    du = torch.randn(r, B, L, 256, device="cuda", dtype=torch.bfloat16) * 0.1
    du[:, :, :s] = 0.0
    out_full, dout_full = mamba3_mixer_jvp_kernel(
        mixer, u, du, compute_dtype=torch.bfloat16, operand_dtype="bfloat16")
    out_suf, dout_suf = mamba3_mixer_jvp_kernel(
        mixer, u, du[:, :, s:].contiguous(), compute_dtype=torch.bfloat16,
        operand_dtype="bfloat16", token_start=s)
    ro = (out_suf.float() - out_full[:, s:].float()).abs().max() / (
        out_full.float().abs().max() + 1e-9)
    rd = (dout_suf.float() - dout_full[:, :, s:].float()).abs().max() / (
        dout_full.float().abs().max() + 1e-9)
    assert float(ro) < 3e-2, f"suffix primal diverges (rel {float(ro):.3e})"
    assert float(rd) < 3e-2, f"suffix tangent diverges (rel {float(rd):.3e})"


