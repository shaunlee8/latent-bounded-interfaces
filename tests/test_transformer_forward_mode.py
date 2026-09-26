"""Forward-mode region JVP checks for the transformer backend.

The forward-mode interface Jacobian must produce the same A_k as the
reverse-mode path on a transformer LBI model, and the fused-kernel tangent
map must match the torch.func reference on both tangent-basis classes
(L-constant, where the first-block norm collapse applies, and general
L-varying, where it is disabled).
"""

from __future__ import annotations

import pytest
import torch

from tests.helpers import cos_rel, worst_cos_rel

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _build_model():
    from train.config import LBITrainingConfig
    from train.model_builders import build_lbi_model

    cfg = LBITrainingConfig(
        vocab_size=64, backbone="transformer", layers=4, dim=256, n_heads=4,
        d_conv=4, region_size=2, message_dim=8, message_hidden_dim=16,
        dtype="bfloat16", interface_type="vector_mlp",
    )
    return build_lbi_model(cfg).to(device="cuda", dtype=torch.bfloat16)


@requires_cuda
def test_forward_mode_Ak_matches_reverse_on_real_transformer() -> None:
    from backward.pullbacks import (
        ForwardModeInterfacePullbackProvider,
        NativeInterfacePullbackProvider,
        materialize_interface_state_jacobian_t_graph,
    )

    torch.manual_seed(0)
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 64), dtype=torch.long, device="cuda")
    _, cache = model.forward_with_cache(input_ids)
    assert len(cache["region_caches"]) == model.num_regions

    fwd = ForwardModeInterfacePullbackProvider().materialize_state_jacobian_t(model=model, cache=cache)
    graph = materialize_interface_state_jacobian_t_graph(model=model, cache=cache)
    native = NativeInterfacePullbackProvider().materialize_state_jacobian_t(model=model, cache=cache)

    assert len(fwd) == len(graph)
    for a, e in zip(fwd, graph):
        assert a.shape == e.shape

    cos_g, rel_g = worst_cos_rel(fwd, graph)
    cos_n, rel_n = worst_cos_rel(fwd, native)
    assert cos_g > 0.999 and rel_g < 3e-2, f"forward-mode A_k vs graph: cos {cos_g:.5f} rel {rel_g:.3e}"
    assert cos_n > 0.999 and rel_n < 3e-2, f"forward-mode A_k vs native: cos {cos_n:.5f} rel {rel_n:.3e}"


@requires_cuda
def test_kernel_matches_reference_constant_basis() -> None:
    """L-constant tangent basis (the A_k access pattern): the kernel path,
    including the first-block norm collapse, must match the fp32 reference."""
    torch.manual_seed(1)
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 64), dtype=torch.long, device="cuda")
    _, cache = model.forward_with_cache(input_ids)
    region_cache = cache["region_caches"][0].backend_cache
    backend = model.region_backend

    B, L, D = region_cache.region_input.shape
    P = 4
    basis = (torch.randn(B, P, 1, D, device="cuda", dtype=torch.bfloat16) * 0.1
             ).expand(-1, -1, L, -1).contiguous()
    ref = backend.region_output_jvp(cache=region_cache, region_input_tangent_basis=basis)
    backend.forward_mode_use_kernel = True
    try:
        kern = backend.region_output_jvp(cache=region_cache, region_input_tangent_basis=basis)
        pooled = backend.region_output_jvp(cache=region_cache,
                                           region_input_tangent_basis=basis, pooled=True)
    finally:
        backend.forward_mode_use_kernel = False
    rel = ((kern.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    assert rel < 3e-2, f"kernel vs reference (constant basis): rel {rel:.3e}"
    # The in-path pooled mean rounds its output to bf16; the fp32 mean over
    # the same values differs by that final rounding.
    prel = ((pooled.float() - kern.float().mean(dim=2, keepdim=True)).abs().max()
            / kern.float().abs().max()).item()
    assert prel < 5e-3, f"pooled vs mean of unpooled: rel {prel:.3e}"


@requires_cuda
def test_kernel_matches_reference_general_basis() -> None:
    """A general L-varying tangent basis takes the norm-JVP path (no first-block
    collapse) and the kernel path must still match the reference."""
    torch.manual_seed(2)
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 64), dtype=torch.long, device="cuda")
    _, cache = model.forward_with_cache(input_ids)
    region_cache = cache["region_caches"][0].backend_cache
    backend = model.region_backend

    B, L, D = region_cache.region_input.shape
    P = 4
    basis = torch.randn(B, P, L, D, device="cuda", dtype=torch.bfloat16) * 0.1
    ref = backend.region_output_jvp(cache=region_cache, region_input_tangent_basis=basis)
    backend.forward_mode_use_kernel = True
    try:
        kern = backend.region_output_jvp(cache=region_cache, region_input_tangent_basis=basis)
    finally:
        backend.forward_mode_use_kernel = False
    rel = ((kern.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    assert rel < 3e-2, f"kernel vs reference (general basis): rel {rel:.3e}"

@requires_cuda
def test_native_vjp_matches_autograd_lowering() -> None:
    """The cache-native region backward (parameter grads + input cotangent
    from cached activations, flash island attention backward) must match the
    autograd-replay lowering."""
    torch.manual_seed(5)
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 64), dtype=torch.long, device="cuda")
    _, cache = model.forward_with_cache(input_ids)
    rc = cache["region_caches"][0].backend_cache
    backend = model.region_backend

    g = torch.randn_like(rc.region_output).to(torch.bfloat16) * 0.1
    grads_n, gin_n = backend.parameter_vjp_with_input_cotangent(
        cache=rc, output_cotangent=g)
    grads_a = backend.lowering.parameter_vjp(
        backend=backend, cache=rc, output_cotangent=g)
    gin_a = backend.lowering.input_pullback_basis(
        backend=backend, cache=rc,
        output_cotangent_basis=g.unsqueeze(1)).squeeze(1)

    assert set(grads_n) == set(grads_a), (
        f"missing: {set(grads_a) ^ set(grads_n)}")
    worst = 0.0
    for n in grads_a:
        rel = ((grads_n[n].float() - grads_a[n].float()).abs().max()
               / (grads_a[n].float().abs().max() + 1e-9)).item()
        assert rel < 4e-2, f"{n}: rel {rel:.3e}"
        worst = max(worst, rel)
    grel = ((gin_n.float() - gin_a.float()).abs().max()
            / gin_a.float().abs().max()).item()
    assert grel < 4e-2, f"input cotangent: rel {grel:.3e}"


@requires_cuda
def test_region_jvp_shapes_and_finiteness() -> None:
    torch.manual_seed(3)
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 64), dtype=torch.long, device="cuda")
    _, cache = model.forward_with_cache(input_ids)
    region_cache = cache["region_caches"][0].backend_cache

    B, L, D = region_cache.region_input.shape
    P = 5
    basis = torch.randn(B, P, L, D, device="cuda", dtype=torch.bfloat16)
    out = model.region_backend.region_output_jvp(cache=region_cache, region_input_tangent_basis=basis)
    assert out.shape == (B, P, L, D)
    assert torch.isfinite(out.float()).all()


@requires_cuda
def test_kernel_cuda_attention_matches_triton(monkeypatch) -> None:
    """The primary CUDA flash-JVP chain vs the triton alternative at a
    full-length region (L=2048, hd64)."""
    from backends import transformer_forward_mode as fm

    if fm._resolve_cuda_attn_jvp() is None:
        pytest.skip("CUDA flash-JVP unavailable (needs sm_90a + toolchain)")

    torch.manual_seed(5)
    model = _build_model()
    backend = model.region_backend
    x = torch.randn(2, 2048, 256, device="cuda", dtype=torch.bfloat16)
    basis = torch.randn(2, 8, 2048, 256, device="cuda", dtype=torch.bfloat16) * 0.02
    _, cache = backend.forward_region(region_input=x, region_index=0)

    monkeypatch.setattr(fm, "_CUDA_ATTN_JVP", False)   # Triton kernel
    ref = fm.transformer_region_output_jvp_kernel(
        backend, cache=cache, region_input_tangent_basis=basis)
    monkeypatch.setattr(fm, "_CUDA_ATTN_JVP", None)    # re-resolve the CUDA kernel
    got = fm.transformer_region_output_jvp_kernel(
        backend, cache=cache, region_input_tangent_basis=basis)

    cos, rel = cos_rel(got, ref)
    assert cos > 0.9999 and rel < 2e-2, (cos, rel)


@requires_cuda
def test_flash_jvp_query_suffix_matches_full() -> None:
    # The suffix-query flash JVP (full-length K/V, suffix tangents) must equal
    # the square kernel's suffix rows when the tangent prefix is zero.
    from backbones.transformer.ops.triton.region_jvp import flash_attention_jvp

    torch.manual_seed(5)
    B, H, L, HD, r, s = 2, 4, 256, 64, 3, 64
    q, k, v = (torch.randn(B, H, L, HD, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    dq, dk, dv = (torch.randn(r, B, H, L, HD, device="cuda", dtype=torch.bfloat16) * 0.1
                  for _ in range(3))
    for t in (dq, dk, dv):
        t[:, :, :, :s] = 0.0
    o_full, do_full = flash_attention_jvp(q, k, v, dq, dk, dv, 0.125)
    o_suf, do_suf = flash_attention_jvp(
        q, k, v,
        dq[:, :, :, s:].contiguous(), dk[:, :, :, s:].contiguous(),
        dv[:, :, :, s:].contiguous(), 0.125, query_start=s)
    assert torch.equal(o_suf, o_full[:, :, s:])
    assert torch.allclose(do_suf.float(), do_full[:, :, :, s:].float(), atol=2e-3, rtol=2e-2)
    # And the tangent prefix of the full result is exactly zero.
    assert do_full[:, :, :, :s].float().abs().max().item() == 0.0


