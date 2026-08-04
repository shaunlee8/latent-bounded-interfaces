"""Forward-mode region JVP gates for the transformer backend.

The forward-mode interface Jacobian must produce the SAME A_k as the
reverse-mode path on a real transformer LBI model, and the fused-kernel
tangent map must match the torch.func reference on both tangent-basis
classes (L-constant, where the first-block norm collapse applies, and
general L-varying, where it is disabled).
"""

from __future__ import annotations

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _build_model():
    from train.config import LBITrainingConfig
    from train.model_builders import build_lbi_model

    cfg = LBITrainingConfig(
        vocab_size=64, backbone="transformer", layers=4, dim=256, n_heads=4,
        d_conv=4, region_size=2, message_dim=8, message_hidden_dim=16,
        dtype="bfloat16",
    )
    return build_lbi_model(cfg).to(device="cuda", dtype=torch.bfloat16)


def _cmp(A, B):
    worst_cos, worst_rel = 1.0, 0.0
    for a, b in zip(A, B):
        a, b = a.float(), b.float()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        rel = ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()
        worst_cos, worst_rel = min(worst_cos, cos), max(worst_rel, rel)
    return worst_cos, worst_rel


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

    cos_g, rel_g = _cmp(fwd, graph)
    cos_n, rel_n = _cmp(fwd, native)
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
def test_kernel_matches_reference_general_basis(monkeypatch) -> None:
    """General L-varying tangent basis: the norm collapse must be disabled
    (LBI_FWDMODE_BCAST=0) and the kernel path must still match."""
    monkeypatch.setenv("LBI_FWDMODE_BCAST", "0")
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
def test_kernel_cache_fed_matches_recompute(monkeypatch) -> None:
    """The cache-fed tangent thread (primal operating points read from the
    region forward's caches) must match the recompute path; both must match
    the reference. The operating points differ only by the cached chain's
    fp32 residuals versus the recomputed bf16 chain."""
    torch.manual_seed(4)
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
        monkeypatch.setenv("LBI_FWDMODE_CACHEPRE", "1")
        fed = backend.region_output_jvp(cache=region_cache, region_input_tangent_basis=basis)
        monkeypatch.setenv("LBI_FWDMODE_CACHEPRE", "0")
        rec = backend.region_output_jvp(cache=region_cache, region_input_tangent_basis=basis)
    finally:
        backend.forward_mode_use_kernel = False
    rel_fr = ((fed.float() - rec.float()).abs().max() / rec.float().abs().max()).item()
    rel_ref = ((fed.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    assert rel_fr < 2e-2, f"cache-fed vs recompute: rel {rel_fr:.3e}"
    assert rel_ref < 3e-2, f"cache-fed vs reference: rel {rel_ref:.3e}"


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

    monkeypatch.delenv("LBI_FWDMODE_CUDA_ATTN", raising=False)
    if fm._resolve_cuda_attn_jvp() is None:
        pytest.skip("CUDA flash-JVP unavailable (needs sm_90a + toolchain)")

    torch.manual_seed(5)
    model = _build_model()
    backend = model.region_backend
    x = torch.randn(2, 2048, 256, device="cuda", dtype=torch.bfloat16)
    basis = torch.randn(2, 8, 2048, 256, device="cuda", dtype=torch.bfloat16) * 0.02
    _, cache = backend.forward_region(region_input=x, region_index=0)

    monkeypatch.setenv("LBI_FWDMODE_CUDA_ATTN", "0")
    ref = fm.transformer_region_output_jvp_kernel(
        backend, cache=cache, region_input_tangent_basis=basis)
    monkeypatch.setenv("LBI_FWDMODE_CUDA_ATTN", "1")
    got = fm.transformer_region_output_jvp_kernel(
        backend, cache=cache, region_input_tangent_basis=basis)

    cos, rel = _cmp([got], [ref])
    assert cos > 0.9999 and rel < 2e-2, (cos, rel)
