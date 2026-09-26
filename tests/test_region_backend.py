from __future__ import annotations

import torch
import pytest

from backbones.general import BackboneSpec, build_backbone_stack
from backends import (
    MAMBA3_SCAN_INPUT_NAMES,
    Mamba3RegionBackend,
    Mamba3RegionCache,
    TorchAutogradMamba3MixerLowering,
    TransformerRegionBackend,
    TransformerRegionCache,
    NativeMamba3MixerLowering,
    mamba3_block_input_pullback_native,
    mamba3_block_param_vjp_native,
    mamba3_siso_scan_input_pullback_basis,
)
from backward import (
    NativeInterfacePullbackProvider,
    NativeLocalVJPProvider,
    ScanADEngine,
    TorchAutogradLocalVJPProvider,
)
from interfaces.vector_mlp import VectorMLPInterface
from train.config import LBITrainingConfig
from train.model_builders import build_lbi_model
from models.lbi_language_model import LBILanguageModel, build_region_ranges


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba-3 Triton kernels require CUDA")

# bf16 kernels with atomic-add reductions: parity tolerances far below any
# structural error, per-parameter max-relative (REL) or allclose (ABS).
REL_TOL = 3e-2
REL_TOL_NATIVE_LOCAL = 6e-2
ABS_TOL_BASIS = 4e-2
ABS_TOL_LOWERING = 2e-2
ABS_TOL_LONG_REDUCTION = 5e-2


def _spec() -> BackboneSpec:
    return BackboneSpec(
        name="transformer",
        dim=16,
        layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_intermediate=32,
        attn_head_dim=4,
    )


def _mamba3_spec() -> BackboneSpec:
    return BackboneSpec(
        name="mamba3",
        dim=64,
        layers=4,
        d_state=64,
        expand=2,
        headdim=64,
        ngroups=1,
        chunk_size=16,
    )


def test_transformer_region_backend_matches_forward_range_and_records_layer_cache() -> None:
    torch.manual_seed(41)
    spec = _spec()
    stack = build_backbone_stack(spec)
    region_ranges = build_region_ranges(spec.layers, 2)
    backend = TransformerRegionBackend(backbone=stack, region_ranges=region_ranges)
    x = torch.randn(2, 8, spec.dim)

    for region_index, (start, end) in enumerate(region_ranges):
        actual, cache = backend.forward_region(region_input=x, region_index=region_index)
        expected = stack.forward_range(x, start, end)
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
        assert isinstance(cache, TransformerRegionCache)
        assert cache.region_index == region_index
        assert cache.layer_range == (start, end)
        assert cache.region_input is x
        assert torch.allclose(cache.region_output, actual, atol=1e-6, rtol=1e-6)
        assert [layer_cache.layer_index for layer_cache in cache.layer_caches] == list(range(start, end))
        for layer_cache in cache.layer_caches:
            assert layer_cache.hidden_input.shape == x.shape
            assert layer_cache.hidden_output.shape == x.shape


def test_lbi_transformer_model_uses_transformer_region_backend() -> None:
    cfg = LBITrainingConfig(
        vocab_size=64,
        backbone="transformer",
        layers=4,
        dim=16,
        n_heads=4,
        n_kv_heads=2,
        d_intermediate=32,
        attn_head_dim=4,
        region_size=2,
        message_dim=8,
        message_hidden_dim=16,
    )
    model = build_lbi_model(cfg)
    assert isinstance(model.region_backend, TransformerRegionBackend)
    assert any(key.startswith("region_backend.backbone") for key in model.state_dict())

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    _, cache = model.forward_with_cache(input_ids)
    backend_cache = cache["region_caches"][0].backend_cache
    assert isinstance(backend_cache, TransformerRegionCache)
    assert backend_cache.layer_range == (0, 2)



def test_transformer_region_backend_cache_exposes_attention_and_mlp_tensors() -> None:
    torch.manual_seed(43)
    spec = _spec()
    stack = build_backbone_stack(spec)
    backend = TransformerRegionBackend(backbone=stack, region_ranges=build_region_ranges(spec.layers, 2))
    x = torch.randn(2, 8, spec.dim)

    _, cache = backend.forward_region(region_input=x, region_index=0)
    first = cache.layer_caches[0]

    assert first.attention.norm_input.shape == x.shape
    assert first.attention.norm_output.shape == x.shape
    assert first.attention.qkv.shape[:2] == x.shape[:2]
    assert first.attention.q.shape == (2, spec.n_heads, 8, spec.attn_head_dim)
    assert first.attention.k.shape == (2, spec.n_kv_heads, 8, spec.attn_head_dim)
    assert first.attention.v.shape == (2, spec.n_kv_heads, 8, spec.attn_head_dim)
    assert first.attention.q_rope.shape == first.attention.q.shape
    assert first.attention.k_rope.shape == first.attention.k.shape
    assert first.attention.k_expanded.shape == first.attention.q.shape
    assert first.attention.v_expanded.shape == first.attention.q.shape
    assert first.attention.out_proj_input.shape == x.shape
    assert first.attention.out_proj_output.shape == x.shape
    assert first.mlp.norm_input.shape == x.shape
    assert first.mlp.norm_output.shape == x.shape
    assert first.mlp.up.shape == first.mlp.gate.shape == first.mlp.activation.shape == first.mlp.down_input.shape
    assert first.mlp.down_output.shape == x.shape


def test_transformer_region_backend_delegates_derivatives_to_lowering() -> None:
    torch.manual_seed(59)
    spec = _spec()
    stack = build_backbone_stack(spec)
    region_ranges = build_region_ranges(spec.layers, 2)

    class RecordingLowering:
        name = "recording"

        def __init__(self) -> None:
            self.input_calls = 0
            self.param_calls = 0

        def input_pullback_basis(self, *, backend, cache, output_cotangent_basis):
            self.input_calls += 1
            assert backend is transformer_backend
            assert cache.region_index == 0
            return torch.full_like(output_cotangent_basis, 2.0)

        def parameter_vjp(self, *, backend, cache, output_cotangent):
            self.param_calls += 1
            assert backend is transformer_backend
            assert cache.region_index == 0
            return {"sentinel": output_cotangent.sum().detach().reshape(1)}

    lowering = RecordingLowering()
    transformer_backend = TransformerRegionBackend(
        backbone=stack,
        region_ranges=region_ranges,
        lowering=lowering,
    )
    x = torch.randn(2, 8, spec.dim)
    _, cache = transformer_backend.forward_region(region_input=x, region_index=0)
    basis = torch.randn(2, 3, 8, spec.dim)
    cotangent = torch.randn(2, 8, spec.dim)

    assert torch.equal(
        transformer_backend.input_pullback_basis(cache=cache, output_cotangent_basis=basis),
        torch.full_like(basis, 2.0),
    )
    assert transformer_backend.parameter_vjp(cache=cache, output_cotangent=cotangent).keys() == {"sentinel"}
    assert lowering.input_calls == 1
    assert lowering.param_calls == 1


def test_transformer_region_backend_input_pullback_basis_matches_autograd() -> None:
    torch.manual_seed(47)
    spec = _spec()
    stack = build_backbone_stack(spec)
    region_ranges = build_region_ranges(spec.layers, 2)
    backend = TransformerRegionBackend(backbone=stack, region_ranges=region_ranges)
    x = torch.randn(2, 8, spec.dim)
    _, cache = backend.forward_region(region_input=x, region_index=1)
    basis = torch.randn(2, 3, 8, spec.dim)

    actual = backend.input_pullback_basis(cache=cache, output_cotangent_basis=basis)

    expected_cols = []
    start, end = region_ranges[1]
    for basis_index in range(basis.shape[1]):
        x_req = x.detach().requires_grad_(True)
        y = stack.forward_range(x_req, start, end)
        grad = torch.autograd.grad(
            y,
            x_req,
            grad_outputs=basis[:, basis_index].to(dtype=y.dtype),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        expected_cols.append(grad.unsqueeze(1))
    expected = torch.cat(expected_cols, dim=1)
    assert actual.shape == basis.shape
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_transformer_region_backend_parameter_vjp_matches_autograd() -> None:
    torch.manual_seed(53)
    spec = _spec()
    stack = build_backbone_stack(spec)
    region_ranges = build_region_ranges(spec.layers, 2)
    backend = TransformerRegionBackend(backbone=stack, region_ranges=region_ranges)
    x = torch.randn(2, 8, spec.dim)
    _, cache = backend.forward_region(region_input=x, region_index=0)
    output_cotangent = torch.randn(2, 8, spec.dim)

    actual = backend.parameter_vjp(cache=cache, output_cotangent=output_cotangent)

    start, end = region_ranges[0]
    y = stack.forward_range(x.detach(), start, end)
    params = backend.parameters_for_region(0)
    grads = torch.autograd.grad(
        y,
        params,
        grad_outputs=output_cotangent.to(dtype=y.dtype),
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    name_by_id = {id(param): name for name, param in backend.named_parameters()}
    expected = {name_by_id[id(param)]: grad for param, grad in zip(params, grads) if grad is not None}

    assert actual.keys() == expected.keys()
    for name in actual:
        assert torch.allclose(actual[name], expected[name], atol=1e-6, rtol=1e-6), name


@requires_cuda
def test_mamba3_region_backend_matches_forward_range_and_records_layer_cache() -> None:
    torch.manual_seed(71)
    spec = _mamba3_spec()
    stack = build_backbone_stack(spec).to(device="cuda", dtype=torch.bfloat16)
    region_ranges = build_region_ranges(spec.layers, 2)
    backend = Mamba3RegionBackend(backbone=stack, region_ranges=region_ranges)
    x = torch.randn(2, 32, spec.dim, device="cuda", dtype=torch.bfloat16)

    for region_index, (start, end) in enumerate(region_ranges):
        actual, cache = backend.forward_region(region_input=x, region_index=region_index)
        expected = stack.forward_range(x, start, end)
        assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)
        assert isinstance(cache, Mamba3RegionCache)
        assert cache.region_index == region_index
        assert cache.layer_range == (start, end)
        assert cache.region_input is x
        assert torch.allclose(cache.region_output, actual, atol=1e-2, rtol=1e-2)
        assert [lc.layer_index for lc in cache.layer_caches] == list(range(start, end))


@requires_cuda
def test_mamba3_region_backend_cache_exposes_mixer_abi_tensors() -> None:
    torch.manual_seed(73)
    spec = _mamba3_spec()
    stack = build_backbone_stack(spec).to(device="cuda", dtype=torch.bfloat16)
    backend = Mamba3RegionBackend(backbone=stack, region_ranges=build_region_ranges(spec.layers, 2))
    x = torch.randn(2, 32, spec.dim, device="cuda", dtype=torch.bfloat16)

    _, cache = backend.forward_region(region_input=x, region_index=0)
    mixer = cache.layer_caches[0].block.mixer_cache

    bsz, seqlen, dim = x.shape
    assert mixer.input_u.shape == (bsz, seqlen, dim)
    assert mixer.output.shape == (bsz, seqlen, dim)
    # Frozen forward-cache ABI consumed by the kernel lowerings.
    for field in ("in_proj", "z", "x", "B", "C", "dd_dt", "dd_A", "trap", "angles", "ADT", "DT", "y_inner"):
        assert getattr(mixer, field) is not None, field


@requires_cuda
def test_mamba3_region_backend_input_pullback_basis_matches_autograd() -> None:
    torch.manual_seed(77)
    spec = _mamba3_spec()
    stack = build_backbone_stack(spec).to(device="cuda", dtype=torch.bfloat16)
    region_ranges = build_region_ranges(spec.layers, 2)
    backend = Mamba3RegionBackend(backbone=stack, region_ranges=region_ranges)
    x = torch.randn(2, 32, spec.dim, device="cuda", dtype=torch.bfloat16)
    _, cache = backend.forward_region(region_input=x, region_index=1)
    basis = torch.randn(2, 3, 32, spec.dim, device="cuda", dtype=torch.bfloat16)

    actual = backend.input_pullback_basis(cache=cache, output_cotangent_basis=basis)

    expected_cols = []
    start, end = region_ranges[1]
    for basis_index in range(basis.shape[1]):
        x_req = x.detach().requires_grad_(True)
        y = stack.forward_range(x_req, start, end)
        grad = torch.autograd.grad(
            y,
            x_req,
            grad_outputs=basis[:, basis_index].to(dtype=y.dtype),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        expected_cols.append(grad.unsqueeze(1))
    expected = torch.cat(expected_cols, dim=1)
    assert actual.shape == basis.shape
    assert torch.allclose(actual, expected, atol=ABS_TOL_BASIS, rtol=ABS_TOL_BASIS)


@requires_cuda
def test_mamba3_region_backend_parameter_vjp_matches_autograd() -> None:
    torch.manual_seed(79)
    spec = _mamba3_spec()
    stack = build_backbone_stack(spec).to(device="cuda", dtype=torch.bfloat16)
    region_ranges = build_region_ranges(spec.layers, 2)
    backend = Mamba3RegionBackend(backbone=stack, region_ranges=region_ranges)
    x = torch.randn(2, 32, spec.dim, device="cuda", dtype=torch.bfloat16)
    _, cache = backend.forward_region(region_input=x, region_index=0)
    output_cotangent = torch.randn(2, 32, spec.dim, device="cuda", dtype=torch.bfloat16)

    actual = backend.parameter_vjp(cache=cache, output_cotangent=output_cotangent)

    start, end = region_ranges[0]
    y = stack.forward_range(x.detach(), start, end)
    params = backend.parameters_for_region(0)
    grads = torch.autograd.grad(
        y,
        params,
        grad_outputs=output_cotangent.to(dtype=y.dtype),
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    name_by_id = {id(param): name for name, param in backend.named_parameters()}
    expected = {name_by_id[id(param)]: grad for param, grad in zip(params, grads) if grad is not None}

    assert actual.keys() == expected.keys()
    for name in actual:
        ref = expected[name].float()
        rel = (actual[name].float() - ref).abs().max() / (ref.abs().max() + 1e-9)
        assert rel <= REL_TOL, f"{name}: rel {rel:.4f}"


@requires_cuda
def test_lbi_mamba3_model_uses_mamba3_region_backend() -> None:
    cfg = LBITrainingConfig(
        vocab_size=64,
        backbone="mamba3",
        layers=4,
        dim=64,
        d_state=64,
        expand=2,
        headdim=64,
        ngroups=1,
        chunk_size=16,
        region_size=2,
        message_dim=8,
        message_hidden_dim=16,
        dtype="bfloat16",
    )
    model = build_lbi_model(cfg).to(device="cuda", dtype=torch.bfloat16)
    assert isinstance(model.region_backend, Mamba3RegionBackend)
    assert any(key.startswith("region_backend.backbone") for key in model.state_dict())

    input_ids = torch.randint(0, 64, (2, 32), dtype=torch.long, device="cuda")
    _, cache = model.forward_with_cache(input_ids)
    backend_cache = cache["region_caches"][0].backend_cache
    assert isinstance(backend_cache, Mamba3RegionCache)
    assert backend_cache.layer_range == (0, 2)


def _mamba3_mixer_and_cache(seed: int):
    torch.manual_seed(seed)
    spec = _mamba3_spec()
    stack = build_backbone_stack(spec).to(device="cuda", dtype=torch.bfloat16)
    backend = Mamba3RegionBackend(backbone=stack, region_ranges=build_region_ranges(spec.layers, 2))
    x = torch.randn(2, 32, spec.dim, device="cuda", dtype=torch.bfloat16)
    _, cache = backend.forward_region(region_input=x, region_index=0)
    mixer = stack.blocks[0].mixer
    mixer_cache = cache.layer_caches[0].block.mixer_cache
    basis = torch.randn(2, 3, 32, spec.dim, device="cuda", dtype=torch.bfloat16)
    return mixer, mixer_cache, basis


@requires_cuda
def test_mamba3_mixer_input_pullback_contract_matches_autograd() -> None:
    mixer, mixer_cache, basis = _mamba3_mixer_and_cache(81)
    lowering = TorchAutogradMamba3MixerLowering()

    actual = lowering.input_pullback_basis(mixer=mixer, cache=mixer_cache, output_cotangent_basis=basis)

    expected_cols = []
    for basis_index in range(basis.shape[1]):
        u = mixer_cache.input_u.detach().requires_grad_(True)
        out = mixer(u)
        grad = torch.autograd.grad(out, u, grad_outputs=basis[:, basis_index].to(dtype=out.dtype))[0]
        expected_cols.append(grad.unsqueeze(1))
    expected = torch.cat(expected_cols, dim=1)
    assert actual.shape == basis.shape
    assert torch.allclose(actual, expected, atol=ABS_TOL_BASIS, rtol=ABS_TOL_BASIS)


@requires_cuda
def test_mamba3_native_mixer_lowering_matches_autograd() -> None:
    mixer, mixer_cache, basis = _mamba3_mixer_and_cache(83)
    native = NativeMamba3MixerLowering().input_pullback_basis(
        mixer=mixer, cache=mixer_cache, output_cotangent_basis=basis
    )
    expected = TorchAutogradMamba3MixerLowering().input_pullback_basis(
        mixer=mixer, cache=mixer_cache, output_cotangent_basis=basis
    )
    assert native.shape == basis.shape
    assert torch.allclose(native, expected, atol=ABS_TOL_LOWERING, rtol=ABS_TOL_LOWERING)


@requires_cuda
def test_mamba3_scan_input_pullback_matches_autograd() -> None:
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

    mixer, mcache, _ = _mamba3_mixer_and_cache(91)
    bsz, seqlen, nheads, headdim_v = mcache.x.shape
    num_basis = 3
    basis = torch.randn(bsz, num_basis, seqlen, nheads, headdim_v, device="cuda", dtype=torch.bfloat16)

    actual = mamba3_siso_scan_input_pullback_basis(mixer=mixer, cache=mcache, output_cotangent_basis=basis)

    scan_inputs = {
        "Q": mcache.C.squeeze(2),
        "K": mcache.B.squeeze(2),
        "V": mcache.x,
        "ADT": mcache.ADT,
        "DT": mcache.DT,
        "Trap": mcache.trap,
        "Angles": mcache.angles,
        "Z": mcache.z,
    }
    expected = {name: [] for name in MAMBA3_SCAN_INPUT_NAMES}
    for basis_index in range(num_basis):
        leaves = {n: scan_inputs[n].detach().requires_grad_(True) for n in MAMBA3_SCAN_INPUT_NAMES}
        out = mamba3_siso_combined(
            leaves["Q"], leaves["K"], leaves["V"], leaves["ADT"], leaves["DT"], leaves["Trap"],
            mixer.C_bias.squeeze(1), mixer.B_bias.squeeze(1), leaves["Angles"], mixer.D, leaves["Z"],
            chunk_size=mixer.chunk_size,
        )
        grads = torch.autograd.grad(
            out,
            [leaves[n] for n in MAMBA3_SCAN_INPUT_NAMES],
            grad_outputs=basis[:, basis_index].to(dtype=out.dtype),
        )
        for name, grad in zip(MAMBA3_SCAN_INPUT_NAMES, grads):
            expected[name].append(grad.unsqueeze(1))

    for name in MAMBA3_SCAN_INPUT_NAMES:
        exp = torch.cat(expected[name], dim=1)
        assert actual[name].shape == exp.shape, name
        assert torch.allclose(actual[name], exp, atol=ABS_TOL_LOWERING, rtol=ABS_TOL_LOWERING), name


@requires_cuda
def test_mamba3_native_block_pullback_matches_autograd() -> None:
    torch.manual_seed(93)
    spec = _mamba3_spec()
    stack = build_backbone_stack(spec).to(device="cuda", dtype=torch.bfloat16)
    block = stack.blocks[0]
    hidden = torch.randn(2, 32, spec.dim, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(2, 32, spec.dim, device="cuda", dtype=torch.bfloat16)
    _, _, block_cache = block.forward_with_cache(hidden, residual=residual)
    num_basis = 3
    g_hidden = torch.randn(2, num_basis, 32, spec.dim, device="cuda", dtype=torch.bfloat16)
    g_residual = torch.randn(2, num_basis, 32, spec.dim, device="cuda", dtype=torch.bfloat16)

    gh, gr = mamba3_block_input_pullback_native(
        block=block, cache=block_cache, output_cotangent_basis=g_hidden, residual_cotangent_basis=g_residual
    )

    exp_h, exp_r = [], []
    for basis_index in range(num_basis):
        h = hidden.detach().requires_grad_(True)
        r = residual.detach().requires_grad_(True)
        out_h, out_r, _ = block.forward_with_cache(h, residual=r)
        grads = torch.autograd.grad(
            [out_h, out_r], [h, r],
            grad_outputs=[g_hidden[:, basis_index].to(out_h.dtype), g_residual[:, basis_index].to(out_r.dtype)],
        )
        exp_h.append(grads[0].unsqueeze(1))
        exp_r.append(grads[1].unsqueeze(1))
    expected_h = torch.cat(exp_h, dim=1)
    expected_r = torch.cat(exp_r, dim=1)

    assert torch.allclose(gh, expected_h, atol=ABS_TOL_LOWERING, rtol=ABS_TOL_LOWERING)
    assert gr is not None
    assert torch.allclose(gr, expected_r, atol=ABS_TOL_LOWERING, rtol=ABS_TOL_LOWERING)


@requires_cuda
def test_mamba3_native_block_param_vjp_matches_autograd() -> None:
    torch.manual_seed(97)
    spec = _mamba3_spec()
    stack = build_backbone_stack(spec).to(device="cuda", dtype=torch.bfloat16)
    block = stack.blocks[0]
    hidden = torch.randn(2, 32, spec.dim, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(2, 32, spec.dim, device="cuda", dtype=torch.bfloat16)
    _, _, block_cache = block.forward_with_cache(hidden, residual=residual)
    g_hidden = torch.randn(2, 32, spec.dim, device="cuda", dtype=torch.bfloat16)
    g_residual = torch.randn(2, 32, spec.dim, device="cuda", dtype=torch.bfloat16)

    _, _, grads = mamba3_block_param_vjp_native(
        block=block, cache=block_cache, output_cotangent=g_hidden, residual_cotangent=g_residual
    )

    # Parameter grads depend only on hidden_out (residual_out is param-independent).
    named = list(block.named_parameters())
    params = [p for _, p in named]
    out_h, _, _ = block.forward_with_cache(hidden.detach(), residual=residual.detach())
    expected = torch.autograd.grad(out_h, params, grad_outputs=g_hidden, allow_unused=True)

    assert set(id(p) for p in grads) == set(id(p) for p in params)
    for (name, param), exp in zip(named, expected):
        assert exp is not None, name
        native = grads[param]
        assert native.shape == param.shape, name
        # in_proj.weight and dt_bias are long bf16 reductions over B*L.
        tol = ABS_TOL_LONG_REDUCTION if name.endswith(("in_proj.weight", "dt_bias")) else ABS_TOL_LOWERING
        assert torch.allclose(native.float(), exp.float(), atol=tol, rtol=tol), name


def _mamba3_lbi_model_and_autograd_grads():
    """A two-region Mamba-3 model on CUDA with its plain-autograd gradients."""
    torch.manual_seed(5)
    interface = VectorMLPInterface(
        feature_dim=64, num_regions=2, interface_width=8, interface_map_hidden_dim=32, update_scale_init=0.5
    )
    model = LBILanguageModel(
        vocab_size=64, layers_per_region=2, backbone_spec=_mamba3_spec(), interface=interface
    ).to(device="cuda", dtype=torch.bfloat16)
    input_ids = torch.randint(0, 64, (2, 32), dtype=torch.long, device="cuda")
    model.zero_grad(set_to_none=True)
    model(input_ids).square().mean().backward()
    autograd_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return model, input_ids, autograd_grads


def _assert_rel_close(grad_map, autograd_grads, tol: float) -> None:
    assert set(autograd_grads) <= {n for n, v in grad_map.items() if v is not None}
    for name, grad in autograd_grads.items():
        rel = (grad_map[name].float() - grad.float()).abs().max() / (grad.float().abs().max() + 1e-9)
        assert rel <= tol, f"{name}: rel {rel:.4f}"


@requires_cuda
def test_lbi_mamba3_scan_engine_native_pullback_matches_autograd() -> None:
    model, input_ids, autograd_grads = _mamba3_lbi_model_and_autograd_grads()
    logits, cache = model.forward_with_cache(input_ids)
    result = ScanADEngine(
        pullback_provider=NativeInterfacePullbackProvider(),
        local_vjp_provider=TorchAutogradLocalVJPProvider(),
    ).backward(model=model, loss=logits.square().mean(), cache=cache)

    assert result.diagnostics["interface_pullback_provider"] == "native"
    _assert_rel_close(result.grad_map, autograd_grads, REL_TOL)


@requires_cuda
def test_lbi_mamba3_scan_engine_native_local_vjp_matches_autograd() -> None:
    # Native local VJPs over a graph-free forward (native_backward=True).
    model, input_ids, autograd_grads = _mamba3_lbi_model_and_autograd_grads()
    logits, cache = model.forward_with_cache(input_ids, native_backward=True)
    result = ScanADEngine(
        pullback_provider=NativeInterfacePullbackProvider(),
        local_vjp_provider=NativeLocalVJPProvider(),
    ).backward(model=model, loss=logits.square().mean(), cache=cache)

    assert result.diagnostics["local_vjp_provider"] == "native_local"
    _assert_rel_close(result.grad_map, autograd_grads, REL_TOL_NATIVE_LOCAL)
