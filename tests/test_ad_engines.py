from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from backbones.general import BackboneSpec
from backward import (
    AutogradEngine,
    NativeInterfacePullbackProvider,
    NativeLocalVJPProvider,
    ScanADEngine,
    TorchGraphInterfacePullbackProvider,
    interface_state_jacobian_t_for_region,
    lbi_scan_backward_step,
    materialize_interface_state_jacobian_t_graph,
    native_initial_backward,
    native_region_backward,
    reduce_region_results,
)
from backward.suffix_scan import propagate_state_adjoint_from_last_region_input
from interfaces.vector_mlp import VectorMLPInterface
from models.lbi_language_model import LBILanguageModel


def _assert_grad_maps_close(
    actual: dict[str, torch.Tensor | None],
    expected: dict[str, torch.Tensor | None],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    assert actual.keys() == expected.keys()
    for name in actual:
        assert (actual[name] is None) == (expected[name] is None), name
        if actual[name] is not None and expected[name] is not None:
            assert torch.allclose(actual[name], expected[name], atol=atol, rtol=rtol), name


def _parameter_grad_map(model: nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if param.grad is None else param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def test_autograd_engine_matches_direct_backward() -> None:
    torch.manual_seed(11)
    direct = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 3))
    engine_model = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 3))
    engine_model.load_state_dict(direct.state_dict())
    x = torch.randn(5, 4)

    direct_loss = direct(x).square().mean()
    direct.zero_grad(set_to_none=True)
    direct_loss.backward()
    expected = _parameter_grad_map(direct)

    engine_loss = engine_model(x).square().mean()
    result = AutogradEngine().backward(model=engine_model, loss=engine_loss)

    _assert_grad_maps_close(result.grad_map, expected)
    _assert_grad_maps_close(_parameter_grad_map(engine_model), expected)
    assert result.diagnostics == {}


def _backbone_spec() -> BackboneSpec:
    return BackboneSpec(
        name="transformer",
        dim=16,
        layers=3,
        n_heads=4,
        n_kv_heads=2,
        d_intermediate=32,
        attn_head_dim=4,
    )


def _build_lbi_model(*, tie_embeddings: bool = False) -> LBILanguageModel:
    interface = VectorMLPInterface(
        feature_dim=16,
        num_regions=3,
        interface_width=8,
        interface_map_hidden_dim=16,
        update_scale_init=0.5,
    )
    return LBILanguageModel(
        vocab_size=64,
        layers_per_region=1,
        backbone_spec=_backbone_spec(),
        interface=interface,
        tie_embeddings=tie_embeddings,
    ).to(dtype=torch.float32)


def test_lbi_canvas_owns_embedding_and_state_keys() -> None:
    torch.manual_seed(21)
    model = _build_lbi_model()
    input_ids = torch.randint(0, 64, (2, 10), dtype=torch.long)

    assert torch.allclose(model.canvas(input_ids), model.canvas.embedding(input_ids), atol=0.0, rtol=0.0)
    assert model.canvas_vjp_parameters() == model.canvas.vjp_parameters()
    assert {id(param) for param in model.canvas_vjp_parameters()} == {id(model.canvas.embedding.weight)}
    assert "canvas.embedding.weight" in model.state_dict()


def test_lbi_readout_owns_norm_and_untied_head_state_keys() -> None:
    torch.manual_seed(25)
    model = _build_lbi_model(tie_embeddings=False)
    features = torch.randn(2, 10, model.hidden_dim)

    assert model.readout.lm_head is not None
    expected_input = model.readout.norm(features)
    if expected_input.dtype != model.readout.lm_head.weight.dtype:
        expected_input = expected_input.to(dtype=model.readout.lm_head.weight.dtype)
    expected = model.readout.lm_head(expected_input)

    assert torch.allclose(model.readout(features, canvas=model.canvas), expected, atol=0.0, rtol=0.0)
    assert model.output_head_vjp_parameters() == model.readout.vjp_parameters()
    assert any(key.startswith("readout.norm") for key in model.state_dict())
    assert "readout.lm_head.weight" in model.state_dict()


def test_lbi_tied_readout_projects_through_canvas_weight() -> None:
    torch.manual_seed(29)
    model = _build_lbi_model(tie_embeddings=True)
    features = torch.randn(2, 10, model.hidden_dim)

    assert model.readout.lm_head is None
    expected_input = model.readout.norm(features)
    if expected_input.dtype != model.canvas.output_weight().dtype:
        expected_input = expected_input.to(dtype=model.canvas.output_weight().dtype)
    expected = torch.nn.functional.linear(expected_input, model.canvas.output_weight())

    assert torch.allclose(model.readout(features, canvas=model.canvas), expected, atol=0.0, rtol=0.0)
    assert {id(param) for param in model.output_head_vjp_parameters()} == {
        id(param) for param in model.readout.norm.parameters() if param.requires_grad
    }
    assert "readout.lm_head.weight" not in model.state_dict()


def test_component_vjp_parameter_groups_match_current_lbi_components() -> None:
    torch.manual_seed(19)
    model = _build_lbi_model()

    assert {id(param) for param in model.output_head_vjp_parameters()} == {
        id(param)
        for module in (model.readout.norm, model.readout.lm_head)
        if module is not None
        for param in module.parameters()
        if param.requires_grad
    }
    assert {id(param) for param in model.canvas_vjp_parameters()} == {id(model.canvas.embedding.weight)}
    assert {id(param) for param in model.interface.initial_vjp_parameters()} == {
        id(param) for param in model.interface.initial_encoder.parameters() if param.requires_grad
    }
    for region_index in range(model.num_regions):
        assert {id(param) for param in model.interface.region_vjp_parameters(region_index)} == {
            id(param)
            for module in (
                model.interface.decoders[region_index],
                model.interface.encoders[region_index],
                model.interface.norms[region_index],
            )
            for param in module.parameters()
            if param.requires_grad
        }
    assert {id(param) for param in model.shared_local_vjp_parameters()} == {
        id(model.canvas.embedding.weight),
        id(model.interface.update_scale),
    }


def test_torch_interface_pullback_providers_match_model_materialization() -> None:
    torch.manual_seed(17)
    model = _build_lbi_model()
    input_ids = torch.randint(0, 64, (2, 10), dtype=torch.long)
    _, cache = model.forward_with_cache(input_ids)

    graph_expected = materialize_interface_state_jacobian_t_graph(model=model, cache=cache)
    graph_actual = TorchGraphInterfacePullbackProvider().materialize_state_jacobian_t(
        model=model,
        cache=cache,
    )
    assert len(graph_actual) == len(graph_expected)
    for actual, expected in zip(graph_actual, graph_expected):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_native_interface_pullback_provider_matches_graph() -> None:
    torch.manual_seed(17)
    model = _build_lbi_model()
    input_ids = torch.randint(0, 64, (2, 10), dtype=torch.long)
    _, cache = model.forward_with_cache(input_ids)

    # NativeInterfacePullbackProvider composes structured pullbacks with the
    # backend's input_pullback_basis; must match the graph provider exactly.
    expected = materialize_interface_state_jacobian_t_graph(model=model, cache=cache)
    actual = NativeInterfacePullbackProvider().materialize_state_jacobian_t(model=model, cache=cache)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a.shape == e.shape
        assert torch.allclose(a, e, atol=1e-5, rtol=1e-5)


def test_scan_engine_from_native_config_matches_autograd() -> None:
    # native_backward=True must build the native providers and
    # match plain autograd over a graph-free forward.
    torch.manual_seed(31)
    model = _build_lbi_model()
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)

    model.zero_grad(set_to_none=True)
    model(input_ids).square().mean().backward()
    autograd_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    cfg = SimpleNamespace(native_backward=True, interface_jacobian_mode="graph")
    engine = ScanADEngine.from_config(cfg)
    assert engine.pullback_provider.name == "native"
    assert engine.local_vjp_provider.name == "native_local"

    logits, cache = model.forward_with_cache(input_ids, native_backward=True)
    result = engine.backward(model=model, loss=logits.square().mean(), cache=cache)

    assert set(autograd_grads) <= {n for n, v in result.grad_map.items() if v is not None}
    for name, grad in autograd_grads.items():
        native = result.grad_map[name]
        assert torch.allclose(native.float(), grad.float(), atol=1e-4, rtol=1e-4), name


def test_native_region_parallel_building_blocks_match_sequential() -> None:
    # The pure per-region functions + reduce_region_results must reproduce the
    # sequential native backward exactly, even in reverse region order.
    torch.manual_seed(41)
    model = _build_lbi_model()
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)

    logits, cache = model.forward_with_cache(input_ids, native_backward=True)
    loss = logits.square().mean()
    seq = ScanADEngine(
        pullback_provider=NativeInterfacePullbackProvider(),
        local_vjp_provider=NativeLocalVJPProvider(),
    ).backward(model=model, loss=loss, cache=cache)

    # Rebuild the same grad map using the pure building blocks, regions reversed.
    logits2, cache2 = model.forward_with_cache(input_ids, native_backward=True)
    loss2 = logits2.square().mean()
    provider = NativeLocalVJPProvider()
    grad_map = provider.new_grad_map(model)
    states = list(cache2["states"])
    num_regions = len(cache2["region_ranges"])

    provider.store_output_head_grads(model=model, loss=loss2, grad_map=grad_map, cache=cache2)
    g_last = provider.state_adjoint_from_loss(model=model, loss=loss2, state=states[-2], cache=cache2)
    sjt = [
        interface_state_jacobian_t_for_region(model=model, region_cache=rc)
        for rc in cache2["region_caches"]
    ]
    g_last = g_last.to(dtype=sjt[0].dtype)
    g_state_inputs = propagate_state_adjoint_from_last_region_input(sjt, g_last, num_regions=num_regions)

    init_grads, canvas_init = native_initial_backward(
        model=model, state0=states[0], state0_adjoint=g_state_inputs[0], cache=cache2
    )
    for name, grad in init_grads.items():
        grad_map[name] = grad

    # Phase 3 in reverse region order, then reduce (order must not matter).
    results = [
        native_region_backward(
            model=model, loss=loss2, region_index=k,
            num_regions=num_regions, state_adjoints=g_state_inputs, cache=cache2,
        )
        for k in reversed(range(num_regions))
    ]
    reduced = reduce_region_results(results)
    for name, grad in reduced.param_grads.items():
        grad_map[name] = grad

    canvas_total = reduced.canvas_cotangent
    if canvas_init is not None:
        canvas_total = canvas_init if canvas_total is None else canvas_total + canvas_init
    provider._canvas_grad = canvas_total
    provider._shared_grads = dict(reduced.shared_partials)
    provider.store_shared_canvas_grads(model=model, loss=loss2, grad_map=grad_map, cache=cache2)

    assert set(grad_map) == set(seq.grad_map)
    for name, grad in seq.grad_map.items():
        par = grad_map[name]
        assert (par is None) == (grad is None), name
        if grad is not None:
            assert torch.allclose(par.float(), grad.float(), atol=1e-5, rtol=1e-5), name


def test_scan_ad_engine_matches_direct_reference_scan_step() -> None:
    torch.manual_seed(23)
    direct = _build_lbi_model()
    engine_model = _build_lbi_model()
    engine_model.load_state_dict(direct.state_dict())
    input_ids = torch.randint(0, 64, (2, 10), dtype=torch.long)

    direct_logits, direct_cache = direct.forward_with_cache(input_ids)
    direct_loss = direct_logits.square().mean()
    expected = lbi_scan_backward_step(
        direct,
        ce_loss=direct_loss,
        cache=direct_cache,
        pullback_provider=TorchGraphInterfacePullbackProvider(),
    )

    engine_logits, engine_cache = engine_model.forward_with_cache(input_ids)
    engine_loss = engine_logits.square().mean()
    result = ScanADEngine(state_jacobian_mode="graph").backward(model=engine_model, loss=engine_loss, cache=engine_cache)

    _assert_grad_maps_close(result.grad_map, expected.grad_map)
    _assert_grad_maps_close(_parameter_grad_map(engine_model), expected.grad_map)
    assert result.diagnostics["interface_pullback_provider"] == "torch_graph"
    assert result.diagnostics["local_vjp_provider"] == "torch_autograd"


def test_vector_interface_structured_pullbacks_match_autograd() -> None:
    torch.manual_seed(31)
    model = _build_lbi_model()
    input_ids = torch.randint(0, 64, (2, 10), dtype=torch.long)
    _, cache = model.forward_with_cache(input_ids)
    region_cache = cache["region_caches"][1]
    bsz = region_cache.state_out.shape[0]
    basis = 3

    g_state_out = torch.randn(bsz, basis, model.interface_width)
    update_actual = model.interface.apply_update_jacobian_t_to_region_output(
        region_cache=region_cache,
        state_out_cotangent_basis=g_state_out,
    )
    update_expected_cols = []
    for basis_index in range(basis):
        grad = torch.autograd.grad(
            region_cache.state_out,
            region_cache.region_output,
            grad_outputs=g_state_out[:, basis_index, :],
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        update_expected_cols.append(grad.unsqueeze(1))
    update_expected = torch.cat(update_expected_cols, dim=1)
    assert torch.allclose(update_actual["g_region_output"], update_expected, atol=1e-6, rtol=1e-6)

    g_region_input = torch.randn(bsz, basis, *region_cache.region_input.shape[1:])
    decode_actual = model.interface.apply_decode_jacobian_t_to_state_input(
        region_cache=region_cache,
        region_input_cotangent_basis=g_region_input,
    )
    decode_expected_cols = []
    for basis_index in range(basis):
        grad = torch.autograd.grad(
            region_cache.region_input,
            region_cache.state_in,
            grad_outputs=g_region_input[:, basis_index, :, :],
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        decode_expected_cols.append(grad.unsqueeze(1))
    decode_expected = torch.cat(decode_expected_cols, dim=1)
    assert torch.allclose(decode_actual["g_state_input"], decode_expected, atol=1e-6, rtol=1e-6)


def _dense_spec() -> BackboneSpec:
    return BackboneSpec(
        name="transformer", dim=16, layers=2, n_heads=4,
        n_kv_heads=2, d_intermediate=32, attn_head_dim=4,
    )


def test_dense_language_model_uses_canvas_and_readout() -> None:
    from models.dense_language_model import DenseLanguageModel

    torch.manual_seed(123)
    dense = DenseLanguageModel(vocab_size=64, backbone_spec=_dense_spec(), tie_embeddings=False)
    state_keys = set(dense.state_dict())

    assert "canvas.embedding.weight" in state_keys
    assert "readout.lm_head.weight" in state_keys
    assert any(key.startswith("readout.norm") for key in state_keys)

    input_ids = torch.randint(0, 64, (3, 12), dtype=torch.long)
    expected = dense.readout(dense.backbone(dense.canvas(input_ids)), canvas=dense.canvas)
    assert torch.allclose(dense(input_ids), expected, atol=0.0, rtol=0.0)


def test_dense_language_model_tied_readout_uses_canvas_output_weight() -> None:
    from models.dense_language_model import DenseLanguageModel

    torch.manual_seed(456)
    dense = DenseLanguageModel(vocab_size=64, backbone_spec=_dense_spec(), tie_embeddings=True)

    assert dense.readout.lm_head is None
    assert dense.canvas.output_weight() is dense.canvas.embedding.weight
    assert "canvas.embedding.weight" in dense.state_dict()
    assert "readout.lm_head.weight" not in dense.state_dict()

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    hidden = dense.backbone(dense.canvas(input_ids))
    logits_input = dense.readout.norm(hidden)
    expected = torch.nn.functional.linear(logits_input, dense.canvas.output_weight())
    assert torch.allclose(dense(input_ids), expected, atol=0.0, rtol=0.0)
