from __future__ import annotations

import pytest
import torch

from backbones.general import BackboneSpec
from interfaces import VectorMLPInterface
from legacy import migrate_lbi1_state_dict, migrate_lbi1_state_key
from backward import (
    materialize_interface_state_jacobian_t_graph,
    materialize_interface_state_jacobian_t_recompute,
    propagate_state_adjoint_from_last_region_input,
)
from backward import lbi_reference_scan_backward_step
from models.lbi_language_model import LBILanguageModel
from legacy.native_region_interface import NativeRegionInterfaceModel
from legacy.backward import _native_backward_step


def _spec() -> BackboneSpec:
    return BackboneSpec(
        name="transformer",
        dim=16,
        layers=3,
        n_heads=4,
        n_kv_heads=2,
        d_intermediate=32,
        attn_head_dim=4,
    )


def _build_legacy() -> NativeRegionInterfaceModel:
    torch.manual_seed(53)
    return NativeRegionInterfaceModel(
        vocab_size=64,
        region_size=1,
        message_dim=8,
        backbone_spec=_spec(),
        message_hidden_dim=16,
        message_scale_init=0.5,
        tie_embeddings=False,
    ).to(dtype=torch.float32)


def _build_owned() -> LBILanguageModel:
    torch.manual_seed(89)
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
        backbone_spec=_spec(),
        interface=interface,
        tie_embeddings=False,
    ).to(dtype=torch.float32)


def _build_migrated_pair() -> tuple[NativeRegionInterfaceModel, LBILanguageModel]:
    legacy = _build_legacy()
    owned = _build_owned()
    owned.load_state_dict(migrate_lbi1_state_dict(legacy.state_dict()), strict=True)
    return legacy, owned


def test_owned_vector_interface_has_stable_spec_and_parameter_schema() -> None:
    legacy, owned = _build_migrated_pair()
    owned_keys = set(owned.state_dict())

    assert owned.interface.spec.state_shape == (8,)
    assert owned.interface.spec.state_flat_dim == 8
    assert owned.interface.spec.region_condition_dim == 16
    assert "interface.initial_encoder.net.0.weight" in owned_keys
    assert "interface.decoders.0.net.0.weight" in owned_keys
    assert "interface.encoders.0.net.0.weight" in owned_keys
    assert "interface.norms.0.weight" in owned_keys
    assert "interface.update_scale" in owned_keys
    assert "input_to_message.net.0.weight" not in owned_keys
    assert "message_alpha" not in owned_keys
    assert set(migrate_lbi1_state_dict(legacy.state_dict())) == owned_keys


def test_migrated_owned_model_matches_lbi1_forward_states_and_caches() -> None:
    legacy, owned = _build_migrated_pair()
    input_ids = torch.randint(0, 64, (2, 10), dtype=torch.long)

    legacy_logits, legacy_cache = legacy.forward_with_cache(input_ids)
    owned_logits, owned_cache = owned.forward_with_cache(input_ids)

    assert torch.allclose(legacy_logits, owned_logits, atol=1e-6, rtol=1e-6)
    for old_state, new_state in zip(legacy_cache["region_messages"], owned_cache["states"]):
        assert torch.allclose(old_state, new_state, atol=1e-6, rtol=1e-6)
    for old_input, new_input in zip(legacy_cache["region_hidden_inputs"], owned_cache["region_inputs"]):
        assert torch.allclose(old_input, new_input, atol=1e-6, rtol=1e-6)
    for old_output, new_output in zip(legacy_cache["boundaries"][1:], owned_cache["region_outputs"]):
        assert torch.allclose(old_output, new_output, atol=1e-6, rtol=1e-6)


def test_migrated_owned_model_matches_lbi1_autograd_for_all_parameters() -> None:
    legacy, owned = _build_migrated_pair()
    input_ids = torch.randint(0, 64, (2, 10), dtype=torch.long)

    legacy_loss = legacy(input_ids).square().mean()
    owned_loss = owned(input_ids).square().mean()
    assert torch.allclose(legacy_loss, owned_loss, atol=1e-7, rtol=1e-7)

    legacy_named = dict(legacy.named_parameters())
    owned_named = dict(owned.named_parameters())
    legacy_grads = torch.autograd.grad(legacy_loss, tuple(legacy_named.values()), allow_unused=True)
    owned_grads = torch.autograd.grad(owned_loss, tuple(owned_named.values()), allow_unused=True)
    owned_grad_by_name = dict(zip(owned_named, owned_grads))

    for legacy_name, legacy_grad in zip(legacy_named, legacy_grads):
        owned_name = migrate_lbi1_state_key(legacy_name)
        assert owned_name in owned_grad_by_name
        owned_grad = owned_grad_by_name[owned_name]
        assert (legacy_grad is None) == (owned_grad is None)
        if legacy_grad is not None and owned_grad is not None:
            assert torch.allclose(legacy_grad, owned_grad, atol=1e-6, rtol=1e-6), owned_name


def test_owned_model_rejects_interface_with_incompatible_region_count() -> None:
    interface = VectorMLPInterface(
        feature_dim=16,
        num_regions=2,
        interface_width=8,
        interface_map_hidden_dim=16,
    )
    with pytest.raises(ValueError, match="region count"):
        LBILanguageModel(
            vocab_size=64,
            layers_per_region=1,
            backbone_spec=_spec(),
            interface=interface,
        )



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


def _autograd_grad_map(model: torch.nn.Module, loss: torch.Tensor) -> dict[str, torch.Tensor | None]:
    named = dict(model.named_parameters())
    grads = torch.autograd.grad(loss, tuple(named.values()), allow_unused=True)
    return {name: None if grad is None else grad.detach().clone() for name, grad in zip(named, grads)}


def test_owned_model_state_jacobians_t_match_lbi1_graph_and_recompute() -> None:
    legacy, owned = _build_migrated_pair()
    input_ids = torch.randint(0, 64, (2, 10), dtype=torch.long)
    _, legacy_cache = legacy.forward_with_cache(input_ids)
    _, owned_cache = owned.forward_with_cache(input_ids)

    for mode, basis_chunk in (("graph", 1), ("recompute", 3)):
        legacy_state_jacobians_t = legacy.materialize_interface_pullback_mats(
            legacy_cache,
            mode=mode,
            basis_chunk=basis_chunk,
        )
        if mode == "graph":
            owned_state_jacobians_t = materialize_interface_state_jacobian_t_graph(model=owned, cache=owned_cache)
        else:
            owned_state_jacobians_t = materialize_interface_state_jacobian_t_recompute(
                model=owned,
                cache=owned_cache,
                basis_chunk=basis_chunk,
            )
        assert len(legacy_state_jacobians_t) == len(owned_state_jacobians_t)
        for legacy_jacobian_t, owned_jacobian_t in zip(legacy_state_jacobians_t, owned_state_jacobians_t):
            assert torch.allclose(legacy_jacobian_t, owned_jacobian_t, atol=1e-6, rtol=1e-6)


def test_owned_model_scan_from_last_input_seed_matches_lbi1() -> None:
    legacy, owned = _build_migrated_pair()
    input_ids = torch.randint(0, 64, (2, 10), dtype=torch.long)
    _, legacy_cache = legacy.forward_with_cache(input_ids)
    _, owned_cache = owned.forward_with_cache(input_ids)
    legacy_state_jacobians_t = legacy.materialize_interface_pullback_mats(legacy_cache, mode="recompute", basis_chunk=3)
    owned_state_jacobians_t = materialize_interface_state_jacobian_t_recompute(model=owned, cache=owned_cache, basis_chunk=3)

    g_last = torch.randn_like(legacy_cache["region_messages"][-2])
    legacy_states = legacy.interface_vjp_scan_from_last_input_seed(legacy_state_jacobians_t, g_last)
    owned_states = propagate_state_adjoint_from_last_region_input(
        owned_state_jacobians_t,
        g_last,
        num_regions=owned.num_regions,
    )

    assert len(legacy_states) == len(owned_states)
    for legacy_grad, owned_grad in zip(legacy_states, owned_states):
        assert torch.allclose(legacy_grad, owned_grad, atol=1e-6, rtol=1e-6)


def test_owned_reference_scan_backward_matches_autograd_and_lbi1_scan() -> None:
    legacy_scan, owned_scan = _build_migrated_pair()
    _, owned_autograd = _build_migrated_pair()
    input_ids = torch.randint(0, 64, (2, 10), dtype=torch.long)

    legacy_logits, legacy_cache = legacy_scan.forward_with_cache(input_ids)
    legacy_loss = legacy_logits.square().mean()
    legacy_backward = _native_backward_step(
        legacy_scan,
        ce_loss=legacy_loss,
        cache=legacy_cache,
        interface_jacobian_mode="recompute",
        jacobian_basis_chunk=3,
    )

    owned_logits, owned_cache = owned_scan.forward_with_cache(input_ids)
    owned_loss = owned_logits.square().mean()
    owned_backward = lbi_reference_scan_backward_step(
        owned_scan,
        ce_loss=owned_loss,
        cache=owned_cache,
        state_jacobian_mode="recompute",
        state_jacobian_basis_chunk=3,
    )

    autograd_loss = owned_autograd(input_ids).square().mean()
    autograd_grads = _autograd_grad_map(owned_autograd, autograd_loss)

    migrated_legacy_grads = {
        migrate_lbi1_state_key(name): None if grad is None else grad.detach().clone()
        for name, grad in legacy_backward.grad_map.items()
    }
    _assert_grad_maps_close(owned_backward.grad_map, migrated_legacy_grads, atol=1e-6, rtol=1e-6)
    _assert_grad_maps_close(owned_backward.grad_map, autograd_grads, atol=1e-6, rtol=1e-6)
    assert owned_backward.interface_scan_rms < 1e-8
    assert legacy_backward.interface_scan_rms < 1e-8


def _formula_model() -> NativeRegionInterfaceModel:
    torch.manual_seed(41)
    return NativeRegionInterfaceModel(
        vocab_size=64,
        region_size=1,
        message_dim=8,
        backbone_spec=BackboneSpec(
            name="transformer", dim=16, layers=3, n_heads=4, n_kv_heads=2,
            d_intermediate=32, attn_head_dim=4,
        ),
        message_hidden_dim=16,
        message_scale_init=0.5,
    ).to(dtype=torch.float32)


def _legacy_formula_forward(
    model: NativeRegionInterfaceModel,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """The LBI-1 paper formula written out directly; anchors the legacy
    oracle that the owned-model parity tests above compare against."""
    x_static = model.embedding(input_ids)
    message = model.input_to_message(model._pool_hidden(x_static))
    messages = [message]
    for region_idx, (start, end) in enumerate(model.region_ranges):
        hidden_bias = model.message_to_hidden[region_idx](message)
        hidden_input = x_static + hidden_bias.unsqueeze(1)
        hidden_output = model.backbone.forward_range(hidden_input, start, end)
        pooled_hidden = model._pool_hidden(hidden_output)
        delta_message = model.hidden_to_message[region_idx](pooled_hidden)
        alpha = torch.tanh(model.message_alpha[region_idx])
        message = model.message_norm[region_idx](message + alpha * delta_message)
        messages.append(message)
    logits_input = model.norm(hidden_output)
    if logits_input.dtype != model.lm_head.weight.dtype:
        logits_input = logits_input.to(dtype=model.lm_head.weight.dtype)
    return model.lm_head(logits_input), messages


def test_legacy_model_matches_paper_formula_without_new_state_keys() -> None:
    model = _formula_model()
    input_ids = torch.randint(0, 64, (2, 10), dtype=torch.long)

    logits, cache = model.forward_with_cache(input_ids)
    reference_logits, reference_messages = _legacy_formula_forward(model, input_ids)

    assert torch.allclose(logits, reference_logits, atol=1e-6, rtol=1e-6)
    for actual, expected in zip(cache["region_messages"], reference_messages):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert model.interface.state_dict() == {}
    assert not any(key.startswith("interface.") for key in model.state_dict())
    assert "input_to_message.net.0.weight" in model.state_dict()
    assert "message_to_hidden.0.net.0.weight" in model.state_dict()
    assert "hidden_to_message.0.net.0.weight" in model.state_dict()

    reloaded = _formula_model()
    reloaded.load_state_dict(model.state_dict(), strict=True)


def test_legacy_model_gradients_match_paper_formula() -> None:
    model = _formula_model()
    input_ids = torch.randint(0, 64, (2, 10), dtype=torch.long)
    params = tuple(model.parameters())

    logits, _ = model.forward_with_cache(input_ids)
    grads = torch.autograd.grad(logits.square().mean(), params, allow_unused=True)

    reference_logits, _ = _legacy_formula_forward(model, input_ids)
    reference_grads = torch.autograd.grad(reference_logits.square().mean(), params, allow_unused=True)

    for actual, expected in zip(grads, reference_grads):
        assert (actual is None) == (expected is None)
        if actual is not None and expected is not None:
            assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
