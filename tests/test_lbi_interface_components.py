from __future__ import annotations

import importlib.util

import pytest
import torch

from backbones.general import BackboneSpec, ReferenceLM, build_backbone_stack
from backbones.mamba3 import Mamba3, Mamba3Block
from models.native_region_interface import NativeRegionInterfaceModel, RegionMessageHead


def _build_model() -> NativeRegionInterfaceModel:
    torch.manual_seed(7)
    return NativeRegionInterfaceModel(
        vocab_size=64,
        region_size=2,
        message_dim=8,
        backbone_spec=BackboneSpec(
            name="transformer",
            dim=16,
            layers=4,
            n_heads=4,
            n_kv_heads=2,
            d_intermediate=32,
            attn_head_dim=4,
        ),
        message_hidden_dim=16,
        message_scale_init=0.5,
    ).to(dtype=torch.float32)


def test_interface_pullback_scan_matches_chain() -> None:
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 12), dtype=torch.long)
    _, cache = model.forward_with_cache(input_ids)
    g_top = torch.randn_like(cache["region_messages"][-1])

    g_chain = model.interface_vjp_chain(cache, g_top)
    g_scan = model.interface_vjp_scan(cache, g_top)

    assert len(g_chain) == len(g_scan)
    for a, b in zip(g_chain, g_scan):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_materialized_interface_matrix_matches_one_step_vjp() -> None:
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 12), dtype=torch.long)
    _, cache = model.forward_with_cache(input_ids)
    mats = model.materialize_interface_pullback_mats(cache)
    region_messages = cache["region_messages"]

    assert len(mats) == len(region_messages) - 1
    for ridx, at in enumerate(mats):
        g_next = torch.randn_like(region_messages[ridx + 1])
        g_ref = torch.autograd.grad(
            region_messages[ridx + 1],
            region_messages[ridx],
            grad_outputs=g_next,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        g_mat = torch.einsum("bij,bj->bi", at, g_next)
        assert torch.allclose(g_ref, g_mat, atol=1e-5, rtol=1e-5)


def test_recomputed_interface_matrix_matches_graph_path() -> None:
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 12), dtype=torch.long)
    _, cache = model.forward_with_cache(input_ids)

    mats_graph = model.materialize_interface_pullback_mats(cache, mode="graph")
    for chunk in (1, 3, 8):
        mats_recompute = model.materialize_interface_pullback_mats(
            cache,
            mode="recompute",
            basis_chunk=chunk,
        )
        assert len(mats_recompute) == len(mats_graph)
        for graph_mat, recompute_mat in zip(mats_graph, mats_recompute):
            assert torch.allclose(graph_mat, recompute_mat, atol=1e-5, rtol=1e-5)


def test_region_forward_caches_capture_region_stages() -> None:
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 12), dtype=torch.long)
    _, cache = model.forward_with_cache(input_ids)

    region_caches = cache["region_caches"]
    assert len(region_caches) == model.n_regions
    for ridx, region_cache in enumerate(region_caches):
        assert torch.allclose(region_cache.message_in, cache["region_messages"][ridx], atol=1e-6, rtol=1e-6)
        assert torch.allclose(region_cache.message_out, cache["region_messages"][ridx + 1], atol=1e-6, rtol=1e-6)
        assert torch.allclose(region_cache.hidden_input, cache["region_hidden_inputs"][ridx], atol=1e-6, rtol=1e-6)
        assert torch.allclose(region_cache.hidden_output, cache["boundaries"][ridx + 1], atol=1e-6, rtol=1e-6)
        assert torch.allclose(region_cache.hidden_bias, model.message_to_hidden[ridx](region_cache.message_in), atol=1e-6, rtol=1e-6)
        assert torch.allclose(region_cache.pooled_hidden, model._pool_hidden(region_cache.hidden_output), atol=1e-6, rtol=1e-6)


def test_region_hidden_inputs_are_message_rebuilt() -> None:
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 12), dtype=torch.long)
    _, cache = model.forward_with_cache(input_ids)

    x_static = cache["boundaries"][0]
    region_hidden_inputs = cache["region_hidden_inputs"]
    region_messages = cache["region_messages"]
    assert len(region_hidden_inputs) == model.n_regions
    for ridx in range(model.n_regions):
        x_expected = x_static + model.message_to_hidden[ridx](region_messages[ridx]).unsqueeze(1)
        assert torch.allclose(region_hidden_inputs[ridx], x_expected, atol=1e-6, rtol=1e-6)


def test_region_message_head_linear_pullback_matches_autograd() -> None:
    model = _build_model()
    torch.manual_seed(11)
    head = RegionMessageHead(5, 7, hidden_dim=0).to(dtype=torch.float32)
    x = torch.randn(3, 5, dtype=torch.float32, requires_grad=True)
    basis = 4
    g_out = torch.randn(3, basis, 7, dtype=torch.float32)

    g_helper = model._region_message_head_input_pullback_matrix(head, x.detach(), g_out)

    y = head(x)
    cols = []
    for pidx in range(basis):
        cols.append(
            torch.autograd.grad(
                y,
                x,
                grad_outputs=g_out[:, pidx, :],
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0].unsqueeze(1)
        )
    g_ref = torch.cat(cols, dim=1)
    assert torch.allclose(g_helper, g_ref, atol=1e-6, rtol=1e-6)



def test_region_message_head_mlp_pullback_matches_autograd() -> None:
    model = _build_model()
    torch.manual_seed(13)
    head = RegionMessageHead(6, 4, hidden_dim=9).to(dtype=torch.float32)
    x = torch.randn(2, 6, dtype=torch.float32, requires_grad=True)
    basis = 3
    g_out = torch.randn(2, basis, 4, dtype=torch.float32)

    g_helper = model._region_message_head_input_pullback_matrix(head, x.detach(), g_out)

    y = head(x)
    cols = []
    for pidx in range(basis):
        cols.append(
            torch.autograd.grad(
                y,
                x,
                grad_outputs=g_out[:, pidx, :],
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0].unsqueeze(1)
        )
    g_ref = torch.cat(cols, dim=1)
    assert torch.allclose(g_helper, g_ref, atol=1e-6, rtol=1e-6)



def test_mean_pool_and_broadcast_pullbacks_match_closed_form() -> None:
    model = _build_model()
    g_pooled = torch.randn(2, 3, 5, dtype=torch.float32)
    seq_len = 7
    g_hidden = model._mean_pool_pullback_matrix(g_pooled, seq_len=seq_len)
    assert g_hidden.shape == (2, 3, seq_len, 5)
    assert torch.allclose(g_hidden.sum(dim=2), g_pooled, atol=1e-6, rtol=1e-6)

    g_hidden_input = torch.randn(2, 3, seq_len, 5, dtype=torch.float32)
    g_bias = model._broadcast_message_pullback_matrix(g_hidden_input)
    assert torch.allclose(g_bias, g_hidden_input.sum(dim=2), atol=1e-6, rtol=1e-6)


def test_region_message_output_pullback_helper_matches_autograd() -> None:
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 12), dtype=torch.long)
    _, cache = model.forward_with_cache(input_ids)
    region_cache = cache["region_caches"][0]
    basis = 3
    g_message_out = torch.randn(region_cache.message_out.shape[0], basis, region_cache.message_out.shape[1])

    helper = model.region_message_output_pullback_to_hidden_output(region_cache, g_message_out)

    g_pre_cols = []
    g_hidden_cols = []
    for pidx in range(basis):
        grad_out = g_message_out[:, pidx, :]
        g_pre_cols.append(
            torch.autograd.grad(
                region_cache.message_out,
                region_cache.pre_norm_message,
                grad_outputs=grad_out,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0].unsqueeze(1)
        )
        g_hidden_cols.append(
            torch.autograd.grad(
                region_cache.message_out,
                region_cache.hidden_output,
                grad_outputs=grad_out,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0].unsqueeze(1)
        )
    g_pre_ref = torch.cat(g_pre_cols, dim=1)
    g_hidden_ref = torch.cat(g_hidden_cols, dim=1)

    assert torch.allclose(helper["g_pre_norm_message"], g_pre_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(helper["g_message_skip"], g_pre_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(helper["g_hidden_output"], g_hidden_ref, atol=1e-5, rtol=1e-5)



def test_region_hidden_input_pullback_helper_matches_autograd() -> None:
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 12), dtype=torch.long)
    _, cache = model.forward_with_cache(input_ids)
    region_cache = cache["region_caches"][0]
    basis = 2
    g_hidden_input = torch.randn(
        region_cache.hidden_input.shape[0],
        basis,
        region_cache.hidden_input.shape[1],
        region_cache.hidden_input.shape[2],
    )

    helper = model.region_hidden_input_pullback_to_message_input(region_cache, g_hidden_input)

    g_msg_cols = []
    for pidx in range(basis):
        grad_out = g_hidden_input[:, pidx, :, :]
        g_msg_cols.append(
            torch.autograd.grad(
                region_cache.hidden_input,
                region_cache.message_in,
                grad_outputs=grad_out,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0].unsqueeze(1)
        )
    g_msg_ref = torch.cat(g_msg_cols, dim=1)

    assert torch.allclose(helper["g_message_input"], g_msg_ref, atol=1e-5, rtol=1e-5)


def test_region_outer_message_input_pullback_combines_components() -> None:
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 12), dtype=torch.long)
    _, cache = model.forward_with_cache(input_ids)
    region_cache = cache["region_caches"][0]
    basis = 2
    g_message_out = torch.randn(region_cache.message_out.shape[0], basis, region_cache.message_out.shape[1])
    g_hidden_input = torch.randn(
        region_cache.hidden_input.shape[0],
        basis,
        region_cache.hidden_input.shape[1],
        region_cache.hidden_input.shape[2],
    )

    combined = model.region_outer_message_input_pullback(region_cache, g_message_out, g_hidden_input)
    tail = model.region_message_output_pullback_to_hidden_output(region_cache, g_message_out)
    head = model.region_hidden_input_pullback_to_message_input(region_cache, g_hidden_input)

    assert torch.allclose(combined["g_pre_norm_message"], tail["g_pre_norm_message"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(combined["g_hidden_output"], tail["g_hidden_output"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(combined["g_message_input"], head["g_message_input"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        combined["g_message_input_total"],
        tail["g_message_skip"] + head["g_message_input"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_scan_from_last_input_seed_matches_manual_chain() -> None:
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 12), dtype=torch.long)
    _, cache = model.forward_with_cache(input_ids)
    mats = model.materialize_interface_pullback_mats(cache)

    g_last = torch.randn_like(cache["region_messages"][-2])
    g_scan = model.interface_vjp_scan_from_last_input_seed(mats, g_last)

    g_manual = [torch.zeros_like(g_last) for _ in range(model.n_regions)]
    g_manual[-1] = g_last
    for ridx in reversed(range(model.n_regions - 1)):
        g_manual[ridx] = torch.einsum("bij,bj->bi", mats[ridx], g_manual[ridx + 1])
    for a, b in zip(g_scan, g_manual):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_compose_pullback_suffix_tree_matches_manual_random() -> None:
    model = NativeRegionInterfaceModel(
        vocab_size=32,
        region_size=1,
        message_dim=4,
        backbone_spec=BackboneSpec(
            name="transformer",
            dim=8,
            layers=5,
            n_heads=2,
            n_kv_heads=1,
            d_intermediate=16,
            attn_head_dim=4,
        ),
        message_hidden_dim=8,
        message_scale_init=0.5,
    ).to(dtype=torch.float32)
    bsz = 3
    rank = model.message_dim
    mats = [torch.randn(bsz, rank, rank) for _ in range(model.n_regions)]

    suffix_tree = model.compose_pullback_suffix(mats)

    eye = torch.eye(rank, dtype=mats[0].dtype).view(1, rank, rank).expand(bsz, rank, rank).clone()
    suffix_manual = [eye.clone() for _ in range(model.n_regions + 1)]
    for ridx in reversed(range(model.n_regions)):
        suffix_manual[ridx] = torch.matmul(mats[ridx], suffix_manual[ridx + 1])

    assert len(suffix_tree) == len(suffix_manual)
    for a, b in zip(suffix_tree, suffix_manual):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_message_ablation_zero_all_blocks_message_carry() -> None:
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 12), dtype=torch.long)

    _, zero_all_cache = model.forward_with_cache(input_ids, message_ablation="zero_all")
    for m in zero_all_cache["region_messages"]:
        assert torch.count_nonzero(m) == 0


def test_message_ablation_noise_and_mask_modify_messages() -> None:
    model = _build_model()
    input_ids = torch.randint(0, 64, (2, 12), dtype=torch.long)
    _, baseline_cache = model.forward_with_cache(input_ids)

    noise_gen = torch.Generator(device=input_ids.device)
    noise_gen.manual_seed(123)
    _, noise_cache = model.forward_with_cache(
        input_ids,
        message_ablation="noise",
        message_noise_std=0.25,
        ablation_generator=noise_gen,
    )
    assert any(
        not torch.allclose(base, noised)
        for base, noised in zip(baseline_cache["region_messages"], noise_cache["region_messages"])
    )

    mask_gen = torch.Generator(device=input_ids.device)
    mask_gen.manual_seed(321)
    _, mask_cache = model.forward_with_cache(
        input_ids,
        message_ablation="mask",
        message_mask_keep_prob=0.5,
        ablation_generator=mask_gen,
    )
    assert any(
        torch.count_nonzero(masked == 0).item() > 0
        for masked in mask_cache["region_messages"]
    )


def test_build_backbone_blocks_mamba2_smoke() -> None:
    if importlib.util.find_spec("triton") is None:
        return
    if importlib.util.find_spec("einops") is None:
        return
    spec = BackboneSpec(
        name="mamba2",
        dim=16,
        layers=2,
        d_state=4,
        d_conv=4,
        expand=2,
        headdim=8,
        ngroups=1,
        chunk_size=16,
        use_mem_eff_path=True,
    )
    stack = build_backbone_stack(spec)
    assert len(stack.blocks) == spec.layers
    assert type(stack.blocks[0]).__name__ == "Mamba2Block"


def test_build_backbone_stack_transformer_smoke() -> None:
    spec = BackboneSpec(
        name="transformer",
        dim=32,
        layers=2,
        n_heads=4,
        n_kv_heads=2,
        mlp_ratio=0.0,
        d_intermediate=0,
        attn_head_dim=10,
        softmax_scale=0.5,
        rope_interleaved=True,
        d_conv=2,
    )
    stack = build_backbone_stack(spec)
    x = torch.randn(2, 16, 32)
    y = stack(x)
    assert y.shape == x.shape

    ref = ReferenceLM(vocab_size=64, backbone_spec=spec)
    ids = torch.randint(0, 64, (2, 16), dtype=torch.long)
    logits = ref(ids)
    assert logits.shape == (2, 16, 64)

    native = NativeRegionInterfaceModel(
        vocab_size=64,
        region_size=1,
        message_dim=8,
        backbone_spec=BackboneSpec(
            name="transformer",
            dim=32,
            layers=2,
            n_heads=4,
            n_kv_heads=2,
            mlp_ratio=0.0,
            d_intermediate=0,
            attn_head_dim=10,
            softmax_scale=0.5,
            rope_interleaved=True,
            d_conv=2,
        ),
        message_hidden_dim=16,
    ).to(dtype=torch.float32)
    native_logits, cache = native.forward_with_cache(ids)
    assert native_logits.shape == (2, 16, 64)
    assert len(cache["region_messages"]) == native.n_regions + 1


def test_build_backbone_stack_hybrid_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.skip("hybrid mamba2/transformer smoke currently requires CUDA")
    device = torch.device("cuda")
    spec = BackboneSpec(
        name="hybrid",
        dim=32,
        layers=4,
        d_state=8,
        d_conv=4,
        expand=2,
        headdim=8,
        ngroups=1,
        chunk_size=16,
        use_mem_eff_path=False,
        n_heads=4,
        n_kv_heads=2,
        d_intermediate=128,
        attn_head_dim=8,
        layer_types=("mamba2", "transformer", "mamba2", "transformer"),
    )
    stack = build_backbone_stack(spec)
    stack = stack.to(device=device, dtype=torch.float32)
    x = torch.randn(2, 16, 32, device=device)
    y = stack(x)
    assert y.shape == x.shape

    ref = ReferenceLM(vocab_size=64, backbone_spec=spec).to(device=device, dtype=torch.float32)
    ids = torch.randint(0, 64, (2, 16), dtype=torch.long, device=device)
    logits = ref(ids)
    assert logits.shape == (2, 16, 64)

    native = NativeRegionInterfaceModel(
        vocab_size=64,
        region_size=2,
        message_dim=8,
        backbone_spec=spec,
        message_hidden_dim=16,
    ).to(device=device, dtype=torch.float32)
    native_logits, cache = native.forward_with_cache(ids)
    assert native_logits.shape == (2, 16, 64)
    assert len(cache["region_messages"]) == native.n_regions + 1


def test_hybrid_backbone_rejects_unsupported_layers() -> None:
    with pytest.raises(ValueError, match="hybrid layer_types must be drawn from"):
        BackboneSpec(
            name="hybrid",
            dim=32,
            layers=2,
            d_state=8,
            d_conv=4,
            expand=2,
            headdim=8,
            n_heads=4,
            layer_types=("unsupported", "transformer"),
        ).validate()


def test_mamba3_forward_with_cache_and_input_pullback_match_autograd() -> None:
    if importlib.util.find_spec("triton") is None:
        return
    if importlib.util.find_spec("einops") is None:
        return
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    mixer = Mamba3(
        d_model=32,
        d_state=16,
        expand=2,
        headdim=16,
        ngroups=1,
        chunk_size=16,
        device=device,
        dtype=torch.bfloat16,
    ).to(device=device, dtype=torch.bfloat16)
    u = torch.randn(2, 16, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    out, cache = mixer.forward_with_cache(u)
    assert out.shape == u.shape
    assert cache.output.shape == out.shape

    basis = 2
    g_out = torch.randn(2, basis, 16, 32, device=device, dtype=torch.bfloat16)

    cols = []
    for pidx in range(basis):
        cols.append(
            torch.autograd.grad(
                out,
                u,
                grad_outputs=g_out[:, pidx, :, :],
                retain_graph=pidx + 1 < basis,
                create_graph=False,
                allow_unused=False,
            )[0].unsqueeze(1)
        )
    g_ref = torch.cat(cols, dim=1)

    u2 = u.detach().clone().requires_grad_(True)
    out2, cache2 = mixer.forward_with_cache(u2)
    g_helper = mixer.input_pullback_matrix(cache2, g_out)
    assert out2.shape == out.shape
    assert torch.allclose(g_helper.float(), g_ref.float(), atol=5e-2, rtol=5e-2)



def test_mamba3_block_forward_with_cache_and_input_pullback_match_autograd() -> None:
    if importlib.util.find_spec("triton") is None:
        return
    if importlib.util.find_spec("einops") is None:
        return
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    block = Mamba3Block(
        dim=32,
        d_state=16,
        expand=2,
        headdim=16,
        ngroups=1,
        chunk_size=16,
        device=device,
        dtype=torch.bfloat16,
    ).to(device=device, dtype=torch.bfloat16)
    hidden = torch.randn(2, 16, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(2, 16, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    hidden_out, residual_out, cache = block.forward_with_cache(hidden, residual=residual)
    assert hidden_out.shape == hidden.shape
    assert residual_out.shape == hidden.shape
    assert cache.hidden_output.shape == hidden_out.shape

    basis = 2
    g_hidden_out = torch.randn(2, basis, 16, 32, device=device, dtype=torch.bfloat16)
    g_residual_out = torch.randn(2, basis, 16, 32, device=device, dtype=torch.bfloat16)

    hidden_cols = []
    residual_cols = []
    for pidx in range(basis):
        grads = torch.autograd.grad(
            (hidden_out, residual_out),
            (hidden, residual),
            grad_outputs=(g_hidden_out[:, pidx, :, :], g_residual_out[:, pidx, :, :]),
            retain_graph=pidx + 1 < basis,
            create_graph=False,
            allow_unused=False,
        )
        hidden_cols.append(grads[0].unsqueeze(1))
        residual_cols.append(grads[1].unsqueeze(1))
    g_hidden_ref = torch.cat(hidden_cols, dim=1)
    g_residual_ref = torch.cat(residual_cols, dim=1)

    hidden2 = hidden.detach().clone().requires_grad_(True)
    residual2 = residual.detach().clone().requires_grad_(True)
    _, _, cache2 = block.forward_with_cache(hidden2, residual=residual2)
    g_hidden_helper, g_residual_helper = block.input_pullback_matrix(cache2, g_hidden_out, g_residual_out)
    assert torch.allclose(g_hidden_helper.float(), g_hidden_ref.float(), atol=5e-2, rtol=5e-2)
    assert g_residual_helper is not None
    assert torch.allclose(g_residual_helper.float(), g_residual_ref.float(), atol=5e-2, rtol=5e-2)


def test_build_backbone_stack_mamba3_forward_smoke() -> None:
    if importlib.util.find_spec("triton") is None:
        return
    if importlib.util.find_spec("einops") is None:
        return
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    spec = BackboneSpec(
        name="mamba3",
        dim=16,
        layers=1,
        d_state=16,
        expand=2,
        headdim=8,
        ngroups=1,
        chunk_size=16,
    )
    stack = build_backbone_stack(spec).to(device=device, dtype=torch.float32)
    x = torch.randn(2, 16, 16, device=device)
    y = stack(x)
    assert y.shape == x.shape

    ref = ReferenceLM(vocab_size=64, backbone_spec=spec).to(device=device, dtype=torch.float32)
    ids = torch.randint(0, 64, (2, 16), device=device)
    logits = ref(ids)
    assert logits.shape == (2, 16, 64)

    native = NativeRegionInterfaceModel(
        vocab_size=64,
        region_size=1,
        message_dim=8,
        backbone_spec=BackboneSpec(
            name="mamba3",
            dim=16,
            layers=2,
            d_state=16,
            expand=2,
            headdim=8,
            ngroups=1,
            chunk_size=16,
        ),
        message_hidden_dim=16,
    ).to(device=device, dtype=torch.float32)
    native_logits, cache = native.forward_with_cache(ids)
    assert native_logits.shape == (2, 16, 64)
    assert len(cache["region_messages"]) == native.n_regions + 1


def test_reference_lm_mamba2_instantiation_smoke() -> None:
    if importlib.util.find_spec("triton") is None:
        return
    if importlib.util.find_spec("einops") is None:
        return
    model = ReferenceLM(
        vocab_size=64,
        backbone_spec=BackboneSpec(
            name="mamba2",
            dim=16,
            layers=2,
            d_state=4,
            d_conv=4,
            expand=2,
            headdim=8,
            ngroups=1,
            chunk_size=16,
            use_mem_eff_path=True,
        ),
    )
    assert len(model.blocks) == 2
    assert type(model.blocks[0]).__name__ == "Mamba2Block"
