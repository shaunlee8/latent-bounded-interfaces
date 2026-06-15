from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from backends import CudaTransformerLowering
from cuda.transformer import attention_input_pullback_basis


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for attention pullback kernel tests")


def _attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float, causal: bool) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
    )


def _autograd_pullback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_cotangent_basis: torch.Tensor,
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq_cols: list[torch.Tensor] = []
    dk_cols: list[torch.Tensor] = []
    dv_cols: list[torch.Tensor] = []
    for basis_index in range(output_cotangent_basis.shape[1]):
        q_req = q.detach().clone().requires_grad_(True)
        k_req = k.detach().clone().requires_grad_(True)
        v_req = v.detach().clone().requires_grad_(True)
        out = _attention_forward(q_req, k_req, v_req, scale=scale, causal=causal)
        dq, dk, dv = torch.autograd.grad(
            out,
            (q_req, k_req, v_req),
            grad_outputs=output_cotangent_basis[:, basis_index],
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        dq_cols.append(dq.unsqueeze(1))
        dk_cols.append(dk.unsqueeze(1))
        dv_cols.append(dv.unsqueeze(1))
    return torch.cat(dq_cols, dim=1), torch.cat(dk_cols, dim=1), torch.cat(dv_cols, dim=1)



@pytest.mark.parametrize("causal", [True, False])
def test_attention_input_pullback_basis_torch_formula_matches_autograd(causal: bool) -> None:
    torch.manual_seed(101)
    q = torch.randn(2, 3, 5, 4, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output_cotangent_basis = torch.randn(2, 2, 3, 5, 4, device="cuda", dtype=torch.float32)
    scale = 1.0 / math.sqrt(q.shape[-1])

    actual = attention_input_pullback_basis(
        q,
        k,
        v,
        output_cotangent_basis,
        softmax_scale=scale,
        causal=causal,
        use_cuda=False,
    )
    expected = _autograd_pullback(q, k, v, output_cotangent_basis, scale=scale, causal=causal)

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.allclose(actual_tensor, expected_tensor, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("causal", [True, False])
def test_attention_input_pullback_basis_simple_cuda_matches_autograd(causal: bool) -> None:
    torch.manual_seed(103)
    q = torch.randn(1, 2, 4, 3, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output_cotangent_basis = torch.randn(1, 3, 2, 4, 3, device="cuda", dtype=torch.float32)
    scale = 1.0 / math.sqrt(q.shape[-1])

    try:
        actual = attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=scale,
            causal=causal,
            use_cuda=True,
            kernel_mode="simple",
        )
    except (AttributeError, TypeError) as exc:
        pytest.skip(f"transformer_lbi_cuda was built without attention pullback binding: {exc}")
    expected = _autograd_pullback(q, k, v, output_cotangent_basis, scale=scale, causal=causal)

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.shape == output_cotangent_basis.shape
        assert torch.allclose(actual_tensor, expected_tensor, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("causal", [True, False])
def test_attention_input_pullback_basis_fa2_loop_matches_autograd(causal: bool) -> None:
    torch.manual_seed(117)
    q = torch.randn(1, 2, 4, 16, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output_cotangent_basis = torch.randn(1, 3, 2, 4, 16, device="cuda", dtype=torch.float32)
    scale = 1.0 / math.sqrt(q.shape[-1])

    actual = attention_input_pullback_basis(
        q,
        k,
        v,
        output_cotangent_basis,
        softmax_scale=scale,
        causal=causal,
        use_cuda=True,
        kernel_mode="fa2_loop",
    )
    expected = _autograd_pullback(q, k, v, output_cotangent_basis, scale=scale, causal=causal)

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.shape == output_cotangent_basis.shape
        assert torch.allclose(actual_tensor, expected_tensor, atol=2e-5, rtol=2e-5)


def test_attention_input_pullback_basis_fa2_p1_matches_autograd_and_rejects_p_gt_1() -> None:
    torch.manual_seed(119)
    q = torch.randn(1, 2, 4, 16, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output_cotangent_basis = torch.randn(1, 1, 2, 4, 16, device="cuda", dtype=torch.float32)
    scale = 1.0 / math.sqrt(q.shape[-1])

    actual = attention_input_pullback_basis(
        q,
        k,
        v,
        output_cotangent_basis,
        softmax_scale=scale,
        causal=True,
        use_cuda=True,
        kernel_mode="fa2_p1",
    )
    expected = _autograd_pullback(q, k, v, output_cotangent_basis, scale=scale, causal=True)

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.shape == output_cotangent_basis.shape
        assert torch.allclose(actual_tensor, expected_tensor, atol=2e-5, rtol=2e-5)

    with pytest.raises(ValueError, match="fa2_p1"):
        attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis.expand(-1, 2, -1, -1, -1).contiguous(),
            softmax_scale=scale,
            causal=True,
            use_cuda=True,
            kernel_mode="fa2_p1",
        )


@pytest.mark.parametrize("causal", [True, False])
def test_attention_input_pullback_basis_fa2_p1_fused_fp16_matches_autograd(causal: bool) -> None:
    torch.manual_seed(121)
    q = torch.randn(1, 2, 16, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output_cotangent_basis = torch.randn(1, 1, 2, 16, 64, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(q.shape[-1])

    actual = attention_input_pullback_basis(
        q,
        k,
        v,
        output_cotangent_basis,
        softmax_scale=scale,
        causal=causal,
        use_cuda=True,
        kernel_mode="fa2_p1",
    )
    expected = _autograd_pullback(q, k, v, output_cotangent_basis, scale=scale, causal=causal)

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.shape == output_cotangent_basis.shape
        assert actual_tensor.dtype == torch.float16
        assert torch.allclose(actual_tensor, expected_tensor, atol=2e-2, rtol=2e-2)


def test_attention_input_pullback_basis_fa2_mma_p1_matches_fused_fa2_p1_when_available() -> None:
    torch.manual_seed(129)
    q = torch.randn(1, 2, 16, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output_cotangent_basis = torch.randn(1, 1, 2, 16, 64, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(q.shape[-1])

    actual = attention_input_pullback_basis(
        q,
        k,
        v,
        output_cotangent_basis,
        softmax_scale=scale,
        causal=True,
        use_cuda=True,
        kernel_mode="fa2_mma_p1",
    )
    expected = attention_input_pullback_basis(
        q,
        k,
        v,
        output_cotangent_basis,
        softmax_scale=scale,
        causal=True,
        use_cuda=True,
        kernel_mode="fa2_p1",
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.shape == output_cotangent_basis.shape
        assert actual_tensor.dtype == torch.float16
        assert torch.allclose(actual_tensor, expected_tensor, atol=3e-2, rtol=3e-2)


def test_attention_input_pullback_basis_fa2_mma_pblock4_matches_fa2_loop_when_available() -> None:
    torch.manual_seed(131)
    q = torch.randn(1, 2, 16, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output_cotangent_basis = torch.randn(1, 4, 2, 16, 64, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(q.shape[-1])

    actual = attention_input_pullback_basis(
        q,
        k,
        v,
        output_cotangent_basis,
        softmax_scale=scale,
        causal=True,
        use_cuda=True,
        kernel_mode="fa2_mma_pblock4",
    )
    expected = attention_input_pullback_basis(
        q,
        k,
        v,
        output_cotangent_basis,
        softmax_scale=scale,
        causal=True,
        use_cuda=True,
        kernel_mode="fa2_loop",
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.shape == output_cotangent_basis.shape
        assert actual_tensor.dtype == torch.float16
        assert torch.allclose(actual_tensor, expected_tensor, atol=3e-2, rtol=3e-2)


def test_attention_input_pullback_basis_fa2_mma_pblock4_rejects_p_gt_4() -> None:
    torch.manual_seed(133)
    q = torch.randn(1, 2, 16, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output_cotangent_basis = torch.randn(1, 5, 2, 16, 64, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(q.shape[-1])

    with pytest.raises(ValueError, match="fa2_mma_pblock4"):
        attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=scale,
            causal=True,
            use_cuda=True,
            kernel_mode="fa2_mma_pblock4",
        )


def test_cuda_transformer_lowering_attention_pullback_matches_autograd() -> None:
    torch.manual_seed(107)
    q = torch.randn(1, 2, 4, 3, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output_cotangent_basis = torch.randn(1, 2, 2, 4, 3, device="cuda", dtype=torch.float32)
    scale = 1.0 / math.sqrt(q.shape[-1])

    lowering = CudaTransformerLowering()
    actual = lowering.attention_input_pullback_basis(
        q=q,
        k=k,
        v=v,
        output_cotangent_basis=output_cotangent_basis,
        softmax_scale=scale,
        causal=True,
        kernel_mode="simple",
    )
    expected = _autograd_pullback(q, k, v, output_cotangent_basis, scale=scale, causal=True)

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.allclose(actual_tensor, expected_tensor, atol=2e-5, rtol=2e-5)
