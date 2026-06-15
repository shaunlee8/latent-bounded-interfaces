from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import transformer_lbi_cuda  # type: ignore
except Exception:  # pragma: no cover - optional extension
    transformer_lbi_cuda = None  # type: ignore[assignment]


def _torch_attention_input_pullback_basis(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_cotangent_basis: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, and v must have matching shape [B, H, T, Dh]")
    if output_cotangent_basis.dim() != 5:
        raise ValueError("output_cotangent_basis must have shape [B, P, H, T, Dh]")
    if output_cotangent_basis.shape[0] != q.shape[0] or output_cotangent_basis.shape[2:] != q.shape[1:]:
        raise ValueError("output_cotangent_basis must have shape [B, P, H, T, Dh] matching q/k/v")

    _, _, seqlen, _ = q.shape
    scores = torch.matmul(q, k.transpose(-1, -2)) * float(softmax_scale)
    if causal:
        mask = torch.ones((seqlen, seqlen), device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask.view(1, 1, seqlen, seqlen), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    d_out = output_cotangent_basis
    dv = torch.einsum("bhij,bphid->bphjd", probs, d_out)
    d_probs = torch.einsum("bphid,bhjd->bphij", d_out, v)
    row_dot = (d_probs * probs.unsqueeze(1)).sum(dim=-1, keepdim=True)
    d_scores = probs.unsqueeze(1) * (d_probs - row_dot)
    dq = torch.einsum("bphij,bhjd->bphid", d_scores, k) * float(softmax_scale)
    dk = torch.einsum("bphij,bhid->bphjd", d_scores, q) * float(softmax_scale)
    return dq, dk, dv


def _sdpa_autograd_attention_input_pullback_basis(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_cotangent_basis: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool = True,
    require_single_basis: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, and v must have matching shape [B, H, T, Dh]")
    if output_cotangent_basis.dim() != 5:
        raise ValueError("output_cotangent_basis must have shape [B, P, H, T, Dh]")
    if output_cotangent_basis.shape[0] != q.shape[0] or output_cotangent_basis.shape[2:] != q.shape[1:]:
        raise ValueError("output_cotangent_basis must have shape [B, P, H, T, Dh] matching q/k/v")
    if require_single_basis and output_cotangent_basis.shape[1] != 1:
        raise ValueError("kernel_mode='fa2_p1' requires output_cotangent_basis.shape[1] == 1")

    dq_cols: list[torch.Tensor] = []
    dk_cols: list[torch.Tensor] = []
    dv_cols: list[torch.Tensor] = []
    for basis_index in range(output_cotangent_basis.shape[1]):
        q_req = q.detach().clone().requires_grad_(True)
        k_req = k.detach().clone().requires_grad_(True)
        v_req = v.detach().clone().requires_grad_(True)
        out = F.scaled_dot_product_attention(
            q_req,
            k_req,
            v_req,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=float(softmax_scale),
        )
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


def _fused_fa2_p1_attention_input_pullback_basis(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_cotangent_basis: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if output_cotangent_basis.shape[1] != 1:
        raise ValueError("kernel_mode='fa2_p1' requires output_cotangent_basis.shape[1] == 1")
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        return _sdpa_autograd_attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=float(softmax_scale),
            causal=causal,
            require_single_basis=True,
        )
    try:
        out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, rng_state, _unused, _debug_mask = (
            torch.ops.aten._scaled_dot_product_flash_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                0.0,
                bool(causal),
                False,
                scale=float(softmax_scale),
            )
        )
        dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            output_cotangent_basis[:, 0].contiguous(),
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            out,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            0.0,
            bool(causal),
            rng_state[0],
            rng_state[1],
            scale=float(softmax_scale),
        )
    except RuntimeError:
        return _sdpa_autograd_attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=float(softmax_scale),
            causal=causal,
            require_single_basis=True,
        )
    return dq.unsqueeze(1), dk.unsqueeze(1), dv.unsqueeze(1)


def _mma_fa2_p1_attention_input_pullback_basis(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_cotangent_basis: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if output_cotangent_basis.shape[1] != 1:
        raise ValueError("kernel_mode='fa2_mma_p1' requires output_cotangent_basis.shape[1] == 1")
    if transformer_lbi_cuda is None or not hasattr(transformer_lbi_cuda, "attention_input_pullback_basis_mma_p1"):
        return _fused_fa2_p1_attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=float(softmax_scale),
            causal=causal,
        )
    if not causal or not q.is_cuda or q.dtype != torch.float16 or q.shape[-1] != 64:
        return _fused_fa2_p1_attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=float(softmax_scale),
            causal=causal,
        )
    try:
        out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            0.0,
            True,
            False,
            scale=float(softmax_scale),
        )
        dq, dk, dv = transformer_lbi_cuda.attention_input_pullback_basis_mma_p1(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            out.contiguous(),
            logsumexp.contiguous(),
            output_cotangent_basis[:, 0].contiguous(),
            float(softmax_scale),
            True,
        )
    except (AttributeError, RuntimeError):
        return _fused_fa2_p1_attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=float(softmax_scale),
            causal=causal,
        )
    return dq.unsqueeze(1), dk.unsqueeze(1), dv.unsqueeze(1)


def _mma_fa2_pblock4_attention_input_pullback_basis(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_cotangent_basis: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    basis = output_cotangent_basis.shape[1]
    if basis < 1 or basis > 4:
        raise ValueError("kernel_mode='fa2_mma_pblock4' requires 1 <= output_cotangent_basis.shape[1] <= 4")
    if transformer_lbi_cuda is None or not hasattr(transformer_lbi_cuda, "attention_input_pullback_basis_mma_pblock4"):
        return _sdpa_autograd_attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=float(softmax_scale),
            causal=causal,
        )
    if not causal or not q.is_cuda or q.dtype != torch.float16 or q.shape[-1] != 64:
        return _sdpa_autograd_attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=float(softmax_scale),
            causal=causal,
        )
    try:
        out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            0.0,
            True,
            False,
            scale=float(softmax_scale),
        )
        dq, dk, dv = transformer_lbi_cuda.attention_input_pullback_basis_mma_pblock4(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            out.contiguous(),
            logsumexp.contiguous(),
            output_cotangent_basis.contiguous(),
            float(softmax_scale),
            True,
        )
    except (AttributeError, RuntimeError):
        return _sdpa_autograd_attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=float(softmax_scale),
            causal=causal,
        )
    return dq, dk, dv


def attention_input_pullback_basis(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_cotangent_basis: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool = True,
    use_cuda: bool = True,
    kernel_mode: str = "fa2_mma_pblock4",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not use_cuda:
        return _torch_attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=float(softmax_scale),
            causal=causal,
        )
    if kernel_mode == "fa2_loop":
        return _sdpa_autograd_attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=float(softmax_scale),
            causal=causal,
        )
    if kernel_mode == "fa2_p1":
        return _fused_fa2_p1_attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=float(softmax_scale),
            causal=causal,
        )
    if kernel_mode == "fa2_mma_p1":
        return _mma_fa2_p1_attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=float(softmax_scale),
            causal=causal,
        )
    if kernel_mode == "fa2_mma_pblock4":
        return _mma_fa2_pblock4_attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=float(softmax_scale),
            causal=causal,
        )
    if kernel_mode == "simple":
        if use_cuda and transformer_lbi_cuda is not None and q.is_cuda:
            return tuple(
                transformer_lbi_cuda.attention_input_pullback_basis_simple(
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    output_cotangent_basis.contiguous(),
                    float(softmax_scale),
                    bool(causal),
                )
            )
        return _torch_attention_input_pullback_basis(
            q,
            k,
            v,
            output_cotangent_basis,
            softmax_scale=float(softmax_scale),
            causal=causal,
        )
    raise ValueError(f"unknown attention pullback kernel_mode: {kernel_mode}")


__all__ = ["attention_input_pullback_basis"]
