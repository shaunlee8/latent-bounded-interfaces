from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .attentive import (
    _RMS_EPS,
    AttentiveInterface,
    GatedPoolEncoder,
    _rms_normalize,
    _rms_normalize_jacobian_apply,
    _softmax_jacobian_apply,
    _stage,
)
from .base import InterfaceSpec, InterfaceStep
from .vector_mlp import (
    _layernorm_input_jacobian_apply,
    _module_input_jacobian_t_apply_autograd,
)


class ChunkRMSNorm(nn.Module):
    """Direction-preserving per-chunk norm: RMS normalization with a learned
    elementwise scale (no centering, no bias). Unlike LayerNorm at small chunk
    widths — which collapses a width-2 chunk to the sign of its within-chunk
    difference — this keeps the chunk vector's direction, so the state carries
    continuous messages and the norm Jacobian stays rank-(width-1)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_normalize(x) * self.weight.to(dtype=x.dtype)


def _chunked_slot_attention_math(canvas, state, w_query, keys, salience, attn_dim, chunk_count, strict):
    """Chunk-masked slot mixture per token plus the salience logit gain
    d(logit)/d(state). Masked slots carry zero attention mass, so the shared
    softmax Jacobian formulas apply unchanged; fully-masked rows (chunk-0
    tokens under `strict`) are zeroed and contribute nothing in either mode."""
    bsz, seq_len, _ = canvas.shape
    total = keys.shape[0]
    queries = _rms_normalize(canvas @ w_query.t())
    keys_n = _rms_normalize(keys)
    bounded = torch.tanh(state * salience)
    logits = queries @ keys_n.t() / math.sqrt(attn_dim) + bounded.unsqueeze(1)
    chunk_len = seq_len // chunk_count
    token_chunk = torch.arange(seq_len, device=canvas.device) // chunk_len
    slot_chunk = torch.arange(total, device=canvas.device) // (total // chunk_count)
    if strict:
        mask = slot_chunk.unsqueeze(0) >= token_chunk.unsqueeze(1)
    else:
        mask = slot_chunk.unsqueeze(0) > token_chunk.unsqueeze(1)
    empty_row = mask.all(dim=1)
    safe_logits = torch.where(mask.unsqueeze(0), torch.finfo(logits.dtype).min, logits)
    safe_logits = torch.where(empty_row.view(1, -1, 1), torch.zeros_like(safe_logits), safe_logits)
    attn = torch.softmax(safe_logits, dim=-1)
    attn = attn * (~empty_row).view(1, -1, 1).to(dtype=attn.dtype)
    return attn, salience * (1.0 - bounded * bounded)


def _chunked_slot_decode_math(canvas, state, w_query, keys, salience, values, attn_dim, chunk_count, strict):
    """Chunk-causal slot decode: token t in chunk i reads a softmax mixture of
    the slots of chunks <= i (leakage radius one chunk) or, with `strict`, of
    chunks < i only (fully adapted: a valid causal model; chunk-0 tokens read
    nothing and receive a zero condition)."""
    attn, _ = _chunked_slot_attention_math(
        canvas, state, w_query, keys, salience, attn_dim, chunk_count, strict
    )
    return (attn * state.unsqueeze(1)) @ values


def _chunk_pool_math(features, w_value, w_score, chunk_count):
    """Per-chunk gated pooling: shared projections, chunk-local softmax over
    tokens. Returns the chunk-shaped value/score/weight pools plus the
    flattened [B, chunks * width] state delta."""
    bsz, seq_len, _ = features.shape
    width = w_value.shape[0]
    chunk_len = seq_len // chunk_count
    normed = _rms_normalize(features)
    values = (normed @ w_value.t()).view(bsz, chunk_count, chunk_len, width)
    scores = (normed @ w_score.t()).view(bsz, chunk_count, chunk_len, width)
    weights = torch.softmax(scores, dim=2)
    delta = (weights * values).sum(dim=2).reshape(bsz, chunk_count * width)
    return values, scores, weights, delta


def _chunk_update_vjp_math(g_delta, features, values, weights, w_value, w_score):
    """Pull a chunk-shaped delta cotangent [B, P, m, r] back to the raw region
    output [B, P, L, D] through the chunk-local softmax pools and the shared
    RMS-normed projections."""
    bsz, chunk_count, chunk_len, _ = values.shape
    lanes = g_delta.shape[1]
    weights_b = weights.unsqueeze(1)
    values_b = values.unsqueeze(1)
    g_delta_c = g_delta.unsqueeze(3)
    g_values = weights_b * g_delta_c
    g_weights = values_b * g_delta_c
    g_scores = _softmax_jacobian_apply(weights_b, g_weights, dim=3)
    g_normed = torch.einsum("bpmlr,rd->bpmld", g_values, w_value) + torch.einsum(
        "bpmlr,rd->bpmld", g_scores, w_score
    )
    g_normed = g_normed.reshape(bsz, lanes, chunk_count * chunk_len, -1)
    return _rms_normalize_jacobian_apply(features.unsqueeze(1), g_normed)


def _chunk_update_jvp_math(t, features, values, weights, w_value, w_score):
    """Push a region-output tangent basis [B, P, L, D] through the chunk-local
    pools -> flattened delta tangent [B, P, m * r]. Exact adjoint of
    `_chunk_update_vjp_math`."""
    bsz, chunk_count, chunk_len, width = values.shape
    lanes = t.shape[1]
    d_normed = _rms_normalize_jacobian_apply(features.unsqueeze(1), t)
    d_values = torch.einsum("bpld,rd->bplr", d_normed, w_value).view(
        bsz, lanes, chunk_count, chunk_len, width
    )
    d_scores = torch.einsum("bpld,rd->bplr", d_normed, w_score).view(
        bsz, lanes, chunk_count, chunk_len, width
    )
    weights_b = weights.unsqueeze(1)
    d_weights = _softmax_jacobian_apply(weights_b, d_scores, dim=3)
    d_delta = (d_weights * values.unsqueeze(1) + weights_b * d_values).sum(dim=3)
    return d_delta.reshape(bsz, lanes, chunk_count * width)


def _chunk_projected_update_jvp_math(
    projected, inner, features, values, score_logits, weights, delta_state, width, chunk_count
):
    """Delta tangent from the projected region tangent ([B, P, L, 2r] value/
    score rows + [B, P, L] RMS inner products) with chunk-local softmax pools;
    algebraically identical to the full-tangent path."""
    bsz, lanes, seq_len, _ = projected.shape
    chunk_len = seq_len // chunk_count
    inv = torch.rsqrt(features.pow(2).mean(dim=-1, keepdim=True) + _RMS_EPS)
    factor = (inv.pow(2).squeeze(-1).unsqueeze(1) * inner) / float(features.shape[-1])
    inv_b = inv.unsqueeze(1)
    values_flat = values.reshape(bsz, 1, seq_len, width)
    scores_flat = score_logits.reshape(bsz, 1, seq_len, width)
    d_values = inv_b * projected[..., :width] - values_flat * factor.unsqueeze(-1)
    d_scores = inv_b * projected[..., width:] - scores_flat * factor.unsqueeze(-1)
    d_values = d_values.view(bsz, lanes, chunk_count, chunk_len, width)
    d_scores = d_scores.view(bsz, lanes, chunk_count, chunk_len, width)
    weights_b = weights.unsqueeze(1)
    p_value = (weights_b * d_values).sum(dim=3)
    p_score = (weights_b * d_scores).sum(dim=3)
    p_weighted = (weights_b * values.unsqueeze(1) * d_scores).sum(dim=3)
    delta_c = delta_state.view(bsz, 1, chunk_count, width)
    return (p_value + p_weighted - p_score * delta_c).reshape(bsz, lanes, chunk_count * width)


class ChunkedAttentiveInterface(AttentiveInterface):
    """Chunk-resolved attentive interface: the [B, chunks * width] state carries
    one rank-`width` message per L/chunks-token span, so chunk-local features
    cross region boundaries at the same total state width (and machinery cost)
    as a flat rank-(chunks * width) interface."""

    def __init__(
        self,
        *,
        feature_dim: int,
        num_regions: int,
        chunk_count: int,
        chunk_width: int,
        attn_dim: int = 64,
        update_scale_init: float = 0.5,
        strict_causal: bool = False,
        chunk_norm: str = "layer",
    ) -> None:
        if chunk_count <= 1:
            raise ValueError("chunk_count must be > 1 (use AttentiveInterface for the global case)")
        if chunk_width <= 0:
            raise ValueError("chunk_width must be > 0")
        if chunk_norm not in {"layer", "rms"}:
            raise ValueError("chunk_norm must be 'layer' or 'rms'")
        super().__init__(
            feature_dim=feature_dim,
            num_regions=num_regions,
            interface_width=chunk_count * chunk_width,
            attn_dim=attn_dim,
            update_scale_init=update_scale_init,
        )
        self.chunk_count = int(chunk_count)
        self.chunk_width = int(chunk_width)
        self.strict_causal = bool(strict_causal)
        self.spec = InterfaceSpec(
            state_shape=(chunk_count * chunk_width,),
            state_flat_dim=chunk_count * chunk_width,
            region_condition_dim=feature_dim,
            condition_is_tokenwise=True,
        )
        # Encoders share one width-`chunk_width` projection pair across chunks;
        # norms run per chunk so no cross-chunk statistics couple the slots
        # (which is what keeps the strict variant exactly causal).
        self.initial_encoder = GatedPoolEncoder(feature_dim, chunk_width)
        self.encoders = nn.ModuleList(
            [GatedPoolEncoder(feature_dim, chunk_width) for _ in range(num_regions)]
        )
        self.chunk_norm_kind = chunk_norm
        norm_cls = ChunkRMSNorm if chunk_norm == "rms" else nn.LayerNorm
        self.norms = nn.ModuleList([norm_cls(chunk_width) for _ in range(num_regions)])

    def _check_seq(self, seq_len: int) -> None:
        if seq_len % self.chunk_count != 0:
            raise ValueError(
                f"sequence length {seq_len} must divide the chunk count {self.chunk_count}"
            )

    def initialize(self, canvas_features: torch.Tensor) -> torch.Tensor:
        self._check_seq(canvas_features.shape[1])
        encoder = self.initial_encoder
        features = canvas_features
        if features.dtype != encoder.value.weight.dtype:
            features = features.to(dtype=encoder.value.weight.dtype)
        return _stage(_chunk_pool_math)(
            features, encoder.value.weight, encoder.score.weight, self.chunk_count
        )[-1]

    def decode(
        self,
        state: torch.Tensor,
        region_index: int,
        *,
        canvas_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if canvas_features is None:
            raise ValueError("chunked interface decode requires canvas_features")
        self._check_seq(canvas_features.shape[1])
        decoder = self.decoders[region_index]
        weight = decoder.query.weight
        canvas = canvas_features
        if canvas.dtype != weight.dtype:
            canvas = canvas.to(dtype=weight.dtype)
        if state.dtype != weight.dtype:
            state = state.to(dtype=weight.dtype)
        return _stage(_chunked_slot_decode_math)(
            canvas, state, weight, decoder.keys, decoder.salience,
            decoder.values.to(dtype=weight.dtype), decoder.attn_dim, self.chunk_count,
            self.strict_causal,
        )

    def update(
        self,
        state: torch.Tensor,
        region_features: torch.Tensor,
        region_index: int,
    ) -> InterfaceStep:
        self._check_seq(region_features.shape[1])
        encoder = self.encoders[region_index]
        features = region_features
        if features.dtype != encoder.value.weight.dtype:
            features = features.to(dtype=encoder.value.weight.dtype)
        pool_values, pool_score_logits, pool_weights, delta_state = _stage(_chunk_pool_math)(
            features, encoder.value.weight, encoder.score.weight, self.chunk_count
        )
        update_scale = torch.tanh(self.update_scale[region_index])
        pre_norm_state = state + update_scale * delta_state
        grouped = pre_norm_state.view(-1, self.chunk_count, self.chunk_width)
        next_state = self.norms[region_index](grouped).reshape(pre_norm_state.shape)
        return InterfaceStep(
            state=next_state,
            diagnostics={
                "pool_values": pool_values,
                "pool_score_logits": pool_score_logits,
                "pool_weights": pool_weights,
                "delta_state": delta_state,
                "update_scale": update_scale,
                "pre_norm_state": pre_norm_state,
            },
        )

    def _decode_primal(self, region_cache: Any, dtype: torch.dtype):
        """Chunk-masked decode attention at the cached operating point; the
        inherited decode JVP/VJP formulas consume it unchanged (masked slots
        carry zero mass, so the softmax Jacobian is exact as written)."""
        decoder = self.decoders[region_cache.region_index]
        state = region_cache.state_in.detach().to(dtype=dtype)
        canvas = region_cache.canvas_features.detach().to(dtype=dtype)
        self._check_seq(canvas.shape[1])
        attn, salience_gain = _stage(_chunked_slot_attention_math)(
            canvas,
            state,
            decoder.query.weight.to(dtype=dtype),
            decoder.keys.to(dtype=dtype),
            decoder.salience.to(dtype=dtype),
            decoder.attn_dim,
            self.chunk_count,
            self.strict_causal,
        )
        return attn, state, salience_gain, decoder

    def _chunk_norm_vjp(
        self, region_index: int, pre_norm_state: torch.Tensor, g_out: torch.Tensor
    ) -> torch.Tensor:
        """Per-chunk LayerNorm transpose with the chunk axis folded into the
        batch; [B, P, m * r] cotangents in and out."""
        bsz, lanes, total = g_out.shape
        m, r = self.chunk_count, self.chunk_width
        pre_folded = pre_norm_state.view(bsz, m, r).reshape(bsz * m, r)
        g_folded = g_out.view(bsz, lanes, m, r).permute(0, 2, 1, 3).reshape(bsz * m, lanes, r)
        g_pre = _module_input_jacobian_t_apply_autograd(self.norms[region_index], pre_folded, g_folded)
        return g_pre.view(bsz, m, lanes, r).permute(0, 2, 1, 3).reshape(bsz, lanes, total)

    def _chunk_norm_jvp(
        self, region_index: int, pre_norm_state: torch.Tensor, d_in: torch.Tensor
    ) -> torch.Tensor:
        """Per-chunk LayerNorm JVP with the chunk axis folded into the batch.
        Runs fp32: at small chunk widths the LN Jacobian is a cancellation of
        two large rsqrt-scaled terms, which half precision destroys."""
        bsz, lanes, total = d_in.shape
        m, r = self.chunk_count, self.chunk_width
        pre_folded = pre_norm_state.view(bsz, m, r).reshape(bsz * m, r)
        d_folded = d_in.view(bsz, lanes, m, r).permute(0, 2, 1, 3).reshape(bsz * m, lanes, r)
        norm = self.norms[region_index]
        if isinstance(norm, ChunkRMSNorm):
            d_out = (
                _rms_normalize_jacobian_apply(
                    pre_folded.float().unsqueeze(1), d_folded.float()
                )
                * norm.weight.float().view(1, 1, -1)
            ).to(dtype=d_in.dtype)
        else:
            d_out = _layernorm_input_jacobian_apply(
                norm, pre_folded.float(), d_folded.float()
            ).to(dtype=d_in.dtype)
        return d_out.view(bsz, m, lanes, r).permute(0, 2, 1, 3).reshape(bsz, lanes, total)

    def apply_update_jacobian_t_to_region_output(
        self,
        *,
        region_cache: Any,
        state_out_cotangent_basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if state_out_cotangent_basis.dim() != 3:
            raise ValueError("state_out_cotangent_basis must have shape [B, P, R].")
        diagnostics = region_cache.interface_step.diagnostics
        encoder = self.encoders[region_cache.region_index]
        g_pre_norm = self._chunk_norm_vjp(
            region_cache.region_index, diagnostics["pre_norm_state"], state_out_cotangent_basis
        )
        g_state_skip = g_pre_norm
        g_delta = g_pre_norm * diagnostics["update_scale"].to(dtype=g_pre_norm.dtype).view(1, 1, 1)
        compute_dtype = torch.promote_types(g_delta.dtype, diagnostics["pool_weights"].dtype)
        bsz, lanes, _ = g_delta.shape
        g_region_output = _stage(_chunk_update_vjp_math)(
            g_delta.to(dtype=compute_dtype).view(bsz, lanes, self.chunk_count, self.chunk_width),
            region_cache.region_output.detach().to(dtype=compute_dtype),
            diagnostics["pool_values"].to(dtype=compute_dtype),
            diagnostics["pool_weights"].to(dtype=compute_dtype),
            encoder.value.weight.to(dtype=compute_dtype),
            encoder.score.weight.to(dtype=compute_dtype),
        )
        return {
            "g_pre_norm_state": g_pre_norm,
            "g_state_skip": g_state_skip,
            "g_delta_state": g_delta,
            "g_region_output": g_region_output.to(dtype=state_out_cotangent_basis.dtype),
        }

    def _apply_update_jacobian_projected(
        self,
        *,
        region_cache: Any,
        projected: torch.Tensor,
        inner: torch.Tensor,
        state_input_tangent_basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        diagnostics = region_cache.interface_step.diagnostics
        compute_dtype = torch.promote_types(projected.dtype, diagnostics["pool_weights"].dtype)
        d_delta = _stage(_chunk_projected_update_jvp_math)(
            projected.to(dtype=compute_dtype),
            inner.to(dtype=compute_dtype),
            region_cache.region_output.detach().to(dtype=compute_dtype),
            diagnostics["pool_values"].to(dtype=compute_dtype),
            diagnostics["pool_score_logits"].to(dtype=compute_dtype),
            diagnostics["pool_weights"].to(dtype=compute_dtype),
            diagnostics["delta_state"].to(dtype=compute_dtype),
            self.chunk_width,
            self.chunk_count,
        )
        return self._finish_update_jvp(region_cache, d_delta, state_input_tangent_basis)

    def _finish_update_jvp(
        self,
        region_cache: Any,
        d_delta: torch.Tensor,
        state_input_tangent_basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        diagnostics = region_cache.interface_step.diagnostics
        update_scale = diagnostics["update_scale"].to(dtype=d_delta.dtype).view(1, 1, 1)
        d_pre_norm = state_input_tangent_basis.to(dtype=d_delta.dtype) + update_scale * d_delta
        d_state_out = self._chunk_norm_jvp(
            region_cache.region_index, diagnostics["pre_norm_state"], d_pre_norm
        )
        return {
            "d_delta_state": d_delta,
            "d_pre_norm_state": d_pre_norm,
            "d_state_out": d_state_out.to(dtype=state_input_tangent_basis.dtype),
        }

    def apply_update_jacobian_to_region_output(
        self,
        *,
        region_cache: Any,
        region_output_tangent_basis: "torch.Tensor | tuple[torch.Tensor, torch.Tensor]",
        state_input_tangent_basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if isinstance(region_output_tangent_basis, tuple):
            projected, inner = region_output_tangent_basis
            return self._apply_update_jacobian_projected(
                region_cache=region_cache,
                projected=projected,
                inner=inner,
                state_input_tangent_basis=state_input_tangent_basis,
            )
        if region_output_tangent_basis.dim() != 4:
            raise ValueError("region_output_tangent_basis must have shape [B, P, L, D].")
        if state_input_tangent_basis.dim() != 3:
            raise ValueError("state_input_tangent_basis must have shape [B, P, R].")
        diagnostics = region_cache.interface_step.diagnostics
        encoder = self.encoders[region_cache.region_index]
        t = region_output_tangent_basis
        compute_dtype = torch.promote_types(t.dtype, diagnostics["pool_weights"].dtype)
        d_delta = _stage(_chunk_update_jvp_math)(
            t.to(dtype=compute_dtype),
            region_cache.region_output.detach().to(dtype=compute_dtype),
            diagnostics["pool_values"].to(dtype=compute_dtype),
            diagnostics["pool_weights"].to(dtype=compute_dtype),
            encoder.value.weight.to(dtype=compute_dtype),
            encoder.score.weight.to(dtype=compute_dtype),
        )
        return self._finish_update_jvp(region_cache, d_delta, state_input_tangent_basis)

    def apply_update_jacobian_skip_only(
        self,
        *,
        region_cache: Any,
        state_input_tangent_basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Update JVP restricted to the skip path (identically zero region-
        output tangent): state basis -> per-chunk LayerNorm JVP. Used for the
        state lanes whose decode support is empty under strict chunking."""
        diagnostics = region_cache.interface_step.diagnostics
        d_state_out = self._chunk_norm_jvp(
            region_cache.region_index,
            diagnostics["pre_norm_state"],
            state_input_tangent_basis.to(dtype=diagnostics["pre_norm_state"].dtype),
        )
        return {"d_state_out": d_state_out.to(dtype=state_input_tangent_basis.dtype)}

    def state_tangent_support_starts(self, seq_len: int) -> torch.Tensor:
        """First token index where each state coordinate's decode tangent can
        be nonzero: chunk-c slots feed tokens of chunks > c (strict) or >= c
        (inclusive). Lanes with start >= seq_len have an empty decode path and
        the forward-mode construction skips their region factor entirely."""
        self._check_seq(seq_len)
        chunk_len = seq_len // self.chunk_count
        chunk_index = torch.arange(self.chunk_count).repeat_interleave(self.chunk_width)
        offset = 1 if self.strict_causal else 0
        return (chunk_index + offset) * chunk_len
