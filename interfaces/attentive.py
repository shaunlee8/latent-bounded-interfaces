from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .base import InterfaceModule, InterfaceSpec, InterfaceStep
from .vector_mlp import (
    _layernorm_input_jacobian_apply,
    _module_input_jacobian_t_apply_autograd,
)


def _softmax_jacobian_apply(probs: torch.Tensor, d_logits: torch.Tensor, dim: int) -> torch.Tensor:
    """Push tangents through a softmax along `dim`; the softmax Jacobian is
    symmetric, so this is also the transpose applied to cotangents."""
    inner = (probs * d_logits).sum(dim=dim, keepdim=True)
    return probs * (d_logits - inner)


_RMS_EPS = 1e-6

# Compiled interface stages: the slot-attention / gated-pool math runs as many
# small eager ops per region; LBI_INTERFACE_COMPILE=1 fuses each stage.
_STAGE_COMPILED: dict = {}


def _stage(fn):
    import os

    if os.environ.get("LBI_INTERFACE_COMPILE", "0") != "1":
        return fn
    compiled = _STAGE_COMPILED.get(fn)
    if compiled is None:
        compiled = torch.compile(fn, dynamic=False)
        _STAGE_COMPILED[fn] = compiled
    return compiled


def _slot_attention_math(canvas, state, w_query, keys, salience, attn_dim):
    """Slot mixture per token plus the salience logit gain d(logit)/d(state)."""
    queries = _rms_normalize(canvas @ w_query.t())
    keys_n = _rms_normalize(keys)
    bounded = torch.tanh(state * salience)
    logits = queries @ keys_n.t() / math.sqrt(attn_dim) + bounded.unsqueeze(1)
    return torch.softmax(logits, dim=-1), salience * (1.0 - bounded * bounded)


def _slot_decode_math(attn, state, values):
    return (attn * state.unsqueeze(1)) @ values


def _gated_pool_math(features, w_value, w_score):
    """RMS-normed value/score projections, per-channel softmax over tokens, and
    the pooled state delta."""
    normed = _rms_normalize(features)
    values = normed @ w_value.t()
    score_logits = normed @ w_score.t()
    weights = torch.softmax(score_logits, dim=1)
    return values, score_logits, weights, (weights * values).sum(dim=1)


def _decode_jvp_math(attn, state, salience_gain, values, d_state):
    d_logits = (d_state * salience_gain.unsqueeze(1)).unsqueeze(2)
    attn_b = attn.unsqueeze(1)
    d_attn = _softmax_jacobian_apply(attn_b, d_logits, dim=-1)
    coeff = d_attn * state.unsqueeze(1).unsqueeze(2) + attn_b * d_state.unsqueeze(2)
    return torch.einsum("bplr,rd->bpld", coeff, values)


def _decode_vjp_math(attn, state, salience_gain, values, g):
    h = torch.einsum("bpld,rd->bplr", g, values)
    attn_b = attn.unsqueeze(1)
    g_state = (attn_b * h).sum(dim=2)
    g_attn = h * state.unsqueeze(1).unsqueeze(2)
    g_logits = _softmax_jacobian_apply(attn_b, g_attn, dim=-1)
    return g_state + g_logits.sum(dim=2) * salience_gain.unsqueeze(1)


def _projected_update_jvp_math(projected, inner, features, values, score_logits, weights, delta_state, width):
    """d_delta from the projected region tangent (value/score rows + RMS inner
    products): P1 + P3 - P2 * delta over the cached softmax pools."""
    inv = torch.rsqrt(features.pow(2).mean(dim=-1, keepdim=True) + _RMS_EPS)
    factor = (inv.pow(2).squeeze(-1).unsqueeze(1) * inner) / float(features.shape[-1])
    inv_b = inv.unsqueeze(1)
    values_b = values.unsqueeze(1)
    logits_b = score_logits.unsqueeze(1)
    d_values = inv_b * projected[..., :width] - values_b * factor.unsqueeze(-1)
    d_scores = inv_b * projected[..., width:] - logits_b * factor.unsqueeze(-1)
    weights_b = weights.unsqueeze(1)
    p_value = (weights_b * d_values).sum(dim=2)
    p_score = (weights_b * d_scores).sum(dim=2)
    p_weighted = (weights_b * values_b * d_scores).sum(dim=2)
    return p_value + p_weighted - p_score * delta_state.unsqueeze(1)


def _rms_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Parameter-free RMS normalization; bounds downstream softmax logits so
    the interface keeps the backbone's LR headroom."""
    return x * torch.rsqrt(x.pow(2).mean(dim=dim, keepdim=True) + _RMS_EPS)


def _rms_normalize_jacobian_apply(x: torch.Tensor, d_in: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Push tangents through `_rms_normalize` at primal `x` (broadcasts over a
    leading basis axis in `d_in`); the Jacobian is symmetric, so this is also
    the transpose applied to cotangents."""
    inv = torch.rsqrt(x.pow(2).mean(dim=dim, keepdim=True) + _RMS_EPS)
    denom = x.pow(2).mean(dim=dim, keepdim=True) + _RMS_EPS
    inner = (x * d_in).mean(dim=dim, keepdim=True)
    return inv * (d_in - x * inner / denom)


class GatedPoolEncoder(nn.Module):
    """Per-channel softmax pooling over tokens: each state channel forms its own
    attention distribution over the sequence via a learned score projection."""

    def __init__(self, feature_dim: int, width: int) -> None:
        super().__init__()
        self.value = nn.Linear(feature_dim, width, bias=False)
        self.score = nn.Linear(feature_dim, width, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError("gated-pool encoder expects features shaped [B, T, D]")
        if features.dtype != self.value.weight.dtype:
            features = features.to(dtype=self.value.weight.dtype)
        return _stage(_gated_pool_math)(features, self.value.weight, self.score.weight)[-1]


class SlotAttentionDecoder(nn.Module):
    """Token-query attention over state channels: each token reads its own
    softmax mixture of the per-channel value directions, scaled by the state."""

    def __init__(self, feature_dim: int, width: int, attn_dim: int) -> None:
        super().__init__()
        self.attn_dim = int(attn_dim)
        self.query = nn.Linear(feature_dim, attn_dim, bias=False)
        self.keys = nn.Parameter(torch.empty(width, attn_dim))
        self.salience = nn.Parameter(torch.zeros(width))
        self.values = nn.Parameter(torch.empty(width, feature_dim))
        bound_k = 1.0 / math.sqrt(attn_dim)
        bound_v = 1.0 / math.sqrt(width)
        nn.init.uniform_(self.keys, -bound_k, bound_k)
        nn.init.uniform_(self.values, -bound_v, bound_v)

    def attention(self, state: torch.Tensor, canvas_features: torch.Tensor) -> torch.Tensor:
        """Slot mixture per token: softmax over channels of RMS-normed
        query-key logits plus a tanh-bounded state salience term. Returns
        [B, L, R]."""
        if state.dim() != 2:
            raise ValueError("slot decoder expects state shaped [B, R]")
        if canvas_features.dim() != 3:
            raise ValueError("slot decoder expects canvas features shaped [B, L, D]")
        weight = self.query.weight
        if canvas_features.dtype != weight.dtype:
            canvas_features = canvas_features.to(dtype=weight.dtype)
        if state.dtype != weight.dtype:
            state = state.to(dtype=weight.dtype)
        attn, _ = _stage(_slot_attention_math)(
            canvas_features, state, weight, self.keys, self.salience, self.attn_dim
        )
        return attn

    def forward(self, state: torch.Tensor, canvas_features: torch.Tensor) -> torch.Tensor:
        attn = self.attention(state, canvas_features)
        state = state.to(dtype=attn.dtype)
        return _stage(_slot_decode_math)(attn, state, self.values.to(dtype=attn.dtype))


class AttentiveInterface(InterfaceModule):
    """Routed vector interface: gated-pool encoders, slot-attention decoders,
    and per-region normalization over a [B, R] boundary state."""

    def __init__(
        self,
        *,
        feature_dim: int,
        num_regions: int,
        interface_width: int,
        attn_dim: int = 64,
        update_scale_init: float = 0.5,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be > 0")
        if num_regions <= 0:
            raise ValueError("num_regions must be > 0")
        if interface_width <= 0:
            raise ValueError("interface_width must be > 0")
        if attn_dim <= 0:
            raise ValueError("attn_dim must be > 0")
        if update_scale_init <= 0.0:
            raise ValueError("update_scale_init must be > 0")
        self.spec = InterfaceSpec(
            state_shape=(interface_width,),
            state_flat_dim=interface_width,
            region_condition_dim=feature_dim,
            condition_is_tokenwise=True,
        )
        self.initial_encoder = GatedPoolEncoder(feature_dim, interface_width)
        self.decoders = nn.ModuleList(
            [SlotAttentionDecoder(feature_dim, interface_width, attn_dim) for _ in range(num_regions)]
        )
        self.encoders = nn.ModuleList(
            [GatedPoolEncoder(feature_dim, interface_width) for _ in range(num_regions)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(interface_width) for _ in range(num_regions)])
        self.update_scale = nn.Parameter(torch.full((num_regions,), float(update_scale_init)))

    @property
    def num_regions(self) -> int:
        return len(self.decoders)

    def validate_region_count(self, num_regions: int) -> None:
        if num_regions != self.num_regions:
            raise ValueError(
                "attentive interface region count must match model region count "
                f"({self.num_regions} != {num_regions})"
            )

    def initialize(self, canvas_features: torch.Tensor) -> torch.Tensor:
        return self.initial_encoder(canvas_features)

    def decode(
        self,
        state: torch.Tensor,
        region_index: int,
        *,
        canvas_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if canvas_features is None:
            raise ValueError("attentive interface decode requires canvas_features")
        return self.decoders[region_index](state, canvas_features)

    def update(
        self,
        state: torch.Tensor,
        region_features: torch.Tensor,
        region_index: int,
    ) -> InterfaceStep:
        encoder = self.encoders[region_index]
        features = region_features
        if features.dtype != encoder.value.weight.dtype:
            features = features.to(dtype=encoder.value.weight.dtype)
        pool_values, pool_score_logits, pool_weights, delta_state = _stage(_gated_pool_math)(
            features, encoder.value.weight, encoder.score.weight
        )
        update_scale = torch.tanh(self.update_scale[region_index])
        pre_norm_state = state + update_scale * delta_state
        next_state = self.norms[region_index](pre_norm_state)
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

    def initial_vjp_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.initial_encoder.parameters() if p.requires_grad]

    def region_vjp_parameters(self, region_index: int) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        params.extend([p for p in self.decoders[region_index].parameters() if p.requires_grad])
        params.extend([p for p in self.encoders[region_index].parameters() if p.requires_grad])
        params.extend([p for p in self.norms[region_index].parameters() if p.requires_grad])
        return params

    def shared_vjp_parameters(self) -> list[nn.Parameter]:
        return [self.update_scale] if self.update_scale.requires_grad else []

    def _decode_primal(
        self, region_cache: Any, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SlotAttentionDecoder]:
        """Recompute the decode attention [B, L, R], state [B, R], and the
        salience logit gain d(logit_c)/d(state_c) [B, R] in `dtype` from the
        cached region inputs (one small GEMM; nothing is stored)."""
        decoder = self.decoders[region_cache.region_index]
        state = region_cache.state_in.detach().to(dtype=dtype)
        canvas = region_cache.canvas_features.detach().to(dtype=dtype)
        attn, salience_gain = _stage(_slot_attention_math)(
            canvas,
            state,
            decoder.query.weight.to(dtype=dtype),
            decoder.keys.to(dtype=dtype),
            decoder.salience.to(dtype=dtype),
            decoder.attn_dim,
        )
        return attn, state, salience_gain, decoder

    def apply_decode_jacobian_t_to_state_input(
        self,
        *,
        region_cache: Any,
        region_input_cotangent_basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if region_input_cotangent_basis.dim() != 4:
            raise ValueError("region_input_cotangent_basis must have shape [B, P, L, D].")
        g = region_input_cotangent_basis
        compute_dtype = torch.promote_types(g.dtype, region_cache.state_in.dtype)
        attn, state, salience_gain, decoder = self._decode_primal(region_cache, compute_dtype)
        g_c = g.to(device=attn.device, dtype=compute_dtype)
        g_state = _stage(_decode_vjp_math)(
            attn, state, salience_gain, decoder.values.to(dtype=compute_dtype), g_c
        )
        return {
            "g_condition": region_input_cotangent_basis,
            "g_state_input": g_state.to(device=g.device, dtype=g.dtype),
        }

    def apply_decode_jacobian_to_state_input(
        self,
        *,
        region_cache: Any,
        state_input_tangent_basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward-mode: push a state tangent basis [B, P, R] through the slot
        decode -> token-wise region-input tangent basis [B, P, L, D]. Exact
        adjoint of `apply_decode_jacobian_t_to_state_input`."""
        if state_input_tangent_basis.dim() != 3:
            raise ValueError("state_input_tangent_basis must have shape [B, P, R].")
        d_state = state_input_tangent_basis
        compute_dtype = torch.promote_types(d_state.dtype, region_cache.state_in.dtype)
        attn, state, salience_gain, decoder = self._decode_primal(region_cache, compute_dtype)
        d_c = d_state.to(device=attn.device, dtype=compute_dtype)
        d_condition = _stage(_decode_jvp_math)(
            attn, state, salience_gain, decoder.values.to(dtype=compute_dtype), d_c
        )
        d_condition = d_condition.to(device=d_state.device, dtype=d_state.dtype)
        return {
            "d_condition": d_condition,
            "d_region_input": d_condition,
        }

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
        g_pre_norm = _module_input_jacobian_t_apply_autograd(
            self.norms[region_cache.region_index],
            diagnostics["pre_norm_state"],
            state_out_cotangent_basis,
        )
        g_state_skip = g_pre_norm
        g_delta = g_pre_norm * diagnostics["update_scale"].to(dtype=g_pre_norm.dtype).view(1, 1, 1)
        compute_dtype = torch.promote_types(g_delta.dtype, diagnostics["pool_weights"].dtype)
        weights = diagnostics["pool_weights"].to(dtype=compute_dtype).unsqueeze(1)
        values = diagnostics["pool_values"].to(dtype=compute_dtype).unsqueeze(1)
        g_delta_c = g_delta.to(dtype=compute_dtype).unsqueeze(2)
        g_values = weights * g_delta_c
        g_weights = values * g_delta_c
        g_scores = _softmax_jacobian_apply(weights, g_weights, dim=2)
        g_normed = torch.einsum(
            "bplr,rd->bpld", g_values, encoder.value.weight.to(dtype=compute_dtype)
        ) + torch.einsum("bplr,rd->bpld", g_scores, encoder.score.weight.to(dtype=compute_dtype))
        features = region_cache.region_output.detach().to(dtype=compute_dtype).unsqueeze(1)
        g_region_output = _rms_normalize_jacobian_apply(features, g_normed)
        return {
            "g_pre_norm_state": g_pre_norm,
            "g_state_skip": g_state_skip,
            "g_delta_state": g_delta,
            "g_region_output": g_region_output.to(dtype=state_out_cotangent_basis.dtype),
        }

    def update_tangent_projection(self, region_cache: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """The region-JVP output contraction the pooled update path consumes:
        stacked value/score projection rows [2R, D] plus the raw region output
        [B, L, D] for the RMS-norm inner-product chain."""
        encoder = self.encoders[region_cache.region_index]
        projection = torch.cat([encoder.value.weight, encoder.score.weight], dim=0)
        return projection.detach(), region_cache.region_output.detach()

    def _apply_update_jacobian_projected(
        self,
        *,
        region_cache: Any,
        projected: torch.Tensor,
        inner: torch.Tensor,
        state_input_tangent_basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward-mode update from the projected region tangent ([B, P, L, 2R]
        value/score rows + [B, P, L] RMS inner products) instead of the full
        [B, P, L, D] basis; algebraically identical to the full-tangent path."""
        diagnostics = region_cache.interface_step.diagnostics
        width = diagnostics["pool_weights"].shape[-1]
        compute_dtype = torch.promote_types(projected.dtype, diagnostics["pool_weights"].dtype)
        d_delta = _stage(_projected_update_jvp_math)(
            projected.to(dtype=compute_dtype),
            inner.to(dtype=compute_dtype),
            region_cache.region_output.detach().to(dtype=compute_dtype),
            diagnostics["pool_values"].to(dtype=compute_dtype),
            diagnostics["pool_score_logits"].to(dtype=compute_dtype),
            diagnostics["pool_weights"].to(dtype=compute_dtype),
            diagnostics["delta_state"].to(dtype=compute_dtype),
            width,
        )
        update_scale = diagnostics["update_scale"].to(dtype=d_delta.dtype).view(1, 1, 1)
        d_pre_norm = state_input_tangent_basis.to(dtype=d_delta.dtype) + update_scale * d_delta
        d_state_out = _layernorm_input_jacobian_apply(
            self.norms[region_cache.region_index],
            diagnostics["pre_norm_state"],
            d_pre_norm,
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
        """Forward-mode: push a region-output tangent basis [B, P, L, D] plus the
        skip state tangent [B, P, R] through the gated-pool update -> state-out
        tangent basis [B, P, R]. Exact adjoint of
        `apply_update_jacobian_t_to_region_output`. Also accepts the projected
        `(projected, inner)` pair from `update_tangent_projection`."""
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
        t_c = t.to(dtype=compute_dtype)
        weights = diagnostics["pool_weights"].to(dtype=compute_dtype).unsqueeze(1)
        values = diagnostics["pool_values"].to(dtype=compute_dtype).unsqueeze(1)
        features = region_cache.region_output.detach().to(dtype=compute_dtype).unsqueeze(1)
        d_normed = _rms_normalize_jacobian_apply(features, t_c)
        d_values = torch.einsum("bpld,rd->bplr", d_normed, encoder.value.weight.to(dtype=compute_dtype))
        d_scores = torch.einsum("bpld,rd->bplr", d_normed, encoder.score.weight.to(dtype=compute_dtype))
        d_weights = _softmax_jacobian_apply(weights, d_scores, dim=2)
        d_delta = (d_weights * values + weights * d_values).sum(dim=2)
        update_scale = diagnostics["update_scale"].to(dtype=d_delta.dtype).view(1, 1, 1)
        d_pre_norm = state_input_tangent_basis.to(dtype=d_delta.dtype) + update_scale * d_delta
        d_state_out = _layernorm_input_jacobian_apply(
            self.norms[region_cache.region_index],
            diagnostics["pre_norm_state"],
            d_pre_norm,
        )
        return {
            "d_delta_state": d_delta,
            "d_pre_norm_state": d_pre_norm,
            "d_state_out": d_state_out.to(dtype=state_input_tangent_basis.dtype),
        }
