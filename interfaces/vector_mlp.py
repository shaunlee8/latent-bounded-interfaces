"""The paper's bounded interface: mean-pool encode of the region output into a
rank-r state, broadcast MLP decode into the next region, and a normalized
residual update, with the structured Jacobian applications of both maps."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from interfaces.base import InterfaceModule, InterfaceSpec, InterfaceStep
from interfaces.jacobians import (
    _layernorm_input_jacobian_apply,
    _module_input_jacobian_t_apply_autograd,
)
from interfaces.mlp_head import VectorMLPHead


def _linear_input_jacobian_t_apply(linear: nn.Linear, g_out: torch.Tensor) -> torch.Tensor:
    if g_out.dim() != 3:
        raise ValueError("linear input Jacobian-transpose apply expects g_out shaped [B, P, D_out].")
    compute_dtype = torch.promote_types(g_out.dtype, linear.weight.dtype)
    g_in = torch.einsum(
        "bpo,oi->bpi",
        g_out.to(device=linear.weight.device, dtype=compute_dtype),
        linear.weight.to(dtype=compute_dtype),
    )
    return g_in.to(device=g_out.device, dtype=g_out.dtype)


def _silu_input_jacobian_t_apply(x: torch.Tensor, g_out: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("SiLU input Jacobian-transpose apply expects x shaped [B, D].")
    if g_out.dim() != 3:
        raise ValueError("SiLU input Jacobian-transpose apply expects g_out shaped [B, P, D].")
    compute_dtype = torch.promote_types(x.dtype, g_out.dtype)
    x_compute = x.to(device=g_out.device, dtype=compute_dtype)
    g_compute = g_out.to(dtype=compute_dtype)
    sig = torch.sigmoid(x_compute)
    deriv = sig * (1.0 + x_compute * (1.0 - sig))
    return (g_compute * deriv.unsqueeze(1)).to(device=g_out.device, dtype=g_out.dtype)


def _interface_map_input_jacobian_t_apply(
    interface_map: nn.Module,
    x: torch.Tensor,
    g_out: torch.Tensor,
) -> torch.Tensor:
    net = getattr(interface_map, "net", None)
    if isinstance(net, nn.Linear):
        return _linear_input_jacobian_t_apply(net, g_out)
    if isinstance(net, nn.Sequential) and len(net) == 3:
        linear1, act, linear2 = net
        if not isinstance(linear1, nn.Linear) or not isinstance(act, nn.SiLU) or not isinstance(linear2, nn.Linear):
            raise TypeError("unsupported vector interface map sequential structure")
        # The primal replay runs in the module weight dtype, as the forward
        # does; an fp32 residual stream would otherwise crash the bf16 GEMM.
        hidden_pre = linear1(x.to(device=linear1.weight.device, dtype=linear1.weight.dtype))
        g_hidden = _linear_input_jacobian_t_apply(linear2, g_out.to(device=linear2.weight.device))
        g_hidden = g_hidden.to(device=hidden_pre.device, dtype=hidden_pre.dtype)
        g_hidden_pre = _silu_input_jacobian_t_apply(hidden_pre, g_hidden)
        g_in = _linear_input_jacobian_t_apply(linear1, g_hidden_pre)
        return g_in.to(device=g_out.device, dtype=g_out.dtype)
    raise TypeError(f"unsupported vector interface map net type: {type(net).__name__}")


def _mean_pool_jacobian_t_apply(g_pooled: torch.Tensor, seq_len: int) -> torch.Tensor:
    if g_pooled.dim() != 3:
        raise ValueError("mean-pool Jacobian-transpose apply expects g_pooled shaped [B, P, D].")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    return g_pooled.unsqueeze(2).expand(-1, -1, seq_len, -1) / float(seq_len)


def _broadcast_condition_jacobian_t_apply(g_region_input: torch.Tensor) -> torch.Tensor:
    if g_region_input.dim() != 4:
        raise ValueError("broadcast-condition Jacobian-transpose apply expects g_region_input shaped [B, P, L, D].")
    return g_region_input.sum(dim=2)


# JVP counterparts of the transpose helpers: together they assemble
# A_k = skip + encode . J_region . decode applied to the identity basis.


def _linear_input_jacobian_apply(linear: nn.Linear, d_in: torch.Tensor) -> torch.Tensor:
    if d_in.dim() != 3:
        raise ValueError("linear input Jacobian apply expects d_in shaped [B, P, D_in].")
    compute_dtype = torch.promote_types(d_in.dtype, linear.weight.dtype)
    d_out = torch.einsum(
        "bpi,oi->bpo",
        d_in.to(device=linear.weight.device, dtype=compute_dtype),
        linear.weight.to(dtype=compute_dtype),
    )
    return d_out.to(device=d_in.device, dtype=d_in.dtype)


def _silu_input_jacobian_apply(x: torch.Tensor, d_in: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("SiLU input Jacobian apply expects x shaped [B, D].")
    if d_in.dim() != 3:
        raise ValueError("SiLU input Jacobian apply expects d_in shaped [B, P, D].")
    compute_dtype = torch.promote_types(x.dtype, d_in.dtype)
    x_compute = x.to(device=d_in.device, dtype=compute_dtype)
    sig = torch.sigmoid(x_compute)
    deriv = sig * (1.0 + x_compute * (1.0 - sig))
    return (d_in.to(dtype=compute_dtype) * deriv.unsqueeze(1)).to(device=d_in.device, dtype=d_in.dtype)


def _interface_map_input_jacobian_apply(
    interface_map: nn.Module,
    x: torch.Tensor,
    d_in: torch.Tensor,
) -> torch.Tensor:
    net = getattr(interface_map, "net", None)
    if isinstance(net, nn.Linear):
        return _linear_input_jacobian_apply(net, d_in)
    if isinstance(net, nn.Sequential) and len(net) == 3:
        linear1, act, linear2 = net
        if not isinstance(linear1, nn.Linear) or not isinstance(act, nn.SiLU) or not isinstance(linear2, nn.Linear):
            raise TypeError("unsupported vector interface map sequential structure")
        # The primal replay runs in the module weight dtype (as the forward);
        # the tangent einsum and SiLU derivative promote internally.
        hidden_pre = linear1(x.to(device=linear1.weight.device, dtype=linear1.weight.dtype))
        d_hidden_pre = _linear_input_jacobian_apply(linear1, d_in)
        d_hidden = _silu_input_jacobian_apply(hidden_pre, d_hidden_pre.to(dtype=hidden_pre.dtype))
        d_out = _linear_input_jacobian_apply(linear2, d_hidden)
        return d_out.to(device=d_in.device, dtype=d_in.dtype)
    raise TypeError(f"unsupported vector interface map net type: {type(net).__name__}")


def _mean_pool_jacobian_apply(d_region_output: torch.Tensor) -> torch.Tensor:
    if d_region_output.dim() != 4:
        raise ValueError("mean-pool Jacobian apply expects d_region_output shaped [B, P, L, D].")
    return d_region_output.mean(dim=2)


def _broadcast_condition_jacobian_apply(d_condition: torch.Tensor, seq_len: int) -> torch.Tensor:
    if d_condition.dim() != 3:
        raise ValueError("broadcast-condition Jacobian apply expects d_condition shaped [B, P, D].")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    return d_condition.unsqueeze(2).expand(-1, -1, seq_len, -1)


class VectorMLPInterface(InterfaceModule):
    """Residual vector interface with MLP decoders, MLP encoders, and per-region normalization."""

    def __init__(
        self,
        *,
        feature_dim: int,
        num_regions: int,
        interface_width: int,
        interface_map_hidden_dim: int = 0,
        update_scale_init: float = 0.5,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be > 0")
        if num_regions <= 0:
            raise ValueError("num_regions must be > 0")
        if interface_width <= 0:
            raise ValueError("interface_width must be > 0")
        if interface_map_hidden_dim < 0:
            raise ValueError("interface_map_hidden_dim must be >= 0")
        if update_scale_init <= 0.0:
            raise ValueError("update_scale_init must be > 0")
        self.spec = InterfaceSpec(
            state_flat_dim=interface_width,
            region_condition_dim=feature_dim,
        )
        self.initial_encoder = VectorMLPHead(feature_dim, interface_width, hidden_dim=interface_map_hidden_dim)
        self.decoders = nn.ModuleList(
            [VectorMLPHead(interface_width, feature_dim, hidden_dim=interface_map_hidden_dim) for _ in range(num_regions)]
        )
        self.encoders = nn.ModuleList(
            [VectorMLPHead(feature_dim, interface_width, hidden_dim=interface_map_hidden_dim) for _ in range(num_regions)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(interface_width) for _ in range(num_regions)])
        self.update_scale = nn.Parameter(torch.full((num_regions,), float(update_scale_init)))

    @property
    def num_regions(self) -> int:
        return len(self.decoders)

    def validate_region_count(self, num_regions: int) -> None:
        if num_regions != self.num_regions:
            raise ValueError(
                "vector interface region count must match model region count "
                f"({self.num_regions} != {num_regions})"
            )

    @staticmethod
    def summarize_features(features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError("vector interface features must have shape [B, T, D]")
        pooled = features.mean(dim=1)
        if pooled.dtype != features.dtype:
            pooled = pooled.to(dtype=features.dtype)
        return pooled

    def initialize(self, canvas_features: torch.Tensor) -> torch.Tensor:
        return self.initial_encoder(self.summarize_features(canvas_features))

    def decode(self, state: torch.Tensor, region_index: int) -> torch.Tensor:
        return self.decoders[region_index](state)

    def update(
        self,
        state: torch.Tensor,
        region_features: torch.Tensor,
        region_index: int,
    ) -> InterfaceStep:
        pooled_features = self.summarize_features(region_features)
        delta_state = self.encoders[region_index](pooled_features)
        update_scale = torch.tanh(self.update_scale[region_index])
        pre_norm_state = state + update_scale * delta_state
        next_state = self.norms[region_index](pre_norm_state)
        return InterfaceStep(
            state=next_state,
            diagnostics={
                "pooled_features": pooled_features,
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

    def apply_update_jacobian_t_to_region_output(
        self,
        *,
        region_cache: Any,
        state_out_cotangent_basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if state_out_cotangent_basis.dim() != 3:
            raise ValueError("state_out_cotangent_basis must have shape [B, P, R].")
        diagnostics = region_cache.interface_step.diagnostics
        g_pre_norm = _module_input_jacobian_t_apply_autograd(
            self.norms[region_cache.region_index],
            diagnostics["pre_norm_state"],
            state_out_cotangent_basis,
        )
        g_state_skip = g_pre_norm
        g_delta_state = g_pre_norm * diagnostics["update_scale"].to(dtype=g_pre_norm.dtype).view(1, 1, 1)
        g_pooled_features = _interface_map_input_jacobian_t_apply(
            self.encoders[region_cache.region_index],
            diagnostics["pooled_features"],
            g_delta_state,
        )
        g_region_output = _mean_pool_jacobian_t_apply(g_pooled_features, seq_len=region_cache.region_output.shape[1])
        return {
            "g_pre_norm_state": g_pre_norm,
            "g_state_skip": g_state_skip,
            "g_delta_state": g_delta_state,
            "g_pooled_features": g_pooled_features,
            "g_region_output": g_region_output,
        }

    def apply_decode_jacobian_t_to_state_input(
        self,
        *,
        region_cache: Any,
        region_input_cotangent_basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if region_input_cotangent_basis.dim() != 4:
            raise ValueError("region_input_cotangent_basis must have shape [B, P, L, D].")
        g_condition = _broadcast_condition_jacobian_t_apply(region_input_cotangent_basis)
        g_state_input = _interface_map_input_jacobian_t_apply(
            self.decoders[region_cache.region_index],
            region_cache.state_in,
            g_condition,
        )
        return {
            "g_condition": g_condition,
            "g_state_input": g_state_input,
        }

    def apply_decode_jacobian_to_state_input(
        self,
        *,
        region_cache: Any,
        state_input_tangent_basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward-mode: push a state tangent basis [B, P, R] through decode ->
        region-input tangent basis [B, P, L, D] (decoder MLP JVP then broadcast).
        Exact adjoint of `apply_decode_jacobian_t_to_state_input`."""
        if state_input_tangent_basis.dim() != 3:
            raise ValueError("state_input_tangent_basis must have shape [B, P, R].")
        d_condition = _interface_map_input_jacobian_apply(
            self.decoders[region_cache.region_index],
            region_cache.state_in,
            state_input_tangent_basis,
        )
        seq_len = region_cache.region_output.shape[1]
        d_region_input = _broadcast_condition_jacobian_apply(d_condition, seq_len)
        return {
            "d_condition": d_condition,
            "d_region_input": d_region_input,
        }

    def apply_update_jacobian_to_region_output(
        self,
        *,
        region_cache: Any,
        region_output_tangent_basis: torch.Tensor,
        state_input_tangent_basis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward-mode: push a region-output tangent basis [B, P, L, D] plus the
        skip state tangent [B, P, R] through update -> state-out tangent basis
        [B, P, R]. Exact adjoint of `apply_update_jacobian_t_to_region_output`
        (which splits g_pre_norm into g_region_output + g_state_skip)."""
        if region_output_tangent_basis.dim() != 4:
            raise ValueError("region_output_tangent_basis must have shape [B, P, L, D].")
        if state_input_tangent_basis.dim() != 3:
            raise ValueError("state_input_tangent_basis must have shape [B, P, R].")
        diagnostics = region_cache.interface_step.diagnostics
        d_pooled_features = _mean_pool_jacobian_apply(region_output_tangent_basis)
        d_delta_state = _interface_map_input_jacobian_apply(
            self.encoders[region_cache.region_index],
            diagnostics["pooled_features"],
            d_pooled_features,
        )
        update_scale = diagnostics["update_scale"].to(dtype=d_delta_state.dtype).view(1, 1, 1)
        # pre_norm = state + update_scale * delta -> d_pre_norm = skip + scale*d_delta.
        d_pre_norm_state = state_input_tangent_basis.to(dtype=d_delta_state.dtype) + update_scale * d_delta_state
        d_state_out = _layernorm_input_jacobian_apply(
            self.norms[region_cache.region_index],
            diagnostics["pre_norm_state"],
            d_pre_norm_state,
        )
        return {
            "d_pooled_features": d_pooled_features,
            "d_delta_state": d_delta_state,
            "d_pre_norm_state": d_pre_norm_state,
            "d_state_out": d_state_out,
        }

