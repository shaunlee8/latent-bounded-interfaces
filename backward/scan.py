from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from backward.base import ADResult, LBIBackwardResult, ScanBackpropModel, assign_grad_map
from backward.local_vjp import LocalVJPProvider, TorchAutogradLocalVJPProvider
from backward.pullbacks import InterfacePullbackProvider, build_interface_pullback_provider
from backward.suffix_scan import (
    apply_jacobian_t,
    interface_state_jacobian_t_stats,
    propagate_state_adjoint_from_last_region_input,
)


def lbi_scan_backward_step(
    model: ScanBackpropModel,
    *,
    ce_loss: torch.Tensor,
    cache: dict[str, Any],
    pullback_provider: InterfacePullbackProvider,
    local_vjp_provider: LocalVJPProvider | None = None,
    compute_interface_jacobian_stats: bool = False,
    include_interface_jacobian_suffix: bool = False,
) -> LBIBackwardResult:
    """Exact scan backward for an LBI model using a supplied interface-pullback provider."""
    local_vjp_provider = local_vjp_provider or TorchAutogradLocalVJPProvider()
    grad_map = local_vjp_provider.new_grad_map(model)

    states = list(cache["states"])
    region_ranges = list(cache["region_ranges"])
    num_regions = len(region_ranges)
    if num_regions == 0:
        raise RuntimeError("LBI scan backward requires at least one region.")

    local_vjp_provider.store_output_head_grads(model=model, loss=ce_loss, grad_map=grad_map)

    if num_regions == 1:
        g_state_inputs = [
            local_vjp_provider.state_adjoint_from_loss(loss=ce_loss, state=states[0])
        ]
        interface_scan_rms = 0.0
        interface_jacobian_stats = None
    else:
        g_last_input = local_vjp_provider.state_adjoint_from_loss(loss=ce_loss, state=states[-2])
        state_jacobians_t = pullback_provider.materialize_state_jacobian_t(model=model, cache=cache)
        g_last_input = g_last_input.to(device=state_jacobians_t[0].device, dtype=state_jacobians_t[0].dtype)
        interface_jacobian_stats = (
            interface_state_jacobian_t_stats(
                state_jacobians_t,
                include_suffix=include_interface_jacobian_suffix,
            )
            if compute_interface_jacobian_stats
            else None
        )
        g_state_inputs = propagate_state_adjoint_from_last_region_input(
            state_jacobians_t,
            g_last_input,
            num_regions=num_regions,
        )
        g_manual = [torch.zeros_like(g_last_input) for _ in range(num_regions)]
        g_manual[-1] = g_last_input
        for region_index in reversed(range(num_regions - 1)):
            g_manual[region_index] = apply_jacobian_t(
                state_jacobians_t[region_index],
                g_manual[region_index + 1],
                out_dtype=g_manual[region_index + 1].dtype,
            )
        diffs = [((a - b) ** 2).mean() for a, b in zip(g_state_inputs, g_manual)]
        interface_scan_rms = float(torch.sqrt(torch.stack(diffs).mean()).item())

    local_vjp_provider.store_initial_interface_grads(
        model=model,
        state0=states[0],
        state0_adjoint=g_state_inputs[0],
        grad_map=grad_map,
    )

    for region_index in range(num_regions):
        local_vjp_provider.store_region_grads(
            model=model,
            loss=ce_loss,
            states=states,
            region_index=region_index,
            num_regions=num_regions,
            state_adjoints=g_state_inputs,
            grad_map=grad_map,
        )

    local_vjp_provider.store_shared_canvas_grads(model=model, loss=ce_loss, grad_map=grad_map)
    return LBIBackwardResult(
        grad_map=grad_map,
        interface_scan_rms=interface_scan_rms,
        interface_jacobian_stats=interface_jacobian_stats,
    )


class ScanADEngine:
    name = "scan"

    def __init__(
        self,
        *,
        pullback_provider: InterfacePullbackProvider | None = None,
        local_vjp_provider: LocalVJPProvider | None = None,
        state_jacobian_mode: str = "graph",
        state_jacobian_basis_chunk: int = 1,
        compute_interface_jacobian_stats: bool = False,
        include_interface_jacobian_suffix: bool = False,
    ) -> None:
        self.pullback_provider = pullback_provider or build_interface_pullback_provider(
            state_jacobian_mode,
            basis_chunk=state_jacobian_basis_chunk,
        )
        self.local_vjp_provider = local_vjp_provider or TorchAutogradLocalVJPProvider()
        self.compute_interface_jacobian_stats = bool(compute_interface_jacobian_stats)
        self.include_interface_jacobian_suffix = bool(include_interface_jacobian_suffix)

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        compute_interface_jacobian_stats: bool = False,
    ) -> "ScanADEngine":
        return cls(
            pullback_provider=build_interface_pullback_provider(
                str(cfg.interface_jacobian_mode),
                basis_chunk=int(cfg.jacobian_basis_chunk),
            ),
            local_vjp_provider=TorchAutogradLocalVJPProvider(),
            compute_interface_jacobian_stats=compute_interface_jacobian_stats,
            include_interface_jacobian_suffix=bool(cfg.log_interface_jacobian_suffix),
        )

    def backward(
        self,
        *,
        model: nn.Module,
        loss: torch.Tensor,
        cache: Any | None = None,
    ) -> ADResult:
        if not isinstance(model, ScanBackpropModel):
            raise TypeError("ScanADEngine requires a model satisfying ScanBackpropModel")
        if cache is None:
            raise ValueError("ScanADEngine requires a forward cache")
        model.zero_grad(set_to_none=True)
        result = lbi_scan_backward_step(
            model,
            ce_loss=loss,
            cache=cache,
            pullback_provider=self.pullback_provider,
            local_vjp_provider=self.local_vjp_provider,
            compute_interface_jacobian_stats=self.compute_interface_jacobian_stats,
            include_interface_jacobian_suffix=self.include_interface_jacobian_suffix,
        )
        assign_grad_map(model, result.grad_map)
        diagnostics: dict[str, Any] = {
            "interface_scan_rms": result.interface_scan_rms,
            "interface_pullback_provider": self.pullback_provider.name,
            "local_vjp_provider": self.local_vjp_provider.name,
        }
        if result.interface_jacobian_stats is not None:
            diagnostics["interface_jacobian_stats"] = dict(result.interface_jacobian_stats)
        return ADResult(grad_map=result.grad_map, diagnostics=diagnostics)


# Alias used by callers that name the scan engine by algorithm role.
ReferenceScanEngine = ScanADEngine
