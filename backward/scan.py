from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from backward.base import ADResult, LBIBackwardResult, ScanBackpropModel, assign_grad_map
from backward.local_vjp import LocalVJPProvider, TorchAutogradLocalVJPProvider
from backward.pullbacks import InterfacePullbackProvider, build_interface_pullback_provider
from backward.suffix_scan import propagate_state_adjoint_from_last_region_input


def lbi_scan_backward_step(
    model: ScanBackpropModel,
    *,
    ce_loss: torch.Tensor,
    cache: dict[str, Any],
    pullback_provider: InterfacePullbackProvider,
    local_vjp_provider: LocalVJPProvider | None = None,
) -> LBIBackwardResult:
    """Exact scan backward for an LBI model using a supplied interface-pullback provider."""
    local_vjp_provider = local_vjp_provider or TorchAutogradLocalVJPProvider()
    grad_map = local_vjp_provider.new_grad_map(model)

    states = list(cache["states"])
    region_ranges = list(cache["region_ranges"])
    num_regions = len(region_ranges)
    if num_regions == 0:
        raise RuntimeError("LBI scan backward requires at least one region.")

    local_vjp_provider.store_output_head_grads(model=model, loss=ce_loss, grad_map=grad_map, cache=cache)

    if num_regions == 1:
        g_state_inputs = [
            local_vjp_provider.state_adjoint_from_loss(model=model, loss=ce_loss, state=states[0], cache=cache)
        ]
    else:
        g_last_input = local_vjp_provider.state_adjoint_from_loss(model=model, loss=ce_loss, state=states[-2], cache=cache)
        state_jacobians_t = pullback_provider.materialize_state_jacobian_t(model=model, cache=cache)
        g_last_input = g_last_input.to(device=state_jacobians_t[0].device, dtype=state_jacobians_t[0].dtype)
        g_state_inputs = propagate_state_adjoint_from_last_region_input(
            state_jacobians_t,
            g_last_input,
            num_regions=num_regions,
        )

    local_vjp_provider.store_initial_interface_grads(
        model=model,
        state0=states[0],
        state0_adjoint=g_state_inputs[0],
        grad_map=grad_map,
        cache=cache,
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
            cache=cache,
        )

    local_vjp_provider.store_shared_canvas_grads(model=model, loss=ce_loss, grad_map=grad_map, cache=cache)
    return LBIBackwardResult(grad_map=grad_map)


class ScanADEngine:
    name = "scan"

    def __init__(
        self,
        *,
        pullback_provider: InterfacePullbackProvider | None = None,
        local_vjp_provider: LocalVJPProvider | None = None,
        state_jacobian_mode: str = "graph",
    ) -> None:
        self.pullback_provider = pullback_provider or build_interface_pullback_provider(state_jacobian_mode)
        self.local_vjp_provider = local_vjp_provider or TorchAutogradLocalVJPProvider()

    @classmethod
    def from_config(cls, cfg: Any) -> "ScanADEngine":
        # native_backward selects the autograd-free providers (A_k construction
        # per interface_jacobian_mode); otherwise autograd providers.
        if bool(getattr(cfg, "native_backward", False)):
            from backward.local_vjp import NativeLocalVJPProvider
            from backward.pullbacks import (
                ForwardModeInterfacePullbackProvider,
                NativeInterfacePullbackProvider,
            )

            mode = str(getattr(cfg, "interface_jacobian_mode", "native")).lower()
            if mode in {"forward", "forward_mode", "fwd", "jvp"}:
                pullback_provider: InterfacePullbackProvider = ForwardModeInterfacePullbackProvider()
            else:
                pullback_provider = NativeInterfacePullbackProvider()
            local_vjp_provider: LocalVJPProvider = NativeLocalVJPProvider()
        else:
            pullback_provider = build_interface_pullback_provider(str(cfg.interface_jacobian_mode))
            local_vjp_provider = TorchAutogradLocalVJPProvider()
        return cls(pullback_provider=pullback_provider, local_vjp_provider=local_vjp_provider)

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
        )
        assign_grad_map(model, result.grad_map)
        diagnostics: dict[str, Any] = {
            "interface_pullback_provider": self.pullback_provider.name,
            "local_vjp_provider": self.local_vjp_provider.name,
        }
        return ADResult(grad_map=result.grad_map, diagnostics=diagnostics)

