from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn


GradMap = dict[str, torch.Tensor | None]


@dataclass
class ADResult:
    grad_map: GradMap
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class LBIBackwardResult:
    grad_map: GradMap
    interface_scan_rms: float
    interface_jacobian_stats: dict[str, float] | None = None



@runtime_checkable
class ScanBackpropModel(Protocol):
    """Structural model interface required by scan-based LBI backprop."""

    num_regions: int
    region_ranges: list[tuple[int, int]]
    interface: Any
    region_backend: Any

    def named_parameters(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        ...

    def output_head_vjp_parameters(self) -> list[nn.Parameter]:
        ...

    def canvas_vjp_parameters(self) -> list[nn.Parameter]:
        ...

    def shared_local_vjp_parameters(self) -> list[nn.Parameter]:
        ...


class ADEngine(Protocol):
    name: str

    def backward(
        self,
        *,
        model: nn.Module,
        loss: torch.Tensor,
        cache: Any | None = None,
    ) -> ADResult:
        ...


def collect_named_grads(model: nn.Module) -> GradMap:
    grad_map: GradMap = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad_map[name] = None if param.grad is None else param.grad.detach().clone()
    return grad_map


def assign_grad_map(model: nn.Module, grad_map: GradMap) -> None:
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = grad_map.get(name)
        param.grad = None if grad is None else grad.detach().clone()


def store_named_grads(
    grad_map: GradMap,
    param_name_map: dict[int, str],
    params: list[nn.Parameter],
    grads: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None],
) -> None:
    for param, grad in zip(params, grads):
        name = param_name_map[id(param)]
        grad_map[name] = None if grad is None else grad.detach().clone()
