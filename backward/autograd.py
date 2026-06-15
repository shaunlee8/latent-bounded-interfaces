from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from backward.base import ADResult, GradMap, collect_named_grads


class AutogradEngine:
    name = "autograd"

    def backward(
        self,
        *,
        model: nn.Module,
        loss: torch.Tensor,
        cache: Any | None = None,
    ) -> ADResult:
        del cache
        model.zero_grad(set_to_none=True)
        loss.backward()
        return ADResult(grad_map=collect_named_grads(model))


def autograd_backward_step(model: nn.Module, *, ce_loss: torch.Tensor) -> GradMap:
    return AutogradEngine().backward(model=model, loss=ce_loss).grad_map
