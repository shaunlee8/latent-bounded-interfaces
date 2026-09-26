from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class CanvasModule(Protocol):
    """Token-to-canvas feature map used by LBI models."""

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...

    def output_weight(self) -> torch.Tensor:
        ...

    def vjp_parameters(self) -> list[nn.Parameter]:
        ...
