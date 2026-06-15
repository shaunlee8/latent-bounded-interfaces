from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn


@dataclass
class CanvasCache:
    input_ids: torch.Tensor
    features: torch.Tensor


class CanvasModule(Protocol):
    """Token-to-canvas feature interface used by LBI models."""

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...

    def forward_with_cache(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, CanvasCache]:
        ...

    def output_weight(self) -> torch.Tensor:
        ...

    def vjp_parameters(self) -> list[nn.Parameter]:
        ...
