from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn

from canvas import CanvasModule


@dataclass
class ReadoutCache:
    features: torch.Tensor
    normed_features: torch.Tensor
    logits: torch.Tensor


class ReadoutModule(Protocol):
    """Maps final region features to logits and exposes readout VJP parameters."""

    def __call__(self, features: torch.Tensor, *, canvas: CanvasModule) -> torch.Tensor: ...

    def forward(self, features: torch.Tensor, *, canvas: CanvasModule) -> torch.Tensor: ...

    def forward_with_cache(self, features: torch.Tensor, *, canvas: CanvasModule) -> tuple[torch.Tensor, ReadoutCache]: ...

    def vjp_parameters(self) -> list[nn.Parameter]: ...
