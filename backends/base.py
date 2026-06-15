from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch
import torch.nn as nn


@dataclass
class RegionForwardCache:
    region_index: int
    layer_range: tuple[int, int]


class RegionBackend(Protocol):
    name: str

    def forward_region(
        self,
        *,
        region_input: torch.Tensor,
        region_index: int,
    ) -> tuple[torch.Tensor, RegionForwardCache]:
        ...

    def parameters_for_region(self, region_index: int) -> Sequence[nn.Parameter]:
        ...

    def count_parameters(self) -> int:
        ...

    def initialize_parameters(self, *, backbone_spec: Any) -> None:
        ...

    def input_pullback_basis(
        self,
        *,
        cache: RegionForwardCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def parameter_vjp(
        self,
        *,
        cache: RegionForwardCache,
        output_cotangent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ...
