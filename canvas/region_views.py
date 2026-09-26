"""Per-region views of the canvas. A view decides what region k reads from
the canvas; the paper's shared view gives every region the full canvas."""

from __future__ import annotations

import torch
import torch.nn as nn


class RegionView(nn.Module):
    """What region k reads from the canvas, and what the initial write sees."""

    kind: str = "shared"

    def __init__(self, num_regions: int) -> None:
        super().__init__()
        if num_regions <= 0:
            raise ValueError("num_regions must be > 0")
        self.num_regions = int(num_regions)

    def apply(self, canvas_features: torch.Tensor, region_index: int) -> torch.Tensor:
        """The canvas as region `region_index` reads it (the additive read)."""
        return canvas_features

    def decode_canvas(self, canvas_features: torch.Tensor, region_index: int) -> torch.Tensor:
        """The canvas recorded with region k's cache."""
        return canvas_features

    def read_vjp(self, g: torch.Tensor, region_index: int) -> torch.Tensor:
        """Pull a cotangent on region k's additive read back to the canvas."""
        return g

    def initial_write(self, canvas_features: torch.Tensor) -> torch.Tensor:
        """What the interface's initial encoder (state m_0) reads."""
        return canvas_features

    def extra_repr(self) -> str:
        return f"kind={self.kind}, num_regions={self.num_regions}"


class SharedView(RegionView):
    kind = "shared"


def build_region_view(kind: str, *, num_regions: int, feature_dim: int, segments: int = 0) -> RegionView:
    del feature_dim, segments
    if kind == "shared":
        return SharedView(num_regions)
    raise ValueError(f"unknown canvas view {kind!r}; this release provides the shared view")


__all__ = ["RegionView", "SharedView", "build_region_view"]
