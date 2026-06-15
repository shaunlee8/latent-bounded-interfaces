from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from backbones.general import BackboneSpec, BackboneStack, build_backbone_stack, init_transformer_module
from backends.base import RegionForwardCache


class BackboneStackRegionBackend(nn.Module):
    """Executes contiguous layer ranges from a BackboneStack as LBI regions."""

    name = "backbone_stack"

    def __init__(
        self,
        *,
        backbone_spec: BackboneSpec | None = None,
        region_ranges: Sequence[tuple[int, int]],
        backbone: BackboneStack | None = None,
    ) -> None:
        super().__init__()
        if backbone is None:
            if backbone_spec is None:
                raise ValueError("backbone_spec is required when backbone is not provided")
            backbone = build_backbone_stack(backbone_spec)
        self.backbone = backbone
        self.region_ranges = list(region_ranges)
        if not self.region_ranges:
            raise ValueError("region backend requires at least one region")

    def _region_range(self, region_index: int) -> tuple[int, int]:
        try:
            return self.region_ranges[region_index]
        except IndexError as exc:
            raise IndexError(f"region_index {region_index} out of range for {len(self.region_ranges)} regions") from exc

    def forward_region(
        self,
        *,
        region_input: torch.Tensor,
        region_index: int,
    ) -> tuple[torch.Tensor, RegionForwardCache]:
        start, end = self._region_range(region_index)
        region_output = self.backbone.forward_range(region_input, start, end)
        return region_output, RegionForwardCache(region_index=region_index, layer_range=(start, end))

    def parameters_for_region(self, region_index: int) -> list[nn.Parameter]:
        start, end = self._region_range(region_index)
        params: list[nn.Parameter] = []
        for layer_index in range(start, end):
            params.extend([p for p in self.backbone.blocks[layer_index].parameters() if p.requires_grad])
        return params

    def count_parameters(self) -> int:
        return int(sum(p.numel() for block in self.backbone.blocks for p in block.parameters()))

    def initialize_parameters(self, *, backbone_spec: BackboneSpec) -> None:
        if backbone_spec.name == "transformer":
            init_transformer_module(self.backbone, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
        elif backbone_spec.name == "hybrid":
            for layer_type, block in zip(backbone_spec.layer_types, self.backbone.blocks):
                if layer_type == "transformer":
                    init_transformer_module(block, n_layers=backbone_spec.layers, n_residuals_per_layer=2)

    def input_pullback_basis(
        self,
        *,
        cache: RegionForwardCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("generic backbone-stack input pullback basis is not implemented")

    def parameter_vjp(
        self,
        *,
        cache: RegionForwardCache,
        output_cotangent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("generic backbone-stack parameter VJP is not implemented")

