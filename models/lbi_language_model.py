from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from backbones.general import BackboneSpec, _make_final_norm, init_transformer_module
from backends import (
    Mamba3RegionBackend,
    RegionForwardCache,
    TransformerRegionBackend,
)
from canvas import TokenEmbeddingCanvas
from canvas.region_views import RegionView, SharedView
from interfaces import InterfaceModule, InterfaceStep
from readouts import NormLMHeadReadout


def build_region_ranges(layers: int, layers_per_region: int) -> list[tuple[int, int]]:
    if layers <= 0:
        raise ValueError("layers must be > 0")
    if layers_per_region <= 0:
        raise ValueError("layers_per_region must be > 0")
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < layers:
        end = min(layers, start + layers_per_region)
        ranges.append((start, end))
        start = end
    return ranges


@dataclass
class LBIRegionCache:
    region_index: int
    layer_range: tuple[int, int]
    backend_cache: RegionForwardCache
    canvas_features: torch.Tensor
    state_in: torch.Tensor
    condition: torch.Tensor
    region_input: torch.Tensor
    region_output: torch.Tensor
    interface_step: InterfaceStep
    state_out: torch.Tensor


class LBILanguageModel(nn.Module):
    """Language model composed from canvas, interface, region backend, and readout modules."""

    def __init__(
        self,
        *,
        vocab_size: int,
        layers_per_region: int,
        backbone_spec: BackboneSpec,
        interface: InterfaceModule,
        tie_embeddings: bool = False,
        region_view: RegionView | None = None,
    ) -> None:
        super().__init__()
        backbone_spec.validate()
        if interface.spec.region_condition_dim != backbone_spec.dim:
            raise ValueError(
                "interface region_condition_dim must match backbone hidden dimension "
                f"({interface.spec.region_condition_dim} != {backbone_spec.dim})"
            )
        self.hidden_dim = backbone_spec.dim
        self.layers = backbone_spec.layers
        self.interface_width = interface.spec.state_flat_dim
        self.tie_embeddings = bool(tie_embeddings)
        self.region_ranges = build_region_ranges(backbone_spec.layers, layers_per_region)
        self.num_regions = len(self.region_ranges)
        interface.validate_region_count(self.num_regions)

        self.canvas = TokenEmbeddingCanvas(
            vocab_size=vocab_size, feature_dim=self.hidden_dim
        )
        # Topology: what each region reads from the canvas (canvas/region_views).
        if region_view is None:
            region_view = SharedView(self.num_regions)
        if region_view.num_regions != self.num_regions:
            raise ValueError(
                f"region view built for {region_view.num_regions} regions, model has {self.num_regions}")
        self.region_view = region_view
        if backbone_spec.name == "transformer":
            self.region_backend = TransformerRegionBackend(backbone_spec=backbone_spec, region_ranges=self.region_ranges)
        elif backbone_spec.name == "mamba3":
            self.region_backend = Mamba3RegionBackend(backbone_spec=backbone_spec, region_ranges=self.region_ranges)
        elif backbone_spec.name == "hybrid":
            from backends.hybrid import HybridRegionBackend

            self.region_backend = HybridRegionBackend(backbone_spec=backbone_spec, region_ranges=self.region_ranges)
        else:
            raise ValueError(f"unsupported backbone for LBI regions: {backbone_spec.name}")
        self.readout = NormLMHeadReadout(
            norm=_make_final_norm(backbone_spec),
            feature_dim=self.hidden_dim,
            vocab_size=vocab_size,
            tie_embeddings=self.tie_embeddings,
        )
        self.interface = interface

        if backbone_spec.name in ("transformer", "hybrid"):
            init_transformer_module(self.canvas, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
            self.region_backend.initialize_parameters(backbone_spec=backbone_spec)
            init_transformer_module(self.readout, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
        if self.tie_embeddings:
            nn.init.normal_(self.canvas.embedding.weight, std=0.02)

    def output_head_vjp_parameters(self) -> list[nn.Parameter]:
        """Readout parameters touched by loss-to-output local VJPs."""
        return list(self.readout.vjp_parameters())

    def viewed_canvas(self, canvas_features: torch.Tensor, region_index: int) -> torch.Tensor:
        """The canvas as region `region_index` reads it (see canvas/region_views)."""
        return self.region_view.apply(canvas_features, region_index)

    def initial_state(self, canvas_features: torch.Tensor) -> torch.Tensor:
        """m_0: the interface's initial write, from what the view lets it read."""
        return self.interface.initialize(self.region_view.initial_write(canvas_features))

    def canvas_vjp_parameters(self) -> list[nn.Parameter]:
        """Canvas parameters touched by local VJPs."""
        return self.canvas.vjp_parameters()

    def shared_local_vjp_parameters(self) -> list[nn.Parameter]:
        """Shared non-region parameters touched at the end of scan backward."""
        params = list(self.canvas_vjp_parameters())
        params.extend(self.interface.shared_vjp_parameters())
        return params

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        *,
        native_backward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # native_backward runs the region body under no_grad (grads rebuilt from
        # frozen backend caches); canvas / initialize / readout stay under grad.
        region_ctx = torch.no_grad() if native_backward else contextlib.nullcontext()

        canvas_features = self.canvas(input_ids)
        state = self.initial_state(canvas_features)
        states: list[torch.Tensor] = [state]
        region_inputs: list[torch.Tensor] = []
        region_outputs: list[torch.Tensor] = []
        region_caches: list[LBIRegionCache] = []

        for region_index, (start, end) in enumerate(self.region_ranges):
            state_in = state
            with region_ctx:
                # Both canvas reads route through the view; the cache records the
                # canvas the region saw.
                decode_canvas = self.region_view.decode_canvas(canvas_features, region_index)
                condition = self.interface.decode(state_in, region_index)
                canvas_read = self.viewed_canvas(canvas_features, region_index)
                condition_wide = condition if condition.dim() == 3 else condition.unsqueeze(1)
                region_input = canvas_read + condition_wide
                region_output, backend_cache = self.region_backend.forward_region(
                    region_input=region_input,
                    region_index=region_index,
                )
                interface_step = self.interface.update(state_in, region_output, region_index)
                state = interface_step.state
            states.append(state)
            region_inputs.append(region_input)
            region_outputs.append(region_output)
            region_caches.append(
                LBIRegionCache(
                    region_index=region_index,
                    layer_range=(start, end),
                    backend_cache=backend_cache,
                    canvas_features=decode_canvas,
                    state_in=state_in,
                    condition=condition,
                    region_input=region_input,
                    region_output=region_output,
                    interface_step=interface_step,
                    state_out=state,
                )
            )

        # Under native_backward the last region output is detached; re-expose it
        # as a grad leaf so the seed's dL/d(region_output) can reach it.
        readout_input = region_outputs[-1]
        if native_backward:
            readout_input = readout_input.detach().requires_grad_(True)
            region_caches[-1].region_output = readout_input
        logits = self.readout(readout_input, canvas=self.canvas)
        return logits, {
            "canvas_features": canvas_features,
            "states": states,
            "region_inputs": region_inputs,
            "region_outputs": region_outputs,
            "region_ranges": self.region_ranges,
            "region_caches": region_caches,
        }

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_cache(input_ids)
        return logits
