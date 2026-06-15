from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from backbones.general import BackboneSpec, _make_final_norm, init_transformer_module
from backends import BackboneStackRegionBackend, RegionForwardCache, TransformerRegionBackend
from canvas import TokenEmbeddingCanvas
from interfaces import InterfaceModule, InterfaceStep
from readouts import NormLMHeadReadout


_VALID_STATE_ABLATIONS = {"none", "zero_all", "noise", "mask"}


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


@dataclass
class LBICache:
    canvas_features: torch.Tensor
    states: list[torch.Tensor]
    region_inputs: list[torch.Tensor]
    region_outputs: list[torch.Tensor]
    region_ranges: list[tuple[int, int]]
    region_caches: list[LBIRegionCache]


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

        self.canvas = TokenEmbeddingCanvas(vocab_size=vocab_size, feature_dim=self.hidden_dim)
        if backbone_spec.name == "transformer":
            self.region_backend = TransformerRegionBackend(backbone_spec=backbone_spec, region_ranges=self.region_ranges)
        else:
            self.region_backend = BackboneStackRegionBackend(backbone_spec=backbone_spec, region_ranges=self.region_ranges)
        self.readout = NormLMHeadReadout(
            norm=_make_final_norm(backbone_spec),
            feature_dim=self.hidden_dim,
            vocab_size=vocab_size,
            tie_embeddings=self.tie_embeddings,
        )
        self.interface = interface

        if backbone_spec.name in {"transformer", "hybrid"}:
            init_transformer_module(self.canvas, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
            self.region_backend.initialize_parameters(backbone_spec=backbone_spec)
            init_transformer_module(self.readout, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
        if self.tie_embeddings:
            nn.init.normal_(self.canvas.embedding.weight, std=0.02)

    def output_head_vjp_parameters(self) -> list[nn.Parameter]:
        """Readout parameters touched by loss-to-output local VJPs."""
        return self.readout.vjp_parameters()

    def canvas_vjp_parameters(self) -> list[nn.Parameter]:
        """Canvas parameters touched by local VJPs."""
        return self.canvas.vjp_parameters()

    def shared_local_vjp_parameters(self) -> list[nn.Parameter]:
        """Shared non-region parameters touched at the end of scan backward."""
        params = list(self.canvas_vjp_parameters())
        params.extend(self.interface.shared_vjp_parameters())
        return params

    def _apply_state_ablation(
        self,
        state: torch.Tensor,
        *,
        mode: str,
        noise_std: float,
        mask_keep_prob: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if mode not in _VALID_STATE_ABLATIONS:
            raise ValueError(f"unsupported state ablation mode: {mode}")
        if mode == "none":
            return state
        if mode == "zero_all":
            return torch.zeros_like(state)
        if mode == "noise":
            if noise_std < 0.0:
                raise ValueError("message_noise_std must be >= 0")
            if noise_std == 0.0:
                return state
            noise = torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
            return state + (noise_std * noise)
        if mask_keep_prob <= 0.0 or mask_keep_prob > 1.0:
            raise ValueError("message_mask_keep_prob must be in (0, 1]")
        if mask_keep_prob == 1.0:
            return state
        mask = torch.rand(state.shape, device=state.device, generator=generator) < mask_keep_prob
        return state * mask.to(dtype=state.dtype)

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        *,
        message_ablation: str = "none",
        message_noise_std: float = 1.0,
        message_mask_keep_prob: float = 0.5,
        ablation_generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        canvas_features = self.canvas(input_ids)
        state = self.interface.initialize(canvas_features)
        state = self._apply_state_ablation(
            state,
            mode=message_ablation,
            noise_std=message_noise_std,
            mask_keep_prob=message_mask_keep_prob,
            generator=ablation_generator,
        )
        states: list[torch.Tensor] = [state]
        region_inputs: list[torch.Tensor] = []
        region_outputs: list[torch.Tensor] = []
        region_caches: list[LBIRegionCache] = []

        for region_index, (start, end) in enumerate(self.region_ranges):
            state_in = state
            condition = self.interface.decode(state_in, region_index)
            region_input = canvas_features + condition.unsqueeze(1)
            region_output, backend_cache = self.region_backend.forward_region(
                region_input=region_input,
                region_index=region_index,
            )
            interface_step = self.interface.update(state_in, region_output, region_index)
            state = self._apply_state_ablation(
                interface_step.state,
                mode=message_ablation,
                noise_std=message_noise_std,
                mask_keep_prob=message_mask_keep_prob,
                generator=ablation_generator,
            )
            states.append(state)
            region_inputs.append(region_input)
            region_outputs.append(region_output)
            region_caches.append(
                LBIRegionCache(
                    region_index=region_index,
                    layer_range=(start, end),
                    backend_cache=backend_cache,
                    canvas_features=canvas_features,
                    state_in=state_in,
                    condition=condition,
                    region_input=region_input,
                    region_output=region_output,
                    interface_step=interface_step,
                    state_out=state,
                )
            )

        logits = self.readout(region_outputs[-1], canvas=self.canvas)
        cache = LBICache(
            canvas_features=canvas_features,
            states=states,
            region_inputs=region_inputs,
            region_outputs=region_outputs,
            region_ranges=self.region_ranges,
            region_caches=region_caches,
        )
        return logits, {
            "canvas_features": cache.canvas_features,
            "states": cache.states,
            "region_inputs": cache.region_inputs,
            "region_outputs": cache.region_outputs,
            "region_ranges": cache.region_ranges,
            "region_caches": cache.region_caches,
        }


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_cache(input_ids)
        return logits
