from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from backbones.general import BackboneSpec, _make_final_norm, init_transformer_module
from backends import (
    BackboneStackRegionBackend,
    Mamba3RegionBackend,
    RegionForwardCache,
    TransformerRegionBackend,
)
from canvas import TokenEmbeddingCanvas
from interfaces import InterfaceModule, InterfaceStep
from readouts import NormLMHeadReadout, StateReadout


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
        canvas_local_mixer: int = 0,
        canvas_region_view: bool = False,
        canvas_state_readout: bool = False,
        canvas_output_readout: bool = False,
        state_readout_attn_dim: int = 64,
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
            vocab_size=vocab_size, feature_dim=self.hidden_dim, local_mixer_kernel=canvas_local_mixer
        )
        # Per-region diagonal view of the shared canvas (identity at init).
        if canvas_region_view:
            self.canvas_view_gain = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.hidden_dim)) for _ in range(self.num_regions)]
            )
            self.canvas_view_bias = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.hidden_dim)) for _ in range(self.num_regions)]
            )
        else:
            self.canvas_view_gain = None
            self.canvas_view_bias = None
        if backbone_spec.name == "transformer":
            self.region_backend = TransformerRegionBackend(backbone_spec=backbone_spec, region_ranges=self.region_ranges)
        elif backbone_spec.name == "mamba3":
            self.region_backend = Mamba3RegionBackend(backbone_spec=backbone_spec, region_ranges=self.region_ranges)
        else:
            self.region_backend = BackboneStackRegionBackend(backbone_spec=backbone_spec, region_ranges=self.region_ranges)
        self.readout = NormLMHeadReadout(
            norm=_make_final_norm(backbone_spec),
            feature_dim=self.hidden_dim,
            vocab_size=vocab_size,
            tie_embeddings=self.tie_embeddings,
        )
        self.interface = interface
        if canvas_state_readout:
            self.state_readout = StateReadout(
                num_regions=self.num_regions,
                state_width=interface.spec.state_flat_dim,
                feature_dim=self.hidden_dim,
                tokenwise=bool(getattr(interface.spec, "condition_is_tokenwise", False)),
                attn_dim=state_readout_attn_dim,
            )
        else:
            self.state_readout = None
        # Full-width readout accumulation: every region's output enters the
        # readout stream through a zero-init gate (autograd engines only).
        if canvas_output_readout:
            self.output_readout_gates = nn.Parameter(torch.zeros(self.num_regions))
        else:
            self.output_readout_gates = None

        if backbone_spec.name in {"transformer", "hybrid"}:
            init_transformer_module(self.canvas, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
            self.region_backend.initialize_parameters(backbone_spec=backbone_spec)
            init_transformer_module(self.readout, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
        if self.tie_embeddings:
            nn.init.normal_(self.canvas.embedding.weight, std=0.02)

    def output_head_vjp_parameters(self) -> list[nn.Parameter]:
        """Readout parameters touched by loss-to-output local VJPs."""
        params = list(self.readout.vjp_parameters())
        if self.state_readout is not None:
            params.extend(p for p in self.state_readout.parameters() if p.requires_grad)
        if self.output_readout_gates is not None and self.output_readout_gates.requires_grad:
            params.append(self.output_readout_gates)
        return params

    def region_view_parameters(self, region_index: int) -> list[nn.Parameter]:
        """Per-region canvas-view parameters touched by that region's local VJP."""
        if self.canvas_view_gain is None:
            return []
        return [self.canvas_view_gain[region_index], self.canvas_view_bias[region_index]]

    def viewed_canvas(self, canvas_features: torch.Tensor, region_index: int) -> torch.Tensor:
        """The canvas as region `region_index` reads it (identity when the
        per-region view is disabled)."""
        if self.canvas_view_gain is None:
            return canvas_features
        gain = self.canvas_view_gain[region_index].to(dtype=canvas_features.dtype)
        bias = self.canvas_view_bias[region_index].to(dtype=canvas_features.dtype)
        return canvas_features * (1.0 + gain) + bias

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
        native_backward: bool = False,
        trim_region_cache: bool = False,
        detach_state_taps: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # native_backward runs the region body under no_grad (the native scan
        # backward rebuilds region grads from the frozen backend caches);
        # canvas / initialize / readout stay under grad. trim_region_cache
        # additionally keeps only each region input and recomputes region
        # forwards on demand.
        region_ctx = torch.no_grad() if native_backward else contextlib.nullcontext()
        if trim_region_cache and not native_backward:
            raise ValueError("trim_region_cache requires native_backward=True")
        prev_recompute = getattr(self.region_backend, "recompute_forward", None)
        if trim_region_cache:
            if prev_recompute is None:
                raise ValueError("region backend does not support trim_region_cache")
            self.region_backend.recompute_forward = True

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
            with region_ctx:
                condition = self.interface.decode(state_in, region_index, canvas_features=canvas_features)
                canvas_read = self.viewed_canvas(canvas_features, region_index)
                region_input = canvas_read + (condition if condition.dim() == 3 else condition.unsqueeze(1))
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

        if trim_region_cache:
            self.region_backend.recompute_forward = prev_recompute

        # The readout needs a grad-connected input: under native_backward the last
        # region output is detached, so re-expose it as a grad leaf (and record it
        # on its cache so the seed's dL/d(region_output) autograd can reach it).
        readout_input = region_outputs[-1]
        if native_backward:
            readout_input = readout_input.detach().requires_grad_(True)
            region_caches[-1].region_output = readout_input
        # Boundary-state taps into the readout stream. The scan engines need the
        # taps severed from the state chain (their direct cotangents become the
        # scan's source terms); the plain autograd engine needs them attached.
        detach_taps = native_backward if detach_state_taps is None else bool(detach_state_taps)
        readout_stream = readout_input
        state_taps: list[torch.Tensor] = []
        state_readout_terms: list[torch.Tensor] = []
        if self.state_readout is not None:
            for state_out in states[1:]:
                tap = state_out.detach().requires_grad_(True) if detach_taps else state_out
                state_taps.append(tap)
            state_readout_terms = self.state_readout.contributions(state_taps, canvas_features)
            for term in state_readout_terms:
                readout_stream = readout_stream + (term if term.dim() == 3 else term.unsqueeze(1))
        output_taps: list[torch.Tensor] = []
        if self.output_readout_gates is not None:
            # Gates are tanh-bounded: signed mixtures stay expressive (learned
            # values sit well inside +-1) while runaway amplification cannot.
            # Scan engines take severed output taps (their direct cotangents
            # become the scan's input-side source terms), mirroring state taps.
            for k, region_output in enumerate(region_outputs):
                tap = region_output.detach().requires_grad_(True) if detach_taps else region_output
                output_taps.append(tap)
                gate = torch.tanh(self.output_readout_gates[k]).to(dtype=tap.dtype)
                readout_stream = readout_stream + gate * tap
        logits = self.readout(readout_stream, canvas=self.canvas)
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
            "state_taps": state_taps,
            "state_readout_terms": state_readout_terms,
            "output_taps": output_taps,
        }


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_cache(input_ids)
        return logits
