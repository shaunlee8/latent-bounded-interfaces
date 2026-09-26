"""Hybrid (mixed Mamba-3 / Transformer) region backend. A mixed region
decomposes into maximal single-family runs, which the family backends execute
against the shared block list; forward and JVP compose the runs left to right,
cotangents and parameter VJPs right to left. The residual stream closes at each
run boundary (output = hidden + residual, reopened with residual=None), which
under fused add-norm differs from continuous threading only by one
residual-precision rounding per boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn

from backbones.general import BackboneSpec, BackboneStack, build_backbone_stack, init_transformer_module

_MIRRORED_ENGINE_FLAGS = ("forward_mode_use_kernel", "compile_forward_stages")


def _split_runs(layer_types: Sequence[str], start: int, end: int) -> list[tuple[str, int, int]]:
    runs: list[tuple[str, int, int]] = []
    run_start = start
    for i in range(start + 1, end + 1):
        if i == end or layer_types[i] != layer_types[run_start]:
            runs.append((layer_types[run_start], run_start, i))
            run_start = i
    return runs


@dataclass
class HybridRegionCache:
    region_index: int
    layer_range: tuple[int, int]
    region_input: torch.Tensor | None = None
    region_output: torch.Tensor | None = None
    # (family, sub-backend run index, family region cache) per run, in order.
    runs: list[tuple[str, int, Any]] = field(default_factory=list)

    @property
    def layer_caches(self) -> list[Any]:
        return [lc for _, _, cache in self.runs for lc in getattr(cache, "layer_caches", [])]


class HybridRegionBackend(nn.Module):
    """Executes mixed Mamba-3/Transformer block ranges via per-family sub-backends."""

    name = "hybrid"

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
        if backbone_spec is None or backbone_spec.layer_types is None:
            raise ValueError("hybrid region backend requires backbone_spec.layer_types")
        self.backbone = backbone
        self.backbone_spec = backbone_spec
        self.region_ranges = list(region_ranges)
        if not self.region_ranges:
            raise ValueError("hybrid region backend requires at least one region")

        # Decompose every region into single-family runs and hand each family
        # its runs as that sub-backend's "regions" over the shared stack.
        layer_types = backbone_spec.layer_types
        self.region_runs: list[list[tuple[str, int]]] = []  # per region: (family, sub run index)
        family_runs: dict[str, list[tuple[int, int]]] = {"mamba3": [], "transformer": []}
        for start, end in self.region_ranges:
            entries: list[tuple[str, int]] = []
            for family, a, b in _split_runs(layer_types, start, end):
                entries.append((family, len(family_runs[family])))
                family_runs[family].append((a, b))
            self.region_runs.append(entries)

        from backends.mamba3 import Mamba3RegionBackend
        from backends.transformer import TransformerRegionBackend

        self._subs = nn.ModuleDict()
        if family_runs["mamba3"]:
            self._subs["mamba3"] = Mamba3RegionBackend(
                backbone=backbone, region_ranges=family_runs["mamba3"]
            )
        if family_runs["transformer"]:
            self._subs["transformer"] = TransformerRegionBackend(
                backbone=backbone, region_ranges=family_runs["transformer"]
            )

    # Engine code configures flags on the model's region backend; mirror the
    # known ones onto the family sub-backends.
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in _MIRRORED_ENGINE_FLAGS and hasattr(self, "_subs"):
            for sub in self._subs.values():
                setattr(sub, name, value)

    def _region_runs(self, region_index: int) -> list[tuple[str, int]]:
        try:
            return self.region_runs[region_index]
        except IndexError as exc:
            raise IndexError(
                f"region_index {region_index} out of range for {len(self.region_runs)} regions"
            ) from exc

    # ---- forward ----
    def forward_region(
        self,
        *,
        region_input: torch.Tensor,
        region_index: int,
    ) -> tuple[torch.Tensor, HybridRegionCache]:
        x = region_input
        runs: list[tuple[str, int, Any]] = []
        for family, sub_index in self._region_runs(region_index):
            x, sub_cache = self._subs[family].forward_region(region_input=x, region_index=sub_index)
            runs.append((family, sub_index, sub_cache))
        return x, HybridRegionCache(
            region_index=region_index,
            layer_range=self.region_ranges[region_index],
            region_input=region_input,
            region_output=x,
            runs=runs,
        )

    # ---- forward-mode construction ----
    def region_output_jvp(
        self,
        *,
        cache: HybridRegionCache,
        region_input_tangent_basis: torch.Tensor,
        compute_dtype: torch.dtype | None = None,
        pooled: bool = False,
        tangent_token_start: int = 0,
        direction_major_out: bool = False,
    ) -> torch.Tensor:
        basis = region_input_tangent_basis
        n = len(cache.runs)
        for i, (family, _sub_index, sub_cache) in enumerate(cache.runs):
            last = i == n - 1
            kwargs: dict[str, Any] = dict(
                cache=sub_cache,
                region_input_tangent_basis=basis,
                compute_dtype=compute_dtype,
                tangent_token_start=tangent_token_start if i == 0 else 0,
            )
            if last:
                kwargs["pooled"] = pooled
            if family == "mamba3":
                kwargs["direction_major_out"] = direction_major_out if last else False
            basis = self._subs[family].region_output_jvp(**kwargs)
        return basis

    # ---- reverse surfaces ----
    def input_pullback_basis(
        self,
        *,
        cache: HybridRegionCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        basis = output_cotangent_basis
        for family, _sub_index, sub_cache in reversed(cache.runs):
            basis = self._subs[family].input_pullback_basis(
                cache=sub_cache, output_cotangent_basis=basis
            )
        return basis

    def parameter_vjp_with_input_cotangent(
        self,
        *,
        cache: HybridRegionCache,
        output_cotangent: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        grads: dict[str, torch.Tensor] = {}
        cot = output_cotangent
        for family, _sub_index, sub_cache in reversed(cache.runs):
            sub = self._subs[family]
            combined = getattr(sub, "parameter_vjp_with_input_cotangent", None)
            if combined is not None:
                run_grads, cot = combined(cache=sub_cache, output_cotangent=cot)
            else:
                run_grads = sub.parameter_vjp(cache=sub_cache, output_cotangent=cot)
                cot = sub.input_pullback_basis(
                    cache=sub_cache, output_cotangent_basis=cot.unsqueeze(1)
                ).squeeze(1)
            if cot.dim() == 4:
                cot = cot.squeeze(1)
            # Sub-backends name parameters under the same shared-stack attribute
            # ("backbone.blocks.N..."), so their keys are already hybrid-relative.
            grads.update(run_grads)
        return grads, cot

    def parameter_vjp(
        self,
        *,
        cache: HybridRegionCache,
        output_cotangent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        grads, _ = self.parameter_vjp_with_input_cotangent(
            cache=cache, output_cotangent=output_cotangent
        )
        return grads

    # ---- bookkeeping ----
    def parameters_for_region(self, region_index: int) -> list[nn.Parameter]:
        start, end = self.region_ranges[region_index]
        params: list[nn.Parameter] = []
        for block in self.backbone.blocks[start:end]:
            params.extend(block.parameters())
        return params

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters())

    def initialize_parameters(self, *, backbone_spec: BackboneSpec) -> None:
        assert backbone_spec.layer_types is not None
        for layer_type, block in zip(backbone_spec.layer_types, self.backbone.blocks):
            if layer_type == "transformer":
                init_transformer_module(block, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
