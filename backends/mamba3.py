"""Mamba-3 SISO region backend and Torch-reference derivative lowering.

Defines the frozen forward-cache ABI and the region backend, with a Torch
autograd reference lowering and the native lowerings (the scan pullback, the
epilogue pullback, and the parameter reductions) as `Mamba3Lowering`
implementations; the reference is the native lowerings' parity oracle.

Forward-cache ABI
-----------------
`forward_region` threads the upstream `(hidden_states, residual)` state through
the block range and records one `Mamba3LayerCache` per block. Each carries the
block-level `Mamba3BlockForwardCache`, whose `mixer_cache` is the frozen
`Mamba3ForwardCache` contract the kernel lowerings consume:

- input_u        [B, L, D]            block-mixer input (in_proj input)
- in_proj        [B, L, C_in]         concatenated z|x|B|C|dd_dt|dd_A|trap|angles
- z              [B, L, H, P]         output gate activation
- x              [B, L, H, P]         SSM value path (V); also feeds the D skip
- B, C           [B, L, 1, G, N]      SISO key/query state projections (pre-norm)
- B_normed, C_normed                  post-RMSNorm B/C
- dd_dt, dd_A    [B, L, H]            pre-activation dt and A channels
- trap           [B, H, L]            trapezoidal-correction term
- angles         [B, L, N, S]         rotational-state angles
- ADT, DT        [B, N, L]            decay (A*dt) and dt, scan transition factors
- y_inner        [B, L, H*P]          scan output before out_proj
- output         [B, L, D]            mixer output (out_proj output)

The scan pullback consumes the transition factors (ADT, DT, angles) and the
Q/K/V tiles (C, B, x); the parameter reductions additionally consume
input_u, y_inner, and z for the projection/norm/bias VJPs. The reference
lowering keeps the full forward cache resident; `recompute_forward` rebuilds
it per region instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn

from backbones.general import BackboneSpec, BackboneStack, build_backbone_stack
from backbones.mamba3.block import Mamba3BlockForwardCache
from backbones.mamba3.mamba3 import Mamba3ForwardCache
from backends.base import RegionForwardCache


@dataclass
class Mamba3LayerCache:
    layer_index: int
    block: Mamba3BlockForwardCache


@dataclass
class Mamba3RegionCache(RegionForwardCache):
    layer_caches: list[Mamba3LayerCache]
    region_input: torch.Tensor
    region_output: torch.Tensor


class Mamba3Lowering(Protocol):
    """Implementation strategy for Mamba-3 region derivative contracts."""

    name: str

    def input_pullback_basis(
        self,
        *,
        backend: "Mamba3RegionBackend",
        cache: Mamba3RegionCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def parameter_vjp(
        self,
        *,
        backend: "Mamba3RegionBackend",
        cache: Mamba3RegionCache,
        output_cotangent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ...


class Mamba3RegionBackend(nn.Module):
    """Executes Mamba-3 SISO block ranges and records per-block region caches."""

    name = "mamba3"

    def __init__(
        self,
        *,
        backbone_spec: BackboneSpec | None = None,
        region_ranges: Sequence[tuple[int, int]],
        backbone: BackboneStack | None = None,
        lowering: Mamba3Lowering | None = None,
        recompute_forward: bool = False,
    ) -> None:
        super().__init__()
        if backbone is None:
            if backbone_spec is None:
                raise ValueError("backbone_spec is required when backbone is not provided")
            backbone = build_backbone_stack(backbone_spec)
        self.backbone = backbone
        self.lowering = lowering or NativeMamba3Lowering()
        self.region_ranges = list(region_ranges)
        if not self.region_ranges:
            raise ValueError("mamba3 region backend requires at least one region")
        # When True, store only the region input and recompute per-layer
        # caches per region on demand (the store-vs-recompute knob).
        self.recompute_forward = bool(recompute_forward)

    def _region_range(self, region_index: int) -> tuple[int, int]:
        try:
            return self.region_ranges[region_index]
        except IndexError as exc:
            raise IndexError(
                f"region_index {region_index} out of range for {len(self.region_ranges)} regions"
            ) from exc

    def forward_region(
        self,
        *,
        region_input: torch.Tensor,
        region_index: int,
    ) -> tuple[torch.Tensor, Mamba3RegionCache]:
        # recompute mode retains only region_input; the native backward
        # rebuilds per-layer caches via materialize_region_cache.
        return self._run_region(region_input, region_index, retain=not self.recompute_forward)

    def _run_region(
        self, region_input: torch.Tensor, region_index: int, *, retain: bool
    ) -> tuple[torch.Tensor, Mamba3RegionCache]:
        start, end = self._region_range(region_index)
        hidden_states = region_input
        residual: torch.Tensor | None = None
        layer_caches: list[Mamba3LayerCache] = []
        for layer_index in range(start, end):
            block = self.backbone.blocks[layer_index]
            hidden_states, residual, block_cache = block.forward_with_cache(
                hidden_states, residual=residual
            )
            if retain:
                layer_caches.append(Mamba3LayerCache(layer_index=layer_index, block=block_cache))
        region_output = (hidden_states + residual) if residual is not None else hidden_states
        return region_output, Mamba3RegionCache(
            region_index=region_index,
            layer_range=(start, end),
            layer_caches=layer_caches,
            region_input=region_input,
            region_output=region_output,
        )

    def materialize_region_cache(self, cache: Mamba3RegionCache) -> Mamba3RegionCache:
        """Return a region cache with per-layer forward caches populated. If the
        cache was trimmed (recompute mode), recompute the region forward once from
        the retained region input under no_grad; otherwise return it unchanged.

        The recomputed cache is scoped to a single region's backward and freed when
        the caller drops it, so peak backward memory holds one region at a time."""
        if cache.layer_caches:
            return cache
        with torch.no_grad():
            _, full = self._run_region(cache.region_input, cache.region_index, retain=True)
        return full

    def parameters_for_region(self, region_index: int) -> list[nn.Parameter]:
        start, end = self._region_range(region_index)
        params: list[nn.Parameter] = []
        for layer_index in range(start, end):
            params.extend([p for p in self.backbone.blocks[layer_index].parameters() if p.requires_grad])
        return params

    def count_parameters(self) -> int:
        return int(sum(p.numel() for block in self.backbone.blocks for p in block.parameters()))

    def initialize_parameters(self, *, backbone_spec: BackboneSpec) -> None:
        # Mamba-3 blocks initialize their own parameters in their constructors.
        del backbone_spec

    def input_pullback_basis(
        self,
        *,
        cache: Mamba3RegionCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        return self.lowering.input_pullback_basis(
            backend=self,
            cache=cache,
            output_cotangent_basis=output_cotangent_basis,
        )

    def parameter_vjp(
        self,
        *,
        cache: Mamba3RegionCache,
        output_cotangent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.lowering.parameter_vjp(
            backend=self,
            cache=cache,
            output_cotangent=output_cotangent,
        )

    def region_output_jvp(
        self,
        *,
        cache: Mamba3RegionCache,
        region_input_tangent_basis: torch.Tensor,
        compute_dtype: "torch.dtype | None" = None,
        pooled: bool = False,
    ) -> torch.Tensor:
        """Forward-mode region map: push a region-input tangent basis [B, P, L, D]
        to a region-output tangent basis [B, P, L, D] at the frozen operating
        point. This is the dual of `input_pullback_basis` and the region factor of
        the forward-mode interface Jacobian. With `forward_mode_use_kernel`
        (and tilelang available) the mixer scan JVP runs through the fused
        dual-scan kernel; otherwise the torch.func reference path.

        `pooled`: return the mean over L with keepdim, [B, P, 1, D]. Exact
        for the A_k path, whose only consumer is the interface meanpool; the
        kernel path folds the last block's finalize + meanpool on-chip.

        `compute_dtype=None` resolves per path: bf16 tangent stream on the
        kernel path (LBI_FWDMODE_CD=float32 escapes), fp32 on the reference
        path."""
        if getattr(self, "forward_mode_use_kernel", False):
            from backends.mamba3_forward_mode import mamba3_region_output_jvp_kernel

            if compute_dtype is None:
                import os
                compute_dtype = (torch.float32
                                 if os.environ.get("LBI_FWDMODE_CD") == "float32"
                                 else torch.bfloat16)
            return mamba3_region_output_jvp_kernel(
                self,
                cache=cache,
                region_input_tangent_basis=region_input_tangent_basis,
                compute_dtype=compute_dtype,
                pooled=pooled,
            )
        from backends.mamba3_forward_mode import mamba3_region_output_jvp

        out = mamba3_region_output_jvp(
            self,
            cache=cache,
            region_input_tangent_basis=region_input_tangent_basis,
            compute_dtype=compute_dtype if compute_dtype is not None else torch.float32,
        )
        return out.mean(dim=2, keepdim=True) if pooled else out

    def parameter_vjp_with_input_cotangent(
        self,
        *,
        cache: Mamba3RegionCache,
        output_cotangent: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Param grads plus the region-input cotangent (`P=1`) from a single
        backward pass (and, under trimming, a single region recompute)."""
        vjp = getattr(self.lowering, "parameter_vjp_with_input_cotangent", None)
        if vjp is None:
            grads = self.parameter_vjp(cache=cache, output_cotangent=output_cotangent)
            g_region_input = self.input_pullback_basis(
                cache=cache, output_cotangent_basis=output_cotangent.unsqueeze(1)
            ).squeeze(1)
            return grads, g_region_input
        return vjp(backend=self, cache=cache, output_cotangent=output_cotangent)


class TorchAutogradMamba3Lowering:
    """Reference lowering implemented with torch.autograd: re-runs the region
    forward from a detached input and differentiates, holding parameters fixed
    for the input pullback. Parity oracle for the native lowerings.
    """

    name = "torch_autograd"

    def input_pullback_basis(
        self,
        *,
        backend: Mamba3RegionBackend,
        cache: Mamba3RegionCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        if output_cotangent_basis.dim() != 4:
            raise ValueError("output_cotangent_basis must have shape [B, P, L, D]")
        region_input = cache.region_input.detach().requires_grad_(True)
        region_output, _ = backend.forward_region(region_input=region_input, region_index=cache.region_index)
        basis_first = (
            output_cotangent_basis.to(device=region_output.device, dtype=region_output.dtype)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        try:
            grad_input = torch.autograd.grad(
                region_output,
                region_input,
                grad_outputs=basis_first,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
                is_grads_batched=True,
            )[0]
            return grad_input.permute(1, 0, 2, 3).contiguous().to(
                device=output_cotangent_basis.device,
                dtype=output_cotangent_basis.dtype,
            )
        except (TypeError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and "doesn't have storage" not in str(exc) and "vmap" not in str(exc):
                raise
        grads: list[torch.Tensor] = []
        for basis_index in range(output_cotangent_basis.shape[1]):
            region_input_i = cache.region_input.detach().requires_grad_(True)
            region_output_i, _ = backend.forward_region(region_input=region_input_i, region_index=cache.region_index)
            grad_i = torch.autograd.grad(
                region_output_i,
                region_input_i,
                grad_outputs=output_cotangent_basis[:, basis_index].to(
                    device=region_output_i.device, dtype=region_output_i.dtype
                ),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            grads.append(grad_i.unsqueeze(1))
        return torch.cat(grads, dim=1).to(device=output_cotangent_basis.device, dtype=output_cotangent_basis.dtype)

    def parameter_vjp(
        self,
        *,
        backend: Mamba3RegionBackend,
        cache: Mamba3RegionCache,
        output_cotangent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        params = backend.parameters_for_region(cache.region_index)
        region_input = cache.region_input.detach()
        region_output, _ = backend.forward_region(region_input=region_input, region_index=cache.region_index)
        grads = torch.autograd.grad(
            region_output,
            params,
            grad_outputs=output_cotangent.to(device=region_output.device, dtype=region_output.dtype),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        name_by_id = {id(param): name for name, param in backend.named_parameters()}
        out: dict[str, torch.Tensor] = {}
        for param, grad in zip(params, grads):
            if grad is not None:
                out[name_by_id[id(param)]] = grad.detach().clone()
        return out


class NativeMamba3Lowering:
    """Native (autograd-free) Mamba-3 region lowering.

    Composes the native block pullbacks (scan pullback + epilogue) and block
    parameter VJPs (scan pullback at P=1 + parameter reductions) over the
    region's block range. `input_pullback_basis` threads
    the P-batched output cotangent backward through the blocks;
    `parameter_vjp` threads the single real adjoint and accumulates parameter
    grads. `TorchAutogradMamba3Lowering` is the parity oracle.
    """

    name = "native"

    def input_pullback_basis(
        self,
        *,
        backend: Mamba3RegionBackend,
        cache: Mamba3RegionCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        if output_cotangent_basis.dim() != 4:
            raise ValueError("output_cotangent_basis must have shape [B, P, L, D]")
        cache = backend.materialize_region_cache(cache)
        # region_output = hidden + residual (last block) -> both seed the cotangent.
        g_hidden = output_cotangent_basis
        g_residual: torch.Tensor | None = output_cotangent_basis
        for layer_cache in reversed(cache.layer_caches):
            block = backend.backbone.blocks[layer_cache.layer_index]
            g_hidden, g_residual = mamba3_block_input_pullback_native(
                block=block,
                cache=layer_cache.block,
                output_cotangent_basis=g_hidden,
                residual_cotangent_basis=g_residual,
            )
        return g_hidden

    def parameter_vjp(
        self,
        *,
        backend: Mamba3RegionBackend,
        cache: Mamba3RegionCache,
        output_cotangent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        grads, _ = self.parameter_vjp_with_input_cotangent(
            backend=backend, cache=cache, output_cotangent=output_cotangent
        )
        return grads

    def parameter_vjp_with_input_cotangent(
        self,
        *,
        backend: Mamba3RegionBackend,
        cache: Mamba3RegionCache,
        output_cotangent: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Param grads AND the region-input cotangent (`P=1`) in one pass. The
        block loop that threads the adjoint back for the parameter reductions
        already produces the region-input cotangent as its final `g_hidden`, so the
        canvas contribution comes for free with no separate `input_pullback_basis`
        (and, under trimming, no second region recompute)."""
        cache = backend.materialize_region_cache(cache)
        g_hidden = output_cotangent
        g_residual: torch.Tensor | None = output_cotangent
        grads_by_param: dict = {}
        for layer_cache in reversed(cache.layer_caches):
            block = backend.backbone.blocks[layer_cache.layer_index]
            g_hidden, g_residual, block_grads = mamba3_block_param_vjp_native(
                block=block,
                cache=layer_cache.block,
                output_cotangent=g_hidden,
                residual_cotangent=g_residual,
            )
            grads_by_param.update(block_grads)
        name_by_id = {id(param): name for name, param in backend.named_parameters()}
        out: dict[str, torch.Tensor] = {}
        for param, grad in grads_by_param.items():
            name = name_by_id.get(id(param))
            if name is not None:
                out[name] = grad.detach().clone()
        return out, g_hidden


class TileLangPBatchedMamba3Lowering(NativeMamba3Lowering):
    """Region lowering that runs the P-batched cotangent walk with the tilelang
    lane-grid mixer pullback instead of the per-lane native path.

    Only `input_pullback_basis` (the P=r A_k factor, Phase 1's hot loop) changes;
    the P=1 parameter VJP inherits the native path, where P-batching buys
    nothing. Wins in the launch-bound region regime (short per-region
    sequences); requires tilelang.
    """

    name = "tilelang_pbatched"

    def input_pullback_basis(
        self,
        *,
        backend: Mamba3RegionBackend,
        cache: Mamba3RegionCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        from backbones.mamba3.ops.tilelang.mamba3.siso_pbatched import (
            mamba3_mixer_input_pullback_pbatched,
        )

        if output_cotangent_basis.dim() != 4:
            raise ValueError("output_cotangent_basis must have shape [B, P, L, D]")
        cache = backend.materialize_region_cache(cache)
        g_hidden = output_cotangent_basis
        g_residual: torch.Tensor | None = output_cotangent_basis
        for layer_cache in reversed(cache.layer_caches):
            block = backend.backbone.blocks[layer_cache.layer_index]
            g_hidden, g_residual = mamba3_block_input_pullback_native(
                block=block,
                cache=layer_cache.block,
                output_cotangent_basis=g_hidden,
                residual_cotangent_basis=g_residual,
                mixer_pullback=lambda **kw: mamba3_mixer_input_pullback_pbatched(**kw),
            )
        return g_hidden


class RegionLocalAutogradMamba3Lowering:
    """Region-local checkpointed-autograd lowering.

    Instead of the custom Triton backward orchestration (`NativeMamba3Lowering`),
    re-run the region forward once from the trim checkpoint (`cache.region_input`)
    under `enable_grad` and drive the OFFICIAL fused mamba backward via
    `torch.autograd.grad`. This is still region-independent and bounded-memory (it
    is per-region activation checkpointing: one region's graph, freed after), so
    it composes with the scan engine / multi-device driver unchanged, with each
    backward pass on the fused kernel. It ignores `layer_caches` entirely
    (always re-forwards), so it is trim-agnostic.
    """

    name = "region_local_autograd"

    @staticmethod
    def _reforward(backend: "Mamba3RegionBackend", cache: Mamba3RegionCache):
        start, end = backend._region_range(cache.region_index)
        region_input = cache.region_input.detach().requires_grad_(True)
        hidden = region_input
        residual: torch.Tensor | None = None
        for layer_index in range(start, end):
            hidden, residual = backend.backbone.blocks[layer_index](hidden, residual=residual)
        region_output = (hidden + residual) if residual is not None else hidden
        return region_input, region_output

    def input_pullback_basis(
        self,
        *,
        backend: "Mamba3RegionBackend",
        cache: Mamba3RegionCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        if output_cotangent_basis.dim() != 4:
            raise ValueError("output_cotangent_basis must have shape [B, P, L, D]")
        with torch.enable_grad():
            region_input, region_output = self._reforward(backend, cache)
            basis = output_cotangent_basis.to(device=region_output.device, dtype=region_output.dtype)
            cols = [
                torch.autograd.grad(region_output, region_input, grad_outputs=basis[:, i], retain_graph=True)[0]
                for i in range(basis.shape[1])
            ]
        return torch.stack(cols, dim=1).to(dtype=output_cotangent_basis.dtype)

    def parameter_vjp(
        self,
        *,
        backend: "Mamba3RegionBackend",
        cache: Mamba3RegionCache,
        output_cotangent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        grads, _ = self.parameter_vjp_with_input_cotangent(
            backend=backend, cache=cache, output_cotangent=output_cotangent
        )
        return grads

    def parameter_vjp_with_input_cotangent(
        self,
        *,
        backend: "Mamba3RegionBackend",
        cache: Mamba3RegionCache,
        output_cotangent: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        params = list(backend.parameters_for_region(cache.region_index))
        with torch.enable_grad():
            region_input, region_output = self._reforward(backend, cache)
            grads = torch.autograd.grad(
                region_output,
                [region_input, *params],
                grad_outputs=output_cotangent.to(device=region_output.device, dtype=region_output.dtype),
                allow_unused=True,
            )
        g_region_input = grads[0]
        name_by_id = {id(param): name for name, param in backend.named_parameters()}
        out: dict[str, torch.Tensor] = {}
        for param, grad in zip(params, grads[1:]):
            name = name_by_id.get(id(param))
            if grad is not None and name is not None:
                out[name] = grad.detach().clone()
        return out, g_region_input.detach()


# Mixer input-pullback contract (the scan pullback composed with the epilogue).


class Mamba3MixerLowering(Protocol):
    """Contract for the Mamba-3 state-mixer matrix-valued input pullback.

    `input_pullback_basis` maps a P-batched cotangent on the mixer output to the
    P-batched cotangent on the mixer input `u`, holding mixer parameters fixed:

        (cache: Mamba3ForwardCache, output_cotangent_basis [B, P, L, D])
            -> input_cotangent_basis [B, P, L, D]

    `D` is the mixer's model dimension; `P` is the cotangent-basis batch (the
    interface rank `r` when materializing an interface Jacobian, or 1 for a
    real adjoint). The native lowering realizes this contract as the scan
    pullback composed with the epilogue pullback. `cache` is the frozen
    `Mamba3ForwardCache` ABI documented in this module; the reference lowering
    is the parity oracle.
    """

    name: str

    def input_pullback_basis(
        self,
        *,
        mixer: nn.Module,
        cache: Mamba3ForwardCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        ...


def _autograd_mixer_input_pullback_basis(
    *,
    mixer: nn.Module,
    cache: Mamba3ForwardCache,
    output_cotangent_basis: torch.Tensor,
) -> torch.Tensor:
    """Reference mixer input pullback: re-run forward from the cached input and
    differentiate w.r.t. that input only (parameters held fixed)."""
    if output_cotangent_basis.dim() != 4:
        raise ValueError("output_cotangent_basis must have shape [B, P, L, D]")
    u = cache.input_u.detach().requires_grad_(True)
    output = mixer(u)
    basis_first = (
        output_cotangent_basis.to(device=output.device, dtype=output.dtype)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    try:
        grad_input = torch.autograd.grad(
            output,
            u,
            grad_outputs=basis_first,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
            is_grads_batched=True,
        )[0]
        return grad_input.permute(1, 0, 2, 3).contiguous().to(
            device=output_cotangent_basis.device,
            dtype=output_cotangent_basis.dtype,
        )
    except (TypeError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and "doesn't have storage" not in str(exc) and "vmap" not in str(exc):
            raise
    grads: list[torch.Tensor] = []
    for basis_index in range(output_cotangent_basis.shape[1]):
        u_i = cache.input_u.detach().requires_grad_(True)
        output_i = mixer(u_i)
        grad_i = torch.autograd.grad(
            output_i,
            u_i,
            grad_outputs=output_cotangent_basis[:, basis_index].to(device=output_i.device, dtype=output_i.dtype),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        grads.append(grad_i.unsqueeze(1))
    return torch.cat(grads, dim=1).to(device=output_cotangent_basis.device, dtype=output_cotangent_basis.dtype)


class TorchAutogradMamba3MixerLowering:
    """Reference mixer input-pullback lowering (scan pullback + epilogue) via autograd."""

    name = "torch_autograd"

    def input_pullback_basis(
        self,
        *,
        mixer: nn.Module,
        cache: Mamba3ForwardCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        return _autograd_mixer_input_pullback_basis(
            mixer=mixer,
            cache=cache,
            output_cotangent_basis=output_cotangent_basis,
        )


class NativeMamba3MixerLowering:
    """Native (autograd-free) mixer input pullback.

    Composes `out_proj^T`, the scan pullback (`mamba3_siso_scan_input_pullback_basis`,
    which drives the official Triton kernels per lane), the preprocess/RMSNorm
    VJPs, and `in_proj^T`. No `torch.autograd` in the path. `allow_reference_fallback`
    forces the autograd reference instead, for parity testing.
    """

    name = "native"

    def __init__(self, *, allow_reference_fallback: bool = False) -> None:
        self.allow_reference_fallback = bool(allow_reference_fallback)
        self._reference = TorchAutogradMamba3MixerLowering()

    def input_pullback_basis(
        self,
        *,
        mixer: nn.Module,
        cache: Mamba3ForwardCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        if self.allow_reference_fallback:
            return self._reference.input_pullback_basis(
                mixer=mixer,
                cache=cache,
                output_cotangent_basis=output_cotangent_basis,
            )
        return mamba3_mixer_input_pullback_native(
            mixer=mixer,
            cache=cache,
            output_cotangent_basis=output_cotangent_basis,
        )


class TileLangPBatchedMamba3MixerLowering:
    """P-batched mixer input pullback via the tilelang lane-grid MIMO backward.

    Composes `out_proj^T`, the lane-grid P-batched scan pullback (one kernel
    pair for all P lanes: shared cotangent-independent pass + per-lane-CTA
    backward), and the closed-form lane-batched preprocess VJP. All P lanes
    complete in a fixed number of launches, which wins in the launch-bound
    short-sequence regime. Requires tilelang (JIT, cached).
    """

    name = "tilelang_pbatched"

    def input_pullback_basis(
        self,
        *,
        mixer: nn.Module,
        cache: Mamba3ForwardCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        from backbones.mamba3.ops.tilelang.mamba3.siso_pbatched import (
            mamba3_mixer_input_pullback_pbatched,
        )

        return mamba3_mixer_input_pullback_pbatched(
            mixer=mixer,
            cache=cache,
            output_cotangent_basis=output_cotangent_basis,
        )


# Backwards-compatible alias (the scan is native Triton; the epilogues are torch).
TritonMamba3MixerLowering = NativeMamba3MixerLowering


# Scan-boundary input pullback (the scan backward recurrence).

# Differentiable scan inputs in SISO-kernel order: the activation adjoints
# the scan pullback produces (parameters Q/K biases, D held fixed).
MAMBA3_SCAN_INPUT_NAMES = ("Q", "K", "V", "ADT", "DT", "Trap", "Angles", "Z")


def _mamba3_scan_inputs_from_cache(mixer: nn.Module, cache: Mamba3ForwardCache) -> dict[str, torch.Tensor]:
    """Reconstruct the SISO scan's differentiable inputs from the frozen cache.

    Contiguous-izes every tensor: the cache holds transpose (ADT/DT/Trap) and
    expand (Angles) views of the in_proj output, and the Triton backward kernels
    consuming these assume packed layouts (strided views silently corrupt the
    non-V/Z gradients)."""
    return {
        "Q": cache.C.squeeze(2).contiguous(),
        "K": cache.B.squeeze(2).contiguous(),
        "V": cache.x.contiguous(),
        "ADT": cache.ADT.contiguous(),
        "DT": cache.DT.contiguous(),
        "Trap": cache.trap.contiguous(),
        "Angles": cache.angles.contiguous(),
        "Z": cache.z.contiguous(),
    }


def mamba3_siso_scan_input_pullback_basis(
    *,
    mixer: nn.Module,
    cache: Mamba3ForwardCache,
    output_cotangent_basis: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Matrix-valued input pullback of the SISO selective scan.

    Maps a P-batched cotangent on the scan output `[B, P, L, H, Dv]` to the
    P-batched activation adjoints on the scan inputs (Q=C, K=B, V=x, ADT, DT,
    Trap, Angles, Z), holding parameters (Q/K biases, D) fixed.

    Recomputes the scan forward once via `mamba3_siso_fwd` and drives the
    Triton backward kernels directly for each of the `P` cotangent lanes,
    discarding parameter (`dD`, `dQ_bias`, `dK_bias`) and inference-state
    grads; the forward is shared across `P`. No `torch.autograd` in the path.
    """
    if output_cotangent_basis.dim() != 5:
        raise ValueError("output_cotangent_basis must have shape [B, P, L, H, Dv]")

    tiles = _mamba3_recompute_scan_tiles(mixer, cache)
    scan = tiles["scan"]
    columns: dict[str, list[torch.Tensor]] = {name: [] for name in MAMBA3_SCAN_INPUT_NAMES}
    for basis_index in range(output_cotangent_basis.shape[1]):
        grad_out = output_cotangent_basis[:, basis_index].to(device=tiles["out_v"].device, dtype=scan["V"].dtype)
        inputs, _params = _mamba3_scan_lane_backward(mixer, tiles, grad_out)
        for name in MAMBA3_SCAN_INPUT_NAMES:
            columns[name].append(inputs[name].unsqueeze(1))

    # Return each adjoint in its scan input's dtype (matching autograd semantics:
    # the kernels compute in fp32 but grads adopt the input dtype).
    return {
        name: torch.cat(columns[name], dim=1).to(dtype=scan[name].dtype)
        for name in MAMBA3_SCAN_INPUT_NAMES
    }


def _mamba3_recompute_scan_tiles(mixer: nn.Module, cache: Mamba3ForwardCache) -> dict:
    """Recompute the cotangent-independent SISO forward tiles from the cache
    (shared by the input pullback and the parameter VJP)."""
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd
    from backbones.mamba3.ops.triton.mamba3.angle_dt import angle_dt_fwd

    scan = _mamba3_scan_inputs_from_cache(mixer, cache)
    q_bias = mixer.C_bias.squeeze(1)
    k_bias = mixer.B_bias.squeeze(1)
    chunk_size = mixer.chunk_size
    angles_cumsum, _ = angle_dt_fwd(
        scan["Angles"], scan["DT"], init_state=None, chunk_size=chunk_size,
        return_output_state=True, cu_seqlens=None,
    )
    _out, out_v, ssm_states, da_cs, da_cs_sum, q_rot, k_scaled, qk_dot, scale, gamma, _final = mamba3_siso_fwd(
        scan["Q"], scan["K"], scan["V"], scan["ADT"], scan["DT"], scan["Trap"],
        q_bias, k_bias, angles_cumsum, mixer.D, scan["Z"], None,
        chunk_size=chunk_size, store_states_adt_outv=True, return_final_states=False,
    )
    return {
        "scan": scan, "q_bias": q_bias, "k_bias": k_bias, "chunk_size": chunk_size,
        "angles_cumsum": angles_cumsum, "out_v": out_v, "ssm_states": ssm_states,
        "da_cs": da_cs, "da_cs_sum": da_cs_sum, "q_rot": q_rot, "k_scaled": k_scaled,
        "qk_dot": qk_dot, "scale": scale, "gamma": gamma,
    }


def _mamba3_scan_lane_backward(mixer: nn.Module, tiles: dict, grad_out: torch.Tensor):
    """One-lane SISO scan backward. Returns (scan-input adjoints, scan-param grads
    `{dD, dC_bias, dB_bias}`). 4a discards the params; the parameter VJP keeps them."""
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_bwd import (
        compute_ddt_dtrap_dinput_states, compute_dqktheta, compute_dqkv, compute_dzdo,
    )
    from backbones.mamba3.ops.triton.mamba3.angle_dt import angle_dt_bwd

    scan = tiles["scan"]
    Z = scan["Z"]
    cs = tiles["chunk_size"]
    if Z is not None:
        dZ, grad_out_scaled = compute_dzdo(grad_out, Z, tiles["out_v"], chunk_size=cs)
    else:
        dZ, grad_out_scaled = None, grad_out
    dQ_mid, dK_mid, dV, dADT, dQK_dot, dD, _ = compute_dqkv(
        q=tiles["q_rot"], k=tiles["k_scaled"], v=scan["V"], da_cs=tiles["da_cs"],
        da_cs_sum=tiles["da_cs_sum"], qk_dot=tiles["qk_dot"], SSM_States=tiles["ssm_states"],
        do=grad_out_scaled, D=mixer.D, chunk_size=cs,
    )
    dQ, dK, dQ_bias, dK_bias, dAngles_Cumsum, dScale, dGamma = compute_dqktheta(
        q=scan["Q"], k=scan["K"], scale=tiles["scale"], gamma=tiles["gamma"],
        q_bias=tiles["q_bias"], k_bias=tiles["k_bias"], angles=tiles["angles_cumsum"],
        dq_in=dQ_mid, dk_in=dK_mid, dqk=dQK_dot, d_ok_state=None, chunk_size=cs,
    )
    dDT, dTrap, _, _, _ = compute_ddt_dtrap_dinput_states(
        dscale=dScale, dgamma=dGamma, dt=scan["DT"], trap=scan["Trap"].float(),
        d_issm_state=None, input_k_state=None, input_v_state=None,
    )
    dAngles, dDT_angle, _ = angle_dt_bwd(
        grad_out=dAngles_Cumsum, angle=scan["Angles"], dt=scan["DT"], has_init_state=False,
        chunk_size=cs, grad_output_state=None,
    )
    dDT = dDT + dDT_angle
    inputs = {"Q": dQ, "K": dK, "V": dV, "ADT": dADT, "DT": dDT, "Trap": dTrap, "Angles": dAngles, "Z": dZ}
    params = {"dD": dD, "dC_bias": dQ_bias, "dB_bias": dK_bias}
    return inputs, params


# Native mixer input pullback (output cotangent -> input u).


def _rmsnorm_vjp(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """VJP of RMSNorm (no bias, no gate) w.r.t. its input `x`.

    `dy` may carry a leading `P` axis that broadcasts against the shared,
    cotangent-independent `x`. Returns `dx` in fp32.
    """
    xf = x.float()
    dyf = dy.float()
    wf = weight.float()
    n = xf.shape[-1]
    r = torch.rsqrt((xf * xf).mean(-1, keepdim=True) + eps)
    g = dyf * wf
    sgx = (g * xf).sum(-1, keepdim=True)
    return r * g - (r * r * r / n) * xf * sgx


def mamba3_mixer_input_pullback_native(
    *,
    mixer: nn.Module,
    cache: Mamba3ForwardCache,
    output_cotangent_basis: torch.Tensor,
) -> torch.Tensor:
    """Native (autograd-free) mixer input pullback.

    Maps a P-batched cotangent on the mixer output `[B, P, L, D]` to the
    cotangent on the mixer input `u`, parameters held fixed. Composes
    `out_proj^T`, the scan pullback (`mamba3_siso_scan_input_pullback_basis`),
    the preprocess/RMSNorm VJPs, and `in_proj^T`. Preprocess intermediates are
    recomputed from the frozen forward cache rather than stored.
    """
    if output_cotangent_basis.dim() != 4:
        raise ValueError("output_cotangent_basis must have shape [B, P, L, D]")
    bsz, num_p, seqlen, _ = output_cotangent_basis.shape
    heads, hdim = mixer.nheads, mixer.headdim
    d_inner, d_state, nheads = mixer.d_inner, mixer.d_state, mixer.nheads

    # 1. out_proj^T -> scan-output cotangent [B, P, L, H, hd]
    g_yinner = output_cotangent_basis.to(mixer.out_proj.weight.dtype) @ mixer.out_proj.weight
    g_scan_out = g_yinner.reshape(bsz, num_p, seqlen, heads, hdim).contiguous()

    # 2. scan pullback
    adj = mamba3_siso_scan_input_pullback_basis(
        mixer=mixer, cache=cache, output_cotangent_basis=g_scan_out
    )

    # recompute the cheap preprocess forward from the cache
    sizes = [d_inner, d_inner, d_state, d_state, nheads, nheads, nheads, mixer.num_rope_angles]
    _z, _x, b_pre, c_pre, _ddt, _ddA, _trap, _ang = torch.split(cache.in_proj, sizes, dim=-1)
    ddA_f = cache.dd_A.float()
    ddt_f = cache.dd_dt.float()
    softplus_a = torch.nn.functional.softplus(ddA_f)
    a_clamped = torch.clamp(-softplus_a, max=-mixer.A_floor)
    clamp_mask = (-softplus_a < -mixer.A_floor).float()
    sig_dt = torch.sigmoid(ddt_f + mixer.dt_bias.float())
    sig_a = torch.sigmoid(ddA_f)

    # 3. RMSNorm VJPs for B/C (scan Q=C, K=B are post-norm)
    g_c_pre = _rmsnorm_vjp(adj["Q"].reshape(bsz, num_p, seqlen, d_state), c_pre.unsqueeze(1), mixer.C_norm.weight, mixer.C_norm.eps)
    g_b_pre = _rmsnorm_vjp(adj["K"].reshape(bsz, num_p, seqlen, d_state), b_pre.unsqueeze(1), mixer.B_norm.weight, mixer.B_norm.eps)

    # 4. z, x (direct in_proj slices, reshaped)
    g_z = adj["Z"].reshape(bsz, num_p, seqlen, d_inner)
    g_x = adj["V"].reshape(bsz, num_p, seqlen, d_inner)

    # 5. ADT = clamp(-softplus(dd_A), max=-A_floor) * softplus(dd_dt + dt_bias)
    dt_val = torch.nn.functional.softplus(ddt_f + mixer.dt_bias.float())  # [B,L,n]
    g_adt = adj["ADT"].transpose(-1, -2).float()  # [B,P,n,L] -> [B,P,L,n]
    g_dt = adj["DT"].transpose(-1, -2).float()
    g_dt_total = g_dt + g_adt * a_clamped.unsqueeze(1)              # dL/dDT
    g_ddt = g_dt_total * sig_dt.unsqueeze(1)                         # through softplus
    g_a = g_adt * dt_val.unsqueeze(1)                               # dL/d_A
    g_ddA = g_a * clamp_mask.unsqueeze(1) * (-sig_a.unsqueeze(1))    # through clamp + (-softplus)

    # 6. trap, angles (expand over heads -> sum)
    g_trap = adj["Trap"].transpose(-1, -2)          # [B,P,h,L] -> [B,P,L,h]
    g_ang = adj["Angles"].sum(dim=-2)               # [B,P,L,nheads,S] -> [B,P,L,S]

    # 8. assemble in_proj-output cotangent and apply in_proj^T
    w_in = mixer.in_proj.weight  # [C_in, d_model]
    dt = w_in.dtype
    g_inproj = torch.cat(
        [g_z.to(dt), g_x.to(dt), g_b_pre.to(dt), g_c_pre.to(dt), g_ddt.to(dt), g_ddA.to(dt), g_trap.to(dt), g_ang.to(dt)],
        dim=-1,
    )
    g_u = g_inproj @ w_in
    return g_u.to(output_cotangent_basis.dtype)


def mamba3_block_input_pullback_native(
    *,
    block: nn.Module,
    cache: Mamba3BlockForwardCache,
    output_cotangent_basis: torch.Tensor,
    residual_cotangent_basis: torch.Tensor | None = None,
    mixer_pullback=None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Native (autograd-free) pullback for one `Mamba3Block`.

    The block is `residual_out = hidden_in + residual_in; hidden_out =
    mixer(RMSNorm(residual_out))`, returning `(hidden_out, residual_out)`. Given
    the P-batched output cotangents `(g_hidden_out, g_residual_out)`, returns the
    input cotangents `(g_hidden_in, g_residual_in)`. Composes the mixer input
    pullback (`mixer_pullback`, default the native per-lane path) and the RMSNorm
    VJP; replaces the autograd path in `Mamba3Block.input_pullback_matrix`.
    """
    if mixer_pullback is None:
        mixer_pullback = mamba3_mixer_input_pullback_native
    # hidden_out path: mixer^T then RMSNorm^T -> cotangent on residual_out.
    g_norm_input = mixer_pullback(
        mixer=block.mixer, cache=cache.mixer_cache, output_cotangent_basis=output_cotangent_basis
    )
    x_norm = cache.residual_output.to(block.norm.weight.dtype).unsqueeze(1)  # [B,1,L,D]
    g_res_from_norm = _rmsnorm_vjp(g_norm_input, x_norm, block.norm.weight, block.norm.eps).to(
        output_cotangent_basis.dtype
    )

    # residual_out is also a returned output -> accumulate the downstream cotangent.
    if residual_cotangent_basis is None:
        g_residual_total = g_res_from_norm
    else:
        g_residual_total = g_res_from_norm + residual_cotangent_basis.to(g_res_from_norm.dtype)

    # residual_out = hidden_in + residual_in -> both inputs receive g_residual_total.
    g_hidden_in = g_residual_total
    if cache.residual_input is None:
        return g_hidden_in, None
    return g_hidden_in, g_hidden_in.clone()


# Native parameter VJP (scan pullback at P=1 plus the parameter reductions).


def _rmsnorm_weight_grad(dy: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    """Weight grad of RMSNorm (no bias/gate): sum over leading dims of
    `dy * normalize(x)`. Returns `[N]`."""
    xf = x.float()
    dyf = dy.float()
    r = torch.rsqrt((xf * xf).mean(-1, keepdim=True) + eps)
    dw = dyf * (xf * r)
    return dw.reshape(-1, dw.shape[-1]).sum(0)


def mamba3_mixer_param_vjp_native(
    *,
    mixer: nn.Module,
    cache: Mamba3ForwardCache,
    output_cotangent: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Native mixer parameter VJP at `P = 1`.

    Given the real output cotangent `[B, L, D]`, returns `(g_u [B, L, D],
    {param -> grad})` for the mixer's parameters: the P=1 scan-param grads
    (`D`, C/B biases) plus the parameter reductions (projection weights, `dt_bias`,
    B/C-norm weights). Grads are keyed by the parameter object.
    """
    bsz, seqlen, _ = output_cotangent.shape
    heads, hdim = mixer.nheads, mixer.headdim
    d_inner, d_state, nheads = mixer.d_inner, mixer.d_state, mixer.nheads
    grads: dict = {}

    # out_proj
    y_inner = cache.y_inner
    grads[mixer.out_proj.weight] = torch.einsum("bld,ble->de", output_cotangent.float(), y_inner.float()).to(mixer.out_proj.weight.dtype)
    g_yinner = output_cotangent.to(mixer.out_proj.weight.dtype) @ mixer.out_proj.weight
    g_scan_out = g_yinner.reshape(bsz, seqlen, heads, hdim).contiguous()

    # scan pullback at P=1: scan-input adjoints + scan-param grads
    tiles = _mamba3_recompute_scan_tiles(mixer, cache)
    inputs, scan_params = _mamba3_scan_lane_backward(mixer, tiles, g_scan_out.to(dtype=tiles["scan"]["V"].dtype))
    grads[mixer.D] = scan_params["dD"].to(mixer.D.dtype)
    grads[mixer.C_bias] = scan_params["dC_bias"].unsqueeze(1).to(mixer.C_bias.dtype)
    grads[mixer.B_bias] = scan_params["dB_bias"].unsqueeze(1).to(mixer.B_bias.dtype)

    # preprocess intermediates
    sizes = [d_inner, d_inner, d_state, d_state, nheads, nheads, nheads, mixer.num_rope_angles]
    _z, _x, b_pre, c_pre, _ddt, _ddA, _trap, _ang = torch.split(cache.in_proj, sizes, dim=-1)
    ddA_f = cache.dd_A.float()
    ddt_f = cache.dd_dt.float()
    softplus_a = torch.nn.functional.softplus(ddA_f)
    a_clamped = torch.clamp(-softplus_a, max=-mixer.A_floor)
    clamp_mask = (-softplus_a < -mixer.A_floor).float()
    sig_dt = torch.sigmoid(ddt_f + mixer.dt_bias.float())
    sig_a = torch.sigmoid(ddA_f)
    dt_val = torch.nn.functional.softplus(ddt_f + mixer.dt_bias.float())

    # norm VJPs (input grad + weight grad)
    dC_post = inputs["Q"].reshape(bsz, seqlen, d_state)
    dB_post = inputs["K"].reshape(bsz, seqlen, d_state)
    g_c_pre = _rmsnorm_vjp(dC_post, c_pre, mixer.C_norm.weight, mixer.C_norm.eps)
    g_b_pre = _rmsnorm_vjp(dB_post, b_pre, mixer.B_norm.weight, mixer.B_norm.eps)
    grads[mixer.C_norm.weight] = _rmsnorm_weight_grad(dC_post, c_pre, mixer.C_norm.eps).to(mixer.C_norm.weight.dtype)
    grads[mixer.B_norm.weight] = _rmsnorm_weight_grad(dB_post, b_pre, mixer.B_norm.eps).to(mixer.B_norm.weight.dtype)

    g_z = inputs["Z"].reshape(bsz, seqlen, d_inner)
    g_x = inputs["V"].reshape(bsz, seqlen, d_inner)
    g_adt = inputs["ADT"].transpose(-1, -2).float()
    g_dt = inputs["DT"].transpose(-1, -2).float()
    g_ddt = (g_dt + g_adt * a_clamped) * sig_dt
    g_ddA = (g_adt * dt_val) * clamp_mask * (-sig_a)
    grads[mixer.dt_bias] = g_ddt.reshape(-1, nheads).sum(0).to(mixer.dt_bias.dtype)
    g_trap = inputs["Trap"].transpose(-1, -2)
    g_ang = inputs["Angles"].sum(dim=-2)

    w_in = mixer.in_proj.weight
    dt = w_in.dtype
    g_inproj = torch.cat(
        [g_z.to(dt), g_x.to(dt), g_b_pre.to(dt), g_c_pre.to(dt), g_ddt.to(dt), g_ddA.to(dt), g_trap.to(dt), g_ang.to(dt)],
        dim=-1,
    )
    grads[w_in] = torch.einsum("blc,bld->cd", g_inproj.float(), cache.input_u.float()).to(dt)
    g_u = g_inproj @ w_in
    return g_u.to(output_cotangent.dtype), grads


def mamba3_block_param_vjp_native(
    *,
    block: nn.Module,
    cache: Mamba3BlockForwardCache,
    output_cotangent: torch.Tensor,
    residual_cotangent: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    """Native block parameter VJP at `P = 1`. Returns
    `(g_hidden_in, g_residual_in, {param -> grad})` for the block parameters."""
    g_u, grads = mamba3_mixer_param_vjp_native(
        mixer=block.mixer, cache=cache.mixer_cache, output_cotangent=output_cotangent
    )
    x_norm = cache.residual_output.to(block.norm.weight.dtype)
    g_res_from_norm = _rmsnorm_vjp(g_u, x_norm, block.norm.weight, block.norm.eps).to(output_cotangent.dtype)
    grads[block.norm.weight] = _rmsnorm_weight_grad(g_u, x_norm, block.norm.eps).to(block.norm.weight.dtype)

    if residual_cotangent is None:
        g_residual_total = g_res_from_norm
    else:
        g_residual_total = g_res_from_norm + residual_cotangent.to(g_res_from_norm.dtype)
    g_hidden_in = g_residual_total
    g_residual_in = None if cache.residual_input is None else g_residual_total.clone()
    return g_hidden_in, g_residual_in, grads
