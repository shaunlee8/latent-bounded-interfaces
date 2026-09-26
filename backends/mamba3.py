"""Mamba-3 SISO region backend, its forward-cache contract, and the derivative
lowerings: a Torch autograd reference and the native lowerings (scan pullback,
epilogue pullback, parameter reductions) as `Mamba3Lowering` implementations.

`forward_region` threads `(hidden_states, residual)` through the block range
and records one `Mamba3LayerCache` per block, whose `mixer_cache` is the
`Mamba3ForwardCache` the lowerings consume:

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
Q/K/V tiles (C, B, x); the parameter reductions additionally consume input_u,
y_inner, and z."""

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
        start, end = self._region_range(region_index)
        hidden_states = region_input
        residual: torch.Tensor | None = None
        layer_caches: list[Mamba3LayerCache] = []
        for layer_index in range(start, end):
            block = self.backbone.blocks[layer_index]
            hidden_states, residual, block_cache = block.forward_with_cache(
                hidden_states, residual=residual
            )
            layer_caches.append(Mamba3LayerCache(layer_index=layer_index, block=block_cache))
        region_output = (hidden_states + residual) if residual is not None else hidden_states
        return region_output, Mamba3RegionCache(
            region_index=region_index,
            layer_range=(start, end),
            layer_caches=layer_caches,
            region_input=region_input,
            region_output=region_output,
        )

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
        tangent_token_start: int = 0,
        direction_major_out: bool = False,
    ) -> torch.Tensor:
        """Forward-mode region map: an input tangent basis [B, P, L, D] to the
        output tangent basis, the dual of `input_pullback_basis`. `pooled` returns
        the mean over L as [B, P, 1, D]; `compute_dtype=None` selects bf16 on the
        kernel path and fp32 on the reference path."""
        if getattr(self, "forward_mode_use_kernel", False):
            from backends.mamba3_forward_mode import mamba3_region_output_jvp_kernel

            if compute_dtype is None:
                compute_dtype = torch.bfloat16
            return mamba3_region_output_jvp_kernel(
                self,
                cache=cache,
                region_input_tangent_basis=region_input_tangent_basis,
                compute_dtype=compute_dtype,
                pooled=pooled,
                tangent_token_start=tangent_token_start,
                direction_major_out=direction_major_out,
            )
        from backends.mamba3_forward_mode import mamba3_region_output_jvp

        out = mamba3_region_output_jvp(
            self,
            cache=cache,
            region_input_tangent_basis=region_input_tangent_basis,
            compute_dtype=compute_dtype if compute_dtype is not None else torch.float32,
        )
        if direction_major_out:
            out = out.transpose(0, 1)
        return out.mean(dim=2, keepdim=True) if pooled else out

    def parameter_vjp_with_input_cotangent(
        self,
        *,
        cache: Mamba3RegionCache,
        output_cotangent: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Param grads plus the region-input cotangent (`P=1`) from a single
        backward pass."""
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
    for the input pullback; the native lowerings are checked against it.
    """

    name = "torch_autograd"

    # Builds a local graph; callers (structured-pullback providers) may run
    # under no_grad.
    @torch.enable_grad()
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
    """Native (autograd-free) Mamba-3 region lowering. `input_pullback_basis`
        threads a basis of P output cotangents backward through the blocks by the
        native block pullbacks; `parameter_vjp` threads the single real adjoint
        and accumulates the parameter grads. `TorchAutogradMamba3Lowering` is the
        reference.
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
        """
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


# Mixer input-pullback contract (the scan pullback composed with the epilogue).


class Mamba3MixerLowering(Protocol):
    """Contract for the Mamba-3 mixer's matrix-valued input pullback:
        `input_pullback_basis(cache, output_cotangent_basis [B, P, L, D])` returns
        the input cotangent basis [B, P, L, D] with the mixer parameters held fixed.
        `P` is the cotangent-basis batch (the interface rank r when materializing
        an interface Jacobian, or 1 for a real adjoint). `cache` is the
        `Mamba3ForwardCache` documented in this module.
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
    """Native (autograd-free) mixer input pullback: `out_proj^T`, the scan
    pullback (`mamba3_siso_scan_input_pullback_basis`), the preprocess and
    RMSNorm VJPs, and `in_proj^T`."""

    name = "native"

    def input_pullback_basis(
        self,
        *,
        mixer: nn.Module,
        cache: Mamba3ForwardCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        return mamba3_mixer_input_pullback_native(
            mixer=mixer,
            cache=cache,
            output_cotangent_basis=output_cotangent_basis,
        )


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
    """Matrix-valued input pullback of the SISO selective scan: a basis of P
        cotangents on the scan output `[B, P, L, H, Dv]` to the activation adjoints
        on the scan inputs (Q=C, K=B, V=x, ADT, DT, Trap, Angles, Z), parameters
        held fixed. The scan forward is recomputed once with `mamba3_siso_fwd` and
        shared across the P directions, each driven through the Triton backward
        kernels.
        """
    if output_cotangent_basis.dim() != 5:
        raise ValueError("output_cotangent_basis must have shape [B, P, L, H, Dv]")

    tiles = _mamba3_recompute_scan_tiles(mixer, cache)
    scan = tiles["scan"]
    columns: dict[str, list[torch.Tensor]] = {name: [] for name in MAMBA3_SCAN_INPUT_NAMES}
    for basis_index in range(output_cotangent_basis.shape[1]):
        grad_out = output_cotangent_basis[:, basis_index].to(device=tiles["out_v"].device, dtype=scan["V"].dtype)
        inputs, _params = _mamba3_scan_direction_backward(mixer, tiles, grad_out)
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


def _mamba3_scan_direction_backward(mixer: nn.Module, tiles: dict, grad_out: torch.Tensor):
    """One-direction SISO scan backward. Returns (scan-input adjoints, scan-param grads
    `{dD, dC_bias, dB_bias}`). The input pullback discards the params; the parameter VJP keeps them."""
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
    """VJP of RMSNorm (no bias, no gate) with respect to its input `x`. `dy`
        may carry a leading `P` axis that broadcasts against the shared `x`;
        returns `dx` in fp32.
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
    """Native (autograd-free) mixer input pullback: a basis of P cotangents on
        the mixer output `[B, P, L, D]` to the cotangent on the mixer input `u`,
        parameters held fixed. Composes `out_proj^T`, the scan pullback, the
        preprocess and RMSNorm VJPs, and `in_proj^T`, recomputing the preprocess
        intermediates from the frozen forward cache.
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
    """Native (autograd-free) pullback for one `Mamba3Block`, whose forward is
        `residual_out = hidden_in + residual_in; hidden_out = mixer(RMSNorm(residual_out))`.
        Given the output cotangents `(g_hidden_out, g_residual_out)`, each with a
        leading P axis, returns the input cotangents `(g_hidden_in, g_residual_in)`
        by composing the mixer input pullback (`mixer_pullback`, default the native
        path) and the RMSNorm VJP.
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


# Compiled VJP peripheral stages: the elementwise/norm/projection tails around the
# opaque scan kernels.
_VJP_COMPILED: dict = {}


def _vjp_compiled(fn):
    c = _VJP_COMPILED.get(fn)
    if c is None:
        c = torch.compile(fn, dynamic=False)
        _VJP_COMPILED[fn] = c
    return c


def _block_norm_vjp(g_u, residual_output, norm_w, eps):
    """Block-norm pullback: input grad and weight grad in one stage."""
    x_norm = residual_output.to(norm_w.dtype)
    return (_rmsnorm_vjp(g_u, x_norm, norm_w, eps),
            _rmsnorm_weight_grad(g_u, x_norm, eps))


def _mixer_vjp_head(ct, y_inner, w_out, nheads, hdim):
    """out_proj weight grad and the scan-output cotangent in head layout.
    The weight-grad GEMM runs on bf16 operands (fp32 accumulation)."""
    g_wout = torch.einsum("bld,ble->de", ct.to(w_out.dtype),
                          y_inner.to(w_out.dtype)).to(w_out.dtype)
    g_scan = (ct.to(w_out.dtype) @ w_out).reshape(
        ct.shape[0], ct.shape[1], nheads, hdim)
    return g_wout, g_scan.contiguous()


def _mixer_vjp_tail(gQ, gK, gZ, gV, gADT, gDT, gTrap, gAng,
                    in_proj_cached, input_u, dd_A, dd_dt,
                    w_in, dt_bias, cnorm_w, bnorm_w,
                    d_inner, d_state, nheads, n_angles, c_eps, b_eps, a_floor):
    """Pre-projection VJP tail: B/C-norm pullbacks, the dt/A activation
    chains, and the in_proj reductions."""
    bsz, seqlen = input_u.shape[0], input_u.shape[1]
    sizes = [d_inner, d_inner, d_state, d_state, nheads, nheads, nheads, n_angles]
    _z, _x, b_pre, c_pre, _ddt, _ddA, _trap, _ang = torch.split(in_proj_cached, sizes, dim=-1)
    ddA_f = dd_A.float()
    ddt_f = dd_dt.float()
    softplus_a = torch.nn.functional.softplus(ddA_f)
    a_clamped = torch.clamp(-softplus_a, max=-a_floor)
    clamp_mask = (-softplus_a < -a_floor).float()
    sig_dt = torch.sigmoid(ddt_f + dt_bias.float())
    sig_a = torch.sigmoid(ddA_f)
    dt_val = torch.nn.functional.softplus(ddt_f + dt_bias.float())

    dC_post = gQ.reshape(bsz, seqlen, d_state)
    dB_post = gK.reshape(bsz, seqlen, d_state)
    g_c_pre = _rmsnorm_vjp(dC_post, c_pre, cnorm_w, c_eps)
    g_b_pre = _rmsnorm_vjp(dB_post, b_pre, bnorm_w, b_eps)
    g_cnw = _rmsnorm_weight_grad(dC_post, c_pre, c_eps)
    g_bnw = _rmsnorm_weight_grad(dB_post, b_pre, b_eps)

    g_z = gZ.reshape(bsz, seqlen, d_inner)
    g_x = gV.reshape(bsz, seqlen, d_inner)
    g_adt = gADT.transpose(-1, -2).float()
    g_dt = gDT.transpose(-1, -2).float()
    g_ddt = (g_dt + g_adt * a_clamped) * sig_dt
    g_ddA = (g_adt * dt_val) * clamp_mask * (-sig_a)
    g_dtb = g_ddt.reshape(-1, nheads).sum(0)
    g_trap = gTrap.transpose(-1, -2)
    g_ang = gAng.sum(dim=-2)

    dt = w_in.dtype
    g_inproj = torch.cat(
        [g_z.to(dt), g_x.to(dt), g_b_pre.to(dt), g_c_pre.to(dt),
         g_ddt.to(dt), g_ddA.to(dt), g_trap.to(dt), g_ang.to(dt)],
        dim=-1,
    )
    g_win = torch.einsum("blc,bld->cd", g_inproj, input_u.to(dt))
    g_u = g_inproj @ w_in
    return g_u, g_win, g_dtb, g_cnw, g_bnw


def mamba3_mixer_param_vjp_native(
    *,
    mixer: nn.Module,
    cache: Mamba3ForwardCache,
    output_cotangent: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Native mixer parameter VJP at `P = 1`: returns `(g_u [B, L, D],
    {param -> grad})`. The scan pullback runs the SISO backward kernels; the
    peripheral on either side runs as two compiled stages."""
    grads: dict = {}
    g_wout, g_scan_out = _vjp_compiled(_mixer_vjp_head)(
        output_cotangent, cache.y_inner, mixer.out_proj.weight,
        mixer.nheads, mixer.headdim)
    grads[mixer.out_proj.weight] = g_wout

    # scan pullback at P=1: scan-input adjoints + scan-param grads
    tiles = _mamba3_recompute_scan_tiles(mixer, cache)
    inputs, scan_params = _mamba3_scan_direction_backward(mixer, tiles, g_scan_out.to(dtype=tiles["scan"]["V"].dtype))
    grads[mixer.D] = scan_params["dD"].to(mixer.D.dtype)
    grads[mixer.C_bias] = scan_params["dC_bias"].unsqueeze(1).to(mixer.C_bias.dtype)
    grads[mixer.B_bias] = scan_params["dB_bias"].unsqueeze(1).to(mixer.B_bias.dtype)

    g_u, g_win, g_dtb, g_cnw, g_bnw = _vjp_compiled(_mixer_vjp_tail)(
        inputs["Q"], inputs["K"], inputs["Z"], inputs["V"], inputs["ADT"],
        inputs["DT"], inputs["Trap"], inputs["Angles"],
        cache.in_proj, cache.input_u, cache.dd_A, cache.dd_dt,
        mixer.in_proj.weight, mixer.dt_bias, mixer.C_norm.weight,
        mixer.B_norm.weight, mixer.d_inner, mixer.d_state, mixer.nheads,
        mixer.num_rope_angles, mixer.C_norm.eps, mixer.B_norm.eps,
        mixer.A_floor)
    grads[mixer.dt_bias] = g_dtb.to(mixer.dt_bias.dtype)
    grads[mixer.C_norm.weight] = g_cnw.to(mixer.C_norm.weight.dtype)
    grads[mixer.B_norm.weight] = g_bnw.to(mixer.B_norm.weight.dtype)
    grads[mixer.in_proj.weight] = g_win
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
    g_res_from_norm, g_norm_w = _vjp_compiled(_block_norm_vjp)(
        g_u, cache.residual_output, block.norm.weight, block.norm.eps)
    g_res_from_norm = g_res_from_norm.to(output_cotangent.dtype)
    grads[block.norm.weight] = g_norm_w.to(block.norm.weight.dtype)

    if residual_cotangent is None:
        g_residual_total = g_res_from_norm
    else:
        g_residual_total = g_res_from_norm + residual_cotangent.to(g_res_from_norm.dtype)
    g_hidden_in = g_residual_total
    g_residual_in = None if cache.residual_input is None else g_residual_total.clone()
    return g_hidden_in, g_residual_in, grads
