from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch


class InterfacePullbackProvider(Protocol):
    name: str

    def materialize_state_jacobian_t(
        self,
        *,
        model: Any,
        cache: dict[str, Any],
    ) -> list[torch.Tensor]:
        ...


def materialize_interface_state_jacobian_t_graph(
    *,
    model: Any,
    cache: dict[str, Any],
) -> list[torch.Tensor]:
    states: Sequence[torch.Tensor] = cache["states"]
    if len(states) != model.num_regions + 1:
        raise ValueError("cache has invalid states length.")
    state_jacobians_t: list[torch.Tensor] = []
    for region_index in range(model.num_regions):
        state_in = states[region_index]
        state_out = states[region_index + 1]
        if state_in.dim() != 2 or state_out.dim() != 2:
            raise ValueError("interface states must be [B, R].")
        bsz, rank = state_out.shape
        eye = torch.eye(rank, device=state_out.device, dtype=state_out.dtype)
        grad_out_batched = eye.unsqueeze(1).expand(rank, bsz, rank)
        try:
            grad_in_batched = torch.autograd.grad(
                state_out,
                state_in,
                grad_outputs=grad_out_batched,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
                is_grads_batched=True,
            )[0]
            state_jacobians_t.append(grad_in_batched.permute(1, 2, 0).contiguous())
        except (TypeError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and "doesn't have storage" not in str(exc):
                raise
            cols: list[torch.Tensor] = []
            for j in range(rank):
                grad_out = eye[j].view(1, rank).expand(bsz, rank)
                grad_in = torch.autograd.grad(
                    state_out,
                    state_in,
                    grad_outputs=grad_out,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                cols.append(grad_in.unsqueeze(-1))
            state_jacobians_t.append(torch.cat(cols, dim=-1))
    return state_jacobians_t


def materialize_interface_state_jacobian_t_native(
    *,
    model: Any,
    cache: dict[str, Any],
) -> list[torch.Tensor]:
    """Build every interface Jacobian `A_k^T` from the region caches with the
    native region pullback, one region at a time."""
    region_caches: Sequence[Any] = cache["region_caches"]
    if len(region_caches) != model.num_regions:
        raise ValueError("cache has invalid region_caches length.")

    return [
        interface_state_jacobian_t_for_region(model=model, region_cache=region_cache)
        for region_cache in region_caches
    ]


def interface_state_jacobian_t_for_region(*, model: Any, region_cache: Any) -> torch.Tensor:
    """Native interface Jacobian `A_k^T` for one region, a function of that
    region's cache alone. Transports the identity `I_r` through `update^T`,
    the native region `input_pullback_basis`, and `decode^T`, plus the skip term."""
    state_in = region_cache.state_in
    if state_in.dim() != 2:
        raise ValueError("interface state inputs must be [B, R].")
    bsz, rank = state_in.shape
    # Identity basis on the state output: [B, P=rank, R].
    eye = torch.eye(rank, device=state_in.device, dtype=state_in.dtype)
    state_out_basis = eye.unsqueeze(0).expand(bsz, rank, rank)

    update = model.interface.apply_update_jacobian_t_to_region_output(
        region_cache=region_cache,
        state_out_cotangent_basis=state_out_basis,
    )
    g_region_input = model.region_backend.input_pullback_basis(
        cache=region_cache.backend_cache,
        output_cotangent_basis=update["g_region_output"],
    )
    decode = model.interface.apply_decode_jacobian_t_to_state_input(
        region_cache=region_cache,
        region_input_cotangent_basis=g_region_input,
    )
    # [B, P=R_out, R_in] -> [B, R_in, R_out] to match the autograd providers.
    a_raw = update["g_state_skip"] + decode["g_state_input"]
    return a_raw.permute(0, 2, 1).contiguous().to(dtype=state_in.dtype)


def interface_state_jacobian_for_region_forward(
    *,
    model: Any,
    region_cache: Any,
    region_output_jvp: Any,
    direction_chunk: int = 0,
) -> torch.Tensor:
    """Forward-mode interface Jacobian for one region in the same `A_k^T`
    [B, R_in, R_out] layout as the reverse-mode providers. The identity basis
    on the state input passes through the decode JVP, the region JVP, and the
    update JVP (skip included); `direction_chunk` > 0 drives the basis in chunks
    of that many directions, which bounds the tangent-field peak memory."""
    state_in = region_cache.state_in
    if state_in.dim() != 2:
        raise ValueError("interface state inputs must be [B, R].")
    bsz, rank = state_in.shape
    # Identity basis on the state input: [B, P=rank, R], column j = e_j.
    eye = torch.eye(rank, device=state_in.device, dtype=state_in.dtype)
    chunk = rank if direction_chunk <= 0 or direction_chunk >= rank else int(direction_chunk)
    outs: list[torch.Tensor] = []
    for start in range(0, rank, chunk):
        state_in_basis = eye[start : start + chunk].unsqueeze(0).expand(bsz, -1, rank)
        decode = model.interface.apply_decode_jacobian_to_state_input(
            region_cache=region_cache,
            state_input_tangent_basis=state_in_basis,
        )
        d_region_output = region_output_jvp(decode["d_region_input"])
        update = model.interface.apply_update_jacobian_to_region_output(
            region_cache=region_cache,
            region_output_tangent_basis=d_region_output,
            state_input_tangent_basis=state_in_basis,
        )
        outs.append(update["d_state_out"])
    d_state_out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
    return d_state_out.contiguous().to(dtype=state_in.dtype)


class NativeInterfacePullbackProvider:
    name = "native"

    @torch.no_grad()
    def materialize_state_jacobian_t(
        self,
        *,
        model: Any,
        cache: dict[str, Any],
    ) -> list[torch.Tensor]:
        # Under no_grad: the pullbacks read live parameters and record nothing.
        del self
        return materialize_interface_state_jacobian_t_native(model=model, cache=cache)


class ForwardModeInterfacePullbackProvider:
    """Forward-mode interface Jacobians: each `A_k` is built as a forward-
    linearized scan (decode JVP, region JVP, update JVP) instead of r reverse-
    mode VJPs. The region factor is the backend's `region_output_jvp`."""

    name = "forward_mode"

    def __init__(self, *, use_kernel: "bool | str" = "auto", direction_chunk: int = 0) -> None:
        self.use_kernel = use_kernel
        # 0 drives all r directions in one fused pass.
        self.direction_chunk = int(direction_chunk)

    def _enable_kernel_path(self, model: Any) -> None:
        """Select the fused forward-mode kernel path on the backend. Under "auto"
        the torch.func reference stays selected when tilelang is absent."""
        backend = getattr(model, "region_backend", None)
        if backend is None or self.use_kernel is False:
            return
        if getattr(backend, "forward_mode_use_kernel", None):
            return
        if self.use_kernel == "auto":
            try:
                import tilelang  # noqa: F401
            except Exception:
                return
        backend.forward_mode_use_kernel = True

    @torch.no_grad()
    def materialize_state_jacobian_t(
        self,
        *,
        model: Any,
        cache: dict[str, Any],
    ) -> list[torch.Tensor]:
        # Under no_grad: the JVP chain reads live parameters, and recording it
        # would retain every region's tangent transients at once.
        self._enable_kernel_path(model)
        region_caches: Sequence[Any] = cache["region_caches"]
        if len(region_caches) != model.num_regions:
            raise ValueError("cache has invalid region_caches length.")
        region_output_jvp = getattr(model.region_backend, "region_output_jvp", None)
        if region_output_jvp is None:
            raise NotImplementedError(
                "forward_mode interface provider requires region_backend.region_output_jvp"
            )

        # The update JVP consumes only the pooled region tangent, so the region
        # JVP returns the mean-pooled form.
        jacobians: list[torch.Tensor] = []
        for region_cache in region_caches:

            def _jvp(region_input_tangent_basis: torch.Tensor, _rc: Any = region_cache) -> Any:
                return region_output_jvp(
                    cache=_rc.backend_cache,
                    region_input_tangent_basis=region_input_tangent_basis,
                    pooled=True,
                )

            jacobians.append(
                interface_state_jacobian_for_region_forward(
                    model=model, region_cache=region_cache, region_output_jvp=_jvp,
                    direction_chunk=self.direction_chunk,
                )
            )
        return jacobians


class TorchGraphInterfacePullbackProvider:
    name = "torch_graph"

    def materialize_state_jacobian_t(
        self,
        *,
        model: Any,
        cache: dict[str, Any],
    ) -> list[torch.Tensor]:
        del self
        return materialize_interface_state_jacobian_t_graph(model=model, cache=cache)


def build_interface_pullback_provider(mode: str) -> InterfacePullbackProvider:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"graph", "torch_graph"}:
        return TorchGraphInterfacePullbackProvider()
    if normalized in {"native", "native_region"}:
        return NativeInterfacePullbackProvider()
    if normalized in {"forward", "forward_mode", "fwd", "jvp"}:
        return ForwardModeInterfacePullbackProvider()
    raise ValueError(
        f"interface pullback provider must be one of: graph, native, forward_mode; got {mode!r}"
    )
