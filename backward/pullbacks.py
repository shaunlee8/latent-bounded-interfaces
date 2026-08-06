from __future__ import annotations

import os
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


def _forward_region_state_map(
    *,
    model: Any,
    region_index: int,
    canvas_features: torch.Tensor,
    state_in: torch.Tensor,
) -> torch.Tensor:
    condition = model.interface.decode(state_in, region_index, canvas_features=canvas_features)
    viewed = getattr(model, "viewed_canvas", None)
    canvas_read = viewed(canvas_features, region_index) if viewed is not None else canvas_features
    region_input = canvas_read + (condition if condition.dim() == 3 else condition.unsqueeze(1))
    region_output, _ = model.region_backend.forward_region(
        region_input=region_input,
        region_index=region_index,
    )
    return model.interface.update(state_in, region_output, region_index).state


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


def materialize_interface_state_jacobian_t_recompute(
    *,
    model: Any,
    cache: dict[str, Any],
    basis_chunk: int,
) -> list[torch.Tensor]:
    if basis_chunk <= 0:
        raise ValueError("basis_chunk must be > 0.")
    region_caches: Sequence[Any] = cache["region_caches"]
    if len(region_caches) != model.num_regions:
        raise ValueError("cache has invalid region_caches length.")

    state_jacobians_t: list[torch.Tensor] = []
    for region_cache in region_caches:
        region_index = region_cache.region_index
        state_base = region_cache.state_in.detach()
        canvas_base = region_cache.canvas_features.detach()
        if state_base.dim() != 2:
            raise ValueError("interface state inputs must be [B, R].")
        if canvas_base.dim() != 3:
            raise ValueError("canvas_features must be [B, L, D].")
        bsz, rank = state_base.shape
        if canvas_base.shape[0] != bsz:
            raise ValueError("canvas_features and state batch sizes do not match.")

        eye = torch.eye(rank, device=state_base.device, dtype=state_base.dtype)
        cols: list[torch.Tensor] = []
        for start in range(0, rank, basis_chunk):
            basis = eye[start : start + basis_chunk]
            chunk = int(basis.shape[0])
            state_rep = (
                state_base.unsqueeze(0)
                .expand(chunk, bsz, rank)
                .reshape(chunk * bsz, rank)
                .contiguous()
                .requires_grad_(True)
            )
            canvas_rep = (
                canvas_base.unsqueeze(0)
                .expand(chunk, *canvas_base.shape)
                .reshape(chunk * bsz, canvas_base.shape[1], canvas_base.shape[2])
                .contiguous()
            )
            state_out_rep = _forward_region_state_map(
                model=model,
                region_index=region_index,
                canvas_features=canvas_rep,
                state_in=state_rep,
            )
            grad_out = (
                basis.unsqueeze(1)
                .expand(chunk, bsz, rank)
                .reshape(chunk * bsz, rank)
                .to(device=state_out_rep.device, dtype=state_out_rep.dtype)
            )
            grad_in = torch.autograd.grad(
                state_out_rep,
                state_rep,
                grad_outputs=grad_out,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            cols.append(grad_in.reshape(chunk, bsz, rank).permute(1, 2, 0).contiguous())
        state_jacobians_t.append(torch.cat(cols, dim=-1).to(device=state_base.device, dtype=state_base.dtype))
    return state_jacobians_t


def materialize_interface_state_jacobian_t_native(
    *,
    model: Any,
    cache: dict[str, Any],
) -> list[torch.Tensor]:
    """Build each interface Jacobian `A_k` natively, without differentiating the
    region forward graph.

    Composes the interface's structured pullbacks with the region backend's
    native `input_pullback_basis`: transport the identity `I_r` as cotangents on
    the state output, apply `update^T` (interface), then `J_region^T` (native
    region pullback), then `decode^T` (interface), and add the skip term. The
    result matches `materialize_interface_state_jacobian_t_graph`/`_recompute`
    but the expensive region factor is native.

    The region pullback pushes the full `P=r` identity basis through the
    region in one pass; splitting `r` into outer chunks repeats the
    cotangent-independent scan-tile recompute per chunk and is a net loss."""
    region_caches: Sequence[Any] = cache["region_caches"]
    if len(region_caches) != model.num_regions:
        raise ValueError("cache has invalid region_caches length.")

    return [
        interface_state_jacobian_t_for_region(model=model, region_cache=region_cache)
        for region_cache in region_caches
    ]


def interface_state_jacobian_t_for_region(*, model: Any, region_cache: Any) -> torch.Tensor:
    """Native interface Jacobian `A_k^T` for a single region. It is a pure
    function of that region's cache alone, so Phase 1 can be dispatched per
    region across streams/GPUs. Transports the identity `I_r` through `update^T`,
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
) -> torch.Tensor:
    """FORWARD-MODE interface Jacobian for a single region, returned in the same
    `A_k^T` [B, R_in, R_out] layout as the reverse-mode providers.

    The forward-linearized construction A_k = skip + encode . J_region .
    decode applied to the identity input basis (the dual of the reverse-mode
    r VJPs). Transport
    `I_r` as a state-input tangent basis through `decode` (JVP), then the region
    forward (`region_output_jvp`, the forward-mode scan/kernel), then `update`
    (JVP, including the skip). `region_output_jvp` maps a region-input tangent
    basis [B, P, L, D] -> region-output tangent basis [B, P, L, D].

    The returned tensor matches the reverse providers EXACTLY (same object,
    opposite mode): update JVP output U[b, p=i, o] = d state_out_o / d state_in_i
    = (A_k^T)[i, o], no transpose needed.

    Interfaces exposing `state_tangent_support_starts` (per-coordinate first
    token of nonzero decode tangent, e.g. strict chunk-causal decodes) get a
    lane-restricted construction: coordinates with an empty decode path skip
    the decode/region/update chain entirely and take the closed-form skip +
    norm row instead. Exact; only the zero lanes are dropped."""
    state_in = region_cache.state_in
    if state_in.dim() != 2:
        raise ValueError("interface state inputs must be [B, R].")
    bsz, rank = state_in.shape
    # Identity basis on the state INPUT: [B, P=rank, R], column j = e_j.
    eye = torch.eye(rank, device=state_in.device, dtype=state_in.dtype)

    active_idx = idle_idx = None
    support_fn = getattr(model.interface, "state_tangent_support_starts", None)
    if support_fn is not None:
        seq_len = int(region_cache.canvas_features.shape[1])
        starts = support_fn(seq_len).to(device=state_in.device)
        idle = starts >= seq_len
        if bool(idle.any()):
            active_idx = torch.nonzero(~idle).squeeze(-1)
            idle_idx = torch.nonzero(idle).squeeze(-1)

    if active_idx is None:
        state_in_basis = eye.unsqueeze(0).expand(bsz, rank, rank)
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
        return update["d_state_out"].contiguous().to(dtype=state_in.dtype)

    basis_active = eye[active_idx].unsqueeze(0).expand(bsz, active_idx.shape[0], rank)
    decode = model.interface.apply_decode_jacobian_to_state_input(
        region_cache=region_cache,
        state_input_tangent_basis=basis_active,
    )
    # Suffix-aware region JVP: lanes grouped by their common support start so
    # backends that honor `tangent_token_start` compute only each group's
    # token suffix. Callables without the kwarg get one plain full call.
    import inspect

    try:
        params = inspect.signature(region_output_jvp).parameters
        accepts_start = "tangent_token_start" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        accepts_start = False
    if accepts_start:
        starts_active = starts[active_idx]
        d_in = decode["d_region_input"]
        groups: list[tuple[int, int, int]] = []
        lo = 0
        for p in range(1, starts_active.shape[0] + 1):
            if p == starts_active.shape[0] or starts_active[p] != starts_active[lo]:
                groups.append((lo, p, int(starts_active[lo])))
                lo = p
        if len(groups) == 1:
            d_region_output = region_output_jvp(d_in, tangent_token_start=groups[0][2])
        else:
            parts = [
                region_output_jvp(d_in[:, g_lo:g_hi], tangent_token_start=g_s)
                for g_lo, g_hi, g_s in groups
            ]
            if isinstance(parts[0], tuple):
                d_region_output = (
                    torch.cat([p[0] for p in parts], dim=1),
                    torch.cat([p[1] for p in parts], dim=1) if parts[0][1] is not None else None,
                )
            else:
                d_region_output = torch.cat(parts, dim=1)
    else:
        d_region_output = region_output_jvp(decode["d_region_input"])
    update = model.interface.apply_update_jacobian_to_region_output(
        region_cache=region_cache,
        region_output_tangent_basis=d_region_output,
        state_input_tangent_basis=basis_active,
    )
    basis_idle = eye[idle_idx].unsqueeze(0).expand(bsz, idle_idx.shape[0], rank)
    skip = model.interface.apply_update_jacobian_skip_only(
        region_cache=region_cache,
        state_input_tangent_basis=basis_idle,
    )
    rows = torch.empty(bsz, rank, rank, device=state_in.device, dtype=state_in.dtype)
    rows[:, active_idx] = update["d_state_out"].to(dtype=state_in.dtype)
    rows[:, idle_idx] = skip["d_state_out"].to(dtype=state_in.dtype)
    return rows.contiguous()


class NativeInterfacePullbackProvider:
    name = "native"

    def __init__(self, *, basis_chunk: int = 1) -> None:
        self.basis_chunk = int(basis_chunk)

    @torch.no_grad()
    def materialize_state_jacobian_t(
        self,
        *,
        model: Any,
        cache: dict[str, Any],
    ) -> list[torch.Tensor]:
        # no_grad is load-bearing (see the forward provider); helpers that
        # need autograd re-enable it locally.
        del self
        return materialize_interface_state_jacobian_t_native(model=model, cache=cache)


class ForwardModeInterfacePullbackProvider:
    """Forward-mode interface Jacobians: build each `A_k` as a forward-
    linearized scan (`decode`-JVP -> region-JVP -> `update`-JVP) instead of r
    reverse-mode VJPs. The region factor is the backend's `region_output_jvp`;
    the interface factors are exact JVPs, adjoints of the reverse methods.
    Same `A_k^T` layout as the other providers."""

    name = "forward_mode"

    def __init__(self, *, basis_chunk: int = 1, use_kernel: "bool | str" = "auto") -> None:
        self.basis_chunk = int(basis_chunk)
        self.use_kernel = use_kernel

    def _enable_kernel_path(self, model: Any) -> None:
        """Request the fused dual-scan kernel path on the backend (best-effort
        under "auto": falls back to the torch.func reference without tilelang)."""
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
        # no_grad is load-bearing: the JVP chain reads live parameters, so
        # recording it would retain every region's tangent transients at once.
        self._enable_kernel_path(model)
        region_caches: Sequence[Any] = cache["region_caches"]
        if len(region_caches) != model.num_regions:
            raise ValueError("cache has invalid region_caches length.")
        region_output_jvp = getattr(model.region_backend, "region_output_jvp", None)
        if region_output_jvp is None:
            raise NotImplementedError(
                "forward_mode interface provider requires region_backend.region_output_jvp "
                "(the native forward-mode scan). No autograd fallback."
            )

        # The region tangent's only consumers are pooling contractions, so ask
        # the backend for the reduced form: the meanpool [B, P, 1, D] for
        # broadcast interfaces (LBI_FWDMODE_POOLED=0 escapes), the projected
        # value/score rows + RMS inner products for token-wise interfaces
        # (LBI_FWDMODE_PROJ=0 escapes to the full [B, P, L, D] basis).
        tokenwise = bool(getattr(model.interface.spec, "condition_is_tokenwise", False))
        pooled = os.environ.get("LBI_FWDMODE_POOLED", "1") == "1" and not tokenwise
        projection_fn = (
            getattr(model.interface, "update_tangent_projection", None)
            if tokenwise and os.environ.get("LBI_FWDMODE_PROJ", "1") == "1"
            else None
        )
        jacobians: list[torch.Tensor] = []
        for region_cache in region_caches:
            projection = inner_map = None
            if projection_fn is not None:
                projection, inner_map = projection_fn(region_cache)

            def _jvp(
                region_input_tangent_basis: torch.Tensor,
                _rc: Any = region_cache,
                _proj: Any = projection,
                _inner: Any = inner_map,
                tangent_token_start: int = 0,
            ) -> Any:
                return region_output_jvp(
                    cache=_rc.backend_cache,
                    region_input_tangent_basis=region_input_tangent_basis,
                    pooled=pooled,
                    output_projection=_proj,
                    output_inner=_inner,
                    tangent_token_start=tangent_token_start,
                )

            jacobians.append(
                interface_state_jacobian_for_region_forward(
                    model=model, region_cache=region_cache, region_output_jvp=_jvp,
                )
            )
        return jacobians


class TorchGraphInterfacePullbackProvider:
    name = "torch_graph"

    def __init__(self, *, basis_chunk: int = 1) -> None:
        self.basis_chunk = int(basis_chunk)

    def materialize_state_jacobian_t(
        self,
        *,
        model: Any,
        cache: dict[str, Any],
    ) -> list[torch.Tensor]:
        del self
        return materialize_interface_state_jacobian_t_graph(model=model, cache=cache)


class TorchRecomputeInterfacePullbackProvider:
    name = "torch_recompute"

    def __init__(self, *, basis_chunk: int = 1) -> None:
        self.basis_chunk = int(basis_chunk)

    def materialize_state_jacobian_t(
        self,
        *,
        model: Any,
        cache: dict[str, Any],
    ) -> list[torch.Tensor]:
        return materialize_interface_state_jacobian_t_recompute(
            model=model,
            cache=cache,
            basis_chunk=self.basis_chunk,
        )


def build_interface_pullback_provider(
    mode: str,
    *,
    basis_chunk: int = 1,
) -> InterfacePullbackProvider:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"graph", "torch_graph", "pytorch_graph"}:
        return TorchGraphInterfacePullbackProvider(basis_chunk=basis_chunk)
    if normalized in {"recompute", "torch_recompute", "pytorch_recompute"}:
        return TorchRecomputeInterfacePullbackProvider(basis_chunk=basis_chunk)
    if normalized in {"native", "native_region"}:
        return NativeInterfacePullbackProvider(basis_chunk=basis_chunk)
    if normalized in {"forward", "forward_mode", "fwd", "jvp"}:
        return ForwardModeInterfacePullbackProvider(basis_chunk=basis_chunk)
    raise ValueError(
        "interface pullback provider must be one of: graph, recompute, native, "
        "forward_mode, torch_graph, torch_recompute"
    )
