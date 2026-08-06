from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from backward.base import GradMap, ScanBackpropModel, store_named_grads


@dataclass
class RegionBackwardResult:
    """Pure per-region Phase-3 output. It is region-independent with no shared
    state, so
    regions can be computed in any order / on separate streams or GPUs and then
    reduced. `param_grads` are keyed by full parameter name; `canvas_cotangent` is
    that region's `[B, L, D]` contribution to `d(loss)/d(canvas_features)`;
    `shared_partials` are this region's contribution to shared params (e.g.
    `update_scale`), keyed by parameter id."""

    param_grads: dict[str, torch.Tensor] = field(default_factory=dict)
    canvas_cotangent: torch.Tensor | None = None
    shared_partials: dict[str, torch.Tensor] = field(default_factory=dict)


def native_region_backward(
    *, model, loss, states, region_index, num_regions, state_adjoints, cache, g_region_output=None
) -> RegionBackwardResult:
    """Compute one region's parameter grads, canvas cotangent, and shared-param
    partials natively. This is a pure function of that region's cache + boundary adjoint,
    with no dependence on other regions or on any accumulator. This is the unit of
    region-parallel work for Phase 3.

    `g_region_output` (the region-output cotangent) is computed device-locally for
    interior regions via `update^T(scanned adjoint)`. For the LAST region it comes
    from the readout path (`autograd(loss, region_output)`) which lives wherever the
    readout does; a multi-device driver can precompute it there and pass it in."""
    region_cache = cache["region_caches"][region_index]

    # `g_region_output`, given or derived, carries only the readout path for the
    # last region; the tapped state's adjoint lambda_K adds its update path.
    has_tap_adjoints = len(state_adjoints) > num_regions
    if g_region_output is None:
        if region_index < num_regions - 1:
            g_state_out = state_adjoints[region_index + 1].to(dtype=region_cache.state_in.dtype)
            update = model.interface.apply_update_jacobian_t_to_region_output(
                region_cache=region_cache,
                state_out_cotangent_basis=g_state_out.unsqueeze(1),
            )
            g_region_output = update["g_region_output"].squeeze(1)
        else:
            g_region_output = torch.autograd.grad(
                loss, region_cache.region_output, retain_graph=True, allow_unused=False
            )[0]
    if region_index == num_regions - 1 and has_tap_adjoints:
        g_state_out = state_adjoints[num_regions].to(dtype=region_cache.state_in.dtype)
        update = model.interface.apply_update_jacobian_t_to_region_output(
            region_cache=region_cache,
            state_out_cotangent_basis=g_state_out.unsqueeze(1),
        )
        g_region_output = g_region_output + update["g_region_output"].squeeze(1).to(dtype=g_region_output.dtype)
    output_tap_grads = cache.get("output_tap_grads") if isinstance(cache, dict) else None
    if output_tap_grads is not None:
        g_region_output = g_region_output + output_tap_grads[region_index].to(dtype=g_region_output.dtype)

    # Region-backend param grads (NATIVE) + region-input cotangent (P=1) from one pass.
    region_backend = model.region_backend
    combined = getattr(region_backend, "parameter_vjp_with_input_cotangent", None)
    if combined is not None:
        region_grads, g_region_input = combined(
            cache=region_cache.backend_cache, output_cotangent=g_region_output
        )
    else:
        region_grads = region_backend.parameter_vjp(
            cache=region_cache.backend_cache, output_cotangent=g_region_output
        )
        g_region_input = region_backend.input_pullback_basis(
            cache=region_cache.backend_cache,
            output_cotangent_basis=g_region_output.unsqueeze(1),
        ).squeeze(1)
    if g_region_input.dim() == 4:
        g_region_input = g_region_input.squeeze(1)

    param_grads = {f"region_backend.{name}": grad.detach().clone() for name, grad in region_grads.items()}

    # Interface region params (decode/encode/norm) + shared update_scale via a
    # scoped-local re-run from the frozen cached inputs (independent of any graph).
    # Token-wise interfaces read the canvas inside decode, so the re-run also
    # recovers that decode-path canvas cotangent (broadcast decode has none).
    interface_params = list(model.interface.region_vjp_parameters(region_index))
    shared_params = list(model.interface.shared_vjp_parameters())
    tokenwise = bool(getattr(model.interface.spec, "condition_is_tokenwise", False))
    g_condition = g_region_input if tokenwise else g_region_input.sum(dim=1)
    canvas_leaf: torch.Tensor | None = None
    with torch.enable_grad():
        state_in = region_cache.state_in.detach()
        if tokenwise:
            canvas_leaf = region_cache.canvas_features.detach().requires_grad_(True)
            condition = model.interface.decode(state_in, region_index, canvas_features=canvas_leaf)
        else:
            condition = model.interface.decode(state_in, region_index)
        outputs: list[torch.Tensor] = [condition]
        seeds: list[torch.Tensor] = [g_condition.to(dtype=condition.dtype)]
        if region_index < num_regions - 1 or has_tap_adjoints:
            update_step = model.interface.update(
                state_in, region_cache.region_output.detach(), region_index
            )
            outputs.append(update_step.state)
            seeds.append(state_adjoints[region_index + 1].to(dtype=update_step.state.dtype))
        all_params = interface_params + shared_params
        grad_inputs = all_params + ([canvas_leaf] if canvas_leaf is not None else [])
        grads = torch.autograd.grad(
            outputs, grad_inputs, grad_outputs=seeds, allow_unused=True, retain_graph=False
        )
    g_canvas_decode = None
    if canvas_leaf is not None:
        g_canvas_decode = grads[-1]
        grads = grads[:-1]

    name_by_id = {id(p): n for n, p in model.named_parameters()}
    for param, grad in zip(interface_params, grads[: len(interface_params)]):
        if grad is not None and id(param) in name_by_id:
            param_grads[name_by_id[id(param)]] = grad.detach().clone()
    shared_partials: dict[str, torch.Tensor] = {}
    for param, grad in zip(shared_params, grads[len(interface_params):]):
        if grad is not None and id(param) in name_by_id:
            shared_partials[name_by_id[id(param)]] = grad.detach()

    # Per-region canvas view: the canvas-additive path scales by the view gain,
    # and the view parameters take their grads natively from g_region_input.
    canvas_cotangent = g_region_input
    view_params = getattr(model, "region_view_parameters", None)
    view_params = list(view_params(region_index)) if view_params is not None else []
    if view_params:
        gain_param, bias_param = view_params
        canvas_base = region_cache.canvas_features.detach().to(dtype=g_region_input.dtype)
        if id(gain_param) in name_by_id:
            param_grads[name_by_id[id(gain_param)]] = (
                (g_region_input * canvas_base).sum(dim=(0, 1)).detach().to(dtype=gain_param.dtype)
            )
        if id(bias_param) in name_by_id:
            param_grads[name_by_id[id(bias_param)]] = (
                g_region_input.sum(dim=(0, 1)).detach().to(dtype=bias_param.dtype)
            )
        gain = gain_param.detach().to(dtype=g_region_input.dtype)
        canvas_cotangent = g_region_input * (1.0 + gain)
    if g_canvas_decode is not None:
        canvas_cotangent = canvas_cotangent + g_canvas_decode.to(dtype=canvas_cotangent.dtype)

    return RegionBackwardResult(
        param_grads=param_grads,
        canvas_cotangent=canvas_cotangent.detach(),
        shared_partials=shared_partials,
    )


def native_initial_backward(
    *, model, state0, state0_adjoint, cache
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """Initial-state glue: pull the seed adjoint through `initialize` to the initial
    params and the canvas-feature cotangent. Returns `(named param grads, canvas
    cotangent partial)`; the caller folds the canvas partial into the reduction."""
    initial_params = list(model.interface.initial_vjp_parameters())
    canvas_features = cache["canvas_features"]
    outputs = torch.autograd.grad(
        state0,
        [*initial_params, canvas_features],
        grad_outputs=state0_adjoint,
        retain_graph=True,
        allow_unused=True,
    )
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    param_grads: dict[str, torch.Tensor] = {}
    for param, grad in zip(initial_params, outputs[:-1]):
        if grad is not None and id(param) in name_by_id:
            param_grads[name_by_id[id(param)]] = grad.detach().clone()
    canvas_partial = outputs[-1].detach() if outputs[-1] is not None else None
    return param_grads, canvas_partial


def reduce_region_results(results: list[RegionBackwardResult]) -> RegionBackwardResult:
    """Reduce independent per-region results into one: merge the (distinct-key)
    param grads, sum the canvas cotangents, and sum the shared-param partials by id.
    This is the explicit reduction a parallel driver runs after the region-parallel
    section, replacing the sequential in-place accumulation."""
    merged: dict[str, torch.Tensor] = {}
    canvas: torch.Tensor | None = None
    shared: dict[str, torch.Tensor] = {}
    for result in results:
        merged.update(result.param_grads)
        if result.canvas_cotangent is not None:
            canvas = (
                result.canvas_cotangent
                if canvas is None
                else canvas + result.canvas_cotangent.to(canvas.dtype)
            )
        for name, grad in result.shared_partials.items():
            shared[name] = grad if name not in shared else shared[name] + grad.to(shared[name].dtype)
    return RegionBackwardResult(param_grads=merged, canvas_cotangent=canvas, shared_partials=shared)


class LocalVJPProvider(Protocol):
    name: str

    def new_grad_map(self, model: ScanBackpropModel) -> GradMap:
        ...

    def store_output_head_grads(
        self,
        *,
        model: ScanBackpropModel,
        loss: torch.Tensor,
        grad_map: GradMap,
        cache: Any = None,
    ) -> None:
        ...

    def state_adjoint_from_loss(
        self,
        *,
        loss: torch.Tensor,
        state: torch.Tensor,
        cache: Any = None,
        model: Any = None,
    ) -> torch.Tensor:
        ...

    def store_initial_interface_grads(
        self,
        *,
        model: ScanBackpropModel,
        state0: torch.Tensor,
        state0_adjoint: torch.Tensor,
        grad_map: GradMap,
        cache: Any = None,
    ) -> None:
        ...

    def store_region_grads(
        self,
        *,
        model: ScanBackpropModel,
        loss: torch.Tensor,
        states: list[torch.Tensor],
        region_index: int,
        num_regions: int,
        state_adjoints: list[torch.Tensor],
        grad_map: GradMap,
        cache: Any = None,
    ) -> None:
        ...

    def store_shared_canvas_grads(
        self,
        *,
        model: ScanBackpropModel,
        loss: torch.Tensor,
        grad_map: GradMap,
        cache: Any = None,
    ) -> None:
        ...


class TorchAutogradLocalVJPProvider:
    """Computes local parameter VJPs with torch.autograd.grad."""

    name = "torch_autograd"

    def _param_name_map(self, model: ScanBackpropModel) -> dict[int, str]:
        return {id(param): name for name, param in model.named_parameters() if param.requires_grad}

    def new_grad_map(self, model: ScanBackpropModel) -> GradMap:
        return {name: None for name, param in model.named_parameters() if param.requires_grad}

    def _store_grads(
        self,
        *,
        model: ScanBackpropModel,
        grad_map: GradMap,
        params: list[torch.nn.Parameter],
        grads: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None],
    ) -> None:
        store_named_grads(grad_map, self._param_name_map(model), params, grads)

    def store_output_head_grads(
        self,
        *,
        model: ScanBackpropModel,
        loss: torch.Tensor,
        grad_map: GradMap,
        cache: Any = None,
    ) -> None:
        head_params = model.output_head_vjp_parameters()
        head_grads = torch.autograd.grad(
            loss,
            head_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        self._store_grads(model=model, grad_map=grad_map, params=head_params, grads=head_grads)

    def output_tap_state_sources(
        self,
        *,
        model: ScanBackpropModel,
        cache: Any,
        tap_grads: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Input-side scan sources for the output-readout path: pull each tap's
        direct cotangent back through the region graph to its input state."""
        sources: list[torch.Tensor] = []
        for region_index, grad in enumerate(tap_grads):
            source = torch.autograd.grad(
                cache["region_outputs"][region_index],
                cache["states"][region_index],
                grad_outputs=grad,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            sources.append(source.detach())
        return sources

    def state_adjoint_from_loss(
        self,
        *,
        loss: torch.Tensor,
        state: torch.Tensor,
        cache: Any = None,
        model: Any = None,
    ) -> torch.Tensor:
        return torch.autograd.grad(
            loss,
            state,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0].detach().to(device=state.device, dtype=state.dtype)

    def store_initial_interface_grads(
        self,
        *,
        model: ScanBackpropModel,
        state0: torch.Tensor,
        state0_adjoint: torch.Tensor,
        grad_map: GradMap,
        cache: Any = None,
    ) -> None:
        initial_params = model.interface.initial_vjp_parameters()
        initial_grads = torch.autograd.grad(
            state0,
            initial_params,
            grad_outputs=state0_adjoint,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        self._store_grads(model=model, grad_map=grad_map, params=initial_params, grads=initial_grads)

    def store_region_grads(
        self,
        *,
        model: ScanBackpropModel,
        loss: torch.Tensor,
        states: list[torch.Tensor],
        region_index: int,
        num_regions: int,
        state_adjoints: list[torch.Tensor],
        grad_map: GradMap,
        cache: Any = None,
    ) -> None:
        region_params: list[torch.nn.Parameter] = list(model.region_backend.parameters_for_region(region_index))
        region_params.extend(model.interface.region_vjp_parameters(region_index))
        view_params = getattr(model, "region_view_parameters", None)
        if view_params is not None:
            region_params.extend(view_params(region_index))

        # Severed taps redirect some loss paths into explicit seeded outputs:
        # the tapped last state (lambda_K) and each region's tapped output.
        outputs: list[torch.Tensor] = []
        seeds: list[torch.Tensor] = []
        if region_index < num_regions - 1:
            outputs.append(states[region_index + 1])
            seeds.append(state_adjoints[region_index + 1])
        else:
            outputs.append(loss)
            seeds.append(torch.ones_like(loss))
            if len(state_adjoints) > num_regions:
                outputs.append(states[num_regions])
                seeds.append(state_adjoints[num_regions])
        tap_grads = (cache or {}).get("output_tap_grads") if cache is not None else None
        if tap_grads is not None:
            outputs.append(cache["region_outputs"][region_index])
            seeds.append(tap_grads[region_index])
        grads = torch.autograd.grad(
            outputs,
            region_params,
            grad_outputs=seeds,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        self._store_grads(model=model, grad_map=grad_map, params=region_params, grads=grads)

    def store_shared_canvas_grads(
        self,
        *,
        model: ScanBackpropModel,
        loss: torch.Tensor,
        grad_map: GradMap,
        cache: Any = None,
    ) -> None:
        shared_params = model.shared_local_vjp_parameters()
        # Detached boundary-state taps sever the loss graph at each tapped
        # state; re-seed those paths with the taps' direct cotangents so shared
        # params keep their through-state contributions.
        taps = list((cache or {}).get("state_taps") or []) if cache is not None else []
        output_tap_grads = (cache or {}).get("output_tap_grads") if cache is not None else None
        if (taps and all(tap.is_leaf and tap.requires_grad for tap in taps)) or output_tap_grads is not None:
            outputs = [loss]
            grad_outputs: list[torch.Tensor] = [torch.ones_like(loss)]
            if taps and all(tap.is_leaf and tap.requires_grad for tap in taps):
                tap_sources = torch.autograd.grad(loss, taps, retain_graph=True, allow_unused=False)
                states = list(cache["states"])
                outputs.extend(states[1:])
                grad_outputs.extend(
                    source.to(dtype=state.dtype) for source, state in zip(tap_sources, states[1:])
                )
            if output_tap_grads is not None:
                outputs.extend(cache["region_outputs"])
                grad_outputs.extend(output_tap_grads)
            shared_grads = torch.autograd.grad(
                outputs,
                shared_params,
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        else:
            shared_grads = torch.autograd.grad(
                loss,
                shared_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        self._store_grads(model=model, grad_map=grad_map, params=shared_params, grads=shared_grads)


class NativeLocalVJPProvider:
    """Local VJP provider that computes the region-backend parameter grads
    natively via `region_backend.parameter_vjp` (no autograd through the region),
    routing the scanned adjoint through the interface's `update^T`.

    The expensive region-backend param grads are native; the cheap glue
    (interface region params, readout, initial state, canvas, seed) uses
    autograd on the retained forward graph.
    """

    name = "native_local"

    def __init__(self) -> None:
        # Accumulated cotangent on canvas_features [B, L, D] over the backward.
        self._canvas_grad: torch.Tensor | None = None
        # Accumulated grads for shared interface params (e.g. update_scale),
        # keyed by param name, summed across regions.
        self._shared_grads: dict[str, torch.Tensor] = {}

    def _param_name_map(self, model: ScanBackpropModel) -> dict[int, str]:
        return {id(param): name for name, param in model.named_parameters() if param.requires_grad}

    def new_grad_map(self, model: ScanBackpropModel) -> GradMap:
        self._canvas_grad = None
        self._shared_grads = {}
        self._name_by_id = {id(p): n for n, p in model.named_parameters()}
        return {name: None for name, param in model.named_parameters() if param.requires_grad}

    def _accumulate_canvas(self, g: torch.Tensor) -> None:
        g = g.detach()
        self._canvas_grad = g if self._canvas_grad is None else self._canvas_grad + g.to(self._canvas_grad.dtype)

    def _store_grads(self, *, model, grad_map, params, grads) -> None:
        store_named_grads(grad_map, self._param_name_map(model), params, grads)

    def store_output_head_grads(self, *, model, loss, grad_map, cache: Any = None) -> None:
        head_params = model.output_head_vjp_parameters()
        head_grads = torch.autograd.grad(loss, head_params, retain_graph=True, allow_unused=True)
        self._store_grads(model=model, grad_map=grad_map, params=head_params, grads=head_grads)

    def output_tap_state_sources(
        self,
        *,
        model,
        cache: Any,
        tap_grads: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Native input-side scan sources: pull each output tap's cotangent
        through the backend's input pullback and the decode transpose."""
        sources: list[torch.Tensor] = []
        for region_index, grad in enumerate(tap_grads):
            region_cache = cache["region_caches"][region_index]
            g_region_input = model.region_backend.input_pullback_basis(
                cache=region_cache.backend_cache,
                output_cotangent_basis=grad.to(dtype=region_cache.region_output.dtype).unsqueeze(1),
            )
            decode = model.interface.apply_decode_jacobian_t_to_state_input(
                region_cache=region_cache,
                region_input_cotangent_basis=g_region_input,
            )
            sources.append(decode["g_state_input"].squeeze(1).detach())
        return sources

    def state_adjoint_from_loss(self, *, loss, state, cache: Any = None, model: Any = None) -> torch.Tensor:
        # Seed: dL/d(last region input state) = decode^T(input_pullback_basis(
        # dL/d region_output[-1])); only the cheap readout path uses autograd.
        region_cache = cache["region_caches"][-1]
        g_region_output = torch.autograd.grad(
            loss, region_cache.region_output, retain_graph=True, allow_unused=False
        )[0].to(dtype=region_cache.region_output.dtype)
        g_region_input = model.region_backend.input_pullback_basis(
            cache=region_cache.backend_cache,
            output_cotangent_basis=g_region_output.unsqueeze(1),
        )
        decode = model.interface.apply_decode_jacobian_t_to_state_input(
            region_cache=region_cache,
            region_input_cotangent_basis=g_region_input,
        )
        return decode["g_state_input"].squeeze(1).detach().to(device=state.device, dtype=state.dtype)

    def store_initial_interface_grads(self, *, model, state0, state0_adjoint, grad_map, cache: Any = None) -> None:
        # state0 = initialize(canvas_features): pull the seed adjoint to the
        # initial params and the canvas-feature cotangent.
        param_grads, canvas_partial = native_initial_backward(
            model=model, state0=state0, state0_adjoint=state0_adjoint, cache=cache
        )
        for name, grad in param_grads.items():
            if name in grad_map:
                grad_map[name] = grad
        if canvas_partial is not None:
            self._accumulate_canvas(canvas_partial)

    def store_region_grads(
        self, *, model, loss, states, region_index, num_regions, state_adjoints, grad_map, cache: Any = None
    ) -> None:
        # Sequential fold of the pure per-region result; the parallel driver
        # instead collects results and calls reduce_region_results.
        result = native_region_backward(
            model=model, loss=loss, states=states, region_index=region_index,
            num_regions=num_regions, state_adjoints=state_adjoints, cache=cache,
        )
        self.apply_region_result(result, grad_map)

    def apply_region_result(self, result: RegionBackwardResult, grad_map: GradMap) -> None:
        for name, grad in result.param_grads.items():
            if name in grad_map:
                grad_map[name] = grad
        if result.canvas_cotangent is not None:
            self._accumulate_canvas(result.canvas_cotangent)
        for name, grad in result.shared_partials.items():
            prev = self._shared_grads.get(name)
            self._shared_grads[name] = grad if prev is None else prev + grad.to(prev.dtype)

    def store_shared_canvas_grads(self, *, model, loss, grad_map, cache: Any = None) -> None:
        canvas_params = list(model.canvas_vjp_parameters())
        interface_shared = list(model.interface.shared_vjp_parameters())
        tied = bool(getattr(model, "tie_embeddings", False))
        has_taps = cache is not None and bool(cache.get("state_readout_terms") or [])

        if self._canvas_grad is None and not (tied or has_taps):
            shared = canvas_params + interface_shared
            grads = torch.autograd.grad(loss, shared, retain_graph=False, allow_unused=True)
            self._store_grads(model=model, grad_map=grad_map, params=shared, grads=grads)
            return

        # Canvas param grads split into two disjoint path families: the frozen
        # region/initialize paths, pulled natively from the accumulated
        # canvas-feature cotangent; and the live loss paths outside the regions
        # (tied readout, boundary-state readout taps), pulled by autograd.
        totals: dict[int, torch.Tensor] = {}
        if canvas_params and self._canvas_grad is not None:
            canvas_features = cache["canvas_features"]
            seeded = torch.autograd.grad(
                canvas_features,
                canvas_params,
                grad_outputs=self._canvas_grad.to(dtype=canvas_features.dtype),
                retain_graph=True,
                allow_unused=True,
            )
            for param, grad in zip(canvas_params, seeded):
                if grad is not None:
                    totals[id(param)] = grad.detach().clone()
        if canvas_params and (tied or has_taps):
            direct = torch.autograd.grad(loss, canvas_params, retain_graph=False, allow_unused=True)
            for param, grad in zip(canvas_params, direct):
                if grad is None:
                    continue
                prev = totals.get(id(param))
                totals[id(param)] = grad.detach().clone() if prev is None else prev + grad.to(prev.dtype)
        name_by_id = self._param_name_map(model)
        for param in canvas_params:
            name = name_by_id.get(id(param))
            grad = totals.get(id(param))
            if name is not None and grad is not None:
                grad_map[name] = grad

        # Shared interface params (e.g. update_scale): accumulated per-region from
        # the scoped-local update re-runs in store_region_grads (no region graph).
        for param in interface_shared:
            name = name_by_id.get(id(param))
            grad = self._shared_grads.get(name) if name is not None else None
            if grad is not None:
                grad_map[name] = grad.detach().clone()
