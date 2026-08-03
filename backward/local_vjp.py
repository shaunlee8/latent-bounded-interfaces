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
    interface_params = list(model.interface.region_vjp_parameters(region_index))
    shared_params = list(model.interface.shared_vjp_parameters())
    g_condition = g_region_input.sum(dim=1)
    with torch.enable_grad():
        state_in = region_cache.state_in.detach()
        condition = model.interface.decode(state_in, region_index)
        outputs: list[torch.Tensor] = [condition]
        seeds: list[torch.Tensor] = [g_condition.to(dtype=condition.dtype)]
        if region_index < num_regions - 1:
            update_step = model.interface.update(
                state_in, region_cache.region_output.detach(), region_index
            )
            outputs.append(update_step.state)
            seeds.append(state_adjoints[region_index + 1].to(dtype=update_step.state.dtype))
        all_params = interface_params + shared_params
        grads = torch.autograd.grad(
            outputs, all_params, grad_outputs=seeds, allow_unused=True, retain_graph=False
        )

    name_by_id = {id(p): n for n, p in model.named_parameters()}
    for param, grad in zip(interface_params, grads[: len(interface_params)]):
        if grad is not None and id(param) in name_by_id:
            param_grads[name_by_id[id(param)]] = grad.detach().clone()
    shared_partials: dict[str, torch.Tensor] = {}
    for param, grad in zip(shared_params, grads[len(interface_params):]):
        if grad is not None and id(param) in name_by_id:
            shared_partials[name_by_id[id(param)]] = grad.detach()

    return RegionBackwardResult(
        param_grads=param_grads,
        canvas_cotangent=g_region_input.detach(),
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

        if region_index < num_regions - 1:
            grads = torch.autograd.grad(
                states[region_index + 1],
                region_params,
                grad_outputs=state_adjoints[region_index + 1],
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
        else:
            grads = torch.autograd.grad(
                loss,
                region_params,
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

        # Tied embeddings also feed the readout (a path not captured by the
        # canvas-feature cotangent), so fall back to full autograd there.
        if getattr(model, "tie_embeddings", False) or self._canvas_grad is None:
            shared = canvas_params + interface_shared
            grads = torch.autograd.grad(loss, shared, retain_graph=False, allow_unused=True)
            self._store_grads(model=model, grad_map=grad_map, params=shared, grads=grads)
            return

        # Untied: canvas param grads NATIVE from the accumulated canvas-feature
        # cotangent. This traverses only the canvas subgraph, never the regions.
        if canvas_params:
            canvas_features = cache["canvas_features"]
            canvas_grads = torch.autograd.grad(
                canvas_features,
                canvas_params,
                grad_outputs=self._canvas_grad.to(dtype=canvas_features.dtype),
                retain_graph=True,
                allow_unused=True,
            )
            self._store_grads(model=model, grad_map=grad_map, params=canvas_params, grads=canvas_grads)

        # Shared interface params (e.g. update_scale): accumulated per-region from
        # the scoped-local update re-runs in store_region_grads (no region graph).
        name_by_id = self._param_name_map(model)
        for param in interface_shared:
            name = name_by_id.get(id(param))
            grad = self._shared_grads.get(name) if name is not None else None
            if grad is not None:
                grad_map[name] = grad.detach().clone()
