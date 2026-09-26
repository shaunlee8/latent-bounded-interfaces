from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from backward.base import GradMap, ScanBackpropModel, store_named_grads


@dataclass
class RegionBackwardResult:
    """One region's local backward output, computable in any order and reduced
    afterwards. `param_grads` are keyed by parameter name, `canvas_cotangent` is
    the region's `[B, L, D]` term of the canvas gradient, and `shared_partials`
    are its terms for shared interface parameters, keyed by name."""

    param_grads: dict[str, torch.Tensor] = field(default_factory=dict)
    canvas_cotangent: torch.Tensor | None = None
    shared_partials: dict[str, torch.Tensor] = field(default_factory=dict)


def native_region_backward(
    *, model, loss, region_index, num_regions, state_adjoints, cache, g_region_output=None
) -> RegionBackwardResult:
    """Compute one region's parameter grads, canvas cotangent, and shared-parameter
    partials from that region's cache and boundary adjoint alone. Interior regions
    derive the region-output cotangent through `update^T`; the last region takes it
    from the readout path, or from `g_region_output` when a driver supplies it."""
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

    # Region-backend parameter grads and the region-input cotangent from one pass.
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

    # Interface region params from a scoped re-run of decode/update on the
    # frozen cached inputs.
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

    # The canvas cotangent is the region-input cotangent pulled back through the view.
    canvas_cotangent = model.region_view.read_vjp(g_region_input, region_index)

    return RegionBackwardResult(
        param_grads=param_grads,
        canvas_cotangent=canvas_cotangent.detach(),
        shared_partials=shared_partials,
    )


def native_initial_backward(
    *, model, state0, state0_adjoint, cache
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """Pull the initial-state adjoint through `initialize` to the initial
    parameters and the canvas features. Returns the named parameter grads and
    the canvas cotangent partial."""
    initial_params = list(model.interface.initial_vjp_parameters())
    canvas_features = cache["canvas_features"]
    # A frozen embedding leaves the canvas features without a graph.
    inputs = [*initial_params] + ([canvas_features] if canvas_features.requires_grad else [])
    outputs = torch.autograd.grad(
        state0,
        inputs,
        grad_outputs=state0_adjoint,
        retain_graph=True,
        allow_unused=True,
    )
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    param_grads: dict[str, torch.Tensor] = {}
    for param, grad in zip(initial_params, outputs[: len(initial_params)]):
        if grad is not None and id(param) in name_by_id:
            param_grads[name_by_id[id(param)]] = grad.detach().clone()
    canvas_partial = None
    if canvas_features.requires_grad and outputs[-1] is not None:
        canvas_partial = outputs[-1].detach()
    return param_grads, canvas_partial


def reduce_region_results(results: list[RegionBackwardResult]) -> RegionBackwardResult:
    """Reduce per-region results into one: merge the parameter grads, sum the
    canvas cotangents, and sum the shared-parameter partials by name."""
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

        # Interior regions are seeded at their output state; the last region at the loss.
        outputs: list[torch.Tensor] = []
        seeds: list[torch.Tensor] = []
        if region_index < num_regions - 1:
            outputs.append(states[region_index + 1])
            seeds.append(state_adjoints[region_index + 1])
        else:
            outputs.append(loss)
            seeds.append(torch.ones_like(loss))
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
        shared_grads = torch.autograd.grad(
            loss,
            shared_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        self._store_grads(model=model, grad_map=grad_map, params=shared_params, grads=shared_grads)


class NativeLocalVJPProvider:
    """Local VJP provider whose region-backend parameter grads come from the
    backend's native VJP, with the scanned adjoint routed through `update^T`.
    The interface, readout, initial-state, and canvas terms use autograd on
    the retained forward graph."""

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
        if not region_cache.region_output.is_leaf:
            raise ValueError(
                "NativeLocalVJPProvider needs a graph-free region forward: "
                "call forward_with_cache(..., native_backward=True)"
            )
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
        del states
        result = native_region_backward(
            model=model, loss=loss, region_index=region_index,
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

        if self._canvas_grad is None and not tied:
            shared = canvas_params + interface_shared
            grads = torch.autograd.grad(loss, shared, retain_graph=False, allow_unused=True)
            self._store_grads(model=model, grad_map=grad_map, params=shared, grads=grads)
            return

        # Canvas param grads split into frozen region/initialize paths (pulled
        # natively) and live loss paths outside the regions (pulled by autograd).
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
        if canvas_params and tied:
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
