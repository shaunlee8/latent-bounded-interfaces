from __future__ import annotations

from typing import Any, Protocol

import torch

from backward.base import GradMap, ScanBackpropModel, store_named_grads


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
    ) -> None:
        ...

    def state_adjoint_from_loss(
        self,
        *,
        loss: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def store_initial_interface_grads(
        self,
        *,
        model: ScanBackpropModel,
        state0: torch.Tensor,
        state0_adjoint: torch.Tensor,
        grad_map: GradMap,
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
    ) -> None:
        ...

    def store_shared_canvas_grads(
        self,
        *,
        model: ScanBackpropModel,
        loss: torch.Tensor,
        grad_map: GradMap,
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
