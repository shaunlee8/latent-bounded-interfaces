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


def _forward_region_state_map(
    *,
    model: Any,
    region_index: int,
    canvas_features: torch.Tensor,
    state_in: torch.Tensor,
) -> torch.Tensor:
    condition = model.interface.decode(state_in, region_index)
    region_input = canvas_features + condition.unsqueeze(1)
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
    raise ValueError("interface pullback provider must be one of: graph, recompute, torch_graph, torch_recompute")
