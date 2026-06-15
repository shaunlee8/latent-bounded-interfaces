from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict

import torch

from legacy.native_region_interface import NativeRegionInterfaceModel


@dataclass
class NativeBackwardResult:
    grad_map: Dict[str, torch.Tensor | None]
    interface_scan_rms: float
    interface_jacobian_stats: Dict[str, float] | None = None


def _assign_grads(params: list[torch.nn.Parameter], grads: list[torch.Tensor | None]) -> None:
    for p, g in zip(params, grads):
        if g is not None:
            p.grad = g.detach()


def _store_named_grads(
    grad_map: Dict[str, torch.Tensor | None],
    param_name_map: Dict[int, str],
    params: list[torch.nn.Parameter],
    grads: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None],
) -> None:
    for param, grad in zip(params, grads):
        name = param_name_map[id(param)]
        grad_map[name] = None if grad is None else grad.detach().clone()


def _interface_jacobian_stats(
    model: NativeRegionInterfaceModel,
    mats: list[torch.Tensor],
    *,
    include_suffix: bool,
) -> Dict[str, float]:
    """Optional appendix diagnostic computed from already-materialized interface pullbacks."""
    if not mats:
        return {}
    with torch.no_grad():
        dense = torch.stack([mat.detach().float() for mat in mats], dim=1)
        frob = torch.linalg.matrix_norm(dense, ord="fro", dim=(-2, -1))
        spec = torch.linalg.matrix_norm(dense, ord=2, dim=(-2, -1))
        rank = max(1, int(dense.shape[-1]))
        stats = {
            "jac_local_spec_mean": float(spec.mean().item()),
            "jac_local_spec_max": float(spec.max().item()),
            "jac_local_frob_mean": float(frob.mean().item()),
            "jac_local_frob_max": float(frob.max().item()),
            "jac_local_frob_normed_mean": float((frob / math.sqrt(rank)).mean().item()),
        }
        if include_suffix:
            suffix = model.compose_pullback_suffix(mats)
            suffix_dense = torch.stack([mat.detach().float() for mat in suffix], dim=1)
            suffix_spec = torch.linalg.matrix_norm(suffix_dense, ord=2, dim=(-2, -1))
            stats.update(
                {
                    "jac_suffix_spec_mean": float(suffix_spec.mean().item()),
                    "jac_suffix_spec_max": float(suffix_spec.max().item()),
                }
            )
        return stats


def _native_backward_step(
    model: NativeRegionInterfaceModel,
    *,
    ce_loss: torch.Tensor,
    cache: Dict[str, Any],
    interface_jacobian_mode: str = "graph",
    jacobian_basis_chunk: int = 1,
    interface_recompute_context: Any | None = None,
    compute_interface_jacobian_stats: bool = False,
    include_interface_jacobian_suffix: bool = False,
) -> NativeBackwardResult:
    """Active LBI paper backward path.

    The forward graph is regionized. This function materializes local message
    pullbacks, composes them with the suffix scan, and assigns parameter
    gradients without dense cross-region activation backpropagation.
    """
    param_name_map = {id(param): name for name, param in model.named_parameters() if param.requires_grad}
    grad_map: Dict[str, torch.Tensor | None] = {
        name: None for name, param in model.named_parameters() if param.requires_grad
    }

    region_messages = list(cache["region_messages"])
    region_ranges = list(cache["region_ranges"])
    n_regions = len(region_ranges)
    if n_regions == 0:
        raise RuntimeError("native_region_interface requires at least one region.")

    tied_lm_head = bool(getattr(model, "tie_embeddings", False) and model.lm_head.weight is model.embedding.weight)
    head_params = [p for p in model.norm.parameters() if p.requires_grad]
    if not tied_lm_head:
        head_params.extend([p for p in model.lm_head.parameters() if p.requires_grad])
    head_grads = torch.autograd.grad(
        ce_loss,
        head_params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    _store_named_grads(grad_map, param_name_map, head_params, head_grads)

    if n_regions == 1:
        g_msg_inputs = [
            torch.autograd.grad(
                ce_loss,
                region_messages[0],
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0].detach().to(device=region_messages[0].device, dtype=region_messages[0].dtype)
        ]
        interface_scan_rms = 0.0
        interface_jacobian_stats = None
    else:
        g_last_input = torch.autograd.grad(
            ce_loss,
            region_messages[-2],
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0].detach()
        ctx = (
            interface_recompute_context
            if interface_jacobian_mode == "recompute" and interface_recompute_context is not None
            else nullcontext()
        )
        with ctx:
            mats = model.materialize_interface_pullback_mats(
                cache,
                mode=interface_jacobian_mode,
                basis_chunk=jacobian_basis_chunk,
            )
        interface_jacobian_stats = (
            _interface_jacobian_stats(
                model,
                list(mats),
                include_suffix=include_interface_jacobian_suffix,
            )
            if compute_interface_jacobian_stats
            else None
        )
        g_last_input = g_last_input.to(device=mats[0].device, dtype=mats[0].dtype)
        g_msg_inputs = model.interface_vjp_scan_from_last_input_seed(mats, g_last_input)
        g_manual = [torch.zeros_like(g_last_input) for _ in range(n_regions)]
        g_manual[-1] = g_last_input
        for ridx in reversed(range(n_regions - 1)):
            g_manual[ridx] = model._apply_pullback_mat(mats[ridx], g_manual[ridx + 1], out_dtype=g_manual[ridx + 1].dtype)
        diffs = [((a - b) ** 2).mean() for a, b in zip(g_msg_inputs, g_manual)]
        interface_scan_rms = float(torch.sqrt(torch.stack(diffs).mean()).item())

    input_params = [p for p in model.input_to_message.parameters() if p.requires_grad]
    input_grads = torch.autograd.grad(
        region_messages[0],
        input_params,
        grad_outputs=g_msg_inputs[0],
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    _store_named_grads(grad_map, param_name_map, input_params, input_grads)

    for ridx, (start, end) in enumerate(region_ranges):
        region_params: list[torch.nn.Parameter] = []
        for li in range(start, end):
            region_params.extend([p for p in model.blocks[li].parameters() if p.requires_grad])
        region_params.extend([p for p in model.message_to_hidden[ridx].parameters() if p.requires_grad])
        region_params.extend([p for p in model.hidden_to_message[ridx].parameters() if p.requires_grad])
        region_params.extend([p for p in model.message_norm[ridx].parameters() if p.requires_grad])

        if ridx < n_regions - 1:
            grads = torch.autograd.grad(
                region_messages[ridx + 1],
                region_params,
                grad_outputs=g_msg_inputs[ridx + 1],
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
        else:
            grads = torch.autograd.grad(
                ce_loss,
                region_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
        _store_named_grads(grad_map, param_name_map, region_params, grads)

    shared_params = [model.embedding.weight, model.message_alpha]
    shared_grads = torch.autograd.grad(
        ce_loss,
        shared_params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    _store_named_grads(grad_map, param_name_map, shared_params, shared_grads)
    return NativeBackwardResult(
        grad_map=grad_map,
        interface_scan_rms=interface_scan_rms,
        interface_jacobian_stats=interface_jacobian_stats,
    )
