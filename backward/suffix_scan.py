from __future__ import annotations

from collections.abc import Sequence

import torch

from cuda.interface import suffix_scan_pullbacks as suffix_scan_jacobian_t


def compose_suffix_jacobian_t(state_jacobians_t: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    count = len(state_jacobians_t)
    if count == 0:
        return []
    bsz, rank, rank2 = state_jacobians_t[0].shape
    if rank != rank2:
        raise ValueError("interface state Jacobian-transpose tensors must be square.")
    stacked_state_jacobians_t = torch.stack(list(state_jacobians_t), dim=1).contiguous()
    suffix = suffix_scan_jacobian_t(stacked_state_jacobians_t)
    if suffix.shape != (bsz, count + 1, rank, rank):
        raise ValueError("suffix scan returned an unexpected shape")
    return [suffix[:, region_index].contiguous() for region_index in range(count + 1)]


def apply_jacobian_t(
    jacobian_t: torch.Tensor,
    cotangent: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    compute_dtype = torch.promote_types(jacobian_t.dtype, cotangent.dtype)
    result = torch.einsum(
        "bij,bj->bi",
        jacobian_t.to(dtype=compute_dtype),
        cotangent.to(device=jacobian_t.device, dtype=compute_dtype),
    )
    if out_dtype is None:
        out_dtype = cotangent.dtype
    return result.to(device=cotangent.device, dtype=out_dtype)


def propagate_state_adjoint_from_last_region_input(
    state_jacobians_t: Sequence[torch.Tensor],
    g_last_input_state: torch.Tensor,
    *,
    num_regions: int,
) -> list[torch.Tensor]:
    if len(state_jacobians_t) != num_regions:
        raise ValueError("state_jacobians_t length mismatch.")
    if num_regions == 0:
        return []
    if num_regions == 1:
        return [g_last_input_state]
    prefix_state_jacobians_t = state_jacobians_t[:-1]
    suffix = compose_suffix_jacobian_t(prefix_state_jacobians_t)
    target_dtype = suffix[0].dtype
    target_device = suffix[0].device
    g_last_input_state = g_last_input_state.to(device=target_device, dtype=target_dtype)
    return [
        apply_jacobian_t(suffix[region_index], g_last_input_state, out_dtype=g_last_input_state.dtype)
        for region_index in range(num_regions)
    ]
