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


def propagate_state_adjoint_with_jacobian_scan(
    state_jacobians_t: Sequence[torch.Tensor],
    g_top_state: torch.Tensor,
) -> list[torch.Tensor]:
    count = len(state_jacobians_t)
    suffix = compose_suffix_jacobian_t(state_jacobians_t)
    if len(suffix) != count + 1:
        raise ValueError("suffix length mismatch.")
    target_dtype = suffix[0].dtype
    target_device = suffix[0].device
    g_top_state = g_top_state.to(device=target_device, dtype=target_dtype)
    g_states: list[torch.Tensor] = []
    for region_index in range(count):
        g_states.append(apply_jacobian_t(suffix[region_index], g_top_state, out_dtype=g_top_state.dtype))
    g_states.append(g_top_state)
    return g_states


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


def propagate_state_adjoint_affine(
    state_jacobians_t: Sequence[torch.Tensor],
    g_last_input_state: torch.Tensor,
    state_sources: "Sequence[torch.Tensor] | None",
    *,
    num_regions: int,
    input_sources: "Sequence[torch.Tensor] | None" = None,
) -> list[torch.Tensor]:
    """Affine state-adjoint recursion for models whose boundary states also
    feed the loss directly: lambda_k = A_k^T lambda_{k+1} + b_k.
    `state_sources[k]` is the direct adjoint of region k's OUTPUT state
    (boundary-state readout taps); `input_sources[k]` is the direct adjoint of
    region k's INPUT state (the output-readout path decode^T J^T g). The
    readout-path seed attaches at the last region's input. Returns
    [lambda_0 .. lambda_K] (length num_regions + 1; the final entry seeds the
    last region's update path)."""
    if len(state_jacobians_t) != num_regions:
        raise ValueError("state_jacobians_t length mismatch.")
    if state_sources is not None and len(state_sources) != num_regions:
        raise ValueError("state_sources length mismatch.")
    if input_sources is not None and len(input_sources) != num_regions:
        raise ValueError("input_sources length mismatch.")
    if num_regions == 0:
        return []
    target_dtype = state_jacobians_t[0].dtype
    target_device = state_jacobians_t[0].device

    def _cast(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=target_device, dtype=target_dtype)

    adjoints: list[torch.Tensor | None] = [None] * (num_regions + 1)
    top = (
        _cast(state_sources[num_regions - 1])
        if state_sources is not None
        else torch.zeros_like(_cast(g_last_input_state))
    )
    adjoints[num_regions] = top
    for region_index in reversed(range(num_regions)):
        lam = apply_jacobian_t(
            state_jacobians_t[region_index],
            adjoints[region_index + 1],
            out_dtype=target_dtype,
        )
        if state_sources is not None and region_index >= 1:
            lam = lam + _cast(state_sources[region_index - 1])
        if input_sources is not None:
            lam = lam + _cast(input_sources[region_index])
        if region_index == num_regions - 1:
            lam = lam + _cast(g_last_input_state)
        adjoints[region_index] = lam
    return adjoints  # type: ignore[return-value]


def propagate_state_adjoint_by_autograd_chain(
    *,
    states: Sequence[torch.Tensor],
    top_state_adjoint: torch.Tensor,
) -> list[torch.Tensor]:
    """Diagnostic state-adjoint propagation through an existing autograd graph."""
    if len(states) < 2:
        raise ValueError("at least two interface states are required.")
    g_states: list[torch.Tensor] = [torch.zeros_like(states[0]) for _ in range(len(states))]
    g_states[-1] = top_state_adjoint
    for region_index in reversed(range(len(states) - 1)):
        g_prev = torch.autograd.grad(
            states[region_index + 1],
            states[region_index],
            grad_outputs=g_states[region_index + 1],
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        g_states[region_index] = g_prev
    return g_states


def interface_state_jacobian_t_stats(
    state_jacobians_t: Sequence[torch.Tensor],
    *,
    include_suffix: bool,
) -> dict[str, float]:
    if not state_jacobians_t:
        return {}
    with torch.no_grad():
        dense = torch.stack([jacobian_t.detach().float() for jacobian_t in state_jacobians_t], dim=1)
        frob = torch.linalg.matrix_norm(dense, ord="fro", dim=(-2, -1))
        spec = torch.linalg.matrix_norm(dense, ord=2, dim=(-2, -1))
        rank = max(1, int(dense.shape[-1]))
        stats = {
            "jac_local_spec_mean": float(spec.mean().item()),
            "jac_local_spec_max": float(spec.max().item()),
            "jac_local_frob_mean": float(frob.mean().item()),
            "jac_local_frob_max": float(frob.max().item()),
            "jac_local_frob_normed_mean": float((frob / (rank ** 0.5)).mean().item()),
        }
        if include_suffix:
            suffix = compose_suffix_jacobian_t(state_jacobians_t)
            suffix_dense = torch.stack([jacobian_t.detach().float() for jacobian_t in suffix], dim=1)
            suffix_spec = torch.linalg.matrix_norm(suffix_dense, ord=2, dim=(-2, -1))
            stats.update(
                {
                    "jac_suffix_spec_mean": float(suffix_spec.mean().item()),
                    "jac_suffix_spec_max": float(suffix_spec.max().item()),
                }
            )
        return stats
