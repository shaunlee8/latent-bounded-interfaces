from __future__ import annotations

from typing import Any

import torch

from backward.base import LBIBackwardResult, ScanBackpropModel
from backward.pullbacks import build_interface_pullback_provider
from backward.scan import ReferenceScanEngine, ScanADEngine, lbi_scan_backward_step


def lbi_reference_scan_backward_step(
    model: ScanBackpropModel,
    *,
    ce_loss: torch.Tensor,
    cache: dict[str, Any],
    state_jacobian_mode: str = "graph",
    state_jacobian_basis_chunk: int = 1,
    compute_interface_jacobian_stats: bool = False,
    include_interface_jacobian_suffix: bool = False,
) -> LBIBackwardResult:
    """Run exact LBI scan backprop with a configured interface-pullback provider."""
    pullback_provider = build_interface_pullback_provider(
        state_jacobian_mode,
        basis_chunk=state_jacobian_basis_chunk,
    )
    return lbi_scan_backward_step(
        model,
        ce_loss=ce_loss,
        cache=cache,
        pullback_provider=pullback_provider,
        compute_interface_jacobian_stats=compute_interface_jacobian_stats,
        include_interface_jacobian_suffix=include_interface_jacobian_suffix,
    )


__all__ = [
    "ReferenceScanEngine",
    "ScanADEngine",
    "lbi_reference_scan_backward_step",
    "lbi_scan_backward_step",
]
