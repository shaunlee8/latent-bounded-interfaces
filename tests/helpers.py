"""Shared test metrics: cosine and max-relative error between two tensors."""

from __future__ import annotations

import torch


def cos_rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a, b = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()
    return cos, rel


def worst_cos_rel(a_list, b_list) -> tuple[float, float]:
    """Minimum cosine and maximum relative error over paired tensors."""
    worst_cos, worst_rel = 1.0, 0.0
    for a, b in zip(a_list, b_list):
        cos, rel = cos_rel(a, b)
        worst_cos, worst_rel = min(worst_cos, cos), max(worst_rel, rel)
    return worst_cos, worst_rel
