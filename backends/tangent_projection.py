from __future__ import annotations

import os
from typing import Callable

import torch
import torch.nn.functional as F


def lane_chunked_projection(
    jvp_full: Callable[[torch.Tensor], torch.Tensor],
    region_input_tangent_basis: torch.Tensor,
    *,
    output_projection: torch.Tensor,
    output_inner: torch.Tensor | None,
    tangent_token_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the region JVP a few lanes at a time and keep only each chunk's
    projected contraction, so the full [B, P, L, D] tangent never exists at
    once (chunk width LBI_FWDMODE_PROJ_CHUNK). With a caller-certified zero
    tangent prefix, only the token suffix is projected and the prefix is
    zero-filled."""
    lanes = region_input_tangent_basis.shape[1]
    seq_len = region_input_tangent_basis.shape[2]
    s = int(tangent_token_start)
    s = s if 0 < s < seq_len else 0
    chunk = max(1, int(os.environ.get("LBI_FWDMODE_PROJ_CHUNK", "2")))
    projected_parts, inner_parts = [], []
    for start in range(0, lanes, chunk):
        full = jvp_full(region_input_tangent_basis[:, start:start + chunk])
        part_projected, part_inner = project_region_output_tangent(
            full[:, :, s:] if s else full,
            output_projection=output_projection,
            output_inner=output_inner[:, s:] if s and output_inner is not None else output_inner,
        )
        del full
        if s:
            part_projected = F.pad(part_projected, (0, 0, s, 0))
            if part_inner is not None:
                part_inner = F.pad(part_inner, (s, 0))
        projected_parts.append(part_projected)
        inner_parts.append(part_inner)
    projected = torch.cat(projected_parts, dim=1)
    inner = torch.cat(inner_parts, dim=1) if inner_parts[0] is not None else None
    return projected, inner


def project_region_output_tangent(
    tangent: torch.Tensor,
    *,
    output_projection: torch.Tensor,
    output_inner: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Contract the region-output tangent basis [B, P, L, D] to the projected
    form its interface consumer needs: fixed rows [B, P, L, Rp] plus an
    optional per-token inner product against a [B, L, D] map (the RMS-norm
    chain). The full tangent is not retained past this call; a kernel epilogue
    can later emit these directly without materializing it."""
    if tangent.dim() != 4:
        raise ValueError("tangent must have shape [B, P, L, D].")
    if output_projection.dim() != 2:
        raise ValueError("output_projection must have shape [Rp, D].")
    compute_dtype = torch.promote_types(tangent.dtype, output_projection.dtype)
    t_c = tangent.to(dtype=compute_dtype)
    projected = torch.einsum("bpld,rd->bplr", t_c, output_projection.to(dtype=compute_dtype))
    inner = None
    if output_inner is not None:
        if output_inner.dim() != 3:
            raise ValueError("output_inner must have shape [B, L, D].")
        inner = torch.einsum("bpld,bld->bpl", t_c, output_inner.to(dtype=compute_dtype))
    return projected, inner
