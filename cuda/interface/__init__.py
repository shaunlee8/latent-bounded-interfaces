from __future__ import annotations

import torch

try:
    import interface_scan_cuda  # type: ignore
except Exception:  # pragma: no cover - optional extension
    interface_scan_cuda = None  # type: ignore[assignment]


def _torch_suffix_scan_mats(mats: torch.Tensor) -> torch.Tensor:
    if mats.dim() != 4:
        raise ValueError("mats must have shape [B, K, R, R]")
    bsz, n_regions, rank, rank2 = mats.shape
    if rank != rank2:
        raise ValueError("mats must be square in the last two dimensions")
    if n_regions == 0:
        eye = torch.eye(rank, device=mats.device, dtype=mats.dtype)
        return eye.view(1, 1, rank, rank).expand(bsz, 1, rank, rank).clone()
    suffix_inclusive = mats.clone()
    offset = 1
    while offset < n_regions:
        prev = suffix_inclusive.clone()
        suffix_inclusive[:, :-offset] = torch.matmul(prev[:, :-offset], prev[:, offset:])
        offset *= 2
    eye = torch.eye(rank, device=mats.device, dtype=mats.dtype)
    suffix = torch.empty((bsz, n_regions + 1, rank, rank), device=mats.device, dtype=mats.dtype)
    suffix[:, :n_regions] = suffix_inclusive
    suffix[:, n_regions] = eye
    return suffix


def suffix_scan_pullbacks(mats: torch.Tensor) -> torch.Tensor:
    if interface_scan_cuda is not None and mats.is_cuda:
        return interface_scan_cuda.suffix_scan_mats(mats.contiguous())
    return _torch_suffix_scan_mats(mats)


__all__ = ["suffix_scan_pullbacks"]

