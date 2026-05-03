from __future__ import annotations

from pathlib import Path

import torch

"""Byte-level text data helpers for legacy and smoke-test data modes.

Canonical paper runs use FineWeb-Edu token shards via data.token_shards.
"""


def build_byte_stream(path: str | Path, *, device: torch.device | None = None) -> torch.Tensor:
    """Load a text file and return a 1D uint8 token stream as int64."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"text file not found: {p}")
    raw = p.read_bytes()
    if len(raw) == 0:
        raise ValueError(f"text file is empty: {p}")
    dev = device if device is not None else torch.device("cpu")
    stream = torch.tensor(list(raw), dtype=torch.long, device=dev)
    return stream


def split_stream_train_val(stream: torch.Tensor, *, val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    if stream.dim() != 1:
        raise ValueError("stream must be 1D")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    n = int(stream.numel())
    n_val = max(1, int(n * val_fraction))
    if n - n_val < 2:
        raise ValueError("stream too small after split")
    return stream[:-n_val].contiguous(), stream[-n_val:].contiguous()


def sample_batch_stream(
    stream: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random contiguous windows from a 1D token stream."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if stream.dim() != 1:
        raise ValueError("stream must be 1D")
    n = int(stream.numel())
    needed = seq_len + 1
    if n <= needed:
        raise ValueError(f"stream length {n} must be > seq_len + 1 ({needed})")

    max_start = n - needed
    starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator, device=stream.device)
    offsets = torch.arange(needed, device=stream.device).unsqueeze(0)
    seq = stream[starts.unsqueeze(1) + offsets]
    x = seq[:, :-1].to(device=device, non_blocking=True)
    y = seq[:, 1:].to(device=device, non_blocking=True)
    return x, y
