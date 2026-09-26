from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over every position."""
    vocab = logits.size(-1)
    return F.cross_entropy(logits.float().reshape(-1, vocab), targets.reshape(-1), reduction="mean")


def evaluate_reference(
    *,
    cfg: Any,
    model: nn.Module,
    val_corpus: Any,
    eval_batches: int,
    generator: torch.Generator,
    device: torch.device,
    sample_batch_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    autocast_context_fn: Callable[[Any, torch.device], AbstractContextManager[Any]],
) -> dict[str, float]:
    """Validation cross-entropy of the dense model."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(eval_batches):
            xb, yb = sample_batch_fn(
                cfg=cfg,
                corpus=val_corpus,
                batch_size=cfg.batch_size,
                generator=generator,
                device=device,
            )
            with autocast_context_fn(cfg, device):
                logits = model(xb)
            losses.append(float(next_token_loss(logits, yb).item()))
    return {"ce_loss": float(sum(losses) / max(1, len(losses)))}


def evaluate_native(
    *,
    cfg: Any,
    model: nn.Module,
    val_corpus: Any,
    eval_batches: int,
    generator: torch.Generator,
    device: torch.device,
    sample_batch_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    autocast_context_fn: Callable[[Any, torch.device], AbstractContextManager[Any]],
) -> dict[str, float]:
    """Validation cross-entropy of the LBI model."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(eval_batches):
            xb, yb = sample_batch_fn(
                cfg=cfg,
                corpus=val_corpus,
                batch_size=cfg.batch_size,
                generator=generator,
                device=device,
            )
            with autocast_context_fn(cfg, device):
                logits, _ = model.forward_with_cache(xb)
            losses.append(float(next_token_loss(logits, yb).item()))
    return {"ce_loss": float(sum(losses) / max(1, len(losses)))}
