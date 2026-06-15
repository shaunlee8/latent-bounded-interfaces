from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab = logits.size(-1)
    return F.cross_entropy(logits.float().reshape(-1, vocab), targets.reshape(-1), reduction="mean")


def resolve_eval_message_ablation_modes(cfg: Any, *, ablation_modes: tuple[str, ...]) -> list[str]:
    if cfg.eval_all_message_ablations:
        return list(ablation_modes)
    if cfg.eval_message_ablation == "none":
        return []
    return [cfg.eval_message_ablation]


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
    """Online/post-hoc CE evaluator for dense backprop_ref runs."""
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
    message_ablation: str = "none",
    ablation_generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Online/post-hoc CE evaluator for LBI runs, with optional message ablations."""
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
                logits, _ = model.forward_with_cache(
                    xb,
                    message_ablation=message_ablation,
                    message_noise_std=cfg.message_noise_std,
                    message_mask_keep_prob=cfg.message_mask_keep_prob,
                    ablation_generator=ablation_generator,
                )
            losses.append(float(next_token_loss(logits, yb).item()))
    return {"ce_loss": float(sum(losses) / max(1, len(losses)))}
