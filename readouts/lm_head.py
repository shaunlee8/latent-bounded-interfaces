from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from canvas import CanvasModule
from .base import ReadoutCache


class NormLMHeadReadout(nn.Module):
    """Applies final normalization and maps final region features to vocabulary logits.

    With tied embeddings, logits are computed with ``canvas.output_weight()``.
    With untied embeddings, this module registers a separate linear LM head.
    """

    def __init__(
        self,
        *,
        norm: nn.Module,
        feature_dim: int,
        vocab_size: int,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.feature_dim = int(feature_dim)
        self.vocab_size = int(vocab_size)
        self.tie_embeddings = bool(tie_embeddings)
        self.lm_head = None if self.tie_embeddings else nn.Linear(self.feature_dim, self.vocab_size, bias=False)

    def forward(self, features: torch.Tensor, *, canvas: CanvasModule) -> torch.Tensor:
        logits_input = self.norm(features)
        if self.tie_embeddings:
            weight = canvas.output_weight()
            if logits_input.dtype != weight.dtype:
                logits_input = logits_input.to(dtype=weight.dtype)
            return F.linear(logits_input, weight)
        if self.lm_head is None:
            raise RuntimeError("untied readout is missing lm_head")
        if logits_input.dtype != self.lm_head.weight.dtype:
            logits_input = logits_input.to(dtype=self.lm_head.weight.dtype)
        return self.lm_head(logits_input)

    def forward_with_cache(self, features: torch.Tensor, *, canvas: CanvasModule) -> tuple[torch.Tensor, ReadoutCache]:
        normed_features = self.norm(features)
        if self.tie_embeddings:
            weight = canvas.output_weight()
            logits_input = normed_features.to(dtype=weight.dtype) if normed_features.dtype != weight.dtype else normed_features
            logits = F.linear(logits_input, weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("untied readout is missing lm_head")
            logits_input = normed_features.to(dtype=self.lm_head.weight.dtype) if normed_features.dtype != self.lm_head.weight.dtype else normed_features
            logits = self.lm_head(logits_input)
        return logits, ReadoutCache(features=features, normed_features=normed_features, logits=logits)

    def vjp_parameters(self) -> list[nn.Parameter]:
        params = [param for param in self.norm.parameters() if param.requires_grad]
        if self.lm_head is not None:
            params.extend(param for param in self.lm_head.parameters() if param.requires_grad)
        return params

    def output_head_parameter_count(self) -> int:
        if self.lm_head is None:
            return 0
        return int(sum(param.numel() for param in self.lm_head.parameters()))
