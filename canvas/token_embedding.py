from __future__ import annotations

import torch
import torch.nn as nn

from .base import CanvasCache


class TokenEmbeddingCanvas(nn.Module):
    """Maps token IDs to static canvas features for every region.

    ``output_weight`` exposes the embedding matrix for tied-output readouts.
    """

    def __init__(self, *, vocab_size: int, feature_dim: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.feature_dim = int(feature_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.feature_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward_with_cache(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, CanvasCache]:
        features = self.forward(input_ids)
        return features, CanvasCache(input_ids=input_ids, features=features)

    def output_weight(self) -> torch.Tensor:
        return self.embedding.weight

    def vjp_parameters(self) -> list[nn.Parameter]:
        return [self.embedding.weight] if self.embedding.weight.requires_grad else []
