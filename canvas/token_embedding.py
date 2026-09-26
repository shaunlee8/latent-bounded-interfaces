from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbeddingCanvas(nn.Module):
    """Maps token IDs to the canvas features every region reads. `output_weight`
    exposes the embedding matrix for the tied readout."""

    def __init__(self, *, vocab_size: int, feature_dim: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.feature_dim = int(feature_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.feature_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def output_weight(self) -> torch.Tensor:
        return self.embedding.weight

    def vjp_parameters(self) -> list[nn.Parameter]:
        return [self.embedding.weight] if self.embedding.weight.requires_grad else []
