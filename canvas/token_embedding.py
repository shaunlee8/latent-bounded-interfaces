from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CanvasCache


class TokenEmbeddingCanvas(nn.Module):
    """Maps token IDs to static canvas features for every region.

    ``output_weight`` exposes the embedding matrix for tied-output readouts.
    ``local_mixer_kernel`` > 0 adds a residual depthwise causal convolution so
    all regions share locally mixed features instead of re-deriving them.
    """

    def __init__(self, *, vocab_size: int, feature_dim: int, local_mixer_kernel: int = 0) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.feature_dim = int(feature_dim)
        self.local_mixer_kernel = int(local_mixer_kernel)
        self.embedding = nn.Embedding(self.vocab_size, self.feature_dim)
        if self.local_mixer_kernel > 0:
            self.local_mixer = nn.Conv1d(
                self.feature_dim,
                self.feature_dim,
                kernel_size=self.local_mixer_kernel,
                groups=self.feature_dim,
            )
        else:
            self.local_mixer = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = self.embedding(input_ids)
        if self.local_mixer is not None:
            x = features.transpose(1, 2)
            x = F.pad(x, (self.local_mixer_kernel - 1, 0))
            weight = self.local_mixer.weight
            if x.dtype != weight.dtype:
                x = x.to(dtype=weight.dtype)
            mixed = F.silu(self.local_mixer(x)).transpose(1, 2)
            features = features + mixed.to(dtype=features.dtype)
        return features

    def forward_with_cache(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, CanvasCache]:
        features = self.forward(input_ids)
        return features, CanvasCache(input_ids=input_ids, features=features)

    def output_weight(self) -> torch.Tensor:
        return self.embedding.weight

    def vjp_parameters(self) -> list[nn.Parameter]:
        params = [self.embedding.weight] if self.embedding.weight.requires_grad else []
        if self.local_mixer is not None:
            params.extend(p for p in self.local_mixer.parameters() if p.requires_grad)
        return params
