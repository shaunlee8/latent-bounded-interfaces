# Portions adapted from state-spaces/mamba (https://github.com/state-spaces/mamba),
# licensed under Apache-2.0. Modified for the bounded-interface/LBI experiments.
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GatedMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = False,
        multiple_of: int = 128,
    ) -> None:
        super().__init__()
        out_features = dim if out_features is None else out_features
        hidden_features = hidden_features if hidden_features is not None else int(8 * dim / 3)
        hidden_features = max(multiple_of, ((hidden_features + multiple_of - 1) // multiple_of) * multiple_of)
        self.fc1 = nn.Linear(dim, 2 * hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(y * F.silu(gate))


class SwiGLUMLP(GatedMLP):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, bias: bool = False) -> None:
        hidden_dim = max(1, int(dim * mlp_ratio))
        super().__init__(dim, hidden_features=hidden_dim, out_features=dim, bias=bias, multiple_of=1)
