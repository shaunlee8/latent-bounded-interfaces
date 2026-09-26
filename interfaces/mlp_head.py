"""The MLP map used by the vector interface's initial encoder, decoders, and encoders."""

from __future__ import annotations

import torch
import torch.nn as nn


class VectorMLPHead(nn.Module):
    """Linear map, or a two-layer SiLU MLP when `hidden_dim` > 0."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        param = next(self.net.parameters(), None)
        if param is not None and x.dtype != param.dtype:
            x = x.to(dtype=param.dtype)
        return self.net(x)
