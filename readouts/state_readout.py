from __future__ import annotations

import torch
import torch.nn as nn

from interfaces.attentive import SlotAttentionDecoder
from interfaces.vector_mlp import VectorMLPHead


class StateReadout(nn.Module):
    """Direct boundary-state taps into the readout stream: every region's
    post-update state feeds the loss through a small gated decoder, so messages
    no longer have to survive the relay through later regions to matter."""

    def __init__(
        self,
        *,
        num_regions: int,
        state_width: int,
        feature_dim: int,
        tokenwise: bool,
        attn_dim: int = 64,
    ) -> None:
        super().__init__()
        if num_regions <= 0:
            raise ValueError("num_regions must be > 0")
        if state_width <= 0:
            raise ValueError("state_width must be > 0")
        self.tokenwise = bool(tokenwise)
        if self.tokenwise:
            self.decoders = nn.ModuleList(
                [SlotAttentionDecoder(feature_dim, state_width, attn_dim) for _ in range(num_regions)]
            )
        else:
            self.decoders = nn.ModuleList(
                [VectorMLPHead(state_width, feature_dim) for _ in range(num_regions)]
            )
        # Zero-init gates make the model exactly the tap-free one at init.
        self.gates = nn.Parameter(torch.zeros(num_regions))

    def contributions(
        self,
        state_taps: list[torch.Tensor],
        canvas_features: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Per-region gated readout terms; token-wise decoders return [B, L, D],
        broadcast decoders [B, D]. Each term's only consumer is the readout sum,
        so its cotangent seeds the scan's direct state sources."""
        if len(state_taps) != len(self.decoders):
            raise ValueError("state tap count must match region count")
        terms: list[torch.Tensor] = []
        for index, (tap, decoder) in enumerate(zip(state_taps, self.decoders)):
            gate = self.gates[index].to(dtype=canvas_features.dtype)
            if self.tokenwise:
                decoded = decoder(tap, canvas_features)
            else:
                decoded = decoder(tap)
            terms.append(gate * decoded.to(dtype=canvas_features.dtype))
        return terms

    def forward(
        self,
        state_taps: list[torch.Tensor],
        canvas_features: torch.Tensor,
    ) -> torch.Tensor:
        total = torch.zeros_like(canvas_features)
        for term in self.contributions(state_taps, canvas_features):
            total = total + (term if term.dim() == 3 else term.unsqueeze(1))
        return total
