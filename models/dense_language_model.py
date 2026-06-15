from __future__ import annotations

import torch
import torch.nn as nn

from backbones.general import BackboneSpec, _make_final_norm, build_backbone_stack, init_transformer_module
from canvas import TokenEmbeddingCanvas
from readouts import NormLMHeadReadout


class DenseLanguageModel(nn.Module):
    """Dense autoregressive language model built from a canvas, backbone stack, and readout."""

    def __init__(self, *, vocab_size: int, backbone_spec: BackboneSpec, tie_embeddings: bool = False) -> None:
        super().__init__()
        backbone_spec.validate()
        self.vocab_size = int(vocab_size)
        self.backbone_spec = backbone_spec
        self.tie_embeddings = bool(tie_embeddings)
        self.canvas = TokenEmbeddingCanvas(vocab_size=self.vocab_size, feature_dim=backbone_spec.dim)
        self.backbone = build_backbone_stack(backbone_spec)
        self.blocks = self.backbone.blocks
        self.readout = NormLMHeadReadout(
            norm=_make_final_norm(backbone_spec),
            feature_dim=backbone_spec.dim,
            vocab_size=self.vocab_size,
            tie_embeddings=self.tie_embeddings,
        )

        if backbone_spec.name == "transformer":
            init_transformer_module(self.canvas, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
            init_transformer_module(self.backbone, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
            init_transformer_module(self.readout, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
        elif backbone_spec.name == "hybrid":
            init_transformer_module(self.canvas, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
            for layer_type, block in zip(backbone_spec.layer_types, self.blocks):
                if layer_type == "transformer":
                    init_transformer_module(block, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
            init_transformer_module(self.readout, n_layers=backbone_spec.layers, n_residuals_per_layer=2)

        if self.tie_embeddings:
            nn.init.normal_(self.canvas.embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.canvas(input_ids)
        x = self.backbone(x)
        return self.readout(x, canvas=self.canvas)
