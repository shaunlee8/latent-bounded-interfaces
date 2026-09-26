from __future__ import annotations

from typing import Any

import torch.nn as nn

from backbones.general import BackboneSpec, infer_message_hidden_dim
from interfaces.vector_mlp import VectorMLPInterface
from models.dense_language_model import DenseLanguageModel
from models.lbi_language_model import LBILanguageModel, build_region_ranges
from train.config import DENSE_VARIANT, LBI_VARIANT, normalize_model_variant


def build_backbone_spec(cfg: Any) -> BackboneSpec:
    layer_types = None
    if cfg.backbone == "hybrid":
        raw = str(getattr(cfg, "layer_types", "") or "").strip()
        if raw:
            layer_types = tuple(t.strip() for t in raw.split(",") if t.strip())
        else:
            if cfg.layers % 4 != 0:
                raise ValueError(
                    "hybrid default pattern (3x mamba3 + 1x transformer) requires layers "
                    "divisible by 4; set layer_types explicitly otherwise"
                )
            layer_types = ("mamba3", "mamba3", "mamba3", "transformer") * (cfg.layers // 4)
    spec = BackboneSpec(
        name=cfg.backbone,
        layer_types=layer_types,
        dim=cfg.dim,
        layers=cfg.layers,
        d_state=cfg.d_state,
        expand=cfg.expand,
        d_conv=cfg.d_conv,
        headdim=cfg.headdim,
        ngroups=cfg.ngroups,
        chunk_size=cfg.chunk_size,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        mlp_ratio=cfg.mlp_ratio,
        d_intermediate=cfg.d_intermediate,
        rope_base=cfg.rope_base,
        attn_head_dim=cfg.attn_head_dim,
        softmax_scale=cfg.softmax_scale,
        rope_interleaved=cfg.rope_interleaved,
        use_flash_attn=cfg.use_flash_attn,
        residual_in_fp32=cfg.residual_in_fp32,
        fused_add_norm=cfg.fused_add_norm,
    )
    spec.validate()
    return spec


def build_dense_model(cfg: Any, *, backbone_spec: BackboneSpec | None = None) -> DenseLanguageModel:
    if backbone_spec is None:
        backbone_spec = build_backbone_spec(cfg)
    return DenseLanguageModel(
        vocab_size=int(cfg.vocab_size),
        backbone_spec=backbone_spec,
        tie_embeddings=bool(cfg.tie_embeddings),
    )


def build_interface(cfg: Any, *, backbone_spec: BackboneSpec, num_regions: int) -> VectorMLPInterface:
    return VectorMLPInterface(
        feature_dim=backbone_spec.dim,
        num_regions=num_regions,
        interface_width=int(cfg.message_dim),
        interface_map_hidden_dim=infer_message_hidden_dim(backbone_spec, int(cfg.message_hidden_dim)),
        update_scale_init=float(cfg.message_scale_init),
    )


def build_lbi_model(cfg: Any, *, backbone_spec: BackboneSpec | None = None) -> LBILanguageModel:
    """Build the LBI language model used by training and evaluation."""
    if backbone_spec is None:
        backbone_spec = build_backbone_spec(cfg)
    region_ranges = build_region_ranges(backbone_spec.layers, int(cfg.region_size))
    interface = build_interface(cfg, backbone_spec=backbone_spec, num_regions=len(region_ranges))
    return LBILanguageModel(
        vocab_size=int(cfg.vocab_size),
        layers_per_region=int(cfg.region_size),
        backbone_spec=backbone_spec,
        interface=interface,
        tie_embeddings=bool(cfg.tie_embeddings),
    )


def build_model_for_regime(cfg: Any) -> nn.Module:
    """Build the dense or LBI model named by `cfg.regime`."""
    variant = normalize_model_variant(cfg.regime)
    if variant == DENSE_VARIANT:
        return build_dense_model(cfg)
    if variant == LBI_VARIANT:
        return build_lbi_model(cfg)
    raise ValueError(f"unsupported model regime: {cfg.regime}")
