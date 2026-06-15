from __future__ import annotations

from typing import Any

import torch.nn as nn

from backbones.general import BackboneSpec, infer_message_hidden_dim
from interfaces.vector_mlp import VectorMLPInterface
from models.dense_language_model import DenseLanguageModel
from models.lbi_language_model import LBILanguageModel, build_region_ranges
from train.config import LEGACY_DENSE_REGIME, LEGACY_LBI_REGIME, normalize_regime_name


def build_backbone_spec(cfg: Any) -> BackboneSpec:
    layer_types = tuple(t.strip() for t in cfg.layer_types.split(",") if t.strip())
    spec = BackboneSpec(
        name=cfg.backbone,
        dim=cfg.dim,
        layers=cfg.layers,
        d_state=cfg.d_state,
        dt_rank=cfg.dt_rank,
        expand=cfg.expand,
        d_conv=cfg.d_conv,
        headdim=cfg.headdim,
        ngroups=cfg.ngroups,
        chunk_size=cfg.chunk_size,
        use_mem_eff_path=cfg.use_mem_eff_path,
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
        layer_types=layer_types,
    )
    spec.validate()
    return spec


def build_dense_model(cfg: Any, *, backbone_spec: BackboneSpec | None = None) -> DenseLanguageModel:
    if backbone_spec is None:
        backbone_spec = build_backbone_spec(cfg)
    return DenseLanguageModel(
        vocab_size=int(cfg.vocab_size),
        backbone_spec=backbone_spec,
        tie_embeddings=bool(getattr(cfg, "tie_embeddings", False)),
    )


def build_lbi_language_model(cfg: Any, *, backbone_spec: BackboneSpec | None = None) -> LBILanguageModel:
    if backbone_spec is None:
        backbone_spec = build_backbone_spec(cfg)
    region_ranges = build_region_ranges(backbone_spec.layers, int(cfg.region_size))
    interface = VectorMLPInterface(
        feature_dim=backbone_spec.dim,
        num_regions=len(region_ranges),
        interface_width=int(cfg.message_dim),
        interface_map_hidden_dim=infer_message_hidden_dim(backbone_spec, int(cfg.message_hidden_dim)),
        update_scale_init=float(cfg.message_scale_init),
    )
    return LBILanguageModel(
        vocab_size=int(cfg.vocab_size),
        layers_per_region=int(cfg.region_size),
        backbone_spec=backbone_spec,
        interface=interface,
        tie_embeddings=bool(getattr(cfg, "tie_embeddings", False)),
    )


def build_legacy_lbi_model(cfg: Any, *, backbone_spec: BackboneSpec | None = None) -> nn.Module:
    """Build a checkpoint-compatible model for state dictionaries with historical key names."""
    from legacy.native_region_interface import NativeRegionInterfaceModel

    if backbone_spec is None:
        backbone_spec = build_backbone_spec(cfg)
    return NativeRegionInterfaceModel(
        vocab_size=int(cfg.vocab_size),
        region_size=int(cfg.region_size),
        message_dim=int(cfg.message_dim),
        backbone_spec=backbone_spec,
        message_hidden_dim=infer_message_hidden_dim(backbone_spec, int(cfg.message_hidden_dim)),
        message_scale_init=float(cfg.message_scale_init),
        tie_embeddings=bool(getattr(cfg, "tie_embeddings", False)),
    )


def build_lbi_model(cfg: Any, *, backbone_spec: BackboneSpec | None = None) -> LBILanguageModel:
    """Build the LBI language model used by training and evaluation."""
    return build_lbi_language_model(cfg, backbone_spec=backbone_spec)


def checkpoint_uses_owned_lbi(checkpoint: dict[str, Any]) -> bool:
    state_dict = checkpoint.get("model_state_dict", {})
    return any(str(key).startswith("interface.") for key in state_dict.keys())


def checkpoint_uses_legacy_lbi(checkpoint: dict[str, Any]) -> bool:
    state_dict = checkpoint.get("model_state_dict", {})
    has_owned_interface = any(str(key).startswith("interface.") for key in state_dict.keys())
    has_legacy_interface = any(
        str(key).startswith(("input_to_message.", "message_to_hidden.", "hidden_to_message.", "message_norm.", "message_alpha"))
        for key in state_dict.keys()
    )
    return has_legacy_interface and not has_owned_interface


def build_model_for_regime(
    cfg: Any,
    *,
    checkpoint: dict[str, Any] | None = None,
    use_legacy_lbi_checkpoint: bool | None = None,
    use_owned_lbi: bool | None = None,
) -> nn.Module:
    """Build the model variant requested by the training or evaluation configuration."""
    backbone_spec = build_backbone_spec(cfg)
    if normalize_regime_name(cfg.regime) == LEGACY_DENSE_REGIME:
        return build_dense_model(cfg, backbone_spec=backbone_spec)
    if normalize_regime_name(cfg.regime) == LEGACY_LBI_REGIME:
        if use_owned_lbi is not None:
            use_legacy_lbi_checkpoint = not bool(use_owned_lbi)
        elif use_legacy_lbi_checkpoint is None and checkpoint is not None:
            use_legacy_lbi_checkpoint = checkpoint_uses_legacy_lbi(checkpoint)
        if use_legacy_lbi_checkpoint:
            return build_legacy_lbi_model(cfg, backbone_spec=backbone_spec)
        return build_lbi_model(cfg, backbone_spec=backbone_spec)
    raise ValueError(f"unsupported model regime: {cfg.regime}")
