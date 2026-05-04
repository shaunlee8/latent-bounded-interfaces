from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn


SUPPORTED_BACKBONES = ("mamba2", "mamba3", "transformer")
_HYBRID_SUPPORTED_LAYER_TYPES = ("mamba2", "mamba3", "transformer")


@dataclass
class BackboneSpec:
    """Shared backbone configuration surface for region-interface experiments."""

    name: str = "transformer"
    dim: int = 64
    layers: int = 4
    d_state: int = 8
    dt_rank: Optional[int] = None

    # Shared Mamba-style knobs.
    expand: int = 2
    d_conv: int = 4
    bias: bool = False
    conv_bias: bool = True

    # Mamba-2 / Mamba-3 style knobs.
    headdim: int = 128
    ngroups: int = 1
    chunk_size: int = 256
    use_mem_eff_path: bool = True

    # Transformer-style knobs.
    n_heads: int = 8
    n_kv_heads: int = 0
    mlp_ratio: float = 4.0
    d_intermediate: int = 0
    rope_base: float = 10000.0
    attn_head_dim: int = 0
    softmax_scale: float = 0.0
    rope_interleaved: bool = False
    use_flash_attn: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    layer_types: tuple[str, ...] = ()

    # Reserved for future Mamba-3-specific wiring.
    extra: Optional[dict[str, Any]] = None

    def validate(self) -> None:
        allowed_backbones = (*SUPPORTED_BACKBONES, "hybrid")
        if self.name not in allowed_backbones:
            allowed = ", ".join(allowed_backbones)
            raise ValueError(f"unsupported backbone '{self.name}', expected one of: {allowed}")
        if self.dim <= 0:
            raise ValueError("dim must be > 0")
        if self.layers <= 0:
            raise ValueError("layers must be > 0")
        if self.d_state <= 0:
            raise ValueError("d_state must be > 0")
        if self.expand <= 0:
            raise ValueError("expand must be > 0")
        if self.d_conv <= 0:
            raise ValueError("d_conv must be > 0")
        if self.headdim <= 0:
            raise ValueError("headdim must be > 0")
        if self.ngroups <= 0:
            raise ValueError("ngroups must be > 0")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        active_layer_types = {self.name}
        if self.name == "hybrid":
            if not self.layer_types:
                raise ValueError("hybrid backbone requires non-empty layer_types")
            if len(self.layer_types) != self.layers:
                raise ValueError("for hybrid, len(layer_types) must equal layers")
            unknown = sorted(set(self.layer_types) - set(_HYBRID_SUPPORTED_LAYER_TYPES))
            if unknown:
                allowed = ", ".join(_HYBRID_SUPPORTED_LAYER_TYPES)
                bad = ", ".join(unknown)
                raise ValueError(f"hybrid layer_types must be drawn from: {allowed}; got: {bad}")
            active_layer_types = set(self.layer_types)
        if active_layer_types & {"mamba2", "mamba3"} and (self.expand * self.dim) % self.headdim != 0:
            raise ValueError("for mamba2/mamba3/hybrid-mamba layers, expand * dim must be divisible by headdim")
        if active_layer_types & {"transformer"}:
            if self.n_heads <= 0:
                raise ValueError("transformer requires n_heads > 0")
            if self.n_kv_heads < 0:
                raise ValueError("transformer requires n_kv_heads >= 0")
            kv_heads = self.n_kv_heads or self.n_heads
            if self.attn_head_dim < 0:
                raise ValueError("transformer requires attn_head_dim >= 0")
            if self.attn_head_dim == 0 and self.dim % self.n_heads != 0:
                raise ValueError("for transformer, dim must be divisible by n_heads")
            if self.n_heads % kv_heads != 0:
                raise ValueError("for transformer, n_heads must be divisible by n_kv_heads")
            if self.mlp_ratio < 0.0:
                raise ValueError("transformer requires mlp_ratio >= 0")
            if self.d_intermediate < 0:
                raise ValueError("transformer requires d_intermediate >= 0")
            if self.mlp_ratio == 0.0 and self.d_intermediate == 0:
                pass
            if self.rope_base <= 0.0:
                raise ValueError("transformer requires rope_base > 0")
            if self.softmax_scale < 0.0:
                raise ValueError("transformer requires softmax_scale >= 0")


class BackboneStack(nn.Module):
    """Backbone stack with full-stack and range execution."""

    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward_range(self, x: torch.Tensor, start: int = 0, end: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_range(x, 0, len(self.blocks))


class LayerwiseBackboneStack(BackboneStack):
    def forward_range(self, x: torch.Tensor, start: int = 0, end: Optional[int] = None) -> torch.Tensor:
        end = len(self.blocks) if end is None else end
        for block in self.blocks[start:end]:
            x = block(x)
        return x


class ResidualBackboneStack(BackboneStack):
    """Stack for blocks that thread upstream-style `(hidden_states, residual)` state."""

    def forward_range(self, x: torch.Tensor, start: int = 0, end: Optional[int] = None) -> torch.Tensor:
        end = len(self.blocks) if end is None else end
        hidden_states = x
        residual = None
        for block in self.blocks[start:end]:
            hidden_states, residual = block(hidden_states, residual=residual)
        return (hidden_states + residual) if residual is not None else hidden_states


def init_transformer_module(module: nn.Module, *, n_layers: int, n_residuals_per_layer: int = 2) -> None:
    def _init_weights(submodule: nn.Module) -> None:
        if isinstance(submodule, nn.Linear):
            if submodule.bias is not None and not getattr(submodule.bias, "_no_reinit", False):
                nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, nn.Embedding):
            nn.init.normal_(submodule.weight, std=0.02)

        for name, param in submodule.named_parameters(recurse=False):
            if name in {"out_proj.weight", "fc2.weight"}:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                with torch.no_grad():
                    param /= math.sqrt(max(1, n_residuals_per_layer * n_layers))

    module.apply(_init_weights)


def build_backbone_stack(spec: BackboneSpec) -> BackboneStack:
    """Build the selected backbone as a reusable stack/segment module."""

    spec.validate()
    if spec.name == "mamba2":
        return _build_mamba2_stack(spec)
    if spec.name == "mamba3":
        return _build_mamba3_stack(spec)
    if spec.name == "transformer":
        return _build_transformer_stack(spec)
    if spec.name == "hybrid":
        return _build_hybrid_stack(spec)
    raise AssertionError("unreachable")


def build_backbone_blocks(spec: BackboneSpec) -> nn.ModuleList:
    """Compatibility helper returning the underlying layer modules for tests/inspection."""

    return build_backbone_stack(spec).blocks


def _make_final_norm(backbone_spec: BackboneSpec) -> nn.Module:
    if backbone_spec.name == "mamba2":
        from backbones.mamba2.ops.triton.layernorm_gated import RMSNorm

        return RMSNorm(backbone_spec.dim, eps=1e-5)
    if backbone_spec.name == "mamba3":
        from backbones.mamba3.ops.triton.layernorm_gated import RMSNorm

        return RMSNorm(backbone_spec.dim, eps=1e-5)
    if backbone_spec.name == "transformer":
        from backbones.transformer import RMSNorm

        return RMSNorm(backbone_spec.dim, eps=1e-5)
    if backbone_spec.name == "hybrid":
        from backbones.transformer import RMSNorm

        return RMSNorm(backbone_spec.dim, eps=1e-5)
    return nn.LayerNorm(backbone_spec.dim)


class ReferenceLM(nn.Module):
    """Shared dense autoregressive reference model built from a backbone stack."""

    def __init__(self, *, vocab_size: int, backbone_spec: BackboneSpec, tie_embeddings: bool = False):
        super().__init__()
        backbone_spec.validate()
        self.vocab_size = vocab_size
        self.backbone_spec = backbone_spec
        self.tie_embeddings = bool(tie_embeddings)
        self.embedding = nn.Embedding(vocab_size, backbone_spec.dim)
        self.backbone = build_backbone_stack(backbone_spec)
        self.blocks = self.backbone.blocks
        self.norm = _make_final_norm(backbone_spec)
        self.lm_head = nn.Linear(backbone_spec.dim, vocab_size, bias=False)
        if backbone_spec.name == "transformer":
            init_transformer_module(self, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
        elif backbone_spec.name == "hybrid":
            init_transformer_module(self.embedding, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
            for layer_type, block in zip(backbone_spec.layer_types, self.blocks):
                if layer_type == "transformer":
                    init_transformer_module(block, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
            init_transformer_module(self.lm_head, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
        if self.tie_embeddings:
            nn.init.normal_(self.embedding.weight, std=0.02)
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.backbone(x)
        return self.lm_head(self.norm(x))


def infer_message_hidden_dim(spec: BackboneSpec, explicit_value: int) -> int:
    if explicit_value < 0:
        raise ValueError("explicit_value must be >= 0")
    if explicit_value > 0:
        return explicit_value
    return spec.dim


def _build_mamba2_stack(spec: BackboneSpec) -> BackboneStack:
    from backbones.mamba2 import Mamba2Block

    blocks = nn.ModuleList(
        [
            Mamba2Block(
                dim=spec.dim,
                d_state=spec.d_state,
                d_conv=spec.d_conv,
                expand=spec.expand,
                headdim=spec.headdim,
                ngroups=spec.ngroups,
                chunk_size=spec.chunk_size,
                use_mem_eff_path=spec.use_mem_eff_path,
                bias=spec.bias,
                conv_bias=spec.conv_bias,
            )
            for _ in range(spec.layers)
        ]
    )
    return ResidualBackboneStack(blocks)


def _build_mamba3_stack(spec: BackboneSpec) -> BackboneStack:
    from backbones.mamba3 import Mamba3Block

    blocks = nn.ModuleList(
        [
            Mamba3Block(
                dim=spec.dim,
                d_state=spec.d_state,
                expand=spec.expand,
                headdim=spec.headdim,
                ngroups=spec.ngroups,
                chunk_size=spec.chunk_size,
            )
            for _ in range(spec.layers)
        ]
    )
    return ResidualBackboneStack(blocks)


def _build_transformer_stack(spec: BackboneSpec) -> BackboneStack:
    from backbones.transformer import TransformerBlock

    kv_heads = spec.n_kv_heads or spec.n_heads
    head_dim = spec.attn_head_dim or (spec.dim // spec.n_heads)
    softmax_scale = spec.softmax_scale if spec.softmax_scale > 0.0 else None
    blocks = nn.ModuleList(
        [
            TransformerBlock(
                dim=spec.dim,
                n_heads=spec.n_heads,
                n_kv_heads=kv_heads,
                head_dim=head_dim,
                d_intermediate=spec.d_intermediate,
                mlp_ratio=spec.mlp_ratio,
                rope_base=spec.rope_base,
                rope_interleaved=spec.rope_interleaved,
                softmax_scale=softmax_scale,
                d_conv=spec.d_conv,
                use_flash_attn=spec.use_flash_attn,
                residual_in_fp32=spec.residual_in_fp32,
                fused_add_norm=spec.fused_add_norm,
                bias=spec.bias,
            )
            for _ in range(spec.layers)
        ]
    )
    return ResidualBackboneStack(blocks)


def _make_hybrid_block(layer_type: str, spec: BackboneSpec) -> nn.Module:
    if layer_type == "mamba2":
        from backbones.mamba2 import Mamba2Block

        return Mamba2Block(
            dim=spec.dim,
            d_state=spec.d_state,
            d_conv=spec.d_conv,
            expand=spec.expand,
            headdim=spec.headdim,
            ngroups=spec.ngroups,
            chunk_size=spec.chunk_size,
            use_mem_eff_path=spec.use_mem_eff_path,
            bias=spec.bias,
            conv_bias=spec.conv_bias,
        )
    if layer_type == "mamba3":
        from backbones.mamba3 import Mamba3Block

        return Mamba3Block(
            dim=spec.dim,
            d_state=spec.d_state,
            expand=spec.expand,
            headdim=spec.headdim,
            ngroups=spec.ngroups,
            chunk_size=spec.chunk_size,
        )
    if layer_type == "transformer":
        from backbones.transformer import TransformerBlock

        kv_heads = spec.n_kv_heads or spec.n_heads
        head_dim = spec.attn_head_dim or (spec.dim // spec.n_heads)
        softmax_scale = spec.softmax_scale if spec.softmax_scale > 0.0 else None
        return TransformerBlock(
            dim=spec.dim,
            n_heads=spec.n_heads,
            n_kv_heads=kv_heads,
            head_dim=head_dim,
            d_intermediate=spec.d_intermediate,
            mlp_ratio=spec.mlp_ratio,
            rope_base=spec.rope_base,
            rope_interleaved=spec.rope_interleaved,
            softmax_scale=softmax_scale,
            d_conv=spec.d_conv,
            use_flash_attn=spec.use_flash_attn,
            residual_in_fp32=spec.residual_in_fp32,
            fused_add_norm=spec.fused_add_norm,
            bias=spec.bias,
        )
    raise ValueError(f"unsupported hybrid layer type: {layer_type}")


def _build_hybrid_stack(spec: BackboneSpec) -> BackboneStack:
    blocks = nn.ModuleList([_make_hybrid_block(layer_type, spec) for layer_type in spec.layer_types])
    return ResidualBackboneStack(blocks)
