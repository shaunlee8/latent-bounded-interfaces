from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


SUPPORTED_BACKBONES = ("mamba3", "transformer", "hybrid")


@dataclass
class BackboneSpec:
    """Shared backbone configuration surface for region-interface experiments."""

    name: str = "transformer"
    dim: int = 64
    layers: int = 4
    d_state: int = 8

    # Shared Mamba-style knobs.
    expand: int = 2
    d_conv: int = 4
    bias: bool = False

    # Mamba-3 knobs.
    headdim: int = 128
    ngroups: int = 1
    chunk_size: int = 256

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

    # Hybrid stacks: per-layer block types ("mamba3" / "transformer"),
    # length == layers. Required for name == "hybrid"; ignored otherwise.
    layer_types: Optional[tuple[str, ...]] = None

    def validate(self) -> None:
        if self.name not in SUPPORTED_BACKBONES:
            allowed = ", ".join(SUPPORTED_BACKBONES)
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
        if self.name == "hybrid":
            if not self.layer_types:
                raise ValueError("hybrid requires layer_types (one entry per layer)")
            if len(self.layer_types) != self.layers:
                raise ValueError("hybrid layer_types length must equal layers")
            bad = set(self.layer_types) - {"mamba3", "transformer"}
            if bad:
                raise ValueError(f"hybrid layer_types must be mamba3/transformer, got {sorted(bad)}")
        if self.name in ("mamba3", "hybrid") and (self.expand * self.dim) % self.headdim != 0:
            raise ValueError("for mamba3, expand * dim must be divisible by headdim")
        if self.name in ("transformer", "hybrid"):
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
    if spec.name == "mamba3":
        return _build_mamba3_stack(spec)
    if spec.name == "transformer":
        return _build_transformer_stack(spec)
    if spec.name == "hybrid":
        return _build_hybrid_stack(spec)
    raise AssertionError("unreachable")


def _make_final_norm(backbone_spec: BackboneSpec) -> nn.Module:
    if backbone_spec.name == "mamba3":
        from backbones.mamba3.ops.triton.layernorm_gated import RMSNorm

        return RMSNorm(backbone_spec.dim, eps=1e-5)
    if backbone_spec.name in ("transformer", "hybrid"):
        from backbones.transformer import RMSNorm

        return RMSNorm(backbone_spec.dim, eps=1e-5)
    raise AssertionError("unreachable")


def infer_message_hidden_dim(spec: BackboneSpec, explicit_value: int) -> int:
    if explicit_value < 0:
        raise ValueError("explicit_value must be >= 0")
    if explicit_value > 0:
        return explicit_value
    return spec.dim


def _make_mamba3_block(spec: BackboneSpec) -> nn.Module:
    from backbones.mamba3 import Mamba3Block

    return Mamba3Block(
        dim=spec.dim,
        d_state=spec.d_state,
        expand=spec.expand,
        headdim=spec.headdim,
        ngroups=spec.ngroups,
        chunk_size=spec.chunk_size,
    )


def _make_transformer_block(spec: BackboneSpec) -> nn.Module:
    from backbones.transformer import TransformerBlock

    kv_heads = spec.n_kv_heads or spec.n_heads
    head_dim = spec.attn_head_dim or (spec.dim // spec.n_heads)
    scale = spec.softmax_scale if spec.softmax_scale > 0.0 else None
    return TransformerBlock(
        dim=spec.dim,
        n_heads=spec.n_heads,
        n_kv_heads=kv_heads,
        head_dim=head_dim,
        d_intermediate=spec.d_intermediate,
        mlp_ratio=spec.mlp_ratio,
        rope_base=spec.rope_base,
        rope_interleaved=spec.rope_interleaved,
        softmax_scale=scale,
        d_conv=spec.d_conv,
        use_flash_attn=spec.use_flash_attn,
        residual_in_fp32=spec.residual_in_fp32,
        fused_add_norm=spec.fused_add_norm,
        bias=spec.bias,
    )


def _build_mamba3_stack(spec: BackboneSpec) -> BackboneStack:
    blocks = nn.ModuleList([_make_mamba3_block(spec) for _ in range(spec.layers)])
    return ResidualBackboneStack(blocks)


def _build_hybrid_stack(spec: BackboneSpec) -> BackboneStack:
    factories = {"mamba3": _make_mamba3_block, "transformer": _make_transformer_block}
    blocks = nn.ModuleList([factories[t](spec) for t in spec.layer_types])
    return ResidualBackboneStack(blocks)


def _build_transformer_stack(spec: BackboneSpec) -> BackboneStack:
    blocks = nn.ModuleList([_make_transformer_block(spec) for _ in range(spec.layers)])
    return ResidualBackboneStack(blocks)
