from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn


SUPPORTED_BACKBONES = ("mamba1", "mamba2", "mamba3")


@dataclass
class BackboneSpec:
    """Shared backbone configuration surface for region-interface experiments."""

    name: str = "mamba1"
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

    # Reserved for future Mamba-3-specific wiring.
    extra: Optional[dict[str, Any]] = None

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
        if self.name in {"mamba2", "mamba3"} and (self.expand * self.dim) % self.headdim != 0:
            raise ValueError("for mamba2/mamba3, expand * dim must be divisible by headdim")


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


class Mamba2BackboneStack(BackboneStack):
    def forward_range(self, x: torch.Tensor, start: int = 0, end: Optional[int] = None) -> torch.Tensor:
        end = len(self.blocks) if end is None else end
        hidden_states = x
        residual = None
        for block in self.blocks[start:end]:
            hidden_states, residual = block(hidden_states, residual=residual)
        return (hidden_states + residual) if residual is not None else hidden_states


def build_backbone_stack(spec: BackboneSpec) -> BackboneStack:
    """Build the selected backbone as a reusable stack/segment module."""

    spec.validate()
    if spec.name == "mamba1":
        return _build_mamba1_stack(spec)
    if spec.name == "mamba2":
        return _build_mamba2_stack(spec)
    if spec.name == "mamba3":
        return _build_mamba3_stack(spec)
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
    return nn.LayerNorm(backbone_spec.dim)


class ReferenceLM(nn.Module):
    """Shared dense autoregressive reference model built from a backbone stack."""

    def __init__(self, *, vocab_size: int, backbone_spec: BackboneSpec):
        super().__init__()
        backbone_spec.validate()
        self.vocab_size = vocab_size
        self.backbone_spec = backbone_spec
        self.embedding = nn.Embedding(vocab_size, backbone_spec.dim)
        self.backbone = build_backbone_stack(backbone_spec)
        self.blocks = self.backbone.blocks
        self.norm = _make_final_norm(backbone_spec)
        self.lm_head = nn.Linear(backbone_spec.dim, vocab_size, bias=False)

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


def _build_mamba1_stack(spec: BackboneSpec) -> BackboneStack:
    from backbones.mamba1.block import MambaResidualBlock
    from backbones.mamba1.config import BlockConfig
    from backbones.mamba1.mamba import make_backend as make_mamba_backend

    dt_rank_cfg = spec.dt_rank if spec.dt_rank is not None else "auto"
    block_cfg = BlockConfig(
        dim=spec.dim,
        d_state=spec.d_state,
        dt_rank=dt_rank_cfg,
        expand=spec.expand,
        d_conv=spec.d_conv,
        bias=spec.bias,
        conv_bias=spec.conv_bias,
        bidirectional=False,
        use_depth_scan=False,
    )
    blocks = nn.ModuleList(
        [MambaResidualBlock(block_cfg, make_mamba_backend(block_cfg)) for _ in range(spec.layers)]
    )
    return LayerwiseBackboneStack(blocks)


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
    return Mamba2BackboneStack(blocks)


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
    return Mamba2BackboneStack(blocks)
