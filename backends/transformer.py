from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
import torch.nn as nn

from backbones.general import BackboneSpec, BackboneStack, build_backbone_stack, init_transformer_module
from backbones.transformer.rope import apply_rope
from backends.base import RegionForwardCache


@dataclass
class TransformerAttentionCache:
    norm_input: torch.Tensor
    norm_output: torch.Tensor
    qkv: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_rope: torch.Tensor
    k_rope: torch.Tensor
    k_expanded: torch.Tensor
    v_expanded: torch.Tensor
    attn_output: torch.Tensor
    out_proj_input: torch.Tensor
    out_proj_output: torch.Tensor


@dataclass
class TransformerMLPCache:
    norm_input: torch.Tensor
    norm_output: torch.Tensor
    fc1_output: torch.Tensor
    up: torch.Tensor
    gate: torch.Tensor
    activation: torch.Tensor
    down_input: torch.Tensor
    down_output: torch.Tensor


@dataclass
class TransformerLayerCache:
    layer_index: int
    hidden_input: torch.Tensor
    residual_input: torch.Tensor | None
    attention: TransformerAttentionCache
    mlp: TransformerMLPCache
    hidden_output: torch.Tensor
    residual_output: torch.Tensor | None


@dataclass
class TransformerRegionCache(RegionForwardCache):
    layer_caches: list[TransformerLayerCache]
    region_input: torch.Tensor
    region_output: torch.Tensor


class TransformerLowering(Protocol):
    """Implementation strategy for Transformer region derivative contracts."""

    name: str

    def input_pullback_basis(
        self,
        *,
        backend: "TransformerRegionBackend",
        cache: TransformerRegionCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def parameter_vjp(
        self,
        *,
        backend: "TransformerRegionBackend",
        cache: TransformerRegionCache,
        output_cotangent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ...


class TransformerRegionBackend(nn.Module):
    """Executes Transformer layer ranges and records per-layer region caches."""

    name = "transformer"

    def __init__(
        self,
        *,
        backbone_spec: BackboneSpec | None = None,
        region_ranges: Sequence[tuple[int, int]],
        backbone: BackboneStack | None = None,
        lowering: TransformerLowering | None = None,
    ) -> None:
        super().__init__()
        if backbone is None:
            if backbone_spec is None:
                raise ValueError("backbone_spec is required when backbone is not provided")
            backbone = build_backbone_stack(backbone_spec)
        self.backbone = backbone
        self.lowering = lowering or TorchAutogradTransformerLowering()
        self.region_ranges = list(region_ranges)
        if not self.region_ranges:
            raise ValueError("transformer region backend requires at least one region")

    def _region_range(self, region_index: int) -> tuple[int, int]:
        try:
            return self.region_ranges[region_index]
        except IndexError as exc:
            raise IndexError(f"region_index {region_index} out of range for {len(self.region_ranges)} regions") from exc

    def _attention_forward_with_cache(self, attn: nn.Module, norm_output: torch.Tensor) -> tuple[torch.Tensor, TransformerAttentionCache]:
        bsz, seqlen, _ = norm_output.shape
        qkv = attn.in_proj(norm_output)
        if attn.d_conv > 0:
            qkv = attn.conv1d(qkv.transpose(1, 2))[..., :seqlen].transpose(1, 2).contiguous()
        q_dim = attn.n_heads * attn.head_dim
        kv_dim = attn.n_kv_heads * attn.head_dim
        q_raw, k_raw, v_raw = qkv.split((q_dim, kv_dim, kv_dim), dim=-1)
        q = q_raw.view(bsz, seqlen, attn.n_heads, attn.head_dim).transpose(1, 2)
        k = k_raw.view(bsz, seqlen, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
        v = v_raw.view(bsz, seqlen, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
        q_rope = apply_rope(q, base=attn.rope_base, interleaved=attn.rope_interleaved)
        k_rope = apply_rope(k, base=attn.rope_base, interleaved=attn.rope_interleaved)
        k_expanded = attn._expand_kv(k_rope)
        v_expanded = attn._expand_kv(v)

        used_flash = False
        if attn._can_use_flash_attn(norm_output):
            try:
                attn_heads = attn._flash_attention(q_rope, k_expanded, v_expanded)
                used_flash = True
            except RuntimeError:
                attn._flash_attn_disabled = True
        if not used_flash:
            scale = attn.softmax_scale if attn.softmax_scale is not None else 1.0 / math.sqrt(attn.head_dim)
            attn_heads = F.scaled_dot_product_attention(
                q_rope,
                k_expanded,
                v_expanded,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=scale,
            ).transpose(1, 2)
        out_proj_input = attn_heads.contiguous().view(bsz, seqlen, attn.out_dim)
        out_proj_output = attn.out_proj(out_proj_input)
        cache = TransformerAttentionCache(
            norm_input=norm_output,
            norm_output=norm_output,
            qkv=qkv,
            q=q,
            k=k,
            v=v,
            q_rope=q_rope,
            k_rope=k_rope,
            k_expanded=k_expanded,
            v_expanded=v_expanded,
            attn_output=attn_heads,
            out_proj_input=out_proj_input,
            out_proj_output=out_proj_output,
        )
        return out_proj_output, cache

    def _mlp_forward_with_cache(self, mlp: nn.Module, norm_output: torch.Tensor) -> tuple[torch.Tensor, TransformerMLPCache]:
        fc1_output = mlp.fc1(norm_output)
        up, gate = fc1_output.chunk(2, dim=-1)
        activation = F.silu(gate)
        down_input = up * activation
        down_output = mlp.fc2(down_input)
        cache = TransformerMLPCache(
            norm_input=norm_output,
            norm_output=norm_output,
            fc1_output=fc1_output,
            up=up,
            gate=gate,
            activation=activation,
            down_input=down_input,
            down_output=down_output,
        )
        return down_output, cache

    def _block_forward_with_cache(
        self,
        *,
        block: nn.Module,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, TransformerAttentionCache, TransformerMLPCache]:
        attn_norm_input = (hidden_states + residual) if residual is not None else hidden_states
        attn_norm_output, residual_after_attn_norm = block._prenorm(hidden_states, residual, block.norm)
        attn_output, attn_cache = self._attention_forward_with_cache(block.attn, attn_norm_output)
        attn_cache.norm_input = attn_norm_input

        mlp_norm_input = attn_output + residual_after_attn_norm
        mlp_norm_output, residual_after_mlp_norm = block._prenorm(attn_output, residual_after_attn_norm, block.norm2)
        mlp_output, mlp_cache = self._mlp_forward_with_cache(block.mlp, mlp_norm_output)
        mlp_cache.norm_input = mlp_norm_input
        return mlp_output, residual_after_mlp_norm, attn_cache, mlp_cache

    def forward_region(
        self,
        *,
        region_input: torch.Tensor,
        region_index: int,
    ) -> tuple[torch.Tensor, TransformerRegionCache]:
        start, end = self._region_range(region_index)
        hidden_states = region_input
        residual: torch.Tensor | None = None
        layer_caches: list[TransformerLayerCache] = []
        for layer_index in range(start, end):
            hidden_input = hidden_states
            residual_input = residual
            hidden_states, residual, attn_cache, mlp_cache = self._block_forward_with_cache(
                block=self.backbone.blocks[layer_index],
                hidden_states=hidden_states,
                residual=residual,
            )
            layer_caches.append(
                TransformerLayerCache(
                    layer_index=layer_index,
                    hidden_input=hidden_input,
                    residual_input=residual_input,
                    attention=attn_cache,
                    mlp=mlp_cache,
                    hidden_output=hidden_states,
                    residual_output=residual,
                )
            )
        region_output = (hidden_states + residual) if residual is not None else hidden_states
        return region_output, TransformerRegionCache(
            region_index=region_index,
            layer_range=(start, end),
            layer_caches=layer_caches,
            region_input=region_input,
            region_output=region_output,
        )

    def parameters_for_region(self, region_index: int) -> list[nn.Parameter]:
        start, end = self._region_range(region_index)
        params: list[nn.Parameter] = []
        for layer_index in range(start, end):
            params.extend([p for p in self.backbone.blocks[layer_index].parameters() if p.requires_grad])
        return params

    def count_parameters(self) -> int:
        return int(sum(p.numel() for block in self.backbone.blocks for p in block.parameters()))

    def initialize_parameters(self, *, backbone_spec: BackboneSpec) -> None:
        init_transformer_module(self.backbone, n_layers=backbone_spec.layers, n_residuals_per_layer=2)

    def input_pullback_basis(
        self,
        *,
        cache: TransformerRegionCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        return self.lowering.input_pullback_basis(
            backend=self,
            cache=cache,
            output_cotangent_basis=output_cotangent_basis,
        )

    def parameter_vjp(
        self,
        *,
        cache: TransformerRegionCache,
        output_cotangent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.lowering.parameter_vjp(
            backend=self,
            cache=cache,
            output_cotangent=output_cotangent,
        )


class TorchAutogradTransformerLowering:
    """Reference Transformer derivative lowering implemented with torch.autograd."""

    name = "torch_autograd"

    def input_pullback_basis(
        self,
        *,
        backend: TransformerRegionBackend,
        cache: TransformerRegionCache,
        output_cotangent_basis: torch.Tensor,
    ) -> torch.Tensor:
        if output_cotangent_basis.dim() != 4:
            raise ValueError("output_cotangent_basis must have shape [B, P, T, D]")
        region_input = cache.region_input.detach().requires_grad_(True)
        region_output, _ = backend.forward_region(region_input=region_input, region_index=cache.region_index)
        basis_first = output_cotangent_basis.to(device=region_output.device, dtype=region_output.dtype).permute(1, 0, 2, 3).contiguous()
        try:
            grad_input = torch.autograd.grad(
                region_output,
                region_input,
                grad_outputs=basis_first,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
                is_grads_batched=True,
            )[0]
            return grad_input.permute(1, 0, 2, 3).contiguous().to(
                device=output_cotangent_basis.device,
                dtype=output_cotangent_basis.dtype,
            )
        except (TypeError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and "doesn't have storage" not in str(exc) and "vmap" not in str(exc):
                raise
        grads: list[torch.Tensor] = []
        for basis_index in range(output_cotangent_basis.shape[1]):
            region_input_i = cache.region_input.detach().requires_grad_(True)
            region_output_i, _ = backend.forward_region(region_input=region_input_i, region_index=cache.region_index)
            grad_i = torch.autograd.grad(
                region_output_i,
                region_input_i,
                grad_outputs=output_cotangent_basis[:, basis_index].to(device=region_output_i.device, dtype=region_output_i.dtype),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            grads.append(grad_i.unsqueeze(1))
        return torch.cat(grads, dim=1).to(device=output_cotangent_basis.device, dtype=output_cotangent_basis.dtype)

    def parameter_vjp(
        self,
        *,
        backend: TransformerRegionBackend,
        cache: TransformerRegionCache,
        output_cotangent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        params = backend.parameters_for_region(cache.region_index)
        region_input = cache.region_input.detach()
        region_output, _ = backend.forward_region(region_input=region_input, region_index=cache.region_index)
        grads = torch.autograd.grad(
            region_output,
            params,
            grad_outputs=output_cotangent.to(device=region_output.device, dtype=region_output.dtype),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        name_by_id = {id(param): name for name, param in backend.named_parameters()}
        out: dict[str, torch.Tensor] = {}
        for param, grad in zip(params, grads):
            if grad is not None:
                out[name_by_id[id(param)]] = grad.detach().clone()
        return out
