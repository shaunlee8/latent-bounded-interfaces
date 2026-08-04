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
    # Pre-conv projection, the operating point of the forward-mode tangent
    # synthesis (qkv above is post-conv when d_conv > 0).
    qkv_preconv: torch.Tensor | None = None
    # Flash softmax stats consumed by the native backward's direct call to
    # the flash backward op.
    attn_out_bhld: torch.Tensor | None = None
    attn_logsumexp: torch.Tensor | None = None
    attn_philox_seed: torch.Tensor | None = None
    attn_philox_offset: torch.Tensor | None = None


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


def softmax_scale(attn):
    """The attention module's softmax scale (1/sqrt(hd) unless overridden)."""
    return attn.softmax_scale if attn.softmax_scale is not None else 1.0 / math.sqrt(attn.head_dim)


def _attention_stage(x, in_w, conv_w, out_w, n_heads, n_kv_heads, head_dim,
                     d_conv, rope_base, interleaved, scale):
    """The attention cache tensors as one pure function of (input, weights);
    weights ride as arguments so a single compiled graph serves every layer."""
    bsz, seqlen, _ = x.shape
    qkv_pre = F.linear(x, in_w)
    if d_conv > 0:
        qkv = F.conv1d(qkv_pre.transpose(1, 2), conv_w, None, padding=d_conv - 1,
                       groups=conv_w.shape[0])[..., :seqlen].transpose(1, 2).contiguous()
    else:
        qkv = qkv_pre
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    q_raw, k_raw, v_raw = qkv.split((q_dim, kv_dim, kv_dim), dim=-1)
    q = q_raw.view(bsz, seqlen, n_heads, head_dim).transpose(1, 2)
    k = k_raw.view(bsz, seqlen, n_kv_heads, head_dim).transpose(1, 2)
    v = v_raw.view(bsz, seqlen, n_kv_heads, head_dim).transpose(1, 2)
    q_rope = apply_rope(q, base=rope_base, interleaved=interleaved)
    k_rope = apply_rope(k, base=rope_base, interleaved=interleaved)
    rep = n_heads // n_kv_heads
    k_exp = k_rope if rep == 1 else k_rope.repeat_interleave(rep, dim=1)
    v_exp = v if rep == 1 else v.repeat_interleave(rep, dim=1)
    out_bhld, lse, seed, offset = _flash_with_stats(q_rope, k_exp, v_exp, scale)
    heads = out_bhld.transpose(1, 2)
    opi = heads.contiguous().view(bsz, seqlen, q_dim)
    opo = F.linear(opi, out_w)
    return (qkv_pre, qkv, q, k, v, q_rope, k_rope, k_exp, v_exp, heads,
            opi, opo, out_bhld, lse, seed, offset)


def _flash_with_stats(q, k, v, scale):
    """Flash attention returning (out, logsumexp, philox seed/offset);
    [B,H,L,hd] in and out. The stats feed the flash backward op directly."""
    out = torch.ops.aten._scaled_dot_product_flash_attention(
        q, k, v, 0.0, True, False, scale=scale)
    return out[0], out[1], out[6], out[7]


def _mlp_stage(x, fc1_w, fc2_w):
    fc1_out = F.linear(x, fc1_w)
    up, gate = fc1_out.chunk(2, dim=-1)
    act = F.silu(gate)
    di = up * act
    do = F.linear(di, fc2_w)
    return fc1_out, up, gate, act, di, do


_COMPILED_STAGES: dict = {}


def _compiled(fn):
    c = _COMPILED_STAGES.get(fn)
    if c is None:
        c = torch.compile(fn, dynamic=False)
        _COMPILED_STAGES[fn] = c
    return c


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
        if (getattr(self, "compile_forward_stages", False)
                and attn.in_proj.bias is None and attn.out_proj.bias is None
                and (attn.d_conv <= 0 or attn.conv1d.bias is None)):
            conv_w = attn.conv1d.weight if attn.d_conv > 0 else attn.in_proj.weight
            (qkv_preconv, qkv, q, k, v, q_rope, k_rope, k_expanded, v_expanded,
             attn_heads, out_proj_input, out_proj_output, out_bhld, lse, seed,
             offset) = _compiled(_attention_stage)(
                norm_output, attn.in_proj.weight, conv_w, attn.out_proj.weight,
                attn.n_heads, attn.n_kv_heads, attn.head_dim, attn.d_conv,
                attn.rope_base, attn.rope_interleaved, softmax_scale(attn))
        else:
            bsz, seqlen, _ = norm_output.shape
            qkv = attn.in_proj(norm_output)
            qkv_preconv = qkv
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
            out_bhld = lse = seed = offset = None
            if attn._can_use_flash_attn(norm_output):
                try:
                    attn_heads = attn._flash_attention(q_rope, k_expanded, v_expanded)
                    used_flash = True
                except RuntimeError:
                    attn._flash_attn_disabled = True
            if not used_flash:
                try:
                    out_bhld, lse, seed, offset = _flash_with_stats(
                        q_rope, k_expanded, v_expanded, softmax_scale(attn))
                    attn_heads = out_bhld.transpose(1, 2)
                except RuntimeError:
                    attn_heads = F.scaled_dot_product_attention(
                        q_rope, k_expanded, v_expanded, attn_mask=None,
                        dropout_p=0.0, is_causal=True, scale=softmax_scale(attn),
                    ).transpose(1, 2)
            out_proj_input = attn_heads.contiguous().view(bsz, seqlen, attn.out_dim)
            out_proj_output = attn.out_proj(out_proj_input)
        cache = TransformerAttentionCache(
            norm_input=norm_output, norm_output=norm_output, qkv=qkv,
            qkv_preconv=qkv_preconv, q=q, k=k, v=v, q_rope=q_rope,
            k_rope=k_rope, k_expanded=k_expanded, v_expanded=v_expanded,
            attn_output=attn_heads, out_proj_input=out_proj_input,
            out_proj_output=out_proj_output, attn_out_bhld=out_bhld,
            attn_logsumexp=lse, attn_philox_seed=seed,
            attn_philox_offset=offset)
        return out_proj_output, cache

    def _mlp_forward_with_cache(self, mlp: nn.Module, norm_output: torch.Tensor) -> tuple[torch.Tensor, TransformerMLPCache]:
        if (getattr(self, "compile_forward_stages", False)
                and mlp.fc1.bias is None and mlp.fc2.bias is None):
            fc1_output, up, gate, activation, down_input, down_output = \
                _compiled(_mlp_stage)(norm_output, mlp.fc1.weight, mlp.fc2.weight)
        else:
            fc1_output = mlp.fc1(norm_output)
            up, gate = fc1_output.chunk(2, dim=-1)
            activation = F.silu(gate)
            down_input = up * activation
            down_output = mlp.fc2(down_input)
        cache = TransformerMLPCache(
            norm_input=norm_output, norm_output=norm_output,
            fc1_output=fc1_output, up=up, gate=gate, activation=activation,
            down_input=down_input, down_output=down_output)
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

    def parameter_vjp_with_input_cotangent(
        self,
        *,
        cache: TransformerRegionCache,
        output_cotangent: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Parameter grads and the region-input cotangent from one reverse
        walk over the cached activations, without replaying the region
        forward. Falls back to the autograd lowering outside the native
        contract or when LBI_TRANSFORMER_NATIVE_VJP=0."""
        import os

        from backends.transformer_native_vjp import (
            native_vjp_supported, transformer_region_parameter_vjp)

        if (os.environ.get("LBI_TRANSFORMER_NATIVE_VJP", "1") != "0"
                and native_vjp_supported(self, cache)):
            return transformer_region_parameter_vjp(self, cache, output_cotangent)
        grads = self.lowering.parameter_vjp(
            backend=self, cache=cache, output_cotangent=output_cotangent)
        g_in = self.lowering.input_pullback_basis(
            backend=self, cache=cache,
            output_cotangent_basis=output_cotangent.unsqueeze(1)).squeeze(1)
        return grads, g_in

    def region_output_jvp(
        self,
        *,
        cache: TransformerRegionCache,
        region_input_tangent_basis: torch.Tensor,
        compute_dtype: torch.dtype | None = None,
        pooled: bool = False,
    ) -> torch.Tensor:
        """Push a region-input tangent basis [B, P, L, D] to the region-output
        tangent basis at the frozen operating point (the dual of
        `input_pullback_basis`). `forward_mode_use_kernel` selects the fused
        kernels over the torch.func reference; `pooled` returns the mean over
        L with keepdim; `compute_dtype=None` resolves to bf16 on the kernel
        path (LBI_FWDMODE_CD=float32 escapes) and fp32 on the reference."""
        if getattr(self, "forward_mode_use_kernel", False):
            from backends.transformer_forward_mode import (
                transformer_region_output_jvp_kernel,
            )

            if compute_dtype is None:
                import os

                compute_dtype = (torch.float32
                                 if os.environ.get("LBI_FWDMODE_CD") == "float32"
                                 else torch.bfloat16)
            return transformer_region_output_jvp_kernel(
                self,
                cache=cache,
                region_input_tangent_basis=region_input_tangent_basis,
                compute_dtype=compute_dtype,
                pooled=pooled,
            )
        from backends.transformer_forward_mode import transformer_region_output_jvp

        out = transformer_region_output_jvp(
            self,
            cache=cache,
            region_input_tangent_basis=region_input_tangent_basis,
            compute_dtype=compute_dtype if compute_dtype is not None else torch.float32,
        )
        return out.mean(dim=2, keepdim=True) if pooled else out


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
