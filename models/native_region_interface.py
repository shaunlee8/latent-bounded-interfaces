from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from cuda.interface import suffix_scan_pullbacks

from backbones.general import BackboneSpec, build_backbone_stack, _make_final_norm, init_transformer_module

_INTERFACE_JACOBIAN_NOTICE = {"batched_vjp_fallback": False}
_VALID_MESSAGE_ABLATIONS = {"none", "zero_all", "noise", "mask"}


def _report_interface_jacobian_notice(kind: str, message: str) -> None:
    if _INTERFACE_JACOBIAN_NOTICE.get(kind, False):
        return
    _INTERFACE_JACOBIAN_NOTICE[kind] = True
    print(message, flush=True)


def build_region_slices(layers: int, region_size: int) -> List[Tuple[int, int]]:
    if layers <= 0:
        raise ValueError("layers must be > 0")
    if region_size <= 0:
        raise ValueError("region_size must be > 0")
    slices: List[Tuple[int, int]] = []
    start = 0
    while start < layers:
        end = min(layers, start + region_size)
        slices.append((start, end))
        start = end
    return slices


class RegionMessageHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0):
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


@dataclass
class RegionForwardCache:
    region_idx: int
    region_range: Tuple[int, int]
    x_static: torch.Tensor
    message_in: torch.Tensor
    hidden_bias: torch.Tensor
    hidden_input: torch.Tensor
    hidden_output: torch.Tensor
    pooled_hidden: torch.Tensor
    delta_message: torch.Tensor
    alpha: torch.Tensor
    pre_norm_message: torch.Tensor
    message_out: torch.Tensor


@dataclass
class RegionInterfaceCache:
    boundaries: List[torch.Tensor]
    region_hidden_inputs: List[torch.Tensor]
    region_messages: List[torch.Tensor]
    region_ranges: List[Tuple[int, int]]
    region_caches: List[RegionForwardCache]


class NativeRegionInterfaceModel(nn.Module):
    """Regionized dense Mamba with a bounded residual message interface between regions."""

    def __init__(
        self,
        *,
        vocab_size: int,
        region_size: int,
        message_dim: int,
        backbone_spec: BackboneSpec,
        message_hidden_dim: int = 0,
        message_scale_init: float = 0.5,
    ):
        super().__init__()
        backbone_spec.validate()
        self.dim = backbone_spec.dim
        self.layers = backbone_spec.layers
        self.message_dim = message_dim
        self.region_ranges = build_region_slices(layers=backbone_spec.layers, region_size=region_size)
        self.n_regions = len(self.region_ranges)

        self.embedding = nn.Embedding(vocab_size, self.dim)
        self.backbone = build_backbone_stack(backbone_spec)
        self.blocks = self.backbone.blocks
        self.norm = _make_final_norm(backbone_spec)
        self.lm_head = nn.Linear(self.dim, vocab_size, bias=False)

        self.input_to_message = RegionMessageHead(self.dim, message_dim, hidden_dim=message_hidden_dim)
        self.message_to_hidden = nn.ModuleList(
            [RegionMessageHead(message_dim, self.dim, hidden_dim=message_hidden_dim) for _ in range(self.n_regions)]
        )
        self.hidden_to_message = nn.ModuleList(
            [RegionMessageHead(self.dim, message_dim, hidden_dim=message_hidden_dim) for _ in range(self.n_regions)]
        )
        self.message_norm = nn.ModuleList([nn.LayerNorm(message_dim) for _ in range(self.n_regions)])
        self.message_alpha = nn.Parameter(torch.full((self.n_regions,), float(message_scale_init)))
        if backbone_spec.name == "transformer":
            init_transformer_module(self.embedding, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
            init_transformer_module(self.backbone, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
            init_transformer_module(self.lm_head, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
        elif backbone_spec.name == "hybrid":
            init_transformer_module(self.embedding, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
            for layer_type, block in zip(backbone_spec.layer_types, self.blocks):
                if layer_type == "transformer":
                    init_transformer_module(block, n_layers=backbone_spec.layers, n_residuals_per_layer=2)
            init_transformer_module(self.lm_head, n_layers=backbone_spec.layers, n_residuals_per_layer=2)

    def _pool_hidden(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        if pooled.dtype != x.dtype:
            pooled = pooled.to(dtype=x.dtype)
        return pooled

    def _apply_message_ablation(
        self,
        m: torch.Tensor,
        *,
        mode: str,
        next_region_idx: int,
        noise_std: float,
        mask_keep_prob: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if mode not in _VALID_MESSAGE_ABLATIONS:
            raise ValueError(f"unsupported message ablation mode: {mode}")
        if mode == "none":
            return m
        if mode == "zero_all":
            return torch.zeros_like(m)
        if mode == "noise":
            if noise_std < 0.0:
                raise ValueError("message_noise_std must be >= 0")
            if noise_std == 0.0:
                return m
            noise = torch.randn(m.shape, device=m.device, dtype=m.dtype, generator=generator)
            return m + (noise_std * noise)
        if mask_keep_prob <= 0.0 or mask_keep_prob > 1.0:
            raise ValueError("message_mask_keep_prob must be in (0, 1]")
        if mask_keep_prob == 1.0:
            return m
        mask = torch.rand(m.shape, device=m.device, generator=generator) < mask_keep_prob
        return m * mask.to(dtype=m.dtype)

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        *,
        message_ablation: str = "none",
        message_noise_std: float = 1.0,
        message_mask_keep_prob: float = 0.5,
        ablation_generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        x_static = self.embedding(input_ids)
        pooled0 = self._pool_hidden(x_static)
        m = self.input_to_message(pooled0)
        m = self._apply_message_ablation(
            m,
            mode=message_ablation,
            next_region_idx=0,
            noise_std=message_noise_std,
            mask_keep_prob=message_mask_keep_prob,
            generator=ablation_generator,
        )

        boundaries: List[torch.Tensor] = [x_static]
        region_hidden_inputs: List[torch.Tensor] = []
        region_messages: List[torch.Tensor] = [m]
        region_caches: List[RegionForwardCache] = []
        for ridx, (start, end) in enumerate(self.region_ranges):
            m_in = m
            hidden_bias = self.message_to_hidden[ridx](m_in)
            # Strict bounded-interface mode: each region hidden stream is rebuilt
            # from static token embedding + current message, with no dense carry.
            x_in = x_static + hidden_bias.unsqueeze(1)
            region_hidden_inputs.append(x_in)
            x_out = self.backbone.forward_range(x_in, start, end)
            boundaries.append(x_out)

            pooled_hidden = self._pool_hidden(x_out)
            delta_m = self.hidden_to_message[ridx](pooled_hidden)
            alpha = torch.tanh(self.message_alpha[ridx])
            pre_norm_message = m_in + alpha * delta_m
            m = self.message_norm[ridx](pre_norm_message)
            m = self._apply_message_ablation(
                m,
                mode=message_ablation,
                next_region_idx=ridx + 1,
                noise_std=message_noise_std,
                mask_keep_prob=message_mask_keep_prob,
                generator=ablation_generator,
            )
            region_messages.append(m)
            region_caches.append(
                RegionForwardCache(
                    region_idx=ridx,
                    region_range=(start, end),
                    x_static=x_static,
                    message_in=m_in,
                    hidden_bias=hidden_bias,
                    hidden_input=x_in,
                    hidden_output=x_out,
                    pooled_hidden=pooled_hidden,
                    delta_message=delta_m,
                    alpha=alpha,
                    pre_norm_message=pre_norm_message,
                    message_out=m,
                )
            )

        logits_in = self.norm(x_out)
        if logits_in.dtype != self.lm_head.weight.dtype:
            logits_in = logits_in.to(dtype=self.lm_head.weight.dtype)
        logits = self.lm_head(logits_in)
        cache = RegionInterfaceCache(
            boundaries=boundaries,
            region_hidden_inputs=region_hidden_inputs,
            region_messages=region_messages,
            region_ranges=self.region_ranges,
            region_caches=region_caches,
        )
        return logits, {
            "boundaries": cache.boundaries,
            "region_hidden_inputs": cache.region_hidden_inputs,
            "region_messages": cache.region_messages,
            "region_ranges": cache.region_ranges,
            "region_caches": cache.region_caches,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        message_ablation: str = "none",
        message_noise_std: float = 1.0,
        message_mask_keep_prob: float = 0.5,
        ablation_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        logits, _ = self.forward_with_cache(
            input_ids,
            message_ablation=message_ablation,
            message_noise_std=message_noise_std,
            message_mask_keep_prob=message_mask_keep_prob,
            ablation_generator=ablation_generator,
        )
        return logits

    def _module_input_pullback_matrix_autograd(
        self,
        module: nn.Module,
        x: torch.Tensor,
        g_out: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("module input pullback expects x shaped [B, D].")
        if g_out.dim() != 3:
            raise ValueError("module input pullback expects g_out shaped [B, P, D_out].")
        bsz, basis = g_out.shape[:2]
        x_rep = (
            x.detach()
            .unsqueeze(1)
            .expand(bsz, basis, x.shape[-1])
            .reshape(bsz * basis, x.shape[-1])
            .requires_grad_(True)
        )
        y_rep = module(x_rep)
        g_rep = g_out.to(device=y_rep.device, dtype=y_rep.dtype).reshape_as(y_rep)
        g_in = torch.autograd.grad(
            y_rep,
            x_rep,
            grad_outputs=g_rep,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        return g_in.reshape(bsz, basis, x.shape[-1]).to(device=g_out.device, dtype=g_out.dtype)

    def _linear_input_pullback_matrix(self, linear: nn.Linear, g_out: torch.Tensor) -> torch.Tensor:
        if g_out.dim() != 3:
            raise ValueError("linear pullback expects g_out shaped [B, P, D_out].")
        compute_dtype = torch.promote_types(g_out.dtype, linear.weight.dtype)
        g_in = torch.einsum(
            "bpo,oi->bpi",
            g_out.to(device=linear.weight.device, dtype=compute_dtype),
            linear.weight.to(dtype=compute_dtype),
        )
        return g_in.to(device=g_out.device, dtype=g_out.dtype)

    def _silu_input_pullback_matrix(self, x: torch.Tensor, g_out: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("SiLU pullback expects x shaped [B, D].")
        if g_out.dim() != 3:
            raise ValueError("SiLU pullback expects g_out shaped [B, P, D].")
        compute_dtype = torch.promote_types(x.dtype, g_out.dtype)
        x_compute = x.to(device=g_out.device, dtype=compute_dtype)
        g_compute = g_out.to(dtype=compute_dtype)
        sig = torch.sigmoid(x_compute)
        deriv = sig * (1.0 + x_compute * (1.0 - sig))
        return (g_compute * deriv.unsqueeze(1)).to(device=g_out.device, dtype=g_out.dtype)

    def _region_message_head_input_pullback_matrix(
        self,
        head: RegionMessageHead,
        x: torch.Tensor,
        g_out: torch.Tensor,
    ) -> torch.Tensor:
        net = head.net
        if isinstance(net, nn.Linear):
            return self._linear_input_pullback_matrix(net, g_out)
        if isinstance(net, nn.Sequential) and len(net) == 3:
            linear1, act, linear2 = net
            if not isinstance(linear1, nn.Linear) or not isinstance(act, nn.SiLU) or not isinstance(linear2, nn.Linear):
                raise TypeError("unsupported RegionMessageHead sequential structure")
            compute_dtype = torch.promote_types(x.dtype, g_out.dtype)
            hidden_pre = linear1(x.to(device=linear1.weight.device, dtype=compute_dtype))
            g_hidden = self._linear_input_pullback_matrix(linear2, g_out.to(device=linear2.weight.device, dtype=compute_dtype))
            g_hidden = g_hidden.to(device=hidden_pre.device, dtype=hidden_pre.dtype)
            g_hidden_pre = self._silu_input_pullback_matrix(hidden_pre, g_hidden)
            g_in = self._linear_input_pullback_matrix(linear1, g_hidden_pre)
            return g_in.to(device=g_out.device, dtype=g_out.dtype)
        raise TypeError(f"unsupported RegionMessageHead net type: {type(net).__name__}")

    def _mean_pool_pullback_matrix(self, g_pooled: torch.Tensor, seq_len: int) -> torch.Tensor:
        if g_pooled.dim() != 3:
            raise ValueError("mean-pool pullback expects g_pooled shaped [B, P, D].")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        return g_pooled.unsqueeze(2).expand(-1, -1, seq_len, -1) / float(seq_len)

    def _broadcast_message_pullback_matrix(self, g_hidden_input: torch.Tensor) -> torch.Tensor:
        if g_hidden_input.dim() != 4:
            raise ValueError("broadcast pullback expects g_hidden_input shaped [B, P, L, D].")
        return g_hidden_input.sum(dim=2)

    def region_message_output_pullback_to_hidden_output(
        self,
        region_cache: RegionForwardCache,
        g_message_out: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if g_message_out.dim() != 3:
            raise ValueError("g_message_out must have shape [B, P, R].")
        g_pre_norm = self._module_input_pullback_matrix_autograd(
            self.message_norm[region_cache.region_idx],
            region_cache.pre_norm_message,
            g_message_out,
        )
        g_message_skip = g_pre_norm
        g_delta_message = g_pre_norm * region_cache.alpha.to(dtype=g_pre_norm.dtype).view(1, 1, 1)
        g_pooled_hidden = self._region_message_head_input_pullback_matrix(
            self.hidden_to_message[region_cache.region_idx],
            region_cache.pooled_hidden,
            g_delta_message,
        )
        g_hidden_output = self._mean_pool_pullback_matrix(g_pooled_hidden, seq_len=region_cache.hidden_output.shape[1])
        return {
            "g_pre_norm_message": g_pre_norm,
            "g_message_skip": g_message_skip,
            "g_delta_message": g_delta_message,
            "g_pooled_hidden": g_pooled_hidden,
            "g_hidden_output": g_hidden_output,
        }

    def region_hidden_input_pullback_to_message_input(
        self,
        region_cache: RegionForwardCache,
        g_hidden_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if g_hidden_input.dim() != 4:
            raise ValueError("g_hidden_input must have shape [B, P, L, D].")
        g_hidden_bias = self._broadcast_message_pullback_matrix(g_hidden_input)
        g_message_input = self._region_message_head_input_pullback_matrix(
            self.message_to_hidden[region_cache.region_idx],
            region_cache.message_in,
            g_hidden_bias,
        )
        return {
            "g_hidden_bias": g_hidden_bias,
            "g_message_input": g_message_input,
        }

    def region_outer_message_input_pullback(
        self,
        region_cache: RegionForwardCache,
        g_message_out: torch.Tensor,
        g_hidden_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        tail = self.region_message_output_pullback_to_hidden_output(region_cache, g_message_out)
        head = self.region_hidden_input_pullback_to_message_input(region_cache, g_hidden_input)
        return {
            **tail,
            **head,
            "g_message_input_total": tail["g_message_skip"] + head["g_message_input"],
        }

    def interface_vjp_chain(self, cache: Dict[str, Any], g_top_message: torch.Tensor) -> List[torch.Tensor]:
        """Reference region-message pullback chain (sequential over regions)."""
        region_messages: Sequence[torch.Tensor] = cache["region_messages"]
        if len(region_messages) != self.n_regions + 1:
            raise ValueError("cache has invalid region_messages length.")
        g_msgs: List[torch.Tensor] = [torch.zeros_like(region_messages[0]) for _ in range(self.n_regions + 1)]
        g_msgs[-1] = g_top_message
        for ridx in reversed(range(self.n_regions)):
            g_prev = torch.autograd.grad(
                region_messages[ridx + 1],
                region_messages[ridx],
                grad_outputs=g_msgs[ridx + 1],
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            g_msgs[ridx] = g_prev
        return g_msgs

    def materialize_interface_pullback_mats(self, cache: Dict[str, Any]) -> List[torch.Tensor]:
        """Materialize per-region pullback matrices A_k^T where g_k = A_k^T g_{k+1}."""
        region_messages: Sequence[torch.Tensor] = cache["region_messages"]
        if len(region_messages) != self.n_regions + 1:
            raise ValueError("cache has invalid region_messages length.")
        mats: List[torch.Tensor] = []
        for ridx in range(self.n_regions):
            m_in = region_messages[ridx]
            m_out = region_messages[ridx + 1]
            if m_in.dim() != 2 or m_out.dim() != 2:
                raise ValueError("region_messages must be [B, R].")
            bsz, rank = m_out.shape
            eye = torch.eye(rank, device=m_out.device, dtype=m_out.dtype)
            grad_out_batched = eye.unsqueeze(1).expand(rank, bsz, rank)
            try:
                grad_in_batched = torch.autograd.grad(
                    m_out,
                    m_in,
                    grad_outputs=grad_out_batched,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                    is_grads_batched=True,
                )[0]
                mats.append(grad_in_batched.permute(1, 2, 0).contiguous())
            except (TypeError, RuntimeError) as exc:
                if isinstance(exc, RuntimeError) and "doesn't have storage" not in str(exc):
                    raise
                _report_interface_jacobian_notice(
                    "batched_vjp_fallback",
                    "[region_interface] falling back to per-basis Jacobian construction because batched VJP is incompatible with the current custom CUDA autograd path",
                )
                cols: List[torch.Tensor] = []
                for j in range(rank):
                    grad_out = eye[j].view(1, rank).expand(bsz, rank)
                    grad_in = torch.autograd.grad(
                        m_out,
                        m_in,
                        grad_outputs=grad_out,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=False,
                    )[0]
                    cols.append(grad_in.unsqueeze(-1))
                mats.append(torch.cat(cols, dim=-1))
        return mats

    def compose_pullback_suffix(self, interface_pullback_mats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Suffix products S_k = A_k^T ... A_{K-1}^T with S_K = I."""
        count = len(interface_pullback_mats)
        if count == 0:
            return []
        bsz, rank, rank2 = interface_pullback_mats[0].shape
        if rank != rank2:
            raise ValueError("interface pullback matrices must be square.")
        mats = torch.stack(list(interface_pullback_mats), dim=1).contiguous()
        suffix = suffix_scan_pullbacks(mats)
        if suffix.shape != (bsz, count + 1, rank, rank):
            raise ValueError("suffix scan returned an unexpected shape")
        return [suffix[:, ridx].contiguous() for ridx in range(count + 1)]

    def _apply_pullback_mat(
        self,
        mat: torch.Tensor,
        vec: torch.Tensor,
        *,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        compute_dtype = torch.promote_types(mat.dtype, vec.dtype)
        result = torch.einsum(
            "bij,bj->bi",
            mat.to(dtype=compute_dtype),
            vec.to(device=mat.device, dtype=compute_dtype),
        )
        if out_dtype is None:
            out_dtype = vec.dtype
        return result.to(device=vec.device, dtype=out_dtype)

    def interface_vjp_scan_from_mats(
        self,
        interface_pullback_mats: Sequence[torch.Tensor],
        g_top_message: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Apply suffix-composed pullback summaries to top message gradient."""
        count = len(interface_pullback_mats)
        suffix = self.compose_pullback_suffix(interface_pullback_mats)
        if len(suffix) != count + 1:
            raise ValueError("suffix length mismatch.")
        target_dtype = suffix[0].dtype
        target_device = suffix[0].device
        g_top_message = g_top_message.to(device=target_device, dtype=target_dtype)
        g_msgs: List[torch.Tensor] = []
        for ridx in range(count):
            g_msgs.append(self._apply_pullback_mat(suffix[ridx], g_top_message, out_dtype=g_top_message.dtype))
        g_msgs.append(g_top_message)
        return g_msgs

    def interface_vjp_scan(self, cache: Dict[str, Any], g_top_message: torch.Tensor) -> List[torch.Tensor]:
        """Region pullback via materialized interface summaries + suffix composition."""
        mats = self.materialize_interface_pullback_mats(cache)
        return self.interface_vjp_scan_from_mats(mats, g_top_message)

    def interface_vjp_scan_from_last_input_seed(
        self,
        interface_pullback_mats: Sequence[torch.Tensor],
        g_last_input_message: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Scan message adjoints with seed at m_{K-1} (last region input message).

        Returns [g_{m0}, ..., g_{m_{K-1}}].
        """
        if len(interface_pullback_mats) != self.n_regions:
            raise ValueError("interface_pullback_mats length mismatch.")
        if self.n_regions == 0:
            return []
        if self.n_regions == 1:
            return [g_last_input_message]
        mats = interface_pullback_mats[:-1]
        suffix = self.compose_pullback_suffix(mats)
        target_dtype = suffix[0].dtype
        target_device = suffix[0].device
        g_last_input_message = g_last_input_message.to(device=target_device, dtype=target_dtype)
        return [
            self._apply_pullback_mat(suffix[ridx], g_last_input_message, out_dtype=g_last_input_message.dtype)
            for ridx in range(self.n_regions)
        ]
