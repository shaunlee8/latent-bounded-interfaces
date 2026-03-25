from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from backbones.general import BackboneSpec, build_backbone_stack

_INTERFACE_JACOBIAN_NOTICE = {"batched_vjp_fallback": False}


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
        return self.net(x)


@dataclass
class RegionInterfaceCache:
    boundaries: List[torch.Tensor]
    region_hidden_inputs: List[torch.Tensor]
    region_messages: List[torch.Tensor]
    region_ranges: List[Tuple[int, int]]


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
        self.norm = nn.LayerNorm(self.dim)
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

    def _pool_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def forward_with_cache(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, Dict[str, Any]]:
        x_static = self.embedding(input_ids)
        pooled0 = self._pool_hidden(x_static)
        m = self.input_to_message(pooled0)

        boundaries: List[torch.Tensor] = [x_static]
        region_hidden_inputs: List[torch.Tensor] = []
        region_messages: List[torch.Tensor] = [m]
        for ridx, (start, end) in enumerate(self.region_ranges):
            # Strict bounded-interface mode: each region hidden stream is rebuilt
            # from static token embedding + current message, with no dense carry.
            x = x_static + self.message_to_hidden[ridx](m).unsqueeze(1)
            region_hidden_inputs.append(x)
            x = self.backbone.forward_range(x, start, end)
            boundaries.append(x)

            delta_m = self.hidden_to_message[ridx](self._pool_hidden(x))
            alpha = torch.tanh(self.message_alpha[ridx])
            m = self.message_norm[ridx](m + alpha * delta_m)
            region_messages.append(m)

        logits = self.lm_head(self.norm(x))
        cache = RegionInterfaceCache(
            boundaries=boundaries,
            region_hidden_inputs=region_hidden_inputs,
            region_messages=region_messages,
            region_ranges=self.region_ranges,
        )
        return logits, {
            "boundaries": cache.boundaries,
            "region_hidden_inputs": cache.region_hidden_inputs,
            "region_messages": cache.region_messages,
            "region_ranges": cache.region_ranges,
        }

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_cache(input_ids)
        return logits

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
        if len(interface_pullback_mats) != self.n_regions:
            raise ValueError("interface_pullback_mats length mismatch.")
        if self.n_regions == 0:
            return []
        bsz, rank, rank2 = interface_pullback_mats[0].shape
        if rank != rank2:
            raise ValueError("interface pullback matrices must be square.")
        eye = torch.eye(rank, device=interface_pullback_mats[0].device, dtype=interface_pullback_mats[0].dtype)
        mats = torch.stack(list(interface_pullback_mats), dim=1).contiguous()
        suffix_inclusive = mats.clone()
        offset = 1
        while offset < self.n_regions:
            prev = suffix_inclusive.clone()
            suffix_inclusive[:, :-offset] = torch.matmul(prev[:, :-offset], prev[:, offset:])
            offset *= 2
        suffix = [suffix_inclusive[:, ridx].contiguous() for ridx in range(self.n_regions)]
        suffix.append(eye.view(1, rank, rank).expand(bsz, rank, rank).clone())
        return suffix

    def interface_vjp_scan_from_mats(
        self,
        interface_pullback_mats: Sequence[torch.Tensor],
        g_top_message: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Apply suffix-composed pullback summaries to top message gradient."""
        suffix = self.compose_pullback_suffix(interface_pullback_mats)
        if len(suffix) != self.n_regions + 1:
            raise ValueError("suffix length mismatch.")
        g_msgs: List[torch.Tensor] = []
        for ridx in range(self.n_regions):
            g_msgs.append(torch.einsum("bij,bj->bi", suffix[ridx], g_top_message))
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
        bsz, rank, rank2 = mats[0].shape
        if rank != rank2:
            raise ValueError("interface pullback matrices must be square.")
        mats_stacked = torch.stack(list(mats), dim=1).contiguous()
        suffix_inclusive = mats_stacked.clone()
        offset = 1
        while offset < (self.n_regions - 1):
            prev = suffix_inclusive.clone()
            suffix_inclusive[:, : -(offset)] = torch.matmul(prev[:, : -(offset)], prev[:, offset:])
            offset *= 2
        eye = torch.eye(rank, device=mats[0].device, dtype=mats[0].dtype)
        suffix: List[torch.Tensor] = [suffix_inclusive[:, ridx].contiguous() for ridx in range(self.n_regions - 1)]
        suffix.append(eye.view(1, rank, rank).expand(bsz, rank, rank).clone())
        return [torch.einsum("bij,bj->bi", suffix[ridx], g_last_input_message) for ridx in range(self.n_regions)]
