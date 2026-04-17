from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from backbones.transformer.rope import apply_rope

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_base: float = 10000.0,
        rope_interleaved: bool = False,
        softmax_scale: float | None = None,
        d_conv: int = 0,
        use_flash_attn: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.rope_interleaved = rope_interleaved
        self.softmax_scale = softmax_scale
        self.d_conv = d_conv
        self.use_flash_attn = use_flash_attn
        self._flash_attn_disabled = flash_attn_func is None
        self.qkv_dim = self.head_dim * (self.n_heads + 2 * self.n_kv_heads)
        self.out_dim = self.head_dim * self.n_heads
        self.in_proj = nn.Linear(dim, self.qkv_dim, bias=bias)
        if self.d_conv > 0:
            self.conv1d = nn.Conv1d(
                self.qkv_dim,
                self.qkv_dim,
                kernel_size=self.d_conv,
                padding=self.d_conv - 1,
                groups=self.qkv_dim,
                bias=bias,
            )
        self.out_proj = nn.Linear(self.out_dim, dim, bias=bias)

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_kv_heads == self.n_heads:
            return x
        repeat_factor = self.n_heads // self.n_kv_heads
        return x.repeat_interleave(repeat_factor, dim=1)

    def _can_use_flash_attn(self, x: torch.Tensor) -> bool:
        if not self.use_flash_attn or self._flash_attn_disabled:
            return False
        if not x.is_cuda:
            return False
        return x.dtype in {torch.float16, torch.bfloat16}

    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if flash_attn_func is None:
            raise RuntimeError("flash_attn is not available")
        scale = self.softmax_scale if self.softmax_scale is not None else None
        return flash_attn_func(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            softmax_scale=scale,
            causal=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        qkv = self.in_proj(x)
        if self.d_conv > 0:
            qkv = self.conv1d(qkv.transpose(1, 2))[..., :seqlen].transpose(1, 2).contiguous()
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        q, k, v = qkv.split((q_dim, kv_dim, kv_dim), dim=-1)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, base=self.rope_base, interleaved=self.rope_interleaved)
        k = apply_rope(k, base=self.rope_base, interleaved=self.rope_interleaved)
        k = self._expand_kv(k)
        v = self._expand_kv(v)
        if self._can_use_flash_attn(x):
            try:
                attn = self._flash_attention(q, k, v)
                attn = attn.contiguous().view(bsz, seqlen, self.out_dim)
                return self.out_proj(attn)
            except RuntimeError:
                self._flash_attn_disabled = True
        scale = self.softmax_scale if self.softmax_scale is not None else 1.0 / math.sqrt(self.head_dim)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.out_dim)
        return self.out_proj(attn)
