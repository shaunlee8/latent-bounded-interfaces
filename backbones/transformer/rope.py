from __future__ import annotations

import torch


def apply_rope(x: torch.Tensor, base: float = 10000.0, interleaved: bool = False) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("RoPE expects x shaped [B, H, L, D].")
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")
    device = x.device
    dtype = x.dtype
    half = head_dim // 2
    positions = torch.arange(x.shape[-2], device=device, dtype=torch.float32)
    freq_seq = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = base ** (-2.0 * freq_seq / float(head_dim))
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().view(1, 1, x.shape[-2], half)
    sin = freqs.sin().view(1, 1, x.shape[-2], half)
    if interleaved:
        x_pairs = x.float().view(*x.shape[:-1], half, 2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        out = torch.stack((rot_even, rot_odd), dim=-1).reshape_as(x_pairs.new_empty(*x.shape))
    else:
        x_even = x[..., :half].float()
        x_odd = x[..., half:].float()
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        out = torch.cat((rot_even, rot_odd), dim=-1)
    return out.to(dtype=dtype)
