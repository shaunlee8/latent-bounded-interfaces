"""Dense sequential backprop row: the same transformer stack with no
interfaces, single GPU, eager bf16 -- fwd-only, fwd+bwd, peak memory.

This is the method-tax baseline for the region-parallel rows: an ordinary
sequentially backpropagated dense model of identical width/depth.

Run: python benchmarks/dense_sequential_baseline.py [--seq-len 2048 --batch 8]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn


def _build_blocks(backbone, dim, head_dim, layers, d_conv, dev, dt):
    if backbone == "transformer":
        from backbones.transformer.block import TransformerBlock

        return [
            TransformerBlock(dim, n_heads=dim // head_dim, n_kv_heads=dim // head_dim,
                             head_dim=head_dim, d_conv=d_conv).to(device=dev, dtype=dt)
            for _ in range(layers)
        ]
    from backbones.mamba3.block import Mamba3Block

    return [
        Mamba3Block(dim, d_state=128, expand=2, headdim=head_dim, ngroups=1,
                    chunk_size=64, device=dev, dtype=dt)
        for _ in range(layers)
    ]


def run(L, B, *, backbone, dim, head_dim, layers, vocab, d_conv):
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    embed = nn.Embedding(vocab, dim, device=dev, dtype=dt)
    blocks = nn.ModuleList(_build_blocks(backbone, dim, head_dim, layers, d_conv, dev, dt))
    norm = nn.RMSNorm(dim, device=dev, dtype=dt)
    head = nn.Linear(dim, vocab, bias=False, device=dev, dtype=dt)
    ids = torch.randint(0, vocab, (B, L), device=dev)

    def fwd():
        h, res = embed(ids), None
        for b in blocks:
            h, res = b(h, residual=res)
        return head(norm((h + res).to(dt))).float().square().mean()

    def timed(fn, w=3, it=10):
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(it):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / it * 1e3

    with torch.no_grad():
        f_ms = timed(fwd)
    torch.cuda.reset_peak_memory_stats()

    def step():
        loss = fwd()
        loss.backward()
        for p in [*embed.parameters(), *blocks.parameters(),
                  *norm.parameters(), *head.parameters()]:
            p.grad = None

    fb_ms = timed(step)
    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"L={L} B={B}: dense fwd {f_ms:7.2f} ms | fwd+bwd {fb_ms:7.2f} ms | "
          f"bwd/f {(fb_ms - f_ms) / f_ms:4.2f} | peak {peak:8.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["mamba3", "transformer"], default="transformer")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--d-conv", type=int, default=4)
    a = ap.parse_args()
    run(a.seq_len, a.batch, backbone=a.backbone, dim=a.dim, head_dim=a.head_dim,
        layers=a.layers, vocab=a.vocab, d_conv=a.d_conv)


if __name__ == "__main__":
    main()
