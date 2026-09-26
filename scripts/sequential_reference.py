"""The sequential reference row of the four-device tables: the same stack
with no interfaces, backpropagated on one GPU in eager bf16; prints forward,
forward+backward, and peak memory.

Run: python scripts/sequential_reference.py [--seq-len 2048 --batch 8]
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

from scripts.region_parallel_common import PAPER_SHAPE


def _build_blocks(backbone, dim, head_dim, layers, dev, dt):
    from backbones.mamba3.block import Mamba3Block
    from backbones.transformer.block import TransformerBlock

    def _tf():
        return TransformerBlock(dim, n_heads=dim // head_dim, n_kv_heads=dim // head_dim,
                                head_dim=head_dim, d_conv=PAPER_SHAPE["d_conv"]).to(device=dev, dtype=dt)

    def _m3():
        return Mamba3Block(dim, d_state=PAPER_SHAPE["d_state"], expand=PAPER_SHAPE["expand"], headdim=head_dim,
                           ngroups=PAPER_SHAPE["ngroups"], chunk_size=PAPER_SHAPE["chunk_size"], device=dev, dtype=dt)

    if backbone == "transformer":
        return [_tf() for _ in range(layers)]
    if backbone == "hybrid":
        if layers % 4 != 0:
            raise ValueError("hybrid requires layers divisible by 4 (3x mamba3 + 1x transformer)")
        return [_tf() if i % 4 == 3 else _m3() for i in range(layers)]
    return [_m3() for _ in range(layers)]


def run(L, B, *, backbone, dim, head_dim, layers, vocab):
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    embed = nn.Embedding(vocab, dim, device=dev, dtype=dt)
    blocks = nn.ModuleList(_build_blocks(backbone, dim, head_dim, layers, dev, dt))
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
    ap.add_argument("--backbone", choices=["mamba3", "transformer", "hybrid"], default="mamba3")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--head-dim", type=int, default=PAPER_SHAPE["headdim"])
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=256)
    a = ap.parse_args()
    run(a.seq_len, a.batch, backbone=a.backbone, dim=a.dim, head_dim=a.head_dim,
        layers=a.layers, vocab=a.vocab)


if __name__ == "__main__":
    main()
