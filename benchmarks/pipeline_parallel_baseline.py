"""Pipeline-parallel backprop baseline (the fair opponent for LBI region-parallel).

Shards the SAME mamba stack across GPUs -- one "region's worth" of layers per GPU,
matching the LBI sharding -- but with standard pipeline parallelism: full
activations cross stage boundaries and the backward is the usual sequential stage
chain, scheduled by torch's official GPipe/1F1B (a fair, bubble-minimizing
schedule). Compare its forward+backward wall-clock and peak memory to
`nccl_region_parallel.py` at the matched config.

The stack uses the residual-fold identity so each stage has single-tensor I/O:
passing `hidden + residual` (residual reset) across a boundary is exactly the
continuous residual stream.

Run: python benchmarks/pipeline_parallel_baseline.py [args]   (mp.spawn)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

from backbones.mamba3.block import Mamba3Block


def _build_blocks(a, n, device, dtype):
    if a.backbone == "transformer":
        from backbones.transformer.block import TransformerBlock

        heads = a.dim // a.headdim
        return [
            TransformerBlock(a.dim, n_heads=heads, n_kv_heads=heads,
                             head_dim=a.headdim, d_conv=4).to(device=device, dtype=dtype)
            for _ in range(n)
        ]
    return [
        Mamba3Block(a.dim, d_state=a.d_state, expand=2, headdim=a.headdim, ngroups=1,
                    chunk_size=a.chunk_size, device=device, dtype=dtype)
        for _ in range(n)
    ]


class Stage(nn.Module):
    """One pipeline stage: optional embed (first) -> backbone blocks (folded
    residual) -> optional final norm + lm_head (last). Single-tensor I/O."""

    def __init__(self, a, layer_lo, layer_hi, *, is_first, is_last, device, dtype):
        super().__init__()
        self.is_first, self.is_last = is_first, is_last
        self.dtype = dtype
        if is_first:
            self.embed = nn.Embedding(a.vocab, a.dim, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(_build_blocks(a, layer_hi - layer_lo, device, dtype))
        if is_last:
            self.norm = nn.RMSNorm(a.dim, device=device, dtype=dtype)
            self.lm_head = nn.Linear(a.dim, a.vocab, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        hidden = self.embed(x) if self.is_first else x
        residual = None
        for b in self.blocks:
            hidden, residual = b(hidden, residual=residual)
        hidden = (hidden + residual) if residual is not None else hidden
        # The folded residual may ride an fp32 stream; the boundary contract
        # (and the head) is the block dtype.
        hidden = hidden.to(self.dtype)
        if self.is_last:
            return self.lm_head(self.norm(hidden))
        return hidden


def worker(rank, world_size, a):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(a.port))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    torch.manual_seed(a.seed + rank)

    total_layers = a.regions * a.layers_per_region
    lps = total_layers // world_size
    lo, hi = rank * lps, (rank + 1) * lps if rank < world_size - 1 else total_layers
    stage_mod = Stage(a, lo, hi, is_first=(rank == 0), is_last=(rank == world_size - 1),
                      device=dev, dtype=torch.bfloat16)

    # Example input is the PER-MICROBATCH shape (the schedule splits along dim 0).
    mb = a.batch // a.microbatches
    if rank == 0:
        example = torch.randint(0, a.vocab, (mb, a.seq_len), device=dev)
    else:
        example = torch.zeros(mb, a.seq_len, a.dim, device=dev, dtype=torch.bfloat16)
    stage = PipelineStage(stage_mod, rank, world_size, dev, input_args=(example,))

    def loss_fn(output, target):
        return output.float().square().mean()

    sched = ScheduleGPipe(stage, n_microbatches=a.microbatches, loss_fn=loss_fn)
    input_ids = torch.randint(0, a.vocab, (a.batch, a.seq_len), device=dev)
    target = torch.zeros(a.batch, device=dev, dtype=torch.bfloat16)  # ignored by loss_fn

    def step():
        if rank == 0:
            sched.step(input_ids)
        elif rank == world_size - 1:
            sched.step(target=target, losses=[])
        else:
            sched.step()

    # time forward+backward, peak memory
    for _ in range(a.warmup):
        step(); torch.cuda.synchronize(dev); dist.barrier()
    torch.cuda.reset_peak_memory_stats(dev)
    ts = []
    for _ in range(a.iters):
        dist.barrier(); torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        step(); torch.cuda.synchronize(dev); dist.barrier()
        ts.append((time.perf_counter() - t0) * 1000)
    step_ms = sum(ts) / len(ts)
    peak = torch.cuda.max_memory_allocated(dev) / 1e6

    peak_t = torch.tensor([peak], device=dev)
    dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)
    if rank == 0:
        act_bytes = a.batch * a.seq_len * a.dim * 2  # bf16 activation per boundary per microbatch
        bubble = (world_size - 1) / (a.microbatches + world_size - 1)
        print(f"regions/stages={world_size} dim={a.dim} seq={a.seq_len} batch={a.batch} "
              f"layers/stage={lps} microbatches={a.microbatches}", flush=True)
        print(f"pipeline-parallel fwd+bwd: {step_ms:8.2f} ms/step | peak mem {float(peak_t.item()):8.1f} MB | "
              f"bubble~{bubble:.0%} | per-boundary activation {act_bytes/1e6:.1f} MB x {a.microbatches} microbatches "
              f"x 2 (fwd+bwd) x {world_size - 1} boundaries", flush=True)
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["mamba3", "transformer"], default="mamba3")
    p.add_argument("--regions", type=int, default=4)  # == pipeline stages / GPUs
    p.add_argument("--world-size", type=int, default=torch.cuda.device_count())
    p.add_argument("--layers-per-region", type=int, default=6)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--headdim", type=int, default=64)
    p.add_argument("--d-state", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--microbatches", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=29517)
    a = p.parse_args()
    mp.spawn(worker, args=(a.world_size, a), nprocs=a.world_size, join=True)


if __name__ == "__main__":
    main()
