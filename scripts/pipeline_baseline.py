"""Pipeline-parallel baseline of the four-device rows: the same stack sharded
one region's worth of blocks per GPU, but with standard pipeline parallelism,
where full activations cross the stage boundaries and the backward is the
sequential stage chain scheduled by torch's GPipe or 1F1B. The stack folds the
residual at each boundary so every stage has single-tensor I/O.

Run: python scripts/pipeline_baseline.py [args]   (mp.spawn)"""

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
from torch.distributed.pipelining import (PipelineStage, Schedule1F1B, ScheduleGPipe,
                                          ScheduleInterleaved1F1B, ScheduleLoopedBFS)

from scripts import comm_bytes
from scripts.region_parallel_common import PAPER_SHAPE

from backbones.mamba3.block import Mamba3Block


def _build_blocks(a, n, device, dtype, offset=0):
    from backbones.transformer.block import TransformerBlock

    heads = a.dim // a.headdim

    def _tf():
        return TransformerBlock(a.dim, n_heads=heads, n_kv_heads=heads, head_dim=a.headdim,
                                d_conv=PAPER_SHAPE["d_conv"]).to(device=device, dtype=dtype)

    def _m3():
        return Mamba3Block(a.dim, d_state=a.d_state, expand=PAPER_SHAPE["expand"], headdim=a.headdim,
                           ngroups=PAPER_SHAPE["ngroups"], chunk_size=a.chunk_size, device=device, dtype=dtype)

    if a.backbone == "transformer":
        return [_tf() for _ in range(n)]
    if a.backbone == "hybrid":
        # 3x mamba3 + 1x transformer by GLOBAL layer index; offset places the stage.
        return [_tf() if (offset + i) % 4 == 3 else _m3() for i in range(n)]
    return [_m3() for _ in range(n)]


class Stage(nn.Module):
    """One pipeline stage: optional embed (first) -> backbone blocks (folded
    residual) -> optional final norm + lm_head (last). Single-tensor I/O."""

    def __init__(self, a, layer_lo, layer_hi, *, is_first, is_last, device, dtype):
        super().__init__()
        self.is_first, self.is_last = is_first, is_last
        self.dtype = dtype
        if is_first:
            self.embed = nn.Embedding(a.vocab, a.dim, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(_build_blocks(a, layer_hi - layer_lo, device, dtype, offset=layer_lo))
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
    spd = a.stages_per_device
    num_stages = world_size * spd
    lps = total_layers // num_stages
    mb = a.batch // a.microbatches

    def make_stage(sidx):
        # Interleaved placement (Megatron-style virtual pipeline): stage s
        # lives on device s % world_size, so every stage boundary crosses a
        # device and the number of cuts is num_stages - 1.
        lo, hi = sidx * lps, (sidx + 1) * lps if sidx < num_stages - 1 else total_layers
        mod = Stage(a, lo, hi, is_first=(sidx == 0), is_last=(sidx == num_stages - 1),
                    device=dev, dtype=torch.bfloat16)
        # Example input is the PER-MICROBATCH shape (the schedule splits along dim 0).
        if sidx == 0:
            example = torch.randint(0, a.vocab, (mb, a.seq_len), device=dev)
        else:
            example = torch.zeros(mb, a.seq_len, a.dim, device=dev, dtype=torch.bfloat16)
        return PipelineStage(mod, sidx, num_stages, dev, input_args=(example,))

    def loss_fn(output, target):
        return output.float().square().mean()

    if spd == 1:
        stage = make_stage(rank)
        Sched = Schedule1F1B if a.schedule == "1f1b" else ScheduleGPipe
        sched = Sched(stage, n_microbatches=a.microbatches, loss_fn=loss_fn)
    else:
        stages = [make_stage(rank + i * world_size) for i in range(spd)]
        Sched = ScheduleInterleaved1F1B if a.schedule == "1f1b" else ScheduleLoopedBFS
        sched = Sched(stages, n_microbatches=a.microbatches, loss_fn=loss_fn)
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
    comm_bytes.install()
    for _ in range(a.warmup):
        step(); torch.cuda.synchronize(dev); dist.barrier()
    comm_bytes.reset()
    torch.cuda.reset_peak_memory_stats(dev)
    ts = []
    for _ in range(a.iters):
        dist.barrier(); torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        step(); torch.cuda.synchronize(dev); dist.barrier()
        ts.append((time.perf_counter() - t0) * 1000)
    step_ms = sum(ts) / len(ts)
    counted = comm_bytes.total_bytes() / a.iters
    peak = torch.cuda.max_memory_allocated(dev) / 1e6

    peak_t = torch.tensor([peak], device=dev)
    dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)
    counted_t = torch.tensor([counted], device=dev, dtype=torch.float64)
    dist.all_reduce(counted_t, op=dist.ReduceOp.SUM)
    if rank == 0:
        act_bytes = mb * a.seq_len * a.dim * 2  # bf16 activation per boundary per microbatch
        bubble = (num_stages - 1) / (a.microbatches + num_stages - 1)
        print(f"regions/stages={num_stages} devices={world_size} stages/device={spd} dim={a.dim} "
              f"seq={a.seq_len} batch={a.batch} layers/stage={lps} microbatches={a.microbatches} "
              f"schedule={a.schedule} warmup={a.warmup} iters={a.iters}", flush=True)
        print(f"pipeline-parallel fwd+bwd: {step_ms:8.2f} ms/step | peak mem {float(peak_t.item()):8.1f} MB | "
              f"bubble~{bubble:.0%} | per-boundary activation {act_bytes/1e6:.1f} MB x {a.microbatches} microbatches "
              f"x 2 (fwd+bwd) x {num_stages - 1} boundaries", flush=True)
        print(f"counted comm/step: {float(counted_t.item()) / 1e6:.2f} MB "
              f"(outbound bytes at the dist call boundary, summed over ranks)", flush=True)
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["mamba3", "transformer", "hybrid"], default="mamba3")
    p.add_argument("--regions", type=int, default=4)  # == pipeline stages / GPUs
    p.add_argument("--world-size", type=int, default=torch.cuda.device_count())
    p.add_argument("--layers-per-region", type=int, default=2)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--headdim", type=int, default=PAPER_SHAPE["headdim"])
    p.add_argument("--d-state", type=int, default=PAPER_SHAPE["d_state"])
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=PAPER_SHAPE["chunk_size"])
    p.add_argument("--microbatches", type=int, default=4)
    p.add_argument("--schedule", choices=["gpipe", "1f1b"], default="gpipe")
    p.add_argument("--stages-per-device", type=int, default=1,
                   help="virtual pipeline: stages per device with interleaved placement "
                        "(stage s on device s %% world_size); 1 = one stage per device; "
                        ">1 uses torch's looped-BFS (gpipe) or interleaved-1F1B (1f1b) "
                        "multi-stage schedules")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=29517)
    a = p.parse_args()
    mp.spawn(worker, args=(a.world_size, a), nprocs=a.world_size, join=True)


if __name__ == "__main__":
    main()
