"""Region-sharded LBI training step: rank k owns region k.

The forward is a chain of [B, r] bf16 state messages: each rank receives its
input state, runs its region body once (the caches double as the backward's
operating points), updates the interface state, and sends it on; A_k is then
constructed by forward mode, overlapping the later ranks' chain turns. The
loss lives on the last rank, the seed adjoint broadcasts back as one [B, r]
tensor, and the backward is the region-parallel scan (all-reduce the rank-r
A_k, local suffix scan, per-rank region param grads). No rank forwards
another rank's region, so per-rank compute and cache memory drop K-fold
versus the replicated-forward driver (`nccl_region_parallel.py`).

Run: python benchmarks/sharded_forward_region_parallel.py [args]  (mp.spawn;
world size must equal --regions)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from backward import NativeInterfacePullbackProvider, NativeLocalVJPProvider, ScanADEngine
from backward.local_vjp import native_initial_backward, native_region_backward
from backward.pullbacks import interface_state_jacobian_for_region_forward
from backward.suffix_scan import propagate_state_adjoint_from_last_region_input
from benchmarks.nccl_region_parallel import (
    _reduce_shared, _time_barrier, _worst_rel, build_model)
from models.lbi_language_model import LBIRegionCache


def sharded_step(model, provider, input_ids, rank, num_regions, dev, microbatches=1):
    """One region-sharded training step; returns (grad_map, loss-or-None).
    With microbatches > 1 the state chain runs fill-drain ([B/m, r] messages,
    chain wall ~ (m+K-1)/(m*K) * F_full); A_k construction runs after the
    chain loop, off the downstream ranks' critical path."""
    m = microbatches
    B = input_ids.shape[0]
    if B % m != 0:
        raise ValueError("batch must divide microbatches")
    mb = B // m

    # Canvas + initial state are small grad-connected graphs, replicated.
    canvas = model.canvas(input_ids)
    state0 = model.interface.initialize(canvas)
    rr = state0.shape[1]

    # ---- forward chain (fill-drain over microbatches) ----
    rcs, s_ins = [], []
    for j in range(m):
        sl = slice(j * mb, (j + 1) * mb)
        if rank == 0:
            s_in = state0[sl].detach().contiguous()
        else:
            s_in = torch.empty(mb, rr, device=dev, dtype=state0.dtype)
            dist.recv(s_in, src=rank - 1)
        with torch.no_grad():
            condition = model.interface.decode(s_in, rank)
            region_input = canvas[sl].detach() + condition.unsqueeze(1)
            region_output, backend_cache = model.region_backend.forward_region(
                region_input=region_input, region_index=rank
            )
            step = model.interface.update(s_in, region_output, rank)
            s_out = step.state.to(dtype=state0.dtype).contiguous()
        if rank < num_regions - 1:
            dist.send(s_out, dst=rank + 1)
        rcs.append(LBIRegionCache(
            region_index=rank,
            layer_range=tuple(model.region_ranges[rank]),
            backend_cache=backend_cache,
            canvas_features=canvas[sl].detach(),
            state_in=s_in,
            condition=condition,
            region_input=region_input,
            region_output=region_output,
            interface_step=step,
            state_out=s_out,
        ))
        s_ins.append(s_in)

    # ---- A_k, per microbatch, off the chain's critical path ----
    A = torch.zeros(num_regions, B, rr, rr, device=dev, dtype=state0.dtype)
    with torch.no_grad():
        for j in range(m):
            sl = slice(j * mb, (j + 1) * mb)

            def _jvp(basis, _bc=rcs[j].backend_cache):
                return model.region_backend.region_output_jvp(
                    cache=_bc, region_input_tangent_basis=basis, pooled=True
                )

            A[rank, sl] = interface_state_jacobian_for_region_forward(
                model=model, region_cache=rcs[j], region_output_jvp=_jvp
            ).to(A.dtype)

    # ---- loss + seed adjoints (last rank), broadcast ----
    grad_map = provider.new_grad_map(model)
    loss = None
    last_g_ro = [None] * m
    g_last = torch.empty(B, rr, device=dev, dtype=state0.dtype)
    if rank == num_regions - 1:
        losses, ro_inputs = [], []
        for j in range(m):
            ro = rcs[j].region_output.detach().requires_grad_(True)
            rcs[j].region_output = ro
            logits = model.readout(ro, canvas=model.canvas)
            losses.append(logits.float().square().mean())
            ro_inputs.append(ro)
        # Equal chunks: the mean of chunk means is the full-batch mean.
        loss = sum(losses) / m
        provider.store_output_head_grads(model=model, loss=loss, grad_map=grad_map,
                                         cache={"region_caches": rcs})
        for j in range(m):
            sl = slice(j * mb, (j + 1) * mb)
            g_last[sl] = provider.state_adjoint_from_loss(
                model=model, loss=loss, state=s_ins[j],
                cache={"region_caches": [rcs[j]]})
            last_g_ro[j] = torch.autograd.grad(loss, ro_inputs[j], retain_graph=True)[0]
    dist.broadcast(g_last, src=num_regions - 1)

    # ---- Phase 1 reduce + Phase 2 suffix scan (local, tiny) ----
    dist.all_reduce(A, op=dist.ReduceOp.SUM)
    sjt = [A[k] for k in range(num_regions)]
    gsi = propagate_state_adjoint_from_last_region_input(
        sjt, g_last.to(dtype=sjt[0].dtype), num_regions=num_regions
    )

    init_grads, canvas_init = native_initial_backward(
        model=model, state0=state0, state0_adjoint=gsi[0],
        cache={"canvas_features": canvas},
    )
    for n, g in init_grads.items():
        if n in grad_map:
            grad_map[n] = g

    # ---- Phase 3: own region, per microbatch (param grads sum over B;
    # canvas cotangent chunks concatenate along B) ----
    param_sums: dict[str, torch.Tensor] = {}
    shared_sums: dict[str, torch.Tensor] = {}
    canvas_ct = torch.zeros_like(canvas)
    for j in range(m):
        sl = slice(j * mb, (j + 1) * mb)
        cache_j = {"region_caches": [None] * num_regions, "canvas_features": canvas[sl]}
        cache_j["region_caches"][rank] = rcs[j]
        res_j = native_region_backward(
            model=model, loss=loss, states=None, region_index=rank,
            num_regions=num_regions, state_adjoints=[g[sl] for g in gsi],
            cache=cache_j, g_region_output=last_g_ro[j],
        )
        for n, g in res_j.param_grads.items():
            param_sums[n] = g if n not in param_sums else param_sums[n] + g
        for n, g in res_j.shared_partials.items():
            shared_sums[n] = g if n not in shared_sums else shared_sums[n] + g
        if res_j.canvas_cotangent is not None:
            canvas_ct[sl] = res_j.canvas_cotangent.to(dtype=canvas_ct.dtype)
    for n, g in param_sums.items():
        if n in grad_map:
            grad_map[n] = g

    # Canvas cotangent: one dtype on every rank before the collective.
    dist.all_reduce(canvas_ct, op=dist.ReduceOp.SUM)
    if canvas_init is not None:
        canvas_ct = canvas_ct + canvas_init

    shared = _reduce_shared(model, shared_sums, dev)

    provider._canvas_grad = canvas_ct
    provider._shared_grads = shared
    provider.store_shared_canvas_grads(model=model, loss=loss, grad_map=grad_map,
                                       cache={"canvas_features": canvas})
    return grad_map, loss


def worker(rank, world_size, a):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(a.port))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    torch.manual_seed(a.seed)
    if world_size != a.regions:
        raise ValueError("sharded forward requires world_size == regions (one region per rank)")

    model = build_model(a, dev)
    if a.phase1 == "forward":
        model.region_backend.forward_mode_use_kernel = True
    if os.environ.get("LBI_TURN_COMPILE", "0") != "0":
        model.region_backend.compile_forward_stages = True
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    input_ids = torch.randint(0, a.vocab, (a.batch, a.seq_len), device=dev)
    dist.broadcast(input_ids, src=0)
    provider = NativeLocalVJPProvider()

    # Correctness: full local sequential reference vs this rank's shard.
    logits_r, cache_r = model.forward_with_cache(input_ids, native_backward=True)
    ref = ScanADEngine(
        pullback_provider=NativeInterfacePullbackProvider(), local_vjp_provider=NativeLocalVJPProvider()
    ).backward(model=model, loss=logits_r.square().mean(), cache=cache_r)
    del logits_r, cache_r
    torch.cuda.empty_cache()

    gm, _ = sharded_step(model, provider, input_ids, rank, a.regions, dev,
                         microbatches=a.microbatches)
    worst = _worst_rel(gm, ref.grad_map, dev)
    del ref
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(dev)
    step_ms = _time_barrier(
        lambda: sharded_step(model, provider, input_ids, rank, a.regions, dev,
                             microbatches=a.microbatches),
        dev, a.warmup, a.iters,
    )
    peak = torch.cuda.max_memory_allocated(dev) / 1e6
    peak_t = torch.tensor([peak], device=dev)
    dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)

    if rank == 0:
        rr = a.rank
        state_bytes = a.batch * rr * 2 * (world_size - 1)
        a_bytes = a.regions * a.batch * rr * rr * 2
        canvas_bytes = a.batch * a.seq_len * a.dim * 2
        comm_mb = (state_bytes + a_bytes + canvas_bytes) / 1e6
        ok = "OK" if worst < 6e-2 else "FAIL"
        print(f"regions={a.regions} world_size={world_size} dim={a.dim} seq={a.seq_len} "
              f"batch={a.batch} rank_r={rr} layers/region={a.layers_per_region} phase1={a.phase1} "
              f"microbatches={a.microbatches}", flush=True)
        print(f"correctness: worst rel {worst:.4f} ({ok})", flush=True)
        print(f"sharded fwd+bwd: {step_ms:8.2f} ms/step | peak mem {float(peak_t.item()):8.1f} MB | "
              f"cross-GPU comm/step ~{comm_mb:.2f} MB (state chain + A_k + canvas)", flush=True)
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["mamba3", "transformer"], default="transformer")
    p.add_argument("--regions", type=int, default=4)
    p.add_argument("--world-size", type=int, default=torch.cuda.device_count())
    p.add_argument("--layers-per-region", type=int, default=2)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--headdim", type=int, default=64)
    p.add_argument("--d-state", type=int, default=128)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--lowering", choices=["native", "autograd"], default="native")
    p.add_argument("--phase1", choices=["native", "forward"], default="forward")
    p.add_argument("--microbatches", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=29617)
    a = p.parse_args()
    mp.spawn(worker, args=(a.world_size, a), nprocs=a.world_size, join=True)


if __name__ == "__main__":
    main()
