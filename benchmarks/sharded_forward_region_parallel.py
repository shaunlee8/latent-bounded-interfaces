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
from backward.suffix_scan import (
    propagate_state_adjoint_affine,
    propagate_state_adjoint_from_last_region_input,
)
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
            condition = model.interface.decode(s_in, rank, canvas_features=canvas[sl].detach())
            canvas_read = model.viewed_canvas(canvas[sl].detach(), rank)
            region_input = canvas_read + (condition if condition.dim() == 3 else condition.unsqueeze(1))
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

    # Boundary-state taps read every region's output state at the readout
    # rank: share them with one tiny all-reduce ([K, B, r]).
    taps_on = model.state_readout is not None
    s_all = None
    if taps_on:
        s_all = torch.zeros(num_regions, B, rr, device=dev, dtype=state0.dtype)
        for j in range(m):
            sl = slice(j * mb, (j + 1) * mb)
            s_all[rank, sl] = rcs[j].state_out
        dist.all_reduce(s_all, op=dist.ReduceOp.SUM)

    # Output readout: the readout stream adds tanh-gated region outputs. The
    # sum is linear, so each rank contributes its own gated output to ONE
    # [B, L, D] all-reduce (no per-region gather). The reduce is issued async
    # and overlaps the A_k construction below; only the readout rank waits.
    out_on = model.output_readout_gates is not None
    gated_sum = None
    gate_own = None
    gated_work = None
    if out_on:
        gate_own = torch.tanh(model.output_readout_gates[rank].detach()).to(dtype=state0.dtype)
        gated_sum = torch.zeros_like(canvas, dtype=state0.dtype)
        for j in range(m):
            sl = slice(j * mb, (j + 1) * mb)
            gated_sum[sl] = gate_own * rcs[j].region_output.to(dtype=state0.dtype)
        gated_work = dist.all_reduce(gated_sum, op=dist.ReduceOp.SUM, async_op=True)

    # ---- A_k, per microbatch, off the chain's critical path ----
    # Token-wise interfaces consume the projected tangent contraction; the
    # broadcast interface keeps the fused meanpool epilogue.
    tokenwise = bool(getattr(model.interface.spec, "condition_is_tokenwise", False))
    A = torch.zeros(num_regions, B, rr, rr, device=dev, dtype=state0.dtype)
    with torch.no_grad():
        for j in range(m):
            sl = slice(j * mb, (j + 1) * mb)
            projection = inner_map = None
            if tokenwise:
                projection, inner_map = model.interface.update_tangent_projection(rcs[j])

            def _jvp(basis, _bc=rcs[j].backend_cache, _proj=projection, _inner=inner_map,
                     tangent_token_start=0):
                return model.region_backend.region_output_jvp(
                    cache=_bc, region_input_tangent_basis=basis,
                    pooled=_proj is None, output_projection=_proj, output_inner=_inner,
                    tangent_token_start=tangent_token_start,
                )

            A[rank, sl] = interface_state_jacobian_for_region_forward(
                model=model, region_cache=rcs[j], region_output_jvp=_jvp
            ).to(A.dtype)
    # Phase-1 reduce, issued async: the [K, B, r, r] all-reduce overlaps the
    # readout-rank loss/seed work and the tap-source pullbacks below.
    a_work = dist.all_reduce(A, op=dist.ReduceOp.SUM, async_op=True)

    # ---- loss + seed adjoints (last rank), broadcast ----
    grad_map = provider.new_grad_map(model)
    loss = None
    last_g_ro = [None] * m
    last_comb = [None] * m
    g_last = torch.empty(B, rr, device=dev, dtype=state0.dtype)
    b_all = torch.zeros(num_regions, B, rr, device=dev, dtype=state0.dtype) if taps_on else None
    g_canvas_taps = None
    if rank == num_regions - 1:
        if gated_work is not None:
            gated_work.wait()
        losses, ro_inputs, tap_lists = [], [], []
        for j in range(m):
            sl = slice(j * mb, (j + 1) * mb)
            ro = rcs[j].region_output.detach().requires_grad_(True)
            rcs[j].region_output = ro
            readout_stream = ro
            taps_j: list[torch.Tensor] = []
            if taps_on:
                taps_j = [s_all[k, sl].detach().requires_grad_(True) for k in range(num_regions)]
                for term in model.state_readout.contributions(taps_j, canvas[sl]):
                    readout_stream = readout_stream + (term if term.dim() == 3 else term.unsqueeze(1))
            if out_on:
                # The gated-output sum enters detached (the severed-leaf analog):
                # its cotangents route through the tap machinery below, while
                # the live `ro` leaf keeps its coefficient-1 readout path.
                readout_stream = readout_stream + gated_sum[sl].detach().to(dtype=ro.dtype)
            logits = model.readout(readout_stream, canvas=model.canvas)
            losses.append(logits.float().square().mean())
            ro_inputs.append(ro)
            tap_lists.append(taps_j)
        # Equal chunks: the mean of chunk means is the full-batch mean.
        loss = sum(losses) / m
        provider.store_output_head_grads(model=model, loss=loss, grad_map=grad_map,
                                         cache={"region_caches": rcs})
        for j in range(m):
            sl = slice(j * mb, (j + 1) * mb)
            last_g_ro[j] = torch.autograd.grad(loss, ro_inputs[j], retain_graph=True)[0]
            if out_on:
                # Fused seed: the readout path (coefficient 1) and the last
                # region's own tap path (coefficient tanh(gate)) pull back to
                # the input state through one decode^T J^T pass.
                last_comb[j] = (1.0 + gate_own.to(dtype=last_g_ro[j].dtype)) * last_g_ro[j]
                g_ri = model.region_backend.input_pullback_basis(
                    cache=rcs[j].backend_cache,
                    output_cotangent_basis=last_comb[j].unsqueeze(1))
                dec = model.interface.apply_decode_jacobian_t_to_state_input(
                    region_cache=rcs[j], region_input_cotangent_basis=g_ri)
                g_last[sl] = dec["g_state_input"].squeeze(1).detach().to(dtype=g_last.dtype)
            else:
                g_last[sl] = provider.state_adjoint_from_loss(
                    model=model, loss=loss, state=s_ins[j],
                    cache={"region_caches": [rcs[j]]})
            if taps_on:
                tap_grads = torch.autograd.grad(loss, tap_lists[j], retain_graph=True)
                for k in range(num_regions):
                    b_all[k, sl] = tap_grads[k].to(dtype=b_all.dtype)
        if taps_on:
            # The taps' canvas-query cotangent (folded into the canvas reduction).
            g_canvas_taps = torch.autograd.grad(
                loss, canvas, retain_graph=True, allow_unused=True
            )[0]
    dist.broadcast(g_last, src=num_regions - 1)
    if taps_on:
        dist.broadcast(b_all, src=num_regions - 1)

    # Output-readout backward: one [B, L, D] broadcast of the stream cotangent;
    # every rank forms its own tap cotangent, gate grad, and input-side scan
    # source locally (the last rank's source already rode the fused seed).
    out_tap_grads = None
    b_out = None
    if out_on:
        g_stream = torch.empty_like(canvas, dtype=state0.dtype)
        if rank == num_regions - 1:
            for j in range(m):
                sl = slice(j * mb, (j + 1) * mb)
                g_stream[sl] = last_g_ro[j].to(dtype=g_stream.dtype)
        dist.broadcast(g_stream, src=num_regions - 1)
        out_tap_grads = [
            gate_own * g_stream[slice(j * mb, (j + 1) * mb)] for j in range(m)
        ]
        dot = torch.zeros((), device=dev, dtype=torch.float32)
        for j in range(m):
            sl = slice(j * mb, (j + 1) * mb)
            dot = dot + (g_stream[sl].float() * rcs[j].region_output.float()).sum()
        g_gates = torch.zeros(num_regions, device=dev, dtype=torch.float32)
        g_gates[rank] = dot * (1.0 - gate_own.float() ** 2)
        dist.all_reduce(g_gates, op=dist.ReduceOp.SUM)
        if "output_readout_gates" in grad_map:
            grad_map["output_readout_gates"] = g_gates.to(dtype=model.output_readout_gates.dtype)
        b_out = torch.zeros(num_regions, B, rr, device=dev, dtype=state0.dtype)
        if rank < num_regions - 1:
            for j in range(m):
                sl = slice(j * mb, (j + 1) * mb)
                g_ri = model.region_backend.input_pullback_basis(
                    cache=rcs[j].backend_cache,
                    output_cotangent_basis=out_tap_grads[j].to(
                        dtype=rcs[j].region_output.dtype).unsqueeze(1))
                dec = model.interface.apply_decode_jacobian_t_to_state_input(
                    region_cache=rcs[j], region_input_cotangent_basis=g_ri)
                b_out[rank, sl] = dec["g_state_input"].squeeze(1).detach().to(dtype=b_out.dtype)
        dist.all_reduce(b_out, op=dist.ReduceOp.SUM)

    # ---- Phase 2 suffix scan (local, tiny); Phase-1 reduce was issued async ----
    a_work.wait()
    if gated_work is not None:
        gated_work.wait()
    sjt = [A[k] for k in range(num_regions)]
    if taps_on or out_on:
        gsi = propagate_state_adjoint_affine(
            sjt, g_last.to(dtype=sjt[0].dtype),
            [b_all[k] for k in range(num_regions)] if taps_on else None,
            num_regions=num_regions,
            input_sources=[b_out[k] for k in range(num_regions)] if out_on else None,
        )
    else:
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
        if out_on and rank < num_regions - 1:
            # Interior ranks add their tap cotangent to the derived region-
            # output cotangent; the last rank's is fused into `last_comb`.
            tap_list = [None] * num_regions
            tap_list[rank] = out_tap_grads[j].to(dtype=rcs[j].region_output.dtype)
            cache_j["output_tap_grads"] = tap_list
        res_j = native_region_backward(
            model=model, loss=loss, states=None, region_index=rank,
            num_regions=num_regions, state_adjoints=[g[sl] for g in gsi],
            cache=cache_j, g_region_output=(last_comb[j] if out_on else last_g_ro[j]),
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
    if g_canvas_taps is not None:
        canvas_ct = canvas_ct + g_canvas_taps.to(dtype=canvas_ct.dtype)
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
    if model.output_readout_gates is not None:
        # Zero-init gates would make the readout path vacuously correct; move
        # them off zero so the parity check exercises the tap machinery.
        with torch.no_grad():
            model.output_readout_gates.add_(0.2 * torch.randn_like(model.output_readout_gates))
    if a.phase1 == "forward":
        model.region_backend.forward_mode_use_kernel = True
    if os.environ.get("LBI_TURN_COMPILE", "0") != "0":
        model.region_backend.compile_forward_stages = True
        os.environ.setdefault("LBI_INTERFACE_COMPILE", "1")
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
        # Boundary-state taps add the state gather + source broadcast (KB).
        tap_bytes = 2 * a.regions * a.batch * rr * 2 if a.canvas_state_readout else 0
        # Output readout: gated-sum all-reduce + stream-cotangent broadcast
        # ([B, L, D] each) + the tiny input-source reduce.
        readout_bytes = (
            (2 * a.batch * a.seq_len * a.dim + a.regions * a.batch * rr) * 2
            if a.canvas_output_readout else 0
        )
        comm_mb = (state_bytes + a_bytes + canvas_bytes + tap_bytes + readout_bytes) / 1e6
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
    p.add_argument("--interface", choices=["vector_mlp", "attentive", "chunked"],
                   default="vector_mlp")
    p.add_argument("--interface-chunks", type=int, default=4)
    p.add_argument("--interface-inclusive", action="store_true")
    p.add_argument("--canvas-state-readout", action="store_true")
    p.add_argument("--canvas-region-view", action="store_true")
    p.add_argument("--canvas-local-mixer", type=int, default=0)
    p.add_argument("--canvas-output-readout", action="store_true")
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
