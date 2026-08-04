"""Multi-process (NCCL) region-parallel native backward.

One process per GPU -> each has its own GIL and dispatches only its regions'
kernels, removing the single-process serial-dispatch ceiling. Regions are
round-robin assigned to ranks; the model is replicated (weights broadcast from
rank 0 so all ranks are identical). Each rank:

  * runs the forward locally (redundant, replicated) under native_backward+trim,
  * Phase 1: computes A_k for its regions; all-reduce shares the rank-r Jacobians,
  * Phase 2: runs the O(K) suffix scan locally,
  * Phase 3: computes its regions' param grads + canvas/shared partials,
  * reduces: all-reduce the canvas cotangent and shared-param partials; region
    param grads stay local (no gradient all-reduce).

Region param grads live on their owning rank (as in real region-parallel
training). Each rank self-verifies its own regions + the reduced quantities against
a locally computed sequential reference. Timing compares the distributed backward
(with collectives) to the single-rank sequential backward.

Run: torchrun --standalone --nproc_per_node=4 benchmarks/nccl_region_parallel.py [args]
  or: python benchmarks/nccl_region_parallel.py [args]   (uses mp.spawn)
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

from backbones.general import BackboneSpec
from backward import NativeInterfacePullbackProvider, NativeLocalVJPProvider, ScanADEngine
from backward.local_vjp import native_initial_backward, native_region_backward, reduce_region_results
from backward.pullbacks import interface_state_jacobian_t_for_region
from backward.suffix_scan import propagate_state_adjoint_from_last_region_input
from interfaces import VectorMLPInterface
from models.lbi_language_model import LBILanguageModel


def build_model(a, dev):
    interface = VectorMLPInterface(
        feature_dim=a.dim, num_regions=a.regions, interface_width=a.rank,
        interface_map_hidden_dim=a.dim, update_scale_init=0.5,
    )
    if a.backbone == "transformer":
        spec = BackboneSpec(
            name="transformer", dim=a.dim, layers=a.regions * a.layers_per_region,
            n_heads=a.dim // a.headdim, d_conv=4,
        )
    else:
        spec = BackboneSpec(
            name="mamba3", dim=a.dim, layers=a.regions * a.layers_per_region,
            d_state=a.d_state, expand=2, headdim=a.headdim, ngroups=1, chunk_size=a.chunk_size,
        )
    model = LBILanguageModel(
        vocab_size=a.vocab, layers_per_region=a.layers_per_region, backbone_spec=spec, interface=interface,
    ).to(device=dev, dtype=torch.bfloat16)
    if a.lowering == "autograd" and a.backbone == "mamba3":
        from backends import RegionLocalAutogradMamba3Lowering

        model.region_backend.lowering = RegionLocalAutogradMamba3Lowering()
    return model


def compute_region_jacobians(model, region_caches, my_regions, A, phase1_mode):
    """Phase 1: A_k for my regions. `native` = the reverse-mode native
    pullback; `forward` = the forward-mode construction through the backend's
    region_output_jvp kernel path."""
    if phase1_mode == "forward":
        from backward.pullbacks import interface_state_jacobian_for_region_forward

        with torch.no_grad():
            for k in my_regions:
                rc = region_caches[k]

                def _jvp(basis, _rc=rc):
                    return model.region_backend.region_output_jvp(
                        cache=_rc.backend_cache,
                        region_input_tangent_basis=basis, pooled=True)

                A[k] = interface_state_jacobian_for_region_forward(
                    model=model, region_cache=rc, region_output_jvp=_jvp,
                ).to(A.dtype)
    else:
        for k in my_regions:
            A[k] = interface_state_jacobian_t_for_region(
                model=model, region_cache=region_caches[k])


def distributed_backward(model, loss, cache, my_regions, num_regions, dev,
                         phase1_mode="native"):
    provider = NativeLocalVJPProvider()
    grad_map = provider.new_grad_map(model)
    states = list(cache["states"])
    region_caches = cache["region_caches"]

    # Glue (local, redundant): readout head, seed, last-region readout cotangent.
    provider.store_output_head_grads(model=model, loss=loss, grad_map=grad_map, cache=cache)
    g_last = provider.state_adjoint_from_loss(model=model, loss=loss, state=states[-2], cache=cache)
    last_g_ro = torch.autograd.grad(loss, region_caches[-1].region_output, retain_graph=True)[0]

    # Phase 1: A_k for my regions; all-reduce so every rank has all A_k (rank-r, tiny).
    bsz, rr = states[-2].shape
    A = torch.zeros(num_regions, bsz, rr, rr, device=dev, dtype=states[-2].dtype)
    compute_region_jacobians(model, region_caches, my_regions, A, phase1_mode)
    dist.all_reduce(A, op=dist.ReduceOp.SUM)

    # Phase 2: suffix scan (local).
    sjt = [A[k] for k in range(num_regions)]
    g_last = g_last.to(dtype=sjt[0].dtype)
    gsi = propagate_state_adjoint_from_last_region_input(sjt, g_last, num_regions=num_regions)

    # Initial-state glue (local).
    init_grads, canvas_init = native_initial_backward(
        model=model, state0=states[0], state0_adjoint=gsi[0], cache=cache
    )
    for n, g in init_grads.items():
        if n in grad_map:
            grad_map[n] = g

    # Phase 3: my regions (param grads stay local).
    results = []
    for k in my_regions:
        g_ro = last_g_ro if k == num_regions - 1 else None
        results.append(native_region_backward(
            model=model, loss=None, states=None, region_index=k, num_regions=num_regions,
            state_adjoints=gsi, cache=cache, g_region_output=g_ro,
        ))
    reduced = reduce_region_results(results)
    for n, g in reduced.param_grads.items():
        if n in grad_map:
            grad_map[n] = g

    # Reduce canvas cotangent across ranks; add the (replicated) initial
    # contribution. The collective needs one dtype on every rank: the last
    # region's cotangent arrives fp32 when the region output rides an fp32
    # residual stream, while other regions produce bf16.
    canvas = reduced.canvas_cotangent
    if canvas is None:
        canvas = torch.zeros_like(cache["canvas_features"])
    canvas = canvas.to(dtype=cache["canvas_features"].dtype)
    dist.all_reduce(canvas, op=dist.ReduceOp.SUM)
    if canvas_init is not None:
        canvas = canvas + canvas_init

    shared = _reduce_shared(model, reduced.shared_partials, dev)

    provider._canvas_grad = canvas
    provider._shared_grads = shared
    provider.store_shared_canvas_grads(model=model, loss=loss, grad_map=grad_map, cache=cache)
    return grad_map


def _time_barrier(fn, dev, warmup, iters):
    for _ in range(warmup):
        fn(); torch.cuda.synchronize(dev); dist.barrier()
    ts = []
    for _ in range(iters):
        dist.barrier(); torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        fn(); torch.cuda.synchronize(dev); dist.barrier()
        ts.append((time.perf_counter() - t0) * 1000)
    return sum(ts) / len(ts)


def _worst_rel(grad_map, ref_map, dev):
    """Worst relative gradient error vs a reference map, maxed across ranks."""
    worst = 0.0
    for n, g in grad_map.items():
        r = ref_map.get(n)
        if g is None or r is None:
            continue
        rel = (g.float() - r.float()).abs().max() / (r.float().abs().max() + 1e-9)
        worst = max(worst, float(rel))
    worst_t = torch.tensor([worst], device=dev)
    dist.all_reduce(worst_t, op=dist.ReduceOp.MAX)
    return float(worst_t.item())


def _reduce_shared(model, partials, dev):
    """All-reduce the shared-parameter partial grads (zeros where absent)."""
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    shared = {}
    for p in model.interface.shared_vjp_parameters():
        name = name_by_id[id(p)]
        part = partials.get(name)
        buf = (part if part is not None else torch.zeros_like(p)).to(dev)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        shared[name] = buf
    return shared


def worker(rank, world_size, a):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(a.port))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    torch.manual_seed(a.seed)

    model = build_model(a, dev)
    if a.phase1 == "forward":
        model.region_backend.forward_mode_use_kernel = True
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    input_ids = torch.randint(0, a.vocab, (a.batch, a.seq_len), device=dev)
    dist.broadcast(input_ids, src=0)

    num_regions = a.regions
    my_regions = list(range(rank, num_regions, world_size))

    # Sequential reference (local) for correctness.
    # Cache trimming is a Mamba-3 backend capability (the transformer backend
    # keeps its per-layer caches; they are the JVP operating point).
    trim = a.backbone == "mamba3"
    logits_r, cache_r = model.forward_with_cache(input_ids, native_backward=True, trim_region_cache=trim)
    ref = ScanADEngine(
        pullback_provider=NativeInterfacePullbackProvider(), local_vjp_provider=NativeLocalVJPProvider()
    ).backward(model=model, loss=logits_r.square().mean(), cache=cache_r)

    # Distributed backward once for correctness.
    logits, cache = model.forward_with_cache(input_ids, native_backward=True, trim_region_cache=trim)
    loss = logits.square().mean()
    gm = distributed_backward(model, loss, cache, my_regions, num_regions, dev,
                              phase1_mode=a.phase1)
    worst = _worst_rel(gm, ref.grad_map, dev)

    # Timing: distributed (parallel) vs single-rank sequential.
    logits_s, cache_s = model.forward_with_cache(input_ids, native_backward=True, trim_region_cache=trim)
    loss_s = logits_s.square().mean()
    engine = ScanADEngine(
        pullback_provider=NativeInterfacePullbackProvider(), local_vjp_provider=NativeLocalVJPProvider()
    )
    logits_d, cache_d = model.forward_with_cache(input_ids, native_backward=True, trim_region_cache=trim)
    loss_d = logits_d.square().mean()

    par_ms = _time_barrier(
        lambda: distributed_backward(model, loss_d, cache_d, my_regions, num_regions, dev,
                                     phase1_mode=a.phase1), dev, a.warmup, a.iters
    )
    seq_ms = _time_barrier(
        lambda: engine.backward(model=model, loss=loss_s, cache=cache_s), dev, a.warmup, a.iters
    )

    # Phase-1-only timing (the component the forward-mode construction moved).
    _states_d = list(cache_d["states"])
    A_buf = torch.zeros(num_regions, _states_d[-2].shape[0], a.rank, a.rank,
                        device=dev, dtype=_states_d[-2].dtype)

    def phase1_only():
        compute_region_jacobians(model, cache_d["region_caches"], my_regions, A_buf, a.phase1)
        dist.all_reduce(A_buf, op=dist.ReduceOp.SUM)

    p1_ms = _time_barrier(phase1_only, dev, a.warmup, a.iters)

    # Full step (forward + region-parallel backward) for the pipeline-parallel
    # comparison. NOTE: the forward here is redundant per rank (each rank forwards
    # the whole model); a real LBI forward pipeline (rank-r message across regions)
    # would be cheaper, so this over-counts LBI's forward.
    def full_step():
        lg, c = model.forward_with_cache(input_ids, native_backward=True, trim_region_cache=trim)
        distributed_backward(model, lg.square().mean(), c, my_regions, num_regions, dev)

    torch.cuda.reset_peak_memory_stats(dev)
    full_ms = _time_barrier(full_step, dev, a.warmup, a.iters)
    peak = torch.cuda.max_memory_allocated(dev) / 1e6
    peak_t = torch.tensor([peak], device=dev)
    dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)

    if rank == 0:
        rr = a.rank
        # Cross-GPU comm per step: all-reduce A_k (K x [B,r,r]) + canvas ([B,L,D]) + shared.
        a_bytes = num_regions * a.batch * rr * rr * 2
        canvas_bytes = a.batch * a.seq_len * a.dim * 2
        comm_mb = (a_bytes + canvas_bytes) / 1e6
        print(f"regions={num_regions} world_size={world_size} dim={a.dim} seq={a.seq_len} batch={a.batch} "
              f"rank_r={rr} layers/region={a.layers_per_region} lowering={a.lowering} phase1={a.phase1}", flush=True)
        print(f"phase 1 (A_k, {a.phase1}) distributed: {p1_ms:8.2f} ms", flush=True)
        print(f"correctness: worst rel {worst:.4f} "
              f"({'OK' if worst < 6e-2 else 'FAIL'})", flush=True)
        print(f"backward: sequential (1 rank) {seq_ms:8.2f} ms | region-parallel ({world_size} ranks) "
              f"{par_ms:8.2f} ms | speedup {seq_ms / par_ms:.2f}x", flush=True)
        print(f"full step (fwd+bwd): {full_ms:8.2f} ms | peak mem {float(peak_t.item()):8.1f} MB | "
              f"cross-GPU comm/step ~{comm_mb:.2f} MB (rank-r A_k + canvas), bubble 0%", flush=True)
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["mamba3", "transformer"], default="mamba3")
    p.add_argument("--regions", type=int, default=4)
    p.add_argument("--world-size", type=int, default=torch.cuda.device_count())
    p.add_argument("--layers-per-region", type=int, default=6)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--headdim", type=int, default=64)
    p.add_argument("--d-state", type=int, default=128)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--lowering", choices=["native", "autograd"], default="autograd")
    p.add_argument("--phase1", choices=["native", "forward"], default="native")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=29513)
    a = p.parse_args()

    if "RANK" in os.environ:  # launched by torchrun
        worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), a)
    else:
        mp.spawn(worker, args=(a.world_size, a), nprocs=a.world_size, join=True)


if __name__ == "__main__":
    main()
