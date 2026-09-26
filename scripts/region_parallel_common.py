"""Helpers shared by the region-parallel step (scripts/region_parallel_step.py):
model construction for the four-device rows, barrier timing, the worst relative
gradient error against a reference, and the shared-parameter reduction."""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from backbones.general import BackboneSpec
from interfaces.vector_mlp import VectorMLPInterface
from models.lbi_language_model import LBILanguageModel

# The paper's region shape for every four-device row and profile: Mamba-3
# state 128, head dim 64, chunk 64, expansion 2, one group; Transformer conv 4.
PAPER_SHAPE = dict(d_state=128, headdim=64, chunk_size=64, expand=2, ngroups=1, d_conv=4)
UPDATE_SCALE_INIT = 0.5   # interface update-scale initialization (tanh(0.5))
PARITY_GATE_REL_L2 = 2e-2  # relative L2 gate of the four-device correctness check


def build_model(a, dev):
    interface = VectorMLPInterface(
        feature_dim=a.dim, num_regions=a.regions, interface_width=a.rank,
        interface_map_hidden_dim=a.dim, update_scale_init=UPDATE_SCALE_INIT,
    )
    if a.backbone == "transformer":
        spec = BackboneSpec(
            name="transformer", dim=a.dim, layers=a.regions * a.layers_per_region,
            n_heads=a.dim // a.headdim, d_conv=PAPER_SHAPE["d_conv"],
        )
    elif a.backbone == "hybrid":
        layers = a.regions * a.layers_per_region
        if layers % 4 != 0:
            raise ValueError("hybrid requires total layers divisible by 4 (3x mamba3 + 1x transformer pattern)")
        spec = BackboneSpec(
            name="hybrid", dim=a.dim, layers=layers,
            layer_types=("mamba3", "mamba3", "mamba3", "transformer") * (layers // 4),
            d_state=a.d_state, expand=PAPER_SHAPE["expand"], headdim=a.headdim,
            ngroups=PAPER_SHAPE["ngroups"], chunk_size=a.chunk_size,
            n_heads=a.dim // a.headdim, d_conv=PAPER_SHAPE["d_conv"],
        )
    else:
        spec = BackboneSpec(
            name="mamba3", dim=a.dim, layers=a.regions * a.layers_per_region,
            d_state=a.d_state, expand=PAPER_SHAPE["expand"], headdim=a.headdim,
            ngroups=PAPER_SHAPE["ngroups"], chunk_size=a.chunk_size,
        )
    return LBILanguageModel(
        vocab_size=a.vocab, layers_per_region=a.layers_per_region, backbone_spec=spec, interface=interface,
    ).to(device=dev, dtype=torch.bfloat16)


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


def _parity(grad_map, ref_map, dev):
    """Gradient parity against a reference map: the relative L2 error of the
    full gradient vector (every parameter, summed across ranks), the worst
    per-parameter max-abs ratio across ranks, and that parameter's name.
    Every reference gradient must be present on some rank."""
    have = torch.zeros(len(ref_map), device=dev)
    names = sorted(n for n, r in ref_map.items() if r is not None)
    diff_sq = torch.zeros((), device=dev, dtype=torch.float64)
    ref_sq = torch.zeros((), device=dev, dtype=torch.float64)
    worst, worst_name = 0.0, ""
    rank = dist.get_rank()
    for i, n in enumerate(names):
        g = grad_map.get(n)
        if g is None:
            continue
        have[i] = 1.0
        r = ref_map[n]
        d = (g.float() - r.float())
        rel = float(d.abs().max() / (r.float().abs().max() + 1e-9))
        if rel > worst:
            worst, worst_name = rel, n
        # Region parameters live on one rank; replicated ones are counted on rank 0.
        if n.startswith("region_backend.") or rank == 0:
            diff_sq = diff_sq + d.double().square().sum()
            ref_sq = ref_sq + r.float().double().square().sum()
    dist.all_reduce(have, op=dist.ReduceOp.MAX)
    missing = [names[i] for i in range(len(names)) if have[i] < 0.5]
    if missing:
        raise RuntimeError(f"gradients missing on every rank: {missing[:8]}")
    dist.all_reduce(diff_sq, op=dist.ReduceOp.SUM)
    dist.all_reduce(ref_sq, op=dist.ReduceOp.SUM)
    worst_t = torch.tensor([worst], device=dev)
    dist.all_reduce(worst_t, op=dist.ReduceOp.MAX)
    if float(worst_t.item()) > worst:
        worst_name = ""
    rel_l2 = float((diff_sq.sqrt() / ref_sq.sqrt().clamp_min(1e-30)).item())
    return rel_l2, float(worst_t.item()), worst_name


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
