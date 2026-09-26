"""The region-parallel training step of Table 3 (rank k owns region k). The
forward is a chain of [B, r] state messages, each rank running its region once
and constructing A_k by forward mode while the later ranks take their turns; the
backward is the seed broadcast from the last rank, the all-reduce of the A_k,
the local suffix scan, and each rank's region backward.

Run: python scripts/region_parallel_step.py [args]  (mp.spawn;
world size must equal --regions)"""

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

from backward import NativeInterfacePullbackProvider, NativeLocalVJPProvider, ScanADEngine
from backward.local_vjp import native_initial_backward, native_region_backward
from backward.pullbacks import interface_state_jacobian_for_region_forward
from backward.suffix_scan import propagate_state_adjoint_from_last_region_input
from scripts import comm_bytes
from scripts.region_parallel_common import (
    PAPER_SHAPE, PARITY_GATE_REL_L2, _parity, _reduce_shared, _time_barrier, build_model)
from models.lbi_language_model import LBIRegionCache


class CanvasReducer:
    """The canvas-cotangent collective across steps. With accum == 1 each
    microbatch's chunk is all-reduced asynchronously as soon as its region
    backward finishes; with accum == N the cotangents are summed locally and
    reduced once per N steps (fuse_shared folds the shared-parameter partials
    into that one collective)."""

    def __init__(self, accum: int = 1, fuse_shared: bool = False):
        self.accum = max(1, int(accum))
        self.fuse_shared = bool(fuse_shared)
        self._shared_meta = None   # [(name, shape, dtype, numel)]
        self._shared_buf = None    # fp32 flat accumulator
        self.calls = 0
        self.boundary = False
        self._buf = None
        self._init = None
        self._works = []

    def begin(self) -> None:
        self.calls += 1
        self.boundary = (self.calls % self.accum == 0)
        self._works = []

    def submit_shared(self, model, partials, dev) -> None:
        """Accumulate this step's shared-parameter partials (fuse_shared)."""
        if self._shared_meta is None:
            name_by_id = {id(p): n for n, p in model.named_parameters()}
            self._shared_meta = [(name_by_id[id(p)], tuple(p.shape), p.dtype, p.numel())
                                 for p in model.interface.shared_vjp_parameters()]
        flats = []
        for name, shape, dtype, numel in self._shared_meta:
            part = partials.get(name)
            flats.append((part.to(dev) if part is not None else torch.zeros(shape, device=dev, dtype=dtype)).reshape(-1).float())
        flat = torch.cat(flats) if flats else torch.zeros(0, device=dev)
        self._shared_buf = flat if self._shared_buf is None else self._shared_buf + flat

    def finish_shared(self, dev):
        """Reduced shared-parameter grads for this window (dict), or None
        inside a window; call after finish()."""
        if self._shared_meta is None:
            return None
        if self.accum > 1 and not self.boundary:
            return None
        buf = self._shared_buf
        self._shared_buf = None
        if buf is None or buf.numel() == 0:
            return {}
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        out, off = {}, 0
        for name, shape, dtype, numel in self._shared_meta:
            out[name] = buf[off:off + numel].reshape(shape).to(dtype); off += numel
        return out

    def submit(self, chunk) -> None:
        """A complete contiguous slice of this step's local cotangent."""
        if self.accum == 1:
            self._works.append(dist.all_reduce(chunk, op=dist.ReduceOp.SUM, async_op=True))

    def finish(self, canvas_ct, canvas_init):
        """Returns the reduced total for this step, or None inside a window."""
        if self.accum == 1:
            for w in self._works:
                w.wait()
            return canvas_ct if canvas_init is None else canvas_ct + canvas_init
        self._buf = canvas_ct if self._buf is None else self._buf + canvas_ct
        if canvas_init is not None:
            self._init = canvas_init if self._init is None else self._init + canvas_init
        if not self.boundary:
            return None
        dist.all_reduce(self._buf, op=dist.ReduceOp.SUM)
        total = self._buf if self._init is None else self._buf + self._init
        self._buf = None
        self._init = None
        return total


def sharded_step(model, provider, input_ids, rank, num_regions, dev, microbatches=1,
                 reducer=None, direction_chunk=0):
    """One region-sharded training step; returns (grad_map, loss-or-None).
    With microbatches > 1 the state chain runs fill-drain ([B/m, r] messages,
    chain wall ~ (m+K-1)/(m*K) * F_full); A_k construction runs after the
    chain loop, off the downstream ranks' critical path. `reducer` carries
    the canvas collective's accumulation state across steps."""
    if reducer is None:
        reducer = CanvasReducer(1)
    reducer.begin()
    m = microbatches
    trace_mem = os.environ.get("LBI_MEM_TRACE", "0") == "1" and rank == 0

    def _mem(tag):
        # LBI_MEM_TRACE=1: per-phase allocated / peak on rank 0 (GB).
        if trace_mem:
            torch.cuda.synchronize(dev)
            print(f"[mem] {tag:<28s} alloc {torch.cuda.memory_allocated(dev) / 2**30:6.2f} GB"
                  f"  peak since last reading {torch.cuda.max_memory_allocated(dev) / 2**30:6.2f} GB", flush=True)
            torch.cuda.reset_peak_memory_stats(dev)

    if trace_mem:
        pbytes = sum(p_.numel() * p_.element_size() for p_ in model.parameters())
        print(f"[mem] parameters (replicated)   {pbytes / 2**30:6.2f} GB", flush=True)
    _mem("step start")
    B = input_ids.shape[0]
    if B % m != 0:
        raise ValueError("batch must divide microbatches")
    mb = B // m

    # Canvas + initial state are small grad-connected graphs, replicated.
    canvas = model.canvas(input_ids)
    state0 = model.initial_state(canvas)
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
            canvas_read = model.viewed_canvas(canvas[sl].detach(), rank)
            region_input = canvas_read + condition.unsqueeze(1)
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

    _mem("after forward chain (caches)")
    # A_k per microbatch, off the chain's critical path, from the fused
    # mean-pooled region JVP.
    A = torch.zeros(num_regions, B, rr, rr, device=dev, dtype=state0.dtype)
    with torch.no_grad():
        for j in range(m):
            sl = slice(j * mb, (j + 1) * mb)

            def _jvp(basis, _bc=rcs[j].backend_cache):
                return model.region_backend.region_output_jvp(
                    cache=_bc, region_input_tangent_basis=basis, pooled=True,
                )

            A[rank, sl] = interface_state_jacobian_for_region_forward(
                model=model, region_cache=rcs[j], region_output_jvp=_jvp,
                direction_chunk=direction_chunk,
            ).to(A.dtype)
    # Phase-1 reduce, issued async: the [K, B, r, r] all-reduce overlaps the
    # readout rank's loss and seed work below.
    a_work = dist.all_reduce(A, op=dist.ReduceOp.SUM, async_op=True)
    _mem("after A_k construction")

    # ---- loss + seed adjoint (last rank), broadcast ----
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
            last_g_ro[j] = torch.autograd.grad(loss, ro_inputs[j], retain_graph=True)[0]
            g_last[sl] = provider.state_adjoint_from_loss(
                model=model, loss=loss, state=s_ins[j],
                cache={"region_caches": [rcs[j]]})
    dist.broadcast(g_last, src=num_regions - 1)

    # ---- Phase 2 suffix scan (local, tiny); the Phase-1 reduce was issued async ----
    a_work.wait()
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

    _mem("before region backward")
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
            model=model, loss=loss, region_index=rank,
            num_regions=num_regions, state_adjoints=[g[sl] for g in gsi],
            cache=cache_j, g_region_output=last_g_ro[j],
        )
        for n, g in res_j.param_grads.items():
            param_sums[n] = g if n not in param_sums else param_sums[n] + g
        for n, g in res_j.shared_partials.items():
            shared_sums[n] = g if n not in shared_sums else shared_sums[n] + g
        if res_j.canvas_cotangent is not None:
            canvas_ct[sl] = res_j.canvas_cotangent.to(dtype=canvas_ct.dtype)
        # Canvas cotangent chunk: one dtype on every rank before the collective.
        reducer.submit(canvas_ct[sl])
    for n, g in param_sums.items():
        if n in grad_map:
            grad_map[n] = g

    if reducer.fuse_shared:
        reducer.submit_shared(model, shared_sums, dev)
        canvas_total = reducer.finish(canvas_ct, canvas_init)
        shared = reducer.finish_shared(dev)
    else:
        shared = _reduce_shared(model, shared_sums, dev)
        canvas_total = reducer.finish(canvas_ct, canvas_init)
    _mem("after region backward")

    provider._shared_grads = shared
    if canvas_total is None:
        provider._canvas_grad = None
        return grad_map, loss
    provider._canvas_grad = canvas_total
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
    model.region_backend.forward_mode_use_kernel = True
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
                         microbatches=a.microbatches, reducer=CanvasReducer(1),
                         direction_chunk=a.direction_chunk)
    rel_l2, worst, worst_name = _parity(gm, ref.grad_map, dev)
    del ref
    torch.cuda.empty_cache()

    comm_bytes.install()
    comm_bytes.reset()
    torch.cuda.reset_peak_memory_stats(dev)
    if a.canvas_accum > 1 and (a.warmup % a.canvas_accum or a.iters % a.canvas_accum):
        raise ValueError("--warmup and --iters must be multiples of --canvas-accum so the "
                         "timed mean holds whole accumulation windows")
    if a.fuse_shared is None:
        a.fuse_shared = a.canvas_accum > 1
    reducer = CanvasReducer(a.canvas_accum, fuse_shared=a.fuse_shared)
    _step = lambda: sharded_step(model, provider, input_ids, rank, a.regions, dev,
                                 microbatches=a.microbatches, reducer=reducer,
                                 direction_chunk=a.direction_chunk)
    if a.timing == "block":
        # One device sync + barrier around the whole timed block instead of
        # around every step, so the window collective's cost lands inside the
        # measured time without a per-step synchronize.
        for _ in range(a.warmup):
            _step()
        torch.cuda.synchronize(dev); dist.barrier()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            _step()
        torch.cuda.synchronize(dev); dist.barrier()
        step_ms = (time.perf_counter() - t0) * 1000 / a.iters
    else:
        step_ms = _time_barrier(_step, dev, a.warmup, a.iters)
    counted = comm_bytes.total_bytes() / (a.warmup + a.iters)
    peak = torch.cuda.max_memory_allocated(dev) / 1e6
    peak_t = torch.tensor([peak], device=dev)
    dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)
    counted_t = torch.tensor([counted], device=dev, dtype=torch.float64)
    dist.all_reduce(counted_t, op=dist.ReduceOp.SUM)

    if rank == 0:
        rr = a.rank
        ok = "OK" if rel_l2 < PARITY_GATE_REL_L2 else "FAIL"
        print(f"regions={a.regions} world_size={world_size} dim={a.dim} seq={a.seq_len} "
              f"batch={a.batch} rank_r={rr} layers/region={a.layers_per_region} "
              f"microbatches={a.microbatches} canvas_accum={a.canvas_accum} "
              f"fuse_shared={int(a.fuse_shared)} timing={a.timing} "
              f"direction_chunk={a.direction_chunk} warmup={a.warmup} iters={a.iters}", flush=True)
        print(f"correctness: rel l2 {rel_l2:.2e} ({ok}); worst per-parameter max-abs ratio "
              f"{worst:.4f} ({worst_name or 'other rank'})", flush=True)
        print(f"sharded fwd+bwd: {step_ms:8.2f} ms/step | peak mem {float(peak_t.item()):8.1f} MB", flush=True)
        print(f"counted comm/step: {float(counted_t.item()) / 1e6:.2f} MB "
              f"(outbound bytes at the dist call boundary, summed over ranks)", flush=True)
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["mamba3", "transformer", "hybrid"], default="mamba3")
    p.add_argument("--regions", type=int, default=4)
    p.add_argument("--world-size", type=int, default=torch.cuda.device_count())
    p.add_argument("--layers-per-region", type=int, default=2)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--headdim", type=int, default=PAPER_SHAPE["headdim"])
    p.add_argument("--d-state", type=int, default=PAPER_SHAPE["d_state"])
    p.add_argument("--rank", "--interface-rank", dest="rank", type=int, default=16, help="interface rank r")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=PAPER_SHAPE["chunk_size"])
    p.add_argument("--microbatches", type=int, default=1)
    p.add_argument("--canvas-accum", "--window", dest="canvas_accum", type=int, default=1,
                   help="reduce the canvas cotangent once per N steps (gradient "
                        "accumulation on the canvas term); warmup and iters must be "
                        "multiples of N")
    p.add_argument("--fuse-shared", dest="fuse_shared", action="store_true", default=None,
                   help="fold the shared-parameter partial gradients into the canvas window: "
                        "one flat collective per window instead of one all-reduce per shared "
                        "parameter per step (exact under the sum); on by default whenever "
                        "--canvas-accum > 1, --no-fuse-shared restores the per-step reduce")
    p.add_argument("--no-fuse-shared", dest="fuse_shared", action="store_false")
    p.add_argument("--direction-chunk", type=int, default=0,
                   help="construct A_k in chunks of this many tangent directions "
                        "(0 = all r in one fused pass, the measured default)")
    p.add_argument("--timing", choices=["step", "block"], default="step",
                   help="step: device sync + barrier around every step; block: once "
                        "around the whole timed block (the window rows)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=29617)
    a = p.parse_args()
    mp.spawn(worker, args=(a.world_size, a), nprocs=a.world_size, join=True)


if __name__ == "__main__":
    main()
