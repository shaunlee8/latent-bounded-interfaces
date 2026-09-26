"""The engine rows and the construction constant of Section 4.2 and Appendix B:
ms/step and peak memory per engine (autograd, and the scan engine with the
forward-mode, native reverse-mode, and graph constructions), plus the
standalone A_k construction time."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from backward import ScanADEngine
from scripts.region_parallel_common import PAPER_SHAPE
from train.config import LBITrainingConfig
from train.model_builders import build_lbi_model


def _cfg(args: argparse.Namespace) -> LBITrainingConfig:
    return LBITrainingConfig(
        variants="lbi",
        backbone=args.backbone,
        dtype="bfloat16",
        layers=args.layers,
        dim=args.dim,
        d_state=PAPER_SHAPE["d_state"],
        headdim=PAPER_SHAPE["headdim"],
        region_size=2,
        message_dim=args.rank,
        message_hidden_dim=0,
        chunk_size=PAPER_SHAPE["chunk_size"],
        vocab_size=32000,
        tie_embeddings=False,  # the measured instances use an untied readout
        seq_len=args.seq_len,
        batch_size=args.batch,
        device="cuda",
    )


def _loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
    )


def _timed(fn, iters: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3 / iters
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    return ms, peak_gb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mamba3")
    ap.add_argument("--layers", type=int, default=18)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--direction-chunk", type=int, default=0,
                    help="forward-mode A_k construction in chunks of this many "
                         "directions (0 = all r at once, the measured default)")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    cfg = _cfg(args)
    model = build_lbi_model(cfg).to(device="cuda", dtype=torch.bfloat16)
    model.region_backend.forward_mode_use_kernel = True
    ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len), device="cuda")
    rows = []

    def autograd_step() -> None:
        model.zero_grad(set_to_none=True)
        _loss(model(ids), ids).backward()

    ms, peak = _timed(autograd_step, args.iters, args.warmup)
    rows.append({"engine": "autograd", "ms_per_step": round(ms, 1), "peak_gb": round(peak, 2)})
    print(json.dumps(rows[-1]), flush=True)

    for mode in ("forward", "native", "graph"):
        engine = ScanADEngine.from_config(
            SimpleNamespace(
                native_backward=True,
                interface_jacobian_mode=mode,
            )
        )

        if args.direction_chunk and hasattr(engine.pullback_provider, "direction_chunk"):
            engine.pullback_provider.direction_chunk = int(args.direction_chunk)

        def scan_step() -> None:
            model.zero_grad(set_to_none=True)
            logits, cache = model.forward_with_cache(ids, native_backward=True)
            engine.backward(model=model, loss=_loss(logits, ids), cache=cache)

        try:
            ms, peak = _timed(scan_step, args.iters, args.warmup)
            row = {"engine": f"scan/{mode}", "ms_per_step": round(ms, 1), "peak_gb": round(peak, 2)}

            # Standalone construction: forward once, time the A_k materialization.
            logits, cache = model.forward_with_cache(ids, native_backward=True)

            def construct() -> None:
                engine.pullback_provider.materialize_state_jacobian_t(model=model, cache=cache)

            cms, cpeak = _timed(construct, args.iters, args.warmup)
            row.update({"construction_ms": round(cms, 1), "construction_peak_gb": round(cpeak, 2)})
        except Exception as exc:
            row = {"engine": f"scan/{mode}", "error": repr(exc)[:200]}
        rows.append(row)
        print(json.dumps(row), flush=True)
        torch.cuda.empty_cache()

    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
