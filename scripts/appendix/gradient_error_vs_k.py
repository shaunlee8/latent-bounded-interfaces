"""The gradient parity table (Appendix D): gradient error of the scan engine
against autograd as the region count K, dtype, and construction vary; one JSON
row per cell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from backward import ScanADEngine
from scripts.region_parallel_common import PAPER_SHAPE
from train.config import LBITrainingConfig
from train.model_builders import build_lbi_model

VOCAB_SIZE = 256  # random-token instances; the readout width does not enter the parity


def _make_cfg(args: argparse.Namespace, regions: int, dtype_name: str) -> LBITrainingConfig:
    return LBITrainingConfig(
        variants="lbi",
        backbone=args.backbone,
        dtype=dtype_name,
        layers=args.region_size * regions,
        dim=args.dim,
        d_state=PAPER_SHAPE["d_state"],
        headdim=PAPER_SHAPE["headdim"],
        region_size=args.region_size,
        message_dim=args.rank,
        message_hidden_dim=0,
        chunk_size=PAPER_SHAPE["chunk_size"],
        vocab_size=VOCAB_SIZE,
        tie_embeddings=False,  # the measured instances use an untied readout
        seq_len=args.seq_len,
        batch_size=args.batch,
        device=args.device,
    )


def _grad_metrics(ref: dict[str, torch.Tensor], got: dict[str, torch.Tensor]) -> dict:
    """Per-parameter relative errors (worst / median), the relative L2 error and
    cosine of the full concatenated gradient vector, the max-abs error, and the
    number of parameters whose scan-engine gradient is missing or non-finite
    (those are excluded from the other metrics and must be reported)."""
    rels, flat_r, flat_g = [], [], []
    max_abs = torch.tensor(0.0)
    missing, nonfinite = 0, 0
    for name, g_ref in ref.items():
        g = got.get(name)
        if g is None:
            missing += 1
            continue
        g_ref = g_ref.float()
        g = g.float()
        if not bool(torch.isfinite(g).all()):
            nonfinite += 1
            continue
        max_abs = torch.maximum(max_abs, (g - g_ref).abs().max().cpu())
        denom = float(g_ref.norm()) + 1e-30
        rels.append(float((g - g_ref).norm()) / denom)
        flat_r.append(g_ref.flatten())
        flat_g.append(g.flatten())
    r = torch.cat(flat_r)
    g = torch.cat(flat_g)
    cos = float(torch.dot(r, g) / (r.norm() * g.norm() + 1e-30))
    rels_t = torch.tensor(rels)
    return {
        "max_abs": float(max_abs),
        "worst_rel_l2": float(rels_t.max()),
        "median_rel_l2": float(rels_t.median()),
        "rel_l2_full": float((g - r).norm() / (r.norm() + 1e-30)),
        "cosine": cos,
        "n_params": len(ref),
        "n_missing": missing,
        "n_nonfinite": nonfinite,
    }


def run_cell(args: argparse.Namespace, regions: int, dtype: torch.dtype, mode: str) -> dict:
    torch.manual_seed(args.seed)
    dtype_name = {torch.float32: "float32", torch.bfloat16: "bfloat16"}[dtype]
    cfg = _make_cfg(args, regions, dtype_name)
    device = torch.device(args.device)
    model = build_lbi_model(cfg).to(device=device, dtype=dtype)
    model.region_backend.forward_mode_use_kernel = True
    ids = torch.randint(0, VOCAB_SIZE, (args.batch, args.seq_len), device=device)

    def loss_of(logits: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
        )

    model.zero_grad(set_to_none=True)
    loss_of(model(ids)).backward()
    ref = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    engine_cfg = SimpleNamespace(
        native_backward=True,
        interface_jacobian_mode=mode,
    )
    engine = ScanADEngine.from_config(engine_cfg)
    logits, cache = model.forward_with_cache(ids, native_backward=True)
    result = engine.backward(model=model, loss=loss_of(logits), cache=cache)
    got = {n: v for n, v in result.grad_map.items() if v is not None}

    row = {
        "regions": regions,
        "layers": cfg.layers,
        "dtype": str(dtype).split(".")[-1],
        "mode": mode,
        "backbone": args.backbone,
        "dim": args.dim,
        "rank": args.rank,
    }
    row.update(_grad_metrics(ref, got))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mamba3")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--region-size", type=int, default=2)
    ap.add_argument("--regions", type=int, nargs="+", default=[2, 3, 4, 7, 10, 14])
    ap.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"])
    ap.add_argument("--modes", nargs="+", default=["forward", "graph"])
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    rows = []
    for mode in args.modes:
        for name in args.dtypes:
            for k in args.regions:
                try:
                    row = run_cell(args, k, dt[name], mode)
                except Exception as exc:  # keep the sweep alive; record the cell
                    row = {"regions": k, "dtype": name, "mode": mode, "error": repr(exc)[:200]}
                print(json.dumps(row), flush=True)
                rows.append(row)
                torch.cuda.empty_cache()
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
