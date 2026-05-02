from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import torch

from backbones.general import ReferenceLM, infer_message_hidden_dim
from models.native_region_interface import NativeRegionInterfaceModel
from train.train_region_interface import (
    RegionInterfaceConfig,
    _autocast_context,
    _build_corpora,
    _load_checkpoint,
    _make_backbone_spec,
    _next_token_loss,
    _resolve_device,
    _sample_batch_any,
)


def _run_sort_key(path: Path) -> tuple[int, int, str]:
    parent = path.parent.name
    if parent.startswith("dense_seed"):
        return (0, _parse_suffix_int(parent, "dense_seed", default=0), parent)
    if parent.startswith("lbi_r") and "_seed" in parent:
        rank = _parse_suffix_int(parent.split("_seed", 1)[0], "lbi_r", default=10**9)
        seed = _parse_suffix_int(parent.rsplit("_seed", 1)[-1], "", default=0)
        return (1, rank, f"{seed:08d}_{parent}")
    return (2, 0, str(path))


def _parse_suffix_int(text: str, prefix: str, *, default: int) -> int:
    try:
        return int(text[len(prefix) :])
    except ValueError:
        return default


def _discover_run_dirs(family_dir: Path) -> list[Path]:
    candidates = [
        path
        for path in family_dir.glob("*/*")
        if path.is_dir()
        and (path / "config.json").exists()
        and (path / "summary.json").exists()
        and path.name in {"backprop_ref", "native_region_interface"}
    ]
    return sorted(candidates, key=_run_sort_key)


def _load_config(run_dir: Path, *, args: argparse.Namespace) -> RegionInterfaceConfig:
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(RegionInterfaceConfig)}
    cfg = RegionInterfaceConfig(**{key: value for key, value in raw.items() if key in allowed})

    overrides: dict[str, Any] = {
        "eval_batches": int(args.eval_batches),
        "device": str(args.device),
    }
    if args.tokenizer_path:
        overrides["tokenizer_path"] = str(args.tokenizer_path)
    if args.batch_size is not None:
        overrides["batch_size"] = int(args.batch_size)
    if args.seq_len is not None:
        overrides["seq_len"] = int(args.seq_len)
    return replace(cfg, **overrides)


def _checkpoint_from_summary(run_dir: Path, checkpoint_name: str) -> Path:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    key = "best_checkpoint_path" if checkpoint_name == "best" else "latest_checkpoint_path"
    fallback_key = "latest_checkpoint_path" if checkpoint_name == "best" else "best_checkpoint_path"
    for candidate_key in (key, fallback_key):
        candidate = str(summary.get(candidate_key, "")).strip()
        if candidate:
            path = Path(candidate).expanduser()
            if path.exists():
                return path

    for filename in (f"{checkpoint_name}.pt", "best.pt", "latest.pt"):
        candidate = run_dir / "checkpoints" / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no {checkpoint_name}.pt checkpoint found for {run_dir}")


def _build_model(cfg: RegionInterfaceConfig) -> torch.nn.Module:
    backbone_spec = _make_backbone_spec(cfg)
    if cfg.regime == "backprop_ref":
        return ReferenceLM(
            vocab_size=cfg.vocab_size,
            backbone_spec=backbone_spec,
            tie_embeddings=getattr(cfg, "tie_embeddings", False),
        )
    if cfg.regime == "native_region_interface":
        return NativeRegionInterfaceModel(
            vocab_size=cfg.vocab_size,
            region_size=cfg.region_size,
            message_dim=cfg.message_dim,
            backbone_spec=backbone_spec,
            message_hidden_dim=infer_message_hidden_dim(backbone_spec, cfg.message_hidden_dim),
            message_scale_init=cfg.message_scale_init,
            tie_embeddings=getattr(cfg, "tie_embeddings", False),
        )
    raise ValueError(f"unsupported eval regime: {cfg.regime}")


def _safe_exp(value: float) -> float:
    try:
        return float(math.exp(value))
    except OverflowError:
        return float("inf")


def _evaluate_loaded_model(
    *,
    cfg: RegionInterfaceConfig,
    model: torch.nn.Module,
    val_corpus: Any,
    generator_seed: int,
    device: torch.device,
) -> dict[str, float]:
    generator_device = device if cfg.data_mode == "synthetic" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(generator_seed)

    model.eval()
    loss_sum = 0.0
    token_count = 0
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            xb, yb = _sample_batch_any(
                cfg=cfg,
                corpus=val_corpus,
                batch_size=cfg.batch_size,
                generator=generator,
                device=device,
            )
            with _autocast_context(cfg, device):
                if isinstance(model, NativeRegionInterfaceModel):
                    logits, _ = model.forward_with_cache(xb)
                else:
                    logits = model(xb)
            loss = _next_token_loss(logits, yb)
            batch_tokens = int(yb.numel())
            loss_sum += float(loss.item()) * batch_tokens
            token_count += batch_tokens
    wall_time_s = time.perf_counter() - start
    ce_loss = loss_sum / max(1, token_count)
    return {
        "ce_loss": float(ce_loss),
        "ppl": _safe_exp(float(ce_loss)),
        "eval_tokens": float(token_count),
        "wall_time_s": float(wall_time_s),
        "tokens_per_s": float(token_count / wall_time_s) if wall_time_s > 0.0 else 0.0,
    }


def _run_label(run_dir: Path, cfg: RegionInterfaceConfig) -> str:
    variant = run_dir.parent.name
    if cfg.regime == "backprop_ref":
        return variant.replace("dense_", "dense_")
    return variant


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_run(run_dir: Path, *, args: argparse.Namespace) -> dict[str, Any]:
    cfg = _load_config(run_dir, args=args)
    device = _resolve_device(cfg)
    checkpoint_path = _checkpoint_from_summary(run_dir, args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path, device=device)

    model = _build_model(cfg).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, val_corpus = _build_corpora(cfg, device=device)
    metrics = _evaluate_loaded_model(
        cfg=cfg,
        model=model,
        val_corpus=val_corpus,
        generator_seed=int(args.eval_seed),
        device=device,
    )

    summary = _read_json(run_dir / "summary.json")
    model_info = _read_json(run_dir / "model_info.json")
    row: dict[str, Any] = {
        "label": _run_label(run_dir, cfg),
        "run_dir": str(run_dir),
        "regime": cfg.regime,
        "backbone": cfg.backbone,
        "seed": cfg.seed,
        "message_dim": cfg.message_dim if cfg.regime == "native_region_interface" else "",
        "region_size": cfg.region_size if cfg.regime == "native_region_interface" else "",
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_best_val_ce_loss": float(checkpoint.get("best_val_ce_loss", float("nan"))),
        "checkpoint_best_val_step": int(checkpoint.get("best_val_step", -1)),
        "summary_best_val_ce_loss": summary.get("best_val_ce_loss", ""),
        "summary_best_val_step": summary.get("best_val_step", ""),
        "summary_final_val_ce_loss": summary.get("final_val_ce_loss", ""),
        "posthoc_val_ce_loss": metrics["ce_loss"],
        "posthoc_val_ppl": metrics["ppl"],
        "eval_batches": cfg.eval_batches,
        "eval_batch_size": cfg.batch_size,
        "eval_seq_len": cfg.seq_len,
        "eval_tokens": int(metrics["eval_tokens"]),
        "eval_seed": int(args.eval_seed),
        "eval_wall_time_s": metrics["wall_time_s"],
        "eval_tokens_per_s": metrics["tokens_per_s"],
        "total_params": model_info.get("total_params", ""),
    }

    del model
    del checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "lm_eval_summary.json"
    csv_path = output_dir / "lm_eval_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_table(rows: list[dict[str, Any]]) -> None:
    cols = [
        ("label", "run"),
        ("posthoc_val_ce_loss", "posthoc_ce"),
        ("posthoc_val_ppl", "ppl"),
        ("summary_best_val_ce_loss", "train_best_ce"),
        ("checkpoint_step", "ckpt_step"),
        ("eval_tokens", "eval_tokens"),
    ]
    print("\t".join(header for _, header in cols))
    for row in rows:
        out: list[str] = []
        for key, _ in cols:
            value = row.get(key, "")
            if isinstance(value, float):
                out.append(f"{value:.6g}")
            else:
                out.append(str(value))
        print("\t".join(out))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-hoc LM evaluation for saved paper checkpoints.")
    parser.add_argument(
        "--family",
        type=str,
        required=True,
        help="Family directory containing dense_seed*/ and lbi_r*_seed*/ runs, or a single regime run directory.",
    )
    parser.add_argument("--output-dir", type=str, default="", help="Directory for lm_eval_summary.{csv,json}.")
    parser.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    parser.add_argument("--eval-batches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help="Optional tokenizer directory/model override used when rebuilding tokenized corpora for evaluation.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Specific run directory to evaluate. May be repeated; when set, --family is only used for default output naming.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    family = Path(args.family).expanduser().resolve()
    if args.run_dir:
        run_dirs = [Path(item).expanduser().resolve() for item in args.run_dir]
    elif (family / "config.json").exists() and (family / "summary.json").exists():
        run_dirs = [family]
    else:
        run_dirs = _discover_run_dirs(family)
    if not run_dirs:
        raise FileNotFoundError(f"no completed paper runs found under: {family}")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path("out/evals/region_interface") / family.name
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        print(f"[eval] {run_dir}", flush=True)
        rows.append(_evaluate_run(run_dir, args=args))
    _write_outputs(rows, output_dir)
    _print_table(rows)
    print(f"[eval] wrote {output_dir / 'lm_eval_summary.csv'}")
    print(f"[eval] wrote {output_dir / 'lm_eval_summary.json'}")


if __name__ == "__main__":
    main()
