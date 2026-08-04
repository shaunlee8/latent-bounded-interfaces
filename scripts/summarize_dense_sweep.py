from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_dense_sweep(sweep_dir: Path, *, output_csv: Path | None = None) -> Path:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(sweep_dir.glob("**/backprop_ref/summary.json")):
        run_dir = summary_path.parent
        variant_dir = run_dir.parent
        try:
            summary = _load_json(summary_path)
            config = _load_json(run_dir / "config.json")
            model_info = _load_json(run_dir / "model_info.json")
        except FileNotFoundError:
            continue
        rows.append(
            {
                "variant": variant_dir.name,
                "run_dir": str(run_dir),
                "backbone": config.get("backbone", ""),
                "seed": config.get("seed", ""),
                "steps": summary.get("steps", config.get("steps", "")),
                "best_val_ce_loss": summary.get("best_val_ce_loss", ""),
                "best_val_step": summary.get("best_val_step", ""),
                "final_val_ce_loss": summary.get("final_val_ce_loss", ""),
                "final_train_ce_loss": summary.get("final_train_ce_loss", ""),
                "lr_model": config.get("lr_model", ""),
                "lr_schedule": config.get("lr_schedule", ""),
                "warmup_steps": config.get("warmup_steps", ""),
                "min_lr_ratio": config.get("min_lr_ratio", ""),
                "weight_decay": config.get("weight_decay", ""),
                "grad_clip": config.get("grad_clip", ""),
                "batch_size": config.get("batch_size", ""),
                "seq_len": config.get("seq_len", ""),
                "total_params": model_info.get("total_params", ""),
                "save_checkpoints": summary.get("save_checkpoints", config.get("save_checkpoints", "")),
            }
        )
    rows.sort(key=lambda row: float(row["best_val_ce_loss"]) if row["best_val_ce_loss"] != "" else float("inf"))
    output_csv = output_csv or (sweep_dir / "dense_tuning_summary.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "backbone",
        "seed",
        "steps",
        "best_val_ce_loss",
        "best_val_step",
        "final_val_ce_loss",
        "final_train_ce_loss",
        "lr_model",
        "lr_schedule",
        "warmup_steps",
        "min_lr_ratio",
        "weight_decay",
        "grad_clip",
        "batch_size",
        "seq_len",
        "total_params",
        "save_checkpoints",
        "run_dir",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output_csv


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize dense tuning sweep results.")
    parser.add_argument("sweep_dir", type=Path, help="Sweep family directory or sweep root to scan.")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    output_csv = summarize_dense_sweep(args.sweep_dir, output_csv=args.output_csv)
    print(output_csv)


if __name__ == "__main__":
    main()
