from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Dict

from backward import autograd_backward_step as _autograd_backward_step
from train.checkpointing import infer_existing_root_dir as _infer_existing_root_dir
from train.config import (
    DENSE_VARIANT,
    LBITrainingConfig,
    LBI_VARIANT,
    build_arg_parser as _build_arg_parser,
    config_from_args as _from_args,
    output_name_for_variant,
    resolve_model_variants,
)
from train.data import resolve_runtime_vocab_size as _resolve_runtime_vocab_size
from train.eval import next_token_loss as _next_token_loss
from train.runners import (
    _autocast_context,
    _resolve_device,
    run_dense_training,
    run_lbi_training,
)


# Training entrypoint for dense and LBI model variants.
# Runs are selected by model variants and written to dense/ and lbi/ subdirectories.



def _prepare_run_dir(cfg: LBITrainingConfig, run_root_name: str) -> Path:
    root = Path(cfg.output_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = cfg.run_name if cfg.run_name else f"{run_root_name}_{stamp}"
    out = root / base
    out.mkdir(parents=True, exist_ok=True)
    return out




def _root_name_for_variants(variants: tuple[str, ...]) -> str:
    return "_".join(variants) if variants else "lbi"


def run_lbi_experiment(cfg: LBITrainingConfig) -> Dict[str, Any]:
    """Run one configured experiment for the selected dense and/or LBI variants."""
    cfg.vocab_size = _resolve_runtime_vocab_size(cfg)
    variants = resolve_model_variants(cfg)
    root_dir = _infer_existing_root_dir(cfg.resume_from) if cfg.resume_from else _prepare_run_dir(
        cfg,
        run_root_name=_root_name_for_variants(variants),
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for variant in variants:
        run_cfg = replace(cfg, variants=variant, regime=variant)
        run_dir = root_dir / output_name_for_variant(variant)
        run_dir.mkdir(parents=True, exist_ok=True)
        if variant == DENSE_VARIANT:
            runs.append(run_dense_training(run_cfg, run_dir=run_dir))
        elif variant == LBI_VARIANT:
            runs.append(run_lbi_training(run_cfg, run_dir=run_dir))
        else:
            raise AssertionError(f"unsupported model variant: {variant}")
    summary = {"variants": list(variants), "root_dir": str(root_dir), "runs": runs}
    (root_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


# Public alias used by script entrypoints.
run_training = run_lbi_experiment


def main() -> None:
    cfg = _from_args(_build_arg_parser().parse_args())
    summary = run_lbi_experiment(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
