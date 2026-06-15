from __future__ import annotations

import json
from typing import Any, Dict

from config.lbi2 import from_legacy_region_interface_config
from train.config import LBITrainingConfig, RegionInterfaceConfig, build_arg_parser as _build_arg_parser, config_from_args as _from_args
from train.lbi import (
    _autocast_context,
    _autograd_backward_step,
    _next_token_loss,
    _resolve_device,
    run_lbi_experiment,
)


def _prepare_legacy_cfg(cfg: LBITrainingConfig) -> LBITrainingConfig:
    # Keep the LBI-1 config bridge here so train.lbi is variant-based and LBI-2-only.
    from_legacy_region_interface_config(cfg)
    if not str(getattr(cfg, "variants", "")).strip():
        cfg.variants = cfg.regime
    return cfg


def run_training(cfg: LBITrainingConfig) -> Dict[str, Any]:
    return run_lbi_experiment(_prepare_legacy_cfg(cfg))


def main() -> None:
    cfg = _from_args(_build_arg_parser().parse_args())
    summary = run_training(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))


__all__ = [
    "LBITrainingConfig",
    "RegionInterfaceConfig",
    "_autocast_context",
    "_autograd_backward_step",
    "_next_token_loss",
    "_resolve_device",
    "main",
    "run_training",
]


if __name__ == "__main__":
    main()
