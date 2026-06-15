from __future__ import annotations

# Command-line entrypoint matching the historical train_region_interface module path.
from legacy.train_region_interface import (
    LBITrainingConfig,
    RegionInterfaceConfig,
    _autocast_context,
    _autograd_backward_step,
    _next_token_loss,
    _resolve_device,
    main,
    run_training,
)
from train.lbi import run_lbi_experiment

__all__ = [
    "LBITrainingConfig",
    "RegionInterfaceConfig",
    "_autocast_context",
    "_autograd_backward_step",
    "_next_token_loss",
    "_resolve_device",
    "main",
    "run_lbi_experiment",
    "run_training",
]


if __name__ == "__main__":
    main()
