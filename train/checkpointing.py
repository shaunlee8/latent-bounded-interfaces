from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from train.config import compatible_output_names_for_variant


def checkpoint_dir(run_dir: Path, cfg: Any) -> Path:
    if cfg.checkpoint_root:
        output_root = Path(cfg.output_dir).expanduser().resolve()
        checkpoint_root = Path(cfg.checkpoint_root).expanduser().resolve()
        try:
            relative_dir = run_dir.resolve().relative_to(output_root.parent)
        except ValueError:
            relative_dir = Path(run_dir.name)
        out = checkpoint_root / relative_dir / "checkpoints"
    else:
        out = run_dir / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    return out


def checkpoint_path_from_summary(summary_path: Path, *, preference: str) -> Path | None:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    preferred_key = "latest_checkpoint_path" if preference == "latest" else "best_checkpoint_path"
    fallback_key = "best_checkpoint_path" if preference == "latest" else "latest_checkpoint_path"
    for key in (preferred_key, fallback_key):
        candidate = str(summary.get(key, "")).strip()
        if candidate:
            path = Path(candidate).expanduser()
            if path.exists():
                return path
    return None


def save_checkpoint(
    *,
    run_dir: Path,
    filename: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    device: torch.device,
    train_generator: torch.Generator,
    eval_generator: torch.Generator,
    step: int,
    tokens_seen: int,
    best_val_ce_loss: float,
    best_val_step: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "best_val_ce_loss": float(best_val_ce_loss),
        "best_val_step": int(best_val_step),
        "config": cfg.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "train_generator_state": train_generator.get_state(),
        "eval_generator_state": eval_generator.get_state(),
    }
    if extra:
        payload.update(extra)
    path = checkpoint_dir(run_dir, cfg) / filename
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, *, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device)


def resolve_checkpoint_path(
    load_from: str,
    *,
    regime: str,
    preference: str,
) -> Path | None:
    if not load_from:
        return None
    source = Path(load_from).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"checkpoint source does not exist: {source}")
    if source.is_file():
        return source

    candidates: list[Path] = []
    preferred_name = "latest.pt" if preference == "latest" else "best.pt"
    fallback_name = "best.pt" if preference == "latest" else "latest.pt"
    compatible_names = compatible_output_names_for_variant(regime)
    if (source / "metrics.csv").exists():
        candidates.extend(
            [
                source / "checkpoints" / preferred_name,
                source / "checkpoints" / fallback_name,
            ]
        )
    else:
        for output_name in compatible_names:
            candidates.extend(
                [
                    source / output_name / "checkpoints" / preferred_name,
                    source / output_name / "checkpoints" / fallback_name,
                ]
            )
        candidates.extend(
            [
                source / "checkpoints" / preferred_name,
                source / "checkpoints" / fallback_name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if (source / "metrics.csv").exists():
        from_summary = checkpoint_path_from_summary(source / "summary.json", preference=preference)
        if from_summary is not None:
            return from_summary
    else:
        for output_name in compatible_output_names_for_variant(regime):
            from_regime_summary = checkpoint_path_from_summary(source / output_name / "summary.json", preference=preference)
            if from_regime_summary is not None:
                return from_regime_summary
        from_summary = checkpoint_path_from_summary(source / "summary.json", preference=preference)
        if from_summary is not None:
            return from_summary
    return None


def infer_existing_root_dir(load_from: str) -> Path:
    source = Path(load_from).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"checkpoint source does not exist: {source}")
    if source.is_file():
        if source.parent.name == "checkpoints":
            return source.parent.parent.parent
        return source.parent
    if (source / "metrics.csv").exists():
        return source.parent
    if (source / "checkpoints").exists():
        return source.parent
    return source


def maybe_restore_training_state(
    *,
    mode: str,
    load_from: str,
    regime: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_generator: torch.Generator,
    eval_generator: torch.Generator,
) -> dict[str, Any]:
    if mode not in {"resume", "init"}:
        raise ValueError("mode must be one of: resume, init")
    checkpoint_path = resolve_checkpoint_path(
        load_from,
        regime=regime,
        preference="latest" if mode == "resume" else "best",
    )
    if checkpoint_path is None:
        return {
            "checkpoint_path": "",
            "start_step": 0,
            "tokens_seen": 0,
            "best_val_ce_loss": float("inf"),
            "best_val_step": -1,
        }

    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if mode == "resume":
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        torch_rng_state = checkpoint.get("torch_rng_state")
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state.cpu())
        if device.type == "cuda":
            cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
            if cuda_rng_state_all:
                torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state_all])
        train_gen_state = checkpoint.get("train_generator_state")
        if train_gen_state is not None:
            train_generator.set_state(train_gen_state.cpu())
        eval_gen_state = checkpoint.get("eval_generator_state")
        if eval_gen_state is not None:
            eval_generator.set_state(eval_gen_state.cpu())
        return {
            "checkpoint_path": str(checkpoint_path),
            "start_step": int(checkpoint.get("step", 0)),
            "tokens_seen": int(checkpoint.get("tokens_seen", 0)),
            "best_val_ce_loss": float(checkpoint.get("best_val_ce_loss", float("inf"))),
            "best_val_step": int(checkpoint.get("best_val_step", -1)),
        }
    return {
        "checkpoint_path": str(checkpoint_path),
        "start_step": 0,
        "tokens_seen": 0,
        "best_val_ce_loss": float("inf"),
        "best_val_step": -1,
    }
