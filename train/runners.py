from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import torch
from backward import AutogradEngine, ScanADEngine

from data.token_shards import TokenShardCorpus
from models.lbi_language_model import LBILanguageModel
from train.config import DENSE_VARIANT, LBITrainingConfig, LBI_VARIANT, LEGACY_DENSE_REGIME, LEGACY_LBI_REGIME
from train.checkpointing import (
    maybe_restore_training_state as _maybe_restore_training_state,
    save_checkpoint as _save_checkpoint,
)
from train.data import (
    CORPORA_ROOT,
    build_corpora as _build_corpora,
    resolve_text_paths as _resolve_text_paths,
    resolve_token_shard_manifests as _resolve_token_shard_manifests,
    resolve_token_shards_dir as _resolve_token_shards_dir,
    resolve_tokenizer_path as _resolve_tokenizer_path,
    sample_batch_any as _sample_batch_any,
)
from train.eval import (
    evaluate_native as _evaluate_native,
    evaluate_reference as _evaluate_reference,
    next_token_loss as _next_token_loss,
    resolve_eval_message_ablation_modes as _resolve_eval_message_ablation_modes,
)
from train.metrics import write_csv_row as _write_csv, write_run_metadata as _write_run_metadata
from train.model_builders import build_dense_model, build_lbi_model

_EVAL_MESSAGE_ABLATION_MODES = ("zero_all", "noise", "mask")


def _lbi_cache_state_norm(cache: dict[str, Any]) -> float:
    states = cache.get("states", [])
    if not states:
        return 0.0
    return float(torch.stack([state.norm(dim=-1).mean() for state in states]).mean().item())




def _resolve_device(cfg: LBITrainingConfig) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")




def _autocast_context(cfg: LBITrainingConfig, device: torch.device):
    if cfg.dtype == "bfloat16":
        if device.type != "cuda":
            raise RuntimeError("bfloat16 currently requires CUDA")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()



def _lr_multiplier(cfg: LBITrainingConfig, step: int) -> float:
    if step <= 0:
        return 0.0
    warmup_steps = cfg.warmup_steps
    if warmup_steps > 0 and step <= warmup_steps:
        return float(step) / float(warmup_steps)
    if cfg.lr_schedule == "constant":
        return 1.0
    decay_steps = max(1, cfg.steps - warmup_steps)
    decay_progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    if cfg.lr_schedule == "linear":
        decay_multiplier = 1.0 - decay_progress
    elif cfg.lr_schedule == "cosine":
        decay_multiplier = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    else:
        raise ValueError(f"unsupported lr_schedule: {cfg.lr_schedule}")
    return cfg.min_lr_ratio + ((1.0 - cfg.min_lr_ratio) * decay_multiplier)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _scheduled_lr(cfg: LBITrainingConfig, step: int) -> float:
    return float(cfg.lr_model * _lr_multiplier(cfg, step))





def run_dense_training(cfg: LBITrainingConfig, *, run_dir: Path) -> Dict[str, Any]:
    """Paper dense baseline training loop."""
    device = _resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    model = build_dense_model(cfg).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_model, weight_decay=cfg.weight_decay)
    ad_engine = AutogradEngine()
    train_corpus, val_corpus = _build_corpora(cfg, device=device)
    _write_run_metadata(
        cfg=cfg,
        run_dir=run_dir,
        model=model,
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        corpora_root=CORPORA_ROOT,
        resolve_text_paths_fn=_resolve_text_paths,
        resolve_tokenizer_path_fn=_resolve_tokenizer_path,
        resolve_token_shard_manifests_fn=_resolve_token_shard_manifests,
        resolve_token_shards_dir_fn=_resolve_token_shards_dir,
    )

    gen_device = device if cfg.data_mode == "synthetic" else torch.device("cpu")
    train_gen = torch.Generator(device=gen_device)
    train_gen.manual_seed(cfg.seed + 101)
    eval_gen = torch.Generator(device=gen_device)
    eval_gen.manual_seed(cfg.seed + 202)
    restore = _maybe_restore_training_state(
        mode="resume" if cfg.resume_from else "init",
        load_from=cfg.resume_from or cfg.init_from,
        regime=DENSE_VARIANT,
        model=model,
        optimizer=optimizer,
        device=device,
        train_generator=train_gen,
        eval_generator=eval_gen,
    )

    metrics_csv = run_dir / "metrics.csv"
    metrics_jsonl = run_dir / "metrics.jsonl"
    final_train = float("nan")
    final_val = float("nan")
    best_val = float(restore["best_val_ce_loss"])
    best_val_step = int(restore["best_val_step"])
    best_checkpoint_path = ""
    latest_checkpoint_path = ""
    train_tokens_per_step = int(cfg.batch_size * cfg.seq_len)
    total_train_tokens = int(restore["tokens_seen"])
    start_step = int(restore["start_step"])
    resumed_from = str(restore["checkpoint_path"]) if cfg.resume_from else ""
    initialized_from = str(restore["checkpoint_path"]) if cfg.init_from else ""
    _set_optimizer_lr(optimizer, _scheduled_lr(cfg, start_step))

    for step in range(start_step + 1, cfg.steps + 1):
        t0 = time.perf_counter()
        model.train()
        _set_optimizer_lr(optimizer, _scheduled_lr(cfg, step))
        xb, yb = _sample_batch_any(
            cfg=cfg,
            corpus=train_corpus,
            batch_size=cfg.batch_size,
            generator=train_gen,
            device=device,
        )
        with _autocast_context(cfg, device):
            logits = model(xb)
        ce_loss = _next_token_loss(logits, yb)
        ad_engine.backward(model=model, loss=ce_loss)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        t1 = time.perf_counter()

        final_train = float(ce_loss.item())
        total_train_tokens = int(step * train_tokens_per_step)
        row = {
            "step": step,
            "tokens_seen": total_train_tokens,
            "split": "train",
            "ce_loss": final_train,
            "message_norm": "",
            "scan_align": "",
            "tokens_per_s": float(train_tokens_per_step / max(1e-9, (t1 - t0))),
            "wall_time_s": float(t1 - t0),
        }
        if (step % cfg.log_every) == 0 or step == 1:
            _write_csv(metrics_csv, row)
            with metrics_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

        if (step % cfg.eval_every) == 0 or step == cfg.steps:
            val = _evaluate_reference(
                cfg=cfg,
                model=model,
                val_corpus=val_corpus,
                eval_batches=cfg.eval_batches,
                generator=eval_gen,
                device=device,
                sample_batch_fn=_sample_batch_any,
                autocast_context_fn=_autocast_context,
            )
            final_val = float(val["ce_loss"])
            if final_val < best_val:
                best_val = final_val
                best_val_step = step
                if cfg.save_checkpoints:
                    best_checkpoint_path = str(
                        _save_checkpoint(
                            run_dir=run_dir,
                            filename="best.pt",
                            model=model,
                            optimizer=optimizer,
                            cfg=cfg,
                            device=device,
                            train_generator=train_gen,
                            eval_generator=eval_gen,
                            step=step,
                            tokens_seen=total_train_tokens,
                            best_val_ce_loss=best_val,
                            best_val_step=best_val_step,
                            extra={
                                "variant": DENSE_VARIANT,
                                "regime": DENSE_VARIANT,
                                "legacy_regime": LEGACY_DENSE_REGIME,
                                "train_ce_loss": final_train,
                                "val_ce_loss": final_val,
                                "is_best": True,
                            },
                        )
                    )
            val_row = {
                "step": step,
                "tokens_seen": total_train_tokens,
                "split": "val",
                "ce_loss": final_val,
                "message_norm": "",
                "scan_align": "",
                "tokens_per_s": 0.0,
                "wall_time_s": 0.0,
            }
            _write_csv(metrics_csv, val_row)
            with metrics_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(val_row, sort_keys=True) + "\n")

        if cfg.save_checkpoints and ((step % cfg.save_every) == 0 or step == cfg.steps):
            latest_path = _save_checkpoint(
                run_dir=run_dir,
                filename="latest.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                device=device,
                train_generator=train_gen,
                eval_generator=eval_gen,
                step=step,
                tokens_seen=total_train_tokens,
                best_val_ce_loss=best_val,
                best_val_step=best_val_step,
                extra={"variant": DENSE_VARIANT, "regime": DENSE_VARIANT, "legacy_regime": LEGACY_DENSE_REGIME, "train_ce_loss": final_train},
            )
            latest_checkpoint_path = str(latest_path)

    summary = {
        "variant": DENSE_VARIANT,
        "regime": DENSE_VARIANT,
        "legacy_regime": LEGACY_DENSE_REGIME,
        "task": cfg.task,
        "steps": int(cfg.steps),
        "start_step": start_step,
        "final_train_ce_loss": final_train,
        "final_val_ce_loss": final_val,
        "best_val_ce_loss": best_val,
        "best_val_step": best_val_step,
        "train_tokens_per_step": train_tokens_per_step,
        "total_train_tokens": total_train_tokens,
        "save_checkpoints": bool(cfg.save_checkpoints),
        "best_checkpoint_path": best_checkpoint_path,
        "latest_checkpoint_path": latest_checkpoint_path,
        "resumed_from": resumed_from,
        "initialized_from": initialized_from,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_lbi_training(cfg: LBITrainingConfig, *, run_dir: Path) -> Dict[str, Any]:
    """Training loop for the LBI model variant."""
    device = _resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    model = build_lbi_model(cfg).to(device=device, dtype=torch.float32)
    if not isinstance(model, LBILanguageModel):
        raise TypeError("LBI training requires LBILanguageModel")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_model, weight_decay=cfg.weight_decay)
    train_corpus, val_corpus = _build_corpora(cfg, device=device)
    _write_run_metadata(
        cfg=cfg,
        run_dir=run_dir,
        model=model,
        train_corpus=train_corpus,
        val_corpus=val_corpus,
        corpora_root=CORPORA_ROOT,
        resolve_text_paths_fn=_resolve_text_paths,
        resolve_tokenizer_path_fn=_resolve_tokenizer_path,
        resolve_token_shard_manifests_fn=_resolve_token_shard_manifests,
        resolve_token_shards_dir_fn=_resolve_token_shards_dir,
    )

    gen_device = device if cfg.data_mode == "synthetic" else torch.device("cpu")
    train_gen = torch.Generator(device=gen_device)
    train_gen.manual_seed(cfg.seed + 101)
    eval_gen = torch.Generator(device=gen_device)
    eval_gen.manual_seed(cfg.seed + 202)
    restore = _maybe_restore_training_state(
        mode="resume" if cfg.resume_from else "init",
        load_from=cfg.resume_from or cfg.init_from,
        regime=LBI_VARIANT,
        model=model,
        optimizer=optimizer,
        device=device,
        train_generator=train_gen,
        eval_generator=eval_gen,
    )

    metrics_csv = run_dir / "metrics.csv"
    metrics_jsonl = run_dir / "metrics.jsonl"
    final_train = float("nan")
    final_val = float("nan")
    best_val = float(restore["best_val_ce_loss"])
    best_val_step = int(restore["best_val_step"])
    final_eval_message_ablations: Dict[str, float] = {}
    best_checkpoint_path = ""
    latest_checkpoint_path = ""
    final_interface_jacobian_stats: Dict[str, float] = {}
    train_tokens_per_step = int(cfg.batch_size * cfg.seq_len)
    total_train_tokens = int(restore["tokens_seen"])
    start_step = int(restore["start_step"])
    resumed_from = str(restore["checkpoint_path"]) if cfg.resume_from else ""
    initialized_from = str(restore["checkpoint_path"]) if cfg.init_from else ""
    _set_optimizer_lr(optimizer, _scheduled_lr(cfg, start_step))

    for step in range(start_step + 1, cfg.steps + 1):
        t0 = time.perf_counter()
        model.train()
        _set_optimizer_lr(optimizer, _scheduled_lr(cfg, step))
        xb, yb = _sample_batch_any(
            cfg=cfg,
            corpus=train_corpus,
            batch_size=cfg.batch_size,
            generator=train_gen,
            device=device,
        )
        with _autocast_context(cfg, device):
            logits, cache = model.forward_with_cache(xb)
        ce_loss = _next_token_loss(logits, yb)
        log_interface_jacobian = (
            cfg.log_interface_jacobian_every > 0
            and ((step % cfg.log_interface_jacobian_every) == 0 or step == 1 or step == cfg.steps)
        )
        ad_engine = ScanADEngine.from_config(
            cfg,
            compute_interface_jacobian_stats=log_interface_jacobian,
        )
        ad_result = ad_engine.backward(model=model, loss=ce_loss, cache=cache)
        interface_scan_rms = float(ad_result.diagnostics.get("interface_scan_rms", 0.0))
        interface_jacobian_stats = ad_result.diagnostics.get("interface_jacobian_stats")
        if interface_jacobian_stats is not None:
            final_interface_jacobian_stats = dict(interface_jacobian_stats)

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        t1 = time.perf_counter()

        final_train = float(ce_loss.item())
        total_train_tokens = int(step * train_tokens_per_step)
        msg_norm = _lbi_cache_state_norm(cache)
        row = {
            "step": step,
            "tokens_seen": total_train_tokens,
            "split": "train",
            "ce_loss": final_train,
            "message_norm": msg_norm,
            "scan_align": interface_scan_rms,
            "tokens_per_s": float(train_tokens_per_step / max(1e-9, (t1 - t0))),
            "wall_time_s": float(t1 - t0),
        }
        if interface_jacobian_stats is not None:
            row.update(interface_jacobian_stats)
        if (step % cfg.log_every) == 0 or step == 1 or log_interface_jacobian:
            _write_csv(metrics_csv, row)
            with metrics_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        if (step % cfg.eval_every) == 0 or step == cfg.steps:
            eval_step_gen = torch.Generator(device=gen_device)
            eval_step_gen.manual_seed(cfg.seed + 202 + (step * 11))
            val = _evaluate_native(
                cfg=cfg,
                model=model,
                val_corpus=val_corpus,
                eval_batches=cfg.eval_batches,
                generator=eval_step_gen,
                device=device,
                sample_batch_fn=_sample_batch_any,
                autocast_context_fn=_autocast_context,
            )
            final_val = float(val["ce_loss"])
            if final_val < best_val:
                best_val = final_val
                best_val_step = step
                if cfg.save_checkpoints:
                    best_checkpoint_path = str(
                        _save_checkpoint(
                            run_dir=run_dir,
                            filename="best.pt",
                            model=model,
                            optimizer=optimizer,
                            cfg=cfg,
                            device=device,
                            train_generator=train_gen,
                            eval_generator=eval_gen,
                            step=step,
                            tokens_seen=total_train_tokens,
                            best_val_ce_loss=best_val,
                            best_val_step=best_val_step,
                            extra={
                                "variant": LBI_VARIANT,
                                "regime": LBI_VARIANT,
                                "legacy_regime": LEGACY_LBI_REGIME,
                                "owned_lbi_model": True,
                                "train_ce_loss": final_train,
                                "val_ce_loss": final_val,
                                "interface_scan_rms": interface_scan_rms,
                                "interface_jacobian_mode": cfg.interface_jacobian_mode,
                                "jacobian_basis_chunk": cfg.jacobian_basis_chunk,
                                "interface_jacobian_stats": final_interface_jacobian_stats,
                                "is_best": True,
                            },
                        )
                    )
            val_row = {
                "step": step,
                "tokens_seen": total_train_tokens,
                "split": "val",
                "ce_loss": final_val,
                "message_norm": "",
                "scan_align": "",
                "tokens_per_s": 0.0,
                "wall_time_s": 0.0,
            }
            _write_csv(metrics_csv, val_row)
            with metrics_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(val_row, sort_keys=True) + "\n")

            for ablation_idx, message_ablation in enumerate(_resolve_eval_message_ablation_modes(cfg, ablation_modes=_EVAL_MESSAGE_ABLATION_MODES)):
                ablation_batch_gen = torch.Generator(device=gen_device)
                ablation_batch_gen.manual_seed(cfg.seed + 202 + (step * 11))
                eval_ablation_gen = torch.Generator(device=device)
                eval_ablation_gen.manual_seed(cfg.seed + 303 + (step * 17) + ablation_idx)
                ablated = _evaluate_native(
                    cfg=cfg,
                    model=model,
                    val_corpus=val_corpus,
                    eval_batches=cfg.eval_batches,
                    generator=ablation_batch_gen,
                    device=device,
                    sample_batch_fn=_sample_batch_any,
                    autocast_context_fn=_autocast_context,
                    message_ablation=message_ablation,
                    ablation_generator=eval_ablation_gen,
                )
                split_name = f"val_{message_ablation}"
                final_eval_message_ablations[split_name] = float(ablated["ce_loss"])
                ablation_row = {
                    "step": step,
                    "tokens_seen": total_train_tokens,
                    "split": split_name,
                    "ce_loss": float(ablated["ce_loss"]),
                    "message_norm": "",
                    "scan_align": "",
                    "tokens_per_s": 0.0,
                    "wall_time_s": 0.0,
                }
                _write_csv(metrics_csv, ablation_row)
                with metrics_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(ablation_row, sort_keys=True) + "\n")

        if cfg.save_checkpoints and ((step % cfg.save_every) == 0 or step == cfg.steps):
            latest_path = _save_checkpoint(
                run_dir=run_dir,
                filename="latest.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                device=device,
                train_generator=train_gen,
                eval_generator=eval_gen,
                step=step,
                tokens_seen=total_train_tokens,
                best_val_ce_loss=best_val,
                best_val_step=best_val_step,
                extra={
                    "variant": LBI_VARIANT,
                    "regime": LBI_VARIANT,
                    "legacy_regime": LEGACY_LBI_REGIME,
                    "owned_lbi_model": True,
                    "train_ce_loss": final_train,
                    "interface_scan_rms": interface_scan_rms,
                    "interface_jacobian_mode": cfg.interface_jacobian_mode,
                    "jacobian_basis_chunk": cfg.jacobian_basis_chunk,
                    "interface_jacobian_stats": final_interface_jacobian_stats,
                },
            )
            latest_checkpoint_path = str(latest_path)

    summary = {
        "variant": LBI_VARIANT,
        "regime": LBI_VARIANT,
        "legacy_regime": LEGACY_LBI_REGIME,
        "task": cfg.task,
        "steps": int(cfg.steps),
        "start_step": start_step,
        "final_train_ce_loss": final_train,
        "final_val_ce_loss": final_val,
        "best_val_ce_loss": best_val,
        "best_val_step": best_val_step,
        "train_tokens_per_step": train_tokens_per_step,
        "total_train_tokens": total_train_tokens,
        "save_checkpoints": bool(cfg.save_checkpoints),
        "interface_jacobian_mode": cfg.interface_jacobian_mode,
        "jacobian_basis_chunk": int(cfg.jacobian_basis_chunk),
        "log_interface_jacobian_every": int(cfg.log_interface_jacobian_every),
        "log_interface_jacobian_suffix": bool(cfg.log_interface_jacobian_suffix),
        "final_interface_jacobian_stats": final_interface_jacobian_stats,
        "best_checkpoint_path": best_checkpoint_path,
        "latest_checkpoint_path": latest_checkpoint_path,
        "final_eval_message_ablations": final_eval_message_ablations,
        "resumed_from": resumed_from,
        "initialized_from": initialized_from,
        "run_dir": str(run_dir),
        "owned_lbi_model": True,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary

