from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import torch
from backward import AutogradEngine, ScanADEngine

from train.config import DENSE_VARIANT, LBITrainingConfig, LBI_VARIANT
from train.checkpointing import (
    maybe_restore_training_state as _maybe_restore_training_state,
    save_checkpoint as _save_checkpoint,
)
from train.data import build_corpora as _build_corpora, sample_batch_any as _sample_batch_any
from train.eval import (
    evaluate_native as _evaluate_native,
    evaluate_reference as _evaluate_reference,
    next_token_loss as _next_token_loss,
)
from train.metrics import write_csv_row as _write_csv, write_run_metadata as _write_run_metadata
from train.model_builders import build_dense_model, build_lbi_model


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
    horizon = cfg.lr_schedule_steps if cfg.lr_schedule_steps > 0 else cfg.steps
    decay_steps = max(1, horizon - warmup_steps)
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
        group["lr"] = lr * float(group.get("lr_mult", 1.0))


def _scheduled_lr(cfg: LBITrainingConfig, step: int) -> float:
    return float(cfg.lr_model * _lr_multiplier(cfg, step))


def _halt_on_nonfinite_step(*, ce_loss, grad_norm, step, xb, run_dir) -> None:
    """Halt on the first non-finite loss or gradient norm, saving the step and
    batch token ids so the failing sample reproduces offline."""
    loss_ok = bool(torch.isfinite(ce_loss.detach()))
    norm_ok = grad_norm is None or bool(torch.isfinite(grad_norm))
    if loss_ok and norm_ok:
        return
    path = Path(run_dir) / f"nonfinite_step{int(step)}.pt"
    torch.save({"step": int(step),
                "loss_finite": loss_ok,
                "grad_norm_finite": norm_ok,
                "input_ids": xb.detach().cpu()}, path)
    raise RuntimeError(
        f"non-finite training step {step} (loss finite={loss_ok}, "
        f"grad norm finite={norm_ok}); batch saved to {path}"
    )


def run_dense_training(cfg: LBITrainingConfig, *, run_dir: Path) -> Dict[str, Any]:
    """Paper dense baseline training loop."""
    device = _resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    model = build_dense_model(cfg).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_model, weight_decay=cfg.weight_decay)
    ad_engine = AutogradEngine()
    train_corpus, val_corpus = _build_corpora(cfg)
    _write_run_metadata(cfg=cfg, run_dir=run_dir, model=model, train_corpus=train_corpus, val_corpus=val_corpus)

    train_gen = torch.Generator(device="cpu")
    train_gen.manual_seed(cfg.seed + 101)
    eval_gen = torch.Generator(device="cpu")
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
        total_norm = None
        if cfg.grad_clip > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        _halt_on_nonfinite_step(ce_loss=ce_loss, grad_norm=total_norm, step=step, xb=xb, run_dir=run_dir)
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
                extra={"variant": DENSE_VARIANT, "regime": DENSE_VARIANT, "train_ce_loss": final_train},
            )
            latest_checkpoint_path = str(latest_path)

    summary = {
        "variant": DENSE_VARIANT,
        "regime": DENSE_VARIANT,
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

    # Native backward runs the region kernels in the model dtype (bf16 on
    # Mamba-3) with fp32 master weights in the optimizer; the graph forward
    # trains fp32 weights under autocast.
    if cfg.native_backward:
        model_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
    else:
        model_dtype = torch.float32
    model = build_lbi_model(cfg).to(device=device, dtype=model_dtype)
    if str(cfg.interface_jacobian_mode).lower() == "forward":
        model.region_backend.forward_mode_use_kernel = True

    params = list(model.parameters())
    masters = None
    canvas_params = list(model.canvas_vjp_parameters()) if cfg.canvas_grad_window > 0 else []
    window_acc = {p: None for p in canvas_params}
    if cfg.native_backward and model_dtype is torch.bfloat16:
        masters = [p.detach().clone().float().requires_grad_(False) for p in params]
        opt_params = masters
    else:
        opt_params = params
    if cfg.canvas_grad_window > 0 and cfg.canvas_grad_window_lr_mult != 1.0:
        # The windowed canvas group trains at effective batch x N and takes
        # its own learning-rate multiplier.
        canvas_ids = {id(p) for p in canvas_params}
        canvas_opt = [q for p, q in zip(params, opt_params) if id(p) in canvas_ids]
        other_opt = [q for p, q in zip(params, opt_params) if id(p) not in canvas_ids]
        optimizer = torch.optim.AdamW(
            [{"params": other_opt}, {"params": canvas_opt, "lr_mult": float(cfg.canvas_grad_window_lr_mult)}],
            lr=cfg.lr_model, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(opt_params, lr=cfg.lr_model, weight_decay=cfg.weight_decay)
    train_corpus, val_corpus = _build_corpora(cfg)
    _write_run_metadata(cfg=cfg, run_dir=run_dir, model=model, train_corpus=train_corpus, val_corpus=val_corpus)

    train_gen = torch.Generator(device="cpu")
    train_gen.manual_seed(cfg.seed + 101)
    eval_gen = torch.Generator(device="cpu")
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
    if masters is not None:
        # Restored (or freshly built) weights are the source of truth; the
        # fp32 masters mirror them from here on.
        with torch.no_grad():
            for mp, p in zip(masters, params):
                mp.copy_(p.float())

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

    autograd_backward = cfg.lbi_backward == "autograd"
    ad_engine = None if autograd_backward else ScanADEngine.from_config(cfg)

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
        # Native mode runs in the model's one dtype so the frozen cache stays
        # in the native kernels' dtype; the graph forward runs under autocast.
        forward_ctx = nullcontext() if cfg.native_backward else _autocast_context(cfg, device)
        with forward_ctx:
            logits, cache = model.forward_with_cache(
                xb,
                native_backward=cfg.native_backward,
            )
        ce_loss = _next_token_loss(logits, yb)
        if autograd_backward:
            for p in params:
                p.grad = None
            ce_loss.backward()
        else:
            ad_engine.backward(model=model, loss=ce_loss, cache=cache)

        if cfg.canvas_grad_window > 0:
            # N-step window: the canvas gradient accumulates and the optimizer
            # sees its window mean at the boundary (the last step drains a
            # partial window); in between the canvas parameters carry no gradient.
            n_win = cfg.canvas_grad_window
            at_boundary = (step % n_win == 0) or (step == cfg.steps)
            n_in_window = n_win if step % n_win == 0 else step % n_win
            for p in canvas_params:
                if p.grad is not None:
                    window_acc[p] = p.grad.detach().clone() if window_acc[p] is None else window_acc[p].add_(p.grad.detach())
                if at_boundary and window_acc[p] is not None:
                    p.grad = window_acc[p].div_(n_in_window)
                    window_acc[p] = None
                else:
                    p.grad = None

        total_norm = None
        if masters is not None:
            for mp, p in zip(masters, params):
                mp.grad = None if p.grad is None else p.grad.float()
            if cfg.grad_clip > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(masters, cfg.grad_clip)
            _halt_on_nonfinite_step(ce_loss=ce_loss, grad_norm=total_norm, step=step, xb=xb, run_dir=run_dir)
            optimizer.step()
            with torch.no_grad():
                for mp, p in zip(masters, params):
                    p.copy_(mp.to(p.dtype))
        else:
            if cfg.grad_clip > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            _halt_on_nonfinite_step(ce_loss=ce_loss, grad_norm=total_norm, step=step, xb=xb, run_dir=run_dir)
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
            "tokens_per_s": float(train_tokens_per_step / max(1e-9, (t1 - t0))),
            "wall_time_s": float(t1 - t0),
        }
        if (step % cfg.log_every) == 0 or step == 1:
            _write_csv(metrics_csv, row)
            with metrics_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        if (step % cfg.eval_every) == 0 or step == cfg.steps:
            eval_step_gen = torch.Generator(device="cpu")
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
                                "train_ce_loss": final_train,
                                "val_ce_loss": final_val,
                                "interface_jacobian_mode": cfg.interface_jacobian_mode,
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
                extra={
                    "variant": LBI_VARIANT,
                    "regime": LBI_VARIANT,
                    "train_ce_loss": final_train,
                    "interface_jacobian_mode": cfg.interface_jacobian_mode,
                },
            )
            latest_checkpoint_path = str(latest_path)

    summary = {
        "variant": LBI_VARIANT,
        "regime": LBI_VARIANT,
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
        "best_checkpoint_path": best_checkpoint_path,
        "latest_checkpoint_path": latest_checkpoint_path,
        "resumed_from": resumed_from,
        "initialized_from": initialized_from,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary

