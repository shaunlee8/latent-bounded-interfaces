from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.general import BackboneSpec, ReferenceLM, infer_message_hidden_dim
from data.bpe_tokenizer import build_sentencepiece_bpe_stream, load_or_train_sentencepiece_bpe_tokenizer
from data.data_paths import CORPORA_ROOT, TOKENIZERS_ROOT
from data.lm_data import build_byte_stream, sample_batch_stream, split_stream_train_val
from data.train_data import build_synthetic_corpus, sample_batch
from models.native_region_interface import NativeRegionInterfaceModel

_BUNDLED_TEXT_CORPORA: Dict[str, tuple[str, str]] = {
    "tiny_shakespeare": ("tiny_shakespeare_train.txt", "tiny_shakespeare_val.txt"),
    "enwik8": ("enwik8_train.bin", "enwik8_val.bin"),
    "wikitext103_raw": ("wikitext103_raw_train.txt", "wikitext103_raw_val.txt"),
    "tinystories": ("tinystories_train.txt", "tinystories_val.txt"),
}


@dataclass
class RegionInterfaceConfig:
    regime: str = "all"  # native_region_interface | all
    backbone: str = "mamba1"
    seed: int = 7
    device: str = "auto"  # auto | cpu | cuda
    dtype: str = "float32"
    output_dir: str = "out/region_interface"
    run_name: str = ""
    task: str = "copy"
    data_mode: str = "synthetic"  # synthetic | text_byte | text_bpe
    text_corpus: str = "tiny_shakespeare"
    train_text_path: str = ""
    val_text_path: str = ""
    val_split: float = 0.1
    tokenizer_path: str = ""
    train_tokenizer: bool = False
    tokenizer_train_bytes: int = 8 * 1024 * 1024
    vocab_size: int = 256
    seq_len: int = 64
    train_sequences: int = 2048
    val_sequences: int = 256
    batch_size: int = 8
    steps: int = 200
    eval_every: int = 20
    eval_batches: int = 4
    log_every: int = 5
    save_every: int = 100
    lr_model: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    layers: int = 4
    dim: int = 64
    d_state: int = 8
    dt_rank: int | None = None
    expand: int = 2
    d_conv: int = 4
    headdim: int = 128
    ngroups: int = 1
    chunk_size: int = 256
    use_mem_eff_path: bool = True
    include_reference_run: bool = True
    region_size: int = 2
    message_dim: int = 64
    message_hidden_dim: int = 0
    message_scale_init: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Temporary compatibility alias while the new repo settles.
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train bounded region interfaces on a dense Mamba backbone.")
    p.add_argument("--regime", type=str, default="all")
    p.add_argument("--backbone", type=str, default="mamba1")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--output-dir", type=str, default="out/region_interface")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--task", type=str, default="copy")
    p.add_argument("--data-mode", type=str, default="synthetic")
    p.add_argument("--text-corpus", type=str, default="tiny_shakespeare")
    p.add_argument("--train-text-path", type=str, default="")
    p.add_argument("--val-text-path", type=str, default="")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--tokenizer-path", type=str, default="")
    p.add_argument("--train-tokenizer", action="store_true")
    p.add_argument("--tokenizer-train-bytes", type=int, default=8 * 1024 * 1024)
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--train-sequences", type=int, default=2048)
    p.add_argument("--val-sequences", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--lr-model", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--d-state", type=int, default=8)
    p.add_argument("--dt-rank", type=int, default=-1)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--headdim", type=int, default=128)
    p.add_argument("--ngroups", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--no-mem-eff-path", dest="use_mem_eff_path", action="store_false")
    p.set_defaults(use_mem_eff_path=True)
    p.add_argument("--include-reference-run", action="store_true")
    p.add_argument("--no-reference-run", dest="include_reference_run", action="store_false")
    p.set_defaults(include_reference_run=True)
    p.add_argument("--region-size", type=int, default=2)
    p.add_argument("--message-dim", type=int, default=64)
    p.add_argument("--message-hidden-dim", type=int, default=0)
    p.add_argument("--message-scale-init", type=float, default=0.5)
    return p


def _from_args(args: argparse.Namespace) -> RegionInterfaceConfig:
    cfg = RegionInterfaceConfig(
        regime=args.regime,
        backbone=args.backbone,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        output_dir=args.output_dir,
        run_name=args.run_name,
        task=args.task,
        data_mode=args.data_mode,
        text_corpus=args.text_corpus,
        train_text_path=args.train_text_path,
        val_text_path=args.val_text_path,
        val_split=args.val_split,
        tokenizer_path=args.tokenizer_path,
        train_tokenizer=args.train_tokenizer,
        tokenizer_train_bytes=args.tokenizer_train_bytes,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        log_every=args.log_every,
        save_every=args.save_every,
        lr_model=args.lr_model,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        layers=args.layers,
        dim=args.dim,
        d_state=args.d_state,
        dt_rank=None if args.dt_rank < 0 else args.dt_rank,
        expand=args.expand,
        d_conv=args.d_conv,
        headdim=args.headdim,
        ngroups=args.ngroups,
        chunk_size=args.chunk_size,
        use_mem_eff_path=args.use_mem_eff_path,
        include_reference_run=args.include_reference_run,
        region_size=args.region_size,
        message_dim=args.message_dim,
        message_hidden_dim=args.message_hidden_dim,
        message_scale_init=args.message_scale_init,
    )
    if cfg.regime not in {"native_region_interface", "all"}:
        raise ValueError("regime must be one of: native_region_interface, all")
    if cfg.backbone not in {"mamba1", "mamba2", "mamba3"}:
        raise ValueError("backbone must be one of: mamba1, mamba2, mamba3")
    if cfg.dtype != "float32":
        raise ValueError("Only float32 is supported in this scaffold.")
    if cfg.data_mode not in {"synthetic", "text_byte", "text_bpe"}:
        raise ValueError("data_mode must be one of: synthetic, text_byte, text_bpe")
    if cfg.data_mode == "text_byte":
        if cfg.vocab_size != 256:
            raise ValueError("text_byte mode requires vocab_size=256")
        if not (0.0 < cfg.val_split < 1.0):
            raise ValueError("val_split must be in (0,1)")
    if cfg.data_mode == "text_bpe":
        if cfg.vocab_size <= 256:
            raise ValueError("text_bpe mode requires vocab_size > 256")
        if cfg.tokenizer_train_bytes <= 0:
            raise ValueError("tokenizer_train_bytes must be > 0")
        if not (0.0 < cfg.val_split < 1.0):
            raise ValueError("val_split must be in (0,1)")
    if cfg.data_mode != "synthetic" and (not cfg.train_text_path) and cfg.text_corpus not in _BUNDLED_TEXT_CORPORA:
        allowed = ", ".join(sorted(_BUNDLED_TEXT_CORPORA.keys()))
        raise ValueError(f"text_corpus must be one of: {allowed} when train_text_path is not set")
    if cfg.region_size <= 0:
        raise ValueError("region_size must be > 0.")
    if cfg.message_dim <= 0:
        raise ValueError("message_dim must be > 0.")
    if cfg.message_hidden_dim < 0:
        raise ValueError("message_hidden_dim must be >= 0.")
    if cfg.message_scale_init <= 0.0:
        raise ValueError("message_scale_init must be > 0.")
    return cfg


def _make_backbone_spec(cfg: RegionInterfaceConfig) -> BackboneSpec:
    spec = BackboneSpec(
        name=cfg.backbone,
        dim=cfg.dim,
        layers=cfg.layers,
        d_state=cfg.d_state,
        dt_rank=cfg.dt_rank,
        expand=cfg.expand,
        d_conv=cfg.d_conv,
        headdim=cfg.headdim,
        ngroups=cfg.ngroups,
        chunk_size=cfg.chunk_size,
        use_mem_eff_path=cfg.use_mem_eff_path,
    )
    spec.validate()
    return spec


def _resolve_device(cfg: RegionInterfaceConfig) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction="mean")


def _resolve_text_paths(cfg: RegionInterfaceConfig) -> tuple[Path, Path | None, Path]:
    corpora_root = CORPORA_ROOT
    if cfg.train_text_path:
        train_path = Path(cfg.train_text_path)
        val_path = Path(cfg.val_text_path) if cfg.val_text_path else None
        return train_path, val_path, corpora_root

    train_name, val_name = _BUNDLED_TEXT_CORPORA[cfg.text_corpus]
    train_path = corpora_root / train_name
    val_path = corpora_root / val_name
    if not train_path.exists():
        raise FileNotFoundError(
            f"bundled corpus file missing: {train_path}. "
            f"Run: python -m data.prepare_corpora --corpora {cfg.text_corpus}"
        )
    return train_path, val_path, corpora_root


def _resolve_tokenizer_path(cfg: RegionInterfaceConfig, *, train_path: Path) -> Path:
    if cfg.tokenizer_path:
        return Path(cfg.tokenizer_path)
    if not cfg.train_text_path:
        return TOKENIZERS_ROOT / f"{cfg.text_corpus}_sp_bpe_{cfg.vocab_size}.model"
    return train_path.parent / "tokenizers" / f"{train_path.stem}_sp_bpe_{cfg.vocab_size}.model"


def _build_corpora(
    cfg: RegionInterfaceConfig,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.data_mode == "synthetic":
        train_corpus = build_synthetic_corpus(
            num_sequences=cfg.train_sequences,
            seq_len=cfg.seq_len,
            vocab_size=cfg.vocab_size,
            seed=cfg.seed + 11,
            task=cfg.task,
            device=device,
        )
        val_corpus = build_synthetic_corpus(
            num_sequences=cfg.val_sequences,
            seq_len=cfg.seq_len,
            vocab_size=cfg.vocab_size,
            seed=cfg.seed + 29,
            task=cfg.task,
            device=device,
        )
        return train_corpus, val_corpus

    train_path, val_path, _ = _resolve_text_paths(cfg)
    if cfg.data_mode == "text_byte":
        train_stream = build_byte_stream(train_path, device=torch.device("cpu"))
        if cfg.val_text_path:
            val_stream = build_byte_stream(cfg.val_text_path, device=torch.device("cpu"))
            return train_stream, val_stream
        if val_path is not None and val_path.exists():
            val_stream = build_byte_stream(val_path, device=torch.device("cpu"))
            return train_stream, val_stream
        return split_stream_train_val(train_stream, val_fraction=cfg.val_split)

    tokenizer_path = _resolve_tokenizer_path(cfg, train_path=train_path)
    tokenizer = load_or_train_sentencepiece_bpe_tokenizer(
        tokenizer_path=tokenizer_path,
        train_path=train_path,
        vocab_size=cfg.vocab_size,
        train_bytes=cfg.tokenizer_train_bytes,
        force_train=cfg.train_tokenizer,
    )
    train_stream = build_sentencepiece_bpe_stream(train_path, tokenizer=tokenizer, device=torch.device("cpu"))
    if cfg.val_text_path:
        val_stream = build_sentencepiece_bpe_stream(cfg.val_text_path, tokenizer=tokenizer, device=torch.device("cpu"))
        return train_stream, val_stream
    if val_path is not None and val_path.exists():
        val_stream = build_sentencepiece_bpe_stream(val_path, tokenizer=tokenizer, device=torch.device("cpu"))
        return train_stream, val_stream
    return split_stream_train_val(train_stream, val_fraction=cfg.val_split)


def _sample_batch_any(
    *,
    cfg: RegionInterfaceConfig,
    corpus: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.data_mode == "synthetic":
        return sample_batch(corpus, batch_size=batch_size, generator=generator, device=device)
    return sample_batch_stream(
        corpus,
        batch_size=batch_size,
        seq_len=cfg.seq_len,
        generator=generator,
        device=device,
    )


def _count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _infer_data_info(cfg: RegionInterfaceConfig, *, train_corpus: torch.Tensor, val_corpus: torch.Tensor) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "data_mode": cfg.data_mode,
        "text_corpus": cfg.text_corpus,
        "vocab_size": int(cfg.vocab_size),
        "seq_len": int(cfg.seq_len),
        "train_stream_length": int(train_corpus.numel()),
        "val_stream_length": int(val_corpus.numel()),
    }
    if cfg.data_mode == "synthetic":
        info["task"] = cfg.task
        info["train_sequences"] = int(cfg.train_sequences)
        info["val_sequences"] = int(cfg.val_sequences)
        return info

    train_path, val_path, _ = _resolve_text_paths(cfg)
    info["corpora_root"] = str(CORPORA_ROOT)
    info["train_path"] = str(train_path)
    info["val_path"] = str(val_path) if val_path is not None else ""
    info["train_file_bytes"] = int(train_path.stat().st_size) if train_path.exists() else 0
    info["val_file_bytes"] = int(val_path.stat().st_size) if (val_path is not None and val_path.exists()) else 0
    if cfg.data_mode == "text_bpe":
        tokenizer_path = _resolve_tokenizer_path(cfg, train_path=train_path)
        vocab_path = tokenizer_path.with_suffix(".vocab")
        info["tokenizer_path"] = str(tokenizer_path)
        info["tokenizer_exists"] = bool(tokenizer_path.exists())
        info["tokenizer_train_bytes"] = int(cfg.tokenizer_train_bytes)
        info["tokenizer_vocab_path"] = str(vocab_path)
        info["tokenizer_vocab_exists"] = bool(vocab_path.exists())
    return info


def _infer_model_info(cfg: RegionInterfaceConfig, *, model: nn.Module) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "model_class": type(model).__name__,
        "total_params": _count_parameters(model),
        "layers": int(cfg.layers),
        "dim": int(cfg.dim),
        "d_state": int(cfg.d_state),
        "dt_rank": cfg.dt_rank if cfg.dt_rank is not None else "auto",
    }
    if isinstance(model, NativeRegionInterfaceModel):
        info["n_regions"] = int(model.n_regions)
        info["region_ranges"] = [list(pair) for pair in model.region_ranges]
        info["component_params"] = {
            "embedding": _count_parameters(model.embedding),
            "blocks": int(sum(_count_parameters(block) for block in model.blocks)),
            "input_to_message": _count_parameters(model.input_to_message),
            "message_to_hidden": int(sum(_count_parameters(mod) for mod in model.message_to_hidden)),
            "hidden_to_message": int(sum(_count_parameters(mod) for mod in model.hidden_to_message)),
            "message_norm": int(sum(_count_parameters(mod) for mod in model.message_norm)),
            "message_alpha": int(model.message_alpha.numel()),
            "norm": _count_parameters(model.norm),
            "lm_head": _count_parameters(model.lm_head),
        }
    elif isinstance(model, ReferenceLM):
        info["component_params"] = {
            "embedding": _count_parameters(model.embedding),
            "blocks": int(sum(_count_parameters(block) for block in model.blocks)),
            "norm": _count_parameters(model.norm),
            "lm_head": _count_parameters(model.lm_head),
        }
    return info


def _write_run_metadata(
    *,
    cfg: RegionInterfaceConfig,
    run_dir: Path,
    model: nn.Module,
    train_corpus: torch.Tensor,
    val_corpus: torch.Tensor,
) -> None:
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "data_info.json").write_text(
        json.dumps(_infer_data_info(cfg, train_corpus=train_corpus, val_corpus=val_corpus), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "model_info.json").write_text(
        json.dumps(_infer_model_info(cfg, model=model), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, row: Dict[str, Any]) -> None:
    fieldnames = [
        "step",
        "split",
        "ce_loss",
        "message_norm",
        "scan_align",
        "tokens_per_s",
        "wall_time_s",
    ]
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _prepare_run_dir(cfg: RegionInterfaceConfig, regime_root: str) -> Path:
    root = Path(cfg.output_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = cfg.run_name if cfg.run_name else f"{regime_root}_{stamp}"
    out = root / base
    out.mkdir(parents=True, exist_ok=True)
    return out


def _evaluate_reference(
    *,
    cfg: RegionInterfaceConfig,
    model: nn.Module,
    val_corpus: torch.Tensor,
    eval_batches: int,
    generator: torch.Generator,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(eval_batches):
            xb, yb = _sample_batch_any(
                cfg=cfg,
                corpus=val_corpus,
                batch_size=cfg.batch_size,
                generator=generator,
                device=device,
            )
            logits = model(xb)
            losses.append(float(_next_token_loss(logits, yb).item()))
    return {"ce_loss": float(sum(losses) / max(1, len(losses)))}


def _assign_grads(params: list[torch.nn.Parameter], grads: list[torch.Tensor | None]) -> None:
    for p, g in zip(params, grads):
        if g is not None:
            p.grad = g.detach()


def _run_reference_backprop(cfg: RegionInterfaceConfig, *, run_dir: Path) -> Dict[str, Any]:
    device = _resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    model = ReferenceLM(vocab_size=cfg.vocab_size, backbone_spec=_make_backbone_spec(cfg)).to(
        device=device, dtype=torch.float32
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_model, weight_decay=cfg.weight_decay)
    train_corpus, val_corpus = _build_corpora(cfg, device=device)
    _write_run_metadata(cfg=cfg, run_dir=run_dir, model=model, train_corpus=train_corpus, val_corpus=val_corpus)

    gen_device = device if cfg.data_mode == "synthetic" else torch.device("cpu")
    train_gen = torch.Generator(device=gen_device)
    train_gen.manual_seed(cfg.seed + 101)
    eval_gen = torch.Generator(device=gen_device)
    eval_gen.manual_seed(cfg.seed + 202)

    metrics_csv = run_dir / "metrics.csv"
    metrics_jsonl = run_dir / "metrics.jsonl"
    final_train = float("nan")
    final_val = float("nan")
    best_val = float("inf")
    best_val_step = -1

    for step in range(1, cfg.steps + 1):
        t0 = time.perf_counter()
        model.train()
        xb, yb = _sample_batch_any(
            cfg=cfg,
            corpus=train_corpus,
            batch_size=cfg.batch_size,
            generator=train_gen,
            device=device,
        )
        logits = model(xb)
        ce_loss = _next_token_loss(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        ce_loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        t1 = time.perf_counter()

        final_train = float(ce_loss.item())
        row = {
            "step": step,
            "split": "train",
            "ce_loss": final_train,
            "message_norm": "",
            "scan_align": "",
            "tokens_per_s": float((cfg.batch_size * cfg.seq_len) / max(1e-9, (t1 - t0))),
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
            )
            final_val = float(val["ce_loss"])
            if final_val < best_val:
                best_val = final_val
                best_val_step = step
            val_row = {
                "step": step,
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

    summary = {
        "regime": "backprop_ref",
        "task": cfg.task,
        "steps": int(cfg.steps),
        "final_train_ce_loss": final_train,
        "final_val_ce_loss": final_val,
        "best_val_ce_loss": best_val,
        "best_val_step": best_val_step,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _run_native_region_interface(cfg: RegionInterfaceConfig, *, run_dir: Path) -> Dict[str, Any]:
    device = _resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    backbone_spec = _make_backbone_spec(cfg)
    model = NativeRegionInterfaceModel(
        vocab_size=cfg.vocab_size,
        region_size=cfg.region_size,
        message_dim=cfg.message_dim,
        backbone_spec=backbone_spec,
        message_hidden_dim=infer_message_hidden_dim(backbone_spec, cfg.message_hidden_dim),
        message_scale_init=cfg.message_scale_init,
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_model, weight_decay=cfg.weight_decay)
    train_corpus, val_corpus = _build_corpora(cfg, device=device)
    _write_run_metadata(cfg=cfg, run_dir=run_dir, model=model, train_corpus=train_corpus, val_corpus=val_corpus)

    gen_device = device if cfg.data_mode == "synthetic" else torch.device("cpu")
    train_gen = torch.Generator(device=gen_device)
    train_gen.manual_seed(cfg.seed + 101)
    eval_gen = torch.Generator(device=gen_device)
    eval_gen.manual_seed(cfg.seed + 202)

    metrics_csv = run_dir / "metrics.csv"
    metrics_jsonl = run_dir / "metrics.jsonl"
    final_train = float("nan")
    final_val = float("nan")
    best_val = float("inf")
    best_val_step = -1

    for step in range(1, cfg.steps + 1):
        t0 = time.perf_counter()
        model.train()
        xb, yb = _sample_batch_any(
            cfg=cfg,
            corpus=train_corpus,
            batch_size=cfg.batch_size,
            generator=train_gen,
            device=device,
        )
        logits, cache = model.forward_with_cache(xb)
        ce_loss = _next_token_loss(logits, yb)
        optimizer.zero_grad(set_to_none=True)

        region_messages = list(cache["region_messages"])
        region_ranges = list(cache["region_ranges"])
        n_regions = len(region_ranges)
        if n_regions == 0:
            raise RuntimeError("native_region_interface requires at least one region.")

        head_params = [p for p in list(model.norm.parameters()) + list(model.lm_head.parameters()) if p.requires_grad]
        head_grads = torch.autograd.grad(
            ce_loss,
            head_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        _assign_grads(head_params, list(head_grads))

        if n_regions == 1:
            g_msg_inputs = [
                torch.autograd.grad(
                    ce_loss,
                    region_messages[0],
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0].detach()
            ]
            interface_scan_rms = 0.0
        else:
            g_last_input = torch.autograd.grad(
                ce_loss,
                region_messages[-2],
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0].detach()
            mats = model.materialize_interface_pullback_mats(cache)
            g_msg_inputs = model.interface_vjp_scan_from_last_input_seed(mats, g_last_input)
            g_manual = [torch.zeros_like(g_last_input) for _ in range(n_regions)]
            g_manual[-1] = g_last_input
            for ridx in reversed(range(n_regions - 1)):
                g_manual[ridx] = torch.einsum("bij,bj->bi", mats[ridx], g_manual[ridx + 1])
            diffs = [((a - b) ** 2).mean() for a, b in zip(g_msg_inputs, g_manual)]
            interface_scan_rms = float(torch.sqrt(torch.stack(diffs).mean()).item())

        input_params = [p for p in model.input_to_message.parameters() if p.requires_grad]
        input_grads = torch.autograd.grad(
            region_messages[0],
            input_params,
            grad_outputs=g_msg_inputs[0],
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        _assign_grads(input_params, list(input_grads))

        for ridx, (start, end) in enumerate(region_ranges):
            region_params: list[torch.nn.Parameter] = []
            for li in range(start, end):
                region_params.extend([p for p in model.blocks[li].parameters() if p.requires_grad])
            region_params.extend([p for p in model.message_to_hidden[ridx].parameters() if p.requires_grad])
            region_params.extend([p for p in model.hidden_to_message[ridx].parameters() if p.requires_grad])
            region_params.extend([p for p in model.message_norm[ridx].parameters() if p.requires_grad])

            if ridx < n_regions - 1:
                grads = torch.autograd.grad(
                    region_messages[ridx + 1],
                    region_params,
                    grad_outputs=g_msg_inputs[ridx + 1],
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
            else:
                grads = torch.autograd.grad(
                    ce_loss,
                    region_params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
            _assign_grads(region_params, list(grads))

        shared_params = [model.embedding.weight, model.message_alpha]
        shared_grads = torch.autograd.grad(
            ce_loss,
            shared_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        _assign_grads(shared_params, list(shared_grads))

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        t1 = time.perf_counter()

        final_train = float(ce_loss.item())
        msgs = cache.get("region_messages", [])
        msg_norm = float(torch.stack([m.norm(dim=-1).mean() for m in msgs]).mean().item()) if msgs else 0.0
        row = {
            "step": step,
            "split": "train",
            "ce_loss": final_train,
            "message_norm": msg_norm,
            "scan_align": interface_scan_rms,
            "tokens_per_s": float((cfg.batch_size * cfg.seq_len) / max(1e-9, (t1 - t0))),
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
            )
            final_val = float(val["ce_loss"])
            if final_val < best_val:
                best_val = final_val
                best_val_step = step
            val_row = {
                "step": step,
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

    summary = {
        "regime": "native_region_interface",
        "task": cfg.task,
        "steps": int(cfg.steps),
        "final_train_ce_loss": final_train,
        "final_val_ce_loss": final_val,
        "best_val_ce_loss": best_val,
        "best_val_step": best_val_step,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_training(cfg: RegionInterfaceConfig) -> Dict[str, Any]:
    regimes = ["native_region_interface"]
    root_dir = _prepare_run_dir(cfg, regime_root=cfg.regime)
    runs = []
    if cfg.include_reference_run:
        ref_dir = root_dir / "backprop_ref"
        ref_dir.mkdir(parents=True, exist_ok=True)
        runs.append(_run_reference_backprop(cfg, run_dir=ref_dir))
    for regime in regimes:
        run_dir = root_dir / regime
        run_dir.mkdir(parents=True, exist_ok=True)
        runs.append(_run_native_region_interface(cfg, run_dir=run_dir))
    summary = {"regime": cfg.regime, "root_dir": str(root_dir), "runs": runs}
    (root_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    cfg = _from_args(_build_arg_parser().parse_args())
    summary = run_training(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
