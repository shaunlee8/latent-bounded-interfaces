from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.general import BackboneSpec, ReferenceLM, infer_message_hidden_dim
from data.bpe_tokenizer import build_token_stream, load_text_tokenizer
from data.data_paths import CORPORA_ROOT, LLAMA31_TOKENIZER_ROOT, TOKENIZERS_ROOT
from data.lm_data import build_byte_stream, sample_batch_stream, split_stream_train_val
from data.token_shards import TokenShardCorpus, corpus_numel, sample_batch_token_shards
from data.train_data import build_synthetic_corpus, sample_batch
from models.native_region_interface import NativeRegionInterfaceModel

_BUNDLED_TEXT_CORPORA: Dict[str, tuple[str, str]] = {
    "tiny_shakespeare": ("tiny_shakespeare_train.txt", "tiny_shakespeare_val.txt"),
    "enwik8": ("enwik8_train.bin", "enwik8_val.bin"),
    "wikitext103_raw": ("wikitext103_raw_train.txt", "wikitext103_raw_val.txt"),
    "tinystories": ("tinystories_train.txt", "tinystories_val.txt"),
    "fineweb_edu": ("fineweb_edu/fineweb_edu_train.txt", "fineweb_edu/fineweb_edu_val.txt"),
}

_EVAL_MESSAGE_ABLATION_MODES = ("zero_all", "noise", "mask")


@dataclass
class NativeBackwardResult:
    grad_map: Dict[str, torch.Tensor | None]
    interface_scan_rms: float


@dataclass
class RegionInterfaceConfig:
    regime: str = "all"  # backprop_ref | native_region_interface | all
    backbone: str = "mamba1"
    layer_types: str = ""
    seed: int = 7
    device: str = "auto"  # auto | cpu | cuda
    dtype: str = "float32"
    output_dir: str = "out/region_interface"
    checkpoint_root: str = ""
    run_name: str = ""
    resume_from: str = ""
    init_from: str = ""
    task: str = "copy"
    data_mode: str = "synthetic"  # synthetic | text_byte | text_bpe | text_bpe_sharded
    text_corpus: str = "tiny_shakespeare"
    train_text_path: str = ""
    val_text_path: str = ""
    val_split: float = 0.1
    tokenizer_path: str = ""
    tokenizer_type: str = "sentencepiece"
    token_shards_dir: str = ""
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
    save_checkpoints: bool = True
    lr_model: float = 1e-3
    lr_schedule: str = "constant"  # constant | cosine | linear
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
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
    n_heads: int = 8
    n_kv_heads: int = 0
    mlp_ratio: float = 4.0
    d_intermediate: int = 0
    rope_base: float = 10000.0
    attn_head_dim: int = 0
    softmax_scale: float = 0.0
    rope_interleaved: bool = False
    use_flash_attn: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    include_reference_run: bool = True
    region_size: int = 2
    message_dim: int = 64
    message_hidden_dim: int = 0
    message_scale_init: float = 0.5
    eval_message_ablation: str = "none"
    eval_all_message_ablations: bool = False
    message_noise_std: float = 1.0
    message_mask_keep_prob: float = 0.5

    def __post_init__(self) -> None:
        _validate_cfg(self)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Temporary compatibility alias while the new repo settles.
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train bounded region interfaces on a dense Mamba backbone.")
    p.add_argument("--regime", type=str, default="all")
    p.add_argument("--backbone", type=str, default="mamba1")
    p.add_argument("--layer-types", type=str, default="")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--output-dir", type=str, default="out/region_interface")
    p.add_argument("--checkpoint-root", type=str, default="")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--resume-from", type=str, default="")
    p.add_argument("--init-from", type=str, default="")
    p.add_argument("--task", type=str, default="copy")
    p.add_argument("--data-mode", type=str, default="synthetic")
    p.add_argument("--text-corpus", type=str, default="tiny_shakespeare")
    p.add_argument("--train-text-path", type=str, default="")
    p.add_argument("--val-text-path", type=str, default="")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--tokenizer-path", type=str, default="")
    p.add_argument("--tokenizer-type", type=str, default="sentencepiece")
    p.add_argument("--token-shards-dir", type=str, default="")
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
    p.add_argument("--save-checkpoints", dest="save_checkpoints", action="store_true")
    p.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false")
    p.set_defaults(save_checkpoints=True)
    p.add_argument("--lr-model", type=float, default=1e-3)
    p.add_argument("--lr-schedule", type=str, default="constant")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--min-lr-ratio", type=float, default=0.0)
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
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-kv-heads", type=int, default=0)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--d-intermediate", type=int, default=0)
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--attn-head-dim", type=int, default=0)
    p.add_argument("--softmax-scale", type=float, default=0.0)
    p.add_argument("--rope-interleaved", action="store_true")
    p.add_argument("--use-flash-attn", dest="use_flash_attn", action="store_true")
    p.add_argument("--no-flash-attn", dest="use_flash_attn", action="store_false")
    p.set_defaults(use_flash_attn=True)
    p.add_argument("--residual-in-fp32", dest="residual_in_fp32", action="store_true")
    p.add_argument("--no-residual-in-fp32", dest="residual_in_fp32", action="store_false")
    p.set_defaults(residual_in_fp32=True)
    p.add_argument("--fused-add-norm", dest="fused_add_norm", action="store_true")
    p.add_argument("--no-fused-add-norm", dest="fused_add_norm", action="store_false")
    p.set_defaults(fused_add_norm=True)
    p.add_argument("--include-reference-run", action="store_true")
    p.add_argument("--no-reference-run", dest="include_reference_run", action="store_false")
    p.set_defaults(include_reference_run=True)
    p.add_argument("--region-size", type=int, default=2)
    p.add_argument("--message-dim", type=int, default=64)
    p.add_argument("--message-hidden-dim", type=int, default=0)
    p.add_argument("--message-scale-init", type=float, default=0.5)
    p.add_argument("--eval-message-ablation", type=str, default="none")
    p.add_argument("--eval-all-message-ablations", action="store_true")
    p.add_argument("--message-noise-std", type=float, default=1.0)
    p.add_argument("--message-mask-keep-prob", type=float, default=0.5)
    return p


def _validate_cfg(cfg: RegionInterfaceConfig) -> None:
    if cfg.regime not in {"backprop_ref", "native_region_interface", "all"}:
        raise ValueError("regime must be one of: backprop_ref, native_region_interface, all")
    if cfg.backbone not in {"mamba1", "mamba2", "mamba3", "transformer", "hybrid"}:
        raise ValueError("backbone must be one of: mamba1, mamba2, mamba3, transformer, hybrid")
    if cfg.backbone == "hybrid" and not cfg.layer_types.strip():
        raise ValueError("hybrid backbone requires --layer-types")
    if cfg.dtype not in {"float32", "bfloat16"}:
        raise ValueError("dtype must be one of: float32, bfloat16")
    if cfg.resume_from and cfg.init_from:
        raise ValueError("resume_from and init_from are mutually exclusive")
    if cfg.data_mode not in {"synthetic", "text_byte", "text_bpe", "text_bpe_sharded"}:
        raise ValueError("data_mode must be one of: synthetic, text_byte, text_bpe, text_bpe_sharded")
    if cfg.data_mode == "text_byte":
        if cfg.vocab_size != 256:
            raise ValueError("text_byte mode requires vocab_size=256")
        if not (0.0 < cfg.val_split < 1.0):
            raise ValueError("val_split must be in (0,1)")
    if cfg.data_mode in {"text_bpe", "text_bpe_sharded"}:
        if cfg.tokenizer_type not in {"sentencepiece", "llama31"}:
            raise ValueError("tokenizer_type must be one of: sentencepiece, llama31")
        if cfg.tokenizer_type == "sentencepiece" and cfg.vocab_size <= 256:
            raise ValueError(f"{cfg.data_mode} mode requires vocab_size > 256")
        if cfg.data_mode == "text_bpe":
            if cfg.tokenizer_type == "sentencepiece" and cfg.tokenizer_train_bytes <= 0:
                raise ValueError("tokenizer_train_bytes must be > 0")
            if not (0.0 < cfg.val_split < 1.0):
                raise ValueError("val_split must be in (0,1)")
        if cfg.tokenizer_type == "llama31" and cfg.train_tokenizer:
            raise ValueError("llama31 tokenizer does not support --train-tokenizer")
    if cfg.data_mode != "synthetic" and (not cfg.train_text_path) and cfg.text_corpus not in _BUNDLED_TEXT_CORPORA:
        allowed = ", ".join(sorted(_BUNDLED_TEXT_CORPORA.keys()))
        raise ValueError(f"text_corpus must be one of: {allowed} when train_text_path is not set")
    if cfg.text_corpus == "fineweb_edu" and cfg.data_mode != "text_bpe_sharded":
        raise ValueError("fineweb_edu currently requires data_mode=text_bpe_sharded")
    if cfg.seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if cfg.steps <= 0:
        raise ValueError("steps must be > 0")
    if cfg.eval_every <= 0:
        raise ValueError("eval_every must be > 0")
    if cfg.log_every <= 0:
        raise ValueError("log_every must be > 0")
    if cfg.save_checkpoints and cfg.save_every <= 0:
        raise ValueError("save_every must be > 0 when save_checkpoints is enabled")
    if cfg.lr_model <= 0.0:
        raise ValueError("lr_model must be > 0")
    if cfg.lr_schedule not in {"constant", "cosine", "linear"}:
        raise ValueError("lr_schedule must be one of: constant, cosine, linear")
    if cfg.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if cfg.warmup_steps > cfg.steps:
        raise ValueError("warmup_steps must be <= steps")
    if cfg.min_lr_ratio < 0.0 or cfg.min_lr_ratio > 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if cfg.region_size <= 0:
        raise ValueError("region_size must be > 0.")
    if cfg.message_dim <= 0:
        raise ValueError("message_dim must be > 0.")
    if cfg.message_hidden_dim < 0:
        raise ValueError("message_hidden_dim must be >= 0.")
    if cfg.message_scale_init <= 0.0:
        raise ValueError("message_scale_init must be > 0.")
    if cfg.eval_message_ablation not in {"none", *_EVAL_MESSAGE_ABLATION_MODES}:
        raise ValueError("eval_message_ablation must be one of: none, zero_all, noise, mask")
    if cfg.message_noise_std < 0.0:
        raise ValueError("message_noise_std must be >= 0.")
    if cfg.message_mask_keep_prob <= 0.0 or cfg.message_mask_keep_prob > 1.0:
        raise ValueError("message_mask_keep_prob must be in (0, 1].")
    hybrid_layer_types = tuple(t.strip() for t in cfg.layer_types.split(",") if t.strip()) if cfg.layer_types else ()
    needs_mamba3_constraints = cfg.backbone == "mamba3" or (cfg.backbone == "hybrid" and "mamba3" in hybrid_layer_types)
    if needs_mamba3_constraints:
        if cfg.dtype != "bfloat16":
            raise ValueError("mamba3 currently requires dtype=bfloat16")
        if cfg.device == "cpu":
            raise ValueError("mamba3 currently requires CUDA")
        if cfg.d_state < 16:
            raise ValueError("mamba3 currently requires d_state >= 16")
        if cfg.headdim < 16:
            raise ValueError("mamba3 currently requires headdim >= 16")
        if cfg.chunk_size < 16:
            raise ValueError("mamba3 currently requires chunk_size >= 16")


def _from_args(args: argparse.Namespace) -> RegionInterfaceConfig:
    return RegionInterfaceConfig(
        regime=args.regime,
        backbone=args.backbone,
        layer_types=args.layer_types,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        output_dir=args.output_dir,
        checkpoint_root=args.checkpoint_root,
        run_name=args.run_name,
        resume_from=args.resume_from,
        init_from=args.init_from,
        task=args.task,
        data_mode=args.data_mode,
        text_corpus=args.text_corpus,
        train_text_path=args.train_text_path,
        val_text_path=args.val_text_path,
        val_split=args.val_split,
        tokenizer_path=args.tokenizer_path,
        tokenizer_type=args.tokenizer_type,
        token_shards_dir=args.token_shards_dir,
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
        save_checkpoints=args.save_checkpoints,
        lr_model=args.lr_model,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
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
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        mlp_ratio=args.mlp_ratio,
        d_intermediate=args.d_intermediate,
        rope_base=args.rope_base,
        attn_head_dim=args.attn_head_dim,
        softmax_scale=args.softmax_scale,
        rope_interleaved=args.rope_interleaved,
        use_flash_attn=args.use_flash_attn,
        residual_in_fp32=args.residual_in_fp32,
        fused_add_norm=args.fused_add_norm,
        include_reference_run=args.include_reference_run,
        region_size=args.region_size,
        message_dim=args.message_dim,
        message_hidden_dim=args.message_hidden_dim,
        message_scale_init=args.message_scale_init,
        eval_message_ablation=args.eval_message_ablation,
        eval_all_message_ablations=args.eval_all_message_ablations,
        message_noise_std=args.message_noise_std,
        message_mask_keep_prob=args.message_mask_keep_prob,
    )


def _make_backbone_spec(cfg: RegionInterfaceConfig) -> BackboneSpec:
    layer_types = tuple(t.strip() for t in cfg.layer_types.split(",") if t.strip())
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
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        mlp_ratio=cfg.mlp_ratio,
        d_intermediate=cfg.d_intermediate,
        rope_base=cfg.rope_base,
        attn_head_dim=cfg.attn_head_dim,
        softmax_scale=cfg.softmax_scale,
        rope_interleaved=cfg.rope_interleaved,
        use_flash_attn=cfg.use_flash_attn,
        residual_in_fp32=cfg.residual_in_fp32,
        fused_add_norm=cfg.fused_add_norm,
        layer_types=layer_types,
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




def _autocast_context(cfg: RegionInterfaceConfig, device: torch.device):
    if cfg.dtype == "bfloat16":
        if device.type != "cuda":
            raise RuntimeError("bfloat16 currently requires CUDA")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

def _next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab = logits.size(-1)
    return F.cross_entropy(logits.float().reshape(-1, vocab), targets.reshape(-1), reduction="mean")


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
    if cfg.tokenizer_type == "llama31":
        legacy = TOKENIZERS_ROOT / "llama31"
        if LLAMA31_TOKENIZER_ROOT.exists():
            return LLAMA31_TOKENIZER_ROOT
        if legacy.exists():
            return legacy
        return LLAMA31_TOKENIZER_ROOT
    if not cfg.train_text_path:
        return TOKENIZERS_ROOT / f"{cfg.text_corpus}_sp_bpe_{cfg.vocab_size}.model"
    return train_path.parent / "tokenizers" / f"{train_path.stem}_sp_bpe_{cfg.vocab_size}.model"


def _resolve_token_shards_dir(cfg: RegionInterfaceConfig, *, train_path: Path) -> Path:
    tag = "llama31" if cfg.tokenizer_type == "llama31" else f"sp_bpe_{cfg.vocab_size}"
    if cfg.token_shards_dir:
        return Path(cfg.token_shards_dir)
    if not cfg.train_text_path:
        return CORPORA_ROOT / cfg.text_corpus / "tokens" / f"{cfg.text_corpus}_{tag}"
    return train_path.parent / "tokens" / f"{train_path.stem}_{tag}"


def _resolve_token_shard_manifests(cfg: RegionInterfaceConfig, *, train_path: Path) -> tuple[Path, Path]:
    shard_dir = _resolve_token_shards_dir(cfg, train_path=train_path)
    return shard_dir / "train_manifest.json", shard_dir / "val_manifest.json"


def _resolve_runtime_vocab_size(cfg: RegionInterfaceConfig) -> int:
    if cfg.data_mode not in {"text_bpe", "text_bpe_sharded"} or cfg.tokenizer_type != "llama31":
        return int(cfg.vocab_size)
    train_path, _, _ = _resolve_text_paths(cfg)
    tokenizer_path = _resolve_tokenizer_path(cfg, train_path=train_path)
    tokenizer = load_text_tokenizer(
        tokenizer_type=cfg.tokenizer_type,
        tokenizer_path=tokenizer_path,
        train_path=train_path,
        vocab_size=cfg.vocab_size,
        train_bytes=cfg.tokenizer_train_bytes,
        force_train=False,
    )
    actual_vocab_size = int(tokenizer.vocab_size)
    if cfg.vocab_size != actual_vocab_size:
        print(
            f"[tokenizer] overriding vocab_size from {cfg.vocab_size} to llama31 tokenizer vocab_size={actual_vocab_size} "
            f"from {tokenizer_path}",
            flush=True,
        )
    return actual_vocab_size


def _build_corpora(
    cfg: RegionInterfaceConfig,
    *,
    device: torch.device,
) -> tuple[torch.Tensor | TokenShardCorpus, torch.Tensor | TokenShardCorpus]:
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

    if cfg.data_mode == "text_bpe_sharded":
        train_manifest, val_manifest = _resolve_token_shard_manifests(cfg, train_path=train_path)
        if not train_manifest.exists():
            raise FileNotFoundError(
                f"token shard manifest missing: {train_manifest}. "
                f"Run: python -m data.pretokenize_corpus --text-corpus {cfg.text_corpus} --vocab-size {cfg.vocab_size}"
            )
        if not val_manifest.exists():
            raise FileNotFoundError(
                f"token shard manifest missing: {val_manifest}. "
                f"Run: python -m data.pretokenize_corpus --text-corpus {cfg.text_corpus} --vocab-size {cfg.vocab_size}"
            )
        return TokenShardCorpus(train_manifest), TokenShardCorpus(val_manifest)

    tokenizer_path = _resolve_tokenizer_path(cfg, train_path=train_path)
    tokenizer = load_text_tokenizer(
        tokenizer_type=cfg.tokenizer_type,
        tokenizer_path=tokenizer_path,
        train_path=train_path,
        vocab_size=cfg.vocab_size,
        train_bytes=cfg.tokenizer_train_bytes,
        force_train=cfg.train_tokenizer,
    )
    train_stream = build_token_stream(train_path, tokenizer=tokenizer, device=torch.device("cpu"))
    if cfg.val_text_path:
        val_stream = build_token_stream(cfg.val_text_path, tokenizer=tokenizer, device=torch.device("cpu"))
        return train_stream, val_stream
    if val_path is not None and val_path.exists():
        val_stream = build_token_stream(val_path, tokenizer=tokenizer, device=torch.device("cpu"))
        return train_stream, val_stream
    return split_stream_train_val(train_stream, val_fraction=cfg.val_split)


def _sample_batch_any(
    *,
    cfg: RegionInterfaceConfig,
    corpus: torch.Tensor | TokenShardCorpus,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.data_mode == "synthetic":
        return sample_batch(corpus, batch_size=batch_size, generator=generator, device=device)
    if cfg.data_mode == "text_bpe_sharded":
        return sample_batch_token_shards(
            corpus,
            batch_size=batch_size,
            seq_len=cfg.seq_len,
            generator=generator,
            device=device,
        )
    return sample_batch_stream(
        corpus,
        batch_size=batch_size,
        seq_len=cfg.seq_len,
        generator=generator,
        device=device,
    )


def _count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _infer_data_info(
    cfg: RegionInterfaceConfig,
    *,
    train_corpus: torch.Tensor | TokenShardCorpus,
    val_corpus: torch.Tensor | TokenShardCorpus,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "data_mode": cfg.data_mode,
        "text_corpus": cfg.text_corpus,
        "vocab_size": int(cfg.vocab_size),
        "seq_len": int(cfg.seq_len),
        "train_stream_length": corpus_numel(train_corpus),
        "val_stream_length": corpus_numel(val_corpus),
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
    if cfg.data_mode in {"text_bpe", "text_bpe_sharded"}:
        tokenizer_path = _resolve_tokenizer_path(cfg, train_path=train_path)
        vocab_path = tokenizer_path.with_suffix(".vocab") if tokenizer_path.suffix == ".model" else Path("")
        info["tokenizer_path"] = str(tokenizer_path)
        info["tokenizer_type"] = cfg.tokenizer_type
        info["tokenizer_exists"] = bool(tokenizer_path.exists())
        info["tokenizer_vocab_path"] = str(vocab_path)
        info["tokenizer_vocab_exists"] = bool(vocab_path.exists()) if str(vocab_path) else False
    if cfg.data_mode == "text_bpe":
        info["tokenizer_train_bytes"] = int(cfg.tokenizer_train_bytes)
    if cfg.data_mode == "text_bpe_sharded":
        train_manifest, val_manifest = _resolve_token_shard_manifests(cfg, train_path=train_path)
        info["token_shards_dir"] = str(_resolve_token_shards_dir(cfg, train_path=train_path))
        info["train_manifest"] = str(train_manifest)
        info["train_manifest_exists"] = bool(train_manifest.exists())
        info["val_manifest"] = str(val_manifest)
        info["val_manifest_exists"] = bool(val_manifest.exists())
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
    train_corpus: torch.Tensor | TokenShardCorpus,
    val_corpus: torch.Tensor | TokenShardCorpus,
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
        "tokens_seen",
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


def _lr_multiplier(cfg: RegionInterfaceConfig, step: int) -> float:
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


def _scheduled_lr(cfg: RegionInterfaceConfig, step: int) -> float:
    return float(cfg.lr_model * _lr_multiplier(cfg, step))


def _checkpoint_dir(run_dir: Path, cfg: RegionInterfaceConfig) -> Path:
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


def _checkpoint_path_from_summary(summary_path: Path, *, preference: str) -> Path | None:
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


def _save_checkpoint(
    *,
    run_dir: Path,
    filename: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: RegionInterfaceConfig,
    device: torch.device,
    train_generator: torch.Generator,
    eval_generator: torch.Generator,
    step: int,
    tokens_seen: int,
    best_val_ce_loss: float,
    best_val_step: int,
    extra: Dict[str, Any] | None = None,
) -> Path:
    payload: Dict[str, Any] = {
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
    path = _checkpoint_dir(run_dir, cfg) / filename
    torch.save(payload, path)
    return path


def _load_checkpoint(path: Path, *, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)


def _resolve_checkpoint_path(
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
    if (source / "metrics.csv").exists():
        candidates.extend(
            [
                source / "checkpoints" / preferred_name,
                source / "checkpoints" / fallback_name,
            ]
        )
    else:
        candidates.extend(
            [
                source / regime / "checkpoints" / preferred_name,
                source / regime / "checkpoints" / fallback_name,
                source / "checkpoints" / preferred_name,
                source / "checkpoints" / fallback_name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if (source / "metrics.csv").exists():
        from_summary = _checkpoint_path_from_summary(source / "summary.json", preference=preference)
        if from_summary is not None:
            return from_summary
    else:
        from_regime_summary = _checkpoint_path_from_summary(source / regime / "summary.json", preference=preference)
        if from_regime_summary is not None:
            return from_regime_summary
        from_summary = _checkpoint_path_from_summary(source / "summary.json", preference=preference)
        if from_summary is not None:
            return from_summary
    return None


def _infer_existing_root_dir(load_from: str) -> Path:
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


def _maybe_restore_training_state(
    *,
    mode: str,
    load_from: str,
    regime: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_generator: torch.Generator,
    eval_generator: torch.Generator,
) -> Dict[str, Any]:
    if mode not in {"resume", "init"}:
        raise ValueError("mode must be one of: resume, init")
    checkpoint_path = _resolve_checkpoint_path(
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

    checkpoint = _load_checkpoint(checkpoint_path, device=device)
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


def _prepare_run_dir(cfg: RegionInterfaceConfig, regime_root: str) -> Path:
    root = Path(cfg.output_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = cfg.run_name if cfg.run_name else f"{regime_root}_{stamp}"
    out = root / base
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_eval_message_ablation_modes(cfg: RegionInterfaceConfig) -> list[str]:
    if cfg.eval_all_message_ablations:
        return list(_EVAL_MESSAGE_ABLATION_MODES)
    if cfg.eval_message_ablation == "none":
        return []
    return [cfg.eval_message_ablation]


def _evaluate_reference(
    *,
    cfg: RegionInterfaceConfig,
    model: nn.Module,
    val_corpus: torch.Tensor | TokenShardCorpus,
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
            with _autocast_context(cfg, device):
                logits = model(xb)
            losses.append(float(_next_token_loss(logits, yb).item()))
    return {"ce_loss": float(sum(losses) / max(1, len(losses)))}


def _evaluate_native(
    *,
    cfg: RegionInterfaceConfig,
    model: NativeRegionInterfaceModel,
    val_corpus: torch.Tensor | TokenShardCorpus,
    eval_batches: int,
    generator: torch.Generator,
    device: torch.device,
    message_ablation: str = "none",
    ablation_generator: torch.Generator | None = None,
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
            with _autocast_context(cfg, device):
                logits, _ = model.forward_with_cache(
                    xb,
                    message_ablation=message_ablation,
                    message_noise_std=cfg.message_noise_std,
                    message_mask_keep_prob=cfg.message_mask_keep_prob,
                    ablation_generator=ablation_generator,
                )
            losses.append(float(_next_token_loss(logits, yb).item()))
    return {"ce_loss": float(sum(losses) / max(1, len(losses)))}


def _assign_grads(params: list[torch.nn.Parameter], grads: list[torch.Tensor | None]) -> None:
    for p, g in zip(params, grads):
        if g is not None:
            p.grad = g.detach()


def _collect_named_grads(model: nn.Module) -> Dict[str, torch.Tensor | None]:
    grad_map: Dict[str, torch.Tensor | None] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad_map[name] = None if param.grad is None else param.grad.detach().clone()
    return grad_map


def _assign_grad_map(model: nn.Module, grad_map: Dict[str, torch.Tensor | None]) -> None:
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = grad_map.get(name)
        param.grad = None if grad is None else grad.detach().clone()


def _store_named_grads(
    grad_map: Dict[str, torch.Tensor | None],
    param_name_map: Dict[int, str],
    params: list[torch.nn.Parameter],
    grads: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None],
) -> None:
    for param, grad in zip(params, grads):
        name = param_name_map[id(param)]
        grad_map[name] = None if grad is None else grad.detach().clone()


def _native_backward_step(
    model: NativeRegionInterfaceModel,
    *,
    ce_loss: torch.Tensor,
    cache: Dict[str, Any],
) -> NativeBackwardResult:
    param_name_map = {id(param): name for name, param in model.named_parameters() if param.requires_grad}
    grad_map: Dict[str, torch.Tensor | None] = {
        name: None for name, param in model.named_parameters() if param.requires_grad
    }

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
    _store_named_grads(grad_map, param_name_map, head_params, head_grads)

    if n_regions == 1:
        g_msg_inputs = [
            torch.autograd.grad(
                ce_loss,
                region_messages[0],
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0].detach().to(device=region_messages[0].device, dtype=region_messages[0].dtype)
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
        g_last_input = g_last_input.to(device=mats[0].device, dtype=mats[0].dtype)
        g_msg_inputs = model.interface_vjp_scan_from_last_input_seed(mats, g_last_input)
        g_manual = [torch.zeros_like(g_last_input) for _ in range(n_regions)]
        g_manual[-1] = g_last_input
        for ridx in reversed(range(n_regions - 1)):
            g_manual[ridx] = model._apply_pullback_mat(mats[ridx], g_manual[ridx + 1], out_dtype=g_manual[ridx + 1].dtype)
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
    _store_named_grads(grad_map, param_name_map, input_params, input_grads)

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
        _store_named_grads(grad_map, param_name_map, region_params, grads)

    shared_params = [model.embedding.weight, model.message_alpha]
    shared_grads = torch.autograd.grad(
        ce_loss,
        shared_params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    _store_named_grads(grad_map, param_name_map, shared_params, shared_grads)
    return NativeBackwardResult(grad_map=grad_map, interface_scan_rms=interface_scan_rms)


def _autograd_backward_step(model: nn.Module, *, ce_loss: torch.Tensor) -> Dict[str, torch.Tensor | None]:
    model.zero_grad(set_to_none=True)
    ce_loss.backward()
    return _collect_named_grads(model)


def _run_reference_backprop(cfg: RegionInterfaceConfig, *, run_dir: Path) -> Dict[str, Any]:
    device = _resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    model = ReferenceLM(vocab_size=cfg.vocab_size, backbone_spec=_make_backbone_spec(cfg)).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_model, weight_decay=cfg.weight_decay)
    train_corpus, val_corpus = _build_corpora(cfg, device=device)
    _write_run_metadata(cfg=cfg, run_dir=run_dir, model=model, train_corpus=train_corpus, val_corpus=val_corpus)

    gen_device = device if cfg.data_mode == "synthetic" else torch.device("cpu")
    train_gen = torch.Generator(device=gen_device)
    train_gen.manual_seed(cfg.seed + 101)
    eval_gen = torch.Generator(device=gen_device)
    eval_gen.manual_seed(cfg.seed + 202)
    restore = _maybe_restore_training_state(
        mode="resume" if cfg.resume_from else "init",
        load_from=cfg.resume_from or cfg.init_from,
        regime="backprop_ref",
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
        optimizer.zero_grad(set_to_none=True)
        ce_loss.backward()
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
                                "regime": "backprop_ref",
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
            _save_checkpoint(
                run_dir=run_dir,
                filename=f"step_{step:07d}.pt",
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
                extra={"regime": "backprop_ref", "train_ce_loss": final_train},
            )
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
                extra={"regime": "backprop_ref", "train_ce_loss": final_train},
            )
            latest_checkpoint_path = str(latest_path)

    summary = {
        "regime": "backprop_ref",
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
    restore = _maybe_restore_training_state(
        mode="resume" if cfg.resume_from else "init",
        load_from=cfg.resume_from or cfg.init_from,
        regime="native_region_interface",
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
        optimizer.zero_grad(set_to_none=True)
        backward_result = _native_backward_step(model, ce_loss=ce_loss, cache=cache)
        _assign_grad_map(model, backward_result.grad_map)
        interface_scan_rms = backward_result.interface_scan_rms

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        t1 = time.perf_counter()

        final_train = float(ce_loss.item())
        total_train_tokens = int(step * train_tokens_per_step)
        msgs = cache.get("region_messages", [])
        msg_norm = float(torch.stack([m.norm(dim=-1).mean() for m in msgs]).mean().item()) if msgs else 0.0
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
        if (step % cfg.log_every) == 0 or step == 1:
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
                                "regime": "native_region_interface",
                                "train_ce_loss": final_train,
                                "val_ce_loss": final_val,
                                "interface_scan_rms": interface_scan_rms,
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

            for ablation_idx, message_ablation in enumerate(_resolve_eval_message_ablation_modes(cfg)):
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
            _save_checkpoint(
                run_dir=run_dir,
                filename=f"step_{step:07d}.pt",
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
                    "regime": "native_region_interface",
                    "train_ce_loss": final_train,
                    "interface_scan_rms": interface_scan_rms,
                },
            )
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
                    "regime": "native_region_interface",
                    "train_ce_loss": final_train,
                    "interface_scan_rms": interface_scan_rms,
                },
            )
            latest_checkpoint_path = str(latest_path)

    summary = {
        "regime": "native_region_interface",
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
        "final_eval_message_ablations": final_eval_message_ablations,
        "resumed_from": resumed_from,
        "initialized_from": initialized_from,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_training(cfg: RegionInterfaceConfig) -> Dict[str, Any]:
    cfg.vocab_size = _resolve_runtime_vocab_size(cfg)
    root_dir = _infer_existing_root_dir(cfg.resume_from) if cfg.resume_from else _prepare_run_dir(cfg, regime_root=cfg.regime)
    root_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    active_regimes = (
        ["backprop_ref", "native_region_interface"]
        if cfg.regime == "all"
        else [cfg.regime]
    )
    for regime in active_regimes:
        run_dir = root_dir / regime
        run_dir.mkdir(parents=True, exist_ok=True)
        if regime == "backprop_ref":
            runs.append(_run_reference_backprop(cfg, run_dir=run_dir))
        elif regime == "native_region_interface":
            runs.append(_run_native_region_interface(cfg, run_dir=run_dir))
        else:
            raise AssertionError(f"unsupported active regime: {regime}")
    summary = {"regime": cfg.regime, "root_dir": str(root_dir), "runs": runs}
    (root_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    cfg = _from_args(_build_arg_parser().parse_args())
    summary = run_training(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
