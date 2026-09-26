from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict

DENSE_VARIANT = "dense"
LBI_VARIANT = "lbi"
ALL_VARIANTS = "all"
VARIANTS = (DENSE_VARIANT, LBI_VARIANT)


def normalize_model_variant(value: str) -> str:
    value = str(value).strip()
    if value not in VARIANTS:
        raise ValueError(f"variant must be one of: {', '.join((ALL_VARIANTS, *VARIANTS))}; got {value!r}")
    return value


def parse_model_variants(value: str) -> tuple[str, ...]:
    value = str(value).strip()
    if not value or value == ALL_VARIANTS:
        return VARIANTS
    deduped: list[str] = []
    for part in value.split(","):
        if part.strip() and normalize_model_variant(part) not in deduped:
            deduped.append(normalize_model_variant(part))
    return tuple(deduped) or VARIANTS


def resolve_model_variants(cfg: object) -> tuple[str, ...]:
    variants_value = str(getattr(cfg, "variants", "")).strip()
    if variants_value:
        return parse_model_variants(variants_value)
    return parse_model_variants(str(getattr(cfg, "regime", ALL_VARIANTS)))


def output_name_for_variant(variant: str) -> str:
    return normalize_model_variant(variant)


@dataclass
class LBITrainingConfig:
    """Training configuration. Defaults are the paper's settings wherever the
    paper fixes one; model sizes and step counts come from the launcher."""

    variants: str = ""  # dense | lbi | dense,lbi; empty uses regime
    regime: str = "all"
    backbone: str = "transformer"
    layer_types: str = ""  # hybrid only: comma list per layer; empty = 3x mamba3 + 1x transformer repeating
    seed: int = 7
    device: str = "auto"  # auto | cpu | cuda
    dtype: str = "float32"
    output_dir: str = "out/region_interface"
    checkpoint_root: str = ""
    run_name: str = ""
    resume_from: str = ""
    init_from: str = ""
    # Data: FineWeb-Edu token shards under the LLaMA 32k tokenizer, or explicit paths.
    text_corpus: str = "fineweb_edu"
    train_text_path: str = ""
    val_text_path: str = ""
    tokenizer_path: str = ""
    token_shards_dir: str = ""
    vocab_size: int = 32000
    tie_embeddings: bool = True
    seq_len: int = 64
    batch_size: int = 8
    steps: int = 200
    eval_every: int = 20
    eval_batches: int = 4
    log_every: int = 5
    save_every: int = 5000
    save_checkpoints: bool = True
    lr_model: float = 1e-3
    lr_schedule: str = "cosine"  # constant | cosine | linear
    warmup_steps: int = 0
    # Decay horizon for the schedule; 0 follows `steps`. A truncated run set to
    # a longer run's step count follows that run's learning-rate trajectory.
    lr_schedule_steps: int = 0
    min_lr_ratio: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    layers: int = 4
    dim: int = 64
    d_state: int = 8
    expand: int = 2
    d_conv: int = 4
    headdim: int = 128
    ngroups: int = 1
    chunk_size: int = 256
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
    # Interface: two blocks per region, rank-16 vector MLP interface.
    region_size: int = 2
    message_dim: int = 16
    message_hidden_dim: int = 0
    message_scale_init: float = 0.5
    interface_type: str = "vector_mlp"
    # A_k construction: "forward" = fused forward-mode construction; "graph" =
    # autograd through the retained forward graph.
    interface_jacobian_mode: str = "graph"
    # LBI backward engine: "scan" = the region-decomposed scan engine;
    # "autograd" = end-to-end backprop.
    lbi_backward: str = "scan"
    # native_backward: native local VJPs over a graph-free bf16 forward, with
    # fp32 master weights in the optimizer.
    native_backward: bool = False
    # N-step window: the canvas (token-embedding) gradient is applied once per
    # N optimizer steps as the window mean, the canvas parameters untouched in
    # between. 0 = every step.
    canvas_grad_window: int = 0
    # Learning-rate multiplier for the windowed canvas parameter group. 1 = none.
    canvas_grad_window_lr_mult: float = 1.0

    def __post_init__(self) -> None:
        validate_config(self)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the dense and bounded-interface language models.")
    p.add_argument("--variants", type=str, default="all")
    p.add_argument("--backbone", type=str, default="transformer")
    p.add_argument("--layer-types", type=str, default="")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--output-dir", type=str, default="out/region_interface")
    p.add_argument("--checkpoint-root", type=str, default="")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--resume-from", type=str, default="")
    p.add_argument("--init-from", type=str, default="")
    p.add_argument("--text-corpus", type=str, default="fineweb_edu")
    p.add_argument("--train-text-path", type=str, default="")
    p.add_argument("--val-text-path", type=str, default="")
    p.add_argument("--tokenizer-path", type=str, default="")
    p.add_argument("--token-shards-dir", type=str, default="")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--tie-embeddings", dest="tie_embeddings", action="store_true")
    p.add_argument("--no-tie-embeddings", dest="tie_embeddings", action="store_false")
    p.set_defaults(tie_embeddings=True)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--save-checkpoints", dest="save_checkpoints", action="store_true")
    p.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false")
    p.set_defaults(save_checkpoints=True)
    p.add_argument("--lr-model", type=float, default=1e-3)
    p.add_argument("--lr-schedule", type=str, default="cosine")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--lr-schedule-steps", type=int, default=0)
    p.add_argument("--min-lr-ratio", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--d-state", type=int, default=8)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--headdim", type=int, default=128)
    p.add_argument("--ngroups", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=256)
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
    p.add_argument("--region-size", type=int, default=2)
    p.add_argument("--message-dim", "--interface-rank", dest="message_dim", type=int, default=16,
                   help="interface rank r")
    p.add_argument("--message-hidden-dim", type=int, default=0)
    p.add_argument("--message-scale-init", type=float, default=0.5)
    p.add_argument("--interface-type", type=str, default="vector_mlp")
    p.add_argument("--interface-jacobian-mode", type=str, default="graph")
    p.add_argument("--native-backward", action="store_true")
    p.add_argument("--lbi-backward", "--gradient-engine", dest="lbi_backward", type=str, default="scan",
                   choices=["scan", "autograd"],
                   help="gradient engine: scan = region-parallel construction + suffix scan; autograd = end-to-end reference")
    p.add_argument("--canvas-grad-window", type=int, default=0)
    p.add_argument("--canvas-grad-window-lr-mult", type=float, default=1.0)
    return p


def validate_config(cfg: LBITrainingConfig) -> None:
    resolve_model_variants(cfg)
    if cfg.backbone not in {"mamba3", "transformer", "hybrid"}:
        raise ValueError("backbone must be one of: mamba3, transformer, hybrid")
    if cfg.dtype not in {"float32", "bfloat16"}:
        raise ValueError("dtype must be one of: float32, bfloat16")
    if cfg.resume_from and cfg.init_from:
        raise ValueError("resume_from and init_from are mutually exclusive")
    if cfg.text_corpus != "fineweb_edu" and not cfg.train_text_path:
        raise ValueError("text_corpus must be fineweb_edu unless train_text_path is set")
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
    if cfg.lr_schedule_steps < 0:
        raise ValueError("lr_schedule_steps must be >= 0")
    if 0 < cfg.lr_schedule_steps < cfg.warmup_steps:
        raise ValueError("lr_schedule_steps must be 0 or >= warmup_steps")
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
    if cfg.interface_type != "vector_mlp":
        raise ValueError(f"interface_type must be vector_mlp (the paper interface); got {cfg.interface_type!r}")
    if cfg.interface_jacobian_mode not in {"graph", "forward"}:
        raise ValueError("interface_jacobian_mode must be one of: graph, forward")
    if cfg.lbi_backward not in {"scan", "autograd"}:
        raise ValueError("lbi_backward must be one of: scan, autograd")
    if cfg.lbi_backward == "autograd" and cfg.native_backward:
        raise ValueError("lbi_backward=autograd requires the graph forward (native_backward=False)")
    if cfg.canvas_grad_window < 0:
        raise ValueError("canvas_grad_window must be >= 0.")
    if cfg.canvas_grad_window_lr_mult <= 0:
        raise ValueError("canvas_grad_window_lr_mult must be > 0.")
    if cfg.canvas_grad_window_lr_mult != 1.0 and cfg.canvas_grad_window <= 0:
        raise ValueError("canvas_grad_window_lr_mult requires canvas_grad_window > 0.")
    if cfg.backbone in {"mamba3", "hybrid"}:
        if cfg.dtype != "bfloat16":
            raise ValueError("mamba3 requires dtype=bfloat16")
        if cfg.device == "cpu":
            raise ValueError("mamba3 requires CUDA")
        if cfg.d_state < 16:
            raise ValueError("mamba3 requires d_state >= 16")
        if cfg.headdim < 16:
            raise ValueError("mamba3 requires headdim >= 16")
        if cfg.chunk_size < 16:
            raise ValueError("mamba3 requires chunk_size >= 16")


def config_from_args(args: argparse.Namespace) -> LBITrainingConfig:
    return LBITrainingConfig(
        variants=args.variants,
        regime=args.variants,
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
        text_corpus=args.text_corpus,
        train_text_path=args.train_text_path,
        val_text_path=args.val_text_path,
        tokenizer_path=args.tokenizer_path,
        token_shards_dir=args.token_shards_dir,
        vocab_size=args.vocab_size,
        tie_embeddings=args.tie_embeddings,
        seq_len=args.seq_len,
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
        lr_schedule_steps=args.lr_schedule_steps,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        layers=args.layers,
        dim=args.dim,
        d_state=args.d_state,
        expand=args.expand,
        d_conv=args.d_conv,
        headdim=args.headdim,
        ngroups=args.ngroups,
        chunk_size=args.chunk_size,
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
        region_size=args.region_size,
        message_dim=args.message_dim,
        message_hidden_dim=args.message_hidden_dim,
        message_scale_init=args.message_scale_init,
        interface_type=args.interface_type,
        interface_jacobian_mode=args.interface_jacobian_mode,
        native_backward=args.native_backward,
        lbi_backward=args.lbi_backward,
        canvas_grad_window=args.canvas_grad_window,
        canvas_grad_window_lr_mult=args.canvas_grad_window_lr_mult,
    )
