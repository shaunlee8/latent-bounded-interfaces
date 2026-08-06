from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict

from train.data import BUNDLED_TEXT_CORPORA


EVAL_MESSAGE_ABLATION_MODES = ("zero_all", "noise", "mask")

DENSE_VARIANT = "dense"
LBI_VARIANT = "lbi"
ALL_VARIANTS = "all"
LEGACY_DENSE_REGIME = "backprop_ref"
LEGACY_LBI_REGIME = "native_region_interface"

# Regime constants accepted by command-line and checkpoint readers.
DENSE_REGIME = DENSE_VARIANT
LBI_REGIME = LBI_VARIANT
ALL_REGIME = ALL_VARIANTS

VARIANT_ALIASES = {
    DENSE_VARIANT: DENSE_VARIANT,
    LBI_VARIANT: LBI_VARIANT,
    LEGACY_DENSE_REGIME: DENSE_VARIANT,
    LEGACY_LBI_REGIME: LBI_VARIANT,
}
LEGACY_OUTPUT_NAME_BY_VARIANT = {
    DENSE_VARIANT: LEGACY_DENSE_REGIME,
    LBI_VARIANT: LEGACY_LBI_REGIME,
}
NEW_OUTPUT_NAME_BY_VARIANT = {
    DENSE_VARIANT: DENSE_VARIANT,
    LBI_VARIANT: LBI_VARIANT,
}


def normalize_model_variant(value: str) -> str:
    try:
        return VARIANT_ALIASES[str(value)]
    except KeyError as exc:
        allowed = ", ".join(sorted([ALL_VARIANTS, *VARIANT_ALIASES]))
        raise ValueError(f"variant must be one of: {allowed}") from exc


def parse_model_variants(value: str) -> tuple[str, ...]:
    value = str(value).strip()
    if not value or value == ALL_VARIANTS:
        return (DENSE_VARIANT, LBI_VARIANT)
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return (DENSE_VARIANT, LBI_VARIANT)
    variants = tuple(normalize_model_variant(part) for part in parts)
    deduped: list[str] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return tuple(deduped)


def resolve_model_variants(cfg: object) -> tuple[str, ...]:
    variants_value = str(getattr(cfg, "variants", "")).strip()
    if variants_value:
        return parse_model_variants(variants_value)
    return parse_model_variants(str(getattr(cfg, "regime", ALL_VARIANTS)))


def output_name_for_variant(variant: str) -> str:
    return NEW_OUTPUT_NAME_BY_VARIANT[normalize_model_variant(variant)]


def legacy_output_name_for_variant(variant: str) -> str:
    return LEGACY_OUTPUT_NAME_BY_VARIANT[normalize_model_variant(variant)]


def compatible_output_names_for_variant(variant: str) -> tuple[str, ...]:
    variant = normalize_model_variant(variant)
    return (output_name_for_variant(variant), legacy_output_name_for_variant(variant))


def variant_from_output_name(name: str) -> str:
    return normalize_model_variant(name)


def normalize_regime_name(regime: str) -> str:
    """Legacy helper returning the old output/checkpoint directory name."""
    if str(regime).strip() == ALL_VARIANTS:
        return ALL_VARIANTS
    return legacy_output_name_for_variant(regime)


def display_regime_name(regime: str) -> str:
    if str(regime).strip() == ALL_VARIANTS:
        return ALL_VARIANTS
    return normalize_model_variant(regime)


@dataclass
class LBITrainingConfig:
    variants: str = ""  # dense | lbi | dense,lbi; empty uses regime
    regime: str = "all"  # accepted aliases: backprop_ref | native_region_interface | all
    backbone: str = "transformer"
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
    tie_embeddings: bool = False
    seq_len: int = 64
    train_sequences: int = 2048
    val_sequences: int = 256
    batch_size: int = 8
    steps: int = 200
    eval_every: int = 20
    eval_batches: int = 4
    log_every: int = 5
    save_every: int = 5000
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
    message_dim: int = 2
    message_hidden_dim: int = 0
    message_scale_init: float = 0.5
    # Interface family: "vector_mlp" = mean-pool encode + broadcast MLP decode,
    # "attentive" = gated-pool encode + token-query slot-attention decode
    # (the default, with strict chunks and the bounded output readout below).
    interface_type: str = "attentive"
    interface_attn_dim: int = 64
    # Chunk-resolved attentive interface: >1 splits the sequence into chunks
    # carrying one rank-`message_dim` message each (state width chunks * r).
    # Strict mode reads only earlier chunks: a fully causal (adapted) model.
    interface_chunks: int = 4
    interface_chunks_strict: bool = True
    # Per-chunk norm on the boundary state: "layer" (LayerNorm; at width 2 it
    # is a.e. a sign function of the within-chunk difference) or "rms"
    # (direction-preserving RMS with learned scale).
    interface_chunk_norm: str = "layer"
    # Canvas upgrades: direct boundary-state taps into the readout stream, a
    # per-region diagonal canvas view, and a shared causal local-mixer conv
    # (kernel width; 0 disables).
    canvas_state_readout: bool = False
    canvas_region_view: bool = False
    canvas_local_mixer: int = 0
    # Full-width readout accumulation (tanh-bounded gates).
    canvas_output_readout: bool = True
    # A_k lowering: "forward" = the forward-mode construction (kernel path),
    # "native" = the reverse-mode native pullback, "graph"/"recompute" = the
    # autograd reverse constructions (oracles / fallback).
    interface_jacobian_mode: str = "graph"
    jacobian_basis_chunk: int = 1
    # LBI backward engine: "scan" = the region-decomposed scan engine (the
    # parallelizable path; pays the per-step A_k construction), "autograd" =
    # plain end-to-end backprop through the region graph (the cheapest exact
    # single-device engine; no A_k, no interface-Jacobian diagnostics).
    lbi_backward: str = "scan"
    # Compiled turn stages for the region forward (weights-as-arguments
    # graphs); pays a first-step compile, wins on long runs.
    compile_turn: bool = False
    # Native-backward training keeps fp32 master weights in the optimizer
    # (the model runs bf16); matches the dense arm's precision regime.
    fp32_master: bool = True
    # torch.compile the dense baseline model.
    compile_dense: bool = False
    # Autograd-free native backward: native interface pullback + native local VJP
    # over a forward that builds no region autograd graph (native_backward). With
    # trim_region_cache, per-layer forward caches are dropped and recomputed on
    # demand (store-vs-recompute; requires backend support, e.g. mamba3).
    native_backward: bool = False
    trim_region_cache: bool = False
    log_interface_jacobian_every: int = 0
    log_interface_jacobian_suffix: bool = False
    eval_message_ablation: str = "none"
    # Train-time message ablation: "zero_all" trains the ensemble control
    # (regions read the canvas only; the readout taps stay live).
    train_message_ablation: str = "none"
    eval_all_message_ablations: bool = False
    message_noise_std: float = 1.0
    message_mask_keep_prob: float = 0.5

    def __post_init__(self) -> None:
        validate_config(self)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train bounded region interfaces on a dense Mamba backbone.")
    p.add_argument("--variants", type=str, default="all")
    p.add_argument("--regime", type=str, default="", help="Legacy alias for --variants.")
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
    p.add_argument("--tie-embeddings", dest="tie_embeddings", action="store_true")
    p.add_argument("--no-tie-embeddings", dest="tie_embeddings", action="store_false")
    p.set_defaults(tie_embeddings=False)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--train-sequences", type=int, default=2048)
    p.add_argument("--val-sequences", type=int, default=256)
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
    p.add_argument("--interface-type", type=str, default="vector_mlp", choices=["vector_mlp", "attentive"])
    p.add_argument("--interface-attn-dim", type=int, default=64)
    p.add_argument("--canvas-state-readout", action="store_true")
    p.add_argument("--canvas-region-view", action="store_true")
    p.add_argument("--canvas-local-mixer", type=int, default=0)
    p.add_argument("--canvas-output-readout", action="store_true")
    p.add_argument("--interface-chunks", type=int, default=1)
    p.add_argument("--interface-chunks-strict", action="store_true")
    p.add_argument("--interface-chunk-norm", type=str, default="layer")
    p.add_argument("--interface-jacobian-mode", type=str, default="graph")
    p.add_argument("--jacobian-basis-chunk", type=int, default=1)
    p.add_argument("--native-backward", action="store_true")
    p.add_argument("--trim-region-cache", action="store_true")
    p.add_argument("--lbi-backward", type=str, default="scan", choices=["scan", "autograd"])
    p.add_argument("--compile-turn", action="store_true")
    p.add_argument("--no-fp32-master", action="store_true")
    p.add_argument("--compile-dense", action="store_true")
    p.add_argument("--log-interface-jacobian-every", type=int, default=0)
    p.add_argument("--log-interface-jacobian-suffix", action="store_true")
    p.add_argument("--eval-message-ablation", type=str, default="none")
    p.add_argument("--train-message-ablation", type=str, default="none")
    p.add_argument("--eval-all-message-ablations", action="store_true")
    p.add_argument("--message-noise-std", type=float, default=1.0)
    p.add_argument("--message-mask-keep-prob", type=float, default=0.5)
    return p


# Enumerate all invalid configurations.
def validate_config(cfg: LBITrainingConfig) -> None:
    resolve_model_variants(cfg)
    if cfg.backbone not in {"mamba2", "mamba3", "transformer", "hybrid"}:
        raise ValueError("backbone must be one of: mamba2, mamba3, transformer, hybrid")
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
        if cfg.tokenizer_type not in {"sentencepiece", "llama", "llama31"}:
            raise ValueError("tokenizer_type must be one of: sentencepiece, llama, llama31")
        if cfg.tokenizer_type == "sentencepiece" and cfg.vocab_size <= 256:
            raise ValueError(f"{cfg.data_mode} mode requires vocab_size > 256")
        if cfg.data_mode == "text_bpe":
            if cfg.tokenizer_type == "sentencepiece" and cfg.tokenizer_train_bytes <= 0:
                raise ValueError("tokenizer_train_bytes must be > 0")
            if not (0.0 < cfg.val_split < 1.0):
                raise ValueError("val_split must be in (0,1)")
        if cfg.tokenizer_type in {"llama", "llama31"} and cfg.train_tokenizer:
            raise ValueError(f"{cfg.tokenizer_type} tokenizer does not support --train-tokenizer")
    if cfg.data_mode != "synthetic" and (not cfg.train_text_path) and cfg.text_corpus not in BUNDLED_TEXT_CORPORA:
        allowed = ", ".join(sorted(BUNDLED_TEXT_CORPORA.keys()))
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
    if cfg.interface_type not in {"vector_mlp", "attentive"}:
        raise ValueError("interface_type must be one of: vector_mlp, attentive")
    if cfg.interface_attn_dim <= 0:
        raise ValueError("interface_attn_dim must be > 0.")
    if cfg.canvas_local_mixer < 0 or cfg.canvas_local_mixer == 1:
        raise ValueError("canvas_local_mixer must be 0 (off) or a kernel width >= 2.")
    if cfg.interface_chunks < 1:
        raise ValueError("interface_chunks must be >= 1.")
    if cfg.interface_chunks > 1:
        if cfg.interface_type != "attentive":
            raise ValueError("interface_chunks > 1 requires interface_type=attentive")
        if cfg.canvas_state_readout:
            raise ValueError("interface_chunks > 1 does not support canvas_state_readout (v1)")
        if cfg.seq_len % cfg.interface_chunks != 0:
            raise ValueError("interface_chunks must divide seq_len")
    if cfg.interface_chunks_strict and cfg.interface_chunks <= 1:
        raise ValueError("interface_chunks_strict requires interface_chunks > 1")
    if cfg.interface_chunk_norm not in {"layer", "rms"}:
        raise ValueError("interface_chunk_norm must be one of: layer, rms")
    if cfg.interface_chunk_norm != "layer" and cfg.interface_chunks <= 1:
        raise ValueError("interface_chunk_norm requires interface_chunks > 1")
    if cfg.interface_jacobian_mode not in {"graph", "recompute", "forward"}:
        raise ValueError("interface_jacobian_mode must be one of: graph, recompute, forward")
    if cfg.jacobian_basis_chunk <= 0:
        raise ValueError("jacobian_basis_chunk must be > 0.")
    if cfg.trim_region_cache and not cfg.native_backward:
        raise ValueError("trim_region_cache requires native_backward=True")
    if cfg.log_interface_jacobian_every < 0:
        raise ValueError("log_interface_jacobian_every must be >= 0.")
    if cfg.eval_message_ablation not in {"none", *EVAL_MESSAGE_ABLATION_MODES}:
        raise ValueError("eval_message_ablation must be one of: none, zero_all, noise, mask")
    if cfg.train_message_ablation not in {"none", *EVAL_MESSAGE_ABLATION_MODES}:
        raise ValueError("train_message_ablation must be one of: none, zero_all, noise, mask")
    if cfg.train_message_ablation != "none" and cfg.lbi_backward != "autograd":
        raise ValueError("train_message_ablation requires lbi_backward=autograd (the scan engines assume live states)")
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


def config_from_args(args: argparse.Namespace) -> LBITrainingConfig:
    return LBITrainingConfig(
        variants=args.variants if not args.regime else args.regime,
        regime=args.regime or args.variants,
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
        tie_embeddings=args.tie_embeddings,
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
        interface_type=args.interface_type,
        interface_attn_dim=args.interface_attn_dim,
        canvas_state_readout=args.canvas_state_readout,
        canvas_region_view=args.canvas_region_view,
        canvas_local_mixer=args.canvas_local_mixer,
        canvas_output_readout=args.canvas_output_readout,
        interface_chunks=args.interface_chunks,
        interface_chunks_strict=args.interface_chunks_strict,
        interface_chunk_norm=args.interface_chunk_norm,
        interface_jacobian_mode=args.interface_jacobian_mode,
        jacobian_basis_chunk=args.jacobian_basis_chunk,
        native_backward=args.native_backward,
        trim_region_cache=args.trim_region_cache,
        lbi_backward=args.lbi_backward,
        compile_turn=args.compile_turn,
        fp32_master=not args.no_fp32_master,
        compile_dense=args.compile_dense,
        log_interface_jacobian_every=args.log_interface_jacobian_every,
        log_interface_jacobian_suffix=args.log_interface_jacobian_suffix,
        eval_message_ablation=args.eval_message_ablation,
        train_message_ablation=args.train_message_ablation,
        eval_all_message_ablations=args.eval_all_message_ablations,
        message_noise_std=args.message_noise_std,
        message_mask_keep_prob=args.message_mask_keep_prob,
    )


# Historical public config name retained by train_region_interface imports.
RegionInterfaceConfig = LBITrainingConfig


# Historical helper names used by train_region_interface imports.
_build_arg_parser = build_arg_parser
_validate_cfg = validate_config
_from_args = config_from_args
