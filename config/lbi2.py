from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ModelVariant(str, Enum):
    DENSE = "dense"
    LBI = "lbi"


class BackwardEngine(str, Enum):
    AUTOGRAD = "autograd"
    REFERENCE_SCAN = "reference_scan"
    CUDA_PARALLEL = "cuda_parallel"
    DISTRIBUTED_PARALLEL = "distributed_parallel"


class BackendType(str, Enum):
    TRANSFORMER = "transformer"
    MAMBA3_SISO = "mamba3_siso"
    # Retained only while existing LBI-1 entrypoints pass through this bridge.
    MAMBA2 = "mamba2"
    HYBRID = "hybrid"


class InterfaceType(str, Enum):
    VECTOR_MLP = "vector_mlp"


class CanvasType(str, Enum):
    EMBEDDING = "embedding"


class LocalPullbackMode(str, Enum):
    GRAPH = "graph"
    RECOMPUTE = "recompute"


@dataclass(frozen=True)
class ModelConfig:
    backend: BackendType
    layers: int
    hidden_dim: int
    vocab_size: int
    tie_embeddings: bool
    layer_types: tuple[str, ...]
    d_state: int
    dt_rank: int | None
    expand: int
    d_conv: int
    headdim: int
    ngroups: int
    chunk_size: int
    n_heads: int
    n_kv_heads: int
    mlp_ratio: float
    d_intermediate: int
    attn_head_dim: int
    interface_type: InterfaceType
    canvas_type: CanvasType
    layers_per_region: int
    interface_width: int
    interface_map_hidden_dim: int
    interface_update_scale: float


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    dtype: str
    seq_len: int
    batch_size: int
    steps: int
    eval_every: int
    eval_batches: int
    log_every: int
    save_every: int
    learning_rate: float
    learning_rate_schedule: str
    warmup_steps: int
    min_learning_rate_ratio: float
    weight_decay: float
    grad_clip: float


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    output_dir: str
    checkpoint_root: str
    run_name: str
    resume_from: str
    init_from: str
    save_checkpoints: bool
    model_variants: tuple[ModelVariant, ...]
    dense_backward_engine: BackwardEngine
    lbi_backward_engine: BackwardEngine
    local_pullback_mode: LocalPullbackMode
    pullback_basis_chunk: int
    log_interface_jacobian_every: int
    log_interface_jacobian_suffix: bool


@dataclass(frozen=True)
class LBI2Config:
    model: ModelConfig
    training: TrainingConfig
    runtime: RuntimeConfig


_LEGACY_BACKENDS = {
    "transformer": BackendType.TRANSFORMER,
    "mamba3": BackendType.MAMBA3_SISO,
    "mamba2": BackendType.MAMBA2,
    "hybrid": BackendType.HYBRID,
}

_LEGACY_REGIMES = {
    "backprop_ref": (ModelVariant.DENSE,),
    "native_region_interface": (ModelVariant.LBI,),
    "all": (ModelVariant.DENSE, ModelVariant.LBI),
}


def from_legacy_region_interface_config(cfg: Any) -> LBI2Config:
    """Translate an LBI-1 trainer config into the provisional LBI-2 vocabulary.

    This adapter is intentionally one-way and temporary. It exists only while
    the current trainer remains the executable implementation.
    """

    try:
        backend = _LEGACY_BACKENDS[cfg.backbone]
    except KeyError as exc:
        raise ValueError(f"unsupported legacy backbone: {cfg.backbone}") from exc
    try:
        model_variants = _LEGACY_REGIMES[cfg.regime]
    except KeyError as exc:
        raise ValueError(f"unsupported legacy regime: {cfg.regime}") from exc

    layer_types = tuple(layer.strip() for layer in cfg.layer_types.split(",") if layer.strip())
    return LBI2Config(
        model=ModelConfig(
            backend=backend,
            layers=cfg.layers,
            hidden_dim=cfg.dim,
            vocab_size=cfg.vocab_size,
            tie_embeddings=cfg.tie_embeddings,
            layer_types=layer_types,
            d_state=cfg.d_state,
            dt_rank=cfg.dt_rank,
            expand=cfg.expand,
            d_conv=cfg.d_conv,
            headdim=cfg.headdim,
            ngroups=cfg.ngroups,
            chunk_size=cfg.chunk_size,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            mlp_ratio=cfg.mlp_ratio,
            d_intermediate=cfg.d_intermediate,
            attn_head_dim=cfg.attn_head_dim,
            interface_type=InterfaceType.VECTOR_MLP,
            canvas_type=CanvasType.EMBEDDING,
            layers_per_region=cfg.region_size,
            interface_width=cfg.message_dim,
            interface_map_hidden_dim=cfg.message_hidden_dim,
            interface_update_scale=cfg.message_scale_init,
        ),
        training=TrainingConfig(
            seed=cfg.seed,
            dtype=cfg.dtype,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            steps=cfg.steps,
            eval_every=cfg.eval_every,
            eval_batches=cfg.eval_batches,
            log_every=cfg.log_every,
            save_every=cfg.save_every,
            learning_rate=cfg.lr_model,
            learning_rate_schedule=cfg.lr_schedule,
            warmup_steps=cfg.warmup_steps,
            min_learning_rate_ratio=cfg.min_lr_ratio,
            weight_decay=cfg.weight_decay,
            grad_clip=cfg.grad_clip,
        ),
        runtime=RuntimeConfig(
            device=cfg.device,
            output_dir=cfg.output_dir,
            checkpoint_root=cfg.checkpoint_root,
            run_name=cfg.run_name,
            resume_from=cfg.resume_from,
            init_from=cfg.init_from,
            save_checkpoints=cfg.save_checkpoints,
            model_variants=model_variants,
            dense_backward_engine=BackwardEngine.AUTOGRAD,
            lbi_backward_engine=BackwardEngine.REFERENCE_SCAN,
            local_pullback_mode=LocalPullbackMode(cfg.interface_jacobian_mode),
            pullback_basis_chunk=cfg.jacobian_basis_chunk,
            log_interface_jacobian_every=cfg.log_interface_jacobian_every,
            log_interface_jacobian_suffix=cfg.log_interface_jacobian_suffix,
        ),
    )
