from config.lbi2 import (
    BackendType,
    BackwardEngine,
    CanvasType,
    InterfaceType,
    LocalPullbackMode,
    ModelVariant,
    from_legacy_region_interface_config,
)
from train.lbi import LBITrainingConfig


def test_transformer_canonical_lbi_config_maps_to_lbi2_vocabulary() -> None:
    legacy = LBITrainingConfig(
        regime="native_region_interface",
        backbone="transformer",
        layers=12,
        dim=512,
        vocab_size=32000,
        tie_embeddings=True,
        n_heads=8,
        n_kv_heads=4,
        attn_head_dim=64,
        d_intermediate=2048,
        region_size=2,
        message_dim=32,
        message_hidden_dim=0,
        steps=20000,
        lr_model=6e-4,
        lr_schedule="cosine",
        warmup_steps=1000,
        min_lr_ratio=0.1,
        interface_jacobian_mode="recompute",
        jacobian_basis_chunk=32,
    )

    cfg = from_legacy_region_interface_config(legacy)

    assert cfg.model.backend is BackendType.TRANSFORMER
    assert cfg.model.interface_type is InterfaceType.VECTOR_MLP
    assert cfg.model.canvas_type is CanvasType.EMBEDDING
    assert cfg.model.layers_per_region == 2
    assert cfg.model.interface_width == 32
    assert cfg.model.interface_map_hidden_dim == 0
    assert cfg.training.learning_rate == 6e-4
    assert cfg.training.warmup_steps == 1000
    assert cfg.runtime.model_variants == (ModelVariant.LBI,)
    assert cfg.runtime.lbi_backward_engine is BackwardEngine.REFERENCE_SCAN
    assert cfg.runtime.local_pullback_mode is LocalPullbackMode.RECOMPUTE
    assert cfg.runtime.pullback_basis_chunk == 32


def test_mamba3_siso_canonical_lbi_lower_lr_maps_explicitly() -> None:
    legacy = LBITrainingConfig(
        regime="native_region_interface",
        backbone="mamba3",
        device="cuda",
        dtype="bfloat16",
        layers=14,
        dim=768,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        chunk_size=64,
        vocab_size=32000,
        tie_embeddings=True,
        region_size=2,
        message_dim=16,
        steps=20000,
        lr_model=3e-4,
        lr_schedule="cosine",
        warmup_steps=500,
        min_lr_ratio=0.1,
    )

    cfg = from_legacy_region_interface_config(legacy)

    assert cfg.model.backend is BackendType.MAMBA3_SISO
    assert cfg.model.layers == 14
    assert cfg.model.hidden_dim == 768
    assert cfg.model.interface_width == 16
    assert cfg.training.learning_rate == 3e-4
    assert cfg.training.warmup_steps == 500


def test_dense_and_combined_legacy_regimes_map_to_variant_selection() -> None:
    dense = from_legacy_region_interface_config(LBITrainingConfig(regime="backprop_ref"))
    combined = from_legacy_region_interface_config(LBITrainingConfig(regime="all"))

    assert dense.runtime.model_variants == (ModelVariant.DENSE,)
    assert dense.runtime.dense_backward_engine is BackwardEngine.AUTOGRAD
    assert combined.runtime.model_variants == (ModelVariant.DENSE, ModelVariant.LBI)


def test_normal_lbi_builder_is_owned_only_even_with_legacy_env(monkeypatch) -> None:
    from models.lbi_language_model import LBILanguageModel
    from legacy.native_region_interface import NativeRegionInterfaceModel
    from train.model_builders import build_lbi_model, build_model_for_regime

    cfg = LBITrainingConfig(
        regime="native_region_interface",
        backbone="transformer",
        layers=2,
        dim=16,
        n_heads=4,
        n_kv_heads=2,
        attn_head_dim=4,
        d_intermediate=32,
        vocab_size=64,
        region_size=1,
        message_dim=8,
        message_hidden_dim=16,
    )

    monkeypatch.setenv("LBI1_LEGACY_MODEL", "1")

    assert isinstance(build_lbi_model(cfg), LBILanguageModel)
    assert isinstance(build_model_for_regime(cfg), LBILanguageModel)

    legacy_checkpoint = {"model_state_dict": {"input_to_message.weight": None}}
    assert isinstance(build_model_for_regime(cfg, checkpoint=legacy_checkpoint), NativeRegionInterfaceModel)

def test_lbi2_regime_names_are_primary_aliases() -> None:
    from train.config import LEGACY_DENSE_REGIME, LEGACY_LBI_REGIME, normalize_regime_name

    assert normalize_regime_name("dense") == LEGACY_DENSE_REGIME
    assert normalize_regime_name("lbi") == LEGACY_LBI_REGIME
    assert normalize_regime_name("backprop_ref") == LEGACY_DENSE_REGIME
    assert normalize_regime_name("native_region_interface") == LEGACY_LBI_REGIME
    assert LBITrainingConfig(regime="dense").regime == "dense"
    assert LBITrainingConfig(regime="lbi").regime == "lbi"

