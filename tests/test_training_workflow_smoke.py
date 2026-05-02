from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from data.bpe_tokenizer import load_text_tokenizer
from data.pretokenize_corpus import pretokenize_sentencepiece_corpus, pretokenize_text_corpus
from scripts.plot_message_ablations import generate_message_ablation_plots
from scripts.plot_training_curves import generate_training_plots
from train.train_region_interface import RegionInterfaceConfig, run_training


def _smoke_root() -> Path:
    root = Path("out/smoke")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _tiny_transformer_kwargs() -> dict[str, object]:
    return {
        "backbone": "transformer",
        "dim": 16,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_intermediate": 32,
        "attn_head_dim": 4,
    }


def test_training_workflow_runs_dense_and_lbi_smoke(tmp_path: Path) -> None:
    smoke_root = tmp_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RegionInterfaceConfig(
        regime="all",
        seed=5,
        device=device,
        dtype="float32",
        output_dir=str(smoke_root),
        run_name="smoke_region_interface",
        **_tiny_transformer_kwargs(),
        task="copy",
        vocab_size=64,
        seq_len=16,
        train_sequences=64,
        val_sequences=16,
        batch_size=4,
        steps=2,
        eval_every=1,
        eval_batches=1,
        log_every=1,
        save_every=2,
        layers=2,
        d_state=8,
        region_size=1,
        message_dim=8,
        message_hidden_dim=16,
        interface_jacobian_mode="recompute",
        jacobian_basis_chunk=2,
        eval_all_message_ablations=True,
        lr_model=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
    )
    summary = run_training(cfg)
    assert summary["regime"] == "all"
    assert len(summary["runs"]) == 2
    run_by_regime = {r["regime"]: r for r in summary["runs"]}
    assert "backprop_ref" in run_by_regime
    assert "native_region_interface" in run_by_regime
    run = run_by_regime["native_region_interface"]
    ref_run = run_by_regime["backprop_ref"]
    assert run["steps"] == cfg.steps
    assert math.isfinite(run["final_train_ce_loss"])
    assert math.isfinite(run["final_val_ce_loss"])
    run_dir = Path(run["run_dir"])
    root_dir = Path(summary["root_dir"])
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "data_info.json").exists()
    assert (run_dir / "model_info.json").exists()
    assert "best_val_ce_loss" in run
    assert "best_val_step" in run
    metrics_text = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "tokens_seen" in metrics_text
    assert "val_zero_all" in metrics_text
    assert "val_noise" in metrics_text
    assert "val_mask" in metrics_text
    assert Path(run["best_checkpoint_path"]).exists()
    assert Path(run["latest_checkpoint_path"]).exists()
    assert Path(ref_run["best_checkpoint_path"]).exists()
    assert Path(ref_run["latest_checkpoint_path"]).exists()
    assert not list(Path(run["latest_checkpoint_path"]).parent.glob("step_*.pt"))
    assert not list(Path(ref_run["latest_checkpoint_path"]).parent.glob("step_*.pt"))

    plots_dir = generate_training_plots([root_dir], output_dir=root_dir / "plots_smoke", dpi=80)
    assert (plots_dir / "ce_vs_tokens.png").exists()
    assert (plots_dir / "ce_vs_step.png").exists()
    assert (plots_dir / "tokens_per_s_vs_step.png").exists()
    assert (plots_dir / "val_summary_bar.png").exists()
    assert (plots_dir / "group_summary.csv").exists()

    ablation_dir = generate_message_ablation_plots([root_dir], output_dir=root_dir / "ablation_plots_smoke", dpi=80)
    assert (ablation_dir / "transformer_ablation_ce_vs_tokens.png").exists()
    assert (ablation_dir / "transformer_ablation_final_ce.png").exists()
    assert (ablation_dir / "ablation_summary.csv").exists()


def test_train_region_interface_resume_and_init_from_checkpoint(tmp_path: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_cfg = RegionInterfaceConfig(
        regime="native_region_interface",
        seed=31,
        device=device,
        dtype="float32",
        output_dir=str(tmp_path),
        run_name="resume_region_interface",
        **_tiny_transformer_kwargs(),
        task="copy",
        vocab_size=64,
        seq_len=16,
        train_sequences=64,
        val_sequences=16,
        batch_size=4,
        steps=2,
        eval_every=1,
        eval_batches=1,
        log_every=1,
        save_every=1,
        layers=2,
        d_state=8,
        region_size=1,
        message_dim=8,
        message_hidden_dim=16,
        include_reference_run=False,
        lr_model=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
    )
    first_summary = run_training(base_cfg)
    first_run = first_summary["runs"][0]
    latest_checkpoint = first_run["latest_checkpoint_path"]
    assert Path(latest_checkpoint).exists()

    resume_cfg = RegionInterfaceConfig(
        **{
            **base_cfg.to_dict(),
            "steps": 4,
            "resume_from": latest_checkpoint,
        }
    )
    resumed_summary = run_training(resume_cfg)
    resumed_run = resumed_summary["runs"][0]
    assert resumed_run["start_step"] == 2
    assert resumed_run["resumed_from"] == latest_checkpoint
    assert resumed_run["initialized_from"] == ""
    assert resumed_run["total_train_tokens"] == 4 * base_cfg.batch_size * base_cfg.seq_len
    resumed_metrics = Path(resumed_run["run_dir"]) / "metrics.csv"
    with resumed_metrics.open("r", encoding="utf-8", newline="") as f:
        train_rows = [row for row in csv.DictReader(f) if row["split"] == "train"]
    assert len(train_rows) == 4
    assert train_rows[-1]["step"] == "4"

    init_cfg = RegionInterfaceConfig(
        **{
            **base_cfg.to_dict(),
            "output_dir": str(tmp_path / "init_runs"),
            "run_name": "init_region_interface",
            "steps": 1,
            "init_from": latest_checkpoint,
        }
    )
    init_summary = run_training(init_cfg)
    init_run = init_summary["runs"][0]
    assert init_run["start_step"] == 0
    assert init_run["resumed_from"] == ""
    assert init_run["initialized_from"] == latest_checkpoint
    assert init_run["total_train_tokens"] == base_cfg.batch_size * base_cfg.seq_len
    assert Path(init_run["run_dir"]).parent != Path(resumed_run["run_dir"]).parent


def test_train_region_interface_external_checkpoint_root(tmp_path: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_root = tmp_path / "home_runs"
    checkpoint_root = tmp_path / "u2_checkpoints"
    cfg = RegionInterfaceConfig(
        regime="native_region_interface",
        seed=41,
        device=device,
        dtype="float32",
        output_dir=str(output_root),
        checkpoint_root=str(checkpoint_root),
        run_name="external_checkpoint_run",
        **_tiny_transformer_kwargs(),
        task="copy",
        vocab_size=64,
        seq_len=16,
        train_sequences=64,
        val_sequences=16,
        batch_size=4,
        steps=2,
        eval_every=1,
        eval_batches=1,
        log_every=1,
        save_every=1,
        layers=2,
        d_state=8,
        region_size=1,
        message_dim=8,
        message_hidden_dim=16,
        include_reference_run=False,
    )
    summary = run_training(cfg)
    run = summary["runs"][0]
    run_dir = Path(run["run_dir"])
    best_checkpoint = Path(run["best_checkpoint_path"])
    latest_checkpoint = Path(run["latest_checkpoint_path"])
    assert run_dir.exists()
    assert not (run_dir / "checkpoints").exists()
    assert best_checkpoint.exists()
    assert latest_checkpoint.exists()
    assert checkpoint_root in best_checkpoint.parents
    assert checkpoint_root in latest_checkpoint.parents

    resumed_cfg = RegionInterfaceConfig(
        **{
            **cfg.to_dict(),
            "steps": 3,
            "resume_from": str(run_dir),
        }
    )
    resumed_summary = run_training(resumed_cfg)
    resumed_run = resumed_summary["runs"][0]
    assert resumed_run["start_step"] == 2
    assert resumed_run["resumed_from"] == str(latest_checkpoint)
    assert resumed_run["total_train_tokens"] == 3 * cfg.batch_size * cfg.seq_len


def test_train_region_interface_backprop_ref_only_smoke(tmp_path: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RegionInterfaceConfig(
        regime="backprop_ref",
        seed=9,
        device=device,
        dtype="float32",
        output_dir=str(tmp_path),
        run_name="smoke_backprop_ref_only",
        **_tiny_transformer_kwargs(),
        task="copy",
        vocab_size=64,
        seq_len=16,
        train_sequences=64,
        val_sequences=16,
        batch_size=4,
        steps=2,
        eval_every=1,
        eval_batches=1,
        log_every=1,
        save_every=1,
        layers=2,
        d_state=8,
        region_size=1,
        message_dim=8,
        message_hidden_dim=16,
    )
    summary = run_training(cfg)
    assert summary["regime"] == "backprop_ref"
    assert len(summary["runs"]) == 1
    run = summary["runs"][0]
    assert run["regime"] == "backprop_ref"
    assert math.isfinite(run["final_train_ce_loss"])
    assert math.isfinite(run["final_val_ce_loss"])
    assert Path(run["best_checkpoint_path"]).exists()
    assert Path(run["latest_checkpoint_path"]).exists()


def test_train_region_interface_can_disable_checkpoint_saving(tmp_path: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RegionInterfaceConfig(
        regime="backprop_ref",
        seed=10,
        device=device,
        dtype="float32",
        output_dir=str(tmp_path),
        run_name="smoke_no_checkpoints",
        **_tiny_transformer_kwargs(),
        task="copy",
        vocab_size=64,
        seq_len=16,
        train_sequences=64,
        val_sequences=16,
        batch_size=4,
        steps=2,
        eval_every=1,
        eval_batches=1,
        log_every=1,
        save_every=0,
        save_checkpoints=False,
        layers=2,
        d_state=8,
        region_size=1,
        message_dim=8,
        message_hidden_dim=16,
    )
    summary = run_training(cfg)
    run = summary["runs"][0]
    run_dir = Path(run["run_dir"])
    assert run["save_checkpoints"] is False
    assert math.isfinite(run["final_train_ce_loss"])
    assert math.isfinite(run["final_val_ce_loss"])
    assert math.isfinite(run["best_val_ce_loss"])
    assert run["best_val_step"] > 0
    assert run["best_checkpoint_path"] == ""
    assert run["latest_checkpoint_path"] == ""
    assert not (run_dir / "checkpoints").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "summary.json").exists()


def test_fineweb_edu_requires_sharded_mode() -> None:
    with pytest.raises(ValueError, match="fineweb_edu currently requires data_mode=text_bpe_sharded"):
        RegionInterfaceConfig(
            data_mode="text_bpe",
            text_corpus="fineweb_edu",
            vocab_size=8192,
        )


def test_llama31_rejects_train_tokenizer_cfg() -> None:
    with pytest.raises(ValueError, match="llama31 tokenizer does not support --train-tokenizer"):
        RegionInterfaceConfig(
            data_mode="text_bpe_sharded",
            text_corpus="fineweb_edu",
            tokenizer_type="llama31",
            train_tokenizer=True,
            vocab_size=8192,
        )


def test_llama_rejects_train_tokenizer_cfg() -> None:
    with pytest.raises(ValueError, match="llama tokenizer does not support --train-tokenizer"):
        RegionInterfaceConfig(
            data_mode="text_bpe_sharded",
            text_corpus="fineweb_edu",
            tokenizer_type="llama",
            train_tokenizer=True,
            vocab_size=32000,
        )


def test_load_text_tokenizer_llama31_disallows_training(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="llama31 tokenizer does not support --train-tokenizer"):
        load_text_tokenizer(
            tokenizer_type="llama31",
            tokenizer_path=tmp_path / "llama31",
            train_path=tmp_path / "train.txt",
            vocab_size=32000,
            train_bytes=4096,
            force_train=True,
        )


def test_load_text_tokenizer_llama_disallows_training(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="llama tokenizer does not support --train-tokenizer"):
        load_text_tokenizer(
            tokenizer_type="llama",
            tokenizer_path=tmp_path / "llama",
            train_path=tmp_path / "train.txt",
            vocab_size=32000,
            train_bytes=4096,
            force_train=True,
        )


def test_train_region_interface_text_bpe_sharded_route_smoke(tmp_path: Path) -> None:
    if importlib.util.find_spec("sentencepiece") is None:
        pytest.skip("sentencepiece not installed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smoke_root = tmp_path
    train_path = smoke_root / "smoke_text_bpe_sharded_train.txt"
    val_path = smoke_root / "smoke_text_bpe_sharded_val.txt"
    tokenizer_path = smoke_root / "smoke_text_bpe_sharded.model"
    token_shards_dir = smoke_root / "smoke_text_bpe_sharded_tokens"
    train_path.write_text("hello world\n" * 128 + "bounded interfaces learn\n" * 128, encoding="utf-8")
    val_path.write_text("hello bounded world\n" * 32, encoding="utf-8")
    pretokenize_text_corpus(
        text_corpus="tiny_shakespeare",
        train_text_path=str(train_path),
        val_text_path=str(val_path),
        tokenizer_type="sentencepiece",
        tokenizer_path=str(tokenizer_path),
        vocab_size=320,
        tokenizer_train_bytes=4096,
        train_tokenizer=True,
        shard_tokens=128,
        output_dir=str(token_shards_dir),
    )
    cfg = RegionInterfaceConfig(
        regime="native_region_interface",
        seed=22,
        device=device,
        dtype="float32",
        output_dir=str(smoke_root),
        run_name="smoke_region_interface_text_bpe_sharded",
        **_tiny_transformer_kwargs(),
        data_mode="text_bpe_sharded",
        train_text_path=str(train_path),
        val_text_path=str(val_path),
        tokenizer_path=str(tokenizer_path),
        token_shards_dir=str(token_shards_dir),
        vocab_size=320,
        seq_len=16,
        batch_size=2,
        steps=2,
        eval_every=1,
        eval_batches=1,
        log_every=1,
        save_every=2,
        layers=2,
        d_state=8,
        region_size=1,
        message_dim=8,
        message_hidden_dim=16,
        include_reference_run=False,
    )
    summary = run_training(cfg)
    run = summary["runs"][0]
    assert run["regime"] == "native_region_interface"
    assert math.isfinite(run["final_train_ce_loss"])
    assert math.isfinite(run["final_val_ce_loss"])
    assert (token_shards_dir / "train_manifest.json").exists()
    assert (token_shards_dir / "val_manifest.json").exists()


def test_train_region_interface_text_bpe_route_smoke(tmp_path: Path) -> None:
    if importlib.util.find_spec("sentencepiece") is None:
        pytest.skip("sentencepiece not installed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smoke_root = tmp_path
    train_path = smoke_root / "smoke_text_bpe_train.txt"
    val_path = smoke_root / "smoke_text_bpe_val.txt"
    tokenizer_path = smoke_root / "smoke_text_bpe.model"
    train_path.write_text("hello world\n" * 64 + "bounded interfaces learn\n" * 64, encoding="utf-8")
    val_path.write_text("hello bounded world\n" * 16, encoding="utf-8")
    cfg = RegionInterfaceConfig(
        regime="native_region_interface",
        seed=21,
        device=device,
        dtype="float32",
        output_dir=str(smoke_root),
        run_name="smoke_region_interface_text_bpe",
        **_tiny_transformer_kwargs(),
        data_mode="text_bpe",
        train_text_path=str(train_path),
        val_text_path=str(val_path),
        tokenizer_path=str(tokenizer_path),
        train_tokenizer=True,
        tokenizer_train_bytes=4096,
        vocab_size=320,
        seq_len=16,
        batch_size=2,
        steps=2,
        eval_every=1,
        eval_batches=1,
        log_every=1,
        save_every=2,
        layers=2,
        d_state=8,
        region_size=1,
        message_dim=8,
        message_hidden_dim=16,
        include_reference_run=False,
    )
    summary = run_training(cfg)
    run = summary["runs"][0]
    assert run["regime"] == "native_region_interface"
    assert math.isfinite(run["final_train_ce_loss"])
    assert math.isfinite(run["final_val_ce_loss"])
    assert tokenizer_path.exists()


def test_train_region_interface_mamba2_smoke(tmp_path: Path) -> None:
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton not installed")
    if importlib.util.find_spec("einops") is None:
        pytest.skip("einops not installed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smoke_root = tmp_path
    cfg = RegionInterfaceConfig(
        regime="all",
        backbone="mamba2",
        seed=11,
        device=device,
        dtype="float32",
        output_dir=str(smoke_root),
        run_name="smoke_region_interface_mamba2",
        task="copy",
        vocab_size=64,
        seq_len=16,
        train_sequences=64,
        val_sequences=16,
        batch_size=4,
        steps=2,
        eval_every=1,
        eval_batches=1,
        log_every=1,
        save_every=2,
        layers=2,
        dim=16,
        d_state=4,
        d_conv=4,
        expand=2,
        headdim=8,
        ngroups=1,
        chunk_size=16,
        use_mem_eff_path=(device == "cuda"),
        region_size=1,
        message_dim=8,
        message_hidden_dim=16,
        lr_model=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
    )
    summary = run_training(cfg)
    assert summary["regime"] == "all"
    assert len(summary["runs"]) == 2
    run_by_regime = {r["regime"]: r for r in summary["runs"]}
    assert "backprop_ref" in run_by_regime
    assert "native_region_interface" in run_by_regime
    run = run_by_regime["native_region_interface"]
    assert math.isfinite(run["final_train_ce_loss"])
    assert math.isfinite(run["final_val_ce_loss"])


def test_train_region_interface_transformer_smoke(tmp_path: Path) -> None:
    smoke_root = tmp_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RegionInterfaceConfig(
        regime="all",
        backbone="transformer",
        seed=13,
        device=device,
        dtype="float32",
        output_dir=str(smoke_root),
        run_name="smoke_region_interface_transformer",
        task="copy",
        vocab_size=64,
        seq_len=16,
        train_sequences=64,
        val_sequences=16,
        batch_size=4,
        steps=2,
        eval_every=1,
        eval_batches=1,
        log_every=1,
        save_every=2,
        layers=2,
        dim=32,
        d_state=8,
        n_heads=4,
        n_kv_heads=2,
        mlp_ratio=0.0,
        d_intermediate=0,
        attn_head_dim=10,
        softmax_scale=0.5,
        rope_interleaved=True,
        d_conv=2,
        region_size=1,
        message_dim=8,
        message_hidden_dim=16,
        lr_model=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
    )
    summary = run_training(cfg)
    assert summary["regime"] == "all"
    assert len(summary["runs"]) == 2
    run_by_regime = {r["regime"]: r for r in summary["runs"]}
    assert "backprop_ref" in run_by_regime
    assert "native_region_interface" in run_by_regime
    run = run_by_regime["native_region_interface"]
    assert math.isfinite(run["final_train_ce_loss"])
    assert math.isfinite(run["final_val_ce_loss"])


def test_train_region_interface_hybrid_smoke(tmp_path: Path) -> None:
    smoke_root = tmp_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RegionInterfaceConfig(
        regime="all",
        backbone="hybrid",
        layer_types="mamba2,transformer",
        seed=19,
        device=device,
        dtype="float32",
        output_dir=str(smoke_root),
        run_name="smoke_region_interface_hybrid",
        task="copy",
        vocab_size=64,
        seq_len=16,
        train_sequences=64,
        val_sequences=16,
        batch_size=4,
        steps=2,
        eval_every=1,
        eval_batches=1,
        log_every=1,
        save_every=2,
        layers=2,
        dim=32,
        d_state=8,
        d_conv=4,
        expand=2,
        headdim=8,
        ngroups=1,
        chunk_size=16,
        use_mem_eff_path=(device == "cuda"),
        n_heads=4,
        n_kv_heads=2,
        d_intermediate=128,
        attn_head_dim=8,
        region_size=1,
        message_dim=8,
        message_hidden_dim=16,
        lr_model=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
    )
    summary = run_training(cfg)
    assert summary["regime"] == "all"
    assert len(summary["runs"]) == 2
    run_by_regime = {r["regime"]: r for r in summary["runs"]}
    assert "backprop_ref" in run_by_regime
    assert "native_region_interface" in run_by_regime
    run = run_by_regime["native_region_interface"]
    assert math.isfinite(run["final_train_ce_loss"])
    assert math.isfinite(run["final_val_ce_loss"])


def test_train_region_interface_mamba3_smoke(tmp_path: Path) -> None:
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton not installed")
    if importlib.util.find_spec("einops") is None:
        pytest.skip("einops not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for mamba3 smoke")
    smoke_root = tmp_path
    cfg = RegionInterfaceConfig(
        regime="all",
        backbone="mamba3",
        seed=17,
        device="cuda",
        dtype="bfloat16",
        output_dir=str(smoke_root),
        run_name="smoke_region_interface_mamba3",
        task="copy",
        vocab_size=64,
        seq_len=16,
        train_sequences=64,
        val_sequences=16,
        batch_size=2,
        steps=1,
        eval_every=1,
        eval_batches=1,
        log_every=1,
        save_every=1,
        layers=2,
        dim=16,
        d_state=16,
        expand=2,
        headdim=16,
        ngroups=1,
        chunk_size=16,
        region_size=1,
        message_dim=8,
        message_hidden_dim=16,
        lr_model=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
    )
    summary = run_training(cfg)
    assert summary["regime"] == "all"
    assert len(summary["runs"]) == 2
    run_by_regime = {r["regime"]: r for r in summary["runs"]}
    assert "backprop_ref" in run_by_regime
    assert "native_region_interface" in run_by_regime
    for run in run_by_regime.values():
        assert math.isfinite(run["final_train_ce_loss"])
        assert math.isfinite(run["final_val_ce_loss"])
