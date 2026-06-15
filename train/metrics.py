from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch.nn as nn

from data.token_shards import corpus_numel
from models.dense_language_model import DenseLanguageModel
from models.lbi_language_model import LBILanguageModel


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def canvas_component_params(model: nn.Module) -> int:
    canvas = getattr(model, "canvas", None)
    return count_parameters(canvas) if isinstance(canvas, nn.Module) else 0


def readout_lm_head_component_params(model: nn.Module) -> int:
    readout = getattr(model, "readout", None)
    if readout is None:
        return 0
    if hasattr(readout, "output_head_parameter_count"):
        return int(readout.output_head_parameter_count())
    lm_head = getattr(readout, "lm_head", None)
    return count_parameters(lm_head) if isinstance(lm_head, nn.Module) else 0


def readout_norm_component_params(model: nn.Module) -> int:
    readout = getattr(model, "readout", None)
    norm = getattr(readout, "norm", None)
    return count_parameters(norm) if isinstance(norm, nn.Module) else 0


def readout_uses_tied_output_weight(model: nn.Module) -> bool:
    readout = getattr(model, "readout", None)
    return bool(getattr(readout, "tie_embeddings", False) and getattr(readout, "lm_head", None) is None)


def infer_data_info(
    cfg: Any,
    *,
    train_corpus: Any,
    val_corpus: Any,
    corpora_root: Path,
    resolve_text_paths_fn: Callable[[Any], tuple[Path, Path | None, Path]],
    resolve_tokenizer_path_fn: Callable[..., Path],
    resolve_token_shard_manifests_fn: Callable[..., tuple[Path, Path]],
    resolve_token_shards_dir_fn: Callable[..., Path],
) -> dict[str, Any]:
    info: dict[str, Any] = {
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

    train_path, val_path, _ = resolve_text_paths_fn(cfg)
    info["corpora_root"] = str(corpora_root)
    info["train_path"] = str(train_path)
    info["val_path"] = str(val_path) if val_path is not None else ""
    info["train_file_bytes"] = int(train_path.stat().st_size) if train_path.exists() else 0
    info["val_file_bytes"] = int(val_path.stat().st_size) if (val_path is not None and val_path.exists()) else 0
    if cfg.data_mode in {"text_bpe", "text_bpe_sharded"}:
        tokenizer_path = resolve_tokenizer_path_fn(cfg, train_path=train_path)
        vocab_path = tokenizer_path.with_suffix(".vocab") if tokenizer_path.suffix == ".model" else Path("")
        info["tokenizer_path"] = str(tokenizer_path)
        info["tokenizer_type"] = cfg.tokenizer_type
        info["tokenizer_exists"] = bool(tokenizer_path.exists())
        info["tokenizer_vocab_path"] = str(vocab_path)
        info["tokenizer_vocab_exists"] = bool(vocab_path.exists()) if str(vocab_path) else False
    if cfg.data_mode == "text_bpe":
        info["tokenizer_train_bytes"] = int(cfg.tokenizer_train_bytes)
    if cfg.data_mode == "text_bpe_sharded":
        train_manifest, val_manifest = resolve_token_shard_manifests_fn(cfg, train_path=train_path)
        info["token_shards_dir"] = str(resolve_token_shards_dir_fn(cfg, train_path=train_path))
        info["train_manifest"] = str(train_manifest)
        info["train_manifest_exists"] = bool(train_manifest.exists())
        info["val_manifest"] = str(val_manifest)
        info["val_manifest_exists"] = bool(val_manifest.exists())
    return info


def infer_model_info(cfg: Any, *, model: nn.Module) -> dict[str, Any]:
    info: dict[str, Any] = {
        "model_class": type(model).__name__,
        "total_params": count_parameters(model),
        "layers": int(cfg.layers),
        "dim": int(cfg.dim),
        "d_state": int(cfg.d_state),
        "dt_rank": cfg.dt_rank if cfg.dt_rank is not None else "auto",
        "tie_embeddings": bool(cfg.tie_embeddings),
    }
    if isinstance(model, LBILanguageModel):
        canvas_params = canvas_component_params(model)
        lm_head_params = readout_lm_head_component_params(model)
        norm_params = readout_norm_component_params(model)
        region_backend = getattr(model, "region_backend", None)
        if region_backend is None:
            raise TypeError("LBILanguageModel is missing region_backend")
        backbone_params = int(region_backend.count_parameters())
        interface = model.interface
        interface_params = count_parameters(interface)
        info["n_regions"] = int(model.num_regions)
        info["region_ranges"] = [list(pair) for pair in model.region_ranges]
        info["backbone_params"] = backbone_params
        info["tokenizer_params"] = canvas_params + lm_head_params
        info["interface_params"] = interface_params
        info["component_params"] = {
            "canvas": canvas_params,
            "canvas.embedding": canvas_params,
            "region_backend": backbone_params,
            "interface.initial_encoder": count_parameters(interface.initial_encoder),
            "interface.decoders": int(sum(count_parameters(mod) for mod in interface.decoders)),
            "interface.encoders": int(sum(count_parameters(mod) for mod in interface.encoders)),
            "interface.norms": int(sum(count_parameters(mod) for mod in interface.norms)),
            "interface.update_scale": int(interface.update_scale.numel()),
            "readout.norm": norm_params,
            "readout.lm_head": lm_head_params,
        }
        info["lm_head_tied_to_embedding"] = readout_uses_tied_output_weight(model)
        info["owned_lbi_model"] = True
    elif isinstance(model, DenseLanguageModel):
        canvas_params = canvas_component_params(model)
        lm_head_params = readout_lm_head_component_params(model)
        norm_params = readout_norm_component_params(model)
        backbone_params = int(sum(count_parameters(block) for block in model.blocks))
        info["backbone_params"] = backbone_params
        info["tokenizer_params"] = canvas_params + lm_head_params
        info["interface_params"] = 0
        info["component_params"] = {
            "canvas": canvas_params,
            "canvas.embedding": canvas_params,
            "blocks": backbone_params,
            "readout.norm": norm_params,
            "readout.lm_head": lm_head_params,
        }
        info["lm_head_tied_to_embedding"] = readout_uses_tied_output_weight(model)
    return info


def write_run_metadata(
    *,
    cfg: Any,
    run_dir: Path,
    model: nn.Module,
    train_corpus: Any,
    val_corpus: Any,
    corpora_root: Path,
    resolve_text_paths_fn: Callable[[Any], tuple[Path, Path | None, Path]],
    resolve_tokenizer_path_fn: Callable[..., Path],
    resolve_token_shard_manifests_fn: Callable[..., tuple[Path, Path]],
    resolve_token_shards_dir_fn: Callable[..., Path],
) -> None:
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "data_info.json").write_text(
        json.dumps(
            infer_data_info(
                cfg,
                train_corpus=train_corpus,
                val_corpus=val_corpus,
                corpora_root=corpora_root,
                resolve_text_paths_fn=resolve_text_paths_fn,
                resolve_tokenizer_path_fn=resolve_tokenizer_path_fn,
                resolve_token_shard_manifests_fn=resolve_token_shard_manifests_fn,
                resolve_token_shards_dir_fn=resolve_token_shards_dir_fn,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_dir / "model_info.json").write_text(
        json.dumps(infer_model_info(cfg, model=model), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_csv_row(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "step",
        "tokens_seen",
        "split",
        "ce_loss",
        "message_norm",
        "scan_align",
        "jac_local_spec_mean",
        "jac_local_spec_max",
        "jac_local_frob_mean",
        "jac_local_frob_max",
        "jac_local_frob_normed_mean",
        "jac_suffix_spec_mean",
        "jac_suffix_spec_max",
        "tokens_per_s",
        "wall_time_s",
    ]
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
