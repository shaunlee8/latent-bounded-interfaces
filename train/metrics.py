from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch.nn as nn

from data.token_shards import corpus_numel
from models.dense_language_model import DenseLanguageModel
from models.lbi_language_model import LBILanguageModel
from train.data import (
    resolve_text_paths,
    resolve_token_shard_manifests,
    resolve_token_shards_dir,
    resolve_tokenizer_path,
)


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def readout_lm_head_component_params(model: nn.Module) -> int:
    return int(model.readout.output_head_parameter_count())


def readout_uses_tied_output_weight(model: nn.Module) -> bool:
    readout = model.readout
    return bool(readout.tie_embeddings and readout.lm_head is None)


def infer_data_info(cfg: Any, *, train_corpus: Any, val_corpus: Any) -> dict[str, Any]:
    train_path, val_path = resolve_text_paths(cfg)
    train_manifest, val_manifest = resolve_token_shard_manifests(cfg, train_path=train_path)
    tokenizer_path = resolve_tokenizer_path(cfg)
    return {
        "text_corpus": cfg.text_corpus,
        "vocab_size": int(cfg.vocab_size),
        "seq_len": int(cfg.seq_len),
        "train_stream_length": corpus_numel(train_corpus),
        "val_stream_length": corpus_numel(val_corpus),
        "train_path": str(train_path),
        "val_path": str(val_path) if val_path is not None else "",
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_exists": bool(tokenizer_path.exists()),
        "token_shards_dir": str(resolve_token_shards_dir(cfg, train_path=train_path)),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
    }


def infer_model_info(cfg: Any, *, model: nn.Module) -> dict[str, Any]:
    info: dict[str, Any] = {
        "model_class": type(model).__name__,
        "total_params": count_parameters(model),
        "layers": int(cfg.layers),
        "dim": int(cfg.dim),
        "d_state": int(cfg.d_state),
        "tie_embeddings": bool(cfg.tie_embeddings),
    }
    canvas_params = count_parameters(model.canvas)
    lm_head_params = readout_lm_head_component_params(model)
    norm_params = count_parameters(model.readout.norm)
    info["tokenizer_params"] = canvas_params + lm_head_params
    info["lm_head_tied_to_embedding"] = readout_uses_tied_output_weight(model)
    if isinstance(model, LBILanguageModel):
        backbone_params = int(model.region_backend.count_parameters())
        interface = model.interface
        info["n_regions"] = int(model.num_regions)
        info["region_ranges"] = [list(pair) for pair in model.region_ranges]
        info["backbone_params"] = backbone_params
        info["interface_params"] = count_parameters(interface)
        info["component_params"] = {
            "canvas": canvas_params,
            "region_backend": backbone_params,
            "interface.initial_encoder": count_parameters(interface.initial_encoder),
            "interface.decoders": int(sum(count_parameters(mod) for mod in interface.decoders)),
            "interface.encoders": int(sum(count_parameters(mod) for mod in interface.encoders)),
            "interface.norms": int(sum(count_parameters(mod) for mod in interface.norms)),
            "interface.update_scale": int(interface.update_scale.numel()),
            "readout.norm": norm_params,
            "readout.lm_head": lm_head_params,
        }
    elif isinstance(model, DenseLanguageModel):
        backbone_params = int(sum(count_parameters(block) for block in model.blocks))
        info["backbone_params"] = backbone_params
        info["interface_params"] = 0
        info["component_params"] = {
            "canvas": canvas_params,
            "blocks": backbone_params,
            "readout.norm": norm_params,
            "readout.lm_head": lm_head_params,
        }
    return info


def write_run_metadata(*, cfg: Any, run_dir: Path, model: nn.Module, train_corpus: Any, val_corpus: Any) -> None:
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "data_info.json").write_text(
        json.dumps(infer_data_info(cfg, train_corpus=train_corpus, val_corpus=val_corpus), indent=2, sort_keys=True),
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
        "tokens_per_s",
        "wall_time_s",
    ]
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
