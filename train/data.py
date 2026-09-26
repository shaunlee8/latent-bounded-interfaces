from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from data.tokenizer import load_llama_tokenizer
from data.data_paths import CORPORA_ROOT, LLAMA_TOKENIZER_ROOT
from data.token_shards import TokenShardCorpus, sample_batch_token_shards

FINEWEB_EDU_TEXT = ("fineweb_edu/fineweb_edu_train.txt", "fineweb_edu/fineweb_edu_val.txt")


def resolve_text_paths(cfg: Any) -> tuple[Path, Path | None]:
    """The train and validation text files: explicit paths, else the FineWeb-Edu export."""
    if cfg.train_text_path:
        train_path = Path(cfg.train_text_path)
        val_path = Path(cfg.val_text_path) if cfg.val_text_path else None
        return train_path, val_path
    train_name, val_name = FINEWEB_EDU_TEXT
    return CORPORA_ROOT / train_name, CORPORA_ROOT / val_name


def resolve_tokenizer_path(cfg: Any) -> Path:
    return Path(cfg.tokenizer_path) if cfg.tokenizer_path else LLAMA_TOKENIZER_ROOT


def resolve_token_shards_dir(cfg: Any, *, train_path: Path) -> Path:
    if cfg.token_shards_dir:
        return Path(cfg.token_shards_dir)
    if not cfg.train_text_path:
        return CORPORA_ROOT / cfg.text_corpus / "tokens" / f"{cfg.text_corpus}_llama"
    return train_path.parent / "tokens" / f"{train_path.stem}_llama"


def resolve_token_shard_manifests(cfg: Any, *, train_path: Path) -> tuple[Path, Path]:
    shard_dir = resolve_token_shards_dir(cfg, train_path=train_path)
    return shard_dir / "train_manifest.json", shard_dir / "val_manifest.json"


def resolve_runtime_vocab_size(cfg: Any) -> int:
    """The LLaMA tokenizer's vocabulary size, which overrides `cfg.vocab_size`."""
    tokenizer_path = resolve_tokenizer_path(cfg)
    tokenizer = load_llama_tokenizer(tokenizer_path)
    actual_vocab_size = int(tokenizer.vocab_size)
    if cfg.vocab_size != actual_vocab_size:
        print(
            f"[tokenizer] overriding vocab_size from {cfg.vocab_size} to the tokenizer's {actual_vocab_size} "
            f"from {tokenizer_path}",
            flush=True,
        )
    return actual_vocab_size


def build_corpora(cfg: Any) -> tuple[TokenShardCorpus, TokenShardCorpus]:
    """The train and validation token-shard corpora."""
    train_path, _ = resolve_text_paths(cfg)
    train_manifest, val_manifest = resolve_token_shard_manifests(cfg, train_path=train_path)
    for manifest in (train_manifest, val_manifest):
        if not manifest.exists():
            raise FileNotFoundError(
                f"token shard manifest missing: {manifest}. "
                f"Run: python -m data.pretokenize_corpus --text-corpus {cfg.text_corpus}"
            )
    return TokenShardCorpus(train_manifest), TokenShardCorpus(val_manifest)


def sample_batch_any(
    *,
    cfg: Any,
    corpus: TokenShardCorpus,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return sample_batch_token_shards(
        corpus,
        batch_size=batch_size,
        seq_len=cfg.seq_len,
        generator=generator,
        device=device,
    )
