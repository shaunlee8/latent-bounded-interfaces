from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from data.bpe_tokenizer import build_token_stream, load_text_tokenizer
from data.data_paths import CORPORA_ROOT, LLAMA31_TOKENIZER_ROOT, LLAMA_TOKENIZER_ROOT, TOKENIZERS_ROOT
from data.lm_data import build_byte_stream, sample_batch_stream, split_stream_train_val
from data.token_shards import TokenShardCorpus, sample_batch_token_shards
from data.train_data import build_synthetic_corpus, sample_batch

BUNDLED_TEXT_CORPORA: dict[str, tuple[str, str]] = {
    "tiny_shakespeare": ("tiny_shakespeare_train.txt", "tiny_shakespeare_val.txt"),
    "enwik8": ("enwik8_train.bin", "enwik8_val.bin"),
    "wikitext103_raw": ("wikitext103_raw_train.txt", "wikitext103_raw_val.txt"),
    "tinystories": ("tinystories_train.txt", "tinystories_val.txt"),
    "fineweb_edu": ("fineweb_edu/fineweb_edu_train.txt", "fineweb_edu/fineweb_edu_val.txt"),
}


def resolve_text_paths(cfg: Any) -> tuple[Path, Path | None, Path]:
    corpora_root = CORPORA_ROOT
    if cfg.train_text_path:
        train_path = Path(cfg.train_text_path)
        val_path = Path(cfg.val_text_path) if cfg.val_text_path else None
        return train_path, val_path, corpora_root

    train_name, val_name = BUNDLED_TEXT_CORPORA[cfg.text_corpus]
    train_path = corpora_root / train_name
    val_path = corpora_root / val_name
    if not train_path.exists():
        raise FileNotFoundError(
            f"bundled corpus file missing: {train_path}. "
            "Prepare the corpus with `python -m data.export_fineweb_edu` or pass explicit "
            "`--train-text-path`/`--val-text-path` values."
        )
    return train_path, val_path, corpora_root


def resolve_tokenizer_path(cfg: Any, *, train_path: Path) -> Path:
    if cfg.tokenizer_path:
        return Path(cfg.tokenizer_path)
    if cfg.tokenizer_type == "llama":
        legacy = TOKENIZERS_ROOT / "llama"
        if LLAMA_TOKENIZER_ROOT.exists():
            return LLAMA_TOKENIZER_ROOT
        if legacy.exists():
            return legacy
        return LLAMA_TOKENIZER_ROOT
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


def resolve_token_shards_dir(cfg: Any, *, train_path: Path) -> Path:
    if cfg.tokenizer_type in {"llama", "llama31"}:
        tag = cfg.tokenizer_type
    else:
        tag = f"sp_bpe_{cfg.vocab_size}"
    if cfg.token_shards_dir:
        return Path(cfg.token_shards_dir)
    if not cfg.train_text_path:
        return CORPORA_ROOT / cfg.text_corpus / "tokens" / f"{cfg.text_corpus}_{tag}"
    return train_path.parent / "tokens" / f"{train_path.stem}_{tag}"


def resolve_token_shard_manifests(cfg: Any, *, train_path: Path) -> tuple[Path, Path]:
    shard_dir = resolve_token_shards_dir(cfg, train_path=train_path)
    return shard_dir / "train_manifest.json", shard_dir / "val_manifest.json"


def resolve_runtime_vocab_size(cfg: Any) -> int:
    if cfg.data_mode not in {"text_bpe", "text_bpe_sharded"} or cfg.tokenizer_type not in {"llama", "llama31"}:
        return int(cfg.vocab_size)
    train_path, _, _ = resolve_text_paths(cfg)
    tokenizer_path = resolve_tokenizer_path(cfg, train_path=train_path)
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
            f"[tokenizer] overriding vocab_size from {cfg.vocab_size} to {cfg.tokenizer_type} tokenizer vocab_size={actual_vocab_size} "
            f"from {tokenizer_path}",
            flush=True,
        )
    return actual_vocab_size


def build_corpora(
    cfg: Any,
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

    train_path, val_path, _ = resolve_text_paths(cfg)
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
        train_manifest, val_manifest = resolve_token_shard_manifests(cfg, train_path=train_path)
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

    tokenizer_path = resolve_tokenizer_path(cfg, train_path=train_path)
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


def sample_batch_any(
    *,
    cfg: Any,
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
