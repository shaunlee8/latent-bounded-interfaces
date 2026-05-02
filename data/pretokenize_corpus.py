from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.bpe_tokenizer import TextTokenizerBackend, load_text_tokenizer
from data.data_paths import CORPORA_ROOT, LLAMA31_TOKENIZER_ROOT, LLAMA_TOKENIZER_ROOT, TOKENIZERS_ROOT

_BUNDLED_TEXT_CORPORA = {
    "tiny_shakespeare": ("tiny_shakespeare_train.txt", "tiny_shakespeare_val.txt"),
    "enwik8": ("enwik8_train.bin", "enwik8_val.bin"),
    "wikitext103_raw": ("wikitext103_raw_train.txt", "wikitext103_raw_val.txt"),
    "tinystories": ("tinystories_train.txt", "tinystories_val.txt"),
    "fineweb_edu": ("fineweb_edu/fineweb_edu_train.txt", "fineweb_edu/fineweb_edu_val.txt"),
}


def _resolve_text_paths(text_corpus: str, train_text_path: str, val_text_path: str) -> tuple[Path, Path | None]:
    if train_text_path:
        return Path(train_text_path), Path(val_text_path) if val_text_path else None
    train_name, val_name = _BUNDLED_TEXT_CORPORA[text_corpus]
    return CORPORA_ROOT / train_name, CORPORA_ROOT / val_name


def _resolve_tokenizer_path(
    text_corpus: str,
    train_path: Path,
    tokenizer_path: str,
    vocab_size: int,
    tokenizer_type: str,
) -> Path:
    if tokenizer_path:
        return Path(tokenizer_path)
    if tokenizer_type == "llama":
        legacy = TOKENIZERS_ROOT / "llama"
        if LLAMA_TOKENIZER_ROOT.exists():
            return LLAMA_TOKENIZER_ROOT
        if legacy.exists():
            return legacy
        return LLAMA_TOKENIZER_ROOT
    if tokenizer_type == "llama31":
        legacy = TOKENIZERS_ROOT / "llama31"
        if LLAMA31_TOKENIZER_ROOT.exists():
            return LLAMA31_TOKENIZER_ROOT
        if legacy.exists():
            return legacy
        return LLAMA31_TOKENIZER_ROOT
    if train_path.is_relative_to(CORPORA_ROOT):
        return TOKENIZERS_ROOT / f"{text_corpus}_sp_bpe_{vocab_size}.model"
    return train_path.parent / "tokenizers" / f"{train_path.stem}_sp_bpe_{vocab_size}.model"


def _tokenizer_tag(tokenizer_type: str, vocab_size: int) -> str:
    if tokenizer_type in {"llama", "llama31"}:
        return tokenizer_type
    return f"sp_bpe_{vocab_size}"


def _default_output_dir(text_corpus: str, train_path: Path, vocab_size: int, tokenizer_type: str) -> Path:
    tag = _tokenizer_tag(tokenizer_type, vocab_size)
    if train_path.is_relative_to(CORPORA_ROOT):
        base = train_path.parent if train_path.parent.name == text_corpus else (CORPORA_ROOT / text_corpus)
        return base / "tokens" / f"{text_corpus}_{tag}"
    return train_path.parent / "tokens" / f"{train_path.stem}_{tag}"


def _write_shards(split_name: str, src_path: Path, tokenizer: TextTokenizerBackend, out_dir: Path, shard_tokens: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[dict[str, object]] = []
    buffer: list[int] = []
    shard_idx = 0
    total_tokens = 0

    def flush() -> None:
        nonlocal buffer, shard_idx, total_tokens
        if not buffer:
            return
        arr = np.asarray(buffer, dtype=np.int32)
        shard_name = f"{split_name}_{shard_idx:04d}.bin"
        shard_path = out_dir / shard_name
        arr.tofile(shard_path)
        shard_paths.append({"path": shard_name, "num_tokens": int(arr.size)})
        total_tokens += int(arr.size)
        shard_idx += 1
        buffer = []

    with src_path.open("rb") as f:
        for raw_line in f:
            if not raw_line:
                continue
            if not raw_line.endswith(b"\n"):
                raw_line = raw_line + b"\n"
            ids = tokenizer.encode_bytes(raw_line)
            if not ids:
                continue
            buffer.extend(ids)
            if len(buffer) >= shard_tokens:
                flush()
        flush()

    if not shard_paths:
        raise ValueError(f"no tokens were produced for split {split_name}: {src_path}")

    manifest_path = out_dir / f"{split_name}_manifest.json"
    manifest = {
        "split": split_name,
        "dtype": "int32",
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "num_tokens_total": int(total_tokens),
        "shards": shard_paths,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[{split_name}] wrote {len(shard_paths)} shards total_tokens={total_tokens} manifest={manifest_path}", flush=True)
    return manifest_path


def pretokenize_sentencepiece_corpus(
    *,
    text_corpus: str,
    train_text_path: str,
    val_text_path: str,
    tokenizer_path: str,
    vocab_size: int,
    tokenizer_train_bytes: int,
    train_tokenizer: bool,
    shard_tokens: int,
    output_dir: str,
) -> tuple[Path, Path]:
    return pretokenize_text_corpus(
        text_corpus=text_corpus,
        train_text_path=train_text_path,
        val_text_path=val_text_path,
        tokenizer_type="sentencepiece",
        tokenizer_path=tokenizer_path,
        vocab_size=vocab_size,
        tokenizer_train_bytes=tokenizer_train_bytes,
        train_tokenizer=train_tokenizer,
        shard_tokens=shard_tokens,
        output_dir=output_dir,
    )


def pretokenize_text_corpus(
    *,
    text_corpus: str,
    train_text_path: str,
    val_text_path: str,
    tokenizer_type: str,
    tokenizer_path: str,
    vocab_size: int,
    tokenizer_train_bytes: int,
    train_tokenizer: bool,
    shard_tokens: int,
    output_dir: str,
) -> tuple[Path, Path]:
    train_path, val_path = _resolve_text_paths(text_corpus, train_text_path, val_text_path)
    if not train_path.exists():
        raise FileNotFoundError(f"train text not found: {train_path}")
    if val_path is None or not val_path.exists():
        raise FileNotFoundError(f"val text not found: {val_path}")
    resolved_tokenizer_path = _resolve_tokenizer_path(
        text_corpus,
        train_path,
        tokenizer_path,
        vocab_size,
        tokenizer_type,
    )
    tokenizer = load_text_tokenizer(
        tokenizer_type=tokenizer_type,
        tokenizer_path=resolved_tokenizer_path,
        train_path=train_path,
        vocab_size=vocab_size,
        train_bytes=tokenizer_train_bytes,
        force_train=train_tokenizer,
    )
    out_dir = Path(output_dir) if output_dir else _default_output_dir(text_corpus, train_path, vocab_size, tokenizer_type)
    train_manifest = _write_shards("train", train_path, tokenizer, out_dir, shard_tokens)
    val_manifest = _write_shards("val", val_path, tokenizer, out_dir, shard_tokens)
    return train_manifest, val_manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Pretokenize a text corpus into token shards.")
    p.add_argument("--text-corpus", type=str, default="fineweb_edu")
    p.add_argument("--train-text-path", type=str, default="")
    p.add_argument("--val-text-path", type=str, default="")
    p.add_argument("--tokenizer-type", type=str, default="sentencepiece")
    p.add_argument("--tokenizer-path", type=str, default="")
    p.add_argument("--train-tokenizer", action="store_true")
    p.add_argument("--tokenizer-train-bytes", type=int, default=8 * 1024 * 1024)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--shard-tokens", type=int, default=5_000_000)
    p.add_argument("--output-dir", type=str, default="")
    args = p.parse_args()
    pretokenize_text_corpus(
        text_corpus=args.text_corpus,
        train_text_path=args.train_text_path,
        val_text_path=args.val_text_path,
        tokenizer_type=args.tokenizer_type,
        tokenizer_path=args.tokenizer_path,
        vocab_size=args.vocab_size,
        tokenizer_train_bytes=args.tokenizer_train_bytes,
        train_tokenizer=args.train_tokenizer,
        shard_tokens=args.shard_tokens,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
