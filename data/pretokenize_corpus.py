"""Pretokenize the FineWeb-Edu text export (or explicit text files) into int32
token shards with the LLaMA tokenizer, writing a manifest per split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.data_paths import CORPORA_ROOT, LLAMA_TOKENIZER_ROOT
from data.tokenizer import LlamaTokenizer, load_llama_tokenizer

FINEWEB_EDU_TEXT = ("fineweb_edu/fineweb_edu_train.txt", "fineweb_edu/fineweb_edu_val.txt")


def _resolve_text_paths(text_corpus: str, train_text_path: str, val_text_path: str) -> tuple[Path, Path | None]:
    if train_text_path:
        return Path(train_text_path), Path(val_text_path) if val_text_path else None
    if text_corpus != "fineweb_edu":
        raise ValueError(f"text_corpus must be fineweb_edu unless --train-text-path is set; got {text_corpus!r}")
    train_name, val_name = FINEWEB_EDU_TEXT
    return CORPORA_ROOT / train_name, CORPORA_ROOT / val_name


def _default_output_dir(text_corpus: str, train_path: Path) -> Path:
    if train_path.is_relative_to(CORPORA_ROOT):
        base = train_path.parent if train_path.parent.name == text_corpus else (CORPORA_ROOT / text_corpus)
        return base / "tokens" / f"{text_corpus}_llama"
    return train_path.parent / "tokens" / f"{train_path.stem}_llama"


def _write_shards(split_name: str, src_path: Path, tokenizer: LlamaTokenizer, out_dir: Path, shard_tokens: int) -> Path:
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
        arr.tofile(out_dir / shard_name)
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


def pretokenize_text_corpus(
    *,
    text_corpus: str,
    train_text_path: str,
    val_text_path: str,
    tokenizer_path: str,
    shard_tokens: int,
    output_dir: str,
) -> tuple[Path, Path]:
    train_path, val_path = _resolve_text_paths(text_corpus, train_text_path, val_text_path)
    if not train_path.exists():
        raise FileNotFoundError(f"train text not found: {train_path}")
    if val_path is None or not val_path.exists():
        raise FileNotFoundError(f"val text not found: {val_path}")
    tokenizer = load_llama_tokenizer(Path(tokenizer_path) if tokenizer_path else LLAMA_TOKENIZER_ROOT)
    out_dir = Path(output_dir) if output_dir else _default_output_dir(text_corpus, train_path)
    train_manifest = _write_shards("train", train_path, tokenizer, out_dir, shard_tokens)
    val_manifest = _write_shards("val", val_path, tokenizer, out_dir, shard_tokens)
    return train_manifest, val_manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Pretokenize a text corpus into token shards with the LLaMA tokenizer.")
    p.add_argument("--text-corpus", type=str, default="fineweb_edu")
    p.add_argument("--train-text-path", type=str, default="")
    p.add_argument("--val-text-path", type=str, default="")
    p.add_argument("--tokenizer-path", type=str, default="")
    p.add_argument("--shard-tokens", type=int, default=5_000_000)
    p.add_argument("--output-dir", type=str, default="")
    args = p.parse_args()
    pretokenize_text_corpus(
        text_corpus=args.text_corpus,
        train_text_path=args.train_text_path,
        val_text_path=args.val_text_path,
        tokenizer_path=args.tokenizer_path,
        shard_tokens=args.shard_tokens,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
