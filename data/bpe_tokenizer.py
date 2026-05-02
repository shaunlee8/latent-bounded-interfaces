from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Protocol

import torch

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover - optional dependency
    spm = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None  # type: ignore[assignment]


class TextTokenizerBackend(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode_bytes(self, raw: bytes) -> list[int]: ...


_PRETRAINED_HF_TOKENIZER_TYPES = {"llama", "llama31"}


def _validate_loaded_tokenizer_vocab_size(*, tokenizer_type: str, requested_vocab_size: int, actual_vocab_size: int) -> None:
    if tokenizer_type in _PRETRAINED_HF_TOKENIZER_TYPES:
        return
    if requested_vocab_size != actual_vocab_size:
        raise ValueError(
            f"{tokenizer_type} tokenizer vocab mismatch: requested vocab_size={requested_vocab_size}, "
            f"but loaded tokenizer has vocab_size={actual_vocab_size}"
        )


def _split_bytes_into_pieces(raw: bytes) -> list[bytes]:
    pieces: list[bytes] = []
    start = 0
    in_space = None
    for idx, value in enumerate(raw):
        is_space = chr(value).isspace()
        if in_space is None:
            in_space = is_space
            start = idx
            continue
        if is_space != in_space:
            pieces.append(raw[start:idx])
            start = idx
            in_space = is_space
    if raw:
        pieces.append(raw[start:])
    return [piece for piece in pieces if piece]


def _merge_pair(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    merged: list[int] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


@dataclass
class ByteBPETokenizer:
    merges: list[tuple[int, int, int]]
    token_bytes: dict[int, bytes]

    @property
    def vocab_size(self) -> int:
        return len(self.token_bytes)

    @property
    def pair_to_merge(self) -> dict[tuple[int, int], tuple[int, int]]:
        return {(a, b): (rank, new_id) for rank, (a, b, new_id) in enumerate(self.merges)}

    def encode_piece(self, piece: bytes) -> list[int]:
        if not piece:
            return []
        ranks = self.pair_to_merge
        word = tuple(piece[i] for i in range(len(piece)))
        while len(word) >= 2:
            best_idx = -1
            best_rank = None
            best_new_id = -1
            for idx in range(len(word) - 1):
                hit = ranks.get((word[idx], word[idx + 1]))
                if hit is None:
                    continue
                rank, new_id = hit
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_idx = idx
                    best_new_id = new_id
            if best_idx < 0:
                break
            word = word[:best_idx] + (best_new_id,) + word[best_idx + 2 :]
        return list(word)

    def encode_bytes(self, raw: bytes) -> list[int]:
        cache: dict[bytes, list[int]] = {}
        ids: list[int] = []
        for piece in _split_bytes_into_pieces(raw):
            token_ids = cache.get(piece)
            if token_ids is None:
                token_ids = self.encode_piece(piece)
                cache[piece] = token_ids
            ids.extend(token_ids)
        return ids

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "merges": self.merges,
            "token_bytes": {str(k): list(v) for k, v in self.token_bytes.items()},
        }
        out_path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(int(x) for x in row) for row in payload["merges"]]
        token_bytes = {int(k): bytes(v) for k, v in payload["token_bytes"].items()}
        return cls(merges=merges, token_bytes=token_bytes)


def train_byte_bpe_tokenizer(
    path: str | Path,
    *,
    vocab_size: int,
    max_bytes: int,
    min_pair_count: int = 2,
) -> ByteBPETokenizer:
    if vocab_size <= 256:
        raise ValueError("byte BPE vocab_size must be > 256")
    raw = Path(path).read_bytes()
    if len(raw) == 0:
        raise ValueError(f"empty tokenizer training file: {path}")
    sample = raw[:max_bytes]
    piece_counts = Counter(_split_bytes_into_pieces(sample))
    words = {tuple(piece): freq for piece, freq in piece_counts.items()}
    token_bytes = {idx: bytes([idx]) for idx in range(256)}
    merges: list[tuple[int, int, int]] = []
    next_id = 256

    while next_id < vocab_size:
        pair_counts: Counter[tuple[int, int]] = Counter()
        for word, freq in words.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += freq
        if not pair_counts:
            break
        (left, right), count = pair_counts.most_common(1)[0]
        if count < min_pair_count:
            break
        token_bytes[next_id] = token_bytes[left] + token_bytes[right]
        new_words: dict[tuple[int, ...], int] = {}
        for word, freq in words.items():
            merged = _merge_pair(word, (left, right), next_id)
            new_words[merged] = new_words.get(merged, 0) + freq
        words = new_words
        merges.append((left, right, next_id))
        next_id += 1

    return ByteBPETokenizer(merges=merges, token_bytes=token_bytes)


def load_or_train_byte_bpe_tokenizer(
    *,
    tokenizer_path: str | Path,
    train_path: str | Path,
    vocab_size: int,
    train_bytes: int,
    force_train: bool = False,
    min_pair_count: int = 2,
) -> ByteBPETokenizer:
    path = Path(tokenizer_path)
    if path.exists() and not force_train:
        return ByteBPETokenizer.load(path)
    tokenizer = train_byte_bpe_tokenizer(
        train_path,
        vocab_size=vocab_size,
        max_bytes=train_bytes,
        min_pair_count=min_pair_count,
    )
    tokenizer.save(path)
    return tokenizer


def build_bpe_stream(
    path: str | Path,
    *,
    tokenizer: ByteBPETokenizer,
    device: torch.device | None = None,
) -> torch.Tensor:
    raw = Path(path).read_bytes()
    if len(raw) == 0:
        raise ValueError(f"text file is empty: {path}")
    ids = tokenizer.encode_bytes(raw)
    if not ids:
        raise ValueError(f"tokenizer produced empty stream for: {path}")
    dev = device if device is not None else torch.device("cpu")
    return torch.tensor(ids, dtype=torch.long, device=dev)


class SentencePieceBPETokenizer:
    def __init__(self, model_path: str | Path) -> None:
        if spm is None:
            raise ImportError("sentencepiece is required for text_bpe mode. Install it with: pip install sentencepiece")
        self.model_path = str(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=self.model_path)

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())

    def encode_bytes(self, raw: bytes) -> list[int]:
        text = raw.decode("utf-8", errors="replace")
        return list(self.processor.encode(text, out_type=int))


class HFAutoTextTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        if AutoTokenizer is None:
            raise ImportError("transformers is required for HF tokenizer support. Install it with: pip install transformers")
        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, local_files_only=True)

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    def encode_bytes(self, raw: bytes) -> list[int]:
        text = raw.decode("utf-8", errors="replace")
        return list(self.tokenizer.encode(text, add_special_tokens=False))


def train_sentencepiece_bpe_tokenizer(
    train_path: str | Path,
    *,
    model_prefix: str | Path,
    vocab_size: int,
    train_bytes: int,
) -> SentencePieceBPETokenizer:
    if spm is None:
        raise ImportError("sentencepiece is required for text_bpe mode. Install it with: pip install sentencepiece")
    src = Path(train_path)
    raw = src.read_bytes()
    if len(raw) == 0:
        raise ValueError(f"empty tokenizer training file: {src}")
    sample = raw[:train_bytes]
    out_prefix = Path(model_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spm_train_") as td:
        sample_path = Path(td) / "tokenizer_sample.txt"
        sample_path.write_text(sample.decode("utf-8", errors="replace"), encoding="utf-8")
        spm.SentencePieceTrainer.train(
            input=str(sample_path),
            model_prefix=str(out_prefix),
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            unk_id=0,
            byte_fallback=True,
            normalization_rule_name="identity",
            max_sentence_length=1_000_000,
        )
    return SentencePieceBPETokenizer(f"{out_prefix}.model")


def load_or_train_sentencepiece_bpe_tokenizer(
    *,
    tokenizer_path: str | Path,
    train_path: str | Path,
    vocab_size: int,
    train_bytes: int,
    force_train: bool = False,
) -> SentencePieceBPETokenizer:
    model_path = Path(tokenizer_path)
    if model_path.suffix != ".model":
        model_path = model_path.with_suffix(".model")
    if model_path.exists() and not force_train:
        return SentencePieceBPETokenizer(model_path)
    prefix = model_path.with_suffix("")
    return train_sentencepiece_bpe_tokenizer(
        train_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        train_bytes=train_bytes,
    )


def build_sentencepiece_bpe_stream(
    path: str | Path,
    *,
    tokenizer: SentencePieceBPETokenizer,
    device: torch.device | None = None,
) -> torch.Tensor:
    raw = Path(path).read_bytes()
    if len(raw) == 0:
        raise ValueError(f"text file is empty: {path}")
    ids = tokenizer.encode_bytes(raw)
    if not ids:
        raise ValueError(f"tokenizer produced empty stream for: {path}")
    dev = device if device is not None else torch.device("cpu")
    return torch.tensor(ids, dtype=torch.long, device=dev)


def load_text_tokenizer(
    *,
    tokenizer_type: str,
    tokenizer_path: str | Path,
    train_path: str | Path,
    vocab_size: int,
    train_bytes: int,
    force_train: bool = False,
) -> TextTokenizerBackend:
    if tokenizer_type == "sentencepiece":
        tokenizer = load_or_train_sentencepiece_bpe_tokenizer(
            tokenizer_path=tokenizer_path,
            train_path=train_path,
            vocab_size=vocab_size,
            train_bytes=train_bytes,
            force_train=force_train,
        )
        _validate_loaded_tokenizer_vocab_size(
            tokenizer_type=tokenizer_type,
            requested_vocab_size=vocab_size,
            actual_vocab_size=tokenizer.vocab_size,
        )
        return tokenizer
    if tokenizer_type == "llama31":
        if force_train:
            raise ValueError("llama31 tokenizer does not support --train-tokenizer; provide a local tokenizer path instead")
        model_path = Path(tokenizer_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"llama31 tokenizer path not found: {model_path}. "
                "Provide a local HF-compatible tokenizer directory containing files such as tokenizer.json "
                "and tokenizer_config.json."
            )
        return HFAutoTextTokenizer(model_path)
    if tokenizer_type == "llama":
        if force_train:
            raise ValueError("llama tokenizer does not support --train-tokenizer; provide a local tokenizer path instead")
        model_path = Path(tokenizer_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"llama tokenizer path not found: {model_path}. "
                "Provide a local HF-compatible 32k LLaMA tokenizer directory containing files such as tokenizer.json "
                "and tokenizer_config.json."
            )
        return HFAutoTextTokenizer(model_path)
    raise ValueError(f"unsupported tokenizer_type: {tokenizer_type}")


def build_token_stream(
    path: str | Path,
    *,
    tokenizer: TextTokenizerBackend,
    device: torch.device | None = None,
) -> torch.Tensor:
    raw = Path(path).read_bytes()
    if len(raw) == 0:
        raise ValueError(f"text file is empty: {path}")
    ids = tokenizer.encode_bytes(raw)
    if not ids:
        raise ValueError(f"tokenizer produced empty stream for: {path}")
    dev = device if device is not None else torch.device("cpu")
    return torch.tensor(ids, dtype=torch.long, device=dev)
