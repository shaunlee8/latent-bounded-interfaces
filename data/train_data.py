from __future__ import annotations

import torch

"""Synthetic data helpers for smoke tests and debug experiments.

Canonical paper runs use FineWeb-Edu token shards via data.token_shards.
"""


def _marker_tokens(vocab_size: int) -> tuple[int, int, int]:
    if vocab_size < 8:
        raise ValueError("vocab_size must be >= 8 for long-range synthetic tasks.")
    bos = vocab_size - 1
    sep = vocab_size - 2
    pad = vocab_size - 3
    return bos, sep, pad


def _build_arith_corpus(
    *,
    num_sequences: int,
    total_len: int,
    vocab_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    starts = torch.randint(0, vocab_size, (num_sequences, 1), generator=generator, device=device)
    strides = torch.randint(1, min(vocab_size, 8), (num_sequences, 1), generator=generator, device=device)
    offsets = torch.arange(total_len, device=device).unsqueeze(0)
    seq = (starts + strides * offsets) % vocab_size
    return seq.to(torch.long)


def _build_copy_style_corpus(
    *,
    num_sequences: int,
    total_len: int,
    vocab_size: int,
    generator: torch.Generator,
    device: torch.device,
    mode: str,
) -> torch.Tensor:
    bos, sep, pad = _marker_tokens(vocab_size)
    content_vocab = vocab_size - 3
    payload_len = max(2, (total_len - 2) // 2)
    payload = torch.randint(0, content_vocab, (num_sequences, payload_len), generator=generator, device=device)

    if mode == "copy":
        out_tokens = payload
    elif mode == "reverse":
        out_tokens = payload.flip(dims=(1,))
    elif mode == "selective_copy":
        out_tokens = payload[:, ::2]
    else:
        raise ValueError("mode must be one of: copy, reverse, selective_copy")

    seq = torch.full((num_sequences, total_len), fill_value=pad, device=device, dtype=torch.long)
    seq[:, 0] = bos
    seq[:, 1 : 1 + payload_len] = payload
    sep_pos = 1 + payload_len
    seq[:, sep_pos] = sep
    start = sep_pos + 1
    end = min(total_len, start + out_tokens.size(1))
    copy_len = max(0, end - start)
    if copy_len > 0:
        seq[:, start:end] = out_tokens[:, :copy_len]
    return seq


def _build_induction_corpus(
    *,
    num_sequences: int,
    total_len: int,
    vocab_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    bos, sep, pad = _marker_tokens(vocab_size)
    content_vocab = vocab_size - 3
    # Keep room for BOS, prompt/query separators, query key, and target slot.
    max_pairs = max(1, (total_len - 6) // 2)
    pair_count = max_pairs

    keys = torch.randint(0, content_vocab, (num_sequences, pair_count), generator=generator, device=device)
    vals = torch.randint(0, content_vocab, (num_sequences, pair_count), generator=generator, device=device)
    kv = torch.stack((keys, vals), dim=-1).reshape(num_sequences, pair_count * 2)

    query_idx = torch.randint(0, pair_count, (num_sequences,), generator=generator, device=device)
    query_key = keys.gather(1, query_idx.unsqueeze(1)).squeeze(1)
    query_val = vals.gather(1, query_idx.unsqueeze(1)).squeeze(1)

    seq = torch.full((num_sequences, total_len), fill_value=pad, device=device, dtype=torch.long)
    pos = 0
    seq[:, pos] = bos
    pos += 1
    end_kv = min(total_len, pos + kv.size(1))
    kv_len = end_kv - pos
    if kv_len > 0:
        seq[:, pos:end_kv] = kv[:, :kv_len]
    pos = end_kv
    if pos < total_len:
        seq[:, pos] = sep
        pos += 1
    if pos < total_len:
        seq[:, pos] = query_key
        pos += 1
    if pos < total_len:
        seq[:, pos] = sep
        pos += 1
    if pos < total_len:
        seq[:, pos] = query_val
    return seq


def build_synthetic_corpus(
    *,
    num_sequences: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
    task: str = "arith",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build deterministic token sequences of shape [N, seq_len + 1]."""
    if num_sequences <= 0:
        raise ValueError("num_sequences must be > 0")
    if seq_len <= 1:
        raise ValueError("seq_len must be > 1")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be > 1")
    dev = device if device is not None else torch.device("cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    total_len = seq_len + 1

    if task == "arith":
        return _build_arith_corpus(
            num_sequences=num_sequences,
            total_len=total_len,
            vocab_size=vocab_size,
            generator=gen,
            device=dev,
        )
    if task in ("copy", "reverse", "selective_copy"):
        return _build_copy_style_corpus(
            num_sequences=num_sequences,
            total_len=total_len,
            vocab_size=vocab_size,
            generator=gen,
            device=dev,
            mode=task,
        )
    if task == "induction":
        return _build_induction_corpus(
            num_sequences=num_sequences,
            total_len=total_len,
            vocab_size=vocab_size,
            generator=gen,
            device=dev,
        )
    raise ValueError("task must be one of: arith, copy, reverse, selective_copy, induction")


def sample_batch(
    corpus: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample a token batch and return (x, y) for next-token CE."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    n = corpus.size(0)
    idx = torch.randint(0, n, (batch_size,), generator=generator, device=corpus.device)
    seq = corpus.index_select(0, idx)
    x = seq[:, :-1].to(device=device, non_blocking=True)
    y = seq[:, 1:].to(device=device, non_blocking=True)
    return x, y
