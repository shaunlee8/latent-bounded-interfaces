from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class TokenShardInfo:
    path: str
    num_tokens: int


class TokenShardCorpus:
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.dtype = manifest.get("dtype", "int32")
        if self.dtype != "int32":
            raise ValueError(f"unsupported shard dtype: {self.dtype}")
        self.shards = [
            TokenShardInfo(path=item["path"], num_tokens=int(item["num_tokens"]))
            for item in manifest["shards"]
        ]
        if not self.shards:
            raise ValueError(f"no shards listed in manifest: {self.manifest_path}")
        self.num_tokens_total = int(manifest.get("num_tokens_total", sum(s.num_tokens for s in self.shards)))
        self._arrays: dict[int, np.memmap] = {}

    def _resolve_path(self, shard_idx: int) -> Path:
        return (self.manifest_path.parent / self.shards[shard_idx].path).resolve()

    def _array(self, shard_idx: int) -> np.memmap:
        arr = self._arrays.get(shard_idx)
        if arr is None:
            info = self.shards[shard_idx]
            arr = np.memmap(self._resolve_path(shard_idx), dtype=np.int32, mode="r", shape=(info.num_tokens,))
            self._arrays[shard_idx] = arr
        return arr

    def valid_starts_per_shard(self, seq_len: int) -> list[int]:
        needed = seq_len + 1
        return [max(0, info.num_tokens - needed + 1) for info in self.shards]


def sample_batch_token_shards(
    corpus: TokenShardCorpus,
    *,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")

    valid = corpus.valid_starts_per_shard(seq_len)
    total_valid = sum(valid)
    if total_valid <= 0:
        raise ValueError("no shard contains enough tokens for the requested seq_len")

    cumulative: list[int] = []
    running = 0
    for count in valid:
        running += count
        cumulative.append(running)

    tickets = torch.randint(0, total_valid, (batch_size,), generator=generator, device=torch.device("cpu")).tolist()
    needed = seq_len + 1
    windows: list[torch.Tensor] = []
    for ticket in tickets:
        shard_idx = bisect_right(cumulative, ticket)
        prev = 0 if shard_idx == 0 else cumulative[shard_idx - 1]
        local_start = ticket - prev
        arr = corpus._array(shard_idx)
        window = np.asarray(arr[local_start : local_start + needed], dtype=np.int64)
        windows.append(torch.from_numpy(window.copy()))

    seq = torch.stack(windows, dim=0)
    x = seq[:, :-1].to(device=device, non_blocking=True)
    y = seq[:, 1:].to(device=device, non_blocking=True)
    return x, y


def corpus_numel(corpus: Any) -> int:
    if isinstance(corpus, torch.Tensor):
        return int(corpus.numel())
    if isinstance(corpus, TokenShardCorpus):
        return int(corpus.num_tokens_total)
    raise TypeError(f"unsupported corpus type: {type(corpus).__name__}")
