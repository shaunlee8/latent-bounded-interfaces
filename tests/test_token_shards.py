from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data.token_shards import TokenShardCorpus, corpus_numel, sample_batch_token_shards


def test_sample_batch_token_shards_reads_windows(tmp_path: Path) -> None:
    shard_dir = tmp_path / "tokens"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard0 = np.arange(12, dtype=np.int32)
    shard1 = np.arange(100, 116, dtype=np.int32)
    (shard_dir / "train_0000.bin").write_bytes(shard0.tobytes())
    (shard_dir / "train_0001.bin").write_bytes(shard1.tobytes())
    manifest = {
        "split": "train",
        "dtype": "int32",
        "num_tokens_total": int(shard0.size + shard1.size),
        "shards": [
            {"path": "train_0000.bin", "num_tokens": int(shard0.size)},
            {"path": "train_0001.bin", "num_tokens": int(shard1.size)},
        ],
    }
    manifest_path = shard_dir / "train_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    corpus = TokenShardCorpus(manifest_path)
    assert corpus_numel(corpus) == 28
    x, y = sample_batch_token_shards(
        corpus,
        batch_size=4,
        seq_len=4,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    assert x.shape == (4, 4)
    assert y.shape == (4, 4)
    assert torch.all(y[:, :-1] == x[:, 1:])
