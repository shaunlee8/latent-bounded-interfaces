from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT = Path(os.environ.get("LBI_DATA_ROOT", "data")).expanduser()

# Backward-compatible alias for older helper scripts.
U2_DATA_ROOT = DATA_ROOT
CORPORA_ROOT = Path(os.environ.get("LBI_CORPORA_ROOT", U2_DATA_ROOT / "corpora")).expanduser()
TOKENIZERS_ROOT = Path(os.environ.get("LBI_TOKENIZERS_ROOT", CORPORA_ROOT / "tokenizers")).expanduser()
PRETRAINED_TOKENIZERS_ROOT = Path(
    os.environ.get("LBI_PRETRAINED_TOKENIZERS_ROOT", U2_DATA_ROOT / "tokenizers")
).expanduser()
LLAMA_TOKENIZER_ROOT = Path(
    os.environ.get("LBI_LLAMA_TOKENIZER_ROOT", PRETRAINED_TOKENIZERS_ROOT / "llama")
).expanduser()
LLAMA31_TOKENIZER_ROOT = Path(
    os.environ.get("LBI_LLAMA31_TOKENIZER_ROOT", PRETRAINED_TOKENIZERS_ROOT / "llama-3.1")
).expanduser()
