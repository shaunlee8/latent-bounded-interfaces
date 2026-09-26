from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("LBI_DATA_ROOT", "data")).expanduser()
CORPORA_ROOT = Path(os.environ.get("LBI_CORPORA_ROOT", DATA_ROOT / "corpora")).expanduser()
LLAMA_TOKENIZER_ROOT = Path(
    os.environ.get("LBI_LLAMA_TOKENIZER_ROOT", DATA_ROOT / "tokenizers" / "llama")
).expanduser()
