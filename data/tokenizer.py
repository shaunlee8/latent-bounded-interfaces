"""The LLaMA 32k tokenizer, loaded from a local Hugging Face tokenizer directory."""

from __future__ import annotations

from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None  # type: ignore[assignment]


class LlamaTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        if AutoTokenizer is None:
            raise ImportError("transformers is required for the LLaMA tokenizer. Install it with: pip install transformers")
        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, local_files_only=True)

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    def encode_bytes(self, raw: bytes) -> list[int]:
        text = raw.decode("utf-8", errors="replace")
        return list(self.tokenizer.encode(text, add_special_tokens=False))


def load_llama_tokenizer(tokenizer_path: str | Path) -> LlamaTokenizer:
    model_path = Path(tokenizer_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"LLaMA tokenizer path not found: {model_path}. "
            "Provide a local Hugging Face tokenizer directory containing tokenizer.json and tokenizer_config.json."
        )
    return LlamaTokenizer(model_path)
