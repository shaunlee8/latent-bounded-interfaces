# Data And Tokenization

The canonical paper runs use FineWeb-Edu text exported locally, then pretokenized into int32 token shards with the original 32k LLaMA tokenizer.

## Paths

Set:

```bash
export LBI_DATA_ROOT=/path/to/lbi_data
export LBI_LLAMA_TOKENIZER_ROOT=${LBI_DATA_ROOT}/tokenizers/llama
```

The expected text layout is:

```text
${LBI_DATA_ROOT}/corpora/fineweb_edu/fineweb_edu_train.txt
${LBI_DATA_ROOT}/corpora/fineweb_edu/fineweb_edu_val.txt
```

The expected token-shard layout is:

```text
${LBI_DATA_ROOT}/corpora/fineweb_edu/tokens/fineweb_edu_llama/
  train_manifest.json
  val_manifest.json
  train_*.bin
  val_*.bin
```

The tokenizer directory is not committed. It should be a local Hugging Face-compatible original LLaMA tokenizer directory. The paper configuration uses:

- `TOKENIZER_TYPE=llama`
- `VOCAB_SIZE=32000`
- tied input/output embeddings

The paper uses the original 32k LLaMA tokenizer. Reviewers should request access to the official Meta Llama 2 Hugging Face repository:

```text
https://huggingface.co/meta-llama/Llama-2-7b-hf
```

After access is granted:

```bash
huggingface-cli login

huggingface-cli download meta-llama/Llama-2-7b-hf \
  --include "config.json" \
  --include "tokenizer.*" \
  --include "special_tokens_map.json" \
  --include "tokenizer_config.json" \
  --local-dir ${LBI_LLAMA_TOKENIZER_ROOT}
```

Validate the local tokenizer directory:

```bash
python - <<'PY'
import os
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    os.environ["LBI_LLAMA_TOKENIZER_ROOT"],
    use_fast=True,
    local_files_only=True,
)
print("vocab size:", len(tok))
assert len(tok) == 32000
PY
```

## Export FineWeb-Edu

```bash
${PYTHON_BIN:-python} -m data.export_fineweb_edu \
  --output-dir ${LBI_DATA_ROOT}/corpora/fineweb_edu \
  --cache-dir ${LBI_DATA_ROOT}/hf_cache \
  --config sample-10BT \
  --train-bytes 1000000000 \
  --val-bytes 100000000 \
  --force-exit-after-export
```

This streams `HuggingFaceFW/fineweb-edu` and writes bounded train/validation text files. The raw dataset and Hugging Face cache are not part of the repository.

`--force-exit-after-export` exits immediately after the files are written and flushed. It avoids a known `datasets`/`pyarrow` teardown crash on some hosts where Python aborts after successfully writing the requested byte counts.

## Pretokenize

```bash
${PYTHON_BIN:-python} -m data.pretokenize_corpus \
  --text-corpus fineweb_edu \
  --tokenizer-type llama \
  --vocab-size 32000 \
  --tokenizer-path ${LBI_LLAMA_TOKENIZER_ROOT} \
  --shard-tokens 5000000
```

Training uses `data_mode=text_bpe_sharded`, which reads `train_manifest.json` and `val_manifest.json` through `data.token_shards.TokenShardCorpus`.

## File Roles

- `data_paths.py`: environment-configurable roots for corpora and tokenizer directories.
- `export_fineweb_edu.py`: FineWeb-Edu text export.
- `pretokenize_corpus.py`: tokenizer loading and token-shard creation.
- `token_shards.py`: memory-mapped int32 shard sampler used by paper training.
- `bpe_tokenizer.py`: SentencePiece and Hugging Face tokenizer wrappers.
- `lm_data.py`: legacy: byte-level helpers for old data modes.
- `train_data.py`: legacy: synthetic sequence-task helpers for smoke tests and debug runs.

Canonical paper runs do not use `text_byte` or synthetic data; those paths are kept so the smoke tests and debug modes remain importable.
