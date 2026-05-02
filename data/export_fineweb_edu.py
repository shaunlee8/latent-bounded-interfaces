from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from data.data_paths import CORPORA_ROOT, DATA_ROOT


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream a bounded FineWeb-Edu subset into local train/val text files.")
    p.add_argument("--config", type=str, default="sample-10BT", help="FineWeb-Edu config to stream, e.g. sample-10BT")
    p.add_argument(
        "--output-subdir",
        type=str,
        default="fineweb_edu",
        help="subdirectory under CORPORA_ROOT for exported train/val text files",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="absolute or relative output directory for exported train/val text files; overrides --output-subdir",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=str(DATA_ROOT / "hf_cache"),
        help="Hugging Face datasets cache directory",
    )
    p.add_argument("--train-bytes", type=int, default=1_000_000_000, help="target raw UTF-8 bytes for train split")
    p.add_argument("--val-bytes", type=int, default=100_000_000, help="target raw UTF-8 bytes for val split")
    p.add_argument(
        "--dataset-name",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="HF dataset id; override only if mirroring elsewhere",
    )
    p.add_argument(
        "--force-exit-after-export",
        action="store_true",
        help="force os._exit(0) after export; useful only for hosts with datasets/pyarrow teardown crashes",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("datasets is required. Install it with: pip install datasets") from exc

        out_dir = Path(args.output_dir).expanduser() if args.output_dir else CORPORA_ROOT / args.output_subdir
        cache_dir = Path(args.cache_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_path = out_dir / "fineweb_edu_train.txt"
        val_path = out_dir / "fineweb_edu_val.txt"

        ds = load_dataset(args.dataset_name, args.config, split="train", streaming=True, cache_dir=str(cache_dir))

        written_train = 0
        written_val = 0
        n_docs = 0
        with train_path.open("wb") as f_train, val_path.open("wb") as f_val:
            for ex in ds:
                text = ex.get("text", "")
                if not text:
                    continue
                blob = (text.strip() + "\n").encode("utf-8", errors="ignore")
                if len(blob) <= 1:
                    continue
                if written_train < args.train_bytes:
                    remaining = args.train_bytes - written_train
                    chunk = blob[:remaining]
                    if chunk:
                        f_train.write(chunk)
                        written_train += len(chunk)
                elif written_val < args.val_bytes:
                    remaining = args.val_bytes - written_val
                    chunk = blob[:remaining]
                    if chunk:
                        f_val.write(chunk)
                        written_val += len(chunk)
                else:
                    break
                n_docs += 1
                if n_docs % 10000 == 0:
                    print(
                        f"[fineweb_edu] docs={n_docs} train_bytes={written_train} val_bytes={written_val}",
                        flush=True,
                    )

        print(f"[fineweb_edu] wrote train={train_path} bytes={written_train}", flush=True)
        print(f"[fineweb_edu] wrote val={val_path} bytes={written_val}", flush=True)
    finally:
        if args.force_exit_after_export:
            # Some hosts can crash at interpreter finalization inside datasets/pyarrow teardown.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
