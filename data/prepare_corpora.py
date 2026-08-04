from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

from data.data_paths import CORPORA_ROOT


ENWIK8_URL = "https://mattmahoney.net/dc/enwik8.zip"
WIKITEXT103_RAW_URLS = [
    "https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/main/wikitext-103-raw-v1.zip",
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
]
TINYSTORIES_TRAIN_URLS = [
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
]
TINYSTORIES_VAL_URLS = [
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
]


def _download_file(url: str, dst: Path, max_redirects: int = 5) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request) as resp, dst.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1024 * 1024)
        return
    except urllib.error.HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308} and max_redirects > 0:
            location = exc.headers.get("Location")
            if location:
                _download_file(location, dst, max_redirects=max_redirects - 1)
                return
        raise


def _download_first(urls: list[str], dst: Path) -> str:
    last_err: Exception | None = None
    for url in urls:
        try:
            _download_file(url, dst)
            return url
        except Exception as exc:  # pragma: no cover - network dependent
            last_err = exc
    raise RuntimeError(
        f"failed downloading from all candidates: {urls}; last_error={type(last_err).__name__}: {last_err}"
    ) from last_err


def _write_bytes_subset(src: bytes, dst: Path, max_bytes: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    chunk = src[:max_bytes]
    dst.write_bytes(chunk)
    return len(chunk)


def _write_file_subset(src: Path, dst: Path, max_bytes: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as f:
        chunk = f.read(max_bytes)
    dst.write_bytes(chunk)
    return len(chunk)


def _prepare_enwik8(corpora_dir: Path) -> None:
    train_out = corpora_dir / "enwik8_train.bin"
    val_out = corpora_dir / "enwik8_val.bin"
    with tempfile.TemporaryDirectory(prefix="enwik8_") as td:
        td_path = Path(td)
        zip_path = td_path / "enwik8.zip"
        print(f"[enwik8] downloading {ENWIK8_URL}")
        _download_file(ENWIK8_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open("enwik8") as f:
                data = f.read()
        n = len(data)
        if n < 10_000_000:
            raise RuntimeError(f"unexpected enwik8 size: {n}")
        n_val = max(1, n // 20)  # 5%
        train_n = _write_bytes_subset(data[:-n_val], train_out, len(data) - n_val)
        val_n = _write_bytes_subset(data[-n_val:], val_out, n_val)
    print(f"[enwik8] wrote train={train_n} bytes val={val_n} bytes")


def _prepare_wikitext103_raw(corpora_dir: Path, max_bytes: int) -> None:
    train_out = corpora_dir / "wikitext103_raw_train.txt"
    val_out = corpora_dir / "wikitext103_raw_val.txt"
    train_cap = int(max_bytes * 0.93)
    val_cap = max_bytes - train_cap
    with tempfile.TemporaryDirectory(prefix="wikitext103_") as td:
        td_path = Path(td)
        zip_path = td_path / "wikitext-103-raw-v1.zip"
        source_url = _download_first(WIKITEXT103_RAW_URLS, zip_path)
        print(f"[wikitext103_raw] downloaded from {source_url}")
        with zipfile.ZipFile(zip_path) as zf:
            train_member = "wikitext-103-raw/wiki.train.raw"
            val_member = "wikitext-103-raw/wiki.valid.raw"
            with zf.open(train_member) as f:
                train_data = f.read()
            with zf.open(val_member) as f:
                val_data = f.read()
    train_n = _write_bytes_subset(train_data, train_out, train_cap)
    val_n = _write_bytes_subset(val_data, val_out, val_cap)
    print(f"[wikitext103_raw] wrote train={train_n} bytes val={val_n} bytes (target_total={max_bytes})")


def _prepare_tinystories(corpora_dir: Path, max_bytes: int) -> None:
    train_out = corpora_dir / "tinystories_train.txt"
    val_out = corpora_dir / "tinystories_val.txt"
    train_cap = int(max_bytes * 0.93)
    val_cap = max_bytes - train_cap
    with tempfile.TemporaryDirectory(prefix="tinystories_") as td:
        td_path = Path(td)
        train_tmp = td_path / "tinystories_train.txt"
        val_tmp = td_path / "tinystories_val.txt"
        train_url = _download_first(TINYSTORIES_TRAIN_URLS, train_tmp)
        val_url = _download_first(TINYSTORIES_VAL_URLS, val_tmp)
        print(f"[tinystories] downloaded train from {train_url}")
        print(f"[tinystories] downloaded val from {val_url}")
        train_n = _write_file_subset(train_tmp, train_out, train_cap)
        val_n = _write_file_subset(val_tmp, val_out, val_cap)
    print(f"[tinystories] wrote train={train_n} bytes val={val_n} bytes (target_total={max_bytes})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare bundled corpora for byte-level training.")
    parser.add_argument(
        "--corpora",
        type=str,
        nargs="+",
        default=["enwik8", "wikitext103_raw", "tinystories"],
        help="subset of: enwik8 wikitext103_raw tinystories",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=150 * 1024 * 1024,
        help="target total bytes for each subset corpus (except full enwik8).",
    )
    args = parser.parse_args()
    corpora_dir = CORPORA_ROOT
    corpora_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.corpora)
    valid = {"enwik8", "wikitext103_raw", "tinystories"}
    bad = selected - valid
    if bad:
        raise ValueError(f"unknown corpora: {sorted(bad)}")

    if "enwik8" in selected:
        _prepare_enwik8(corpora_dir)
    if "wikitext103_raw" in selected:
        _prepare_wikitext103_raw(corpora_dir, max_bytes=args.max_bytes)
    if "tinystories" in selected:
        _prepare_tinystories(corpora_dir, max_bytes=args.max_bytes)

    print("done")


if __name__ == "__main__":
    main()
