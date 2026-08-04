from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable, Iterable

from data.data_paths import U2_DATA_ROOT


DEFAULT_OUTPUT_DIR = U2_DATA_ROOT / "downstream"
DEFAULT_CACHE_DIR = U2_DATA_ROOT / "hf_cache"


def _load_dataset(*args: Any, cache_dir: Path, split: str, **kwargs: Any):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required. Install it with: pip install datasets") from exc
    return load_dataset(*args, split=split, cache_dir=str(cache_dir), **kwargs)


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"expected string field, got {type(value).__name__}")
    return " ".join(value.replace("\n", " ").split())


def _choice_suffix(text: str) -> str:
    text = _clean_text(text)
    if not text:
        return text
    if text[0].isspace() or text[0] in ".,;:!?)]}":
        return text
    return " " + text


def _answer_to_index(answer: Any, *, labels: list[str] | None, choices: list[str]) -> int:
    if isinstance(answer, int):
        idx = answer
    elif isinstance(answer, str):
        stripped = answer.strip()
        if labels and stripped in labels:
            idx = labels.index(stripped)
        elif stripped.isdigit():
            idx = int(stripped)
        elif len(stripped) == 1 and "A" <= stripped.upper() <= "Z":
            idx = ord(stripped.upper()) - ord("A")
        elif stripped in choices:
            idx = choices.index(stripped)
        else:
            raise ValueError(f"could not map answer to choice index: {answer!r}")
    else:
        raise ValueError(f"unsupported answer type: {type(answer).__name__}")
    if idx < 0 or idx >= len(choices):
        raise ValueError(f"answer index {idx} out of range for {len(choices)} choices")
    return idx


def _convert_hellaswag(example: dict[str, Any]) -> dict[str, Any]:
    prompt = _clean_text(example.get("ctx", ""))
    endings = example.get("endings")
    if not isinstance(endings, list) or not endings:
        raise ValueError("HellaSwag example missing endings list")
    choices = [_choice_suffix(str(item)) for item in endings]
    answer = _answer_to_index(example.get("label"), labels=None, choices=choices)
    return {
        "id": str(example.get("ind", example.get("id", ""))),
        "task": "hellaswag",
        "prompt": prompt,
        "choices": choices,
        "answer": answer,
    }


def _convert_piqa(example: dict[str, Any]) -> dict[str, Any]:
    goal = _clean_text(example.get("goal", ""))
    choices = [_choice_suffix(str(example.get("sol1", ""))), _choice_suffix(str(example.get("sol2", "")))]
    answer = _answer_to_index(example.get("label"), labels=None, choices=choices)
    return {
        "id": str(example.get("id", "")),
        "task": "piqa",
        "prompt": f"Question: {goal}\nAnswer:",
        "choices": choices,
        "answer": answer,
    }


def _convert_arc_easy(example: dict[str, Any]) -> dict[str, Any]:
    question = _clean_text(example.get("question", ""))
    raw_choices = example.get("choices")
    if not isinstance(raw_choices, dict):
        raise ValueError("ARC example missing choices dict")
    texts = raw_choices.get("text")
    labels = raw_choices.get("label")
    if not isinstance(texts, list) or not texts:
        raise ValueError("ARC example missing choices.text list")
    label_list = [str(item) for item in labels] if isinstance(labels, list) else None
    choices = [_choice_suffix(str(item)) for item in texts]
    answer = _answer_to_index(example.get("answerKey"), labels=label_list, choices=choices)
    return {
        "id": str(example.get("id", "")),
        "task": "arc_easy",
        "prompt": f"Question: {question}\nAnswer:",
        "choices": choices,
        "answer": answer,
    }


TaskLoader = Callable[[Path, str], Iterable[dict[str, Any]]]
TaskConverter = Callable[[dict[str, Any]], dict[str, Any]]


def _load_hellaswag(cache_dir: Path, split: str):
    return _load_dataset("hellaswag", cache_dir=cache_dir, split=split)


def _load_piqa(cache_dir: Path, split: str):
    return _load_dataset("piqa", cache_dir=cache_dir, split=split)


def _load_arc_easy(cache_dir: Path, split: str):
    return _load_dataset("ai2_arc", "ARC-Easy", cache_dir=cache_dir, split=split)


TASKS: dict[str, tuple[TaskLoader, TaskConverter]] = {
    "hellaswag": (_load_hellaswag, _convert_hellaswag),
    "piqa": (_load_piqa, _convert_piqa),
    "arc_easy": (_load_arc_easy, _convert_arc_easy),
}


def _parse_tasks(value: str) -> list[str]:
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    if not tasks:
        raise ValueError("at least one task is required")
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        allowed = ", ".join(sorted(TASKS))
        raise ValueError(f"unsupported tasks: {', '.join(unknown)}. Allowed: {allowed}")
    return tasks


def _iter_examples(dataset: Iterable[dict[str, Any]], *, limit: int, shuffle: bool, seed: int) -> Iterable[dict[str, Any]]:
    if not shuffle:
        for idx, example in enumerate(dataset):
            if limit > 0 and idx >= limit:
                break
            yield example
        return

    items = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(items)
    for idx, example in enumerate(items):
        if limit > 0 and idx >= limit:
            break
        yield example


def _write_task(
    *,
    task: str,
    split: str,
    output_dir: Path,
    cache_dir: Path,
    limit: int,
    shuffle: bool,
    seed: int,
) -> dict[str, Any]:
    loader, converter = TASKS[task]
    dataset = loader(cache_dir, split)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task}_{split}.jsonl"
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for raw in _iter_examples(dataset, limit=limit, shuffle=shuffle, seed=seed):
            converted = converter(dict(raw))
            handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
    return {
        "task": task,
        "split": split,
        "path": str(output_path),
        "examples": count,
        "limit": limit,
        "shuffle": shuffle,
        "seed": seed,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare local multiple-choice downstream JSONL files.")
    parser.add_argument("--tasks", type=str, default="hellaswag,piqa,arc_easy")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=0, help="Maximum examples per task; 0 means full split.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before applying --limit.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    tasks = _parse_tasks(args.tasks)
    output_dir = Path(args.output_dir).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = [
        _write_task(
            task=task,
            split=args.split,
            output_dir=output_dir,
            cache_dir=cache_dir,
            limit=int(args.limit),
            shuffle=bool(args.shuffle),
            seed=int(args.seed),
        )
        for task in tasks
    ]
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for record in records:
        print(f"[prepare] {record['task']} {record['split']}: {record['examples']} examples -> {record['path']}")
    print(f"[prepare] wrote {manifest_path}")


if __name__ == "__main__":
    main()
