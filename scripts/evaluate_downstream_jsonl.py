from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data.bpe_tokenizer import load_text_tokenizer
from scripts.evaluate_paper_checkpoints import (
    _build_model,
    _checkpoint_from_summary,
    _discover_run_dirs,
    _load_config,
    _run_label,
)
from train.checkpointing import load_checkpoint as _load_checkpoint
from train.config import LBI_VARIANT, normalize_model_variant
from train.data import resolve_text_paths as _resolve_text_paths, resolve_tokenizer_path as _resolve_tokenizer_path
from train.lbi import _autocast_context, _resolve_device


def _load_examples(path: Path, *, limit: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if limit > 0 and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"no examples loaded from {path}")
    return examples


def _choice_texts(example: dict[str, Any]) -> list[str]:
    choices = example.get("choices", example.get("endings", example.get("options")))
    if not isinstance(choices, list) or not choices:
        raise ValueError("each example must contain a non-empty choices/options/endings list")
    out: list[str] = []
    for choice in choices:
        if isinstance(choice, str):
            out.append(choice)
        elif isinstance(choice, dict):
            text = choice.get("text", choice.get("value", choice.get("choice")))
            if not isinstance(text, str):
                raise ValueError(f"unsupported choice object: {choice}")
            out.append(text)
        else:
            raise ValueError(f"unsupported choice type: {type(choice).__name__}")
    return out


def _gold_index(example: dict[str, Any], choices: list[str]) -> int:
    raw = example.get("answer", example.get("label", example.get("gold", example.get("target"))))
    if raw is None:
        raise ValueError("each example must contain answer/label/gold/target")
    if isinstance(raw, int):
        gold = raw
    elif isinstance(raw, str):
        stripped = raw.strip()
        if stripped.isdigit():
            gold = int(stripped)
        elif len(stripped) == 1 and "A" <= stripped.upper() <= "Z":
            gold = ord(stripped.upper()) - ord("A")
        elif stripped in choices:
            gold = choices.index(stripped)
        else:
            raise ValueError(f"could not map string answer to a choice: {raw!r}")
    else:
        raise ValueError(f"unsupported answer type: {type(raw).__name__}")
    if gold < 0 or gold >= len(choices):
        raise ValueError(f"gold index {gold} out of range for {len(choices)} choices")
    return gold


def _prompt_text(example: dict[str, Any]) -> str:
    prompt = example.get("prompt", example.get("ctx", example.get("query", example.get("question"))))
    if not isinstance(prompt, str):
        raise ValueError("each example must contain a string prompt/ctx/query/question field")
    return prompt


def _load_tokenizer_for_cfg(cfg: Any):
    if cfg.data_mode == "text_byte":
        return None
    if cfg.data_mode not in {"text_bpe", "text_bpe_sharded"}:
        raise ValueError("downstream JSONL eval requires a text tokenizer config")
    train_path, _, _ = _resolve_text_paths(cfg)
    tokenizer_path = _resolve_tokenizer_path(cfg, train_path=train_path)
    return load_text_tokenizer(
        tokenizer_type=cfg.tokenizer_type,
        tokenizer_path=tokenizer_path,
        train_path=train_path,
        vocab_size=cfg.vocab_size,
        train_bytes=cfg.tokenizer_train_bytes,
        force_train=False,
    )


def _encode(text: str, tokenizer: Any | None) -> list[int]:
    if tokenizer is None:
        return list(text.encode("utf-8"))
    return list(tokenizer.encode_bytes(text.encode("utf-8")))


def _candidate_score(
    *,
    cfg: Any,
    model: torch.nn.Module,
    tokenizer: Any | None,
    prompt: str,
    choice: str,
    device: torch.device,
) -> tuple[float, float, int]:
    prompt_ids = _encode(prompt, tokenizer)
    choice_ids = _encode(choice, tokenizer)
    if not prompt_ids:
        raise ValueError("prompt tokenization produced no tokens; prepend context for causal scoring")
    if not choice_ids:
        raise ValueError("choice tokenization produced no tokens")

    full = prompt_ids + choice_ids
    if len(full) < 2:
        raise ValueError("prompt + choice must contain at least two tokens")

    offset = max(0, len(full) - (cfg.seq_len + 1))
    window = full[offset:]
    x = torch.tensor(window[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(window[1:], dtype=torch.long, device=device).unsqueeze(0)

    target_global = torch.arange(offset + 1, offset + 1 + y.numel(), device=device)
    choice_start = len(prompt_ids)
    mask = target_global >= choice_start
    if not bool(mask.any().item()):
        raise ValueError("no choice tokens fit in the scoring window")

    with torch.no_grad():
        with _autocast_context(cfg, device):
            if hasattr(model, "forward_with_cache"):
                logits, _ = model.forward_with_cache(x)
            else:
                logits = model(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        token_log_probs = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1).squeeze(0)
        selected = token_log_probs[mask]
    total = float(selected.sum().item())
    mean = float(selected.mean().item())
    return total, mean, int(selected.numel())


def _evaluate_examples(
    *,
    cfg: Any,
    model: torch.nn.Module,
    tokenizer: Any | None,
    examples: list[dict[str, Any]],
    device: torch.device,
    score_normalization: str,
) -> dict[str, Any]:
    correct = 0
    gold_logprob_sum = 0.0
    gold_token_count = 0
    start = time.perf_counter()
    for example in examples:
        prompt = _prompt_text(example)
        choices = _choice_texts(example)
        gold = _gold_index(example, choices)
        choice_scores: list[float] = []
        choice_totals: list[float] = []
        choice_token_counts: list[int] = []
        for choice in choices:
            total, mean, token_count = _candidate_score(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                choice=choice,
                device=device,
            )
            choice_scores.append(mean if score_normalization == "mean" else total)
            choice_totals.append(total)
            choice_token_counts.append(token_count)
        pred = max(range(len(choice_scores)), key=choice_scores.__getitem__)
        correct += int(pred == gold)
        gold_logprob_sum += choice_totals[gold]
        gold_token_count += choice_token_counts[gold]
    wall = time.perf_counter() - start
    n = len(examples)
    gold_mean_logprob = gold_logprob_sum / max(1, gold_token_count)
    return {
        "examples": n,
        "accuracy": correct / n,
        "correct": correct,
        "gold_mean_logprob": gold_mean_logprob,
        "gold_ppl": float(math.exp(-gold_mean_logprob)),
        "scored_gold_tokens": gold_token_count,
        "wall_time_s": wall,
        "examples_per_s": n / wall if wall > 0.0 else 0.0,
    }


def _evaluate_run(run_dir: Path, *, args: argparse.Namespace, examples: list[dict[str, Any]]) -> dict[str, Any]:
    cfg_args = argparse.Namespace(eval_batches=1, batch_size=1, seq_len=args.seq_len,
                                  device=args.device, tokenizer_path=args.tokenizer_path)
    cfg = _load_config(run_dir, args=cfg_args)
    device = _resolve_device(cfg)
    checkpoint_path = _checkpoint_from_summary(run_dir, args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path, device=device)
    model = _build_model(cfg, checkpoint=checkpoint).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = _load_tokenizer_for_cfg(cfg)
    metrics = _evaluate_examples(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=device,
        score_normalization=args.score_normalization,
    )
    row = {
        "label": _run_label(run_dir, cfg),
        "run_dir": str(run_dir),
        "regime": cfg.regime,
        "backbone": cfg.backbone,
        "seed": cfg.seed,
        "message_dim": cfg.message_dim if normalize_model_variant(cfg.regime) == LBI_VARIANT else "",
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "task_path": str(Path(args.task_jsonl).expanduser()),
        "score_normalization": args.score_normalization,
        **metrics,
    }
    del model
    del checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "downstream_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows:
        with (output_dir / "downstream_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("run\taccuracy\tgold_ppl\texamples\tckpt_step")
    for row in rows:
        print(
            f"{row['label']}\t{row['accuracy']:.6g}\t{row['gold_ppl']:.6g}\t"
            f"{row['examples']}\t{row['checkpoint_step']}"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved paper checkpoints on a local multiple-choice JSONL task.")
    parser.add_argument("--family", type=str, required=True)
    parser.add_argument("--task-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--tokenizer-path", type=str, default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--score-normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--run-dir", action="append", default=[])
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    family = Path(args.family).expanduser().resolve()
    task_path = Path(args.task_jsonl).expanduser().resolve()
    if args.run_dir:
        run_dirs = [Path(item).expanduser().resolve() for item in args.run_dir]
    elif (family / "config.json").exists() and (family / "summary.json").exists():
        run_dirs = [family]
    else:
        run_dirs = _discover_run_dirs(family)
    if not run_dirs:
        raise FileNotFoundError(f"no completed paper runs found under: {family}")
    examples = _load_examples(task_path, limit=args.limit)
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else Path("out/evals/region_interface") / family.name / task_path.stem
    )
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        print(f"[downstream] {run_dir}", flush=True)
        rows.append(_evaluate_run(run_dir, args=args, examples=examples))
    _write_outputs(rows, output_dir)
    _print_table(rows)
    print(f"[downstream] wrote {output_dir / 'downstream_summary.csv'}")
    print(f"[downstream] wrote {output_dir / 'downstream_summary.json'}")


if __name__ == "__main__":
    main()
