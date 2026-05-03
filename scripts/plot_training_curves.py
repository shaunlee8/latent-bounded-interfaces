#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


_MAIN_SPLITS = {"train", "val"}


def _set_paper_style(font_scale: float = 1.0) -> None:
    def scaled(value: float) -> float:
        return value * font_scale

    plt.rcParams.update(
        {
            "font.size": scaled(12),
            "axes.labelsize": scaled(13),
            "axes.titlesize": scaled(13),
            "xtick.labelsize": scaled(11),
            "ytick.labelsize": scaled(11),
            "legend.fontsize": scaled(10.5),
            "legend.title_fontsize": scaled(10.5),
            "lines.linewidth": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _save_figure(fig, output_path: Path, *, dpi: int, pdf: bool = False) -> None:
    fig.savefig(output_path, dpi=dpi)
    if pdf:
        fig.savefig(output_path.with_suffix(".pdf"))


@dataclass
class RunArtifact:
    experiment_root: Path
    run_dir: Path
    variant: str
    message_dim: int | None
    region_size: int | None
    lr_model: float | None
    backbone: str
    regime: str
    seed: int
    seq_len: int
    total_params: int
    summary: Dict[str, Any]
    metrics_rows: List[Dict[str, str]]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot aggregated LBI training curves from one or more run roots.")
    p.add_argument("run_roots", nargs="+", type=Path, help="Run roots or per-regime run directories.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write plots into. Defaults to <first_run_root>/plots.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI for saved figures.",
    )
    p.add_argument(
        "--label-mode",
        choices=("auto", "variant", "backbone", "regime", "backbone_regime", "region_size", "rank_region"),
        default="auto",
        help="How to group runs when aggregating across seeds.",
    )
    p.add_argument(
        "--regimes",
        type=str,
        default="",
        help="Optional comma-separated regime filter, e.g. native_region_interface.",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Rolling-mean window for paper-style train CE overlays. Use 1 to disable smoothing.",
    )
    p.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the default paper font sizes.",
    )
    p.add_argument(
        "--include-variant-regex",
        type=str,
        default="",
        help=(
            "Optional regex applied to the run variant directory name, e.g. "
            "'^(dense_seed[789]|lbi_r(16|32|64)_seed[789])$'."
        ),
    )
    p.add_argument(
        "--exclude-variant-regex",
        type=str,
        default="",
        help="Optional regex applied to the run variant directory name after include filtering.",
    )
    return p.parse_args()


def _load_metrics_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: str | float | int | None) -> float | None:
    if value == "" or value is None:
        return None
    x = float(value)
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _to_int(value: str | float | int | None) -> int | None:
    x = _to_float(value)
    return None if x is None else int(x)


def _discover_run_dirs(paths: Sequence[Path]) -> List[tuple[Path, Path]]:
    discovered: List[tuple[Path, Path]] = []
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"path does not exist: {path}")
        if (path / "metrics.csv").exists():
            discovered.append((path.parent, path))
            continue
        run_dirs = sorted(p for p in path.rglob("metrics.csv") if p.is_file())
        if not run_dirs:
            raise RuntimeError(f"no per-regime run directories with metrics.csv found under {path}")
        for metrics_path in run_dirs:
            run_dir = metrics_path.parent
            discovered.append((path, run_dir))
    return discovered


def _variant_from_run_dir(experiment_root: Path, run_dir: Path, config: Dict[str, Any], regime: str) -> str:
    try:
        rel = run_dir.relative_to(experiment_root)
    except ValueError:
        rel = Path(run_dir.name)
    parts = rel.parts
    variant = parts[0] if len(parts) > 1 else run_dir.parent.name
    if variant and variant != "." and variant != run_dir.name:
        return variant
    if regime == "backprop_ref":
        return f"dense_seed{config.get('seed', 'unknown')}"
    msg = config.get("message_dim", "unknown")
    return f"lbi_r{msg}_seed{config.get('seed', 'unknown')}"


def _load_run_artifacts(
    paths: Sequence[Path],
    *,
    regimes: set[str] | None,
    include_variant_regex: str = "",
    exclude_variant_regex: str = "",
) -> List[RunArtifact]:
    runs: List[RunArtifact] = []
    include_re = re.compile(include_variant_regex) if include_variant_regex else None
    exclude_re = re.compile(exclude_variant_regex) if exclude_variant_regex else None
    for experiment_root, run_dir in _discover_run_dirs(paths):
        config = _load_json(run_dir / "config.json")
        model_info = _load_json(run_dir / "model_info.json")
        summary = _load_json(run_dir / "summary.json")
        regime = str(summary.get("regime", run_dir.name))
        if regimes is not None and regime not in regimes:
            continue
        variant = _variant_from_run_dir(experiment_root, run_dir, config, regime)
        if include_re is not None and include_re.search(variant) is None:
            continue
        if exclude_re is not None and exclude_re.search(variant) is not None:
            continue
        message_dim = int(config["message_dim"]) if regime == "native_region_interface" and "message_dim" in config else None
        region_size = int(config["region_size"]) if regime == "native_region_interface" and "region_size" in config else None
        lr_model = float(config["lr_model"]) if "lr_model" in config else None
        runs.append(
            RunArtifact(
                experiment_root=experiment_root,
                run_dir=run_dir,
                variant=variant,
                message_dim=message_dim,
                region_size=region_size,
                lr_model=lr_model,
                backbone=str(config.get("backbone", "unknown")),
                regime=regime,
                seed=int(config.get("seed", -1)),
                seq_len=int(config.get("seq_len", 0)),
                total_params=int(model_info.get("total_params", 0)),
                summary=summary,
                metrics_rows=_load_metrics_csv(run_dir / "metrics.csv"),
            )
        )
    if not runs:
        raise RuntimeError("no matching runs found for the requested inputs")
    return runs


def _label_for_run(
    run: RunArtifact,
    *,
    label_mode: str,
    backbones: set[str],
    regimes: set[str],
    lbi_region_sizes: set[int],
) -> str:
    if label_mode == "backbone":
        return run.backbone
    if label_mode == "regime":
        return run.regime
    if label_mode == "backbone_regime":
        return f"{run.backbone}:{run.regime}"
    if label_mode == "region_size":
        if run.regime == "backprop_ref":
            return "dense"
        if run.region_size is not None:
            return f"region={run.region_size}"
        return run.variant
    if label_mode == "rank_region":
        if run.regime == "backprop_ref":
            return "dense"
        rank = f"r={run.message_dim}" if run.message_dim is not None else "r=?"
        if len(lbi_region_sizes) <= 1:
            return f"LBI {rank}"
        region = f"region={run.region_size}" if run.region_size is not None else "region=?"
        return f"LBI {rank}, {region}"
    if label_mode == "variant":
        if run.regime == "backprop_ref":
            return "dense"
        return f"LBI r={run.message_dim}"
    if run.regime == "backprop_ref":
        return "dense"
    if run.regime == "native_region_interface" and run.message_dim is not None:
        return f"LBI r={run.message_dim}"
    if len(backbones) > 1 and len(regimes) > 1:
        return f"{run.backbone}:{run.regime}"
    if len(backbones) > 1:
        return run.backbone
    if len(regimes) > 1:
        return run.regime
    return run.backbone


def _extract_series(rows: Sequence[Dict[str, str]], *, split: str, x_key: str, y_key: str) -> List[tuple[float, float]]:
    points: List[tuple[float, float]] = []
    for row in rows:
        if row.get("split") != split:
            continue
        x_val = _to_float(row.get(x_key))
        y_val = _to_float(row.get(y_key))
        if x_val is None or y_val is None:
            continue
        points.append((x_val, y_val))
    return points


def _aggregate_series(
    runs_by_label: Dict[str, List[RunArtifact]],
    *,
    split: str,
    x_key: str,
    y_key: str,
) -> Dict[str, List[Dict[str, float]]]:
    aggregated: Dict[str, List[Dict[str, float]]] = {}
    for label, runs in sorted(runs_by_label.items()):
        by_x: Dict[float, List[float]] = {}
        for run in runs:
            for x_val, y_val in _extract_series(run.metrics_rows, split=split, x_key=x_key, y_key=y_key):
                by_x.setdefault(x_val, []).append(y_val)
        rows = []
        for x_val in sorted(by_x.keys()):
            ys = by_x[x_val]
            rows.append(
                {
                    "x": x_val,
                    "mean": float(np.mean(ys)),
                    "std": float(np.std(ys)) if len(ys) > 1 else 0.0,
                    "count": float(len(ys)),
                }
            )
        aggregated[label] = rows
    return aggregated


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _paper_color(label: str, idx: int) -> str:
    normalized = label.lower().replace(" ", "")
    palette = {
        "dense": "#1f77b4",
        "lbir=8": "#17becf",
        "lbir=16": "#ff7f0e",
        "lbir=32": "#2ca02c",
        "lbir=64": "#d62728",
        "lbir=128": "#9467bd",
    }
    if normalized in palette:
        return palette[normalized]
    fallback = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]
    return fallback[idx % len(fallback)]


def _paper_backbone_title(backbones: set[str]) -> str:
    if len(backbones) != 1:
        return "FineWeb-Edu"
    backbone = next(iter(backbones))
    labels = {
        "mamba2": "Mamba-2",
        "mamba3": "Mamba-3",
        "transformer": "Transformer",
        "hybrid": "Hybrid",
    }
    return labels.get(backbone, backbone)


def _smooth_values(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    if window <= 1 or arr.size <= 2:
        return arr
    effective = min(window, arr.size)
    kernel = np.ones(effective, dtype=np.float64) / float(effective)
    padded = np.pad(arr, (effective - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _style_paper_axis(ax) -> None:
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.08, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _format_tokens_axis(ax) -> None:
    def formatter(value: float, _pos: int) -> str:
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.0f}M"
        if abs(value) >= 1_000:
            return f"{value / 1_000:.0f}k"
        return f"{value:.0f}"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter))


def _plot_ce(
    runs_by_label: Dict[str, List[RunArtifact]],
    *,
    output_path: Path,
    x_key: str,
    x_label: str,
    dpi: int,
) -> None:
    train = _aggregate_series(runs_by_label, split="train", x_key=x_key, y_key="ce_loss")
    val = _aggregate_series(runs_by_label, split="val", x_key=x_key, y_key="ce_loss")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.get_cmap("tab10")
    for idx, label in enumerate(sorted(runs_by_label.keys())):
        color = colors(idx % 10)
        train_rows = train.get(label, [])
        val_rows = val.get(label, [])
        if train_rows:
            xs = [row["x"] for row in train_rows]
            ys = [row["mean"] for row in train_rows]
            std = [row["std"] for row in train_rows]
            ax.plot(xs, ys, color=color, label=f"{label} train")
            ax.fill_between(xs, np.array(ys) - np.array(std), np.array(ys) + np.array(std), color=color, alpha=0.15)
        if val_rows:
            xs = [row["x"] for row in val_rows]
            ys = [row["mean"] for row in val_rows]
            std = [row["std"] for row in val_rows]
            ax.plot(xs, ys, linestyle="--", color=color, label=f"{label} val")
            ax.fill_between(xs, np.array(ys) - np.array(std), np.array(ys) + np.array(std), color=color, alpha=0.12)
    ax.set_title(f"Cross-Entropy vs {x_label}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cross-Entropy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def _plot_paper_val_ce(
    runs_by_label: Dict[str, List[RunArtifact]],
    *,
    output_path: Path,
    x_key: str,
    x_label: str,
    dpi: int,
    title_prefix: str,
) -> None:
    val = _aggregate_series(runs_by_label, split="val", x_key=x_key, y_key="ce_loss")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for idx, label in enumerate(sorted(runs_by_label.keys())):
        rows = val.get(label, [])
        if not rows:
            continue
        color = _paper_color(label, idx)
        xs = np.array([row["x"] for row in rows], dtype=np.float64)
        ys = np.array([row["mean"] for row in rows], dtype=np.float64)
        std = np.array([row["std"] for row in rows], dtype=np.float64)
        counts = np.array([row["count"] for row in rows], dtype=np.float64)
        ax.plot(xs, ys, color=color, linewidth=2.4, marker="o", markersize=3.2, markevery=max(1, len(xs) // 18), label=label)
        if np.any(counts > 1):
            ax.fill_between(xs, ys - std, ys + std, color=color, alpha=0.12, linewidth=0)
    ax.set_title(f"{title_prefix} Online Validation CE")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cross-Entropy")
    if x_key == "tokens_seen":
        _format_tokens_axis(ax)
    _style_paper_axis(ax)
    ax.legend(frameon=False, ncol=2 if len(runs_by_label) > 3 else 1)
    fig.tight_layout()
    _save_figure(fig, output_path, dpi=dpi, pdf=True)
    plt.close(fig)


def _plot_paper_ce_with_train_overlay(
    runs_by_label: Dict[str, List[RunArtifact]],
    *,
    output_path: Path,
    x_key: str,
    x_label: str,
    dpi: int,
    smooth_window: int,
    title_prefix: str,
) -> None:
    train = _aggregate_series(runs_by_label, split="train", x_key=x_key, y_key="ce_loss")
    val = _aggregate_series(runs_by_label, split="val", x_key=x_key, y_key="ce_loss")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for idx, label in enumerate(sorted(runs_by_label.keys())):
        color = _paper_color(label, idx)
        train_rows = train.get(label, [])
        val_rows = val.get(label, [])
        if train_rows:
            xs = np.array([row["x"] for row in train_rows], dtype=np.float64)
            ys = _smooth_values([row["mean"] for row in train_rows], smooth_window)
            ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.9, zorder=3, label=label)
        if val_rows:
            xs = np.array([row["x"] for row in val_rows], dtype=np.float64)
            ys = np.array([row["mean"] for row in val_rows], dtype=np.float64)
            std = np.array([row["std"] for row in val_rows], dtype=np.float64)
            counts = np.array([row["count"] for row in val_rows], dtype=np.float64)
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=":",
                linewidth=1.25,
                alpha=0.75,
                marker="o",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                markersize=4.2,
                markevery=max(1, len(xs) // 14),
                zorder=4,
            )
            if np.any(counts > 1):
                ax.fill_between(xs, ys - std, ys + std, color=color, alpha=0.035, linewidth=0)
                ax.plot(xs, ys + std, color=color, linestyle=":", linewidth=0.7, alpha=0.26, zorder=2)
                ax.plot(xs, ys - std, color=color, linestyle=":", linewidth=0.7, alpha=0.26, zorder=2)
    ax.set_title(f"{title_prefix} Train and Online Validation CE")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cross-Entropy")
    if x_key == "tokens_seen":
        _format_tokens_axis(ax)
    _style_paper_axis(ax)
    method_legend = ax.legend(frameon=False, ncol=2 if len(runs_by_label) > 3 else 1, loc="upper right")
    style_handles = [
        Line2D([0], [0], color="#444444", linewidth=2.45, alpha=0.9, label="train"),
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=":",
            linewidth=1.25,
            alpha=0.75,
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#444444",
            markeredgewidth=1.1,
            markersize=4.2,
            label="online val",
        ),
    ]
    ax.add_artist(method_legend)
    ax.legend(handles=style_handles, frameon=False, loc="lower left")
    fig.tight_layout()
    _save_figure(fig, output_path, dpi=dpi, pdf=True)
    plt.close(fig)


def _plot_train_metric(
    runs_by_label: Dict[str, List[RunArtifact]],
    *,
    y_key: str,
    title: str,
    y_label: str,
    output_path: Path,
    dpi: int,
) -> None:
    aggregated = _aggregate_series(runs_by_label, split="train", x_key="step", y_key=y_key)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.get_cmap("tab10")
    plotted = False
    for idx, label in enumerate(sorted(runs_by_label.keys())):
        rows = aggregated.get(label, [])
        if not rows:
            continue
        plotted = True
        xs = [row["x"] for row in rows]
        ys = [row["mean"] for row in rows]
        std = [row["std"] for row in rows]
        color = colors(idx % 10)
        ax.plot(xs, ys, color=color, label=label)
        ax.fill_between(xs, np.array(ys) - np.array(std), np.array(ys) + np.array(std), color=color, alpha=0.15)
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def _plot_message_diagnostics(
    runs_by_label: Dict[str, List[RunArtifact]],
    *,
    output_path: Path,
    dpi: int,
) -> None:
    msg = _aggregate_series(runs_by_label, split="train", x_key="step", y_key="message_norm")
    scan = _aggregate_series(runs_by_label, split="train", x_key="step", y_key="scan_align")
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    colors = plt.get_cmap("tab10")
    plotted = False
    for idx, label in enumerate(sorted(runs_by_label.keys())):
        msg_rows = msg.get(label, [])
        scan_rows = scan.get(label, [])
        if not msg_rows and not scan_rows:
            continue
        plotted = True
        color = colors(idx % 10)
        if msg_rows:
            xs = [row["x"] for row in msg_rows]
            ys = [row["mean"] for row in msg_rows]
            std = [row["std"] for row in msg_rows]
            axes[0].plot(xs, ys, color=color, label=label)
            axes[0].fill_between(xs, np.array(ys) - np.array(std), np.array(ys) + np.array(std), color=color, alpha=0.15)
        if scan_rows:
            xs = [row["x"] for row in scan_rows]
            ys = [row["mean"] for row in scan_rows]
            std = [row["std"] for row in scan_rows]
            axes[1].plot(xs, ys, color=color, label=label)
            axes[1].fill_between(xs, np.array(ys) - np.array(std), np.array(ys) + np.array(std), color=color, alpha=0.15)
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_title("Message Norm vs Step")
    axes[0].set_ylabel("Message Norm")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_title("Scan Alignment vs Step")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Scan Align")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    _save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def _aggregate_run_summary_rows(runs_by_label: Dict[str, List[RunArtifact]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, runs in sorted(runs_by_label.items()):
        final_vals = [float(run.summary["final_val_ce_loss"]) for run in runs]
        best_vals = [float(run.summary["best_val_ce_loss"]) for run in runs]
        params = [int(run.total_params) for run in runs]
        seq_lens = sorted({run.seq_len for run in runs})
        rows.append(
            {
                "label": label,
                "num_runs": len(runs),
                "backbones": ",".join(sorted({run.backbone for run in runs})),
                "regimes": ",".join(sorted({run.regime for run in runs})),
                "seeds": ",".join(str(run.seed) for run in sorted(runs, key=lambda item: item.seed)),
                "variants": ",".join(run.variant for run in sorted(runs, key=lambda item: (item.variant, item.seed))),
                "region_sizes": ",".join(
                    str(v) for v in sorted({run.region_size for run in runs if run.region_size is not None})
                ),
                "lr_models": ",".join(
                    f"{v:g}" for v in sorted({run.lr_model for run in runs if run.lr_model is not None})
                ),
                "seq_lens": ",".join(str(v) for v in seq_lens),
                "mean_total_params": float(np.mean(params)),
                "std_total_params": float(np.std(params)) if len(params) > 1 else 0.0,
                "mean_final_val_ce": float(np.mean(final_vals)),
                "std_final_val_ce": float(np.std(final_vals)) if len(final_vals) > 1 else 0.0,
                "mean_best_val_ce": float(np.mean(best_vals)),
                "std_best_val_ce": float(np.std(best_vals)) if len(best_vals) > 1 else 0.0,
            }
        )
    return rows


def _plot_val_summary(summary_rows: Sequence[Dict[str, Any]], *, output_path: Path, dpi: int) -> None:
    if not summary_rows:
        return
    labels = [row["label"] for row in summary_rows]
    final_means = np.array([row["mean_final_val_ce"] for row in summary_rows], dtype=np.float64)
    final_std = np.array([row["std_final_val_ce"] for row in summary_rows], dtype=np.float64)
    best_means = np.array([row["mean_best_val_ce"] for row in summary_rows], dtype=np.float64)
    best_std = np.array([row["std_best_val_ce"] for row in summary_rows], dtype=np.float64)
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(labels)), 4.8))
    ax.bar(x - (width / 2.0), final_means, width=width, yerr=final_std, label="final val", capsize=4)
    ax.bar(x + (width / 2.0), best_means, width=width, yerr=best_std, label="best val", capsize=4)
    ax.set_title("Validation Cross-Entropy Summary")
    ax.set_ylabel("Cross-Entropy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def generate_training_plots(
    run_roots: Sequence[Path],
    *,
    output_dir: Path | None = None,
    dpi: int = 150,
    label_mode: str = "auto",
    regimes: Iterable[str] | None = None,
    smooth_window: int = 25,
    font_scale: float = 1.0,
    include_variant_regex: str = "",
    exclude_variant_regex: str = "",
) -> Path:
    _set_paper_style(font_scale=font_scale)
    regime_filter = None if regimes is None else {regime for regime in regimes if regime}
    runs = _load_run_artifacts(
        run_roots,
        regimes=regime_filter,
        include_variant_regex=include_variant_regex,
        exclude_variant_regex=exclude_variant_regex,
    )
    backbones = {run.backbone for run in runs}
    title_prefix = _paper_backbone_title(backbones)
    regimes_present = {run.regime for run in runs}
    lbi_region_sizes = {run.region_size for run in runs if run.regime == "native_region_interface" and run.region_size is not None}
    runs_by_label: Dict[str, List[RunArtifact]] = {}
    for run in runs:
        label = _label_for_run(
            run,
            label_mode=label_mode,
            backbones=backbones,
            regimes=regimes_present,
            lbi_region_sizes=lbi_region_sizes,
        )
        runs_by_label.setdefault(label, []).append(run)

    base_root = run_roots[0].resolve()
    default_dir = base_root / "plots" if base_root.is_dir() else base_root.parent / "plots"
    out_dir = (output_dir or default_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_ce(runs_by_label, output_path=out_dir / "ce_vs_tokens.png", x_key="tokens_seen", x_label="Tokens Seen", dpi=dpi)
    _plot_ce(runs_by_label, output_path=out_dir / "ce_vs_step.png", x_key="step", x_label="Step", dpi=dpi)
    _plot_paper_val_ce(
        runs_by_label,
        output_path=out_dir / "paper_val_ce_vs_tokens.png",
        x_key="tokens_seen",
        x_label="Tokens Seen",
        dpi=dpi,
        title_prefix=title_prefix,
    )
    _plot_paper_val_ce(
        runs_by_label,
        output_path=out_dir / "paper_val_ce_vs_step.png",
        x_key="step",
        x_label="Step",
        dpi=dpi,
        title_prefix=title_prefix,
    )
    _plot_paper_ce_with_train_overlay(
        runs_by_label,
        output_path=out_dir / "paper_ce_vs_tokens.png",
        x_key="tokens_seen",
        x_label="Tokens Seen",
        dpi=dpi,
        smooth_window=smooth_window,
        title_prefix=title_prefix,
    )
    _plot_train_metric(
        runs_by_label,
        y_key="tokens_per_s",
        title="Tokens/s vs Step",
        y_label="Tokens/s",
        output_path=out_dir / "tokens_per_s_vs_step.png",
        dpi=dpi,
    )
    _plot_message_diagnostics(runs_by_label, output_path=out_dir / "message_diagnostics.png", dpi=dpi)
    summary_rows = _aggregate_run_summary_rows(runs_by_label)
    _write_csv(out_dir / "group_summary.csv", summary_rows)
    (out_dir / "group_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
    _plot_val_summary(summary_rows, output_path=out_dir / "val_summary_bar.png", dpi=dpi)
    return out_dir


def main() -> None:
    args = _parse_args()
    regimes = [part.strip() for part in args.regimes.split(",") if part.strip()]
    out_dir = generate_training_plots(
        args.run_roots,
        output_dir=args.output_dir,
        dpi=args.dpi,
        label_mode=args.label_mode,
        regimes=regimes or None,
        smooth_window=args.smooth_window,
        font_scale=args.font_scale,
        include_variant_regex=args.include_variant_regex,
        exclude_variant_regex=args.exclude_variant_regex,
    )
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
