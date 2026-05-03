#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


_RUN_RE = re.compile(r"lbi_r(?P<rank>\d+)_jac_seed(?P<seed>\d+)")
_SUMMARY_FIELDS = (
    "jac_local_spec_mean",
    "jac_suffix_spec_mean",
    "jac_local_frob_normed_mean",
)


@dataclass
class JacobianRun:
    rank: int
    seed: int
    run_dir: Path
    rows: List[Dict[str, str]]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot appendix Jacobian diagnostics for LBI runs.")
    p.add_argument("run_root", type=Path, help="Jacobian diagnostic family directory.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write plots and tables. Defaults to <run_root>/jacobian_plots.",
    )
    p.add_argument("--min-step", type=int, default=200, help="Minimum step to include in plots/tables.")
    p.add_argument("--dpi", type=int, default=150, help="Output DPI for PNG figures.")
    p.add_argument("--font-scale", type=float, default=1.0, help="Multiplier for paper font sizes.")
    p.add_argument("--include-rank-regex", type=str, default="", help="Optional regex over rank strings, e.g. '^(16|32)$'.")
    return p.parse_args()


def _set_paper_style(font_scale: float) -> None:
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
            "lines.linewidth": 2.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _load_metrics_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str | float | int | None) -> float | None:
    if value == "" or value is None:
        return None
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


def _discover_runs(root: Path, *, include_rank_regex: str = "") -> List[JacobianRun]:
    if not root.exists():
        raise FileNotFoundError(f"run root does not exist: {root}")
    rank_re = re.compile(include_rank_regex) if include_rank_regex else None
    runs: List[JacobianRun] = []
    for metrics_path in sorted(root.glob("lbi_r*_jac_seed*/native_region_interface/metrics.csv")):
        match = _RUN_RE.search(str(metrics_path))
        if match is None:
            continue
        rank = int(match.group("rank"))
        seed = int(match.group("seed"))
        if rank_re is not None and rank_re.search(str(rank)) is None:
            continue
        rows = _load_metrics_csv(metrics_path)
        if not any(row.get("jac_local_spec_mean") for row in rows):
            continue
        runs.append(JacobianRun(rank=rank, seed=seed, run_dir=metrics_path.parent, rows=rows))
    if not runs:
        raise RuntimeError(f"no Jacobian diagnostic metrics found under {root}")
    return runs


def _extract_series(run: JacobianRun, *, field: str, min_step: int) -> List[tuple[float, float]]:
    series: List[tuple[float, float]] = []
    for row in run.rows:
        if row.get("split") != "train":
            continue
        step = _to_float(row.get("step"))
        x = _to_float(row.get("tokens_seen"))
        y = _to_float(row.get(field))
        if step is None or x is None or y is None:
            continue
        if step < min_step:
            continue
        series.append((x, y))
    return series


def _aggregate_by_rank(
    runs: Sequence[JacobianRun],
    *,
    field: str,
    min_step: int,
) -> Dict[int, List[Dict[str, float]]]:
    by_rank: Dict[int, Dict[float, List[float]]] = {}
    for run in runs:
        rank_rows = by_rank.setdefault(run.rank, {})
        for x, y in _extract_series(run, field=field, min_step=min_step):
            rank_rows.setdefault(x, []).append(y)

    out: Dict[int, List[Dict[str, float]]] = {}
    for rank, by_x in sorted(by_rank.items()):
        out[rank] = [
            {
                "x": x,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)) if len(values) > 1 else 0.0,
                "count": float(len(values)),
            }
            for x, values in sorted(by_x.items())
        ]
    return out


def _paper_color(rank: int, idx: int) -> str:
    palette = {
        16: "#ff7f0e",
        32: "#2ca02c",
        64: "#d62728",
        128: "#9467bd",
    }
    fallback = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]
    return palette.get(rank, fallback[idx % len(fallback)])


def _format_tokens_axis(ax) -> None:
    def formatter(value: float, _pos: int) -> str:
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.0f}M"
        if abs(value) >= 1_000:
            return f"{value / 1_000:.0f}k"
        return f"{value:.0f}"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _style_axis(ax) -> None:
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.08, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_spec_diagnostics(
    runs: Sequence[JacobianRun],
    *,
    output_path: Path,
    min_step: int,
    dpi: int,
) -> None:
    fields = [
        ("jac_local_spec_mean", "Local spectral norm"),
        ("jac_suffix_spec_mean", "Suffix spectral norm"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=True)
    for ax, (field, title) in zip(axes, fields):
        aggregated = _aggregate_by_rank(runs, field=field, min_step=min_step)
        for idx, (rank, rows) in enumerate(sorted(aggregated.items())):
            if not rows:
                continue
            color = _paper_color(rank, idx)
            xs = np.array([row["x"] for row in rows], dtype=np.float64)
            ys = np.array([row["mean"] for row in rows], dtype=np.float64)
            std = np.array([row["std"] for row in rows], dtype=np.float64)
            ax.plot(xs, ys, color=color, linewidth=2.35, label=f"r={rank}")
            if np.any(std > 0):
                ax.fill_between(xs, ys - std, ys + std, color=color, alpha=0.10, linewidth=0)
                ax.plot(xs, ys + std, color=color, linewidth=0.7, alpha=0.25)
                ax.plot(xs, ys - std, color=color, linewidth=0.7, alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("Tokens Seen")
        ax.set_ylabel("Spectral Norm")
        _format_tokens_axis(ax)
        _style_axis(ax)
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _final_row_for_field(run: JacobianRun, *, field: str, min_step: int) -> float | None:
    values = [
        _to_float(row.get(field))
        for row in run.rows
        if row.get("split") == "train" and _to_float(row.get("step")) is not None and float(row["step"]) >= min_step
    ]
    values = [value for value in values if value is not None]
    return values[-1] if values else None


def _write_final_summary(runs: Sequence[JacobianRun], *, output_dir: Path, min_step: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ranks = sorted({run.rank for run in runs})
    for rank in ranks:
        rank_runs = sorted([run for run in runs if run.rank == rank], key=lambda item: item.seed)
        row: Dict[str, Any] = {
            "rank": rank,
            "num_seeds": len(rank_runs),
            "seeds": ",".join(str(run.seed) for run in rank_runs),
        }
        for field in _SUMMARY_FIELDS:
            values = [_final_row_for_field(run, field=field, min_step=min_step) for run in rank_runs]
            values = [value for value in values if value is not None]
            row[f"{field}_mean"] = float(np.mean(values)) if values else ""
            row[f"{field}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else ""
        rows.append(row)

    csv_path = output_dir / "jacobian_final_summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tex_path = output_dir / "jacobian_final_summary.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{c|c|c|c}\n")
        f.write("Rank $r$ & Local $\\|J\\|_2$ & Suffix $\\|J\\|_2$ & Local $\\|J\\|_F/\\sqrt{r}$ \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(
                f"{row['rank']} & "
                f"{_format_pm(row['jac_local_spec_mean_mean'], row['jac_local_spec_mean_std'])} & "
                f"{_format_pm(row['jac_suffix_spec_mean_mean'], row['jac_suffix_spec_mean_std'])} & "
                f"{_format_pm(row['jac_local_frob_normed_mean_mean'], row['jac_local_frob_normed_mean_std'])} \\\\\n"
            )
        f.write("\\end{tabular}\n")
    return rows


def _format_pm(mean: Any, std: Any) -> str:
    if mean == "" or std == "":
        return "--"
    return f"{float(mean):.3f} $\\pm$ {float(std):.3f}"


def generate_jacobian_diagnostic_plots(
    run_root: Path,
    *,
    output_dir: Path | None = None,
    min_step: int = 200,
    dpi: int = 150,
    font_scale: float = 1.0,
    include_rank_regex: str = "",
) -> Path:
    _set_paper_style(font_scale)
    runs = _discover_runs(run_root.resolve(), include_rank_regex=include_rank_regex)
    out_dir = (output_dir or run_root / "jacobian_plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_spec_diagnostics(runs, output_path=out_dir / "jacobian_spec_mean_vs_tokens.png", min_step=min_step, dpi=dpi)
    _write_final_summary(runs, output_dir=out_dir, min_step=min_step)
    return out_dir


def main() -> None:
    args = _parse_args()
    out_dir = generate_jacobian_diagnostic_plots(
        args.run_root,
        output_dir=args.output_dir,
        min_step=args.min_step,
        dpi=args.dpi,
        font_scale=args.font_scale,
        include_rank_regex=args.include_rank_regex,
    )
    print(f"wrote Jacobian diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
