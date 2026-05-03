from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Sequence

_PAPER_TABLE_ARCHITECTURES = (
    ("mamba2", "Mamba-2"),
    ("mamba3", "Mamba-3"),
    ("transformer", "Transformer"),
    ("hybrid", "Hybrid"),
)
_PAPER_BACKBONE_LABELS = dict(_PAPER_TABLE_ARCHITECTURES)
_DEFAULT_REPORT_SEEDS = tuple(range(1000, 1020))
_DEFAULT_REPORT_BATCHES_PER_SEED = 5


def _load_grad_parity_module(repo_root: Path) -> ModuleType:
    module_path = repo_root / "tests" / "test_lbi_gradient_parity.py"
    spec = importlib.util.spec_from_file_location("grad_parity_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load grad parity module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(repo_root))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_markdown(path: Path, rows: Sequence[dict[str, object]]) -> None:
    headers = [
        "case_name",
        "backbone",
        "region_mode",
        "interface_jacobian_mode",
        "jacobian_basis_chunk",
        "device",
        "dtype",
        "status",
        "num_trials",
        "num_trials_passed",
        "worst_loss_abs_diff",
        "min_global_grad_cosine",
        "max_relative_l2_error",
        "worst_max_abs_grad_diff",
        "worst_max_rel_grad_diff",
        "median_p95_rel_grad_diff",
        "worst_interface_scan_rms",
        "skip_reason",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row.get(header, "")) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def _format_latex_number(value: object, *, precision: int = 3) -> str:
    if value in ("", None):
        return r"--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return r"--"
    if number == 0.0:
        return r"$0$"
    abs_number = abs(number)
    if 1e-3 <= abs_number < 1e3:
        return f"${number:.{precision}g}$"
    mantissa = number
    exponent = 0
    while abs(mantissa) >= 10.0:
        mantissa /= 10.0
        exponent += 1
    while abs(mantissa) < 1.0:
        mantissa *= 10.0
        exponent -= 1
    return rf"${mantissa:.{precision - 1}f}\times 10^{{{exponent}}}$"


def _paper_table_rows(case_summary_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    multi_region_by_backbone = {
        str(row.get("backbone", "")): row
        for row in case_summary_rows
        if row.get("region_mode") == "multi_region"
    }
    rows: list[dict[str, object]] = []
    for backbone, architecture in _PAPER_TABLE_ARCHITECTURES:
        source = multi_region_by_backbone.get(backbone, {})
        rows.append(
            {
                "architecture": architecture,
                "backbone": backbone,
                "case_name": source.get("case_name", ""),
                "status": source.get("status", "missing"),
                "num_trials": source.get("num_trials", ""),
                "num_trials_passed": source.get("num_trials_passed", ""),
                "max_abs_delta_loss": source.get("worst_loss_abs_diff", ""),
                "grad_linf": source.get("worst_max_abs_grad_diff", ""),
                "grad_relative_l2": source.get("max_relative_l2_error", ""),
                "grad_cosine": source.get("min_global_grad_cosine", ""),
                "skip_reason": source.get("skip_reason", ""),
            }
        )
    return rows


def _write_paper_table_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    _write_csv(path, rows)


def _write_paper_table_latex(path: Path, rows: Sequence[dict[str, object]]) -> None:
    num_trials = sorted({str(row.get("num_trials", "")) for row in rows if row.get("num_trials", "") not in ("", None)})
    trial_note = f" Each entry is the worst case over {num_trials[0]} trials." if len(num_trials) == 1 else ""
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\begin{tabular}{c|c|c|c|c}",
        r"Architecture & $\max |\Delta\mathcal{L}|$ & $\|\Delta g\|_\infty$ & $\|\Delta g\|_2 / \|g\|_2$ & $\cos \mathrm{sim}$ \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    str(row["architecture"]),
                    _format_latex_number(row["max_abs_delta_loss"]),
                    _format_latex_number(row["grad_linf"]),
                    _format_latex_number(row["grad_relative_l2"]),
                    _format_latex_number(row["grad_cosine"], precision=5),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\end{tabular}",
            r"\caption{",
            rf"Gradient parity over multi-region cases.{trial_note} $\Delta\mathcal{{L}} = \mathcal{{L}}_\mathrm{{BI}} - \mathcal{{L}}_\mathrm{{autograd}}$ and $\Delta g = \nabla^\mathrm{{BI}}_\theta - \nabla^\mathrm{{autograd}}_\theta$.",
            r"}",
            r"\label{tab:gradient-parity}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _maybe_write_cdf_plot(path: Path, aggregate_metrics_list: Sequence[object]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping grad parity CDF plot")
        return

    plt.figure(figsize=(8, 5))
    plotted = 0
    for aggregate in aggregate_metrics_list:
        if getattr(aggregate, "region_mode", "") != "multi_region":
            continue
        label = _PAPER_BACKBONE_LABELS.get(str(getattr(aggregate, "backbone", "")))
        if label is None:
            continue
        values = [
            max(float(parameter_metrics.max_rel_diff), 1e-12)
            for trial in aggregate.trial_metrics
            for parameter_metrics in trial.parameter_metrics
            if parameter_metrics.has_ref_grad and parameter_metrics.has_native_grad
        ]
        if not values:
            continue
        xs = sorted(values)
        ys = [(idx + 1) / len(xs) for idx in range(len(xs))]
        plt.plot(xs, ys, label=label)
        plotted += 1
    if plotted == 0:
        plt.close()
        print("no canonical multi-region grad parity cases available; skipping CDF plot")
        return
    plt.xscale("log")
    plt.xlabel("Per-parameter max relative gradient error")
    plt.ylabel("CDF")
    plt.title("Multi-Region Gradient Parity Error Distribution")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def generate_report(
    *,
    repo_root: Path,
    output_dir: Path,
    case_names: Sequence[str] | None,
    plot: bool,
    preset: str,
    seeds: Sequence[int] | None,
    batches_per_seed: int | None,
    interface_jacobian_mode: str,
    jacobian_basis_chunk: int,
) -> None:
    module = _load_grad_parity_module(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    if interface_jacobian_mode not in {"graph", "recompute"}:
        raise ValueError("interface_jacobian_mode must be one of: graph, recompute")
    if jacobian_basis_chunk <= 0:
        raise ValueError("jacobian_basis_chunk must be > 0")

    case_summary_rows: list[dict[str, object]] = []
    trial_summary_rows: list[dict[str, object]] = []
    per_parameter_rows: list[dict[str, object]] = []
    successful_aggregates: list[object] = []

    selected_seeds = tuple(seeds) if seeds is not None else (_DEFAULT_REPORT_SEEDS if preset == "report" else None)
    selected_batches_per_seed = (
        int(batches_per_seed)
        if batches_per_seed is not None
        else (_DEFAULT_REPORT_BATCHES_PER_SEED if preset == "report" else None)
    )
    if selected_batches_per_seed is not None and selected_batches_per_seed <= 0:
        raise ValueError("batches_per_seed must be > 0")

    for case in module.get_grad_parity_cases(case_names, preset=preset):
        if selected_seeds is not None or selected_batches_per_seed is not None:
            case = dataclasses.replace(
                case,
                seeds=selected_seeds if selected_seeds is not None else case.seeds,
                batches_per_seed=(
                    selected_batches_per_seed
                    if selected_batches_per_seed is not None
                    else case.batches_per_seed
                ),
            )
        skip_reason = module.grad_parity_skip_reason(case)
        if skip_reason is not None:
            case_summary_rows.append(
                {
                    "case_name": case.name,
                    "backbone": case.backbone_name,
                    "region_mode": case.region_mode,
                    "interface_jacobian_mode": interface_jacobian_mode,
                    "jacobian_basis_chunk": jacobian_basis_chunk,
                    "device": case.device_type,
                    "dtype": case.dtype_name,
                    "status": "skipped",
                    "skip_reason": skip_reason,
                    "num_trials": "",
                    "num_trials_passed": "",
                    "worst_loss_abs_diff": "",
                    "worst_max_abs_grad_diff": "",
                    "worst_max_rel_grad_diff": "",
                    "median_p95_rel_grad_diff": "",
                    "worst_interface_scan_rms": "",
                }
            )
            continue

        aggregate = module.compute_grad_parity_case_aggregate(
            case,
            interface_jacobian_mode=interface_jacobian_mode,
            jacobian_basis_chunk=jacobian_basis_chunk,
        )
        module.assert_grad_parity_aggregate(case, aggregate)
        successful_aggregates.append(aggregate)
        case_summary_rows.append(
            {
                **aggregate.to_summary_row(),
                "status": "ok",
                "skip_reason": "",
            }
        )
        for trial in aggregate.trial_metrics:
            trial_summary_rows.append(
                {
                    **trial.to_summary_row(),
                    "status": "ok" if trial.trial_passed else "failed",
                }
            )
            per_parameter_rows.extend(trial.per_parameter_rows())

    _write_csv(output_dir / "case_summary.csv", case_summary_rows)
    _write_csv(output_dir / "trial_summary.csv", trial_summary_rows)
    _write_csv(output_dir / "per_parameter.csv", per_parameter_rows)
    (output_dir / "case_summary.json").write_text(json.dumps(case_summary_rows, indent=2) + "\n")
    _write_summary_markdown(output_dir / "case_summary.md", case_summary_rows)
    (output_dir / "report_config.json").write_text(
        json.dumps(
            {
                "preset": preset,
                "interface_jacobian_mode": interface_jacobian_mode,
                "jacobian_basis_chunk": jacobian_basis_chunk,
                "seeds": list(selected_seeds) if selected_seeds is not None else None,
                "batches_per_seed": selected_batches_per_seed,
                "cases": list(case_names) if case_names is not None else None,
            },
            indent=2,
        )
        + "\n"
    )

    # Preserve the older summary filenames as aliases to the aggregate case view.
    _write_csv(output_dir / "summary.csv", case_summary_rows)
    (output_dir / "summary.json").write_text(json.dumps(case_summary_rows, indent=2) + "\n")
    _write_summary_markdown(output_dir / "summary.md", case_summary_rows)

    paper_rows = _paper_table_rows(case_summary_rows)
    _write_paper_table_csv(output_dir / "paper_table.csv", paper_rows)
    _write_paper_table_latex(output_dir / "paper_table.tex", paper_rows)

    if plot and successful_aggregates:
        _maybe_write_cdf_plot(output_dir / "grad_rel_error_cdf.png", successful_aggregates)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grad parity summary tables and optional appendix plots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/grad_parity"),
        help="Directory to write summary tables and optional plots into.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional subset of grad parity case names to run.",
    )
    parser.add_argument(
        "--preset",
        choices=("report", "test"),
        default="report",
        help="Case preset to run. `report` uses the stronger multi-seed defaults; `test` mirrors pytest.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate the appendix-style CDF plot if matplotlib is installed.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Override parity seeds. Defaults to 20 seeds for report preset and case defaults for test preset.",
    )
    parser.add_argument(
        "--batches-per-seed",
        type=int,
        default=None,
        help="Override batches per seed. Defaults to 5 for report preset and case defaults for test preset.",
    )
    parser.add_argument(
        "--interface-jacobian-mode",
        choices=("graph", "recompute"),
        default="graph",
        help="Interface Jacobian materialization path used by native backward.",
    )
    parser.add_argument(
        "--jacobian-basis-chunk",
        type=int,
        default=1,
        help="Basis chunk size for recompute-mode interface Jacobian materialization.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    generate_report(
        repo_root=repo_root,
        output_dir=args.output_dir,
        case_names=args.cases,
        plot=args.plot,
        preset=args.preset,
        seeds=args.seeds,
        batches_per_seed=args.batches_per_seed,
        interface_jacobian_mode=args.interface_jacobian_mode,
        jacobian_basis_chunk=args.jacobian_basis_chunk,
    )


if __name__ == "__main__":
    main()
