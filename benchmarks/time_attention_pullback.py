from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from cuda.transformer import attention_input_pullback_basis

OUT_DIR = REPO_ROOT / "out" / "benchmarks" / "attention_pullback"


def _parse_int_list(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def _attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float, causal: bool) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
    )


def _autograd_pullback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_cotangent_basis: torch.Tensor,
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # This is a semantic baseline for J_attn^T dO_basis. It performs one
    # PyTorch backward per basis vector, so it measures framework/autograd
    # overhead as well as the underlying attention backward implementation.
    dq_cols: list[torch.Tensor] = []
    dk_cols: list[torch.Tensor] = []
    dv_cols: list[torch.Tensor] = []
    for basis_index in range(output_cotangent_basis.shape[1]):
        q_req = q.detach().clone().requires_grad_(True)
        k_req = k.detach().clone().requires_grad_(True)
        v_req = v.detach().clone().requires_grad_(True)
        out = _attention_forward(q_req, k_req, v_req, scale=scale, causal=causal)
        dq, dk, dv = torch.autograd.grad(
            out,
            (q_req, k_req, v_req),
            grad_outputs=output_cotangent_basis[:, basis_index],
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        dq_cols.append(dq.unsqueeze(1))
        dk_cols.append(dk.unsqueeze(1))
        dv_cols.append(dv.unsqueeze(1))
    return torch.cat(dq_cols, dim=1), torch.cat(dk_cols, dim=1), torch.cat(dv_cols, dim=1)


def _torch_formula_pullback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_cotangent_basis: torch.Tensor,
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return attention_input_pullback_basis(
        q,
        k,
        v,
        output_cotangent_basis,
        softmax_scale=scale,
        causal=causal,
        use_cuda=False,
    )


def _cuda_pullback(
    mode: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_cotangent_basis: torch.Tensor,
    *,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return attention_input_pullback_basis(
        q,
        k,
        v,
        output_cotangent_basis,
        softmax_scale=scale,
        causal=causal,
        use_cuda=True,
        kernel_mode=mode,
    )


def _time_cuda(fn: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]], *, warmup: int, iters: int) -> tuple[float, float, int]:
    for _ in range(warmup):
        out = fn()
        for tensor in out:
            tensor.record_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()

    times_ms: list[float] = []
    peak_bytes = 0
    for _ in range(iters):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        for tensor in out:
            tensor.record_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
        peak_bytes = max(peak_bytes, int(torch.cuda.max_memory_allocated()))
    return statistics.median(times_ms), min(times_ms), peak_bytes


def _max_abs_error(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> float:
    return max(float((a - e).abs().max().item()) for a, e in zip(actual, expected))


def _run_backend(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output_cotangent_basis: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    warmup: int,
    iters: int,
) -> tuple[float, float, int]:
    if backend in {"simple", "fa2_loop", "fa2_p1", "fa2_mma_p1", "fa2_mma_pblock4"}:
        fn = lambda: _cuda_pullback(backend, q, k, v, output_cotangent_basis, scale=scale, causal=causal)
    elif backend == "torch_formula":
        fn = lambda: _torch_formula_pullback(q, k, v, output_cotangent_basis, scale=scale, causal=causal)
    elif backend == "autograd":
        fn = lambda: _autograd_pullback(q, k, v, output_cotangent_basis, scale=scale, causal=causal)
    else:
        raise ValueError(f"unknown backend: {backend}")
    return _time_cuda(fn, warmup=warmup, iters=iters)


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[int, int, int, int, int, bool], dict[str, float]] = {}
    for row in rows:
        key = (
            int(row["B"]),
            int(row["H"]),
            int(row["T"]),
            int(row["Dh"]),
            int(row["P"]),
            bool(row["causal"]),
        )
        grouped.setdefault(key, {})[str(row["backend"])] = float(row["median_ms"])

    comparisons: list[dict[str, object]] = []
    preferred_primary = ("fa2_mma_pblock4", "fa2_mma_p1", "fa2_p1", "fa2_loop", "autograd", "torch_formula", "simple")
    for (bsz, heads, seqlen, headdim, basis, causal), timings in sorted(grouped.items()):
        primary = next((name for name in preferred_primary if name in timings), None)
        if primary is None:
            continue
        primary_ms = timings[primary]
        entry: dict[str, object] = {
            "B": bsz,
            "H": heads,
            "T": seqlen,
            "Dh": headdim,
            "P": basis,
            "causal": causal,
            "primary_backend": primary,
            "primary_ms": primary_ms,
        }
        for backend, timing_ms in timings.items():
            entry[f"{backend}_ms"] = timing_ms
            if primary_ms > 0.0 and backend != primary:
                entry[f"primary_speedup_vs_{backend}"] = timing_ms / primary_ms
        comparisons.append(entry)
    return {"num_shapes": len(comparisons), "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmark the Transformer attention input pullback kernel.")
    parser.add_argument("--batches", default="1", help="Comma-separated batch sizes")
    parser.add_argument("--heads", default="8", help="Comma-separated attention head counts")
    parser.add_argument("--seq-lens", default="128,512", help="Comma-separated sequence lengths")
    parser.add_argument("--head-dims", default="64", help="Comma-separated head dimensions")
    parser.add_argument("--basis-sizes", default="4", help="Comma-separated P basis sizes")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--causal", choices=["true", "false", "both"], default="true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--include-simple", action="store_true", help="Also time the scalar CUDA simple mode")
    parser.add_argument("--include-autograd", action="store_true", help="Also time PyTorch autograd P-pass VJP baseline")
    parser.add_argument("--include-fa2-loop", action="store_true", help="Also time the SDPA-backed FA-2 loop adapter")
    parser.add_argument("--include-fa2-p1", action="store_true", help="Also time the P=1 SDPA-backed adapter; requires --basis-sizes 1")
    parser.add_argument("--include-fa2-mma-p1", action="store_true", help="Also time the CUTLASS/CuTe P=1 implementation; requires --basis-sizes 1 and float16")
    parser.add_argument("--include-fa2-mma-pblock4", action="store_true", help="Also time the CUTLASS/CuTe P-block implementation; requires 1 <= --basis-sizes <= 4 and float16")
    parser.add_argument("--skip-torch-formula", action="store_true", help="Do not time the vectorized PyTorch formula baseline")
    parser.add_argument("--no-default-kernel", action="store_true", help="Do not time the default fa2_mma_pblock4 kernel")
    parser.add_argument("--check", action="store_true", help="Check the default kernel against autograd once per shape")
    parser.add_argument("--check-max-tokens", type=int, default=131072, help="Skip autograd checks above B*H*T*Dh*P")
    parser.add_argument("--atol", type=float, default=3e-4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for benchmarks/time_attention_pullback.py")

    batches = _parse_int_list(args.batches)
    heads = _parse_int_list(args.heads)
    seq_lens = _parse_int_list(args.seq_lens)
    head_dims = _parse_int_list(args.head_dims)
    basis_sizes = _parse_int_list(args.basis_sizes)
    causal_values = [True, False] if args.causal == "both" else [args.causal == "true"]

    backends = [] if args.no_default_kernel else ["fa2_mma_pblock4"]
    if not args.skip_torch_formula:
        backends.append("torch_formula")
    if args.include_simple:
        backends.append("simple")
    if args.include_fa2_loop:
        backends.append("fa2_loop")
    if args.include_fa2_p1:
        backends.append("fa2_p1")
    if args.include_fa2_mma_p1:
        backends.append("fa2_mma_p1")
    if args.include_fa2_mma_pblock4 and "fa2_mma_pblock4" not in backends:
        backends.append("fa2_mma_pblock4")
    if args.include_autograd:
        backends.append("autograd")
    if args.dtype != "float32" and "simple" in backends:
        raise SystemExit("simple currently requires --dtype float32; omit --include-simple for fp16/bf16")
    if not backends:
        raise SystemExit("no backends selected")

    rows: list[dict[str, object]] = []
    for bsz in batches:
        for n_heads in heads:
            for seqlen in seq_lens:
                for headdim in head_dims:
                    for basis in basis_sizes:
                        for causal in causal_values:
                            torch.manual_seed(0)
                            dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
                            q = torch.randn(bsz, n_heads, seqlen, headdim, device="cuda", dtype=dtype)
                            k = torch.randn_like(q)
                            v = torch.randn_like(q)
                            output_cotangent_basis = torch.randn(
                                bsz,
                                basis,
                                n_heads,
                                seqlen,
                                headdim,
                                device="cuda",
                                dtype=dtype,
                            )
                            scale = 1.0 / math.sqrt(headdim)

                            max_abs = None
                            check_size = bsz * n_heads * seqlen * headdim * basis
                            if args.check and check_size <= args.check_max_tokens:
                                check_backend = "fa2_mma_pblock4" if "fa2_mma_pblock4" in backends else backends[0]
                                actual = _cuda_pullback(
                                    check_backend,
                                    q,
                                    k,
                                    v,
                                    output_cotangent_basis,
                                    scale=scale,
                                    causal=causal,
                                )
                                expected = _autograd_pullback(q, k, v, output_cotangent_basis, scale=scale, causal=causal)
                                max_abs = _max_abs_error(actual, expected)
                                if max_abs > args.atol:
                                    raise AssertionError(
                                        f"max_abs={max_abs:.6g} exceeds atol={args.atol} for "
                                        f"B={bsz} H={n_heads} T={seqlen} Dh={headdim} P={basis} causal={causal}"
                                    )

                            for backend in backends:
                                median_ms, best_ms, peak_bytes = _run_backend(
                                    backend,
                                    q,
                                    k,
                                    v,
                                    output_cotangent_basis,
                                    scale=scale,
                                    causal=causal,
                                    warmup=args.warmup,
                                    iters=args.iters,
                                )
                                rows.append(
                                    {
                                        "B": bsz,
                                        "H": n_heads,
                                        "T": seqlen,
                                        "Dh": headdim,
                                        "P": basis,
                                        "causal": causal,
                                        "dtype": args.dtype,
                                        "backend": backend,
                                        "median_ms": median_ms,
                                        "best_ms": best_ms,
                                        "peak_memory_bytes": peak_bytes,
                                        "max_abs_error_vs_autograd": max_abs,
                                    }
                                )
                                print(
                                    f"B={bsz} H={n_heads} T={seqlen} Dh={headdim} P={basis} "
                                    f"causal={causal} backend={backend} median={median_ms:.3f}ms best={best_ms:.3f}ms"
                                )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"attention_pullback_{stamp}.csv"
    latest_csv = OUT_DIR / "latest.csv"
    summary_path = OUT_DIR / f"attention_pullback_{stamp}_summary.json"
    latest_summary = OUT_DIR / "latest_summary.json"

    fieldnames = [
        "B",
        "H",
        "T",
        "Dh",
        "P",
        "causal",
        "dtype",
        "backend",
        "median_ms",
        "best_ms",
        "peak_memory_bytes",
        "max_abs_error_vs_autograd",
    ]
    for path in (csv_path, latest_csv):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = _summarize(rows)
    for path in (summary_path, latest_summary):
        with path.open("w") as f:
            json.dump(summary, f, indent=2)

    print(f"wrote raw results: {csv_path}")
    print(f"wrote summary: {summary_path}")
    print(f"shapes: {summary['num_shapes']}")
    for entry in summary["comparisons"][:20]:
        parts = [
            f"B={entry['B']}",
            f"H={entry['H']}",
            f"T={entry['T']}",
            f"Dh={entry['Dh']}",
            f"P={entry['P']}",
            f"causal={entry['causal']}",
            f"primary={entry['primary_backend']}:{entry['primary_ms']:.3f}ms",
        ]
        for key, value in sorted(entry.items()):
            if key.startswith("primary_speedup_vs_"):
                parts.append(f"{key}={value:.3f}x")
        print("  " + " ".join(parts))


if __name__ == "__main__":
    main()
