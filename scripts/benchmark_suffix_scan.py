from __future__ import annotations

import argparse
import csv
import itertools
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


try:
    import interface_scan_cuda  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "Could not import interface_scan_cuda. Build the extension first with:\n"
        "  PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/interface/build.sh"
    ) from exc


DEFAULT_SHAPES: list[tuple[int, int, int]] = [
    (1, 4, 8),
    (1, 8, 16),
    (2, 8, 32),
    (4, 16, 32),
    (8, 16, 64),
    (16, 32, 64),
]


FULL_BS = [1, 2, 4, 8, 16]
FULL_KS = [4, 8, 16, 32]
FULL_RS = [8, 16, 32, 64]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_shape_list(s: str) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []

    for item in s.split(";"):
        item = item.strip()
        if not item:
            continue

        parts = [int(x.strip()) for x in item.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid shape '{item}'. Expected format B,K,R.")

        shapes.append((parts[0], parts[1], parts[2]))

    return shapes


def make_input(B: int, K: int, R: int, *, dtype: torch.dtype) -> torch.Tensor:
    scale = 1.0 / (R ** 0.5)
    return torch.randn(B, K, R, R, device="cuda", dtype=dtype) * scale


@torch.no_grad()
def manual_suffix_scan(mats: torch.Tensor) -> torch.Tensor:
    mats_f32 = mats.to(torch.float32)
    B, K, R, _ = mats_f32.shape

    out = torch.empty((B, K + 1, R, R), device=mats.device, dtype=torch.float32)
    eye = torch.eye(R, device=mats.device, dtype=torch.float32).view(1, R, R).expand(B, R, R)

    running = eye.clone()
    out[:, K] = eye

    for i in reversed(range(K)):
        running = mats_f32[:, i] @ running
        out[:, i] = running

    return out


def float32_tolerances(K: int, R: int) -> tuple[float, float]:
    if R >= 256:
        return 7e-1, 1e-5
    if R >= 128:
        return 2.5e-1, 1e-5
    if K >= 8 or R >= 16:
        return 3e-2, 1e-5
    return 2e-5, 1e-5


@torch.no_grad()
def check_correctness(
    name: str,
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> None:
    actual = fn(x)
    expected = manual_suffix_scan(x)

    B, K, R, _ = x.shape
    atol, rtol = float32_tolerances(K, R)

    if actual.shape != expected.shape:
        raise RuntimeError(
            f"{name}: shape mismatch for B={B}, K={K}, R={R}: "
            f"actual={tuple(actual.shape)}, expected={tuple(expected.shape)}"
        )

    if actual.dtype != torch.float32:
        raise RuntimeError(
            f"{name}: dtype mismatch for B={B}, K={K}, R={R}: "
            f"actual={actual.dtype}, expected=torch.float32"
        )

    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        max_abs = (actual - expected).abs().max().item()
        raise RuntimeError(
            f"{name}: correctness failed for B={B}, K={K}, R={R}: "
            f"max_abs_err={max_abs:.6e}, atol={atol}, rtol={rtol}"
        )


@torch.no_grad()
def time_sync_wall_ms(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        fn(x)

    torch.cuda.synchronize()

    times_ms: list[float] = []

    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()

        fn(x)

        torch.cuda.synchronize()
        end = time.perf_counter()

        times_ms.append((end - start) * 1000.0)

    return statistics.median(times_ms)


@torch.no_grad()
def time_gpu_event_ms(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        fn(x)

    torch.cuda.synchronize()

    times_ms: list[float] = []

    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn(x)
        end.record()

        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return statistics.median(times_ms)

def format_value(value: object, width: int, kind: str) -> str:
    if value is None:
        return f"{'-':>{width}}"

    if kind == "int":
        return f"{int(value):>{width}d}"

    if kind == "ms":
        return f"{float(value):>{width}.4f}"

    if kind == "speedup":
        return f"{float(value):>{width}.2f}x"

    return f"{str(value):>{width}}"


def print_table(rows: list[dict[str, object]], *, has_optimized: bool) -> None:
    if has_optimized:
        columns = [
            ("B", "B", 4, "int"),
            ("K", "K", 4, "int"),
            ("R", "R", 4, "int"),
            ("baseline_wall_ms", "baseline wall ms", 18, "ms"),
            ("optimized_wall_ms", "optimized wall ms", 19, "ms"),
            ("speedup", "speedup", 10, "speedup"),
            ("baseline_event_ms", "baseline event ms", 19, "ms"),
            ("optimized_event_ms", "optimized event ms", 20, "ms"),
        ]
    else:
        columns = [
            ("B", "B", 4, "int"),
            ("K", "K", 4, "int"),
            ("R", "R", 4, "int"),
            ("baseline_wall_ms", "baseline wall ms", 18, "ms"),
            ("baseline_event_ms", "baseline event ms", 19, "ms"),
        ]

    header = "  ".join(f"{title:>{width}}" for _, title, width, _ in columns)
    print(header)
    print("-" * len(header))

    for row in rows:
        line = "  ".join(
            format_value(row[key], width, kind)
            for key, _, width, kind in columns
        )
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CUDA suffix scan implementations.")

    parser.add_argument(
        "--shapes",
        default=None,
        help=(
            "Semicolon-separated shapes in B,K,R format. "
            "Example: --shapes '1,4,8;4,16,32;16,32,64'"
        ),
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Run the full B/K/R grid instead of the small default shape set.",
    )
    parser.add_argument("--B", default=None, help="Comma-separated B values for custom grid.")
    parser.add_argument("--K", default=None, help="Comma-separated K values for custom grid.")
    parser.add_argument("--R", default=None, help="Comma-separated R values for custom grid.")

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--skip-check", action="store_true", help="Skip correctness checks.")

    parser.add_argument("--baseline-fn", default="suffix_scan_mats")
    parser.add_argument("--optimized-fn", default="suffix_scan_jacobian")
    parser.add_argument(
        "--no-optimized",
        action="store_true",
        help="Do not attempt to benchmark the optimized function.",
    )

    parser.add_argument("--csv", type=Path, default=None, help="Optional path to write CSV results.")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    if not hasattr(interface_scan_cuda, args.baseline_fn):
        raise SystemExit(f"Missing baseline function: interface_scan_cuda.{args.baseline_fn}")

    baseline_fn = getattr(interface_scan_cuda, args.baseline_fn)

    optimized_fn = None
    has_optimized = False
    if not args.no_optimized and hasattr(interface_scan_cuda, args.optimized_fn):
        optimized_fn = getattr(interface_scan_cuda, args.optimized_fn)
        has_optimized = True

    if args.shapes is not None:
        shapes = parse_shape_list(args.shapes)
    elif args.B is not None or args.K is not None or args.R is not None:
        Bs = parse_int_list(args.B) if args.B is not None else FULL_BS
        Ks = parse_int_list(args.K) if args.K is not None else FULL_KS
        Rs = parse_int_list(args.R) if args.R is not None else FULL_RS
        shapes = list(itertools.product(Bs, Ks, Rs))
    elif args.full_grid:
        shapes = list(itertools.product(FULL_BS, FULL_KS, FULL_RS))
    else:
        shapes = DEFAULT_SHAPES

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print()
    print("CUDA suffix scan benchmark")
    print("==========================")
    print(f"GPU:          {torch.cuda.get_device_name()}")
    print(f"dtype:        {args.dtype}")
    print(f"baseline:     interface_scan_cuda.{args.baseline_fn}")

    if has_optimized:
        print(f"optimized:    interface_scan_cuda.{args.optimized_fn}")
    else:
        print(f"optimized:    not found: interface_scan_cuda.{args.optimized_fn}")

    print(f"warmup:       {args.warmup}")
    print(f"iters:        {args.iters}")
    print(f"num shapes:   {len(shapes)}")
    print()

    rows: list[dict[str, object]] = []

    for B, K, R in shapes:
        x = make_input(B, K, R, dtype=dtype)

        if not args.skip_check:
            check_correctness(args.baseline_fn, baseline_fn, x)
            if optimized_fn is not None:
                check_correctness(args.optimized_fn, optimized_fn, x)

        baseline_wall_ms = time_sync_wall_ms(
            baseline_fn,
            x,
            warmup=args.warmup,
            iters=args.iters,
        )
        baseline_event_ms = time_gpu_event_ms(
            baseline_fn,
            x,
            warmup=args.warmup,
            iters=args.iters,
        )

        optimized_wall_ms = None
        optimized_event_ms = None
        speedup = None

        if optimized_fn is not None:
            optimized_wall_ms = time_sync_wall_ms(
                optimized_fn,
                x,
                warmup=args.warmup,
                iters=args.iters,
            )
            optimized_event_ms = time_gpu_event_ms(
                optimized_fn,
                x,
                warmup=args.warmup,
                iters=args.iters,
            )
            speedup = baseline_wall_ms / optimized_wall_ms

        rows.append(
            {
                "B": B,
                "K": K,
                "R": R,
                "dtype": args.dtype,
                "baseline_wall_ms": baseline_wall_ms,
                "optimized_wall_ms": optimized_wall_ms,
                "speedup": speedup,
                "baseline_event_ms": baseline_event_ms,
                "optimized_event_ms": optimized_event_ms,
            }
        )

    print_table(rows, has_optimized=has_optimized)

    if has_optimized:
        speedups = [float(row["speedup"]) for row in rows if row["speedup"] is not None]
        if speedups:
            print()
            print("Summary")
            print("-------")
            print(f"median speedup: {statistics.median(speedups):.2f}x")
            print(f"min speedup:    {min(speedups):.2f}x")
            print(f"max speedup:    {max(speedups):.2f}x")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "B",
                    "K",
                    "R",
                    "dtype",
                    "baseline_wall_ms",
                    "optimized_wall_ms",
                    "speedup",
                    "baseline_event_ms",
                    "optimized_event_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print()
        print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
