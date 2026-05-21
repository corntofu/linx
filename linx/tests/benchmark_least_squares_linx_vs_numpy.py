"""Benchmark least-squares solvers: linx vs NumPy.

Usage:
    PYTHONPATH=python python3 tests/benchmark_least_squares_linx_vs_numpy.py
    PYTHONPATH=python python3 tests/benchmark_least_squares_linx_vs_numpy.py --quick
    PYTHONPATH=python python3 tests/benchmark_least_squares_linx_vs_numpy.py --sizes 512x64 2048x256
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
sys.path = [str(PYTHON_DIR)] + [path for path in sys.path if path != str(PYTHON_DIR)]
sys.modules.pop("linx", None)
linx = importlib.import_module("linx")


DEFAULT_SIZES = ["128x32", "512x64", "1024x128", "2048x256"]
QUICK_SIZES = ["128x32", "512x64"]


def parse_size(value: str) -> tuple[int, int]:
    try:
        rows, cols = value.lower().split("x", 1)
        rows_i = int(rows)
        cols_i = int(cols)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid size {value!r}; use MxN") from exc
    if rows_i < cols_i:
        raise argparse.ArgumentTypeError("least-squares benchmark requires M >= N")
    return rows_i, cols_i


def fmt_time(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print(sep)
    print("|" + "|".join(f" {header:^{width}} " for header, width in zip(headers, widths)) + "|")
    print(sep)
    for row in rows:
        print("|" + "|".join(f" {str(cell):>{width}} " for cell, width in zip(row, widths)) + "|")
    print(sep)


def timed(fn, *args, reps: int, warmup: int) -> tuple[float, object]:
    result = None
    for _ in range(warmup):
        result = fn(*args)
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - start)
    return float(np.mean(times)), result


def make_problem(m: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = rng.standard_normal((m, n), dtype=np.float64)
    # Column scaling keeps the normal-equation fallback from becoming too ill-conditioned.
    a /= np.sqrt(np.sum(a * a, axis=0, keepdims=True))
    x_true = rng.standard_normal((n, 1), dtype=np.float64)
    noise = 1e-4 * rng.standard_normal((m, 1), dtype=np.float64)
    b = a @ x_true + noise
    return a, b, x_true


def numpy_lstsq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(a, b, rcond=None)[0]


def run_benchmark(sizes: list[tuple[int, int]], reps: int, warmup: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = []

    for m, n in sizes:
        a, b, _ = make_problem(m, n, rng)

        t_np, x_np = timed(numpy_lstsq, a, b, reps=reps, warmup=warmup)
        t_linx, x_linx = timed(linx.least_squares, a, b, reps=reps, warmup=warmup)

        residual_np = np.linalg.norm(a @ x_np - b)
        residual_linx = np.linalg.norm(a @ x_linx - b)
        x_rel_err = np.linalg.norm(x_linx - x_np) / max(np.linalg.norm(x_np), 1e-12)
        residual_gap = abs(residual_linx - residual_np) / max(np.linalg.norm(b), 1e-12)
        speedup = t_np / t_linx if t_linx > 0 else float("inf")

        rows.append(
            [
                f"{m}x{n}",
                fmt_time(t_np),
                fmt_time(t_linx),
                f"{speedup:.2f}x",
                f"{x_rel_err:.2e}",
                f"{residual_gap:.2e}",
            ]
        )

    print_table(
        ["Size (M x N)", "NumPy lstsq", "linx least_squares", "NumPy/linx", "x rel err", "resid gap/b"],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark linx.least_squares against NumPy np.linalg.lstsq")
    parser.add_argument("--quick", action="store_true", help="Use smaller default sizes")
    parser.add_argument("--sizes", nargs="+", help="Custom sizes in MxN form, e.g. 512x64 2048x256")
    parser.add_argument("--reps", type=int, default=5, help="Timed repetitions per size")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup repetitions per size")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    size_names = args.sizes if args.sizes else (QUICK_SIZES if args.quick else DEFAULT_SIZES)
    sizes = [parse_size(size) for size in size_names]

    print("=" * 84)
    print("  Least Squares Benchmark: linx.least_squares vs NumPy np.linalg.lstsq")
    print("=" * 84)
    print(f"  backend: {linx.hardware_backend()}")
    print(f"  linx package: {Path(linx.__file__).resolve()}")
    print(f"  NumPy version: {np.__version__}")
    print(f"  reps={args.reps}, warmup={args.warmup}, seed={args.seed}")
    print("=" * 84)
    run_benchmark(sizes, reps=args.reps, warmup=args.warmup, seed=args.seed)


if __name__ == "__main__":
    main()
