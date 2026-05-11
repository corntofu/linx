"""Compare NumPy inverse, linx inverse, and linx Schur inverse.

Run from the lingraphics project root:

    /opt/anaconda3/bin/python3 examples/compare_numpy_linx_schur.py --sizes 64,128,256 --reps 3
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingraphics.backends import create_backend


def parse_sizes(value: str) -> list[int]:
    sizes = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if chunk:
            sizes.append(int(chunk))
    if not sizes:
        raise ValueError("at least one size is required")
    return sizes


def make_matrix(size: int, seed: int = 2026) -> np.ndarray:
    rng = np.random.default_rng(seed + size)
    matrix = rng.normal(0.0, 1.0 / math.sqrt(size), size=(size, size))
    matrix += np.eye(size, dtype=np.float64) * 3.0
    return np.ascontiguousarray(matrix, dtype=np.float64)


def time_call(fn, reps: int, warmup: int):
    for _ in range(warmup):
        fn()
    values = []
    result = None
    for _ in range(reps):
        start = time.perf_counter()
        result = fn()
        values.append((time.perf_counter() - start) * 1000.0)
    return result, sum(values) / len(values), min(values), max(values)


def residual(matrix: np.ndarray, inverse: np.ndarray) -> float:
    eye = np.eye(matrix.shape[0], dtype=np.float64)
    return float(np.linalg.norm(matrix @ inverse - eye, ord="fro") / matrix.shape[0])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="64,128,256")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--min-block", type=int, default=32)
    args = parser.parse_args(argv)

    linx_backend = create_backend("linx")
    rows = []

    for size in parse_sizes(args.sizes):
        matrix = make_matrix(size)
        methods = [
            ("numpy", "np.linalg.inv", lambda: np.linalg.inv(matrix)),
            ("linx", "inverse(lu)", lambda: linx_backend.inverse(matrix, method="lu", min_block=args.min_block)),
            (
                "linx-schur",
                "inverse(schur)",
                lambda: linx_backend.inverse(matrix, method="schur", min_block=args.min_block),
            ),
        ]
        for backend, method, fn in methods:
            inv, mean_ms, min_ms, max_ms = time_call(fn, reps=args.reps, warmup=args.warmup)
            rows.append(
                [
                    backend,
                    method,
                    str(size),
                    f"{mean_ms:.3f}",
                    f"{min_ms:.3f}",
                    f"{max_ms:.3f}",
                    f"{residual(matrix, inv):.2e}",
                ]
            )

    headers = ["Backend", "Method", "Size", "Mean ms", "Min ms", "Max ms", "Residual"]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    sep = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print(sep)
    print("|" + "|".join(f" {header:^{widths[index]}} " for index, header in enumerate(headers)) + "|")
    print(sep)
    for row in rows:
        print("|" + "|".join(f" {value:>{widths[index]}} " for index, value in enumerate(row)) + "|")
    print(sep)
    print(f"linx engine: {linx_backend.info.matrix_engine}")


if __name__ == "__main__":
    main()
