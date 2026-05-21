#!/usr/bin/env python3
"""Run the N=4096 least-squares benchmark for linx vs NumPy.

Default problem:
    A: 4096 x 4096
    b: 4096 x 1

Usage:
    PYTHONPATH=python tests/benchmark_least_squares_n4096.py
    PYTHONPATH=python tests/benchmark_least_squares_n4096.py --m 8192 --n 4096
    PYTHONPATH=python tests/benchmark_least_squares_n4096.py --n 64 --reps 1 --warmup 0
"""

from __future__ import annotations

import argparse

from benchmark_least_squares_linx_vs_numpy import linx, np, run_benchmark


def estimate_gib(m: int, n: int) -> float:
    # A plus b plus two solution vectors and temporary LAPACK/NumPy work not included.
    bytes_min = (m * n + m + 2 * n) * np.dtype(np.float64).itemsize
    return bytes_min / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser(description="N=4096 least-squares benchmark: linx vs NumPy")
    parser.add_argument("--n", type=int, default=4096, help="Number of columns / unknowns")
    parser.add_argument("--m", type=int, default=None, help="Number of rows. Defaults to n.")
    parser.add_argument("--reps", type=int, default=1, help="Timed repetitions")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup repetitions")
    parser.add_argument("--seed", type=int, default=4096, help="Random seed")
    args = parser.parse_args()

    m = args.n if args.m is None else args.m
    if m < args.n:
        raise SystemExit("least-squares benchmark requires m >= n")

    print("=" * 84)
    print(f"  N={args.n} Least Squares Benchmark: linx.least_squares vs NumPy lstsq")
    print("=" * 84)
    print(f"  problem: A={m}x{args.n}, b={m}x1")
    print(f"  estimated minimum input/result memory: {estimate_gib(m, args.n):.2f} GiB")
    print(f"  backend: {linx.hardware_backend()}")
    print(f"  NumPy version: {np.__version__}")
    print("=" * 84)

    run_benchmark([(m, args.n)], reps=args.reps, warmup=args.warmup, seed=args.seed)


if __name__ == "__main__":
    main()
