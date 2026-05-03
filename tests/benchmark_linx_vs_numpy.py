"""Benchmark linx C++ backend vs NumPy for various linear algebra operations.

Usage:
    python tests/benchmark_linx_vs_numpy.py
    python tests/benchmark_linx_vs_numpy.py --quick   # smaller matrices, fewer repetitions
    python tests/benchmark_linx_vs_numpy.py --only-inverse --include-inverse-8192
"""

import argparse
import importlib
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
sys.path = [str(PYTHON_DIR)] + [path for path in sys.path if path != str(PYTHON_DIR)]
sys.modules.pop("linx", None)

linx = importlib.import_module("linx")


def timed(fn, args=(), kwargs=None, reps=5, warmup=2):
    """Return (mean_seconds, stdev_seconds) over *reps* runs."""
    if kwargs is None:
        kwargs = {}
    # warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    arr = np.array(times)
    return float(np.mean(arr)), float(np.std(arr))


def fmt_time(seconds):
    if seconds < 1e-6:
        return f"{seconds*1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds*1e6:.2f} µs"
    if seconds < 1:
        return f"{seconds*1e3:.2f} ms"
    return f"{seconds:.3f} s"


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_line = "|" + "|".join(f" {h:^{w}} " for h, w in zip(headers, widths)) + "|"
    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        print("|" + "|".join(f" {str(c):>{w}} " for c, w in zip(row, widths)) + "|")
    print(sep)


# ── benchmark configurations ──────────────────────────────────────────────

SIZES_QUICK = ["64x64", "256x256"]
SIZES_FULL = ["64x64", "256x256", "512x512", "1024x1024", "2048x2048"]
INVERSE_EXTREME_SIZE = "8192x8192"

MATMUL_SIZES_QUICK = [("64x64", 64, 64, 64)]
MATMUL_SIZES_FULL = [
    ("64x64", 64, 64, 64),
    ("256x256", 256, 256, 256),
    ("512x512", 512, 512, 512),
    ("1024x1024", 1024, 1024, 1024),
    ("2048x2048", 2048, 2048, 2048),
]


def parse_shape(name):
    parts = name.split("x")
    return int(parts[0]), int(parts[1])


def bench_matmul(quick=False):
    print("\n" + "=" * 72)
    print("  MATMUL (C = A @ B)  —  linx vs NumPy")
    print("=" * 72)

    sizes = MATMUL_SIZES_QUICK if quick else MATMUL_SIZES_FULL
    headers = ["Size (M×K×N)", "NumPy", "linx", "NumPy/linx"]
    rows = []

    for name, m, k, n in sizes:
        a_np = np.random.randn(m, k).astype(np.float64)
        b_np = np.random.randn(k, n).astype(np.float64)
        a_linx = a_np.copy()
        b_linx = b_np.copy()

        reps = 3 if max(m, k, n) >= 512 else 10

        t_np, _ = timed(np.dot, (a_np, b_np), reps=reps)
        t_linx, _ = timed(linx.matmul, (a_linx, b_linx), reps=reps)

        ratio = t_np / t_linx if t_linx > 0 else float("inf")
        rows.append([name, fmt_time(t_np), fmt_time(t_linx), f"{ratio:.2f}x"])

    print_table(headers, rows)


def bench_matmul_classic_strassen(quick=False):
    print("\n" + "=" * 72)
    print("  MATMUL — classic vs Strassen  (linx only)")
    print("=" * 72)

    sizes = MATMUL_SIZES_QUICK if quick else MATMUL_SIZES_FULL
    headers = ["Size", "classic", "strassen", "classic/strassen"]
    rows = []

    for name, m, k, n in sizes:
        a = np.random.randn(m, k).astype(np.float64)
        b = np.random.randn(k, n).astype(np.float64)

        reps = 3 if max(m, k, n) >= 512 else 5

        t_classic, _ = timed(linx.matmul, (a, b), reps=reps)
        t_strassen, _ = timed(linx.matmul_strassen, (a, b), kwargs={"threshold": 64}, reps=reps)

        ratio = t_classic / t_strassen if t_strassen > 0 else float("inf")
        rows.append([name, fmt_time(t_classic), fmt_time(t_strassen), f"{ratio:.2f}x"])

    print_table(headers, rows)


def make_inverse_input(n):
    a_np = np.random.randn(n, n).astype(np.float64)
    if n >= 4096:
        # Avoid the extra O(n^3) setup cost and memory spike of A @ A.T for
        # extreme inverse benchmarks; diagonal dominance keeps the input stable.
        a_np *= 1.0 / math.sqrt(n)
        diag = np.arange(n)
        a_np[diag, diag] += 4.0
        return a_np
    return a_np @ a_np.T + np.eye(n, dtype=np.float64) * 0.1


def bench_inverse(quick=False, include_8192=False):
    print("\n" + "=" * 72)
    print("  INVERSE  —  linx Schur variants vs NumPy (np.linalg.inv)")
    print("=" * 72)

    sizes = list(SIZES_QUICK if quick else SIZES_FULL)
    if include_8192 and INVERSE_EXTREME_SIZE not in sizes:
        sizes.append(INVERSE_EXTREME_SIZE)
        print("  Note: 8192x8192 measures Schur variants only; NumPy/LU are skipped.")

    headers = [
        "Size",
        "NumPy (inv)",
        "linx (schur)",
        "schur+strassen",
        "linx (lu)",
        "NumPy/schur",
    ]
    rows = []

    for name in sizes:
        n, _ = parse_shape(name)
        a_np = make_inverse_input(n)
        a_linx = a_np if n >= 4096 else a_np.copy()

        is_extreme = n >= 4096
        reps = 1 if is_extreme else (3 if n >= 512 else 5)
        warmup = 0 if is_extreme else 2

        t_np = None
        t_lu = None
        if not is_extreme:
            t_np, _ = timed(np.linalg.inv, (a_np,), reps=reps, warmup=warmup)

        t_schur, _ = timed(
            linx.inverse,
            (a_linx,),
            kwargs={"method": "schur"},
            reps=reps,
            warmup=warmup,
        )
        t_schur_strassen, _ = timed(
            linx.inverse_schur_strassen,
            (a_linx,),
            kwargs={"min_block": 1024, "strassen_threshold": 4096},
            reps=reps,
            warmup=warmup,
        )
        if not is_extreme:
            t_lu, _ = timed(
                linx.inverse,
                (a_linx,),
                kwargs={"method": "lu"},
                reps=reps,
                warmup=warmup,
            )

        ratio = t_np / t_schur if t_np is not None and t_schur > 0 else None
        rows.append([
            name,
            fmt_time(t_np) if t_np is not None else "skipped",
            fmt_time(t_schur),
            fmt_time(t_schur_strassen),
            fmt_time(t_lu) if t_lu is not None else "skipped",
            f"{ratio:.2f}x" if ratio is not None else "-",
        ])

    print_table(headers, rows)


def bench_elementwise(quick=False):
    print("\n" + "=" * 72)
    print("  ELEMENT-WISE OPS  —  linx vs NumPy")
    print("=" * 72)

    sizes = SIZES_QUICK if quick else SIZES_FULL
    ops = [
        ("add (A+B)", lambda a, b: a + b, lambda a, b: linx.add(a, b)),
        ("sub (A-B)", lambda a, b: a - b, lambda a, b: linx.subtract(a, b)),
        ("hadamard (A*B)", lambda a, b: a * b, lambda a, b: linx.hadamard(a, b)),
        ("scalar (A*s)", lambda a, b: a * 2.5, lambda a, b: linx.scalar_mul(a, 2.5)),
        ("negate (-A)", lambda a, b: -a, lambda a, b: linx.neg(a)),
    ]

    for op_name, np_fn, linx_fn in ops:
        print(f"\n  --- {op_name} ---")
        headers = ["Size", "NumPy", "linx", "NumPy/linx"]
        rows = []

        for name in sizes:
            r, c = parse_shape(name)
            a_np = np.random.randn(r, c).astype(np.float64)
            b_np = np.random.randn(r, c).astype(np.float64)

            reps = 10 if max(r, c) >= 1024 else 20

            t_np, _ = timed(np_fn, (a_np, b_np), reps=reps)
            t_linx, _ = timed(linx_fn, (a_np, b_np), reps=reps)

            ratio = t_np / t_linx if t_linx > 0 else float("inf")
            rows.append([name, fmt_time(t_np), fmt_time(t_linx), f"{ratio:.2f}x"])

        print_table(headers, rows)


def bench_transpose(quick=False):
    print("\n" + "=" * 72)
    print("  TRANSPOSE  —  linx vs NumPy")
    print("=" * 72)

    sizes = SIZES_QUICK if quick else SIZES_FULL
    headers = ["Size", "NumPy (.T)", "linx (transpose)", "NumPy/linx"]
    rows = []

    for name in sizes:
        r, c = parse_shape(name)
        a_np = np.random.randn(r, c).astype(np.float64)

        reps = 20 if max(r, c) <= 512 else 10

        t_np, _ = timed(lambda x: x.T.copy(), (a_np,), reps=reps)
        t_linx, _ = timed(linx.transpose, (a_np,), reps=reps)

        ratio = t_np / t_linx if t_linx > 0 else float("inf")
        rows.append([name, fmt_time(t_np), fmt_time(t_linx), f"{ratio:.2f}x"])

    print_table(headers, rows)


def bench_solve(quick=False):
    print("\n" + "=" * 72)
    print("  SOLVE (Ax = B)  —  linx vs NumPy")
    print("=" * 72)

    sizes = SIZES_QUICK if quick else SIZES_FULL
    headers = ["Size", "NumPy (solve)", "linx (solve)", "NumPy/linx"]
    rows = []

    for name in sizes:
        n, _ = parse_shape(name)
        a_np = np.random.randn(n, n).astype(np.float64)
        b_np = np.random.randn(n, 1).astype(np.float64)
        # Make well-conditioned
        a_np = a_np @ a_np.T + np.eye(n) * 0.5

        reps = 3 if n >= 512 else 5

        t_np, _ = timed(np.linalg.solve, (a_np, b_np), reps=reps)
        t_linx, _ = timed(linx.solve, (a_np, b_np), reps=reps)

        ratio = t_np / t_linx if t_linx > 0 else float("inf")
        rows.append([name, fmt_time(t_np), fmt_time(t_linx), f"{ratio:.2f}x"])

    print_table(headers, rows)


def bench_det_trace_norm(quick=False):
    print("\n" + "=" * 72)
    print("  DET / TRACE / NORM  —  linx vs NumPy")
    print("=" * 72)

    sizes = SIZES_QUICK if quick else SIZES_FULL

    # trace
    headers = ["Size", "NumPy (trace)", "linx (trace)", "NumPy/linx"]
    rows = []
    for name in sizes:
        n, _ = parse_shape(name)
        a_np = np.random.randn(n, n).astype(np.float64)
        reps = 20
        t_np, _ = timed(np.trace, (a_np,), reps=reps)
        t_linx, _ = timed(linx.trace, (a_np,), reps=reps)
        ratio = t_np / t_linx if t_linx > 0 else float("inf")
        rows.append([name, fmt_time(t_np), fmt_time(t_linx), f"{ratio:.2f}x"])
    print("\n  --- trace ---")
    print_table(headers, rows)

    # det
    headers = ["Size", "NumPy (det)", "linx (det)", "NumPy/linx"]
    rows = []
    for name in sizes:
        n, _ = parse_shape(name)
        a_np = np.random.randn(n, n).astype(np.float64)
        a_np = a_np @ a_np.T + np.eye(n) * 0.1
        reps = 3 if n >= 512 else 5
        t_np, _ = timed(np.linalg.det, (a_np,), reps=reps)
        t_linx, _ = timed(linx.det, (a_np,), reps=reps)
        ratio = t_np / t_linx if t_linx > 0 else float("inf")
        rows.append([name, fmt_time(t_np), fmt_time(t_linx), f"{ratio:.2f}x"])
    print("\n  --- det ---")
    print_table(headers, rows)

    # frobenius norm
    headers = ["Size", "NumPy (norm)", "linx (norm)", "NumPy/linx"]
    rows = []
    for name in sizes:
        r, c = parse_shape(name)
        a_np = np.random.randn(r, c).astype(np.float64)
        reps = 20 if max(r, c) <= 512 else 10
        t_np, _ = timed(np.linalg.norm, (a_np,), reps=reps)
        t_linx, _ = timed(linx.frobenius_norm, (a_np,), reps=reps)
        ratio = t_np / t_linx if t_linx > 0 else float("inf")
        rows.append([name, fmt_time(t_np), fmt_time(t_linx), f"{ratio:.2f}x"])
    print("\n  --- frobenius_norm ---")
    print_table(headers, rows)


def bench_matrix_class(quick=False):
    """Test the higher-level Matrix class (operator overloads)."""
    print("\n" + "=" * 72)
    print("  Matrix CLASS OPS  —  linx.Matrix vs ndarray")
    print("=" * 72)

    sizes = SIZES_QUICK if quick else ["256x256", "512x512"]
    headers = ["Size", "ndarray @", "Matrix @", "ndarray +", "Matrix +", "@ speedup"]
    rows = []

    for name in sizes:
        n, _ = parse_shape(name)
        a_np = np.random.randn(n, n).astype(np.float64)
        b_np = np.random.randn(n, n).astype(np.float64)
        a_m = linx.Matrix(a_np)
        b_m = linx.Matrix(b_np)

        reps = 5 if n >= 512 else 10

        t_np_matmul, _ = timed(lambda: a_np @ b_np, reps=reps)
        t_m_matmul, _ = timed(lambda: a_m @ b_m, reps=reps)
        t_np_add, _ = timed(lambda: a_np + b_np, reps=reps)
        t_m_add, _ = timed(lambda: a_m + b_m, reps=reps)

        ratio = t_np_matmul / t_m_matmul if t_m_matmul > 0 else float("inf")
        rows.append([
            name,
            fmt_time(t_np_matmul),
            fmt_time(t_m_matmul),
            fmt_time(t_np_add),
            fmt_time(t_m_add),
            f"{ratio:.2f}x",
        ])

    print_table(headers, rows)


def main():
    parser = argparse.ArgumentParser(description="Benchmark linx C++ vs NumPy")
    parser.add_argument("--quick", action="store_true", help="Quick mode (smaller matrices, fewer reps)")
    parser.add_argument("--only-inverse", action="store_true", help="Run only inverse benchmarks")
    parser.add_argument(
        "--include-inverse-8192",
        action="store_true",
        help="Include an extreme 8192x8192 Schur inverse benchmark",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  linx vs NumPy — Performance Benchmark")
    print("=" * 72)
    backend = linx.hardware_backend()
    if "BLAS" in backend or "Accelerate" in backend:
        matmul_dispatch = "C++ BLAS; C++ Strassen for square n > 4096"
    else:
        matmul_dispatch = "NumPy BLAS fallback; C++ Strassen for square n > 4096"
    print(f"  Backend: {backend}")
    print(f"  matmul dispatch: {matmul_dispatch}")
    print(f"  linx package: {Path(linx.__file__).resolve()}")
    print(f"  NumPy version: {np.__version__}")
    print(f"  Mode: {'QUICK' if args.quick else 'FULL'}")
    print("=" * 72)

    if args.only_inverse:
        bench_inverse(quick=args.quick, include_8192=args.include_inverse_8192)
        print("\n" + "=" * 72)
        print("  Benchmark complete.")
        print("=" * 72 + "\n")
        return

    bench_matmul(quick=args.quick)
    bench_matmul_classic_strassen(quick=args.quick)
    bench_inverse(quick=args.quick, include_8192=args.include_inverse_8192)
    bench_elementwise(quick=args.quick)
    bench_transpose(quick=args.quick)
    bench_solve(quick=args.quick)
    bench_det_trace_norm(quick=args.quick)
    bench_matrix_class(quick=args.quick)

    print("\n" + "=" * 72)
    print("  Benchmark complete.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
