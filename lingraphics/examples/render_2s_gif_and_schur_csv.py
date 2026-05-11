"""Render a 2-second Tralalero/cube GIF and write Schur benchmark CSV.

Run from the lingraphics project root:

    /opt/anaconda3/bin/python3 examples/render_2s_gif_and_schur_csv.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingraphics.video import render_tralalero_cube_video
from lingraphics.benchmark import SchurTiming, benchmark_schur_inverse, parse_sizes


def write_schur_csv(path: Path, results) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baselines = {
        (result.backend, result.size): result.mean_ms
        for result in results
        if isinstance(result, SchurTiming) and result.method == "lu"
    }
    fieldnames = [
        "backend",
        "method",
        "size",
        "mean_ms",
        "min_ms",
        "max_ms",
        "speedup_vs_lu",
        "residual",
        "matrix_engine",
    ]
    rows = []
    for result in results:
        if not isinstance(result, SchurTiming):
            backend, error, size = result
            rows.append(
                {
                    "backend": backend,
                    "method": "error",
                    "size": size,
                    "mean_ms": "error",
                    "min_ms": "",
                    "max_ms": "",
                    "speedup_vs_lu": "",
                    "residual": "",
                    "matrix_engine": error,
                }
            )
            continue
        speedup = ""
        baseline = baselines.get((result.backend, result.size))
        if result.method == "schur" and baseline is not None and result.mean_ms > 0.0:
            speedup = f"{baseline / result.mean_ms:.3f}"
        rows.append(
            {
                "backend": result.backend,
                "method": result.method,
                "size": result.size,
                "mean_ms": f"{result.mean_ms:.3f}",
                "min_ms": f"{result.min_ms:.3f}",
                "max_ms": f"{result.max_ms:.3f}",
                "speedup_vs_lu": speedup,
                "residual": f"{result.residual:.3e}",
                "matrix_engine": result.matrix_engine,
            }
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif-output", type=Path, default=Path("renders/tralalero_cube_2s.gif"))
    parser.add_argument("--csv-output", type=Path, default=Path("renders/schur_512_1024_2048.csv"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--backend", choices=["auto", "numpy", "linx"], default="linx")
    parser.add_argument("--sizes", default="512,1024,2048")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--min-block", type=int, default=32)
    args = parser.parse_args(argv)

    timing = render_tralalero_cube_video(
        output=args.gif_output,
        backend=args.backend,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration_seconds=args.duration,
        normal_inverse_method="schur",
    )
    print(
        "gif,"
        f"output={timing.output},"
        f"backend={timing.backend},"
        f"frames={timing.frame_count},"
        f"duration_seconds={timing.duration_seconds:.2f},"
        f"render_seconds={timing.render_seconds:.3f},"
        f"encode_seconds={timing.encode_seconds:.3f}"
    )

    results = benchmark_schur_inverse(
        sizes=parse_sizes(args.sizes),
        reps=args.reps,
        warmup=args.warmup,
        min_block=args.min_block,
        backends=("numpy", "linx"),
        methods=("lu", "schur"),
    )
    write_schur_csv(args.csv_output, results)


if __name__ == "__main__":
    main()
