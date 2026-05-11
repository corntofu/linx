"""Benchmark inverse algorithms inside the render loop and write CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingraphics.video import render_tralalero_cube_video


MODES = (
    ("numpy-lu", "numpy", "lu"),
    ("linx-lu", "linx", "lu"),
    ("linx-schur", "linx", "schur"),
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="512,1024,2048")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("renders/inverse_task"))
    parser.add_argument("--csv-output", type=Path, default=Path("renders/inverse_task/timing.csv"))
    parser.add_argument("--min-block", type=int, default=32)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for size in _parse_sizes(args.sizes):
        for mode, inverse_backend, inverse_method in MODES:
            output = args.output_dir / f"tralalero_cube_{mode}_n{size}.gif"
            total_start = time.perf_counter()
            timing = render_tralalero_cube_video(
                output=output,
                backend="linx",
                width=args.width,
                height=args.height,
                fps=args.fps,
                duration_seconds=args.duration,
                normal_inverse_method="schur",
                inverse_task_backend=inverse_backend,
                inverse_task_method=inverse_method,
                inverse_task_size=size,
                inverse_task_min_block=args.min_block,
            )
            total_seconds = time.perf_counter() - total_start
            rows.append(
                {
                    "mode": mode,
                    "inverse_backend": inverse_backend,
                    "inverse_method": inverse_method,
                    "inverse_size": size,
                    "width": args.width,
                    "height": args.height,
                    "duration_seconds": f"{timing.duration_seconds:.2f}",
                    "fps": timing.fps,
                    "frames": timing.frame_count,
                    "render_seconds": f"{timing.render_seconds:.3f}",
                    "inverse_seconds": f"{timing.inverse_seconds:.3f}",
                    "inverse_ms_per_frame": f"{(timing.inverse_seconds / timing.frame_count) * 1000.0:.2f}",
                    "encode_seconds": f"{timing.encode_seconds:.3f}",
                    "total_seconds": f"{total_seconds:.3f}",
                    "matrix_engine": timing.matrix_engine,
                    "output": str(timing.output),
                }
            )

    fieldnames = [
        "mode",
        "inverse_backend",
        "inverse_method",
        "inverse_size",
        "width",
        "height",
        "duration_seconds",
        "fps",
        "frames",
        "render_seconds",
        "inverse_seconds",
        "inverse_ms_per_frame",
        "encode_seconds",
        "total_seconds",
        "matrix_engine",
        "output",
    ]
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _parse_sizes(value: str) -> list[int]:
    sizes = []
    for part in value.split(","):
        part = part.strip()
        if part:
            sizes.append(int(part))
    if not sizes:
        raise ValueError("at least one size is required")
    return sizes


if __name__ == "__main__":
    main()
