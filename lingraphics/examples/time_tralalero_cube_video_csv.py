"""Render the 3-second Tralalero/cube animation and print timings as CSV.

Run from the lingraphics project root:

    /opt/anaconda3/bin/python3 examples/time_tralalero_cube_video_csv.py
"""

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
    ("numpy", "numpy", "lu"),
    ("linx", "linx", "lu"),
    ("linx-schur", "linx", "schur"),
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=Path("renders/video_compare"))
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, backend, inverse_method in MODES:
        output = args.output_dir / f"tralalero_cube_{label}.gif"
        total_start = time.perf_counter()
        timing = render_tralalero_cube_video(
            output=output,
            backend=backend,
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration_seconds=args.duration,
            normal_inverse_method=inverse_method,
        )
        total_seconds = time.perf_counter() - total_start
        rows.append(
            {
                "mode": label,
                "backend": backend,
                "normal_inverse_method": inverse_method,
                "matrix_engine": timing.matrix_engine,
                "width": args.width,
                "height": args.height,
                "duration_seconds": f"{timing.duration_seconds:.2f}",
                "fps": timing.fps,
                "frames": timing.frame_count,
                "render_seconds": f"{timing.render_seconds:.3f}",
                "average_frame_ms": f"{timing.average_frame_ms:.2f}",
                "encode_seconds": f"{timing.encode_seconds:.3f}",
                "total_seconds": f"{total_seconds:.3f}",
                "output": str(timing.output),
            }
        )

    fieldnames = [
        "mode",
        "backend",
        "normal_inverse_method",
        "matrix_engine",
        "width",
        "height",
        "duration_seconds",
        "fps",
        "frames",
        "render_seconds",
        "average_frame_ms",
        "encode_seconds",
        "total_seconds",
        "output",
    ]

    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
