"""Render-time benchmark for NumPy and linx backends."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time

import numpy as np

from .backends import create_backend
from .camera import Camera
from .renderer import Renderer
from .scene import SceneHistory


@dataclass(frozen=True)
class RenderTiming:
    backend: str
    matrix_engine: str
    reps: int
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class SchurTiming:
    backend: str
    method: str
    size: int
    min_block: int
    reps: int
    mean_ms: float
    min_ms: float
    max_ms: float
    residual: float
    matrix_engine: str


def build_benchmark_scene(count: int = 8, complexity: str = "mixed") -> SceneHistory:
    history = SceneHistory()
    if complexity == "basic":
        kind_cycle = ("cube", "pyramid")
    elif complexity == "complex":
        kind_cycle = ("sphere", "torus", "tralalero")
    elif complexity == "mixed":
        kind_cycle = ("cube", "pyramid", "sphere", "torus", "tralalero")
    else:
        raise ValueError("complexity must be 'basic', 'mixed', or 'complex'")

    for index in range(count):
        history.add_shape(kind_cycle[index % len(kind_cycle)])
        row = index // 4
        col = index % 4
        history.move_selected(dx=(col - 1.5) * 0.32, dy=row * 0.12, dz=-(row * 0.22))
        history.rotate_selected(dx=8.0 * row, dy=14.0 * index, dz=5.0 * (index % 3))
    return history


def measure_render_time(
    backend: str,
    width: int = 640,
    height: int = 480,
    reps: int = 20,
    warmup: int = 3,
    objects: int = 8,
    complexity: str = "mixed",
) -> RenderTiming:
    history = build_benchmark_scene(objects, complexity=complexity)
    items = history.render_items(include_ids=True, include_ground=True)
    camera = Camera.look_at_perspective(
        eye=(3.2, 2.4, 5.2),
        target=(0.0, 0.0, 0.0),
        aspect=width / height,
        fovy_degrees=46.0,
    )
    renderer = Renderer(width=width, height=height, backend=backend)

    for _ in range(warmup):
        renderer.render_many(items, camera=camera, backface_culling=False)

    timings = []
    for _ in range(reps):
        start = time.perf_counter()
        result = renderer.render_many(items, camera=camera, backface_culling=False)
        timings.append((time.perf_counter() - start) * 1000.0)

    return RenderTiming(
        backend=result.backend.name,
        matrix_engine=result.backend.matrix_engine,
        reps=reps,
        mean_ms=sum(timings) / len(timings),
        min_ms=min(timings),
        max_ms=max(timings),
    )


def benchmark_backends(
    width: int = 640,
    height: int = 480,
    reps: int = 20,
    warmup: int = 3,
    objects: int = 8,
    complexity: str = "mixed",
) -> list[RenderTiming | tuple[str, str]]:
    results: list[RenderTiming | tuple[str, str]] = []
    for backend in ("numpy", "linx"):
        try:
            results.append(
                measure_render_time(
                    backend=backend,
                    width=width,
                    height=height,
                    reps=reps,
                    warmup=warmup,
                    objects=objects,
                    complexity=complexity,
                )
            )
        except Exception as error:
            results.append((backend, str(error)))
    return results


def format_results(results: list[RenderTiming | tuple[str, str]]) -> str:
    lines = [
        "+---------+----------+----------+----------+----------+",
        "| Backend | Mean ms  |  Min ms  |  Max ms  |  Reps    |",
        "+---------+----------+----------+----------+----------+",
    ]
    for result in results:
        if isinstance(result, RenderTiming):
            lines.append(
                f"| {result.backend:<7} | {result.mean_ms:8.2f} | "
                f"{result.min_ms:8.2f} | {result.max_ms:8.2f} | {result.reps:8d} |"
            )
        else:
            backend, error = result
            lines.append(f"| {backend:<7} | {'error':>8} | {'-':>8} | {'-':>8} | {'-':>8} |")
            lines.append(f"  {backend} error: {error}")
    lines.append("+---------+----------+----------+----------+----------+")
    engines = [
        f"{result.backend}: {result.matrix_engine}"
        for result in results
        if isinstance(result, RenderTiming)
    ]
    if engines:
        lines.append("Matrix engines:")
        lines.extend(f"  {engine}" for engine in engines)
    return "\n".join(lines)


def parse_sizes(value: str) -> list[int]:
    sizes = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        size = int(part)
        if size < 2:
            raise ValueError("Schur benchmark sizes must be >= 2")
        sizes.append(size)
    if not sizes:
        raise ValueError("at least one Schur benchmark size is required")
    return sizes


def make_inverse_input(size: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed + size)
    matrix = rng.normal(0.0, 1.0 / math.sqrt(size), size=(size, size))
    matrix += np.eye(size, dtype=np.float64) * 3.0
    return np.ascontiguousarray(matrix, dtype=np.float64)


def measure_inverse_time(
    backend_name: str,
    method: str,
    matrix: np.ndarray,
    reps: int,
    warmup: int,
    min_block: int,
) -> SchurTiming:
    backend = create_backend(backend_name)
    for _ in range(warmup):
        backend.inverse(matrix, method=method, min_block=min_block)

    timings = []
    inverse = None
    for _ in range(reps):
        start = time.perf_counter()
        inverse = backend.inverse(matrix, method=method, min_block=min_block)
        timings.append((time.perf_counter() - start) * 1000.0)

    identity = np.eye(matrix.shape[0], dtype=np.float64)
    residual = float(np.linalg.norm(matrix @ inverse - identity, ord="fro") / matrix.shape[0])
    return SchurTiming(
        backend=backend.name,
        method=method,
        size=matrix.shape[0],
        min_block=min_block,
        reps=reps,
        mean_ms=sum(timings) / len(timings),
        min_ms=min(timings),
        max_ms=max(timings),
        residual=residual,
        matrix_engine=backend.info.matrix_engine,
    )


def benchmark_schur_inverse(
    sizes: list[int],
    reps: int = 3,
    warmup: int = 1,
    min_block: int = 32,
    backends: tuple[str, ...] = ("numpy", "linx"),
    methods: tuple[str, ...] = ("lu", "schur"),
) -> list[SchurTiming | tuple[str, str, int]]:
    results: list[SchurTiming | tuple[str, str, int]] = []
    for size in sizes:
        matrix = make_inverse_input(size)
        for backend in backends:
            for method in methods:
                try:
                    results.append(
                        measure_inverse_time(
                            backend_name=backend,
                            method=method,
                            matrix=matrix,
                            reps=reps,
                            warmup=warmup,
                            min_block=min_block,
                        )
                    )
                except Exception as error:
                    results.append((backend, f"{method}: {error}", size))
    return results


def format_schur_results(results: list[SchurTiming | tuple[str, str, int]]) -> str:
    baselines = {
        (result.backend, result.size): result.mean_ms
        for result in results
        if isinstance(result, SchurTiming) and result.method == "lu"
    }
    lines = [
        "+---------+--------+-------+----------+----------+----------+----------+------------+",
        "| Backend | Method | Size  | Mean ms  |  Min ms  |  Max ms  | Speedup  | Residual   |",
        "+---------+--------+-------+----------+----------+----------+----------+------------+",
    ]
    for result in results:
        if isinstance(result, SchurTiming):
            baseline = baselines.get((result.backend, result.size))
            speedup = "-"
            if baseline is not None and result.method == "schur" and result.mean_ms > 0.0:
                speedup = f"{baseline / result.mean_ms:.2f}x"
            lines.append(
                f"| {result.backend:<7} | {result.method:<6} | {result.size:5d} | "
                f"{result.mean_ms:8.2f} | {result.min_ms:8.2f} | {result.max_ms:8.2f} | "
                f"{speedup:>8} | {result.residual:10.2e} |"
            )
        else:
            backend, error, size = result
            lines.append(
                f"| {backend:<7} | {'error':<6} | {size:5d} | "
                f"{'error':>8} | {'-':>8} | {'-':>8} | {'-':>8} | {'-':>10} |"
            )
            lines.append(f"  {backend} size={size} error: {error}")
    lines.append("+---------+--------+-------+----------+----------+----------+----------+------------+")
    engines = sorted(
        {
            f"{result.backend}: {result.matrix_engine}"
            for result in results
            if isinstance(result, SchurTiming)
        }
    )
    if engines:
        lines.append("Matrix engines:")
        lines.extend(f"  {engine}" for engine in engines)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark lingraphics render time.")
    parser.add_argument("--mode", choices=["render", "schur", "both"], default="render")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--objects", type=int, default=8)
    parser.add_argument("--complexity", choices=["basic", "mixed", "complex"], default="mixed")
    parser.add_argument(
        "--schur-sizes",
        default="64,128,256",
        help="comma-separated square matrix sizes for Schur inverse benchmark",
    )
    parser.add_argument("--schur-reps", type=int, default=3)
    parser.add_argument("--schur-warmup", type=int, default=1)
    parser.add_argument("--schur-min-block", type=int, default=32)
    args = parser.parse_args(argv)

    if args.mode in {"render", "both"}:
        results = benchmark_backends(
            width=args.width,
            height=args.height,
            reps=args.reps,
            warmup=args.warmup,
            objects=args.objects,
            complexity=args.complexity,
        )
        print(format_results(results))

    if args.mode == "both":
        print()

    if args.mode in {"schur", "both"}:
        schur_results = benchmark_schur_inverse(
            sizes=parse_sizes(args.schur_sizes),
            reps=args.schur_reps,
            warmup=args.schur_warmup,
            min_block=args.schur_min_block,
        )
        print(format_schur_results(schur_results))


if __name__ == "__main__":
    main()
