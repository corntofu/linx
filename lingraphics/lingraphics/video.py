"""Render short lingraphics animations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time

from PIL import Image

from .camera import Camera, orbit_eye
from .mesh import Mesh
from .renderer import Renderer
from .transforms import rotate_x, rotate_y, rotate_z, scale, translate


@dataclass(frozen=True)
class VideoTiming:
    output: Path
    backend: str
    matrix_engine: str
    frame_count: int
    duration_seconds: float
    fps: int
    render_seconds: float
    encode_seconds: float

    @property
    def average_frame_ms(self) -> float:
        return (self.render_seconds / self.frame_count) * 1000.0


def render_tralalero_cube_video(
    output: str | Path = "renders/tralalero_cube.gif",
    backend: str = "auto",
    width: int = 320,
    height: int = 240,
    fps: int = 10,
    duration_seconds: float = 3.0,
    normal_inverse_method: str = "schur",
) -> VideoTiming:
    output_path = Path(output)
    if output_path.suffix.lower() != ".gif":
        output_path = output_path.with_suffix(".gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = max(1, int(round(duration_seconds * fps)))
    renderer = Renderer(
        width=width,
        height=height,
        backend=backend,
        normal_inverse_method=normal_inverse_method,
    )
    camera = Camera.look_at_perspective(
        eye=orbit_eye(210.0, 22.0, 5.8),
        target=(0.0, -0.12, -0.15),
        aspect=width / height,
        fovy_degrees=46.0,
    )
    ground = Mesh.ground(size=7.0, y=-1.16)
    tralalero = Mesh.tralalero_tralala()
    cube = Mesh.cube(size=1.0)

    frames: list[Image.Image] = []
    last_backend = None
    render_start = time.perf_counter()
    for frame_index in range(frame_count):
        t = frame_index / max(1, frame_count - 1)
        phase = 2.0 * math.pi * t

        body_bob = 0.07 * math.sin(phase * 2.0)
        tralalero_model = (
            translate(0.0, body_bob, 0.0)
            @ rotate_y(math.pi + 0.18 * math.sin(phase))
            @ rotate_z(0.09 * math.sin(phase * 2.0))
            @ scale(0.88)
        )

        cube_x = 0.72 * math.sin(phase * 1.4)
        cube_y = -0.42 + 0.52 * abs(math.sin(phase * 1.5))
        cube_z = -1.14 + 0.34 * math.cos(phase * 1.4)
        cube_model = (
            translate(cube_x, cube_y, cube_z)
            @ rotate_y(phase * 3.0)
            @ rotate_x(phase * 4.0)
            @ rotate_z(phase * 2.0)
            @ scale(0.34)
        )

        result = renderer.render_many(
            [
                (ground, scale(1.0)),
                (tralalero, tralalero_model, 1),
                (cube, cube_model, 2),
            ],
            camera=camera,
            backface_culling=False,
        )
        last_backend = result.backend
        frames.append(Image.fromarray(result.to_uint8(), mode="RGB"))
    render_seconds = time.perf_counter() - render_start

    encode_start = time.perf_counter()
    frame_duration_ms = int(round(1000.0 / fps))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )
    encode_seconds = time.perf_counter() - encode_start

    return VideoTiming(
        output=output_path,
        backend=last_backend.name if last_backend is not None else backend,
        matrix_engine=last_backend.matrix_engine if last_backend is not None else "unknown",
        frame_count=frame_count,
        duration_seconds=duration_seconds,
        fps=fps,
        render_seconds=render_seconds,
        encode_seconds=encode_seconds,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render a 3-second Tralalero/cube animation.")
    parser.add_argument("--output", type=Path, default=Path("renders/tralalero_cube.gif"))
    parser.add_argument("--backend", choices=["auto", "numpy", "linx"], default="auto")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--normal-inverse-method", choices=["lu", "schur"], default="schur")
    args = parser.parse_args(argv)

    timing = render_tralalero_cube_video(
        output=args.output,
        backend=args.backend,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration_seconds=args.duration,
        normal_inverse_method=args.normal_inverse_method,
    )
    print(f"wrote {timing.output}")
    print(f"backend={timing.backend}")
    print(f"matrix_engine={timing.matrix_engine}")
    print(f"frames={timing.frame_count}")
    print(f"duration_seconds={timing.duration_seconds:.2f}")
    print(f"fps={timing.fps}")
    print(f"render_seconds={timing.render_seconds:.3f}")
    print(f"average_frame_ms={timing.average_frame_ms:.2f}")
    print(f"encode_seconds={timing.encode_seconds:.3f}")


if __name__ == "__main__":
    main()
