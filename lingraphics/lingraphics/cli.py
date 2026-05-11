"""Command line demo renderer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .camera import Camera
from .mesh import Mesh
from .renderer import Renderer
from .transforms import identity, rotate_x, rotate_y


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render a small lingraphics demo scene.")
    parser.add_argument("--backend", choices=["auto", "numpy", "linx"], default="auto")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--angle", type=float, default=32.0, help="rotation angle in degrees")
    parser.add_argument("--output", type=Path, default=Path("renders/cube.ppm"))
    parser.add_argument("--mesh", choices=["cube", "pyramid", "sphere", "torus", "tralalero"], default="cube")
    args = parser.parse_args(argv)

    mesh = {
        "cube": Mesh.cube,
        "pyramid": Mesh.pyramid,
        "sphere": Mesh.uv_sphere,
        "torus": Mesh.torus,
        "tralalero": Mesh.tralalero_tralala,
    }[args.mesh]()
    camera = Camera.look_at_perspective(
        eye=(2.6, 1.9, 3.6),
        target=(0.0, 0.0, 0.0),
        aspect=args.width / args.height,
        fovy_degrees=52.0,
    )
    radians = math.radians(args.angle)
    model = rotate_y(radians) @ rotate_x(radians * 0.55)

    renderer = Renderer(width=args.width, height=args.height, backend=args.backend)
    result = renderer.render_many(
        [(Mesh.ground(), identity()), (mesh, model)],
        camera=camera,
        backface_culling=False,
    )
    result.save(str(args.output))

    info = result.backend
    print(f"wrote {args.output}")
    print(f"backend={info.name}")
    print(f"array_engine={info.array_engine}")
    print(f"matrix_engine={info.matrix_engine}")
    print(f"apple_silicon={info.apple_silicon}")
    print(f"gpu_direct={info.gpu_direct}")
    print(info.note)
