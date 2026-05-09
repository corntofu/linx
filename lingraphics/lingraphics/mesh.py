"""Triangle mesh primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    face_colors: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        colors = np.asarray(self.face_colors, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (M, 3)")
        if colors.shape != (faces.shape[0], 3):
            raise ValueError("face_colors must have shape (M, 3)")
        object.__setattr__(self, "vertices", np.ascontiguousarray(vertices))
        object.__setattr__(self, "faces", np.ascontiguousarray(faces))
        object.__setattr__(self, "face_colors", np.clip(np.ascontiguousarray(colors), 0.0, 1.0))

    @classmethod
    def cube(cls, size: float = 1.6) -> "Mesh":
        half = size * 0.5
        vertices = np.array(
            [
                [-half, -half, -half],
                [half, -half, -half],
                [half, half, -half],
                [-half, half, -half],
                [-half, -half, half],
                [half, -half, half],
                [half, half, half],
                [-half, half, half],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [3, 2, 6],
                [3, 6, 7],
                [1, 5, 6],
                [1, 6, 2],
                [0, 3, 7],
                [0, 7, 4],
            ],
            dtype=np.int64,
        )
        colors = np.array(
            [
                [0.80, 0.23, 0.22],
                [0.80, 0.23, 0.22],
                [0.22, 0.52, 0.93],
                [0.22, 0.52, 0.93],
                [0.24, 0.70, 0.42],
                [0.24, 0.70, 0.42],
                [0.95, 0.72, 0.20],
                [0.95, 0.72, 0.20],
                [0.75, 0.35, 0.88],
                [0.75, 0.35, 0.88],
                [0.24, 0.76, 0.78],
                [0.24, 0.76, 0.78],
            ],
            dtype=np.float64,
        )
        return cls(vertices=vertices, faces=faces, face_colors=colors)

    @classmethod
    def pyramid(cls, size: float = 1.8, height: float = 1.7) -> "Mesh":
        half = size * 0.5
        vertices = np.array(
            [
                [-half, -half, -half],
                [half, -half, -half],
                [half, -half, half],
                [-half, -half, half],
                [0.0, height * 0.5, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
            ],
            dtype=np.int64,
        )
        colors = np.array(
            [
                [0.25, 0.28, 0.33],
                [0.25, 0.28, 0.33],
                [0.92, 0.38, 0.24],
                [0.22, 0.57, 0.88],
                [0.91, 0.69, 0.20],
                [0.28, 0.72, 0.45],
            ],
            dtype=np.float64,
        )
        return cls(vertices=vertices, faces=faces, face_colors=colors)
