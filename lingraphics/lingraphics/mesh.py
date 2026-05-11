"""Triangle mesh primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math

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

    @classmethod
    def ground(cls, size: float = 7.0, y: float = -0.86) -> "Mesh":
        half = size * 0.5
        vertices = np.array(
            [
                [-half, y, -half],
                [half, y, -half],
                [half, y, half],
                [-half, y, half],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)
        colors = np.array(
            [
                [0.20, 0.22, 0.24],
                [0.17, 0.19, 0.21],
            ],
            dtype=np.float64,
        )
        return cls(vertices=vertices, faces=faces, face_colors=colors)

    @classmethod
    def uv_sphere(cls, radius: float = 0.82, rings: int = 12, segments: int = 24) -> "Mesh":
        rings = max(4, int(rings))
        segments = max(8, int(segments))
        vertices = []
        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            y = radius * math.cos(phi)
            radial = radius * math.sin(phi)
            for segment in range(segments):
                theta = 2.0 * math.pi * segment / segments
                vertices.append(
                    [
                        radial * math.cos(theta),
                        y,
                        radial * math.sin(theta),
                    ]
                )
        vertices_arr = np.asarray(vertices, dtype=np.float64)

        raw_faces = []
        for ring in range(rings):
            for segment in range(segments):
                a = ring * segments + segment
                b = ring * segments + ((segment + 1) % segments)
                c = (ring + 1) * segments + segment
                d = (ring + 1) * segments + ((segment + 1) % segments)
                raw_faces.append([a, c, b])
                raw_faces.append([b, c, d])
        faces, colors = _oriented_faces(vertices_arr, raw_faces, _sphere_color)
        return cls(vertices=vertices_arr, faces=faces, face_colors=colors)

    @classmethod
    def torus(
        cls,
        major_radius: float = 0.72,
        minor_radius: float = 0.25,
        major_segments: int = 28,
        minor_segments: int = 10,
    ) -> "Mesh":
        major_segments = max(8, int(major_segments))
        minor_segments = max(6, int(minor_segments))
        vertices = []
        for major in range(major_segments):
            theta = 2.0 * math.pi * major / major_segments
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            for minor in range(minor_segments):
                phi = 2.0 * math.pi * minor / minor_segments
                ring = major_radius + minor_radius * math.cos(phi)
                vertices.append(
                    [
                        ring * cos_theta,
                        minor_radius * math.sin(phi),
                        ring * sin_theta,
                    ]
                )
        vertices_arr = np.asarray(vertices, dtype=np.float64)

        faces = []
        colors = []
        for major in range(major_segments):
            for minor in range(minor_segments):
                a = major * minor_segments + minor
                b = major * minor_segments + ((minor + 1) % minor_segments)
                c = ((major + 1) % major_segments) * minor_segments + minor
                d = ((major + 1) % major_segments) * minor_segments + ((minor + 1) % minor_segments)
                for face in ([a, c, b], [b, c, d]):
                    oriented = _orient_torus_face(vertices_arr, face, major_radius)
                    faces.append(oriented)
                    center = vertices_arr[oriented].mean(axis=0)
                    colors.append(_torus_color(center))
        return cls(
            vertices=vertices_arr,
            faces=np.asarray(faces, dtype=np.int64),
            face_colors=np.asarray(colors, dtype=np.float64),
        )

    @classmethod
    def tralalero_tralala(cls) -> "Mesh":
        blue = np.array([0.18, 0.46, 0.88], dtype=np.float64)
        light_blue = np.array([0.30, 0.68, 0.96], dtype=np.float64)
        fin_blue = np.array([0.10, 0.30, 0.66], dtype=np.float64)
        shoe_blue = np.array([0.08, 0.78, 0.95], dtype=np.float64)
        sole = np.array([0.92, 0.95, 0.98], dtype=np.float64)
        eye_white = np.array([0.97, 0.98, 1.00], dtype=np.float64)
        eye_dark = np.array([0.02, 0.03, 0.05], dtype=np.float64)

        parts = [
            _colored(_transform_mesh(cls.uv_sphere(radius=1.0, rings=10, segments=20), _trs((0.0, 0.0, 0.0), (0.54, 0.33, 1.22))), blue),
            _colored(_transform_mesh(cls.uv_sphere(radius=1.0, rings=8, segments=18), _trs((0.0, 0.02, -1.12), (0.43, 0.28, 0.42))), light_blue),
            _colored(_fin_mesh("dorsal"), fin_blue),
            _colored(_fin_mesh("left"), fin_blue),
            _colored(_fin_mesh("right"), fin_blue),
            _colored(_fin_mesh("tail_top"), fin_blue),
            _colored(_fin_mesh("tail_bottom"), fin_blue),
            _colored(_transform_mesh(cls.uv_sphere(radius=1.0, rings=6, segments=12), _trs((-0.18, 0.16, -1.43), (0.07, 0.07, 0.04))), eye_white),
            _colored(_transform_mesh(cls.uv_sphere(radius=1.0, rings=6, segments=12), _trs((0.18, 0.16, -1.43), (0.07, 0.07, 0.04))), eye_white),
            _colored(_transform_mesh(cls.uv_sphere(radius=1.0, rings=4, segments=10), _trs((-0.18, 0.16, -1.48), (0.03, 0.03, 0.02))), eye_dark),
            _colored(_transform_mesh(cls.uv_sphere(radius=1.0, rings=4, segments=10), _trs((0.18, 0.16, -1.48), (0.03, 0.03, 0.02))), eye_dark),
        ]

        leg_positions = [(-0.33, -0.63, -0.34), (0.33, -0.63, -0.34), (0.0, -0.63, 0.34)]
        for x, y, z in leg_positions:
            parts.append(
                _colored(
                    _transform_mesh(cls.uv_sphere(radius=1.0, rings=6, segments=12), _trs((x, y, z), (0.11, 0.46, 0.13))),
                    blue,
                )
            )
            parts.append(_colored(_transform_mesh(cls.cube(size=1.0), _trs((x, y - 0.48, z - 0.02), (0.30, 0.11, 0.42))), shoe_blue))
            parts.append(_colored(_transform_mesh(cls.cube(size=1.0), _trs((x, y - 0.56, z - 0.02), (0.34, 0.06, 0.46))), sole))

        return _merge_meshes(parts)


def _oriented_faces(vertices: np.ndarray, raw_faces, color_fn):
    faces = []
    colors = []
    for face in raw_faces:
        tri = vertices[np.asarray(face, dtype=np.int64)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if np.linalg.norm(normal) < 1e-12:
            continue
        center = tri.mean(axis=0)
        if np.dot(normal, center) < 0.0:
            face = [face[0], face[2], face[1]]
            tri = vertices[np.asarray(face, dtype=np.int64)]
            center = tri.mean(axis=0)
        faces.append(face)
        colors.append(color_fn(center))
    return np.asarray(faces, dtype=np.int64), np.asarray(colors, dtype=np.float64)


def _orient_torus_face(vertices: np.ndarray, face, major_radius: float):
    tri = vertices[np.asarray(face, dtype=np.int64)]
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    center = tri.mean(axis=0)
    theta = math.atan2(center[2], center[0])
    tube_center = np.array(
        [major_radius * math.cos(theta), 0.0, major_radius * math.sin(theta)],
        dtype=np.float64,
    )
    outward = center - tube_center
    if np.dot(normal, outward) < 0.0:
        return [face[0], face[2], face[1]]
    return face


def _sphere_color(center: np.ndarray) -> np.ndarray:
    t = np.clip((center[1] + 0.82) / 1.64, 0.0, 1.0)
    cool = np.array([0.32, 0.56, 0.92], dtype=np.float64)
    warm = np.array([0.96, 0.48, 0.30], dtype=np.float64)
    return cool * (1.0 - t) + warm * t


def _torus_color(center: np.ndarray) -> np.ndarray:
    t = 0.5 + 0.5 * math.sin(3.0 * math.atan2(center[2], center[0]))
    teal = np.array([0.22, 0.72, 0.78], dtype=np.float64)
    violet = np.array([0.68, 0.40, 0.88], dtype=np.float64)
    return teal * (1.0 - t) + violet * t


def _trs(translation, factors) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = factors[0]
    matrix[1, 1] = factors[1]
    matrix[2, 2] = factors[2]
    matrix[:3, 3] = translation
    return matrix


def _transform_mesh(mesh: Mesh, matrix: np.ndarray) -> Mesh:
    vertices_h = np.concatenate([mesh.vertices, np.ones((mesh.vertices.shape[0], 1), dtype=np.float64)], axis=1)
    vertices = (vertices_h @ matrix.T)[:, :3]
    return Mesh(vertices=vertices, faces=mesh.faces, face_colors=mesh.face_colors)


def _colored(mesh: Mesh, color: np.ndarray) -> Mesh:
    colors = np.tile(np.asarray(color, dtype=np.float64), (mesh.faces.shape[0], 1))
    return Mesh(vertices=mesh.vertices, faces=mesh.faces, face_colors=colors)


def _merge_meshes(meshes) -> Mesh:
    vertices = []
    faces = []
    colors = []
    offset = 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        colors.append(mesh.face_colors)
        offset += mesh.vertices.shape[0]
    return Mesh(
        vertices=np.vstack(vertices),
        faces=np.vstack(faces),
        face_colors=np.vstack(colors),
    )


def _fin_mesh(kind: str) -> Mesh:
    if kind == "dorsal":
        vertices = np.array(
            [
                [-0.08, 0.28, -0.15],
                [0.08, 0.28, -0.15],
                [0.00, 0.72, 0.10],
                [0.00, 0.30, 0.42],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3], [1, 3, 2], [0, 3, 1]], dtype=np.int64)
    elif kind == "left":
        vertices = np.array(
            [
                [-0.42, -0.04, -0.18],
                [-0.92, -0.20, -0.05],
                [-0.46, -0.12, 0.22],
                [-0.38, 0.04, 0.02],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 3], [1, 2, 3], [0, 3, 2], [0, 2, 1]], dtype=np.int64)
    elif kind == "right":
        vertices = np.array(
            [
                [0.42, -0.04, -0.18],
                [0.92, -0.20, -0.05],
                [0.46, -0.12, 0.22],
                [0.38, 0.04, 0.02],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 3, 1], [1, 3, 2], [0, 2, 3], [0, 1, 2]], dtype=np.int64)
    elif kind == "tail_top":
        vertices = np.array(
            [
                [-0.05, 0.02, 1.05],
                [0.05, 0.02, 1.05],
                [0.00, 0.54, 1.55],
                [0.00, 0.02, 1.68],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3], [1, 3, 2], [0, 3, 1]], dtype=np.int64)
    elif kind == "tail_bottom":
        vertices = np.array(
            [
                [-0.05, -0.02, 1.05],
                [0.05, -0.02, 1.05],
                [0.00, -0.54, 1.55],
                [0.00, -0.02, 1.68],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 2, 1], [0, 3, 2], [1, 2, 3], [0, 1, 3]], dtype=np.int64)
    else:
        raise ValueError(f"unknown fin kind {kind!r}")
    colors = np.ones((faces.shape[0], 3), dtype=np.float64)
    return Mesh(vertices=vertices, faces=faces, face_colors=colors)
