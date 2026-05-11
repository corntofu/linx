"""Camera matrix helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Camera:
    view: np.ndarray
    projection: np.ndarray
    eye: np.ndarray | None = None

    @classmethod
    def look_at_perspective(
        cls,
        eye: tuple[float, float, float] = (2.4, 1.8, 3.4),
        target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        up: tuple[float, float, float] = (0.0, 1.0, 0.0),
        fovy_degrees: float = 55.0,
        aspect: float = 1.0,
        near: float = 0.1,
        far: float = 100.0,
    ) -> "Camera":
        return cls(
            view=look_at(eye, target, up),
            projection=perspective(fovy_degrees, aspect, near, far),
            eye=np.asarray(eye, dtype=np.float64),
        )


def look_at(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> np.ndarray:
    eye_v = np.asarray(eye, dtype=np.float64)
    target_v = np.asarray(target, dtype=np.float64)
    up_v = _normalize(np.asarray(up, dtype=np.float64))

    forward = _normalize(target_v - eye_v)
    side = _normalize(np.cross(forward, up_v))
    true_up = np.cross(side, forward)

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = side
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(side, eye_v)
    view[1, 3] = -np.dot(true_up, eye_v)
    view[2, 3] = np.dot(forward, eye_v)
    return view


def perspective(
    fovy_degrees: float,
    aspect: float,
    near: float,
    far: float,
) -> np.ndarray:
    if near <= 0.0 or far <= near:
        raise ValueError("perspective requires 0 < near < far")
    if aspect <= 0.0:
        raise ValueError("aspect must be positive")

    f = 1.0 / math.tan(math.radians(fovy_degrees) * 0.5)
    mat = np.zeros((4, 4), dtype=np.float64)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2.0 * far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat


def orbit_eye(
    yaw_degrees: float,
    pitch_degrees: float,
    radius: float,
    target: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    target_v = np.asarray(target, dtype=np.float64)
    cp = math.cos(pitch)
    offset = np.array(
        [
            radius * math.sin(yaw) * cp,
            radius * math.sin(pitch),
            radius * math.cos(yaw) * cp,
        ],
        dtype=np.float64,
    )
    eye = target_v + offset
    return (float(eye[0]), float(eye[1]), float(eye[2]))


def _normalize(value: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(value)
    if norm < eps:
        raise ValueError("cannot normalize a near-zero vector")
    return value / norm
