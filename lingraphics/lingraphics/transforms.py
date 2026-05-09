"""4x4 transform helpers."""

from __future__ import annotations

import math

import numpy as np


def identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def translate(x: float, y: float, z: float) -> np.ndarray:
    mat = identity()
    mat[:3, 3] = [x, y, z]
    return mat


def scale(x: float, y: float | None = None, z: float | None = None) -> np.ndarray:
    if y is None:
        y = x
    if z is None:
        z = x
    return np.diag([x, y, z, 1.0]).astype(np.float64)


def rotate_x(radians: float) -> np.ndarray:
    c = math.cos(radians)
    s = math.sin(radians)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotate_y(radians: float) -> np.ndarray:
    c = math.cos(radians)
    s = math.sin(radians)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotate_z(radians: float) -> np.ndarray:
    c = math.cos(radians)
    s = math.sin(radians)
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
