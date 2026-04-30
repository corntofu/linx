"""Python interface for the linx C++ linear algebra backend."""

from __future__ import annotations

import numpy as np

from ._linx import (
    condition_number,
    hardware_backend,
    inverse,
    inverse_schur,
    matmul,
    matmul_strassen,
    solve,
)

__all__ = [
    "array",
    "arange",
    "condition_number",
    "eye",
    "hardware_backend",
    "inverse",
    "inverse_schur",
    "matmul",
    "matmul_strassen",
    "ones",
    "solve",
    "zeros",
]


def array(values, dtype=np.float64):
    return np.asarray(values, dtype=dtype)


def zeros(shape, dtype=np.float64):
    return np.zeros(shape, dtype=dtype)


def ones(shape, dtype=np.float64):
    return np.ones(shape, dtype=dtype)


def eye(n, dtype=np.float64):
    return np.eye(n, dtype=dtype)


def arange(*args, dtype=np.float64):
    return np.arange(*args, dtype=dtype)
