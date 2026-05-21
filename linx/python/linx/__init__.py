"""Python interface for the linx C++ linear algebra backend."""

from __future__ import annotations

import numpy as np

from . import _linx as _backend

condition_number = _backend.condition_number
det = _backend.det
frobenius_norm = _backend.frobenius_norm
hardware_backend = _backend.hardware_backend
inverse = _backend.inverse
inverse_schur = _backend.inverse_schur
inverse_schur_strassen = _backend.inverse_schur_strassen
least_squares = _backend.least_squares
matmul_strassen = _backend.matmul_strassen
residual_norm = _backend.residual_norm
solve = _backend.solve
trace = _backend.trace

_cpp_matmul = _backend.matmul
_BACKEND_NAME = hardware_backend()
_HAS_FAST_CPP_MATMUL = "BLAS" in _BACKEND_NAME or "Accelerate" in _BACKEND_NAME
_STRASSEN_MIN_N = 4096


def add(lhs, rhs):
    if hasattr(_backend, "add"):
        return _backend.add(lhs, rhs)
    return _as_array(lhs) + _as_array(rhs)


def subtract(lhs, rhs):
    if hasattr(_backend, "subtract"):
        return _backend.subtract(lhs, rhs)
    return _as_array(lhs) - _as_array(rhs)


def hadamard(lhs, rhs):
    if hasattr(_backend, "hadamard"):
        return _backend.hadamard(lhs, rhs)
    return _as_array(lhs) * _as_array(rhs)


def scalar_mul(matrix, scalar):
    if hasattr(_backend, "scalar_mul"):
        return _backend.scalar_mul(matrix, scalar)
    return _as_array(matrix) * float(scalar)


def transpose(matrix):
    if hasattr(_backend, "transpose"):
        return _backend.transpose(matrix)
    return _as_array(matrix).T.copy()


def neg(matrix):
    if hasattr(_backend, "neg"):
        return _backend.neg(matrix)
    return -_as_array(matrix)


def matmul(lhs, rhs):
    """Multiply two matrices.

    The C++ extension uses BLAS when it is available. If the extension was
    built without BLAS, use NumPy's BLAS-backed matmul instead of the much
    slower portable C++ fallback exposed by the extension.
    """
    lhs_arr = _as_array(lhs)
    rhs_arr = _as_array(rhs)
    use_auto_strassen = (
        lhs_arr.ndim == 2
        and rhs_arr.ndim == 2
        and lhs_arr.shape[0] == lhs_arr.shape[1] == rhs_arr.shape[0] == rhs_arr.shape[1]
        and lhs_arr.shape[0] > _STRASSEN_MIN_N
    )
    if _HAS_FAST_CPP_MATMUL or use_auto_strassen:
        return _cpp_matmul(lhs_arr, rhs_arr)
    return np.matmul(lhs_arr, rhs_arr)

__all__ = [
    "Matrix",
    "add",
    "array",
    "arange",
    "condition_number",
    "det",
    "eye",
    "frobenius_norm",
    "hadamard",
    "hardware_backend",
    "inverse",
    "inverse_schur",
    "inverse_schur_strassen",
    "least_squares",
    "matmul",
    "matmul_strassen",
    "neg",
    "ones",
    "residual_norm",
    "scalar_mul",
    "solve",
    "subtract",
    "trace",
    "transpose",
    "zeros",
]


# ── ndarray ↔ linx.Matrix helpers ────────────────────────────────────────────

def _as_array(obj):
    """Ensure `obj` is a numpy float64 2D contiguous C-order array."""
    if isinstance(obj, Matrix):
        return obj._data
    arr = np.asarray(obj, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.ascontiguousarray(arr)


# ── Matrix class ─────────────────────────────────────────────────────────────

class Matrix:
    """linx Matrix – wraps a NumPy float64 2D array and delegates **every
    arithmetic operation** to the C++ linx backend (SIMD + BLAS + LAPACK).

    - ``+`` / ``-`` → vDSP / SIMD element‑wise
    - ``*`` (scalar) → vDSP / SIMD scalar multiply
    - ``*`` (matrix) → element‑wise Hadamard (vDSP / SIMD)
    - ``@`` → matmul (BLAS `cblas_dgemm` or AVX2/NEON + threads)
    - ``-`` (unary) → vDSP negate
    - ``/`` (scalar) → scalar multiply by reciprocal
    - ``.T`` → vDSP transposition
    - ``.inv()`` → BLAS/LAPACK for ≤512, Schur complement for larger
    """

    __slots__ = ("_data",)

    def __init__(self, data, dtype=np.float64):
        if isinstance(data, Matrix):
            data = data._data
        arr = np.asarray(data, dtype=dtype)
        if arr.ndim > 2:
            raise ValueError("linx.Matrix only supports 1‑D or 2‑D arrays")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self._data = np.ascontiguousarray(arr)  # C‑order for linx backend

    # -- properties ------------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """Return the underlying NumPy array (writable view)."""
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def rows(self):
        return self._data.shape[0]

    @property
    def cols(self):
        return self._data.shape[1]

    @property
    def T(self):
        return Matrix(transpose(self._data))

    # -- dunder methods --------------------------------------------------------

    def __repr__(self):
        return f"Matrix(\n{self._data!r}\n)"

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __add__(self, other):
        return Matrix(add(self._data, _as_array(other)))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return Matrix(subtract(self._data, _as_array(other)))

    def __rsub__(self, other):
        return Matrix(subtract(_as_array(other), self._data))

    def __mul__(self, other):
        if np.isscalar(other):
            return Matrix(scalar_mul(self._data, float(other)))
        return Matrix(hadamard(self._data, _as_array(other)))

    def __rmul__(self, other):
        if np.isscalar(other):
            return Matrix(scalar_mul(self._data, float(other)))
        return Matrix(hadamard(_as_array(other), self._data))

    def __matmul__(self, other):
        return Matrix(matmul(self._data, _as_array(other)))

    def __rmatmul__(self, other):
        return Matrix(matmul(_as_array(other), self._data))

    def __truediv__(self, scalar):
        return Matrix(scalar_mul(self._data, 1.0 / float(scalar)))

    def __neg__(self):
        return Matrix(neg(self._data))

    # -- linear algebra --------------------------------------------------------

    def inv(self, method="schur", min_block=32, regularization=0.0, eps=1e-12):
        """Inverse matrix using C++ linx backend."""
        return Matrix(inverse(self._data, method=method, min_block=min_block,
                              regularization=regularization, eps=eps))

    def inv_schur_strassen(self, min_block=1024, strassen_threshold=4096, eps=1e-12):
        """Inverse matrix using Schur complement with Strassen block multiplies."""
        return Matrix(inverse_schur_strassen(self._data, min_block=min_block,
                                             strassen_threshold=strassen_threshold,
                                             eps=eps))

    def solve(self, b):
        """Solve A @ X = B."""
        return Matrix(solve(self._data, _as_array(b)))

    def least_squares(self, b, eps=1e-12):
        """Solve min ||A @ X - B||_2."""
        return Matrix(least_squares(self._data, _as_array(b), eps=eps))

    def condition_number(self, eps=1e-12):
        """Frobenius condition number estimate."""
        return condition_number(self._data, eps=eps)

    def frobenius_norm(self):
        return frobenius_norm(self._data)

    def residual_norm(self, inv_mat=None):
        if inv_mat is None:
            inv_mat = self.inv()
        return residual_norm(self._data, _as_array(inv_mat))

    # -- factory methods -------------------------------------------------------

    @staticmethod
    def zeros(shape, dtype=np.float64):
        return Matrix(np.zeros(shape, dtype=dtype))

    @staticmethod
    def ones(shape, dtype=np.float64):
        return Matrix(np.ones(shape, dtype=dtype))

    @staticmethod
    def eye(n, dtype=np.float64):
        return Matrix(np.eye(n, dtype=dtype))

    @staticmethod
    def arange(*args, dtype=np.float64):
        return Matrix(np.arange(*args, dtype=dtype))


# ── convenience helpers (backward‑compatible) ────────────────────────────────

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
