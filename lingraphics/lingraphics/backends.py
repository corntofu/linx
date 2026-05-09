"""Array backend selection for lingraphics.

The renderer keeps rasterization in NumPy because Python exposes NumPy arrays
as the common interchange type. The linx backend delegates dense matrix
products to the local C++ linx extension, which is built for Apple
Accelerate/ARM64 in this workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import platform
import sys
from typing import Any


def _configure_thread_env() -> None:
    cores = str(os.cpu_count() or 8)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", cores)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", cores)
    os.environ.setdefault("OMP_NUM_THREADS", cores)


_configure_thread_env()

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class BackendInfo:
    name: str
    array_engine: str
    matrix_engine: str
    apple_silicon: bool
    gpu_direct: bool
    note: str


class BaseBackend:
    name = "base"

    def __init__(self) -> None:
        self.xp = np

    @property
    def info(self) -> BackendInfo:
        raise NotImplementedError

    def asarray(self, value: Any, dtype=np.float64) -> Array:
        return np.asarray(value, dtype=dtype)

    def contiguous(self, value: Any, dtype=np.float64) -> Array:
        return np.ascontiguousarray(value, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype=np.float64) -> Array:
        return np.zeros(shape, dtype=dtype)

    def full(self, shape: tuple[int, ...], fill_value: float, dtype=np.float64) -> Array:
        return np.full(shape, fill_value, dtype=dtype)

    def eye(self, n: int, dtype=np.float64) -> Array:
        return np.eye(n, dtype=dtype)

    def matmul(self, lhs: Any, rhs: Any) -> Array:
        return np.matmul(
            np.ascontiguousarray(lhs, dtype=np.float64),
            np.ascontiguousarray(rhs, dtype=np.float64),
        )

    def normalize(self, value: Any, axis: int = -1, eps: float = 1e-12) -> Array:
        arr = np.asarray(value, dtype=np.float64)
        norm = np.linalg.norm(arr, axis=axis, keepdims=True)
        return arr / np.maximum(norm, eps)


class NumpyBackend(BaseBackend):
    name = "numpy"

    @property
    def info(self) -> BackendInfo:
        blas = "unknown BLAS"
        try:
            config = np.__config__.show(mode="dicts")
            blas_info = config.get("Build Dependencies", {}).get("blas", {})
            blas = blas_info.get("name", blas)
        except Exception:
            pass
        return BackendInfo(
            name=self.name,
            array_engine=f"NumPy {np.__version__}",
            matrix_engine=blas,
            apple_silicon=_is_apple_silicon(),
            gpu_direct=False,
            note=(
                "Vectorized NumPy path. Uses the installed BLAS/SIMD build; "
                "NumPy itself does not directly dispatch to the Apple GPU."
            ),
        )


class LinxBackend(BaseBackend):
    name = "linx"

    def __init__(self, linx_module: Any | None = None) -> None:
        super().__init__()
        self._linx = linx_module or _import_linx()

    @property
    def info(self) -> BackendInfo:
        try:
            matrix_engine = self._linx.hardware_backend()
        except Exception:
            matrix_engine = "linx"
        return BackendInfo(
            name=self.name,
            array_engine=f"NumPy {np.__version__}",
            matrix_engine=matrix_engine,
            apple_silicon=_is_apple_silicon(),
            gpu_direct=False,
            note=(
                "Dense transforms use linx C++ kernels. In this workspace linx "
                "reports Apple Accelerate BLAS/LAPACK on arm64."
            ),
        )

    def matmul(self, lhs: Any, rhs: Any) -> Array:
        lhs_arr = np.ascontiguousarray(lhs, dtype=np.float64)
        rhs_arr = np.ascontiguousarray(rhs, dtype=np.float64)
        return self._linx.matmul(lhs_arr, rhs_arr)


def create_backend(name: str | None = None) -> BaseBackend:
    requested = (name or os.environ.get("LINGRAPHICS_BACKEND") or "auto").lower()
    if requested == "auto":
        try:
            return LinxBackend()
        except Exception:
            return NumpyBackend()
    if requested == "numpy":
        return NumpyBackend()
    if requested == "linx":
        return LinxBackend()
    raise ValueError(f"unknown backend {name!r}; expected 'auto', 'numpy', or 'linx'")


def list_backends() -> list[BackendInfo]:
    infos = [NumpyBackend().info]
    try:
        infos.append(LinxBackend().info)
    except Exception:
        pass
    return infos


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _import_linx() -> Any:
    try:
        return importlib.import_module("linx")
    except ImportError as first_error:
        workspace_root = Path(__file__).resolve().parents[2]
        candidate = workspace_root / "linx" / "python"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            try:
                return importlib.import_module("linx")
            except ImportError:
                pass
        raise RuntimeError(
            "linx backend requested, but the linx Python extension was not found. "
            "Build it with `cd ../linx && python3 setup.py build_ext --inplace`, "
            "or run with `--backend numpy`."
        ) from first_error
