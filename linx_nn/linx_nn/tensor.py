"""Small autograd Tensor with linx-backed matrix operations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import numpy as np

try:
    import linx
except Exception:  # pragma: no cover - keeps docs/imports usable before build
    linx = None


_GRAD_ENABLED = True


@contextmanager
def no_grad():
    """Disable graph construction inside the context."""
    global _GRAD_ENABLED
    previous = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = previous


def _array(data) -> np.ndarray:
    if isinstance(data, Tensor):
        data = data.data
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.ascontiguousarray(arr)


def _ensure_tensor(value) -> "Tensor":
    return value if isinstance(value, Tensor) else Tensor(value)


def _linx_binary(lhs: np.ndarray, rhs: np.ndarray, name: str) -> np.ndarray:
    if linx is not None and lhs.ndim == rhs.ndim == 2 and lhs.shape == rhs.shape:
        fn = getattr(linx, name, None)
        if fn is not None:
            return fn(lhs, rhs)
    if name == "add":
        return lhs + rhs
    if name == "subtract":
        return lhs - rhs
    if name == "hadamard":
        return lhs * rhs
    raise ValueError(f"unknown linx binary op: {name}")


def _linx_scalar(data: np.ndarray, scalar: float) -> np.ndarray:
    if linx is not None and data.ndim == 2:
        fn = getattr(linx, "scalar_mul", None)
        if fn is not None:
            return fn(data, scalar)
    return data * scalar


def _linx_matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if linx is not None and lhs.ndim == rhs.ndim == 2:
        return linx.matmul(lhs, rhs)
    return lhs @ rhs


def _linx_transpose(data: np.ndarray) -> np.ndarray:
    if linx is not None and data.ndim == 2:
        fn = getattr(linx, "transpose", None)
        if fn is not None:
            return fn(data)
    return np.ascontiguousarray(data.T)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return np.ascontiguousarray(grad)


class Tensor:
    """A minimal 2D-first autograd tensor.

    The class intentionally mirrors the core PyTorch feel: tensors hold
    ``data`` and ``grad``, operators build a graph, and ``backward()`` fills
    gradients for tensors created with ``requires_grad=True``.
    """

    __array_priority__ = 100

    def __init__(self, data, requires_grad: bool = False, _children: Iterable["Tensor"] = (), _op: str = ""):
        self.data = _array(data)
        self.requires_grad = bool(requires_grad)
        self.grad: np.ndarray | None = None
        self._prev = set(_children) if _GRAD_ENABLED else set()
        self._op = _op
        self._backward = lambda: None

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        out = Tensor(_linx_transpose(self.data), self.requires_grad, (self,), "transpose")

        def _backward():
            if self.requires_grad:
                self._add_grad(_linx_transpose(out.grad))

        out._backward = _backward
        return out

    def __repr__(self):
        return f"Tensor(data={self.data!r}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        other = _ensure_tensor(other)
        data = _linx_binary(self.data, other.data, "add") if self.shape == other.shape else self.data + other.data
        out = Tensor(data, self.requires_grad or other.requires_grad, (self, other), "add")

        def _backward():
            if self.requires_grad:
                self._add_grad(_unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other._add_grad(_unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = _ensure_tensor(other)
        data = _linx_binary(self.data, other.data, "subtract") if self.shape == other.shape else self.data - other.data
        out = Tensor(data, self.requires_grad or other.requires_grad, (self, other), "sub")

        def _backward():
            if self.requires_grad:
                self._add_grad(_unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other._add_grad(_unbroadcast(-out.grad, other.shape))

        out._backward = _backward
        return out

    def __rsub__(self, other):
        return _ensure_tensor(other) - self

    def __mul__(self, other):
        other = _ensure_tensor(other)
        data = _linx_binary(self.data, other.data, "hadamard") if self.shape == other.shape else self.data * other.data
        out = Tensor(data, self.requires_grad or other.requires_grad, (self, other), "mul")

        def _backward():
            if self.requires_grad:
                self._add_grad(_unbroadcast(out.grad * other.data, self.shape))
            if other.requires_grad:
                other._add_grad(_unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = _ensure_tensor(other)
        return self * other.pow(-1.0)

    def __neg__(self):
        return self * -1.0

    def __matmul__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(_linx_matmul(self.data, other.data), self.requires_grad or other.requires_grad, (self, other), "matmul")

        def _backward():
            if self.requires_grad:
                self._add_grad(_linx_matmul(out.grad, _linx_transpose(other.data)))
            if other.requires_grad:
                other._add_grad(_linx_matmul(_linx_transpose(self.data), out.grad))

        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        return _ensure_tensor(other) @ self

    def pow(self, exponent: float):
        data = np.power(self.data, exponent)
        out = Tensor(data, self.requires_grad, (self,), "pow")

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad * exponent * np.power(self.data, exponent - 1.0))

        out._backward = _backward
        return out

    def relu(self):
        data = np.maximum(self.data, 0.0)
        out = Tensor(data, self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad * (self.data > 0.0))

        out._backward = _backward
        return out

    def sigmoid(self):
        data = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(data, self.requires_grad, (self,), "sigmoid")

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad * data * (1.0 - data))

        out._backward = _backward
        return out

    def tanh(self):
        data = np.tanh(self.data)
        out = Tensor(data, self.requires_grad, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad * (1.0 - data * data))

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(data, self.requires_grad, (self,), "sum")

        def _backward():
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self._add_grad(np.ones_like(self.data) * grad)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            denom = self.data.size
        elif isinstance(axis, tuple):
            denom = np.prod([self.data.shape[idx] for idx in axis])
        else:
            denom = self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(denom))

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = _array(grad)

        topo = []
        visited = set()

        def build(node):
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        self.grad = None

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data.reshape(-1)[0])

    def _add_grad(self, grad: np.ndarray):
        grad = np.ascontiguousarray(grad, dtype=np.float64)
        if self.grad is None:
            self.grad = grad
        elif self.grad.shape == grad.shape:
            self.grad = _linx_binary(self.grad, grad, "add")
        else:
            self.grad = self.grad + grad


class Parameter(Tensor):
    """A trainable tensor."""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


def tensor(data, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)
