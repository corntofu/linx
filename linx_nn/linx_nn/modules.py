"""PyTorch-style neural network modules."""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

from .tensor import Parameter, Tensor


class Module:
    """Base class for neural network layers."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        params = []
        for value in self.__dict__.values():
            params.extend(_collect_parameters(value))
        return params

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()


def _collect_parameters(value):
    if isinstance(value, Parameter):
        return [value]
    if isinstance(value, Module):
        return value.parameters()
    if isinstance(value, dict):
        params = []
        for item in value.values():
            params.extend(_collect_parameters(item))
        return params
    if isinstance(value, (list, tuple, OrderedDict)):
        params = []
        iterable = value.values() if isinstance(value, OrderedDict) else value
        for item in iterable:
            params.extend(_collect_parameters(item))
        return params
    return []


class Linear(Module):
    """Fully connected layer: ``y = x @ weight + bias``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, seed: int | None = None):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        rng = np.random.default_rng(seed)
        limit = math.sqrt(6.0 / in_features)
        self.weight = Parameter(rng.uniform(-limit, limit, size=(in_features, out_features)))
        self.bias = Parameter(np.zeros((1, out_features))) if bias else None

    def forward(self, x) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    def forward(self, x) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x) -> Tensor:
        return x.tanh()


class Sequential(Module):
    """Run modules in order."""

    def __init__(self, *modules: Module):
        self.modules = list(modules)

    def forward(self, x) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x

    def __iter__(self):
        return iter(self.modules)

    def __len__(self):
        return len(self.modules)
