"""PyTorch-style neural network modules."""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

from .tensor import Parameter, Tensor, _ensure_tensor, _linx_matmul, _linx_transpose, linear


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
        return linear(x, self.weight, self.bias)


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

    def train_mse_step(self, x, target, optimizer):
        """Fused ``forward + MSE backward + optimizer.step`` for dense MLPs.

        This path skips general graph construction and is intended for hot CPU
        training loops made from ``Linear`` and elementwise activation modules.
        It overwrites parameter gradients for the current batch.
        """
        loss = self.backward_mse(x, target)
        optimizer.step()
        return loss

    def backward_mse(self, x, target) -> Tensor:
        """Fused MSE backward pass for ``Linear``/activation sequences."""
        x = _ensure_tensor(x)
        target = _ensure_tensor(target)
        activations = [x.data]
        layer_inputs = []
        out = x.data

        for module in self.modules:
            layer_inputs.append(out)
            if isinstance(module, Linear):
                out = _linx_matmul(out, module.weight.data)
                if module.bias is not None:
                    out = out + module.bias.data
            elif isinstance(module, ReLU):
                out = np.maximum(out, 0.0)
            elif isinstance(module, Sigmoid):
                out = 1.0 / (1.0 + np.exp(-out))
            elif isinstance(module, Tanh):
                out = np.tanh(out)
            else:
                raise TypeError(f"fused MSE path does not support {type(module).__name__}")
            activations.append(out)

        diff = out - target.data
        flat = diff.reshape(-1)
        loss = Tensor([[float(np.dot(flat, flat) / diff.size)]])
        grad = diff
        grad *= 2.0 / diff.size

        for idx in range(len(self.modules) - 1, -1, -1):
            module = self.modules[idx]
            layer_input = layer_inputs[idx]

            if isinstance(module, Linear):
                module.weight.grad = _linx_matmul(_linx_transpose(layer_input), grad)
                if module.bias is not None:
                    module.bias.grad = np.ascontiguousarray(grad.sum(axis=0, keepdims=True))
                if idx > 0 or x.requires_grad:
                    grad = _linx_matmul(grad, _linx_transpose(module.weight.data))
                continue

            if isinstance(module, ReLU):
                grad = grad * (layer_input > 0.0)
            elif isinstance(module, Sigmoid):
                activated = activations[idx + 1]
                grad = grad * activated * (1.0 - activated)
            elif isinstance(module, Tanh):
                activated = activations[idx + 1]
                grad = grad * (1.0 - activated * activated)

        if x.requires_grad:
            x.grad = np.ascontiguousarray(grad)
        return loss

    def __iter__(self):
        return iter(self.modules)

    def __len__(self):
        return len(self.modules)
