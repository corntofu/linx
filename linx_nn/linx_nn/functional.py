"""Functional helpers for linx_nn."""

from __future__ import annotations

from .tensor import Tensor, _ensure_tensor, mse_loss_tensor


def relu(x) -> Tensor:
    return _ensure_tensor(x).relu()


def sigmoid(x) -> Tensor:
    return _ensure_tensor(x).sigmoid()


def tanh(x) -> Tensor:
    return _ensure_tensor(x).tanh()


def mse_loss(pred, target) -> Tensor:
    return mse_loss_tensor(pred, target)
