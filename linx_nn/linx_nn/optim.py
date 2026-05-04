"""Optimizers."""

from __future__ import annotations


class SGD:
    """Stochastic gradient descent with optional momentum and weight decay."""

    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0, weight_decay: float = 0.0):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if momentum < 0:
            raise ValueError("momentum must be non-negative")
        self.params = list(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self._velocity = [None for _ in self.params]

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    def step(self):
        for idx, param in enumerate(self.params):
            grad = param.grad
            if grad is None:
                continue

            if self.momentum:
                velocity = self._velocity[idx]
                if velocity is None:
                    velocity = grad.copy()
                    self._velocity[idx] = velocity
                else:
                    velocity *= self.momentum
                    velocity += grad
                if self.weight_decay:
                    velocity += self.weight_decay * param.data
                param.data -= self.lr * velocity
                continue

            if self.weight_decay:
                param.data *= 1.0 - self.lr * self.weight_decay
            param.data -= self.lr * grad
