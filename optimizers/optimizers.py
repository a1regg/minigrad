"""Optimization algorithms for parameter updates."""

import numpy as np
from typing import Sequence
from tensor import Tensor


class Optimizer:
    """Base class for gradient-based optimizers."""

    def __init__(self, params: Sequence[Tensor]):
        self._params = list(params)
        if not self._params:
            raise ValueError("Optimizer: no parameters")

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self._params:
            p.zero_grad()


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum.

    Update rule:
        v = momentum * v + grad
        param = param - lr * v

    When momentum=0, reduces to standard SGD: param = param - lr * grad
    """

    def __init__(self, params, lr=0.1, momentum=0.0):
        super().__init__(params)
        if lr <= 0:
            raise ValueError("SGD: lr must be positive")
        if momentum < 0:
            raise ValueError("SGD: momentum >= 0")
        self.lr, self.m = lr, momentum
        self._vel = [np.zeros_like(p.data) for p in self._params]

    def step(self):
        """Apply parameter updates using accumulated gradients."""
        for p, v in zip(self._params, self._vel):
            if p.grad is None:
                continue
            if self.m > 0:
                v[:] = self.m * v + p.grad
                p.data -= self.lr * v
            else:
                p.data -= self.lr * p.grad
