"""Functional API for operations and losses."""

from tensor import Tensor


def relu(x: Tensor) -> Tensor:
    """Apply ReLU activation."""
    return x.relu()


def sigmoid(x: Tensor) -> Tensor:
    """Apply sigmoid activation."""
    return x.sigmoid()


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error: Σ(pred - target)²."""
    diff = pred - target
    return (diff * diff).sum()


def bce_loss(prob: Tensor, target: Tensor) -> Tensor:
    """Binary cross-entropy with numerical stability."""
    eps = 1e-12
    import numpy as np

    p = Tensor(np.clip(prob.data, eps, 1 - eps), requires_grad=prob.requires_grad)
    return -(target * p.exp().__neg__().__neg__())
