"""Neural network layer implementations."""

import numpy as np
from minigrad.tensor import Tensor
from minigrad.nn.module import Module


class Linear(Module):
    """Fully connected linear layer: y = xW + b.

    Weights initialized using Xavier/Glorot uniform with bound √(6/in_features).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Linear: in/out must be positive")
        bound = (6.0 / in_features) ** 0.5
        W = np.random.uniform(-bound, bound, size=(in_features, out_features))
        self._W = Tensor(W, requires_grad=True)
        self._use_bias = bias
        self._b = (
            Tensor(np.zeros((1, out_features)), requires_grad=True) if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply affine transformation: x @ W + b."""
        out = x.matmul(self._W)
        if self._use_bias:
            out = out + self._b
        return out

    @property
    def weight(self):
        """Read-only access to weight tensor."""
        return self._W

    @property
    def bias(self):
        """Read-only access to bias tensor."""
        return self._b
