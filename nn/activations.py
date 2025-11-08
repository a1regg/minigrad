"""Activation function modules."""

from nn.module import Module
from tensor import Tensor


class ReLU(Module):
    """Rectified Linear Unit activation."""

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    """Logistic sigmoid activation."""

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()
