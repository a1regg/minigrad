"""Base module class for neural network components."""

from typing import List, Iterator
from minigrad.tensor import Tensor


class Module:
    """Base class for all neural network modules.

    Provides parameter collection, gradient zeroing, and train/eval mode switching.
    Subclasses should implement forward() for their specific computation.
    """

    def __init__(self):
        self._training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self) -> List[Tensor]:
        """Recursively collect all parameters with requires_grad=True."""
        ps: List[Tensor] = []
        for v in self.__dict__.values():
            ps.extend(_gather(v))
        return ps

    def zero_grad(self):
        """Zero gradients of all parameters."""
        for p in self.parameters():
            p.zero_grad()

    def train(self):
        self._training = True

    def eval(self):
        self._training = False


def _gather(obj) -> List[Tensor]:
    """Recursively extract trainable tensors from nested structures."""

    out: List[Tensor] = []
    if isinstance(obj, Tensor) and obj.requires_grad:
        out.append(obj)
    elif isinstance(obj, Module):
        out.extend(obj.parameters())
    elif isinstance(obj, (list, tuple)):
        for e in obj:
            out.extend(_gather(e))
    elif isinstance(obj, dict):
        for e in obj.values():
            out.extend(_gather(e))
    return out
