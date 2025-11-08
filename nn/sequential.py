"""Sequential module container."""

from typing import List
from nn.module import Module


class Sequential(Module):
    """Container for sequential execution of modules."""

    def __init__(self, *modules: Module):
        super().__init__()
        if not modules:
            raise ValueError("Sequential requires at least one module")
        self._mods: List[Module] = list(modules)

    def forward(self, x):
        """Apply modules sequentially."""
        for m in self._mods:
            x = m(x)
        return x

    def __iter__(self):
        return iter(self._mods)
