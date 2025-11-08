"""Training loop implementation."""

from typing import Optional, List
import numpy as np
from dataclasses import dataclass
from minigrad.nn.module import Module
from minigrad.tensor import Tensor


@dataclass
class TrainConfig:
    """Configuration for training loop.

    Attributes:
        epochs: Number of training epochs.
        print_every: Print loss every N epochs.
    """

    epochs: int = 1000
    print_every: int = 100


class Trainer:
    """Training loop handler for supervised learning.

    Manages forward pass, loss computation, backpropagation, and optimization
    for both training and evaluation.
    """

    def __init__(self, model: Module, loss: Module, optimizer):
        self.model, self.loss, self.opt = model, loss, optimizer

    def fit(
        self, loader, val_loader=None, cfg: Optional[TrainConfig] = None
    ) -> List[float]:
        """Train model for specified epochs.

        Args:
            loader: Training data loader.
            val_loader: Optional validation data loader.
            cfg: Training configuration.

        Returns:
            List of training loss values per epoch.
        """
        cfg = cfg or TrainConfig()
        history: List[float] = []
        for epoch in range(1, cfg.epochs + 1):
            losses = []
            for X_np, y_np in loader:
                self.opt.zero_grad()
                X = Tensor(X_np)  # data tensors
                y = Tensor(y_np)
                pred = self.model(X)
                L = self.loss(pred, y)  # Tensor scalar
                L.backward()  # AD kicks in
                self.opt.step()
                losses.append(float(L.data))
            mean_loss = float(np.mean(losses))
            history.append(mean_loss)
            if epoch % cfg.print_every == 0 or epoch == 1:
                if val_loader is not None:
                    v = self.evaluate(val_loader)
                    print(f"Epoch {epoch:4d}  train={mean_loss:.6f}  val={v:.6f}")
                else:
                    print(f"Epoch {epoch:4d}  train={mean_loss:.6f}")
        return history

    def evaluate(self, loader) -> float:
        """Compute average loss on evaluation set."""
        losses = []
        for X_np, y_np in loader:
            X, y = Tensor(X_np), Tensor(y_np)
            L = self.loss(self.model(X), y)
            losses.append(float(L.data))
        return float(np.mean(losses))
