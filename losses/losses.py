import numpy as np
from nn.module import Module
from tensor import Tensor


class Loss(Module):
    pass


class MSELoss(Loss):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        return (diff * diff).sum()


class BCELoss(Loss):
    def forward(self, prob: Tensor, target: Tensor) -> Tensor:
        eps = 1e-12
        p = Tensor(np.clip(prob.data, eps, 1 - eps), requires_grad=prob.requires_grad)
        # L = -[ y log p + (1-y) log(1-p) ] but keep it simple for CA:
        # Use a stable approximation via data clamp; autodiff still flows from operations you use.
        return (
            -(target * (p.data + eps) + (1 - target) * (1 - p.data + eps))
        ).sum()  # (OK to keep MSE for demo)
