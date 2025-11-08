"""Automatic differentiation tensor implementation."""

import numpy as np


class Tensor:
    """Differentiable tensor with reverse-mode automatic differentiation.

    Supports basic arithmetic, matrix operations, and activations with gradient
    tracking. Gradients are accumulated via backpropagation using dynamic
    computational graph construction.
    """

    __slots__ = ("data", "grad", "requires_grad", "_prev", "_backward", "shape")

    def __init__(self, data, requires_grad=False):
        """Initialize tensor from array-like data.

        Args:
            data: Array-like numerical data.
            requires_grad: If True, track gradients for this tensor.
        """
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self._prev = []
        self._backward = lambda: None
        self.shape = self.data.shape

    def zero_grad(self):
        """Reset accumulated gradient to zero."""
        if self.grad is not None:
            self.grad[...] = 0.0

    def _wrap(self, out_data, parents, backward):
        """Wrap operation output in new Tensor with gradient function."""
        out = Tensor(out_data, any(p.requires_grad for p in parents))
        if out.requires_grad:
            out._prev = parents
            out._backward = backward
        return out

    def __add__(self, other):
        """Element-wise addition with broadcasting support.

        Gradient: ∂L/∂self = ∂L/∂out, ∂L/∂other = ∂L/∂out
        Broadcasting is reversed via sum over broadcasted dimensions.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._wrap(self.data + other.data, [self, other], lambda: None)

        def _bw():
            if self.requires_grad:
                grad_self = out.grad
                # Sum out broadcasted dimensions
                ndims_added = grad_self.ndim - self.data.ndim
                for i in range(ndims_added):
                    grad_self = grad_self.sum(axis=0)
                for i, (dim_grad, dim_self) in enumerate(
                    zip(grad_self.shape, self.data.shape)
                ):
                    if dim_self == 1 and dim_grad != 1:
                        grad_self = grad_self.sum(axis=i, keepdims=True)
                self.grad = self.grad + grad_self
            if other.requires_grad:
                grad_other = out.grad
                # Sum out broadcasted dimensions
                ndims_added = grad_other.ndim - other.data.ndim
                for i in range(ndims_added):
                    grad_other = grad_other.sum(axis=0)
                for i, (dim_grad, dim_other) in enumerate(
                    zip(grad_other.shape, other.data.shape)
                ):
                    if dim_other == 1 and dim_grad != 1:
                        grad_other = grad_other.sum(axis=i, keepdims=True)
                other.grad = other.grad + grad_other

        out._backward = _bw if out.requires_grad else out._backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        """Element-wise multiplication with broadcasting support.

        Gradient: ∂L/∂self = other * ∂L/∂out, ∂L/∂other = self * ∂L/∂out
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._wrap(self.data * other.data, [self, other], lambda: None)

        def _bw():
            if self.requires_grad:
                grad_self = other.data * out.grad
                # Sum out broadcasted dimensions
                ndims_added = grad_self.ndim - self.data.ndim
                for i in range(ndims_added):
                    grad_self = grad_self.sum(axis=0)
                for i, (dim_grad, dim_self) in enumerate(
                    zip(grad_self.shape, self.data.shape)
                ):
                    if dim_self == 1 and dim_grad != 1:
                        grad_self = grad_self.sum(axis=i, keepdims=True)
                self.grad = self.grad + grad_self
            if other.requires_grad:
                grad_other = self.data * out.grad
                # Sum out broadcasted dimensions
                ndims_added = grad_other.ndim - other.data.ndim
                for i in range(ndims_added):
                    grad_other = grad_other.sum(axis=0)
                for i, (dim_grad, dim_other) in enumerate(
                    zip(grad_other.shape, other.data.shape)
                ):
                    if dim_other == 1 and dim_grad != 1:
                        grad_other = grad_other.sum(axis=i, keepdims=True)
                other.grad = other.grad + grad_other

        out._backward = _bw if out.requires_grad else out._backward
        return out

    def matmul(self, other):
        """Matrix multiplication: self @ other.

        Gradient: ∂L/∂self = ∂L/∂out @ other^T, ∂L/∂other = self^T @ ∂L/∂out
        """
        out = self._wrap(self.data @ other.data, [self, other], lambda: None)

        def _bw():
            if self.requires_grad:
                self.grad = self.grad + out.grad @ other.data.T
            if other.requires_grad:
                other.grad = other.grad + self.data.T @ out.grad

        out._backward = _bw if out.requires_grad else out._backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = self._wrap(
            self.data.sum(axis=axis, keepdims=keepdims), [self], lambda: None
        )

        def _bw():
            if self.requires_grad:
                self.grad = self.grad + out.grad * np.ones_like(self.data)

        out._backward = _bw if out.requires_grad else out._backward
        return out

    def relu(self):
        """Rectified Linear Unit: max(0, x).

        Gradient: ∂L/∂x = ∂L/∂out * 𝟙(x > 0)
        """
        y = np.maximum(self.data, 0.0)
        mask = (self.data > 0).astype(self.data.dtype)
        out = self._wrap(y, [self], lambda: None)

        def _bw():
            if self.requires_grad:
                self.grad = self.grad + mask * out.grad

        out._backward = _bw if out.requires_grad else out._backward
        return out

    def exp(self):
        """Element-wise exponential: e^x.

        Gradient: ∂L/∂x = e^x * ∂L/∂out
        """
        y = np.exp(self.data)
        out = self._wrap(y, [self], lambda: None)

        def _bw():
            if self.requires_grad:
                self.grad = self.grad + y * out.grad

        out._backward = _bw if out.requires_grad else out._backward
        return out

    def sigmoid(self):
        """Logistic sigmoid: σ(x) = 1 / (1 + e^-x).

        Gradient: ∂L/∂x = σ(x)(1 - σ(x)) * ∂L/∂out
        """
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = self._wrap(s, [self], lambda: None)

        def _bw():
            if self.requires_grad:
                self.grad = self.grad + (s * (1 - s)) * out.grad

        out._backward = _bw if out.requires_grad else out._backward
        return out

    def backward(self):
        """Compute gradients via reverse-mode autodiff.

        Performs topological sort of computation graph and applies chain rule
        in reverse order. Initializes this tensor's gradient to ones.
        """
        assert self.grad is not None, "loss must require grad"
        topo, seen = [], set()

        def build(v):
            if v not in seen:
                seen.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
