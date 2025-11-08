# MiniGrad

A minimal deep learning framework implementing automatic differentiation and neural network training from scratch using NumPy. Built for fun.

## Installation

Clone the repository:
```bash
git clone https://github.com/a1regg/minigrad.git
cd minigrad
```

Requirements:
```bash
pip install numpy
```

## Features

- **Automatic Differentiation**: Reverse-mode autodiff (backpropagation) with dynamic computational graph
- **Neural Network Modules**: Modular architecture with layers, activations, and sequential containers
- **Optimizers**: SGD with momentum support
- **Loss Functions**: MSE and Binary Cross-Entropy
- **Data Handling**: Dataset and DataLoader with batching and shuffling
- **Training Loop**: High-level Trainer API with configurable epochs and logging

## MiniGrad Structure

```
.
├── tensor.py              # Core autodiff tensor with gradient tracking
├── nn/
│   ├── module.py          # Base Module class for all neural components
│   ├── layers.py          # Linear (fully-connected) layer
│   ├── activations.py     # ReLU, Sigmoid activation modules
│   ├── sequential.py      # Sequential container for layer composition
│   └── functional.py      # Functional API for stateless operations
├── data/
│   └── data.py            # Dataset and DataLoader implementations
├── losses/
│   └── losses.py          # Loss functions (MSE, BCE)
├── optimizers/
│   └── optimizers.py      # Optimization algorithms (SGD)
├── train/
│   └── train.py           # Training loop and configuration
└── examples/
    └── demo.ipynb         # Interactive Jupyter notebook demo
```

## Quick Start

### XOR Example

The classic XOR problem demonstrates the framework's capability to learn non-linear functions:

```python
import numpy as np
from tensor import Tensor
from nn.layers import Linear
from nn.activations import ReLU, Sigmoid
from nn.sequential import Sequential
from losses.losses import MSELoss
from data.data import TensorDataset, DataLoader
from optimizers.optimizers import SGD
from train.train import Trainer, TrainConfig

# Define model architecture
model = Sequential(
    Linear(2, 8),
    ReLU(),
    Linear(8, 1),
    Sigmoid()
)

# Prepare data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Configure training
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.2, momentum=0.9)
trainer = Trainer(model, loss_fn, optimizer)

# Train
trainer.fit(loader, cfg=TrainConfig(epochs=2000, print_every=200))

# Inference
predictions = model(Tensor(X))
print("Predictions:", np.round(predictions.data, 3))
```

Run the example:
```bash
jupyter notebook examples/demo.ipynb
```

## Core Components

### Tensor (Automatic Differentiation)

The `Tensor` class implements reverse-mode automatic differentiation with dynamic computational graph construction.

```python
from tensor import Tensor

# Create tensors
x = Tensor([[1.0, 2.0]], requires_grad=True)
W = Tensor([[0.5], [0.3]], requires_grad=True)

# Forward pass
y = x.matmul(W)
loss = (y * y).sum()

# Backward pass
loss.backward()

# Access gradients
print(W.grad)  # ∂loss/∂W
```

**Supported Operations:**
- Arithmetic: `+`, `-`, `*` (with broadcasting)
- Matrix: `matmul()`
- Reduction: `sum(axis, keepdims)`
- Activations: `relu()`, `sigmoid()`, `exp()`

**Gradient Formulas:**
- Addition: `∂L/∂x = ∂L/∂out`
- Multiplication: `∂L/∂x = other * ∂L/∂out`
- MatMul: `∂L/∂x = ∂L/∂out @ other.T`
- ReLU: `∂L/∂x = ∂L/∂out * 𝟙(x > 0)`
- Sigmoid: `∂L/∂x = σ(x)(1-σ(x)) * ∂L/∂out`

### Neural Network Modules

#### Linear Layer

Fully-connected layer with Xavier initialization:

```python
from nn.layers import Linear

layer = Linear(in_features=10, out_features=5, bias=True)
```

Computes: `y = xW + b` where `W ∈ ℝ^(in × out)`, `b ∈ ℝ^(1 × out)`

#### Activations

```python
from nn.activations import ReLU, Sigmoid

relu = ReLU()      # max(0, x)
sigmoid = Sigmoid() # 1 / (1 + e^(-x))
```

#### Sequential Container

Chain multiple modules:

```python
from nn.sequential import Sequential

model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)
```

### Loss Functions

#### Mean Squared Error

```python
from losses.losses import MSELoss

loss_fn = MSELoss()
loss = loss_fn(predictions, targets)  # Σ(pred - target)²
```

#### Binary Cross-Entropy

```python
from losses.losses import BCELoss

loss_fn = BCELoss()
loss = loss_fn(probabilities, targets)  # -Σ[y log(p) + (1-y)log(1-p)]
```

### Optimizers

#### SGD with Momentum

```python
from optimizers.optimizers import SGD

optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training step
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Update rule:
```
v = momentum * v + grad
param = param - lr * v
```

### Data Handling

#### Dataset

```python
from data.data import TensorDataset

dataset = TensorDataset(X, y)  # X, y must be 2D arrays
sample = dataset[0]  # Returns (x, y) tuple
```

#### DataLoader

```python
from data.data import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for X_batch, y_batch in loader:
    # Training iteration
    pass
```

### Training

#### Trainer API

```python
from train.train import Trainer, TrainConfig

trainer = Trainer(model, loss_fn, optimizer)

config = TrainConfig(
    epochs=1000,
    print_every=100
)

history = trainer.fit(
    train_loader,
    val_loader=val_loader,  # Optional
    cfg=config
)
```

## Implementation Details

### Automatic Differentiation

The framework uses **reverse-mode automatic differentiation** (backpropagation):

1. **Forward Pass**: Operations build a dynamic computational graph
2. **Backward Pass**: Topological sort + chain rule application in reverse order

Each operation stores:
- Parent tensors (`_prev`)
- Gradient function (`_backward`)

Broadcasting is handled by summing gradients over broadcasted dimensions during backpropagation.

### Memory Management

Uses `__slots__` in `Tensor` class for memory efficiency:
```python
__slots__ = ("data", "grad", "requires_grad", "_prev", "_backward", "shape")
```

### Gradient Accumulation

Gradients accumulate across operations (required for parameter sharing):
```python
self.grad = self.grad + incoming_grad  
```

## Technical Notes

- **Numerical Stability**: BCE loss uses epsilon clipping (`1e-12`) to prevent log(0)
- **Initialization**: Linear layers use Xavier initialization: `U(-√(6/n_in), √(6/n_in))`
- **Broadcasting**: Fully supported in arithmetic operations with proper gradient handling
- **Type Coercion**: Scalars automatically converted to Tensors in operations

## Limitations

- NumPy-only backend (CPU-only, no GPU acceleration)
- Limited optimizer selection (SGD only)
- No convolutional or recurrent layers
- No built-in regularization (dropout, batch norm)
- No automatic mixed precision
- No model serialization/loading

## Acknowledgements

Inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy.
Numpy


