"""Dataset and dataloader implementations."""

import numpy as np


class Dataset:
    """Abstract base class for datasets."""

    pass


class TensorDataset(Dataset):
    """Dataset wrapping tensors for supervised learning.

    Expects 2D arrays for both features and targets with matching first dimension.
    """

    def __init__(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError("TensorDataset: X,y must be 2D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("TensorDataset: first dims differ")
        self.X, self.y = X, y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class DataLoader:
    """Batch iterator with optional shuffling."""

    def __init__(self, ds: Dataset, batch_size=32, shuffle=True):
        if batch_size <= 0:
            raise ValueError("batch_size > 0")
        self.ds, self.bs, self.shuffle = ds, batch_size, shuffle

    def __iter__(self):
        """Yield batches as (X, y) tuples."""
        idx = list(range(len(self.ds)))
        np.random.shuffle(idx) if self.shuffle else None
        for i in range(0, len(idx), self.bs):
            b = idx[i : i + self.bs]
            X = np.stack([self.ds[j][0] for j in b], 0)
            y = np.stack([self.ds[j][1] for j in b], 0)
            yield X, y
