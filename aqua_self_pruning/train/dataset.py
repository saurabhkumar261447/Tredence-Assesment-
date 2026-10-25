import numpy as np
from sklearn.datasets import load_digits
from typing import Tuple

def load_data(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target.astype(np.int64)

    # Standardize features
    X = (X - np.mean(X, axis=0, keepdims=True)) / (np.std(X, axis=0, keepdims=True) + 1e-6)

    # 80/20 train/test split
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
