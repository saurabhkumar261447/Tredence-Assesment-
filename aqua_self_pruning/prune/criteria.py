import numpy as np
from nn.layers import Linear

def compute_taylor_saliency(layer: Linear) -> np.ndarray:
    """
    First-Order Taylor Expansion Importance Criterion:
    Delta L ~ |dL/dW * W|
    Approximates the loss perturbation if connection W_ij is set to 0.
    """
    if layer.weight.grad is None:
        return np.abs(layer.weight.data)
    return np.abs(layer.weight.data * layer.weight.grad)

def compute_magnitude(layer: Linear) -> np.ndarray:
    """Baseline magnitude criterion: |W|"""
    return np.abs(layer.weight.data)
