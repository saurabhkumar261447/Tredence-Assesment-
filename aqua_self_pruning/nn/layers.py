import numpy as np
from engine.tensor import Tensor
from nn.module import Module, Parameter

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # He Normal initialization
        std = np.sqrt(2.0 / in_features)
        self.weight = Parameter(np.random.randn(in_features, out_features) * std)
        self.bias = Parameter(np.zeros((out_features,))) if bias else None
        
        # Binary Pruning Mask: 1.0 = active, 0.0 = pruned
        self.mask = np.ones((in_features, out_features), dtype=np.float64)

    def set_mask(self, mask: np.ndarray) -> None:
        assert mask.shape == self.weight.shape
        self.mask = mask.astype(np.float64)
        # Enforce zero weight under mask immediately
        self.weight.data *= self.mask

    def forward(self, x: Tensor) -> Tensor:
        # Effective weight: W_eff = W * M
        effective_w = self.weight * Tensor(self.mask, requires_grad=False)
        out = x @ effective_w
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward_sparse_inference(self, x_np: np.ndarray) -> np.ndarray:
        """
        Direct sparse inference: only computes non-zero input dot products.
        Avoids dense-times-zero calculations.
        """
        active_w = self.weight.data * self.mask
        B, Din = x_np.shape
        Dout = self.out_features
        out = np.zeros((B, Dout), dtype=np.float64)
        
        for j in range(Dout):
            col = active_w[:, j]
            nonzeros = np.nonzero(col)[0]
            if len(nonzeros) > 0:
                out[:, j] = np.dot(x_np[:, nonzeros], col[nonzeros])
                
        if self.bias is not None:
            out += self.bias.data
        return out

class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers)

    def parameters(self) -> list:
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params

    def forward(self, x: Tensor) -> Tensor:
        for l in self.layers:
            x = l(x)
        return x

class MLP(Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = Linear(in_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.fc3 = Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        h1 = self.fc1(x).gelu()
        h2 = self.fc2(h1).gelu()
        return self.fc3(h2)
