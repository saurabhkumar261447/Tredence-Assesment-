import numpy as np
from typing import List
from nn.module import Parameter
from nn.layers import Linear

class Adam:
    def __init__(
        self,
        params: List[Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self, layers: List[Linear] = None) -> None:
        self.t += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * p.data

            # 1. Update 1st and 2nd moments
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)

            # 2. Update parameters
            step_update = lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)
            p.data -= step_update

        # 3. Synchronize and zero out optimizer momentum buffers for masked weights
        if layers is not None:
            self.enforce_mask(layers)

    def enforce_mask(self, layers: List[Linear]) -> None:
        for layer in layers:
            p = layer.weight
            mask = layer.mask
            p.data *= mask
            
            for i, opt_param in enumerate(self.params):
                if opt_param is p:
                    self.m[i] *= mask
                    self.v[i] *= mask
                    if p.grad is not None:
                        p.grad *= mask
