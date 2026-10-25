from typing import List
from engine.tensor import Tensor

class Parameter(Tensor):
    """Marks a Tensor as a trainable model parameter."""
    def __init__(self, data):
        super().__init__(data, requires_grad=True, op_name="parameter")

class Module:
    def parameters(self) -> List[Parameter]:
        params = []
        for _, val in self.__dict__.items():
            if isinstance(val, Parameter):
                params.append(val)
            elif isinstance(val, Module):
                params.extend(val.parameters())
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()
            
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
