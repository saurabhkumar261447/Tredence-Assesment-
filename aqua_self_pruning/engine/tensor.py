from __future__ import annotations
import numpy as np
from typing import Tuple, List, Set, Optional, Union

def unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Sums gradient across dimensions where broadcasting occurred during the forward pass.
    Handles dimension expansions (leading axes) and size-1 singleton axes.
    """
    if grad.shape == target_shape:
        return grad

    grad_ndim = grad.ndim
    target_ndim = len(target_shape)
    
    # 1. Sum over leading axes that were prepended
    num_leading = grad_ndim - target_ndim
    if num_leading > 0:
        grad = grad.sum(axis=tuple(range(num_leading)))

    # 2. Sum over axes that were expanded from 1
    axes_to_sum = []
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] != 1:
            axes_to_sum.append(i)
            
    if axes_to_sum:
        grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)

    return grad.reshape(target_shape)


class Tensor:
    """
    Core differentiable array wrapping NumPy ndarray.
    Maintains computation graph DAG parents and backward closures.
    """
    def __init__(
        self,
        data: Union[int, float, list, np.ndarray],
        requires_grad: bool = False,
        parents: Optional[Tuple[Tensor, ...]] = None,
        op_name: str = ""
    ):
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float64, copy=False)
        else:
            self.data = np.array(data, dtype=np.float64)
            
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._parents = parents or ()
        self._op_name = op_name
        self._backward_fn = lambda: None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0.0)

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """
        Reverse-mode automatic differentiation with topological ordering.
        """
        if not self.requires_grad:
            return

        if grad is None:
            if self.shape == () or self.shape == (1,):
                grad = np.ones_like(self.data)
            else:
                raise RuntimeError("Grad can only be implicitly created for scalar outputs.")
        else:
            grad = np.array(grad, dtype=np.float64)

        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad

        topo: List[Tensor] = []
        visited: Set[Tensor] = set()

        def build_topo(v: Tensor):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        for node in reversed(topo):
            if node._backward_fn is not None:
                node._backward_fn()

    def __add__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import add
        return add(self, other if isinstance(other, Tensor) else Tensor(other))

    def __radd__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import sub
        return sub(self, other if isinstance(other, Tensor) else Tensor(other))

    def __rsub__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        return (self if isinstance(other, Tensor) else Tensor(other)).__sub__(self)

    def __mul__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import mul
        return mul(self, other if isinstance(other, Tensor) else Tensor(other))

    def __rmul__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import div
        return div(self, other if isinstance(other, Tensor) else Tensor(other))

    def __matmul__(self, other: Tensor) -> Tensor:
        from engine.ops import matmul
        return matmul(self, other)

    def __neg__(self) -> Tensor:
        return self * -1.0

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        from engine.ops import tensor_sum
        return tensor_sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        from engine.ops import tensor_mean
        return tensor_mean(self, axis=axis, keepdims=keepdims)

    def relu(self) -> Tensor:
        from engine.ops import relu
        return relu(self)

    def gelu(self) -> Tensor:
        from engine.ops import gelu
        return gelu(self)

    def tanh(self) -> Tensor:
        from engine.ops import tensor_tanh
        return tensor_tanh(self)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
