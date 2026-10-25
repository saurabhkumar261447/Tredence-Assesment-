import numpy as np
from engine.tensor import Tensor, unbroadcast
from typing import Optional, Union, Tuple

def add(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(
        data=a.data + b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        parents=(a, b),
        op_name="add"
    )
    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            g = unbroadcast(out.grad, a.shape)
            a.grad = g if a.grad is None else a.grad + g
        if b.requires_grad:
            g = unbroadcast(out.grad, b.shape)
            b.grad = g if b.grad is None else b.grad + g
    out._backward_fn = _backward
    return out

def sub(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(
        data=a.data - b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        parents=(a, b),
        op_name="sub"
    )
    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            g = unbroadcast(out.grad, a.shape)
            a.grad = g if a.grad is None else a.grad + g
        if b.requires_grad:
            g = unbroadcast(-out.grad, b.shape)
            b.grad = g if b.grad is None else b.grad + g
    out._backward_fn = _backward
    return out

def mul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(
        data=a.data * b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        parents=(a, b),
        op_name="mul"
    )
    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            g = unbroadcast(out.grad * b.data, a.shape)
            a.grad = g if a.grad is None else a.grad + g
        if b.requires_grad:
            g = unbroadcast(out.grad * a.data, b.shape)
            b.grad = g if b.grad is None else b.grad + g
    out._backward_fn = _backward
    return out

def div(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(
        data=a.data / b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        parents=(a, b),
        op_name="div"
    )
    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            g = unbroadcast(out.grad / b.data, a.shape)
            a.grad = g if a.grad is None else a.grad + g
        if b.requires_grad:
            g = unbroadcast(-out.grad * a.data / (b.data ** 2), b.shape)
            b.grad = g if b.grad is None else b.grad + g
    out._backward_fn = _backward
    return out

def matmul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(
        data=np.matmul(a.data, b.data),
        requires_grad=a.requires_grad or b.requires_grad,
        parents=(a, b),
        op_name="matmul"
    )
    def _backward():
        if out.grad is None:
            return
        if a.requires_grad:
            g = np.matmul(out.grad, np.swapaxes(b.data, -1, -2))
            g = unbroadcast(g, a.shape)
            a.grad = g if a.grad is None else a.grad + g
        if b.requires_grad:
            g = np.matmul(np.swapaxes(a.data, -1, -2), out.grad)
            g = unbroadcast(g, b.shape)
            b.grad = g if b.grad is None else b.grad + g
    out._backward_fn = _backward
    return out

def tensor_sum(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    out_data = np.sum(a.data, axis=axis, keepdims=keepdims)
    out = Tensor(data=out_data, requires_grad=a.requires_grad, parents=(a,), op_name="sum")
    def _backward():
        if out.grad is None or not a.requires_grad:
            return
        grad = out.grad
        if not keepdims and axis is not None:
            axes = (axis,) if isinstance(axis, int) else axis
            shape = list(a.shape)
            for ax in axes:
                shape[ax] = 1
            grad = grad.reshape(shape)
        g = np.broadcast_to(grad, a.shape)
        a.grad = g if a.grad is None else a.grad + g
    out._backward_fn = _backward
    return out

def tensor_mean(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    out_data = np.mean(a.data, axis=axis, keepdims=keepdims)
    out = Tensor(data=out_data, requires_grad=a.requires_grad, parents=(a,), op_name="mean")
    def _backward():
        if out.grad is None or not a.requires_grad:
            return
        numel = float(np.prod(a.shape) if axis is None else np.prod([a.shape[ax] for ax in ((axis,) if isinstance(axis, int) else axis)]))
        grad = out.grad / numel
        if not keepdims and axis is not None:
            axes = (axis,) if isinstance(axis, int) else axis
            shape = list(a.shape)
            for ax in axes:
                shape[ax] = 1
            grad = grad.reshape(shape)
        g = np.broadcast_to(grad, a.shape)
        a.grad = g if a.grad is None else a.grad + g
    out._backward_fn = _backward
    return out

def relu(a: Tensor) -> Tensor:
    out = Tensor(
        data=np.maximum(0.0, a.data),
        requires_grad=a.requires_grad,
        parents=(a,),
        op_name="relu"
    )
    def _backward():
        if out.grad is None or not a.requires_grad:
            return
        g = out.grad * (a.data > 0.0)
        a.grad = g if a.grad is None else a.grad + g
    out._backward_fn = _backward
    return out

def gelu(a: Tensor) -> Tensor:
    x = a.data
    s = np.sqrt(2.0 / np.pi)
    inner = s * (x + 0.044715 * (x ** 3))
    tanh_inner = np.tanh(inner)
    out_data = 0.5 * x * (1.0 + tanh_inner)
    out = Tensor(data=out_data, requires_grad=a.requires_grad, parents=(a,), op_name="gelu")

    def _backward():
        if out.grad is None or not a.requires_grad:
            return
        sech2 = 1.0 - (tanh_inner ** 2)
        dinner_dx = s * (1.0 + 3.0 * 0.044715 * (x ** 2))
        dgelu_dx = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * dinner_dx
        g = out.grad * dgelu_dx
        a.grad = g if a.grad is None else a.grad + g
    out._backward_fn = _backward
    return out

def tensor_tanh(a: Tensor) -> Tensor:
    out_data = np.tanh(a.data)
    out = Tensor(data=out_data, requires_grad=a.requires_grad, parents=(a,), op_name="tanh")
    def _backward():
        if out.grad is None or not a.requires_grad:
            return
        g = out.grad * (1.0 - out_data ** 2)
        a.grad = g if a.grad is None else a.grad + g
    out._backward_fn = _backward
    return out

def softmax_cross_entropy(logits: Tensor, target_indices: np.ndarray) -> Tensor:
    N = logits.shape[0]
    shift_logits = logits.data - np.max(logits.data, axis=-1, keepdims=True)
    exp_logits = np.exp(shift_logits)
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
    probs = exp_logits / sum_exp

    eps = 1e-15
    correct_logprobs = -np.log(probs[np.arange(N), target_indices] + eps)
    loss_val = np.mean(correct_logprobs)

    out = Tensor(data=loss_val, requires_grad=logits.requires_grad, parents=(logits,), op_name="cross_entropy")

    def _backward():
        if out.grad is None or not logits.requires_grad:
            return
        grad_logits = probs.copy()
        grad_logits[np.arange(N), target_indices] -= 1.0
        grad_logits = (grad_logits / N) * out.grad
        logits.grad = grad_logits if logits.grad is None else logits.grad + grad_logits
        
    out._backward_fn = _backward
    return out
