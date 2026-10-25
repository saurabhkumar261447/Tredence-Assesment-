import numpy as np
import pytest
from engine.tensor import Tensor
from engine.ops import add, sub, mul, div, matmul, relu, gelu, tensor_tanh, tensor_sum, tensor_mean, softmax_cross_entropy

def numerical_gradient(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        oldval = x[idx]
        
        x[idx] = oldval + eps
        pos = f(x)
        
        x[idx] = oldval - eps
        neg = f(x)
        
        x[idx] = oldval
        grad[idx] = (pos - neg) / (2.0 * eps)
        it.iternext()
    return grad

def test_add_broadcasting():
    np.random.seed(0)
    a_data = np.random.randn(4, 5)
    b_data = np.random.randn(5)

    def f_a(x):
        return np.sum(x + b_data)
    def f_b(y):
        return np.sum(a_data + y)

    ta = Tensor(a_data, requires_grad=True)
    tb = Tensor(b_data, requires_grad=True)
    out = (ta + tb).sum()
    out.backward()

    expected_ga = numerical_gradient(f_a, a_data.copy())
    expected_gb = numerical_gradient(f_b, b_data.copy())

    assert np.allclose(ta.grad, expected_ga, rtol=1e-5, atol=1e-5)
    assert np.allclose(tb.grad, expected_gb, rtol=1e-5, atol=1e-5)

def test_matmul_grad():
    np.random.seed(1)
    a = np.random.randn(3, 4)
    b = np.random.randn(4, 2)

    def f_a(x):
        return np.sum(x @ b)
    def f_b(y):
        return np.sum(a @ y)

    ta = Tensor(a, requires_grad=True)
    tb = Tensor(b, requires_grad=True)
    out = (ta @ tb).sum()
    out.backward()

    assert np.allclose(ta.grad, numerical_gradient(f_a, a.copy()), rtol=1e-5, atol=1e-5)
    assert np.allclose(tb.grad, numerical_gradient(f_b, b.copy()), rtol=1e-5, atol=1e-5)

def test_activations_and_ops():
    np.random.seed(2)
    x = np.random.randn(3, 3)

    # ReLU
    tx = Tensor(x, requires_grad=True)
    relu(tx).sum().backward()
    assert np.allclose(tx.grad, (x > 0).astype(float))

    # GELU
    tx_gelu = Tensor(x, requires_grad=True)
    gelu(tx_gelu).sum().backward()
    num_grad = numerical_gradient(lambda z: np.sum(0.5 * z * (1.0 + np.tanh(np.sqrt(2/np.pi)*(z + 0.044715*(z**3))))), x.copy())
    assert np.allclose(tx_gelu.grad, num_grad, rtol=1e-4, atol=1e-4)

def test_softmax_cross_entropy():
    np.random.seed(3)
    logits = np.random.randn(5, 3)
    targets = np.array([0, 2, 1, 1, 0])

    t_logits = Tensor(logits, requires_grad=True)
    loss = softmax_cross_entropy(t_logits, targets)
    loss.backward()

    def loss_func(l):
        shift = l - np.max(l, axis=-1, keepdims=True)
        ex = np.exp(shift)
        p = ex / np.sum(ex, axis=-1, keepdims=True)
        return -np.mean(np.log(p[np.arange(5), targets] + 1e-15))

    num_g = numerical_gradient(loss_func, logits.copy())
    assert np.allclose(t_logits.grad, num_g, rtol=1e-5, atol=1e-5)
