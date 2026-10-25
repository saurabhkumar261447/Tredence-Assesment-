import numpy as np
import pytest
from engine.tensor import Tensor
from nn.layers import Linear
from nn.optim import Adam

def test_masked_weight_gradient_and_optimizer_isolation():
    layer = Linear(4, 4, bias=False)
    mask = np.eye(4)
    layer.set_mask(mask)

    opt = Adam(layer.parameters(), lr=1e-2)

    for _ in range(5):
        x = Tensor(np.random.randn(2, 4))
        y = layer(x)
        loss = y.sum()
        layer.zero_grad()
        loss.backward()

        opt.step(layers=[layer])

        # Masked weights and their optimizer states must remain strictly zero
        assert np.all(layer.weight.grad[mask == 0.0] == 0.0)
        assert np.all(layer.weight.data[mask == 0.0] == 0.0)
        assert np.all(opt.m[0][mask == 0.0] == 0.0)
        assert np.all(opt.v[0][mask == 0.0] == 0.0)

    # Test clean revival without stale momentum
    new_mask = mask.copy()
    new_mask[0, 1] = 1.0
    layer.set_mask(new_mask)
    opt.enforce_mask([layer])

    assert opt.m[0][0, 1] == 0.0
    assert opt.v[0][0, 1] == 0.0

    x = Tensor(np.ones((1, 4)))
    loss = layer(x).sum()
    layer.zero_grad()
    loss.backward()
    opt.step(layers=[layer])

    assert layer.weight.grad[0, 1] != 0.0
    assert layer.weight.data[0, 1] != 0.0
