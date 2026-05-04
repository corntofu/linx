import numpy as np

from linx_nn import Linear, ReLU, Sequential, SGD, Tensor, mse_loss


def test_autograd_matmul_gradients():
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[3.0], [4.0]], requires_grad=True)

    y = x @ w
    y.backward()

    np.testing.assert_allclose(x.grad, np.array([[3.0, 4.0]]))
    np.testing.assert_allclose(w.grad, np.array([[1.0], [2.0]]))


def test_linear_mse_training_step_reduces_loss():
    x = Tensor([[0.0], [1.0], [2.0], [3.0]])
    y = Tensor([[1.0], [3.0], [5.0], [7.0]])

    model = Sequential(Linear(1, 1, seed=3))
    opt = SGD(model.parameters(), lr=0.05)

    first = None
    last = None
    for _ in range(120):
        pred = model(x)
        loss = mse_loss(pred, y)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = loss.item()

    assert last < first * 0.05


def test_sequential_nonlinear_forward_shape():
    model = Sequential(Linear(2, 4, seed=1), ReLU(), Linear(4, 1, seed=2))
    x = Tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    out = model(x)

    assert out.shape == (3, 1)
