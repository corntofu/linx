# linx_nn

`linx_nn` is a tiny PyTorch-style neural network layer built on top of the
existing `linx` linear algebra backend.

## What is included

- `Tensor`: NumPy-compatible tensor with reverse-mode autograd
- `Parameter`: trainable tensor
- `Module`: PyTorch-like base class with `parameters()` and `zero_grad()`
- Layers: `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Sequential`
- Loss: `mse_loss`
- Optimizer: `SGD`
- Fused dense MLP training: `Sequential.train_mse_step(...)`

Matrix-heavy paths use `linx.matmul`, `linx.transpose`, and linx elementwise
operations when available, so forward and backward passes reuse the optimized
C++/Accelerate/BLAS backend.

For hot CPU training loops made from `Linear` and elementwise activations,
`Sequential.train_mse_step(x, y, optimizer)` skips general graph construction
and runs a fused `forward + MSE backward + optimizer.step`.

## Example

```python
import numpy as np

from linx_nn import Linear, ReLU, Sequential, SGD, Tensor, mse_loss

x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = Tensor([[0.0], [1.0], [1.0], [0.0]])

model = Sequential(
    Linear(2, 8, seed=1),
    ReLU(),
    Linear(8, 1, seed=2),
)
opt = SGD(model.parameters(), lr=0.05)

for _ in range(500):
    pred = model(x)
    loss = mse_loss(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

print(loss.item())
print(model(x).data)
```

## Run Tests

From the workspace root:

```bash
PYTHONPATH=linx/python:linx_nn python -m pytest linx_nn/tests -q
```

## Benchmark vs PyTorch

```bash
PYTHONPATH=linx/python:linx_nn python linx_nn/benchmarks/benchmark_linx_nn_vs_pytorch.py --iters 300 --warmup 80 --repeats 3
```
