# linx workspace

This workspace now contains two separated projects:

- `linx/`: C++17/NumPy-style linear algebra backend with Python bindings
- `linx_nn/`: PyTorch-like neural network toolkit built on top of `linx`

Typical local import path:

```bash
PYTHONPATH=linx/python:linx_nn python -m pytest linx/tests linx_nn/tests -q
```
