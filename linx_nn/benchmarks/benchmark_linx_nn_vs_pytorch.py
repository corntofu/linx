"""Benchmark linx_nn against PyTorch on a small CPU MLP.

Run from the workspace root:

    PYTHONPATH=linx/python:linx_nn python linx_nn/benchmarks/benchmark_linx_nn_vs_pytorch.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from linx_nn import Linear, ReLU, Sequential, SGD, Tensor, mse_loss


def _timer():
    return time.perf_counter()


def make_data(batch: int, in_features: int, out_features: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, in_features)).astype(np.float64)
    true_w = rng.normal(size=(in_features, out_features)).astype(np.float64)
    y = np.tanh(x @ true_w)
    return x, y


def bench_linx_nn(x_np, y_np, hidden: int, iters: int, warmup: int):
    model = Sequential(
        Linear(x_np.shape[1], hidden, seed=1),
        ReLU(),
        Linear(hidden, y_np.shape[1], seed=2),
    )
    opt = SGD(model.parameters(), lr=1e-3)
    x = Tensor(x_np)
    y = Tensor(y_np)
    samples = []
    last_loss = None

    for step in range(iters + warmup):
        start = _timer()
        pred = model(x)
        loss = mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        elapsed = _timer() - start
        if step >= warmup:
            samples.append(elapsed)
        last_loss = loss.item()

    return samples, last_loss


def bench_torch(x_np, y_np, hidden: int, iters: int, warmup: int):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed") from exc

    torch.set_default_dtype(torch.float64)
    model = torch.nn.Sequential(
        torch.nn.Linear(x_np.shape[1], hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, y_np.shape[1]),
    )
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    samples = []
    last_loss = None

    for step in range(iters + warmup):
        start = _timer()
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        elapsed = _timer() - start
        if step >= warmup:
            samples.append(elapsed)
        last_loss = float(loss.detach().cpu())

    return samples, last_loss


def summarize(name: str, samples, loss: float):
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    return {
        "name": name,
        "mean_ms": mean * 1000.0,
        "median_ms": median * 1000.0,
        "iters_per_sec": 1.0 / mean,
        "loss": loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--in-features", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out-features", type=int, default=64)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    x_np, y_np = make_data(args.batch, args.in_features, args.out_features, args.seed)
    linx_samples, linx_loss = bench_linx_nn(x_np, y_np, args.hidden, args.iters, args.warmup)
    torch_samples, torch_loss = bench_torch(x_np, y_np, args.hidden, args.iters, args.warmup)

    results = [
        summarize("linx_nn", linx_samples, linx_loss),
        summarize("pytorch", torch_samples, torch_loss),
    ]
    fastest = min(results, key=lambda item: item["mean_ms"])

    print("backend,batch,in,hidden,out,mean_ms,median_ms,iters_per_sec,loss")
    for row in results:
        print(
            f"{row['name']},{args.batch},{args.in_features},{args.hidden},{args.out_features},"
            f"{row['mean_ms']:.3f},{row['median_ms']:.3f},{row['iters_per_sec']:.2f},{row['loss']:.6f}"
        )

    ratio = max(results, key=lambda item: item["mean_ms"])["mean_ms"] / fastest["mean_ms"]
    print(f"fastest={fastest['name']} speedup={ratio:.2f}x")


if __name__ == "__main__":
    main()
