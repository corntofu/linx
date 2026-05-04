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


def _make_linx_model(x_np, y_np, hidden: int):
    return Sequential(
        Linear(x_np.shape[1], hidden, seed=1),
        ReLU(),
        Linear(hidden, y_np.shape[1], seed=2),
    )


def _linx_weight(in_features: int, out_features: int, seed: int):
    rng = np.random.default_rng(seed)
    limit = np.sqrt(6.0 / in_features)
    return rng.uniform(-limit, limit, size=(in_features, out_features)).astype(np.float64)


def bench_linx_nn_autograd(x_np, y_np, hidden: int, iters: int, warmup: int):
    model = _make_linx_model(x_np, y_np, hidden)
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


def bench_linx_nn_fused(x_np, y_np, hidden: int, iters: int, warmup: int):
    model = _make_linx_model(x_np, y_np, hidden)
    opt = SGD(model.parameters(), lr=1e-3)
    x = Tensor(x_np)
    y = Tensor(y_np)
    samples = []
    last_loss = None

    for step in range(iters + warmup):
        start = _timer()
        loss = model.train_mse_step(x, y, opt)
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
    with torch.no_grad():
        first_weight = _linx_weight(x_np.shape[1], hidden, seed=1)
        second_weight = _linx_weight(hidden, y_np.shape[1], seed=2)
        model[0].weight.copy_(torch.from_numpy(first_weight.T))
        model[0].bias.zero_()
        model[2].weight.copy_(torch.from_numpy(second_weight.T))
        model[2].bias.zero_()
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


def repeat_bench(name: str, fn, repeats: int, *args):
    best = None
    for _ in range(repeats):
        samples, loss = fn(*args)
        row = summarize(name, samples, loss)
        if best is None or row["mean_ms"] < best["mean_ms"]:
            best = row
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--in-features", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out-features", type=int, default=64)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    x_np, y_np = make_data(args.batch, args.in_features, args.out_features, args.seed)
    results = [
        repeat_bench(
            "linx_nn_fused",
            bench_linx_nn_fused,
            args.repeats,
            x_np,
            y_np,
            args.hidden,
            args.iters,
            args.warmup,
        ),
        repeat_bench(
            "linx_nn_autograd",
            bench_linx_nn_autograd,
            args.repeats,
            x_np,
            y_np,
            args.hidden,
            args.iters,
            args.warmup,
        ),
        repeat_bench("pytorch", bench_torch, args.repeats, x_np, y_np, args.hidden, args.iters, args.warmup),
    ]
    fastest = min(results, key=lambda item: item["mean_ms"])

    print("backend,batch,in,hidden,out,repeats,mean_ms,median_ms,iters_per_sec,loss")
    for row in results:
        print(
            f"{row['name']},{args.batch},{args.in_features},{args.hidden},{args.out_features},{args.repeats},"
            f"{row['mean_ms']:.3f},{row['median_ms']:.3f},{row['iters_per_sec']:.2f},{row['loss']:.6f}"
        )

    ratio = max(results, key=lambda item: item["mean_ms"])["mean_ms"] / fastest["mean_ms"]
    print(f"fastest={fastest['name']} speedup={ratio:.2f}x")


if __name__ == "__main__":
    main()
