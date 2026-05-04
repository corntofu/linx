import numpy as np

from linx_nn import Linear, Sequential, SGD, Tanh, Tensor, mse_loss


def main():
    x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = Tensor([[0.0], [1.0], [1.0], [0.0]])

    model = Sequential(
        Linear(2, 8, seed=7),
        Tanh(),
        Linear(8, 1, seed=11),
    )
    opt = SGD(model.parameters(), lr=0.03, momentum=0.9)

    for epoch in range(2000):
        pred = model(x)
        loss = mse_loss(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 400 == 0:
            print(f"epoch={epoch:04d} loss={loss.item():.6f}")

    pred = model(x).data
    print("prediction:")
    print(np.round(pred, 3))


if __name__ == "__main__":
    main()
