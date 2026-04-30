import numpy as np
import linx


def test_matmul():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    np.testing.assert_allclose(linx.matmul(a, b), a @ b)
    np.testing.assert_allclose(linx.matmul_strassen(a, b, threshold=2), a @ b)


def test_inverse():
    a = np.array(
        [
            [8.0, 1.0, 2.0, 0.5],
            [1.0, 7.0, 0.5, 1.0],
            [2.0, 0.5, 6.0, 1.0],
            [0.5, 1.0, 1.0, 5.0],
        ]
    )
    inv = linx.inverse_schur(a, min_block=2)
    np.testing.assert_allclose(a @ inv, np.eye(4), rtol=1e-8, atol=1e-8)


def test_solve():
    a = np.array([[3.0, 2.0], [1.0, 2.0]])
    b = np.array([[5.0], [5.0]])
    np.testing.assert_allclose(linx.solve(a, b), np.linalg.solve(a, b), atol=1e-12)


def test_condition_number():
    a = np.array([[4.0, 7.0], [2.0, 6.0]])
    assert linx.condition_number(a) > 1.0
    assert "thread" in linx.hardware_backend()


if __name__ == "__main__":
    test_matmul()
    test_inverse()
    test_solve()
    test_condition_number()
    print("all python linx tests passed")
