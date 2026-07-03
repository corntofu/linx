import numpy as np
import linx


def test_matmul():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    np.testing.assert_allclose(linx.matmul(a, b), a @ b)
    np.testing.assert_allclose(linx.matmul_strassen(a, b, threshold=2), a @ b)


def test_elementwise_api_exports():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])

    np.testing.assert_allclose(linx.add(a, b), a + b)
    np.testing.assert_allclose(linx.subtract(a, b), a - b)
    np.testing.assert_allclose(linx.hadamard(a, b), a * b)
    np.testing.assert_allclose(linx.scalar_mul(a, 2.5), a * 2.5)
    np.testing.assert_allclose(linx.neg(a), -a)


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
    inv_strassen = linx.inverse_schur_strassen(a, min_block=2, strassen_threshold=2)
    np.testing.assert_allclose(a @ inv_strassen, np.eye(4), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        linx.inverse(a, method="schur_strassen", min_block=2),
        np.linalg.inv(a),
        rtol=1e-8,
        atol=1e-8,
    )


def test_solve():
    a = np.array([[3.0, 2.0], [1.0, 2.0]])
    b = np.array([[5.0], [5.0]])
    np.testing.assert_allclose(linx.solve(a, b), np.linalg.solve(a, b), atol=1e-12)


def test_least_squares():
    a = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    b = np.array([[6.0], [5.0], [7.0], [10.0]])
    expected = np.linalg.lstsq(a, b, rcond=None)[0]
    np.testing.assert_allclose(linx.least_squares(a, b), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(linx.Matrix(a).least_squares(b).data, expected, rtol=1e-10, atol=1e-10)


def test_condition_number():
    a = np.array([[4.0, 7.0], [2.0, 6.0]])
    assert linx.condition_number(a) > 1.0
    backend = linx.hardware_backend()
    assert isinstance(backend, str) and len(backend) > 0
    summary = linx.cpu_optimization_summary()
    assert isinstance(summary, str) and len(summary) > 0


# ── new Matrix class tests ──────────────────────────────────────────────────

def test_matrix_ops():
    A = linx.Matrix([[1.0, 2.0], [3.0, 4.0]])
    B = linx.Matrix([[5.0, 6.0], [7.0, 8.0]])

    # addition / subtraction
    np.testing.assert_allclose((A + B).data, np.array([[6., 8.], [10., 12.]]))
    np.testing.assert_allclose((A - B).data, np.array([[-4., -4.], [-4., -4.]]))

    # scalar multiplication
    np.testing.assert_allclose((A * 2.5).data, np.array([[2.5, 5.], [7.5, 10.]]))

    # Hadamard product
    np.testing.assert_allclose((A * B).data, np.array([[5., 12.], [21., 32.]]))

    # matmul
    np.testing.assert_allclose((A @ B).data, A.data @ B.data)

    # transpose
    np.testing.assert_allclose(A.T.data, np.array([[1., 3.], [2., 4.]]))

    # negation
    np.testing.assert_allclose((-A).data, np.array([[-1., -2.], [-3., -4.]]))

    # inverse
    invA = A.inv()
    np.testing.assert_allclose((A @ invA).data, np.eye(2), rtol=1e-8, atol=1e-8)
    invA_strassen = A.inv_schur_strassen(min_block=2, strassen_threshold=2)
    np.testing.assert_allclose((A @ invA_strassen).data, np.eye(2), rtol=1e-8, atol=1e-8)

    # norms
    assert A.frobenius_norm() > 0


def test_matrix_factories():
    Z = linx.Matrix.zeros((3, 4))
    assert Z.shape == (3, 4)
    np.testing.assert_allclose(Z.data, 0.0)

    O = linx.Matrix.ones((2, 3))
    np.testing.assert_allclose(O.data, 1.0)

    I = linx.Matrix.eye(3)
    np.testing.assert_allclose(I.data, np.eye(3))


if __name__ == "__main__":
    test_matmul()
    test_inverse()
    test_solve()
    test_least_squares()
    test_condition_number()
    test_matrix_ops()
    test_matrix_factories()
    print("all python linx tests passed")
