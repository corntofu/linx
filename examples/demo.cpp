#include <iostream>

#include "linx/linx.hpp"

int main() {
    using la::Matrix;

    Matrix<double> a{
        {4.0, 7.0, 2.0, 1.0},
        {3.0, 6.0, 1.0, 2.0},
        {2.0, 5.0, 3.0, 1.0},
        {1.0, 2.0, 1.0, 4.0}
    };

    auto inv = la::inverse_schur(a, 2);
    auto identity_like = la::matmul(a, inv);

    std::cout << "backend: " << la::hardware_backend() << "\n\n";
    std::cout << "A:\n" << a << "\n\n";
    std::cout << "inverse(A):\n" << inv << "\n\n";
    std::cout << "A @ inverse(A):\n" << identity_like << "\n\n";
    std::cout << "residual norm: " << la::residual_norm(a, inv) << "\n";
    std::cout << "condition estimate: " << la::condition_number_estimate(a) << "\n";

    return 0;
}
