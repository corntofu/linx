#include <cassert>
#include <cmath>
#include <iostream>

#include "linx/linx.hpp"

namespace {

void test_basic_ops() {
    la::Matrix<double> a{{1.0, 2.0}, {3.0, 4.0}};
    la::Matrix<double> b{{5.0, 6.0}, {7.0, 8.0}};

    assert((a + b).allclose({{6.0, 8.0}, {10.0, 12.0}}));
    assert((b - a).allclose({{4.0, 4.0}, {4.0, 4.0}}));
    assert(a.transpose().allclose({{1.0, 3.0}, {2.0, 4.0}}));
    assert(a.hadamard(b).allclose({{5.0, 12.0}, {21.0, 32.0}}));
}

void test_matmul() {
    la::Matrix<double> a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    la::Matrix<double> b{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
    assert(la::matmul(a, b).allclose({{58.0, 64.0}, {139.0, 154.0}}));
}

void test_strassen_matches_classic() {
    auto a = la::Matrix<double>::arange(8, 8, 1.0);
    auto b = la::Matrix<double>::arange(8, 8, 0.5, 0.25);
    assert(la::matmul_strassen(a, b, 2).allclose(la::matmul_classic(a, b)));
}

void test_inverse_lu() {
    la::Matrix<double> a{{4.0, 7.0}, {2.0, 6.0}};
    auto inv = la::inverse_lu(a);
    assert(inv.allclose({{0.6, -0.7}, {-0.2, 0.4}}));
    assert(la::matmul(a, inv).allclose(la::Matrix<double>::eye(2)));
}

void test_inverse_schur() {
    la::Matrix<double> a{
        {8.0, 1.0, 2.0, 0.5},
        {1.0, 7.0, 0.5, 1.0},
        {2.0, 0.5, 6.0, 1.0},
        {0.5, 1.0, 1.0, 5.0}
    };
    auto inv = la::inverse_schur(a, 2);
    const auto product = la::matmul(a, inv);
    assert(product.allclose(la::Matrix<double>::eye(4), 1e-8, 1e-8));
    assert(la::residual_norm(a, inv) < 1e-10);
    assert(la::condition_number_estimate(a) > 1.0);

    auto inv_strassen = la::inverse_schur_strassen(a, 2, 2);
    assert(la::matmul(a, inv_strassen).allclose(la::Matrix<double>::eye(4), 1e-8, 1e-8));
    assert(!la::hardware_backend().empty());
    assert(!la::cpu_optimization_summary().empty());
    assert(la::auto_strassen_min() >= 1);
    assert(la::auto_strassen_base() >= 1);
}

void test_solve() {
    la::Matrix<double> a{{3.0, 2.0}, {1.0, 2.0}};
    la::Matrix<double> b{{5.0}, {5.0}};
    auto x = la::solve(a, b);
    assert(x.allclose({{0.0}, {2.5}}));
}

void test_least_squares() {
    la::Matrix<double> a{
        {1.0, 1.0},
        {1.0, 2.0},
        {1.0, 3.0},
        {1.0, 4.0}
    };
    la::Matrix<double> b{{6.0}, {5.0}, {7.0}, {10.0}};
    auto x = la::least_squares(a, b);
    assert(x.allclose({{3.5}, {1.4}}, 1e-8, 1e-8));
}

void test_inverse_regularized() {
    la::Matrix<double> a{{1.0, 1.0}, {1.0, 1.0}};
    auto inv = la::inverse_regularized(a, 1e-3);
    assert(inv.rows() == 2);
    assert(inv.cols() == 2);
}

} // namespace

int main() {
    test_basic_ops();
    test_matmul();
    test_strassen_matches_classic();
    test_inverse_lu();
    test_inverse_schur();
    test_solve();
    test_least_squares();
    test_inverse_regularized();

    std::cout << "all linx tests passed\n";
    return 0;
}
