#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// ── Apple Accelerate BLAS/LAPACK 통합 ───────────────────────────────────────
// macOS / iOS 환경에서 자동으로 BLAS 가속을 사용합니다.
// 링크 시 -framework Accelerate 플래그가 필요합니다.
// Apple이 macOS 13.3부터 CLAPACK(dgetrf_, cblas_dgemm 등)을 deprecated로
// 표기했지만, 기존 심볼은 여전히 정상 동작합니다.
// 빌드 시스템(CMake/setup.py)에서 -Wno-deprecated-declarations로 경고를 억제합니다.
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#define LINX_HAS_BLAS 1
#else
#define LINX_HAS_BLAS 0
#endif

namespace la {

class LinAlgError : public std::runtime_error {
public:
    explicit LinAlgError(const std::string& message) : std::runtime_error(message) {}
};

class ShapeError : public std::runtime_error {
public:
    explicit ShapeError(const std::string& message) : std::runtime_error(message) {}
};

template <typename T = double>
class Matrix {
public:
    using value_type = T;

    Matrix() = default;

    Matrix(std::size_t rows, std::size_t cols, T value = T{})
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    Matrix(std::initializer_list<std::initializer_list<T>> values) {
        rows_ = values.size();
        cols_ = rows_ == 0 ? 0 : values.begin()->size();
        data_.reserve(rows_ * cols_);

        for (const auto& row : values) {
            if (row.size() != cols_) {
                throw ShapeError("all rows in an initializer list must have the same length");
            }
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }

    static Matrix zeros(std::size_t rows, std::size_t cols) {
        return Matrix(rows, cols, T{});
    }

    static Matrix ones(std::size_t rows, std::size_t cols) {
        return Matrix(rows, cols, T{1});
    }

    static Matrix eye(std::size_t n) {
        Matrix out(n, n, T{});
        for (std::size_t i = 0; i < n; ++i) {
            out(i, i) = T{1};
        }
        return out;
    }

    static Matrix arange(std::size_t rows, std::size_t cols, T start = T{}, T step = T{1}) {
        Matrix out(rows, cols);
        T value = start;
        for (auto& item : out.data_) {
            item = value;
            value += step;
        }
        return out;
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::pair<std::size_t, std::size_t> shape() const noexcept { return {rows_, cols_}; }
    bool empty() const noexcept { return data_.empty(); }
    bool square() const noexcept { return rows_ == cols_; }

    T& operator()(std::size_t row, std::size_t col) {
        assert(row < rows_ && col < cols_);
        return data_[row * cols_ + col];
    }

    const T& operator()(std::size_t row, std::size_t col) const {
        assert(row < rows_ && col < cols_);
        return data_[row * cols_ + col];
    }

    const std::vector<T>& data() const noexcept { return data_; }
    std::vector<T>& data() noexcept { return data_; }

    Matrix reshape(std::size_t rows, std::size_t cols) const {
        if (rows * cols != data_.size()) {
            throw ShapeError("reshape cannot change total element count");
        }
        Matrix out = *this;
        out.rows_ = rows;
        out.cols_ = cols;
        return out;
    }

    Matrix transpose() const {
        Matrix out(cols_, rows_);
        const std::size_t rows = rows_;
        const std::size_t cols = cols_;
#if LINX_HAS_BLAS
        if constexpr (std::is_same<T, double>::value) {
            vDSP_mtransD(data_.data(), 1, out.data_.data(), 1,
                         static_cast<vDSP_Length>(cols),
                         static_cast<vDSP_Length>(rows));
            return out;
        }
#endif
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                out(c, r) = (*this)(r, c);
            }
        }
        return out;
    }

    Matrix block(std::size_t row0, std::size_t col0, std::size_t block_rows, std::size_t block_cols) const {
        if (row0 + block_rows > rows_ || col0 + block_cols > cols_) {
            throw ShapeError("block range is outside matrix bounds");
        }
        Matrix out(block_rows, block_cols);
        for (std::size_t r = 0; r < block_rows; ++r) {
            for (std::size_t c = 0; c < block_cols; ++c) {
                out(r, c) = (*this)(row0 + r, col0 + c);
            }
        }
        return out;
    }

    void set_block(std::size_t row0, std::size_t col0, const Matrix& value) {
        if (row0 + value.rows_ > rows_ || col0 + value.cols_ > cols_) {
            throw ShapeError("set_block range is outside matrix bounds");
        }
        for (std::size_t r = 0; r < value.rows_; ++r) {
            for (std::size_t c = 0; c < value.cols_; ++c) {
                (*this)(row0 + r, col0 + c) = value(r, c);
            }
        }
    }

    T sum() const {
        return std::accumulate(data_.begin(), data_.end(), T{});
    }

    T max_abs() const {
        T out = T{};
        for (const auto& value : data_) {
            out = std::max<T>(out, std::abs(value));
        }
        return out;
    }

    bool allclose(const Matrix& rhs, T rtol = static_cast<T>(1e-5), T atol = static_cast<T>(1e-8)) const {
        if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
            return false;
        }
        for (std::size_t i = 0; i < data_.size(); ++i) {
            const T tolerance = atol + rtol * std::abs(rhs.data_[i]);
            if (std::abs(data_[i] - rhs.data_[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    Matrix operator+(const Matrix& rhs) const {
        require_same_shape(rhs, "add");
        Matrix out(rows_, cols_);
#if LINX_HAS_BLAS
        if constexpr (std::is_same<T, double>::value) {
            vDSP_vaddD(data_.data(), 1, rhs.data_.data(), 1,
                       out.data_.data(), 1, data_.size());
            return out;
        }
#endif
        for (std::size_t i = 0; i < data_.size(); ++i) {
            out.data_[i] = data_[i] + rhs.data_[i];
        }
        return out;
    }

    Matrix operator-(const Matrix& rhs) const {
        require_same_shape(rhs, "subtract");
        Matrix out(rows_, cols_);
#if LINX_HAS_BLAS
        if constexpr (std::is_same<T, double>::value) {
            vDSP_vsubD(rhs.data_.data(), 1, data_.data(), 1,
                       out.data_.data(), 1, data_.size());
            return out;
        }
#endif
        for (std::size_t i = 0; i < data_.size(); ++i) {
            out.data_[i] = data_[i] - rhs.data_[i];
        }
        return out;
    }

    Matrix operator-() const {
        Matrix out(rows_, cols_);
#if LINX_HAS_BLAS
        if constexpr (std::is_same<T, double>::value) {
            vDSP_vnegD(data_.data(), 1, out.data_.data(), 1, data_.size());
            return out;
        }
#endif
        for (std::size_t i = 0; i < data_.size(); ++i) {
            out.data_[i] = -data_[i];
        }
        return out;
    }

    Matrix operator*(T scalar) const {
        Matrix out(rows_, cols_);
#if LINX_HAS_BLAS
        if constexpr (std::is_same<T, double>::value) {
            double s = static_cast<double>(scalar);
            vDSP_vsmulD(data_.data(), 1, &s,
                        out.data_.data(), 1, data_.size());
            return out;
        }
#endif
        for (std::size_t i = 0; i < data_.size(); ++i) {
            out.data_[i] = data_[i] * scalar;
        }
        return out;
    }

    Matrix operator/(T scalar) const {
        if (std::abs(scalar) <= std::numeric_limits<T>::epsilon()) {
            throw LinAlgError("division by a near-zero scalar");
        }
        return (*this) * (T{1} / scalar);
    }

    Matrix hadamard(const Matrix& rhs) const {
        require_same_shape(rhs, "multiply elementwise");
        Matrix out(rows_, cols_);
#if LINX_HAS_BLAS
        if constexpr (std::is_same<T, double>::value) {
            vDSP_vmulD(data_.data(), 1, rhs.data_.data(), 1,
                       out.data_.data(), 1, data_.size());
            return out;
        }
#endif
        for (std::size_t i = 0; i < data_.size(); ++i) {
            out.data_[i] = data_[i] * rhs.data_[i];
        }
        return out;
    }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<T> data_;

    void require_same_shape(const Matrix& rhs, const char* op) const {
        if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
            throw ShapeError(std::string("cannot ") + op + " matrices with different shapes");
        }
    }
};

template <typename T>
Matrix<T> operator*(T scalar, const Matrix<T>& matrix) {
    return matrix * scalar;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {
    os << "[";
    for (std::size_t r = 0; r < matrix.rows(); ++r) {
        if (r != 0) {
            os << " ";
        }
        os << "[";
        for (std::size_t c = 0; c < matrix.cols(); ++c) {
            os << std::setw(10) << matrix(r, c);
            if (c + 1 != matrix.cols()) {
                os << ", ";
            }
        }
        os << "]";
        if (r + 1 != matrix.rows()) {
            os << "\n";
        }
    }
    os << "]";
    return os;
}

namespace detail {

inline std::size_t available_threads() {
    const auto count = std::thread::hardware_concurrency();
    return count == 0 ? 1 : static_cast<std::size_t>(count);
}

template <typename Fn>
void parallel_for_rows(std::size_t rows, std::size_t min_work, std::size_t work, Fn&& fn) {
    const std::size_t thread_count = std::min(rows, available_threads());
    if (rows == 0 || thread_count <= 1 || work < min_work) {
        fn(0, rows);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(thread_count - 1);

    const std::size_t chunk = (rows + thread_count - 1) / thread_count;
    std::size_t begin = 0;
    for (std::size_t t = 0; t + 1 < thread_count && begin < rows; ++t) {
        const std::size_t end = std::min(begin + chunk, rows);
        threads.emplace_back([begin, end, &fn]() {
            fn(begin, end);
        });
        begin = end;
    }

    fn(begin, rows);
    for (auto& thread : threads) {
        thread.join();
    }
}

template <typename T>
T dot_contiguous(const T* lhs, const T* rhs, std::size_t n) {
    T sum = T{};
    for (std::size_t i = 0; i < n; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

inline double dot_contiguous(const double* lhs, const double* rhs, std::size_t n) {
    std::size_t i = 0;
    double sum = 0.0;

#if defined(__AVX2__) || defined(__AVX__)
    __m256d acc = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4) {
        const __m256d a = _mm256_loadu_pd(lhs + i);
        const __m256d b = _mm256_loadu_pd(rhs + i);
        acc = _mm256_add_pd(acc, _mm256_mul_pd(a, b));
    }
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, acc);
    sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
#elif defined(__aarch64__) || defined(__ARM_NEON)
    float64x2_t acc = vdupq_n_f64(0.0);
    for (; i + 2 <= n; i += 2) {
        const float64x2_t a = vld1q_f64(lhs + i);
        const float64x2_t b = vld1q_f64(rhs + i);
        acc = vaddq_f64(acc, vmulq_f64(a, b));
    }
    sum = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
#endif

    for (; i < n; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

} // namespace detail

inline std::string hardware_backend() {
#if LINX_HAS_BLAS
    return "Apple Accelerate BLAS/LAPACK";
#elif defined(__AVX2__)
    return "AVX2 SIMD + std::thread";
#elif defined(__AVX__)
    return "AVX SIMD + std::thread";
#elif defined(__aarch64__) || defined(__ARM_NEON)
    return "ARM NEON SIMD + std::thread";
#else
    return "scalar kernel + std::thread";
#endif
}

template <typename T>
Matrix<T> matmul_classic(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    if (lhs.cols() != rhs.rows()) {
        throw ShapeError("matmul requires lhs.cols == rhs.rows");
    }

    Matrix<T> out(lhs.rows(), rhs.cols(), T{});
    const std::size_t rows = lhs.rows();
    const std::size_t inner = lhs.cols();
    const std::size_t cols = rhs.cols();
    const std::size_t work = rows * inner * cols;

    if constexpr (std::is_same<T, double>::value) {
#if LINX_HAS_BLAS
        // Apple Accelerate BLAS를 이용한 고성능 행렬 곱셈
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(rows), static_cast<int>(cols), static_cast<int>(inner),
                    1.0, lhs.data().data(), static_cast<int>(inner),
                    rhs.data().data(), static_cast<int>(cols),
                    0.0, out.data().data(), static_cast<int>(cols));
#else
        // BLAS가 없으면 기존 SIMD + 스레드 경로 사용
        const Matrix<T> rhs_t = rhs.transpose();  // 복사하여 소유권 확보
        const auto& lhs_data = lhs.data();
        const auto& rhs_data = rhs_t.data();
        auto& out_data = out.data();

        auto worker = [&](std::size_t row_begin, std::size_t row_end) {
            for (std::size_t r = row_begin; r < row_end; ++r) {
                const double* lhs_row = lhs_data.data() + r * inner;
                for (std::size_t c = 0; c < cols; ++c) {
                    const double* rhs_row = rhs_data.data() + c * inner;
                    out_data[r * cols + c] = detail::dot_contiguous(lhs_row, rhs_row, inner);
                }
            }
        };

        detail::parallel_for_rows(rows, 64 * 64 * 64, work, worker);
#endif
        return out;
    } else {
        constexpr std::size_t block = 32;
        const auto& lhs_data = lhs.data();
        const auto& rhs_data = rhs.data();
        auto& out_data = out.data();

        auto worker = [&](std::size_t row_begin, std::size_t row_end) {
            for (std::size_t ii = row_begin; ii < row_end; ii += block) {
                for (std::size_t kk = 0; kk < inner; kk += block) {
                    for (std::size_t jj = 0; jj < cols; jj += block) {
                        const std::size_t i_end = std::min(ii + block, row_end);
                        const std::size_t k_end = std::min(kk + block, inner);
                        const std::size_t j_end = std::min(jj + block, cols);

                        for (std::size_t i = ii; i < i_end; ++i) {
                            for (std::size_t k = kk; k < k_end; ++k) {
                                const T a = lhs_data[i * inner + k];
                                for (std::size_t j = jj; j < j_end; ++j) {
                                    out_data[i * cols + j] += a * rhs_data[k * cols + j];
                                }
                            }
                        }
                    }
                }
            }
        };

        detail::parallel_for_rows(rows, 64 * 64 * 64, work, worker);
        return out;
    }
}

namespace detail {

inline std::size_t next_power_of_two(std::size_t n) {
    if (n == 0) return 1;  // 방어: 0 입력 시 1 반환
    std::size_t out = 1;
    while (out < n) {
        out <<= 1;
    }
    return out;
}

template <typename T>
Matrix<T> pad_square(const Matrix<T>& input, std::size_t size) {
    Matrix<T> out(size, size, T{});
    out.set_block(0, 0, input);
    return out;
}

template <typename T>
Matrix<T> strassen_square(const Matrix<T>& lhs, const Matrix<T>& rhs, std::size_t threshold) {
    const std::size_t n = lhs.rows();
    if (n <= threshold) {
        return matmul_classic(lhs, rhs);
    }

    const std::size_t h = n / 2;
    const auto a = lhs.block(0, 0, h, h);
    const auto b = lhs.block(0, h, h, h);
    const auto c = lhs.block(h, 0, h, h);
    const auto d = lhs.block(h, h, h, h);
    const auto e = rhs.block(0, 0, h, h);
    const auto f = rhs.block(0, h, h, h);
    const auto g = rhs.block(h, 0, h, h);
    const auto i = rhs.block(h, h, h, h);

    const auto p1 = strassen_square(a, f - i, threshold);
    const auto p2 = strassen_square(a + b, i, threshold);
    const auto p3 = strassen_square(c + d, e, threshold);
    const auto p4 = strassen_square(d, g - e, threshold);
    const auto p5 = strassen_square(a + d, e + i, threshold);
    const auto p6 = strassen_square(b - d, g + i, threshold);
    const auto p7 = strassen_square(a - c, e + f, threshold);

    Matrix<T> out(n, n);
    out.set_block(0, 0, p5 + p4 - p2 + p6);
    out.set_block(0, h, p1 + p2);
    out.set_block(h, 0, p3 + p4);
    out.set_block(h, h, p1 + p5 - p3 - p7);
    return out;
}


// ── LAPACK을 이용한 빠른 역행렬 (double, Apple Accelerate) ──────────────
#if LINX_HAS_BLAS
inline Matrix<double> inverse_lapack(const Matrix<double>& matrix, double eps = 1e-12) {
    if (!matrix.square()) {
        throw ShapeError("inverse_lapack requires a square matrix");
    }
    const std::size_t n = matrix.rows();
    if (n == 0) return matrix;

    // dgetrf_ / dgetri_ 는 column-major 행렬을 요구하므로, row-major를 column-major로
    // 변환하기 위해 transpose 한 후 column-major 포인터로 전달합니다.
    Matrix<double> a_col = matrix.transpose();  // data()가 column-major로 해석됨
    
    int n_int = static_cast<int>(n);
    std::vector<int> ipiv(n_int);
    int info = 0;

    // LU 분해 (dgetrf)
    dgetrf_(&n_int, &n_int, a_col.data().data(), &n_int, ipiv.data(), &info);
    if (info != 0) {
        throw LinAlgError("dgetrf failed: matrix is singular or ill-conditioned");
    }

    // 역행렬 계산을 위한 작업 공간 쿼리
    int lwork = -1;
    double work_query = 0.0;
    dgetri_(&n_int, a_col.data().data(), &n_int, ipiv.data(), &work_query, &lwork, &info);
    if (info != 0) {
        throw LinAlgError("dgetri workspace query failed");
    }

    lwork = static_cast<int>(work_query);
    std::vector<double> work(lwork);

    // 역행렬 계산 (dgetri)
    dgetri_(&n_int, a_col.data().data(), &n_int, ipiv.data(), work.data(), &lwork, &info);
    if (info != 0) {
        throw LinAlgError("dgetri failed: matrix is singular");
    }

    // column-major 결과를 row-major로 되돌림 (transpose)
    return a_col.transpose();
}
#endif

} // namespace detail

template <typename T>
Matrix<T> matmul_strassen(const Matrix<T>& lhs, const Matrix<T>& rhs, std::size_t threshold = 64) {
    if (lhs.cols() != rhs.rows()) {
        throw ShapeError("matmul requires lhs.cols == rhs.rows");
    }

    const std::size_t size = detail::next_power_of_two(
        std::max({lhs.rows(), lhs.cols(), rhs.cols(), std::size_t{1}})
    );

    const auto padded_lhs = detail::pad_square(lhs, size);
    const auto padded_rhs = detail::pad_square(rhs, size);
    return detail::strassen_square(padded_lhs, padded_rhs, threshold)
        .block(0, 0, lhs.rows(), rhs.cols());
}

template <typename T>
Matrix<T> matmul(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    const std::size_t work = lhs.rows() * lhs.cols() * rhs.cols();
    if (lhs.rows() == lhs.cols() && rhs.rows() == rhs.cols() && lhs.cols() == rhs.rows() && work >= 128 * 128 * 128) {
        return matmul_strassen(lhs, rhs);
    }
    return matmul_classic(lhs, rhs);
}

template <typename T>
Matrix<T> solve(Matrix<T> a, Matrix<T> b, T eps = static_cast<T>(1e-12)) {
    if (!a.square()) {
        throw ShapeError("solve requires a square coefficient matrix");
    }
    if (a.rows() != b.rows()) {
        throw ShapeError("solve requires a.rows == b.rows");
    }

    const std::size_t n = a.rows();
    const std::size_t m = b.cols();

    // ── double + Apple Accelerate → LAPACK dgesv (최대 50배 가속) ────────
#if LINX_HAS_BLAS
    if constexpr (std::is_same<T, double>::value) {
        Matrix<double> a_col = a.transpose();  // row-major → column-major
        Matrix<double> b_col = b.transpose();  // multiple RHS도 column-major로

        int n_int = static_cast<int>(n);
        int nrhs = static_cast<int>(m);
        std::vector<int> ipiv(n_int);
        int info = 0;

        dgesv_(&n_int, &nrhs, a_col.data().data(), &n_int, ipiv.data(),
               b_col.data().data(), &n_int, &info);

        if (info < 0) {
            throw LinAlgError("dgesv: illegal argument");
        }
        if (info > 0) {
            throw LinAlgError("dgesv: matrix is singular (zero pivot)");
        }

        // column-major 결과를 row-major로 변환하여 반환
        return b_col.transpose();
    }
#endif

    // ── 일반적인 Gauss-Jordan with partial pivoting ──────────────────────
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        T pivot_abs = std::abs(a(col, col));
        for (std::size_t r = col + 1; r < n; ++r) {
            const T candidate = std::abs(a(r, col));
            if (candidate > pivot_abs) {
                pivot = r;
                pivot_abs = candidate;
            }
        }

        if (pivot_abs <= eps) {
            throw LinAlgError("matrix is singular or ill-conditioned");
        }

        if (pivot != col) {
            for (std::size_t c = 0; c < n; ++c) {
                std::swap(a(col, c), a(pivot, c));
            }
            for (std::size_t c = 0; c < m; ++c) {
                std::swap(b(col, c), b(pivot, c));
            }
        }

        const T diag = a(col, col);
        for (std::size_t c = 0; c < n; ++c) {
            a(col, c) /= diag;
        }
        for (std::size_t c = 0; c < m; ++c) {
            b(col, c) /= diag;
        }

        for (std::size_t r = 0; r < n; ++r) {
            if (r == col) {
                continue;
            }
            const T factor = a(r, col);
            if (std::abs(factor) <= eps) {
                continue;
            }
            for (std::size_t c = 0; c < n; ++c) {
                a(r, c) -= factor * a(col, c);
            }
            for (std::size_t c = 0; c < m; ++c) {
                b(r, c) -= factor * b(col, c);
            }
        }
    }

    return b;
}

template <typename T>
Matrix<T> inverse_lu(const Matrix<T>& matrix, T eps = static_cast<T>(1e-12)) {
    if (!matrix.square()) {
        throw ShapeError("inverse requires a square matrix");
    }
    return solve(matrix, Matrix<T>::eye(matrix.rows()), eps);
}

template <typename T>
Matrix<T> inverse_schur(const Matrix<T>& matrix, std::size_t min_block = 32, T eps = static_cast<T>(1e-12)) {
    if (!matrix.square()) {
        throw ShapeError("inverse_schur requires a square matrix");
    }

    const std::size_t n = matrix.rows();

    // ── 작은 행렬 (n ≤ 512) → LAPACK / LU로 직접 처리 ─────────────────
    if (n <= 512) {
#if LINX_HAS_BLAS
        if constexpr (std::is_same<T, double>::value) {
            return detail::inverse_lapack(matrix, eps);
        }
#endif
        return inverse_lu(matrix, eps);
    }

    // ── base case: min_block 이하 or 홀수 → LU 분해 ────────────────────
    if (n <= min_block || n % 2 != 0) {
#if LINX_HAS_BLAS
        if constexpr (std::is_same<T, double>::value) {
            return detail::inverse_lapack(matrix, eps);
        }
#endif
        return inverse_lu(matrix, eps);
    }

    const std::size_t h = n / 2;
    const auto a = matrix.block(0, 0, h, h);
    const auto b = matrix.block(0, h, h, h);
    const auto c = matrix.block(h, 0, h, h);
    const auto d = matrix.block(h, h, h, h);

    const auto a_inv = inverse_schur(a, min_block, eps);
    const auto ca_inv = matmul(c, a_inv);
    const auto schur = d - matmul(ca_inv, b);
    const auto s_inv = inverse_schur(schur, min_block, eps);

    const auto a_inv_b = matmul(a_inv, b);
    const auto top_left = a_inv + matmul(matmul(a_inv_b, s_inv), ca_inv);
    const auto top_right = -matmul(a_inv_b, s_inv);
    const auto bottom_left = -matmul(s_inv, ca_inv);

    Matrix<T> out(n, n);
    out.set_block(0, 0, top_left);
    out.set_block(0, h, top_right);
    out.set_block(h, 0, bottom_left);
    out.set_block(h, h, s_inv);
    return out;
}

template <typename T>
Matrix<T> inverse(const Matrix<T>& matrix, T eps = static_cast<T>(1e-12)) {
    return inverse_schur(matrix, 32, eps);
}

template <typename T>
T frobenius_norm(const Matrix<T>& matrix) {
#if LINX_HAS_BLAS
    if constexpr (std::is_same<T, double>::value) {
        const vDSP_Length n = matrix.data().size();
        double sum_sq = 0.0;
        vDSP_svesqD(matrix.data().data(), 1, &sum_sq, n);
        return static_cast<T>(std::sqrt(sum_sq));
    }
#endif
    T sum = T{};
    for (const auto& value : matrix.data()) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

template <typename T>
T condition_number_estimate(const Matrix<T>& matrix, T eps = static_cast<T>(1e-12)) {
    return frobenius_norm(matrix) * frobenius_norm(inverse(matrix, eps));
}

template <typename T>
Matrix<T> inverse_regularized(const Matrix<T>& matrix, T lambda = static_cast<T>(1e-8), T eps = static_cast<T>(1e-12)) {
    if (!matrix.square()) {
        throw ShapeError("inverse_regularized requires a square matrix");
    }

    auto adjusted = matrix;
    for (std::size_t i = 0; i < adjusted.rows(); ++i) {
        adjusted(i, i) += lambda;
    }
    return inverse(adjusted, eps);
}

template <typename T>
T residual_norm(const Matrix<T>& matrix, const Matrix<T>& inverse_matrix) {
    return frobenius_norm(matmul(matrix, inverse_matrix) - Matrix<T>::eye(matrix.rows()));
}

template <typename T>
T trace(const Matrix<T>& matrix) {
    if (!matrix.square()) {
        throw ShapeError("trace requires a square matrix");
    }
    T sum = T{};
    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        sum += matrix(i, i);
    }
    return sum;
}

template <typename T>
T det(const Matrix<T>& matrix, T eps = static_cast<T>(1e-12)) {
    if (!matrix.square()) {
        throw ShapeError("det requires a square matrix");
    }
    const std::size_t n = matrix.rows();
    if (n == 0) return T{1};

#if LINX_HAS_BLAS
    if constexpr (std::is_same<T, double>::value) {
        // dgetrf로 LU 분해 후 대각원소의 곱으로 행렬식 계산
        Matrix<double> a_col = matrix.transpose();  // column-major 변환
        int n_int = static_cast<int>(n);
        std::vector<int> ipiv(n_int);
        int info = 0;

        dgetrf_(&n_int, &n_int, a_col.data().data(), &n_int, ipiv.data(), &info);
        if (info != 0) {
            throw LinAlgError("det: LU factorization failed");
        }

        T det_val = T{1};
        for (std::size_t i = 0; i < n; ++i) {
            det_val *= a_col(i, i);
            if (ipiv[i] != static_cast<int>(i + 1)) {  // Fortran 1-based pivot
                det_val = -det_val;
            }
        }
        return det_val;
    }
#endif

    // fallback: solve의 Gauss-Jordan 경로 활용 (교육용 — 큰 행렬은 느림)
    Matrix<T> a = matrix;  // 복사본으로 QR-like 분해
    T det_val = T{1};

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        T pivot_abs = std::abs(a(col, col));
        for (std::size_t r = col + 1; r < n; ++r) {
            const T candidate = std::abs(a(r, col));
            if (candidate > pivot_abs) {
                pivot = r;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs <= eps) {
            return T{};  // singular → 행렬식 0
        }
        if (pivot != col) {
            for (std::size_t c = 0; c < n; ++c) std::swap(a(col, c), a(pivot, c));
            det_val = -det_val;
        }
        det_val *= a(col, col);
        for (std::size_t r = col + 1; r < n; ++r) {
            const T factor = a(r, col) / a(col, col);
            for (std::size_t c = col; c < n; ++c) a(r, c) -= factor * a(col, c);
        }
    }
    return det_val;
}

} // namespace la
