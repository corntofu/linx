#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#define LINX_X86_SIMD 1
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#else
#define LINX_X86_SIMD 0
#endif

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if LINX_X86_SIMD && (defined(_M_X64) || defined(__x86_64__) || defined(__SSE2__) || \
                      (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#define LINX_COMPILED_SSE2 1
#else
#define LINX_COMPILED_SSE2 0
#endif

#if LINX_X86_SIMD && (defined(__AVX__) || defined(_M_AVX) || \
                      (!defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__))))
#define LINX_COMPILED_AVX 1
#else
#define LINX_COMPILED_AVX 0
#endif

#if LINX_X86_SIMD && (defined(__AVX2__) || \
                      (!defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__))))
#define LINX_COMPILED_AVX2 1
#else
#define LINX_COMPILED_AVX2 0
#endif

#if LINX_X86_SIMD && (defined(__FMA__) || defined(__AVX2__) || \
                      (!defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__))))
#define LINX_COMPILED_FMA 1
#else
#define LINX_COMPILED_FMA 0
#endif

#if LINX_X86_SIMD && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#define LINX_TARGET_SSE2 __attribute__((target("sse2")))
#define LINX_TARGET_AVX __attribute__((target("avx")))
#define LINX_TARGET_AVX2 __attribute__((target("avx2")))
#define LINX_TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
#else
#define LINX_TARGET_SSE2
#define LINX_TARGET_AVX
#define LINX_TARGET_AVX2
#define LINX_TARGET_AVX2_FMA
#endif

// ── Apple Accelerate BLAS/LAPACK 통합 ───────────────────────────────────────
// macOS / iOS 환경에서 자동으로 BLAS 가속을 사용합니다.
// 링크 시 -framework Accelerate 플래그가 필요합니다.
// Apple이 macOS 13.3부터 CLAPACK(dgetrf_, cblas_dgemm 등)을 deprecated로
// 표기했지만, 기존 심볼은 여전히 정상 동작합니다.
// 빌드 시스템(CMake/setup.py)에서 -Wno-deprecated-declarations로 경고를 억제합니다.
#if defined(__APPLE__) || defined(LINX_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#define LINX_HAS_BLAS 1
#else
#define LINX_HAS_BLAS 0
#endif

// ── M2‑specific performance tuning ───────────────────────────────────────
// Apple M1/M2/M3 have 4 Firestorm (performance) + 4 Icestorm (efficiency)
// cores with 128‑bit NEON. We tune for: 4 P‑cores, 128 KB L1d, shared L2.
#if defined(__aarch64__) && defined(__APPLE__)
  #define LINX_M2_TILE_ROWS    64    // fits in L1d: 64×16 doubles = 8 KB
  #define LINX_M2_TILE_COLS    256   // 64×256 = 16 384 doubles = 128 KB (fits L2 slice)
  #define LINX_M2_L1_BYTES     131072
  #define LINX_M2_P_CORES      4
  #define LINX_M2_UNROLL       8     // unroll factor for NEON matmul micro‑kernel
  #define LINX_M2_STRASSEN_MIN 4096  // use Strassen for N > 4096
  #define LINX_M2_STRASSEN_BASE 4096
  #define LINX_LAPACK_INVERSE_MAX 1024
#else
  #define LINX_M2_TILE_ROWS    32
  #define LINX_M2_TILE_COLS    128
  #define LINX_M2_L1_BYTES     32768
  #define LINX_M2_P_CORES      2
  #define LINX_M2_UNROLL       4
  #define LINX_M2_STRASSEN_MIN 4096
  #define LINX_M2_STRASSEN_BASE 4096
  #define LINX_LAPACK_INVERSE_MAX 1024
#endif

namespace la {

namespace detail {
inline double dot_contiguous(const double* lhs, const double* rhs, std::size_t n);
inline void vector_add_d(double* dst, const double* a, const double* b, std::size_t n);
inline void vector_sub_d(double* dst, const double* a, const double* b, std::size_t n);
inline void vector_mul_d(double* dst, const double* a, const double* b, std::size_t n);
inline void vector_neg_d(double* dst, const double* a, std::size_t n);
inline void vector_scale_d(double* dst, const double* a, double s, std::size_t n);
inline std::size_t runtime_strassen_min();
inline std::size_t runtime_strassen_base();
} // namespace detail

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
        // Use memcpy for whole rows when block_cols == cols_ (full‑width extract)
        if (block_cols == cols_ && !out.data_.empty()) {
            const std::size_t row_bytes = cols_ * sizeof(T);
            for (std::size_t r = 0; r < block_rows; ++r) {
                std::memcpy(out.data_.data() + r * cols_,
                            data_.data() + (row0 + r) * cols_,
                            row_bytes);
            }
            return out;
        }
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
        // Full-width write can use memcpy
        if (value.cols_ == cols_ && !value.data_.empty()) {
            const std::size_t row_bytes = cols_ * sizeof(T);
            for (std::size_t r = 0; r < value.rows_; ++r) {
                std::memcpy(data_.data() + (row0 + r) * cols_,
                            value.data_.data() + r * cols_,
                            row_bytes);
            }
            return;
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
#else
        if constexpr (std::is_same<T, double>::value) {
            detail::vector_add_d(out.data_.data(), data_.data(), rhs.data_.data(), data_.size());
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
#else
        if constexpr (std::is_same<T, double>::value) {
            detail::vector_sub_d(out.data_.data(), data_.data(), rhs.data_.data(), data_.size());
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
#else
        if constexpr (std::is_same<T, double>::value) {
            detail::vector_neg_d(out.data_.data(), data_.data(), data_.size());
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
#else
        if constexpr (std::is_same<T, double>::value) {
            detail::vector_scale_d(out.data_.data(), data_.data(), static_cast<double>(scalar), data_.size());
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
#else
        if constexpr (std::is_same<T, double>::value) {
            detail::vector_mul_d(out.data_.data(), data_.data(), rhs.data_.data(), data_.size());
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

enum class SimdKernel {
    Scalar,
    SSE2,
    AVX,
    AVX2,
    AVX2FMA,
};

struct CpuFeatures {
    bool is_x86 = false;
    bool is_windows = false;
    bool is_intel = false;
    bool windows_intel_profile = false;
    bool sse2 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool os_avx = false;
    std::size_t l1d_bytes = 0;
    std::size_t preferred_threads = 1;
    SimdKernel selected_kernel = SimdKernel::Scalar;
    std::string vendor = "unknown";
};

inline bool kernel_uses_avx(SimdKernel kernel) {
    return kernel == SimdKernel::AVX ||
           kernel == SimdKernel::AVX2 ||
           kernel == SimdKernel::AVX2FMA;
}

inline bool kernel_uses_avx2(SimdKernel kernel) {
    return kernel == SimdKernel::AVX2 || kernel == SimdKernel::AVX2FMA;
}

inline const char* simd_kernel_name(SimdKernel kernel) {
    switch (kernel) {
        case SimdKernel::AVX2FMA: return "AVX2/FMA";
        case SimdKernel::AVX2: return "AVX2";
        case SimdKernel::AVX: return "AVX";
        case SimdKernel::SSE2: return "SSE2";
        case SimdKernel::Scalar:
        default:
            return "scalar";
    }
}

#if LINX_X86_SIMD
inline std::uint32_t reg_u32(const std::array<int, 4>& regs, std::size_t index) {
    return static_cast<std::uint32_t>(regs[index]);
}

inline std::array<int, 4> cpuid_regs(int leaf, int subleaf = 0) {
    std::array<int, 4> regs{0, 0, 0, 0};
#if defined(_MSC_VER)
    __cpuidex(regs.data(), leaf, subleaf);
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    __cpuid_count(static_cast<unsigned int>(leaf),
                  static_cast<unsigned int>(subleaf),
                  eax, ebx, ecx, edx);
    regs = {
        static_cast<int>(eax),
        static_cast<int>(ebx),
        static_cast<int>(ecx),
        static_cast<int>(edx),
    };
#else
    (void)leaf;
    (void)subleaf;
#endif
    return regs;
}

inline std::uint64_t xgetbv_u64(unsigned int index) {
#if defined(_MSC_VER)
    return static_cast<std::uint64_t>(_xgetbv(index));
#elif defined(__GNUC__) || defined(__clang__)
    std::uint32_t eax = 0;
    std::uint32_t edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#else
    (void)index;
    return 0;
#endif
}

inline std::size_t detect_l1d_cache_bytes(std::uint32_t max_leaf) {
    if (max_leaf < 4) {
        return 0;
    }
    for (int subleaf = 0; subleaf < 8; ++subleaf) {
        const auto regs = cpuid_regs(4, subleaf);
        const std::uint32_t eax = reg_u32(regs, 0);
        const std::uint32_t ebx = reg_u32(regs, 1);
        const std::uint32_t ecx = reg_u32(regs, 2);
        const std::uint32_t cache_type = eax & 0x1fu;
        if (cache_type == 0) {
            break;
        }
        const std::uint32_t cache_level = (eax >> 5) & 0x7u;
        if (cache_type == 1 && cache_level == 1) {
            const std::size_t line_size = (ebx & 0xfffu) + 1u;
            const std::size_t partitions = ((ebx >> 12) & 0x3ffu) + 1u;
            const std::size_t ways = ((ebx >> 22) & 0x3ffu) + 1u;
            const std::size_t sets = ecx + 1u;
            return line_size * partitions * ways * sets;
        }
    }
    return 0;
}
#endif

inline CpuFeatures detect_cpu_features() {
    CpuFeatures features;
    features.preferred_threads = available_threads();
#if LINX_X86_SIMD
    features.is_x86 = true;
#if defined(_WIN32)
    features.is_windows = true;
#endif

    const auto leaf0 = cpuid_regs(0);
    const std::uint32_t max_leaf = reg_u32(leaf0, 0);
    char vendor[13] = {};
    std::memcpy(vendor + 0, &leaf0[1], 4);
    std::memcpy(vendor + 4, &leaf0[3], 4);
    std::memcpy(vendor + 8, &leaf0[2], 4);
    features.vendor = vendor;
    features.is_intel = features.vendor == "GenuineIntel";
    features.windows_intel_profile = features.is_windows && features.is_intel;
    features.l1d_bytes = detect_l1d_cache_bytes(max_leaf);

    bool cpu_avx = false;
    bool cpu_fma = false;
    bool cpu_avx2 = false;
    if (max_leaf >= 1) {
        const auto leaf1 = cpuid_regs(1);
        const std::uint32_t ecx = reg_u32(leaf1, 2);
        const std::uint32_t edx = reg_u32(leaf1, 3);
        features.sse2 = (edx & (1u << 26)) != 0;
        cpu_fma = (ecx & (1u << 12)) != 0;
        const bool osxsave = (ecx & (1u << 27)) != 0;
        cpu_avx = (ecx & (1u << 28)) != 0;
        if (osxsave) {
            const std::uint64_t xcr0 = xgetbv_u64(0);
            features.os_avx = (xcr0 & 0x6u) == 0x6u;
        }
    }
    if (max_leaf >= 7) {
        const auto leaf7 = cpuid_regs(7, 0);
        cpu_avx2 = (reg_u32(leaf7, 1) & (1u << 5)) != 0;
    }

    const bool allow_runtime_simd = !features.is_windows || features.is_intel;
    features.avx = allow_runtime_simd && cpu_avx && features.os_avx;
    features.avx2 = features.avx && cpu_avx2;
    features.fma = features.avx && cpu_fma;
    if (!allow_runtime_simd) {
        features.sse2 = false;
    }

#if LINX_COMPILED_SSE2
    if (features.sse2) {
        features.selected_kernel = SimdKernel::SSE2;
    }
#endif
#if LINX_COMPILED_AVX
    if (features.avx) {
        features.selected_kernel = SimdKernel::AVX;
    }
#endif
#if LINX_COMPILED_AVX2
    if (features.avx2) {
        features.selected_kernel = SimdKernel::AVX2;
    }
#endif
#if LINX_COMPILED_FMA
    if (features.avx2 && features.fma) {
        features.selected_kernel = SimdKernel::AVX2FMA;
    }
#endif
#endif
    return features;
}

inline const CpuFeatures& runtime_cpu_features() {
    static const CpuFeatures features = detect_cpu_features();
    return features;
}

inline std::size_t preferred_threads() {
    return std::max<std::size_t>(1, runtime_cpu_features().preferred_threads);
}

inline bool task_parallel_enabled(std::size_t block_size) {
    const char* disabled = std::getenv("LINX_DISABLE_PROCESSOR_PARALLEL");
    if (disabled != nullptr && disabled[0] != '\0' && disabled[0] != '0') {
        return false;
    }
    return preferred_threads() > 1 && block_size >= 512;
}

template <typename Fn>
void parallel_for_rows(std::size_t rows, std::size_t min_work, std::size_t work, Fn&& fn) {
    const std::size_t thread_count = std::min(rows, preferred_threads());
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

inline std::size_t runtime_strassen_base() {
#if LINX_X86_SIMD
    const auto& features = runtime_cpu_features();
    if (features.windows_intel_profile && kernel_uses_avx2(features.selected_kernel)) {
        return 2048;
    }
    if (features.windows_intel_profile && kernel_uses_avx(features.selected_kernel)) {
        return 3072;
    }
#endif
    return static_cast<std::size_t>(LINX_M2_STRASSEN_BASE);
}

inline std::size_t runtime_strassen_min() {
#if LINX_X86_SIMD
    const auto& features = runtime_cpu_features();
    if (features.windows_intel_profile && kernel_uses_avx2(features.selected_kernel)) {
        return 3072;
    }
#endif
    return static_cast<std::size_t>(LINX_M2_STRASSEN_MIN);
}

inline std::string compiled_simd_description() {
#if LINX_COMPILED_FMA
    return "AVX2/FMA target + AVX2 + AVX + SSE2";
#elif LINX_COMPILED_AVX2
    return "AVX2 target + AVX + SSE2";
#elif LINX_COMPILED_AVX
    return "AVX target + SSE2";
#elif LINX_COMPILED_SSE2
    return "SSE2";
#else
    return "scalar";
#endif
}

inline std::string runtime_backend_description() {
#if LINX_X86_SIMD
    const auto& features = runtime_cpu_features();
    std::string out;
    if (features.is_windows) {
        out = features.is_intel
            ? "Windows Intel CPU autodetect"
            : "Windows x86 CPU autodetect (non-Intel scalar fallback)";
    } else {
        out = "x86 CPU autodetect";
    }
    out += ": ";
    out += simd_kernel_name(features.selected_kernel);
    out += " SIMD + std::thread row/task parallel";
    out += " [compiled ";
    out += compiled_simd_description();
    out += "]";
    return out;
#elif defined(__aarch64__) || defined(__ARM_NEON)
    return "ARM NEON SIMD + std::thread row/task parallel";
#else
    return "scalar kernel + std::thread row/task parallel";
#endif
}

inline std::string cpu_optimization_description() {
#if LINX_X86_SIMD
    const auto& features = runtime_cpu_features();
    std::string out = "vendor=" + features.vendor;
    out += ", selected=";
    out += simd_kernel_name(features.selected_kernel);
    out += ", compiled=" + compiled_simd_description();
    out += ", features=";
    bool wrote = false;
    auto append = [&](const char* name, bool enabled) {
        if (!enabled) {
            return;
        }
        if (wrote) {
            out += "/";
        }
        out += name;
        wrote = true;
    };
    append("SSE2", features.sse2);
    append("AVX", features.avx);
    append("AVX2", features.avx2);
    append("FMA", features.fma);
    if (!wrote) {
        out += "scalar";
    }
    out += ", os_avx=";
    out += features.os_avx ? "yes" : "no";
    out += ", windows_intel_profile=";
    out += features.windows_intel_profile ? "enabled" : "disabled";
    out += ", threads=" + std::to_string(features.preferred_threads);
    if (features.l1d_bytes > 0) {
        out += ", l1d_bytes=" + std::to_string(features.l1d_bytes);
    }
    out += ", strassen_min=" + std::to_string(runtime_strassen_min());
    out += ", strassen_base=" + std::to_string(runtime_strassen_base());
    return out;
#elif defined(__aarch64__) || defined(__ARM_NEON)
    return "selected=ARM NEON, threads=" + std::to_string(available_threads());
#else
    return "selected=scalar, threads=" + std::to_string(available_threads());
#endif
}

template <typename T>
T dot_contiguous(const T* lhs, const T* rhs, std::size_t n) {
    T sum = T{};
    for (std::size_t i = 0; i < n; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

inline double dot_scalar_d(const double* lhs, const double* rhs, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

#if LINX_COMPILED_SSE2
LINX_TARGET_SSE2 inline double dot_sse2_d(const double* lhs, const double* rhs, std::size_t n) {
    std::size_t i = 0;
    __m128d acc = _mm_setzero_pd();
    for (; i + 2 <= n; i += 2) {
        const __m128d a = _mm_loadu_pd(lhs + i);
        const __m128d b = _mm_loadu_pd(rhs + i);
        acc = _mm_add_pd(acc, _mm_mul_pd(a, b));
    }
    alignas(16) double lanes[2];
    _mm_store_pd(lanes, acc);
    double sum = lanes[0] + lanes[1];
    for (; i < n; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}
#endif

#if LINX_COMPILED_AVX
LINX_TARGET_AVX inline double dot_avx_d(const double* lhs, const double* rhs, std::size_t n) {
    std::size_t i = 0;
    __m256d acc = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4) {
        const __m256d a = _mm256_loadu_pd(lhs + i);
        const __m256d b = _mm256_loadu_pd(rhs + i);
        acc = _mm256_add_pd(acc, _mm256_mul_pd(a, b));
    }
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, acc);
    double sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < n; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}
#endif

#if LINX_COMPILED_FMA
LINX_TARGET_AVX2_FMA inline double dot_avx2_fma_d(const double* lhs, const double* rhs, std::size_t n) {
    std::size_t i = 0;
    __m256d acc = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4) {
        const __m256d a = _mm256_loadu_pd(lhs + i);
        const __m256d b = _mm256_loadu_pd(rhs + i);
        acc = _mm256_fmadd_pd(a, b, acc);
    }
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, acc);
    double sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < n; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}
#endif

#if defined(__aarch64__) || defined(__ARM_NEON)
inline double dot_neon_d(const double* lhs, const double* rhs, std::size_t n) {
    std::size_t i = 0;
    float64x2_t acc = vdupq_n_f64(0.0);
    for (; i + 2 <= n; i += 2) {
        const float64x2_t a = vld1q_f64(lhs + i);
        const float64x2_t b = vld1q_f64(rhs + i);
        acc = vaddq_f64(acc, vmulq_f64(a, b));
    }
    double sum = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
    for (; i < n; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}
#endif

inline double dot_contiguous(const double* lhs, const double* rhs, std::size_t n) {
#if LINX_X86_SIMD
    const auto kernel = runtime_cpu_features().selected_kernel;
#if LINX_COMPILED_FMA
    if (kernel == SimdKernel::AVX2FMA) {
        return dot_avx2_fma_d(lhs, rhs, n);
    }
#endif
#if LINX_COMPILED_AVX
    if (kernel_uses_avx(kernel)) {
        return dot_avx_d(lhs, rhs, n);
    }
#endif
#if LINX_COMPILED_SSE2
    if (kernel == SimdKernel::SSE2) {
        return dot_sse2_d(lhs, rhs, n);
    }
#endif
#elif defined(__aarch64__) || defined(__ARM_NEON)
    return dot_neon_d(lhs, rhs, n);
#endif
    return dot_scalar_d(lhs, rhs, n);
}

inline void vector_add_scalar_d(double* dst, const double* a, const double* b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = a[i] + b[i];
}

inline void vector_sub_scalar_d(double* dst, const double* a, const double* b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = a[i] - b[i];
}

inline void vector_mul_scalar_d(double* dst, const double* a, const double* b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = a[i] * b[i];
}

inline void vector_neg_scalar_d(double* dst, const double* a, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = -a[i];
}

inline void vector_scale_scalar_d(double* dst, const double* a, double s, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = a[i] * s;
}

#if LINX_COMPILED_SSE2
LINX_TARGET_SSE2 inline void vector_add_sse2_d(double* dst, const double* a, const double* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const __m128d va = _mm_loadu_pd(a + i);
        const __m128d vb = _mm_loadu_pd(b + i);
        _mm_storeu_pd(dst + i, _mm_add_pd(va, vb));
    }
    vector_add_scalar_d(dst + i, a + i, b + i, n - i);
}

LINX_TARGET_SSE2 inline void vector_sub_sse2_d(double* dst, const double* a, const double* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const __m128d va = _mm_loadu_pd(a + i);
        const __m128d vb = _mm_loadu_pd(b + i);
        _mm_storeu_pd(dst + i, _mm_sub_pd(va, vb));
    }
    vector_sub_scalar_d(dst + i, a + i, b + i, n - i);
}

LINX_TARGET_SSE2 inline void vector_mul_sse2_d(double* dst, const double* a, const double* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const __m128d va = _mm_loadu_pd(a + i);
        const __m128d vb = _mm_loadu_pd(b + i);
        _mm_storeu_pd(dst + i, _mm_mul_pd(va, vb));
    }
    vector_mul_scalar_d(dst + i, a + i, b + i, n - i);
}

LINX_TARGET_SSE2 inline void vector_neg_sse2_d(double* dst, const double* a, std::size_t n) {
    std::size_t i = 0;
    const __m128d zero = _mm_setzero_pd();
    for (; i + 2 <= n; i += 2) {
        const __m128d va = _mm_loadu_pd(a + i);
        _mm_storeu_pd(dst + i, _mm_sub_pd(zero, va));
    }
    vector_neg_scalar_d(dst + i, a + i, n - i);
}

LINX_TARGET_SSE2 inline void vector_scale_sse2_d(double* dst, const double* a, double s, std::size_t n) {
    std::size_t i = 0;
    const __m128d vs = _mm_set1_pd(s);
    for (; i + 2 <= n; i += 2) {
        const __m128d va = _mm_loadu_pd(a + i);
        _mm_storeu_pd(dst + i, _mm_mul_pd(va, vs));
    }
    vector_scale_scalar_d(dst + i, a + i, s, n - i);
}
#endif

#if LINX_COMPILED_AVX
LINX_TARGET_AVX inline void vector_add_avx_d(double* dst, const double* a, const double* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d va = _mm256_loadu_pd(a + i);
        const __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(dst + i, _mm256_add_pd(va, vb));
    }
    vector_add_scalar_d(dst + i, a + i, b + i, n - i);
}

LINX_TARGET_AVX inline void vector_sub_avx_d(double* dst, const double* a, const double* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d va = _mm256_loadu_pd(a + i);
        const __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(dst + i, _mm256_sub_pd(va, vb));
    }
    vector_sub_scalar_d(dst + i, a + i, b + i, n - i);
}

LINX_TARGET_AVX inline void vector_mul_avx_d(double* dst, const double* a, const double* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d va = _mm256_loadu_pd(a + i);
        const __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(dst + i, _mm256_mul_pd(va, vb));
    }
    vector_mul_scalar_d(dst + i, a + i, b + i, n - i);
}

LINX_TARGET_AVX inline void vector_neg_avx_d(double* dst, const double* a, std::size_t n) {
    std::size_t i = 0;
    const __m256d zero = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4) {
        const __m256d va = _mm256_loadu_pd(a + i);
        _mm256_storeu_pd(dst + i, _mm256_sub_pd(zero, va));
    }
    vector_neg_scalar_d(dst + i, a + i, n - i);
}

LINX_TARGET_AVX inline void vector_scale_avx_d(double* dst, const double* a, double s, std::size_t n) {
    std::size_t i = 0;
    const __m256d vs = _mm256_set1_pd(s);
    for (; i + 4 <= n; i += 4) {
        const __m256d va = _mm256_loadu_pd(a + i);
        _mm256_storeu_pd(dst + i, _mm256_mul_pd(va, vs));
    }
    vector_scale_scalar_d(dst + i, a + i, s, n - i);
}
#endif

#if defined(__aarch64__) || defined(__ARM_NEON)
inline void vector_add_neon_d(double* dst, const double* a, const double* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(dst + i, vaddq_f64(va, vb));
    }
    vector_add_scalar_d(dst + i, a + i, b + i, n - i);
}

inline void vector_sub_neon_d(double* dst, const double* a, const double* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(dst + i, vsubq_f64(va, vb));
    }
    vector_sub_scalar_d(dst + i, a + i, b + i, n - i);
}

inline void vector_mul_neon_d(double* dst, const double* a, const double* b, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vst1q_f64(dst + i, vmulq_f64(va, vb));
    }
    vector_mul_scalar_d(dst + i, a + i, b + i, n - i);
}

inline void vector_neg_neon_d(double* dst, const double* a, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        vst1q_f64(dst + i, vnegq_f64(va));
    }
    vector_neg_scalar_d(dst + i, a + i, n - i);
}

inline void vector_scale_neon_d(double* dst, const double* a, double s, std::size_t n) {
    std::size_t i = 0;
    float64x2_t vs = vdupq_n_f64(s);
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(a + i);
        vst1q_f64(dst + i, vmulq_f64(va, vs));
    }
    vector_scale_scalar_d(dst + i, a + i, s, n - i);
}
#endif

inline void vector_add_d(double* dst, const double* a, const double* b, std::size_t n) {
#if LINX_X86_SIMD
    const auto kernel = runtime_cpu_features().selected_kernel;
#if LINX_COMPILED_AVX
    if (kernel_uses_avx(kernel)) {
        vector_add_avx_d(dst, a, b, n);
        return;
    }
#endif
#if LINX_COMPILED_SSE2
    if (kernel == SimdKernel::SSE2) {
        vector_add_sse2_d(dst, a, b, n);
        return;
    }
#endif
#elif defined(__aarch64__) || defined(__ARM_NEON)
    vector_add_neon_d(dst, a, b, n);
    return;
#endif
    vector_add_scalar_d(dst, a, b, n);
}

inline void vector_sub_d(double* dst, const double* a, const double* b, std::size_t n) {
#if LINX_X86_SIMD
    const auto kernel = runtime_cpu_features().selected_kernel;
#if LINX_COMPILED_AVX
    if (kernel_uses_avx(kernel)) {
        vector_sub_avx_d(dst, a, b, n);
        return;
    }
#endif
#if LINX_COMPILED_SSE2
    if (kernel == SimdKernel::SSE2) {
        vector_sub_sse2_d(dst, a, b, n);
        return;
    }
#endif
#elif defined(__aarch64__) || defined(__ARM_NEON)
    vector_sub_neon_d(dst, a, b, n);
    return;
#endif
    vector_sub_scalar_d(dst, a, b, n);
}

inline void vector_mul_d(double* dst, const double* a, const double* b, std::size_t n) {
#if LINX_X86_SIMD
    const auto kernel = runtime_cpu_features().selected_kernel;
#if LINX_COMPILED_AVX
    if (kernel_uses_avx(kernel)) {
        vector_mul_avx_d(dst, a, b, n);
        return;
    }
#endif
#if LINX_COMPILED_SSE2
    if (kernel == SimdKernel::SSE2) {
        vector_mul_sse2_d(dst, a, b, n);
        return;
    }
#endif
#elif defined(__aarch64__) || defined(__ARM_NEON)
    vector_mul_neon_d(dst, a, b, n);
    return;
#endif
    vector_mul_scalar_d(dst, a, b, n);
}

inline void vector_neg_d(double* dst, const double* a, std::size_t n) {
#if LINX_X86_SIMD
    const auto kernel = runtime_cpu_features().selected_kernel;
#if LINX_COMPILED_AVX
    if (kernel_uses_avx(kernel)) {
        vector_neg_avx_d(dst, a, n);
        return;
    }
#endif
#if LINX_COMPILED_SSE2
    if (kernel == SimdKernel::SSE2) {
        vector_neg_sse2_d(dst, a, n);
        return;
    }
#endif
#elif defined(__aarch64__) || defined(__ARM_NEON)
    vector_neg_neon_d(dst, a, n);
    return;
#endif
    vector_neg_scalar_d(dst, a, n);
}

inline void vector_scale_d(double* dst, const double* a, double s, std::size_t n) {
#if LINX_X86_SIMD
    const auto kernel = runtime_cpu_features().selected_kernel;
#if LINX_COMPILED_AVX
    if (kernel_uses_avx(kernel)) {
        vector_scale_avx_d(dst, a, s, n);
        return;
    }
#endif
#if LINX_COMPILED_SSE2
    if (kernel == SimdKernel::SSE2) {
        vector_scale_sse2_d(dst, a, s, n);
        return;
    }
#endif
#elif defined(__aarch64__) || defined(__ARM_NEON)
    vector_scale_neon_d(dst, a, s, n);
    return;
#endif
    vector_scale_scalar_d(dst, a, s, n);
}

} // namespace detail

inline std::string hardware_backend() {
#if LINX_HAS_BLAS && defined(__aarch64__) && defined(__APPLE__)
    return "Apple Accelerate BLAS/LAPACK + std::thread task parallel (arm64)";
#elif LINX_HAS_BLAS
    return "Apple Accelerate BLAS/LAPACK + std::thread task parallel (Intel)";
#else
    return detail::runtime_backend_description();
#endif
}

inline std::string cpu_optimization_summary() {
#if LINX_HAS_BLAS && defined(__aarch64__) && defined(__APPLE__)
    return "selected=Apple Accelerate BLAS/LAPACK, arch=arm64, threads=Accelerate";
#elif LINX_HAS_BLAS
    return "selected=Apple Accelerate BLAS/LAPACK, arch=Intel, threads=Accelerate";
#else
    return detail::cpu_optimization_description();
#endif
}

inline std::size_t auto_strassen_min() {
    return detail::runtime_strassen_min();
}

inline std::size_t auto_strassen_base() {
    return detail::runtime_strassen_base();
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

    if constexpr (std::is_same<T, double>::value) {
#if LINX_HAS_BLAS
        // ── Apple Accelerate BLAS ─────────────────────────────────────
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(rows), static_cast<int>(cols), static_cast<int>(inner),
                    1.0, lhs.data().data(), static_cast<int>(inner),
                    rhs.data().data(), static_cast<int>(cols),
                    0.0, out.data().data(), static_cast<int>(cols));
        return out;
#else
        const Matrix<T> rhs_t = rhs.transpose();
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
        const std::size_t work = rows * inner * cols;
        detail::parallel_for_rows(rows, 64 * 64 * 64, work, worker);
        return out;
#endif
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
                                for (std::size_t j = jj; j < j_end; ++j)
                                    out_data[i * cols + j] += a * rhs_data[k * cols + j];
                            }
                        }
                    }
                }
            }
        };
        const std::size_t work = rows * inner * cols;
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
Matrix<T> pad_square_identity(const Matrix<T>& input, std::size_t size) {
    Matrix<T> out = Matrix<T>::eye(size);
    out.set_block(0, 0, input);
    return out;
}

inline bool strassen_parallel_enabled(std::size_t n, std::size_t depth);

template <typename T>
Matrix<T> strassen_square_impl(const Matrix<T>& lhs, const Matrix<T>& rhs,
                               std::size_t threshold, std::size_t depth) {
    const std::size_t n = lhs.rows();
    if (n <= threshold) {
        return matmul_classic(lhs, rhs);
    }
    if (n % 2 != 0) {
        const std::size_t padded_size = n + 1;
        const auto padded_lhs = pad_square(lhs, padded_size);
        const auto padded_rhs = pad_square(rhs, padded_size);
        return strassen_square_impl(padded_lhs, padded_rhs, threshold, depth)
            .block(0, 0, n, n);
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

    Matrix<T> p1;
    Matrix<T> p2;
    Matrix<T> p3;
    Matrix<T> p4;
    Matrix<T> p5;
    Matrix<T> p6;
    Matrix<T> p7;

    if (strassen_parallel_enabled(n, depth)) {
        auto p1_task = std::async(std::launch::async, [&]() {
            return strassen_square_impl(a, f - i, threshold, depth + 1);
        });
        auto p2_task = std::async(std::launch::async, [&]() {
            return strassen_square_impl(a + b, i, threshold, depth + 1);
        });
        auto p3_task = std::async(std::launch::async, [&]() {
            return strassen_square_impl(c + d, e, threshold, depth + 1);
        });
        auto p4_task = std::async(std::launch::async, [&]() {
            return strassen_square_impl(d, g - e, threshold, depth + 1);
        });
        auto p5_task = std::async(std::launch::async, [&]() {
            return strassen_square_impl(a + d, e + i, threshold, depth + 1);
        });
        auto p6_task = std::async(std::launch::async, [&]() {
            return strassen_square_impl(b - d, g + i, threshold, depth + 1);
        });
        auto p7_task = std::async(std::launch::async, [&]() {
            return strassen_square_impl(a - c, e + f, threshold, depth + 1);
        });
        p1 = p1_task.get();
        p2 = p2_task.get();
        p3 = p3_task.get();
        p4 = p4_task.get();
        p5 = p5_task.get();
        p6 = p6_task.get();
        p7 = p7_task.get();
    } else {
        p1 = strassen_square_impl(a, f - i, threshold, depth + 1);
        p2 = strassen_square_impl(a + b, i, threshold, depth + 1);
        p3 = strassen_square_impl(c + d, e, threshold, depth + 1);
        p4 = strassen_square_impl(d, g - e, threshold, depth + 1);
        p5 = strassen_square_impl(a + d, e + i, threshold, depth + 1);
        p6 = strassen_square_impl(b - d, g + i, threshold, depth + 1);
        p7 = strassen_square_impl(a - c, e + f, threshold, depth + 1);
    }

    Matrix<T> out(n, n);
    out.set_block(0, 0, p5 + p4 - p2 + p6);
    out.set_block(0, h, p1 + p2);
    out.set_block(h, 0, p3 + p4);
    out.set_block(h, h, p1 + p5 - p3 - p7);
    return out;
}

template <typename T>
Matrix<T> strassen_square(const Matrix<T>& lhs, const Matrix<T>& rhs, std::size_t threshold) {
    return strassen_square_impl(lhs, rhs, threshold, 0);
}

inline std::size_t strassen_effective_threshold(std::size_t threshold) {
    return std::max(threshold, runtime_strassen_base());
}

struct ConstMatrixViewD {
    const double* data = nullptr;
    std::size_t stride = 0;
};

struct MatrixViewD {
    double* data = nullptr;
    std::size_t stride = 0;
};

inline ConstMatrixViewD subview(ConstMatrixViewD view, std::size_t row, std::size_t col) {
    return {view.data + row * view.stride + col, view.stride};
}

inline MatrixViewD subview(MatrixViewD view, std::size_t row, std::size_t col) {
    return {view.data + row * view.stride + col, view.stride};
}

inline void add_view(double* dst, ConstMatrixViewD lhs, ConstMatrixViewD rhs,
                     std::size_t n, bool subtract_rhs = false) {
#if LINX_HAS_BLAS
    for (std::size_t r = 0; r < n; ++r) {
        const double* lhs_row = lhs.data + r * lhs.stride;
        const double* rhs_row = rhs.data + r * rhs.stride;
        double* dst_row = dst + r * n;
        if (subtract_rhs) {
            vDSP_vsubD(rhs_row, 1, lhs_row, 1, dst_row, 1, static_cast<vDSP_Length>(n));
        } else {
            vDSP_vaddD(lhs_row, 1, rhs_row, 1, dst_row, 1, static_cast<vDSP_Length>(n));
        }
    }
#else
    for (std::size_t r = 0; r < n; ++r) {
        const double* lhs_row = lhs.data + r * lhs.stride;
        const double* rhs_row = rhs.data + r * rhs.stride;
        double* dst_row = dst + r * n;
        if (subtract_rhs) {
            for (std::size_t c = 0; c < n; ++c) dst_row[c] = lhs_row[c] - rhs_row[c];
        } else {
            for (std::size_t c = 0; c < n; ++c) dst_row[c] = lhs_row[c] + rhs_row[c];
        }
    }
#endif
}

inline void gemm_view(ConstMatrixViewD lhs, ConstMatrixViewD rhs, MatrixViewD out, std::size_t n) {
#if LINX_HAS_BLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
                1.0, lhs.data, static_cast<int>(lhs.stride),
                rhs.data, static_cast<int>(rhs.stride),
                0.0, out.data, static_cast<int>(out.stride));
#else
    for (std::size_t r = 0; r < n; ++r) {
        double* out_row = out.data + r * out.stride;
        for (std::size_t c = 0; c < n; ++c) {
            out_row[c] = 0.0;
        }
        for (std::size_t k = 0; k < n; ++k) {
            const double a = lhs.data[r * lhs.stride + k];
            const double* rhs_row = rhs.data + k * rhs.stride;
            for (std::size_t c = 0; c < n; ++c) {
                out_row[c] += a * rhs_row[c];
            }
        }
    }
#endif
}

inline void zero_view(MatrixViewD out, std::size_t n) {
    for (std::size_t r = 0; r < n; ++r) {
        std::memset(out.data + r * out.stride, 0, n * sizeof(double));
    }
}

inline void accumulate_view(MatrixViewD dst, ConstMatrixViewD src, std::size_t n, double alpha) {
#if LINX_HAS_BLAS
    for (std::size_t r = 0; r < n; ++r) {
        double* dst_row = dst.data + r * dst.stride;
        const double* src_row = src.data + r * src.stride;
        if (alpha > 0.0) {
            vDSP_vaddD(dst_row, 1, src_row, 1, dst_row, 1, static_cast<vDSP_Length>(n));
        } else {
            vDSP_vsubD(src_row, 1, dst_row, 1, dst_row, 1, static_cast<vDSP_Length>(n));
        }
    }
#else
    for (std::size_t r = 0; r < n; ++r) {
        double* dst_row = dst.data + r * dst.stride;
        const double* src_row = src.data + r * src.stride;
        if (alpha > 0.0) {
            for (std::size_t c = 0; c < n; ++c) dst_row[c] += src_row[c];
        } else {
            for (std::size_t c = 0; c < n; ++c) dst_row[c] -= src_row[c];
        }
    }
#endif
}

inline void combine_strassen_quadrants(MatrixViewD out,
                                       ConstMatrixViewD p1, ConstMatrixViewD p2,
                                       ConstMatrixViewD p3, ConstMatrixViewD p4,
                                       ConstMatrixViewD p5, ConstMatrixViewD p6,
                                       ConstMatrixViewD p7,
                                       std::size_t h) {
    for (std::size_t r = 0; r < h; ++r) {
        const double* p1_row = p1.data + r * p1.stride;
        const double* p2_row = p2.data + r * p2.stride;
        const double* p3_row = p3.data + r * p3.stride;
        const double* p4_row = p4.data + r * p4.stride;
        const double* p5_row = p5.data + r * p5.stride;
        const double* p6_row = p6.data + r * p6.stride;
        const double* p7_row = p7.data + r * p7.stride;
        double* c11 = out.data + r * out.stride;
        double* c12 = c11 + h;
        double* c21 = out.data + (r + h) * out.stride;
        double* c22 = c21 + h;

        for (std::size_t c = 0; c < h; ++c) {
            c11[c] = p5_row[c] + p4_row[c] - p2_row[c] + p6_row[c];
            c12[c] = p1_row[c] + p2_row[c];
            c21[c] = p3_row[c] + p4_row[c];
            c22[c] = p1_row[c] + p5_row[c] - p3_row[c] - p7_row[c];
        }
    }
}

inline void strassen_view_impl(ConstMatrixViewD lhs, ConstMatrixViewD rhs,
                               MatrixViewD out, std::size_t n,
                               std::size_t threshold, std::size_t depth);

inline std::size_t strassen_work_estimate(std::size_t n) {
    const std::size_t limit = std::numeric_limits<std::size_t>::max();
    if (n != 0 && n > limit / n) {
        return limit;
    }
    const std::size_t square = n * n;
    if (n != 0 && square > limit / n) {
        return limit;
    }
    return square * n;
}

inline std::size_t strassen_parallel_depth_limit() {
    const std::size_t threads = preferred_threads();
    if (threads >= 128) {
        return 3;
    }
    if (threads >= 32) {
        return 2;
    }
    return threads > 1 ? 1 : 0;
}

inline bool strassen_parallel_enabled(std::size_t n, std::size_t depth) {
    const char* disabled = std::getenv("LINX_DISABLE_STRASSEN_PARALLEL");
    if (disabled != nullptr && disabled[0] != '\0' && disabled[0] != '0') {
        return false;
    }
    if (depth >= strassen_parallel_depth_limit()) {
        return false;
    }
    return task_parallel_enabled(strassen_work_estimate(n));
}

inline std::vector<double> strassen_product(ConstMatrixViewD lhs,
                                            ConstMatrixViewD rhs,
                                            std::size_t n,
                                            std::size_t threshold,
                                            std::size_t depth) {
    std::vector<double> product(n * n);
    strassen_view_impl(lhs, rhs, {product.data(), n}, n, threshold, depth + 1);
    return product;
}

inline std::vector<double> strassen_product_with_operands(
    ConstMatrixViewD lhs, ConstMatrixViewD rhs,
    std::size_t n, std::size_t threshold, std::size_t depth,
    ConstMatrixViewD lhs_rhs, bool lhs_has_sum, bool lhs_subtract_rhs,
    ConstMatrixViewD rhs_rhs, bool rhs_has_sum, bool rhs_subtract_rhs) {

    std::vector<double> lhs_storage;
    std::vector<double> rhs_storage;
    ConstMatrixViewD lhs_arg = lhs;
    ConstMatrixViewD rhs_arg = rhs;

    if (lhs_has_sum) {
        lhs_storage.resize(n * n);
        add_view(lhs_storage.data(), lhs, lhs_rhs, n, lhs_subtract_rhs);
        lhs_arg = {lhs_storage.data(), n};
    }
    if (rhs_has_sum) {
        rhs_storage.resize(n * n);
        add_view(rhs_storage.data(), rhs, rhs_rhs, n, rhs_subtract_rhs);
        rhs_arg = {rhs_storage.data(), n};
    }

    return strassen_product(lhs_arg, rhs_arg, n, threshold, depth);
}

inline void copy_view_to_padded(ConstMatrixViewD src, double* dst,
                                std::size_t n, std::size_t padded_size) {
    std::fill(dst, dst + padded_size * padded_size, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        std::memcpy(dst + r * padded_size,
                    src.data + r * src.stride,
                    n * sizeof(double));
    }
}

inline void copy_padded_to_view(ConstMatrixViewD src, MatrixViewD dst, std::size_t n) {
    for (std::size_t r = 0; r < n; ++r) {
        std::memcpy(dst.data + r * dst.stride,
                    src.data + r * src.stride,
                    n * sizeof(double));
    }
}

inline void strassen_view_padded(ConstMatrixViewD lhs, ConstMatrixViewD rhs,
                                 MatrixViewD out, std::size_t n,
                                 std::size_t threshold, std::size_t depth) {
    const std::size_t padded_size = n + 1;
    std::vector<double> lhs_padded(padded_size * padded_size);
    std::vector<double> rhs_padded(padded_size * padded_size);
    std::vector<double> out_padded(padded_size * padded_size);

    copy_view_to_padded(lhs, lhs_padded.data(), n, padded_size);
    copy_view_to_padded(rhs, rhs_padded.data(), n, padded_size);

    strassen_view_impl({lhs_padded.data(), padded_size},
                       {rhs_padded.data(), padded_size},
                       {out_padded.data(), padded_size},
                       padded_size, threshold, depth);
    copy_padded_to_view({out_padded.data(), padded_size}, out, n);
}

inline void strassen_view_sequential(ConstMatrixViewD lhs, ConstMatrixViewD rhs,
                                     MatrixViewD out, std::size_t n,
                                     std::size_t threshold, std::size_t depth) {
    const std::size_t h = n / 2;
    const auto a11 = subview(lhs, 0, 0);
    const auto a12 = subview(lhs, 0, h);
    const auto a21 = subview(lhs, h, 0);
    const auto a22 = subview(lhs, h, h);
    const auto b11 = subview(rhs, 0, 0);
    const auto b12 = subview(rhs, 0, h);
    const auto b21 = subview(rhs, h, 0);
    const auto b22 = subview(rhs, h, h);
    const auto c11 = subview(out, 0, 0);
    const auto c12 = subview(out, 0, h);
    const auto c21 = subview(out, h, 0);
    const auto c22 = subview(out, h, h);

    std::vector<double> s1(h * h);
    std::vector<double> s2(h * h);
    std::vector<double> product(h * h);

    const ConstMatrixViewD s1_view{s1.data(), h};
    const ConstMatrixViewD s2_view{s2.data(), h};
    const MatrixViewD product_view{product.data(), h};
    const ConstMatrixViewD product_const{product.data(), h};

    zero_view(out, n);

    // P1 = A11 * (B12 - B22) -> C12, C22
    add_view(s1.data(), b12, b22, h, true);
    strassen_view_impl(a11, s1_view, product_view, h, threshold, depth + 1);
    accumulate_view(c12, product_const, h, 1.0);
    accumulate_view(c22, product_const, h, 1.0);

    // P2 = (A11 + A12) * B22 -> C11, C12
    add_view(s1.data(), a11, a12, h);
    strassen_view_impl(s1_view, b22, product_view, h, threshold, depth + 1);
    accumulate_view(c11, product_const, h, -1.0);
    accumulate_view(c12, product_const, h, 1.0);

    // P3 = (A21 + A22) * B11 -> C21, C22
    add_view(s1.data(), a21, a22, h);
    strassen_view_impl(s1_view, b11, product_view, h, threshold, depth + 1);
    accumulate_view(c21, product_const, h, 1.0);
    accumulate_view(c22, product_const, h, -1.0);

    // P4 = A22 * (B21 - B11) -> C11, C21
    add_view(s1.data(), b21, b11, h, true);
    strassen_view_impl(a22, s1_view, product_view, h, threshold, depth + 1);
    accumulate_view(c11, product_const, h, 1.0);
    accumulate_view(c21, product_const, h, 1.0);

    // P5 = (A11 + A22) * (B11 + B22) -> C11, C22
    add_view(s1.data(), a11, a22, h);
    add_view(s2.data(), b11, b22, h);
    strassen_view_impl(s1_view, s2_view, product_view, h, threshold, depth + 1);
    accumulate_view(c11, product_const, h, 1.0);
    accumulate_view(c22, product_const, h, 1.0);

    // P6 = (A12 - A22) * (B21 + B22) -> C11
    add_view(s1.data(), a12, a22, h, true);
    add_view(s2.data(), b21, b22, h);
    strassen_view_impl(s1_view, s2_view, product_view, h, threshold, depth + 1);
    accumulate_view(c11, product_const, h, 1.0);

    // P7 = (A11 - A21) * (B11 + B12) -> C22
    add_view(s1.data(), a11, a21, h, true);
    add_view(s2.data(), b11, b12, h);
    strassen_view_impl(s1_view, s2_view, product_view, h, threshold, depth + 1);
    accumulate_view(c22, product_const, h, -1.0);
}

inline void strassen_view_parallel(ConstMatrixViewD lhs, ConstMatrixViewD rhs,
                                   MatrixViewD out, std::size_t n,
                                   std::size_t threshold, std::size_t depth) {
    const std::size_t h = n / 2;
    const auto a11 = subview(lhs, 0, 0);
    const auto a12 = subview(lhs, 0, h);
    const auto a21 = subview(lhs, h, 0);
    const auto a22 = subview(lhs, h, h);
    const auto b11 = subview(rhs, 0, 0);
    const auto b12 = subview(rhs, 0, h);
    const auto b21 = subview(rhs, h, 0);
    const auto b22 = subview(rhs, h, h);

    // Dependency order:
    // phase 1: prepare independent sum/difference operands for P1..P7.
    // phase 2: run the seven recursive products concurrently.
    // phase 3: join products and combine quadrants.
    auto p1_task = std::async(std::launch::async, [=]() {
        return strassen_product_with_operands(a11, b12, h, threshold, depth,
                                              {}, false, false,
                                              b22, true, true);
    });
    auto p2_task = std::async(std::launch::async, [=]() {
        return strassen_product_with_operands(a11, b22, h, threshold, depth,
                                              a12, true, false,
                                              {}, false, false);
    });
    auto p3_task = std::async(std::launch::async, [=]() {
        return strassen_product_with_operands(a21, b11, h, threshold, depth,
                                              a22, true, false,
                                              {}, false, false);
    });
    auto p4_task = std::async(std::launch::async, [=]() {
        return strassen_product_with_operands(a22, b21, h, threshold, depth,
                                              {}, false, false,
                                              b11, true, true);
    });
    auto p5_task = std::async(std::launch::async, [=]() {
        return strassen_product_with_operands(a11, b11, h, threshold, depth,
                                              a22, true, false,
                                              b22, true, false);
    });
    auto p6_task = std::async(std::launch::async, [=]() {
        return strassen_product_with_operands(a12, b21, h, threshold, depth,
                                              a22, true, true,
                                              b22, true, false);
    });
    auto p7_task = std::async(std::launch::async, [=]() {
        return strassen_product_with_operands(a11, b11, h, threshold, depth,
                                              a21, true, true,
                                              b12, true, false);
    });

    const auto p1 = p1_task.get();
    const auto p2 = p2_task.get();
    const auto p3 = p3_task.get();
    const auto p4 = p4_task.get();
    const auto p5 = p5_task.get();
    const auto p6 = p6_task.get();
    const auto p7 = p7_task.get();

    combine_strassen_quadrants(out,
                               {p1.data(), h}, {p2.data(), h},
                               {p3.data(), h}, {p4.data(), h},
                               {p5.data(), h}, {p6.data(), h},
                               {p7.data(), h}, h);
}

inline void strassen_view_impl(ConstMatrixViewD lhs, ConstMatrixViewD rhs,
                               MatrixViewD out, std::size_t n,
                               std::size_t threshold, std::size_t depth) {
    if (n <= threshold) {
        gemm_view(lhs, rhs, out, n);
        return;
    }
    if (n % 2 != 0) {
        strassen_view_padded(lhs, rhs, out, n, threshold, depth);
        return;
    }

    if (strassen_parallel_enabled(n, depth)) {
        strassen_view_parallel(lhs, rhs, out, n, threshold, depth);
        return;
    }

    strassen_view_sequential(lhs, rhs, out, n, threshold, depth);
}

inline void strassen_view(ConstMatrixViewD lhs, ConstMatrixViewD rhs,
                          MatrixViewD out, std::size_t n, std::size_t threshold) {
    strassen_view_impl(lhs, rhs, out, n, threshold, 0);
}

inline Matrix<double> strassen_square_fast(const Matrix<double>& lhs,
                                           const Matrix<double>& rhs,
                                           std::size_t threshold) {
    const std::size_t n = lhs.rows();
    Matrix<double> out(n, n);
    strassen_view({lhs.data().data(), n}, {rhs.data().data(), n},
                  {out.data().data(), n}, n, strassen_effective_threshold(threshold));
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

    if constexpr (std::is_same<T, double>::value) {
        std::size_t size = std::max({lhs.rows(), lhs.cols(), rhs.cols(), std::size_t{1}});
        const std::size_t cutoff = detail::strassen_effective_threshold(threshold);
        if (size < cutoff) {
            return matmul_classic(lhs, rhs);
        }
        if (lhs.rows() == size && lhs.cols() == size &&
            rhs.rows() == size && rhs.cols() == size &&
            size % 2 == 0) {
            return detail::strassen_square_fast(lhs, rhs, cutoff);
        }
        if (size % 2 != 0) {
            ++size;
        }

        const auto padded_lhs = detail::pad_square(lhs, size);
        const auto padded_rhs = detail::pad_square(rhs, size);
        return detail::strassen_square_fast(padded_lhs, padded_rhs, cutoff)
            .block(0, 0, lhs.rows(), rhs.cols());
    } else {
        const std::size_t size = detail::next_power_of_two(
            std::max({lhs.rows(), lhs.cols(), rhs.cols(), std::size_t{1}})
        );

        const auto padded_lhs = detail::pad_square(lhs, size);
        const auto padded_rhs = detail::pad_square(rhs, size);
        return detail::strassen_square(padded_lhs, padded_rhs, threshold)
            .block(0, 0, lhs.rows(), rhs.cols());
    }
}

template <typename T>
Matrix<T> matmul(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    // Square matrices above the runtime-tuned cutoff use Strassen.
    if (lhs.rows() == lhs.cols() && rhs.rows() == rhs.cols()
        && lhs.cols() == rhs.rows()
        && lhs.rows() > detail::runtime_strassen_min()) {
        return matmul_strassen(lhs, rhs, detail::runtime_strassen_base());
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

        auto& a_data = a.data();
        auto& b_data = b.data();
        const T* pivot_a = a_data.data() + col * n;
        const T* pivot_b = b_data.data() + col * m;
        auto eliminate_rows = [&](std::size_t row_begin, std::size_t row_end) {
            for (std::size_t r = row_begin; r < row_end; ++r) {
                if (r == col) {
                    continue;
                }
                T* row_a = a_data.data() + r * n;
                T* row_b = b_data.data() + r * m;
                const T factor = row_a[col];
                if (std::abs(factor) <= eps) {
                    continue;
                }
                for (std::size_t c = 0; c < n; ++c) {
                    row_a[c] -= factor * pivot_a[c];
                }
                for (std::size_t c = 0; c < m; ++c) {
                    row_b[c] -= factor * pivot_b[c];
                }
            }
        };
        detail::parallel_for_rows(n, 128 * 128, n * (n + m), eliminate_rows);
    }

    return b;
}

template <typename T>
Matrix<T> least_squares(const Matrix<T>& a, const Matrix<T>& b, T eps = static_cast<T>(1e-12)) {
    if (a.rows() != b.rows()) {
        throw ShapeError("least_squares requires a.rows == b.rows");
    }
    if (a.rows() < a.cols()) {
        throw ShapeError("least_squares currently supports overdetermined systems with rows >= cols");
    }

    const std::size_t m = a.rows();
    const std::size_t n = a.cols();
    const std::size_t nrhs = b.cols();

#if LINX_HAS_BLAS
    if constexpr (std::is_same<T, double>::value) {
        Matrix<double> a_col = a.transpose();

        int m_int = static_cast<int>(m);
        int n_int = static_cast<int>(n);
        int nrhs_int = static_cast<int>(nrhs);
        int lda = static_cast<int>(m);
        int ldb = static_cast<int>(std::max(m, n));
        int info = 0;
        char trans = 'N';

        std::vector<double> b_col(static_cast<std::size_t>(ldb) * nrhs, 0.0);
        for (std::size_t r = 0; r < m; ++r) {
            for (std::size_t c = 0; c < nrhs; ++c) {
                b_col[r + c * static_cast<std::size_t>(ldb)] = b(r, c);
            }
        }

        int lwork = -1;
        double work_query = 0.0;
        dgels_(&trans, &m_int, &n_int, &nrhs_int,
               a_col.data().data(), &lda,
               b_col.data(), &ldb,
               &work_query, &lwork, &info);
        if (info != 0) {
            throw LinAlgError("dgels workspace query failed");
        }

        lwork = std::max(1, static_cast<int>(work_query));
        std::vector<double> work(static_cast<std::size_t>(lwork));

        dgels_(&trans, &m_int, &n_int, &nrhs_int,
               a_col.data().data(), &lda,
               b_col.data(), &ldb,
               work.data(), &lwork, &info);
        if (info < 0) {
            throw LinAlgError("dgels: illegal argument");
        }
        if (info > 0) {
            throw LinAlgError("dgels: matrix is rank deficient");
        }

        Matrix<double> x(n, nrhs);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < nrhs; ++c) {
                x(r, c) = b_col[r + c * static_cast<std::size_t>(ldb)];
            }
        }
        return x;
    }
#endif

    const auto a_t = a.transpose();
    return solve(matmul(a_t, a), matmul(a_t, b), eps);
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

    if (n > 1 && n % 2 != 0) {
        const auto padded = detail::pad_square_identity(matrix, n + 1);
        return inverse_schur(padded, min_block, eps).block(0, 0, n, n);
    }

    // ── 작은/중간 행렬은 LAPACK / LU로 직접 처리 ─────────────────
    if (n <= static_cast<std::size_t>(LINX_LAPACK_INVERSE_MAX)) {
#if LINX_HAS_BLAS
        if constexpr (std::is_same<T, double>::value) {
            return detail::inverse_lapack(matrix, eps);
        }
#endif
        return inverse_lu(matrix, eps);
    }

    // ── base case: min_block 이하 → LU 분해 ────────────────────
    if (n <= min_block) {
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

    Matrix<T> ca_inv;
    Matrix<T> a_inv_b;
    if (detail::task_parallel_enabled(h)) {
        auto ca_task = std::async(std::launch::async, [&]() {
            return matmul(c, a_inv);
        });
        a_inv_b = matmul(a_inv, b);
        ca_inv = ca_task.get();
    } else {
        ca_inv = matmul(c, a_inv);
        a_inv_b = matmul(a_inv, b);
    }

    const auto schur = d - matmul(ca_inv, b);
    const auto s_inv = inverse_schur(schur, min_block, eps);

    Matrix<T> a_inv_b_s_inv;
    Matrix<T> s_inv_ca_inv;
    if (detail::task_parallel_enabled(h)) {
        auto right_task = std::async(std::launch::async, [&]() {
            return matmul(a_inv_b, s_inv);
        });
        s_inv_ca_inv = matmul(s_inv, ca_inv);
        a_inv_b_s_inv = right_task.get();
    } else {
        a_inv_b_s_inv = matmul(a_inv_b, s_inv);
        s_inv_ca_inv = matmul(s_inv, ca_inv);
    }

    const auto top_left = a_inv + matmul(a_inv_b_s_inv, ca_inv);
    const auto top_right = -a_inv_b_s_inv;
    const auto bottom_left = -s_inv_ca_inv;

    Matrix<T> out(n, n);
    out.set_block(0, 0, top_left);
    out.set_block(0, h, top_right);
    out.set_block(h, 0, bottom_left);
    out.set_block(h, h, s_inv);
    return out;
}

template <typename T>
Matrix<T> inverse_schur_strassen(const Matrix<T>& matrix,
                                 std::size_t min_block = static_cast<std::size_t>(LINX_LAPACK_INVERSE_MAX),
                                 std::size_t strassen_threshold = static_cast<std::size_t>(LINX_M2_STRASSEN_BASE),
                                 T eps = static_cast<T>(1e-12)) {
    if (!matrix.square()) {
        throw ShapeError("inverse_schur_strassen requires a square matrix");
    }

    const std::size_t n = matrix.rows();
    const std::size_t block = std::max(min_block, std::size_t{1});

    if (n > 1 && n % 2 != 0) {
        const auto padded = detail::pad_square_identity(matrix, n + 1);
        return inverse_schur_strassen(padded, block, strassen_threshold, eps)
            .block(0, 0, n, n);
    }

    if (n <= block) {
#if LINX_HAS_BLAS
        if constexpr (std::is_same<T, double>::value) {
            return detail::inverse_lapack(matrix, eps);
        }
#endif
        return inverse_lu(matrix, eps);
    }

    auto multiply = [strassen_threshold](const Matrix<T>& lhs, const Matrix<T>& rhs) {
        return matmul_strassen(lhs, rhs, strassen_threshold);
    };

    const std::size_t h = n / 2;
    const auto a = matrix.block(0, 0, h, h);
    const auto b = matrix.block(0, h, h, h);
    const auto c = matrix.block(h, 0, h, h);
    const auto d = matrix.block(h, h, h, h);

    const auto a_inv = inverse_schur_strassen(a, block, strassen_threshold, eps);

    Matrix<T> ca_inv;
    Matrix<T> a_inv_b;
    if (detail::task_parallel_enabled(h)) {
        auto ca_task = std::async(std::launch::async, [&]() {
            return multiply(c, a_inv);
        });
        a_inv_b = multiply(a_inv, b);
        ca_inv = ca_task.get();
    } else {
        ca_inv = multiply(c, a_inv);
        a_inv_b = multiply(a_inv, b);
    }

    const auto schur = d - multiply(ca_inv, b);
    const auto s_inv = inverse_schur_strassen(schur, block, strassen_threshold, eps);

    Matrix<T> a_inv_b_s_inv;
    Matrix<T> s_inv_ca_inv;
    if (detail::task_parallel_enabled(h)) {
        auto right_task = std::async(std::launch::async, [&]() {
            return multiply(a_inv_b, s_inv);
        });
        s_inv_ca_inv = multiply(s_inv, ca_inv);
        a_inv_b_s_inv = right_task.get();
    } else {
        a_inv_b_s_inv = multiply(a_inv_b, s_inv);
        s_inv_ca_inv = multiply(s_inv, ca_inv);
    }

    const auto top_left = a_inv + multiply(a_inv_b_s_inv, ca_inv);
    const auto top_right = -a_inv_b_s_inv;
    const auto bottom_left = -s_inv_ca_inv;

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
        const auto n = matrix.data().size();
        double sum_sq = 0.0;
        if (n <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            sum_sq = cblas_ddot(static_cast<int>(n), matrix.data().data(), 1, matrix.data().data(), 1);
        } else {
            vDSP_svesqD(matrix.data().data(), 1, &sum_sq, static_cast<vDSP_Length>(n));
        }
        return static_cast<T>(std::sqrt(sum_sq));
    }
#endif
    // Use SIMD dot product to compute sum of squares
    if constexpr (std::is_same<T, double>::value) {
        double sum_sq = detail::dot_contiguous(matrix.data().data(),
                                                matrix.data().data(),
                                                matrix.data().size());
        return static_cast<T>(std::sqrt(sum_sq));
    }
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
