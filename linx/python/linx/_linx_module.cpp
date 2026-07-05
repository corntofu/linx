#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <exception>
#include <limits>
#include <string>
#include <vector>

#include "linx/linx.hpp"

namespace {

struct PyArrayHandle {
    PyArrayObject* ptr = nullptr;

    explicit PyArrayHandle(PyObject* object) {
        ptr = reinterpret_cast<PyArrayObject*>(
            PyArray_FROM_OTF(object, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST)
        );
    }

    ~PyArrayHandle() {
        Py_XDECREF(ptr);
    }

    PyArrayHandle(const PyArrayHandle&) = delete;
    PyArrayHandle& operator=(const PyArrayHandle&) = delete;
};

la::Matrix<double> matrix_from_array(PyArrayObject* array, const char* name) {
    if (array == nullptr) {
        throw std::invalid_argument(std::string(name) + " must be convertible to a NumPy float64 array");
    }

    if (PyArray_NDIM(array) != 2) {
        throw la::ShapeError(std::string(name) + " must be a 2D array");
    }

    const auto rows = static_cast<std::size_t>(PyArray_DIM(array, 0));
    const auto cols = static_cast<std::size_t>(PyArray_DIM(array, 1));
    const auto* src = static_cast<const double*>(PyArray_DATA(array));

    la::Matrix<double> matrix(rows, cols);
    auto& dst = matrix.data();
    std::copy(src, src + rows * cols, dst.begin());
    return matrix;
}

la::Matrix<double> matrix_from_python(PyObject* object, const char* name) {
    PyArrayHandle array(object);
    if (array.ptr == nullptr) {
        throw std::invalid_argument(std::string(name) + " must be convertible to a NumPy float64 array");
    }

    return matrix_from_array(array.ptr, name);
}

PyObject* matrix_to_python(const la::Matrix<double>& matrix) {
    npy_intp dims[2] = {
        static_cast<npy_intp>(matrix.rows()),
        static_cast<npy_intp>(matrix.cols())
    };

    PyObject* out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (out == nullptr) {
        return nullptr;
    }

    auto* dst = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out)));
    const auto& src = matrix.data();
    std::copy(src.begin(), src.end(), dst);
    return out;
}

bool check_2d(PyArrayObject* array, const char* name) {
    if (PyArray_NDIM(array) != 2) {
        PyErr_Format(PyExc_ValueError, "%s must be a 2D array", name);
        return false;
    }
    return true;
}

bool check_same_shape(PyArrayObject* lhs, PyArrayObject* rhs) {
    if (PyArray_DIM(lhs, 0) != PyArray_DIM(rhs, 0) ||
        PyArray_DIM(lhs, 1) != PyArray_DIM(rhs, 1)) {
        PyErr_SetString(PyExc_ValueError, "operands must have the same shape");
        return false;
    }
    return true;
}

enum class BinaryOp {
    Add,
    Subtract,
    Multiply,
};

PyObject* binary_elementwise(PyObject* lhs_object, PyObject* rhs_object, BinaryOp op, const char* name) {
    PyArrayHandle lhs_array(lhs_object);
    PyArrayHandle rhs_array(rhs_object);
    if (lhs_array.ptr == nullptr || rhs_array.ptr == nullptr) {
        return nullptr;
    }
    if (!check_2d(lhs_array.ptr, "lhs") ||
        !check_2d(rhs_array.ptr, "rhs") ||
        !check_same_shape(lhs_array.ptr, rhs_array.ptr)) {
        return nullptr;
    }

    npy_intp dims[2] = {
        PyArray_DIM(lhs_array.ptr, 0),
        PyArray_DIM(lhs_array.ptr, 1),
    };
    PyObject* out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (out == nullptr) {
        return nullptr;
    }

    const auto* lhs = static_cast<const double*>(PyArray_DATA(lhs_array.ptr));
    const auto* rhs = static_cast<const double*>(PyArray_DATA(rhs_array.ptr));
    auto* dst = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out)));
    const auto count = static_cast<std::size_t>(PyArray_SIZE(lhs_array.ptr));

    Py_BEGIN_ALLOW_THREADS
#if LINX_HAS_BLAS
    const vDSP_Length n = static_cast<vDSP_Length>(count);
    if (op == BinaryOp::Add) {
        vDSP_vaddD(lhs, 1, rhs, 1, dst, 1, n);
    } else if (op == BinaryOp::Subtract) {
        vDSP_vsubD(rhs, 1, lhs, 1, dst, 1, n);
    } else {
        vDSP_vmulD(lhs, 1, rhs, 1, dst, 1, n);
    }
#else
    if (op == BinaryOp::Add) {
        la::detail::vector_add_d(dst, lhs, rhs, count);
    } else if (op == BinaryOp::Subtract) {
        la::detail::vector_sub_d(dst, lhs, rhs, count);
    } else {
        la::detail::vector_mul_d(dst, lhs, rhs, count);
    }
#endif
    Py_END_ALLOW_THREADS

    (void)name;
    return out;
}

PyObject* scalar_elementwise(PyObject* matrix_object, double scalar) {
    PyArrayHandle matrix_array(matrix_object);
    if (matrix_array.ptr == nullptr) {
        return nullptr;
    }
    if (!check_2d(matrix_array.ptr, "matrix")) {
        return nullptr;
    }

    npy_intp dims[2] = {
        PyArray_DIM(matrix_array.ptr, 0),
        PyArray_DIM(matrix_array.ptr, 1),
    };
    PyObject* out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (out == nullptr) {
        return nullptr;
    }

    const auto* src = static_cast<const double*>(PyArray_DATA(matrix_array.ptr));
    auto* dst = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out)));
    const auto count = static_cast<std::size_t>(PyArray_SIZE(matrix_array.ptr));

    Py_BEGIN_ALLOW_THREADS
#if LINX_HAS_BLAS
    vDSP_vsmulD(src, 1, &scalar, dst, 1, static_cast<vDSP_Length>(count));
#else
    la::detail::vector_scale_d(dst, src, scalar, count);
#endif
    Py_END_ALLOW_THREADS

    return out;
}

PyObject* neg_elementwise(PyObject* matrix_object) {
    PyArrayHandle matrix_array(matrix_object);
    if (matrix_array.ptr == nullptr) {
        return nullptr;
    }
    if (!check_2d(matrix_array.ptr, "matrix")) {
        return nullptr;
    }

    npy_intp dims[2] = {
        PyArray_DIM(matrix_array.ptr, 0),
        PyArray_DIM(matrix_array.ptr, 1),
    };
    PyObject* out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (out == nullptr) {
        return nullptr;
    }

    const auto* src = static_cast<const double*>(PyArray_DATA(matrix_array.ptr));
    auto* dst = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out)));
    const auto count = static_cast<std::size_t>(PyArray_SIZE(matrix_array.ptr));

    Py_BEGIN_ALLOW_THREADS
#if LINX_HAS_BLAS
    vDSP_vnegD(src, 1, dst, 1, static_cast<vDSP_Length>(count));
#else
    la::detail::vector_neg_d(dst, src, count);
#endif
    Py_END_ALLOW_THREADS

    return out;
}

PyObject* transpose_array(PyObject* matrix_object) {
    PyArrayHandle matrix_array(matrix_object);
    if (matrix_array.ptr == nullptr) {
        return nullptr;
    }
    if (!check_2d(matrix_array.ptr, "matrix")) {
        return nullptr;
    }

    const npy_intp rows = PyArray_DIM(matrix_array.ptr, 0);
    const npy_intp cols = PyArray_DIM(matrix_array.ptr, 1);
    npy_intp dims[2] = {cols, rows};
    PyObject* out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (out == nullptr) {
        return nullptr;
    }

    const auto* src = static_cast<const double*>(PyArray_DATA(matrix_array.ptr));
    auto* dst = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out)));

    Py_BEGIN_ALLOW_THREADS
#if LINX_HAS_BLAS
    vDSP_mtransD(src, 1, dst, 1,
                 static_cast<vDSP_Length>(cols),
                 static_cast<vDSP_Length>(rows));
#else
    for (npy_intp r = 0; r < rows; ++r) {
        for (npy_intp c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
#endif
    Py_END_ALLOW_THREADS

    return out;
}

#if LINX_HAS_BLAS
PyObject* inverse_lapack_array(PyArrayObject* array, double regularization = 0.0) {
    if (!check_2d(array, "matrix")) {
        return nullptr;
    }
    const npy_intp n = PyArray_DIM(array, 0);
    if (n != PyArray_DIM(array, 1)) {
        PyErr_SetString(PyExc_ValueError, "inverse requires a square matrix");
        return nullptr;
    }
    if (n > static_cast<npy_intp>(std::numeric_limits<int>::max())) {
        PyErr_SetString(PyExc_ValueError, "inverse dimension exceeds LAPACK int limits");
        return nullptr;
    }

    npy_intp dims[2] = {n, n};
    PyObject* out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (out == nullptr) {
        return nullptr;
    }

    const auto* src = static_cast<const double*>(PyArray_DATA(array));
    auto* dst = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out)));
    std::vector<double> column_major(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    std::vector<int> ipiv(static_cast<std::size_t>(n));
    int n_int = static_cast<int>(n);
    int info = 0;
    int lwork = -1;
    double work_query = 0.0;

    Py_BEGIN_ALLOW_THREADS
    vDSP_mtransD(src, 1, column_major.data(), 1,
                 static_cast<vDSP_Length>(n),
                 static_cast<vDSP_Length>(n));
    if (regularization != 0.0) {
        for (npy_intp i = 0; i < n; ++i) {
            column_major[static_cast<std::size_t>(i) * static_cast<std::size_t>(n + 1)] += regularization;
        }
    }
    dgetrf_(&n_int, &n_int, column_major.data(), &n_int, ipiv.data(), &info);
    if (info == 0) {
        dgetri_(&n_int, column_major.data(), &n_int, ipiv.data(), &work_query, &lwork, &info);
    }
    Py_END_ALLOW_THREADS

    if (info != 0) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_ArithmeticError, "LAPACK inverse factorization failed");
        return nullptr;
    }

    lwork = std::max(1, static_cast<int>(work_query));
    std::vector<double> work(static_cast<std::size_t>(lwork));

    Py_BEGIN_ALLOW_THREADS
    dgetri_(&n_int, column_major.data(), &n_int, ipiv.data(), work.data(), &lwork, &info);
    if (info == 0) {
        vDSP_mtransD(column_major.data(), 1, dst, 1,
                     static_cast<vDSP_Length>(n),
                     static_cast<vDSP_Length>(n));
    }
    Py_END_ALLOW_THREADS

    if (info != 0) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_ArithmeticError, "LAPACK inverse failed");
        return nullptr;
    }
    return out;
}
#endif

PyObject* exception_to_python(const std::exception& error) {
    if (dynamic_cast<const la::ShapeError*>(&error) != nullptr) {
        PyErr_SetString(PyExc_ValueError, error.what());
    } else if (dynamic_cast<const la::LinAlgError*>(&error) != nullptr) {
        PyErr_SetString(PyExc_ArithmeticError, error.what());
    } else {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    }
    return nullptr;
}

template <typename Fn>
PyObject* run_matrix_kernel(Fn&& fn) {
    la::Matrix<double> result;
    std::string error_message;
    int error_kind = 0;
    bool ok = true;

    Py_BEGIN_ALLOW_THREADS
    try {
        result = fn();
    } catch (const la::ShapeError& error) {
        error_message = error.what();
        error_kind = 1;
        ok = false;
    } catch (const la::LinAlgError& error) {
        error_message = error.what();
        error_kind = 2;
        ok = false;
    } catch (const std::invalid_argument& error) {
        error_message = error.what();
        error_kind = 1;
        ok = false;
    } catch (const std::exception& error) {
        error_message = error.what();
        error_kind = 3;
        ok = false;
    } catch (...) {
        error_message = "unknown C++ exception";
        error_kind = 3;
        ok = false;
    }
    Py_END_ALLOW_THREADS

    if (!ok) {
        if (error_kind == 1) {
            PyErr_SetString(PyExc_ValueError, error_message.c_str());
        } else if (error_kind == 2) {
            PyErr_SetString(PyExc_ArithmeticError, error_message.c_str());
        } else {
            PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
        }
        return nullptr;
    }
    return matrix_to_python(result);
}

PyObject* py_matmul(PyObject*, PyObject* args) {
    PyObject* lhs_object = nullptr;
    PyObject* rhs_object = nullptr;
    if (!PyArg_ParseTuple(args, "OO:matmul", &lhs_object, &rhs_object)) {
        return nullptr;
    }

#if LINX_HAS_BLAS
    PyArrayHandle lhs_array(lhs_object);
    PyArrayHandle rhs_array(rhs_object);
    if (lhs_array.ptr == nullptr || rhs_array.ptr == nullptr) {
        return nullptr;
    }
    if (PyArray_NDIM(lhs_array.ptr) != 2 || PyArray_NDIM(rhs_array.ptr) != 2) {
        PyErr_SetString(PyExc_ValueError, "matmul arguments must be 2D arrays");
        return nullptr;
    }

    const npy_intp rows = PyArray_DIM(lhs_array.ptr, 0);
    const npy_intp inner = PyArray_DIM(lhs_array.ptr, 1);
    const npy_intp rhs_rows = PyArray_DIM(rhs_array.ptr, 0);
    const npy_intp cols = PyArray_DIM(rhs_array.ptr, 1);
    if (inner != rhs_rows) {
        PyErr_SetString(PyExc_ValueError, "matmul requires lhs.cols == rhs.rows");
        return nullptr;
    }
    if (rows > static_cast<npy_intp>(std::numeric_limits<int>::max()) ||
        inner > static_cast<npy_intp>(std::numeric_limits<int>::max()) ||
        cols > static_cast<npy_intp>(std::numeric_limits<int>::max())) {
        PyErr_SetString(PyExc_ValueError, "matmul dimensions exceed BLAS int limits");
        return nullptr;
    }

    if (rows == inner && inner == cols &&
        rows > static_cast<npy_intp>(LINX_M2_STRASSEN_MIN)) {
        try {
            auto lhs = matrix_from_array(lhs_array.ptr, "lhs");
            auto rhs = matrix_from_array(rhs_array.ptr, "rhs");
            return run_matrix_kernel([&]() {
                return la::matmul_strassen(lhs, rhs, LINX_M2_STRASSEN_BASE);
            });
        } catch (const std::exception& error) {
            return exception_to_python(error);
        }
    }

    npy_intp dims[2] = {rows, cols};
    PyObject* out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (out == nullptr) {
        return nullptr;
    }

    const auto* lhs = static_cast<const double*>(PyArray_DATA(lhs_array.ptr));
    const auto* rhs = static_cast<const double*>(PyArray_DATA(rhs_array.ptr));
    auto* dst = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out)));

    Py_BEGIN_ALLOW_THREADS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(rows), static_cast<int>(cols), static_cast<int>(inner),
                1.0, lhs, static_cast<int>(inner),
                rhs, static_cast<int>(cols),
                0.0, dst, static_cast<int>(cols));
    Py_END_ALLOW_THREADS
    return out;
#else
    try {
        auto lhs = matrix_from_python(lhs_object, "lhs");
        auto rhs = matrix_from_python(rhs_object, "rhs");
        return run_matrix_kernel([&]() {
            return la::matmul(lhs, rhs);
        });
    } catch (const std::exception& error) {
        return exception_to_python(error);
    }
#endif
}

PyObject* py_matmul_strassen(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* lhs_object = nullptr;
    PyObject* rhs_object = nullptr;
    int threshold = 64;

    static const char* kwlist[] = {"lhs", "rhs", "threshold", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OO|i:matmul_strassen",
            const_cast<char**>(kwlist),
            &lhs_object,
            &rhs_object,
            &threshold)) {
        return nullptr;
    }

#if LINX_HAS_BLAS
    PyArrayHandle lhs_array(lhs_object);
    PyArrayHandle rhs_array(rhs_object);
    if (lhs_array.ptr == nullptr || rhs_array.ptr == nullptr) {
        return nullptr;
    }
    if (PyArray_NDIM(lhs_array.ptr) != 2 || PyArray_NDIM(rhs_array.ptr) != 2) {
        PyErr_SetString(PyExc_ValueError, "matmul_strassen arguments must be 2D arrays");
        return nullptr;
    }

    const npy_intp rows = PyArray_DIM(lhs_array.ptr, 0);
    const npy_intp inner = PyArray_DIM(lhs_array.ptr, 1);
    const npy_intp rhs_rows = PyArray_DIM(rhs_array.ptr, 0);
    const npy_intp cols = PyArray_DIM(rhs_array.ptr, 1);
    if (inner != rhs_rows) {
        PyErr_SetString(PyExc_ValueError, "matmul requires lhs.cols == rhs.rows");
        return nullptr;
    }
    if (rows > static_cast<npy_intp>(std::numeric_limits<int>::max()) ||
        inner > static_cast<npy_intp>(std::numeric_limits<int>::max()) ||
        cols > static_cast<npy_intp>(std::numeric_limits<int>::max())) {
        PyErr_SetString(PyExc_ValueError, "matmul dimensions exceed BLAS int limits");
        return nullptr;
    }

    const std::size_t cutoff = std::max(
        threshold < 1 ? std::size_t{1} : static_cast<std::size_t>(threshold),
        static_cast<std::size_t>(LINX_M2_STRASSEN_BASE)
    );
    const auto max_dim = static_cast<std::size_t>(std::max({rows, inner, cols, npy_intp{1}}));
    if (max_dim < cutoff) {
        npy_intp dims[2] = {rows, cols};
        PyObject* out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (out == nullptr) {
            return nullptr;
        }

        const auto* lhs = static_cast<const double*>(PyArray_DATA(lhs_array.ptr));
        const auto* rhs = static_cast<const double*>(PyArray_DATA(rhs_array.ptr));
        auto* dst = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out)));

        Py_BEGIN_ALLOW_THREADS
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(rows), static_cast<int>(cols), static_cast<int>(inner),
                    1.0, lhs, static_cast<int>(inner),
                    rhs, static_cast<int>(cols),
                    0.0, dst, static_cast<int>(cols));
        Py_END_ALLOW_THREADS
        return out;
    }

    try {
        auto lhs = matrix_from_array(lhs_array.ptr, "lhs");
        auto rhs = matrix_from_array(rhs_array.ptr, "rhs");
        return run_matrix_kernel([&]() {
            return la::matmul_strassen(lhs, rhs, cutoff);
        });
    } catch (const std::exception& error) {
        return exception_to_python(error);
    }
#else
    try {
        auto lhs = matrix_from_python(lhs_object, "lhs");
        auto rhs = matrix_from_python(rhs_object, "rhs");
        const std::size_t cutoff = threshold < 1 ? 1 : static_cast<std::size_t>(threshold);
        return run_matrix_kernel([&]() {
            return la::matmul_strassen(lhs, rhs, cutoff);
        });
    } catch (const std::exception& error) {
        return exception_to_python(error);
    }
#endif
}

PyObject* py_solve(PyObject*, PyObject* args) {
    PyObject* a_object = nullptr;
    PyObject* b_object = nullptr;
    if (!PyArg_ParseTuple(args, "OO:solve", &a_object, &b_object)) {
        return nullptr;
    }

    try {
        auto a = matrix_from_python(a_object, "a");
        auto b = matrix_from_python(b_object, "b");
        return run_matrix_kernel([&]() {
            return la::solve(a, b);
        });
    } catch (const std::exception& error) {
        return exception_to_python(error);
    }
}

PyObject* py_least_squares(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* a_object = nullptr;
    PyObject* b_object = nullptr;
    double eps = 1e-12;

    static const char* kwlist[] = {"a", "b", "eps", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OO|d:least_squares",
            const_cast<char**>(kwlist),
            &a_object,
            &b_object,
            &eps)) {
        return nullptr;
    }

    try {
        auto a = matrix_from_python(a_object, "a");
        auto b = matrix_from_python(b_object, "b");
        return run_matrix_kernel([&]() {
            return la::least_squares(a, b, eps);
        });
    } catch (const std::exception& error) {
        return exception_to_python(error);
    }
}

PyObject* py_inverse(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* matrix_object = nullptr;
    const char* method = "schur";
    int min_block = 32;
    double regularization = 0.0;
    double eps = 1e-12;

    static const char* kwlist[] = {"matrix", "method", "min_block", "regularization", "eps", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "O|sidd:inverse",
            const_cast<char**>(kwlist),
            &matrix_object,
            &method,
            &min_block,
            &regularization,
            &eps)) {
        return nullptr;
    }

    try {
        const std::string selected(method);
        if (selected != "lu" && selected != "schur" && selected != "schur_strassen") {
            throw std::invalid_argument("method must be 'schur', 'schur_strassen', or 'lu'");
        }

#if LINX_HAS_BLAS
        PyArrayHandle matrix_array(matrix_object);
        if (matrix_array.ptr == nullptr) {
            return nullptr;
        }
        if (!check_2d(matrix_array.ptr, "matrix")) {
            return nullptr;
        }
        const npy_intp n = PyArray_DIM(matrix_array.ptr, 0);
        if (n != PyArray_DIM(matrix_array.ptr, 1)) {
            PyErr_SetString(PyExc_ValueError, "inverse requires a square matrix");
            return nullptr;
        }
        const bool odd_schur_requires_padding =
            selected != "lu" && n > 1 && (n % 2) != 0;
        if (selected == "lu" ||
            (!odd_schur_requires_padding &&
             n <= static_cast<npy_intp>(LINX_LAPACK_INVERSE_MAX))) {
            return inverse_lapack_array(matrix_array.ptr, regularization);
        }
#endif

        auto matrix = matrix_from_python(matrix_object, "matrix");
        const std::size_t block = min_block < 1 ? 1 : static_cast<std::size_t>(min_block);

        return run_matrix_kernel([&]() {
            if (regularization > 0.0) {
                return la::inverse_regularized(matrix, regularization, eps);
            }
            if (selected == "lu") {
                return la::inverse_lu(matrix, eps);
            }
            if (selected == "schur_strassen") {
                return la::inverse_schur_strassen(matrix, block, LINX_M2_STRASSEN_BASE, eps);
            }
            if (selected == "schur") {
                return la::inverse_schur(matrix, block, eps);
            }
            return la::inverse_schur(matrix, block, eps);
        });
    } catch (const std::exception& error) {
        return exception_to_python(error);
    }
}

PyObject* py_inverse_schur(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* matrix_object = nullptr;
    int min_block = 32;
    double eps = 1e-12;

    static const char* kwlist[] = {"matrix", "min_block", "eps", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "O|id:inverse_schur",
            const_cast<char**>(kwlist),
            &matrix_object,
            &min_block,
            &eps)) {
        return nullptr;
    }

    try {
#if LINX_HAS_BLAS
        PyArrayHandle matrix_array(matrix_object);
        if (matrix_array.ptr == nullptr) {
            return nullptr;
        }
        if (!check_2d(matrix_array.ptr, "matrix")) {
            return nullptr;
        }
        const npy_intp n = PyArray_DIM(matrix_array.ptr, 0);
        if (n != PyArray_DIM(matrix_array.ptr, 1)) {
            PyErr_SetString(PyExc_ValueError, "inverse_schur requires a square matrix");
            return nullptr;
        }
        const bool odd_requires_padding = n > 1 && (n % 2) != 0;
        if (!odd_requires_padding &&
            n <= static_cast<npy_intp>(LINX_LAPACK_INVERSE_MAX)) {
            return inverse_lapack_array(matrix_array.ptr);
        }
#endif
        auto matrix = matrix_from_python(matrix_object, "matrix");
        const std::size_t block = min_block < 1 ? 1 : static_cast<std::size_t>(min_block);
        return run_matrix_kernel([&]() {
            return la::inverse_schur(matrix, block, eps);
        });
    } catch (const std::exception& error) {
        return exception_to_python(error);
    }
}

PyObject* py_inverse_schur_strassen(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* matrix_object = nullptr;
    int min_block = LINX_LAPACK_INVERSE_MAX;
    int strassen_threshold = static_cast<int>(la::auto_strassen_base());
    double eps = 1e-12;

    static const char* kwlist[] = {"matrix", "min_block", "strassen_threshold", "eps", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "O|iid:inverse_schur_strassen",
            const_cast<char**>(kwlist),
            &matrix_object,
            &min_block,
            &strassen_threshold,
            &eps)) {
        return nullptr;
    }

    try {
#if LINX_HAS_BLAS
        PyArrayHandle matrix_array(matrix_object);
        if (matrix_array.ptr == nullptr) {
            return nullptr;
        }
        if (!check_2d(matrix_array.ptr, "matrix")) {
            return nullptr;
        }
        const npy_intp n = PyArray_DIM(matrix_array.ptr, 0);
        if (n != PyArray_DIM(matrix_array.ptr, 1)) {
            PyErr_SetString(PyExc_ValueError, "inverse_schur_strassen requires a square matrix");
            return nullptr;
        }
        const std::size_t block = min_block < 1 ? std::size_t{1} : static_cast<std::size_t>(min_block);
        const bool odd_requires_padding = n > 1 && (n % 2) != 0;
        if (!odd_requires_padding && static_cast<std::size_t>(n) <= block) {
            return inverse_lapack_array(matrix_array.ptr);
        }
        auto matrix = matrix_from_array(matrix_array.ptr, "matrix");
#else
        auto matrix = matrix_from_python(matrix_object, "matrix");
        const std::size_t block = min_block < 1 ? std::size_t{1} : static_cast<std::size_t>(min_block);
#endif
        const std::size_t threshold = strassen_threshold < 1
            ? std::size_t{1}
            : static_cast<std::size_t>(strassen_threshold);
        return run_matrix_kernel([&]() {
            return la::inverse_schur_strassen(matrix, block, threshold, eps);
        });
    } catch (const std::exception& error) {
        return exception_to_python(error);
    }
}

PyObject* py_frobenius_norm(PyObject*, PyObject* args) {
    PyObject* obj = nullptr;
    if (!PyArg_ParseTuple(args, "O:frobenius_norm", &obj))
        return nullptr;

    PyArrayHandle matrix_array(obj);
    if (matrix_array.ptr == nullptr) {
        return nullptr;
    }
    if (!check_2d(matrix_array.ptr, "matrix")) {
        return nullptr;
    }

    const auto* src = static_cast<const double*>(PyArray_DATA(matrix_array.ptr));
    const auto count = static_cast<std::size_t>(PyArray_SIZE(matrix_array.ptr));
    double sum_sq = 0.0;

    Py_BEGIN_ALLOW_THREADS
#if LINX_HAS_BLAS
    if (count <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        sum_sq = cblas_ddot(static_cast<int>(count), src, 1, src, 1);
    } else {
        vDSP_svesqD(src, 1, &sum_sq, static_cast<vDSP_Length>(count));
    }
#else
    sum_sq = la::detail::dot_contiguous(src, src, count);
#endif
    Py_END_ALLOW_THREADS

    return PyFloat_FromDouble(std::sqrt(sum_sq));
}

PyObject* py_residual_norm(PyObject*, PyObject* args) {
    PyObject *mat = nullptr, *inv = nullptr;
    if (!PyArg_ParseTuple(args, "OO:residual_norm", &mat, &inv))
        return nullptr;
    try {
        auto m = matrix_from_python(mat, "matrix");
        auto inv_m = matrix_from_python(inv, "inverse_matrix");
        double val = 0.0;
        Py_BEGIN_ALLOW_THREADS
        val = la::residual_norm(m, inv_m);
        Py_END_ALLOW_THREADS
        return PyFloat_FromDouble(val);
    } catch (const std::exception& e) {
        return exception_to_python(e);
    }
}

PyObject* py_condition_number(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* matrix_object = nullptr;
    double eps = 1e-12;

    static const char* kwlist[] = {"matrix", "eps", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "O|d:condition_number",
            const_cast<char**>(kwlist),
            &matrix_object,
            &eps)) {
        return nullptr;
    }

    try {
        auto matrix = matrix_from_python(matrix_object, "matrix");
        double result = 0.0;
        std::string error_message;
        int error_kind = 0;
        bool ok = true;

        Py_BEGIN_ALLOW_THREADS
        try {
            result = la::condition_number_estimate(matrix, eps);
        } catch (const la::ShapeError& error) {
            error_message = error.what();
            error_kind = 1;
            ok = false;
        } catch (const la::LinAlgError& error) {
            error_message = error.what();
            error_kind = 2;
            ok = false;
        } catch (const std::exception& error) {
            error_message = error.what();
            error_kind = 3;
            ok = false;
        }
        Py_END_ALLOW_THREADS

        if (!ok) {
            if (error_kind == 1) {
                PyErr_SetString(PyExc_ValueError, error_message.c_str());
            } else if (error_kind == 2) {
                PyErr_SetString(PyExc_ArithmeticError, error_message.c_str());
            } else {
                PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
            }
            return nullptr;
        }
        return PyFloat_FromDouble(result);
    } catch (const std::exception& error) {
        return exception_to_python(error);
    }
}

PyObject* py_add(PyObject*, PyObject* args) {
    PyObject *lhs = nullptr, *rhs = nullptr;
    if (!PyArg_ParseTuple(args, "OO:add", &lhs, &rhs))
        return nullptr;
    return binary_elementwise(lhs, rhs, BinaryOp::Add, "add");
}

PyObject* py_subtract(PyObject*, PyObject* args) {
    PyObject *lhs = nullptr, *rhs = nullptr;
    if (!PyArg_ParseTuple(args, "OO:subtract", &lhs, &rhs))
        return nullptr;
    return binary_elementwise(lhs, rhs, BinaryOp::Subtract, "subtract");
}

PyObject* py_hadamard(PyObject*, PyObject* args) {
    PyObject *lhs = nullptr, *rhs = nullptr;
    if (!PyArg_ParseTuple(args, "OO:hadamard", &lhs, &rhs))
        return nullptr;
    return binary_elementwise(lhs, rhs, BinaryOp::Multiply, "hadamard");
}

PyObject* py_scalar_mul(PyObject*, PyObject* args) {
    PyObject* obj = nullptr;
    double scalar = 1.0;
    if (!PyArg_ParseTuple(args, "Od:scalar_mul", &obj, &scalar))
        return nullptr;
    return scalar_elementwise(obj, scalar);
}

PyObject* py_transpose(PyObject*, PyObject* args) {
    PyObject* obj = nullptr;
    if (!PyArg_ParseTuple(args, "O:transpose", &obj))
        return nullptr;
    return transpose_array(obj);
}

PyObject* py_neg(PyObject*, PyObject* args) {
    PyObject* obj = nullptr;
    if (!PyArg_ParseTuple(args, "O:neg", &obj))
        return nullptr;
    return neg_elementwise(obj);
}

PyObject* py_hardware_backend(PyObject*, PyObject*) {
    return PyUnicode_FromString(la::hardware_backend().c_str());
}

PyObject* py_cpu_optimization_summary(PyObject*, PyObject*) {
    return PyUnicode_FromString(la::cpu_optimization_summary().c_str());
}

PyObject* py_trace(PyObject*, PyObject* args) {
    PyObject* obj = nullptr;
    if (!PyArg_ParseTuple(args, "O:trace", &obj))
        return nullptr;

    PyArrayHandle matrix_array(obj);
    if (matrix_array.ptr == nullptr) {
        return nullptr;
    }
    if (!check_2d(matrix_array.ptr, "matrix")) {
        return nullptr;
    }
    const npy_intp rows = PyArray_DIM(matrix_array.ptr, 0);
    const npy_intp cols = PyArray_DIM(matrix_array.ptr, 1);
    if (rows != cols) {
        PyErr_SetString(PyExc_ValueError, "trace requires a square matrix");
        return nullptr;
    }

    const auto* src = static_cast<const double*>(PyArray_DATA(matrix_array.ptr));
    double val = 0.0;
    Py_BEGIN_ALLOW_THREADS
#if LINX_HAS_BLAS
    vDSP_sveD(src, static_cast<vDSP_Stride>(cols + 1), &val, static_cast<vDSP_Length>(rows));
#else
    for (npy_intp i = 0; i < rows; ++i) val += src[i * cols + i];
#endif
    Py_END_ALLOW_THREADS

    return PyFloat_FromDouble(val);
}

PyObject* py_det(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* matrix_object = nullptr;
    double eps = 1e-12;

    static const char* kwlist[] = {"matrix", "eps", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "O|d:det",
            const_cast<char**>(kwlist),
            &matrix_object,
            &eps)) {
        return nullptr;
    }

    try {
        auto matrix = matrix_from_python(matrix_object, "matrix");
        double val = 0.0;
        Py_BEGIN_ALLOW_THREADS
        val = la::det(matrix, eps);
        Py_END_ALLOW_THREADS
        return PyFloat_FromDouble(val);
    } catch (const std::exception& error) {
        return exception_to_python(error);
    }
}

PyMethodDef methods[] = {
    {"matmul", py_matmul, METH_VARARGS, "Multiply two matrices with linx backend."},
    {"matmul_strassen", reinterpret_cast<PyCFunction>(py_matmul_strassen), METH_VARARGS | METH_KEYWORDS, "Multiply with Strassen algorithm."},
    {"add", py_add, METH_VARARGS, "Element-wise addition (vDSP)."},
    {"subtract", py_subtract, METH_VARARGS, "Element-wise subtraction (vDSP)."},
    {"hadamard", py_hadamard, METH_VARARGS, "Element-wise multiplication (vDSP)."},
    {"scalar_mul", py_scalar_mul, METH_VARARGS, "Scalar multiplication (vDSP)."},
    {"transpose", py_transpose, METH_VARARGS, "Matrix transpose (vDSP)."},
    {"neg", py_neg, METH_VARARGS, "Unary negation (vDSP)."},
    {"solve", py_solve, METH_VARARGS, "Solve A @ X = B."},
    {"least_squares", reinterpret_cast<PyCFunction>(py_least_squares), METH_VARARGS | METH_KEYWORDS, "Solve min ||A @ X - B||_2."},
    {"inverse", reinterpret_cast<PyCFunction>(py_inverse), METH_VARARGS | METH_KEYWORDS, "Invert a matrix."},
    {"inverse_schur", reinterpret_cast<PyCFunction>(py_inverse_schur), METH_VARARGS | METH_KEYWORDS, "Invert with Schur complement."},
    {"inverse_schur_strassen", reinterpret_cast<PyCFunction>(py_inverse_schur_strassen), METH_VARARGS | METH_KEYWORDS, "Invert with Schur complement and Strassen matmul."},
    {"condition_number", reinterpret_cast<PyCFunction>(py_condition_number), METH_VARARGS | METH_KEYWORDS, "Frobenius condition number."},
    {"frobenius_norm", py_frobenius_norm, METH_VARARGS, "Frobenius norm (vDSP)."},
    {"residual_norm", py_residual_norm, METH_VARARGS, "Residual norm ||A @ A_inv - I||_F."},
    {"hardware_backend", py_hardware_backend, METH_NOARGS, "Return the selected backend."},
    {"cpu_optimization_summary", py_cpu_optimization_summary, METH_NOARGS, "Return runtime CPU optimization details."},
    {"trace", py_trace, METH_VARARGS, "Trace of a square matrix."},
    {"det", reinterpret_cast<PyCFunction>(py_det), METH_VARARGS | METH_KEYWORDS, "Determinant of a square matrix (LU via LAPACK)."},
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_linx",
    "NumPy bindings for the linx C++ linear algebra library.",
    -1,
    methods
};

} // namespace

PyMODINIT_FUNC PyInit__linx() {
    import_array();
    return PyModule_Create(&module);
}
