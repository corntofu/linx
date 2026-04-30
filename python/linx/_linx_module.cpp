#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <exception>
#include <string>

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

la::Matrix<double> matrix_from_python(PyObject* object, const char* name) {
    PyArrayHandle array(object);
    if (array.ptr == nullptr) {
        throw std::invalid_argument(std::string(name) + " must be convertible to a NumPy float64 array");
    }

    if (PyArray_NDIM(array.ptr) != 2) {
        throw la::ShapeError(std::string(name) + " must be a 2D array");
    }

    const auto rows = static_cast<std::size_t>(PyArray_DIM(array.ptr, 0));
    const auto cols = static_cast<std::size_t>(PyArray_DIM(array.ptr, 1));
    const auto* src = static_cast<const double*>(PyArray_DATA(array.ptr));

    la::Matrix<double> matrix(rows, cols);
    auto& dst = matrix.data();
    std::copy(src, src + rows * cols, dst.begin());
    return matrix;
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

    try {
        auto lhs = matrix_from_python(lhs_object, "lhs");
        auto rhs = matrix_from_python(rhs_object, "rhs");
        return run_matrix_kernel([&]() {
            return la::matmul(lhs, rhs);
        });
    } catch (const std::exception& error) {
        return exception_to_python(error);
    }
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
        auto matrix = matrix_from_python(matrix_object, "matrix");
        const std::size_t block = min_block < 1 ? 1 : static_cast<std::size_t>(min_block);

        return run_matrix_kernel([&]() {
            if (regularization > 0.0) {
                return la::inverse_regularized(matrix, regularization, eps);
            }
            const std::string selected(method);
            if (selected == "lu") {
                return la::inverse_lu(matrix, eps);
            }
            if (selected == "schur") {
                return la::inverse_schur(matrix, block, eps);
            }
            throw std::invalid_argument("method must be 'schur' or 'lu'");
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
        auto matrix = matrix_from_python(matrix_object, "matrix");
        const std::size_t block = min_block < 1 ? 1 : static_cast<std::size_t>(min_block);
        return run_matrix_kernel([&]() {
            return la::inverse_schur(matrix, block, eps);
        });
    } catch (const std::exception& error) {
        return exception_to_python(error);
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

PyObject* py_hardware_backend(PyObject*, PyObject*) {
    return PyUnicode_FromString(la::hardware_backend().c_str());
}

PyMethodDef methods[] = {
    {"matmul", py_matmul, METH_VARARGS, "Multiply two 2D arrays with the C++ linx backend."},
    {"matmul_strassen", reinterpret_cast<PyCFunction>(py_matmul_strassen), METH_VARARGS | METH_KEYWORDS, "Multiply two 2D arrays with Strassen recursion."},
    {"solve", py_solve, METH_VARARGS, "Solve A @ X = B."},
    {"inverse", reinterpret_cast<PyCFunction>(py_inverse), METH_VARARGS | METH_KEYWORDS, "Invert a matrix."},
    {"inverse_schur", reinterpret_cast<PyCFunction>(py_inverse_schur), METH_VARARGS | METH_KEYWORDS, "Invert a matrix with the Schur complement algorithm."},
    {"condition_number", reinterpret_cast<PyCFunction>(py_condition_number), METH_VARARGS | METH_KEYWORDS, "Estimate the Frobenius condition number."},
    {"hardware_backend", py_hardware_backend, METH_NOARGS, "Return the compiled SIMD/thread backend."},
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
