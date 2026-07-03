from pathlib import Path
import os
import platform
import sys

import numpy as np
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parent

define_macros = []
extra_link_args = []

if sys.platform == "win32":
    define_macros.append(("LINX_WINDOWS_INTEL_AUTODETECT", "1"))
    extra_compile_args = [
        "/std:c++17",
        "/O2",
        "/DNDEBUG",
        "/EHsc",
        "/fp:fast",
        "/DWIN32_LEAN_AND_MEAN",
    ]
    if os.environ.get("LINX_WINDOWS_INTEL_AVX2") == "1":
        extra_compile_args.append("/arch:AVX2")
elif sys.platform == "darwin":
    extra_compile_args = [
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-fno-math-errno",
        "-ffast-math",
        "-funroll-loops",
        "-ftree-vectorize",
    ]
    define_macros.append(("LINX_USE_ACCELERATE", "1"))
    extra_compile_args += ["-flto", "-fvectorize", "-Wno-deprecated-declarations"]
    extra_link_args += ["-flto", "-framework", "Accelerate"]
    if platform.machine() == "arm64":
        extra_compile_args += ["-mcpu=apple-m1"]
    else:
        extra_compile_args += ["-march=native"]
else:
    extra_compile_args = [
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-fno-math-errno",
        "-ffast-math",
        "-funroll-loops",
        "-ftree-vectorize",
    ]
    if platform.machine() in {"arm64", "aarch64"}:
        extra_compile_args += ["-mcpu=native"]
    else:
        extra_compile_args += ["-march=native"]


extension = Extension(
    "linx._linx",
    sources=["python/linx/_linx_module.cpp"],
    include_dirs=[str(ROOT / "include"), np.get_include()],
    language="c++",
    define_macros=define_macros,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)


setup(
    name="linx-la",
    version="0.1.0",
    description="NumPy-like C++ linear algebra library with Schur inverse and SIMD matmul",
    packages=["linx"],
    package_dir={"": "python"},
    ext_modules=[extension],
    python_requires=">=3.9",
    install_requires=["numpy"],
)
