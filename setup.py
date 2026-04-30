from pathlib import Path

import numpy as np
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parent


extra_compile_args = [
    "-std=c++17",
    "-O3",
    "-DNDEBUG",
    "-fno-math-errno",
    "-march=native",
]

extra_link_args = []


extension = Extension(
    "linx._linx",
    sources=["python/linx/_linx_module.cpp"],
    include_dirs=[str(ROOT / "include"), np.get_include()],
    language="c++",
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
