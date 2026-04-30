# linx

`linx`는 졸업과제 PDF의 요구사항을 바탕으로 만든 작은 C++17 선형대수 라이브러리입니다. NumPy처럼 읽기 쉬운 행렬 생성/연산 API를 제공하고, Schur Complement 기반 역행렬과 Strassen 행렬곱을 포함합니다.

## 주요 기능

- `la::Matrix<T>`: 동적 크기 행렬, `zeros`, `ones`, `eye`, `arange`
- NumPy식 기본 연산: `+`, `-`, scalar multiply/divide, transpose, block slicing, `allclose`
- 하드웨어 최적화 행렬곱: ARM NEON 또는 AVX SIMD dot kernel, RHS packing, `std::thread` 병렬 row scheduling
- 고속 행렬곱: Strassen matmul
- 선형계 풀이: partial pivoting 기반 Gauss-Jordan `solve`
- 역행렬: LU/Gauss-Jordan fallback, Schur Complement 재귀 역행렬
- 안정성 보조: 조건수 추정, diagonal regularization 역행렬
- 검증 도구: Frobenius norm, residual norm
- Python/NumPy 바인딩: `linx.matmul`, `linx.matmul_strassen`, `linx.inverse`, `linx.solve`

## 빌드 및 실행

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/linx_demo
```

`cmake`가 없는 환경에서는 아래처럼 바로 확인할 수 있습니다.

```bash
clang++ -std=c++17 -Iinclude tests/test_linx.cpp -o /tmp/linx_tests
/tmp/linx_tests
clang++ -std=c++17 -Iinclude examples/demo.cpp -o /tmp/linx_demo
/tmp/linx_demo
```

## Python에서 사용

```bash
python3 setup.py build_ext --inplace
PYTHONPATH=python python3 tests/test_python_linx.py
```

```python
import numpy as np
import linx

a = np.array([[4.0, 7.0], [2.0, 6.0]])
inv = linx.inverse(a)
print(a @ inv)
print(linx.hardware_backend())
```

## 사용 예시

```cpp
#include "linx/linx.hpp"

int main() {
    la::Matrix<double> a{
        {4.0, 7.0},
        {2.0, 6.0}
    };

    auto inv = la::inverse(a);
    auto almost_i = la::matmul(a, inv);
}
```

## 설계 메모

Schur Complement 역행렬은 아래 블록 행렬 공식을 사용합니다.

```text
M = [A B]
    [C D]

S = D - C A^-1 B

M^-1 = [A^-1 + A^-1 B S^-1 C A^-1   -A^-1 B S^-1]
       [-S^-1 C A^-1                  S^-1       ]
```

현재 구현은 과제 프로토타입에 맞춘 C++/Python 버전입니다. 다음 단계로는 Eigen/Accelerate BLAS 백엔드 비교, FastAPI 서버 래핑, OpenGL/WebGL 데모 연동을 붙이면 PDF의 전체 시스템 구조와 자연스럽게 이어집니다.
