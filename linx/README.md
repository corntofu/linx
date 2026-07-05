# linx — C++17 선형대수 라이브러리 (NumPy 스타일)

`linx`는 졸업과제 PDF의 요구사항을 바탕으로 만든 작은 C++17 선형대수 라이브러리입니다.  
NumPy처럼 읽기 쉬운 행렬 생성/연산 API를 제공하고, Apple Accelerate BLAS/LAPACK 가속, Windows Intel CPU 자동 감지 SIMD 경로, Schur Complement 역행렬, Strassen 행렬곱, Python 바인딩을 포함합니다.

---

## 📁 프로젝트 구조 요약

| 경로 | 설명 |
|---|---|
| `include/linx/linx.hpp` | **핵심 라이브러리** — 모든 C++ 템플릿/함수가 단일 헤더에 정의됨 |
| `src/` | (현재 비어 있음 — header-only 라이브러리이므로 소스 분리가 필요 없음) |
| `CMakeLists.txt` | CMake 빌드 설정 — `linx_demo`, `linx_tests` 타겟, macOS Accelerate 연동 |
| `setup.py` | Python 패키지 빌드 스크립트 — `linx._linx` C++ 확장 모듈을 컴파일 |
| `examples/demo.cpp` | C++ 데모 프로그램 — 4×4 역행렬, 잔차 노름, 조건수 출력 |
| `tests/test_linx.cpp` | C++ 단위 테스트 — 기본 연산, 행렬곱, Strassen, 역행렬, solve, 정칙화 역행렬 |
| `tests/test_python_linx.py` | Python 단위 테스트 — C++ 백엔드와 `np.linalg` 정확도 비교, Matrix 클래스 연산 검증 |
| `python/linx/__init__.py` | Python 패키지 진입점 — `Matrix` 클래스, 편의 함수, C++ 확장에서 함수 import |
| `python/linx/_linx_module.cpp` | NumPy/C API 바인딩 — 모든 C++ 함수를 Python에 노출 |
| `python/linx_la.egg-info/` | setuptools 생성 메타데이터 (패키지명 `linx-la`, 의존성 `numpy`) |
| `dist/` | 빌드된 wheel/egg 배포 아티팩트 |
| `.gitignore` | `build/`, `*.so`, `*.pyc` 등 무시 |
| `.vscode/settings.json` | VS Code 편집기 설정 |
| `README.md` | 지금 보고 있는 파일 (이 문서) |

---

## 🔧 핵심 라이브러리: `include/linx/linx.hpp`

**네임스페이스:** `la`  
**요구사항:** C++17, macOS Accelerate framework (옵션), AVX2 또는 NEON SIMD

### `la::Matrix<T>` 클래스

- **생성자:** 크기 지정 (`rows, cols, fill`), 이중 `initializer_list`, 복사/이동
- **팩토리 메서드:** `zeros`, `ones`, `eye`, `arange`
- **요소 접근:** `(row, col)` 연산자
- **슬라이싱:** `block(row0, col0, rows, cols)` → 부분행렬 추출, `set_block(...)` → 부분행렬 덮어쓰기
- **기본 연산:** `+`, `-` (요소별), `-` (단항), `*` / `/` (스칼라), `hadamard` (요소별 곱)
- **기타:** `transpose`, `reshape`, `sum`, `max_abs`, `allclose`

### 자유 함수 (free functions)

| 함수 | 설명 |
|---|---|
| `matmul(lhs, rhs)` | 행렬 곱셈 — 큰 정방행렬은 Strassen, 그 외는 classic |
| `matmul_classic(lhs, rhs)` | BLAS `cblas_dgemm` 또는 AVX2/NEON SIMD + 멀티스레드 |
| `matmul_strassen(lhs, rhs, threshold)` | Strassen 알고리즘 (재귀, 홀수 크기는 padding 후 블록 분할) |
| `solve(a, b)` | `A @ X = B` — LAPACK `dgesv_` 또는 Gauss-Jordan |
| `least_squares(a, b)` | `min ||A·X-B||₂` — LAPACK `dgels_` 또는 normal equations fallback |
| `inverse(matrix)` | 역행렬 — 기본 Schur complement 방식 |
| `inverse_lu(matrix)` | LU 분해 역행렬 |
| `inverse_schur(matrix, min_block)` | Schur complement 재귀 역행렬 (홀수 크기는 `diag(A, I)` padding 후 crop) |
| `inverse_regularized(matrix, lambda)` | `A + λI` 후 역행렬 |
| `det(matrix)` | 행렬식 — LAPACK `dgetrf_` 또는 Gauss-Jordan |
| `trace(matrix)` | 대각합 |
| `frobenius_norm(matrix)` | Frobenius 노름 — Accelerate `vDSP_svesqD` 또는 직접 계산 |
| `condition_number_estimate(matrix)` | 조건수 추정 `‖A‖_F · ‖A⁻¹‖_F` |
| `residual_norm(matrix, inverse)` | `‖A·A⁻¹ - I‖_F` |
| `hardware_backend()` | 런타임에 선택된 백엔드 문자열 반환 |
| `cpu_optimization_summary()` | 런타임 CPU 감지 결과와 선택된 SIMD/threshold 요약 반환 |

### 하드웨어 가속 경로

- **Apple Accelerate (macOS):** `vDSP`(요소별 연산), `cblas_dgemm`(행렬곱), `dgetrf_`/`dgetri_`/`dgesv_`(LU, 역행렬, solve)
- **Windows Intel x86/x64:** CPUID/XGETBV로 `GenuineIntel`, `SSE2`, `AVX`, `AVX2`, `FMA`, OS AVX 상태를 자동 감지한 뒤 요소별 연산, dot/matmul fallback, Strassen threshold를 선택
- **x86_64:** AVX2/FMA, AVX, SSE2 내적/요소별 커널 (`<immintrin.h>`)
- **ARM64:** NEON 내적 커널 (`<arm_neon.h>`)
- **프로세서 병렬화:** `std::thread` 기반 row-chunk 병렬 처리 (`parallel_for_rows`)와 Schur complement의 독립 블록 곱 task 병렬화
- **GPU:** 현재 코어는 CPU/SIMD/BLAS 중심이며, CUDA/Metal/OpenCL 백엔드는 아직 별도 구현 대상

런타임에 선택된 최적화는 아래 함수로 확인할 수 있습니다.

```cpp
std::cout << la::hardware_backend() << "\n";
std::cout << la::cpu_optimization_summary() << "\n";
```

```python
import linx
print(linx.hardware_backend())
print(linx.cpu_optimization_summary())
```

---

## ⚙️ 프로세서 병렬화 적용 지점

Schur complement 역행렬은 모든 단계가 병렬화되는 구조는 아니지만, 아래 블록 연산은 서로 독립이라 CPU task 병렬화가 가능합니다.

| 단계 | 병렬화 상태 |
|---|---|
| `C·A⁻¹` 와 `A⁻¹·B` | 서로 독립 → `std::async`로 병렬 실행 |
| `A⁻¹·B·S⁻¹` 와 `S⁻¹·C·A⁻¹` | 서로 독립 → `std::async`로 병렬 실행 |
| Gauss-Jordan fallback row elimination | pivot row 기준으로 각 row 업데이트 독립 → `parallel_for_rows` 적용 |
| classic matmul fallback | row chunk 단위 `std::thread` 병렬 |
| Strassen 7개 부분 곱셈 | P1..P7 입력 합/차를 먼저 준비한 뒤 `std::async`로 독립 실행, join 후 사분면 합성 |
| Apple Accelerate BLAS/LAPACK | 내부적으로 CPU 벡터화/멀티코어 사용 |

프로세서 task 병렬화를 끄고 비교하려면:

```bash
LINX_DISABLE_PROCESSOR_PARALLEL=1 ./build/linx_demo
```

Strassen 부분 곱셈 병렬화만 끄려면:

```bash
LINX_DISABLE_STRASSEN_PARALLEL=1 ./build/linx_demo
```

### Strassen 부분 곱셈 선행 관계

`A`와 `B`를 각각 2×2 블록으로 나누면 P1..P7은 입력 합/차가 준비된 뒤 서로 독립입니다. 따라서 linx는 아래 순서로 실행합니다.
홀수 크기 블록은 LU/LAPACK fallback으로 끊지 않고 0-padding 후 Strassen을 적용하고, 결과를 원래 크기로 crop합니다.

| 단계 | 작업 |
|---|---|
| 1 | P1..P7에 필요한 입력 합/차 준비 |
| 2 | P1..P7 재귀 곱셈을 병렬 실행 |
| 3 | 모든 future join 후 `C11`, `C12`, `C21`, `C22` 합성 |

| 부분 곱셈 | 선행 입력 | 결과 반영 |
|---|---|---|
| `P1 = A11·(B12-B22)` | `B12-B22` | `C12`, `C22` |
| `P2 = (A11+A12)·B22` | `A11+A12` | `C11`, `C12` |
| `P3 = (A21+A22)·B11` | `A21+A22` | `C21`, `C22` |
| `P4 = A22·(B21-B11)` | `B21-B11` | `C11`, `C21` |
| `P5 = (A11+A22)·(B11+B22)` | `A11+A22`, `B11+B22` | `C11`, `C22` |
| `P6 = (A12-A22)·(B21+B22)` | `A12-A22`, `B21+B22` | `C11` |
| `P7 = (A11-A21)·(B11+B12)` | `A11-A21`, `B11+B12` | `C22` |

## 🔗 Python 바인딩: `python/linx/_linx_module.cpp`

Python C API와 NumPy C API로 구현. 모든 함수는 GIL을 해제(`Py_BEGIN_ALLOW_THREADS`)하고 C++ 연산을 수행하여 멀티스레드 안전성을 확보.

### 노출된 C 함수

`matmul`, `matmul_strassen`, `solve`, `least_squares`, `inverse`, `inverse_schur`, `frobenius_norm`, `residual_norm`, `condition_number`, `add`, `subtract`, `hadamard`, `scalar_mul`, `transpose`, `neg`, `hardware_backend`, `cpu_optimization_summary`, `trace`, `det`

---

## 🐍 Python 패키지: `python/linx/__init__.py`

### `linx.Matrix` 클래스

NumPy float64 2D 배열을 감싸고, 모든 산술 연산을 C++ 백엔드에 위임:

| 연산자/메서드 | C++ 호출 |
|---|---|
| `+`, `-` | `vDSP_vaddD` / `vDSP_vsubD` |
| `*` (스칼라) | `vDSP_vsmulD` |
| `*` (행렬) | Hadamard (`vDSP_vmulD`) |
| `@` | `cblas_dgemm` |
| `-` (단항) | `vDSP_vnegD` |
| `.T` | `vDSP_mtransD` |
| `.inv()` | LAPACK (≤512) 또는 Schur complement |
| `.least_squares(b)` | LAPACK `dgels_` 또는 normal equations fallback |

### 편의 함수

`array`, `zeros`, `ones`, `eye`, `arange` — NumPy ndarray를 직접 반환

---

## 🧪 테스트

### C++ (`tests/test_linx.cpp`)

- 기본 연산 (덧셈, 뺄셈, 전치, Hadamard)
- 행렬곱 정확도
- Strassen vs classic 일치
- LU 역행렬
- Schur complement 역행렬 (4×4)
- `solve` (2×2 선형계)
- 정칙화 역행렬

### Python (`tests/test_python_linx.py`)

- `matmul`, `matmul_strassen` 정확도 (NumPy `@` 연산자와 비교)
- `inverse_schur`, `solve` 정확도 (`np.linalg.solve`와 비교)
- `condition_number`, `hardware_backend` 반환값 검증
- `Matrix` 클래스: `+`, `-`, `*`, `@`, `.T`, `-`, `.inv()`, `.frobenius_norm()`
- 팩토리: `zeros`, `ones`, `eye`

---

## 🚀 빌드 및 실행

### C++ (CMake)

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/linx_demo
```

Windows + MSVC에서 Intel CPU 자동 감지 경로를 빌드하려면:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\build\Release\linx_demo.exe
```

기본 MSVC 빌드는 SSE2까지 안전하게 사용합니다. AVX2 명령어 생성을 명시적으로 켜고 Intel AVX2/FMA CPU에서만 실행할 바이너리를 만들려면:

```powershell
cmake -S . -B build-avx2 -G "Visual Studio 17 2022" -DLINX_WINDOWS_INTEL_AVX2=ON
cmake --build build-avx2 --config Release
```

### C++ (직접 컴파일)

```bash
clang++ -std=c++17 -Iinclude tests/test_linx.cpp -o /tmp/linx_tests
/tmp/linx_tests
clang++ -std=c++17 -Iinclude examples/demo.cpp -o /tmp/linx_demo
/tmp/linx_demo
```

macOS에서 직접 컴파일할 때는 Accelerate를 링크합니다.

```bash
clang++ -std=c++17 -Iinclude tests/test_linx.cpp -framework Accelerate -o /tmp/linx_tests
clang++ -std=c++17 -Iinclude examples/demo.cpp -framework Accelerate -o /tmp/linx_demo
```

### Python

```bash
python3 setup.py build_ext --inplace
PYTHONPATH=python python3 tests/test_python_linx.py
```

Windows Python/MSVC 빌드:

```powershell
py setup.py build_ext --inplace
$env:PYTHONPATH="python"
py tests/test_python_linx.py
```

MSVC에서 AVX2 바이너리를 명시적으로 만들 경우:

```powershell
$env:LINX_WINDOWS_INTEL_AVX2="1"
py setup.py build_ext --inplace
```

최소제곱법만 NumPy와 비교:

```bash
PYTHONPATH=python python3 tests/benchmark_least_squares_linx_vs_numpy.py --quick
PYTHONPATH=python python3 tests/benchmark_linx_vs_numpy.py --quick --only-least-squares
tests/run_least_squares_n4096.sh
```

```python
import numpy as np
import linx

a = np.array([[4.0, 7.0], [2.0, 6.0]])
inv = linx.inverse(a)
print(a @ inv)
print(linx.hardware_backend())
```

---

## 📐 Schur Complement 역행렬 공식

```
M = [A B]    S = D - C·A⁻¹·B
    [C D]

M⁻¹ = [A⁻¹ + A⁻¹·B·S⁻¹·C·A⁻¹   -A⁻¹·B·S⁻¹  ]
      [   -S⁻¹·C·A⁻¹                S⁻¹     ]
```

---

## 📝 설계 메모

- **Header-only:** `include/linx/linx.hpp` 하나만 include 하면 모든 기능 사용 가능
- **템플릿:** `Matrix<T>`는 `double`에 최적화되어 있으며, `float`도 지원
- **BLAS 감지:** `__APPLE__` 매크로로 Accelerate를 자동 활성화하고, 미감지 시 순수 C++ SIMD 경로로 fallback
- **과제 연계:** 현재 C++/Python 구현은 프로토타입이며, 추후 Eigen/Accelerate BLAS 비교, FastAPI 서버 래핑, OpenGL/WebGL 데모와의 연계를 염두에 둠
