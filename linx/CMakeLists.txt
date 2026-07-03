cmake_minimum_required(VERSION 3.16)

project(linx
    VERSION 0.1.0
    DESCRIPTION "A small NumPy-like C++ linear algebra library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LINX_NATIVE_OPTIMIZATION "Build with native CPU optimization flags" ON)
option(LINX_WINDOWS_INTEL_AVX2 "Build MSVC Windows Intel kernels with /arch:AVX2" OFF)

add_library(linx INTERFACE)
target_include_directories(linx INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

find_package(Threads REQUIRED)
target_link_libraries(linx INTERFACE Threads::Threads)

if(WIN32)
    target_compile_definitions(linx INTERFACE LINX_WINDOWS_INTEL_AUTODETECT=1 WIN32_LEAN_AND_MEAN)
endif()

# ── macOS: Accelerate framework (BLAS/LAPACK + vDSP) ──────────────────────
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(linx INTERFACE ${ACCELERATE_FRAMEWORK})
endif()

if(MSVC)
    target_compile_options(linx INTERFACE /O2 /EHsc /fp:fast)
    if(LINX_WINDOWS_INTEL_AVX2)
        target_compile_options(linx INTERFACE /arch:AVX2)
    endif()
elseif(LINX_NATIVE_OPTIMIZATION AND (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU") AND NOT WIN32)
    target_compile_options(linx INTERFACE -O3 -march=native -Wno-deprecated-declarations)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    target_compile_options(linx INTERFACE -O3 -Wno-deprecated-declarations)
endif()

add_executable(linx_demo examples/demo.cpp)
target_link_libraries(linx_demo PRIVATE linx)

add_executable(linx_tests tests/test_linx.cpp)
target_link_libraries(linx_tests PRIVATE linx)

enable_testing()
add_test(NAME linx_tests COMMAND linx_tests)
