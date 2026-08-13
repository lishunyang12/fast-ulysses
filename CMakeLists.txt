cmake_minimum_required(VERSION 3.18)

if (NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif ()

project(fast_ulysses LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;90;100;120")
endif ()
set(FAST_ULYSSES_ARCH_LIST "${CMAKE_CUDA_ARCHITECTURES}")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}" CACHE STRING "Torch prefix" FORCE)
find_package(Torch REQUIRED)

if (NOT DEFINED EXT_SUFFIX)
    set(EXT_SUFFIX ".so")
endif ()
if (NOT DEFINED FAST_ULYSSES_VERSION)
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" FAST_ULYSSES_VERSION LIMIT_COUNT 1)
endif ()

add_library(_C SHARED
    src/bindings.cc
    src/group.cc
    src/transfer.cu
    src/barrier.cu)
set_target_properties(_C PROPERTIES
    PREFIX ""
    SUFFIX "${EXT_SUFFIX}"
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "${FAST_ULYSSES_ARCH_LIST}")

string(REPLACE ";" "," FAST_ULYSSES_ARCH_CSV "${FAST_ULYSSES_ARCH_LIST}")
target_compile_definitions(_C PRIVATE
    FAST_ULYSSES_VERSION="${FAST_ULYSSES_VERSION}"
    FAST_ULYSSES_CUDA_ARCH_LIST="${FAST_ULYSSES_ARCH_CSV}")
target_include_directories(_C PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
find_library(TORCH_PYTHON_LIBRARY torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(_C PRIVATE
    ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Python::Module)
set_target_properties(_C PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN/../torch/lib")
target_compile_options(_C PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>)
