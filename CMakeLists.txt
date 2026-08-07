cmake_minimum_required(VERSION 3.18)
# Fall back to the conventional CUDA location when the caller did not pick a compiler
# (override via -DCMAKE_CUDA_COMPILER=... or the CUDACXX env var).
if (NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif ()
project(fast_ulysses LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "80;90;100;120")
endif ()

# ccache: wrap CXX/CUDA compilers to cache objects and skip recompiling unchanged TUs (skipped if not installed)
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "ccache enabled: ${CCACHE_PROGRAM}")
endif ()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}" CACHE STRING "Torch prefix" FORCE)
find_package(Torch REQUIRED)

# setup.py resolves NVSHMEM and passes both paths, because the install we most want to use
# -- the nvidia-nvshmem-cu13 wheel that torch already depends on -- ships headers and
# libnvshmem_host.so.3 but NO lib/cmake/nvshmem, so find_package(NVSHMEM) cannot see it.
# Linking the host library by path works for every layout and needs no package config.
if (NOT DEFINED NVSHMEM_HOME OR NOT EXISTS "${NVSHMEM_HOME}/include/nvshmem.h")
    message(FATAL_ERROR "NVSHMEM_HOME missing: setup.py must pass -DNVSHMEM_HOME=<absolute>.")
endif ()
if (NOT DEFINED NVSHMEM_HOST_LIB OR NOT EXISTS "${NVSHMEM_HOST_LIB}")
    message(FATAL_ERROR "NVSHMEM_HOST_LIB missing: setup.py must pass -DNVSHMEM_HOST_LIB=<absolute>.")
endif ()

if (NOT DEFINED EXT_SUFFIX)
    set(EXT_SUFFIX ".so")
endif ()

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/fast_ulysses/csrc/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/fast_ulysses/csrc/*.cu")
add_library(_C SHARED ${SOURCES})
set_target_properties(_C PROPERTIES PREFIX "" SUFFIX "${EXT_SUFFIX}"
        CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
# CUDA 13 moved the CCCL headers (cuda/std/*) into include/cccl, where nvshmem.h no longer
# finds them -- and nvshmem.h is included by bindings.cpp, so the HOST compiler needs the
# path too, not just nvcc. Adding it here removes what INSTALL.md used to ask users to pass
# by hand.
find_package(CUDAToolkit REQUIRED)
set(CCCL_INCLUDE_DIRS "")
foreach (dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
    if (EXISTS "${dir}/cccl")
        list(APPEND CCCL_INCLUDE_DIRS "${dir}/cccl")
    endif ()
endforeach ()
target_include_directories(_C PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/fast_ulysses/csrc" "${NVSHMEM_HOME}/include"
        ${CCCL_INCLUDE_DIRS})
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
# Only the HOST library. No device-side nvshmem_* call exists in these kernels, so there is
# no relocatable-device-code link step and libnvshmem_device.a stays out -- which also keeps
# nvshmemi_init_thread out, the symbol that clashes with torch's libtorch_nvshmem.so (see the
# DEVIATION notes in ulysses_group.cu).
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Python::Module
        "${NVSHMEM_HOST_LIB}" cuda)  # cuda: CUDA driver, for cuTensorMapEncodeTiled (TMA)
# The wheel layout is not on any default search path, so bake it in rather than making every
# caller set LD_LIBRARY_PATH.
get_filename_component(NVSHMEM_LIB_DIR "${NVSHMEM_HOST_LIB}" DIRECTORY)
set_target_properties(_C PROPERTIES BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "${NVSHMEM_LIB_DIR}")
# -t0: nvcc compiles the per-arch device passes in parallel (one thread per arch)
target_compile_options(_C PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -t0 --expt-relaxed-constexpr
        -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
        -U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__>)
