#pragma once

#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <torch/version.h>

static_assert(TORCH_VERSION_MAJOR > 2 ||
                  (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 10),
              "fast-ulysses requires torch 2.10 or newer");

#define ULYSSES_CUDA_CHECK(expr)                                                \
    do {                                                                         \
        const cudaError_t err_ = (expr);                                         \
        TORCH_CHECK(err_ == cudaSuccess, "CUDA error (" #expr "): ",           \
                    cudaGetErrorString(err_));                                   \
    } while (0)
