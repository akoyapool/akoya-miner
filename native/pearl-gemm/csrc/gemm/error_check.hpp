#pragma once
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <cstdio>

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    const char* error_str = cudaGetErrorString(code);
    // CUDA has a sticky global error state. We must clear it with cudaGetLastError()
    // before throwing, otherwise subsequent CUDA calls may fail unexpectedly.
    // Note: Input validation (e.g. TORCH_CHECK(dtype==int8)) throws BEFORE any CUDA
    // call happens, so there's no CUDA error state to clear in those cases.
    cudaGetLastError();
    TORCH_CHECK(false, "CUDA error: ", error_str, " at ", file, ":", line);
  }
}

#define CHECK_CUDA_KERNEL_LAUNCH() gpuErrchk(cudaGetLastError())
