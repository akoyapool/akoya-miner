// Legacy DP4A int8 helpers for sm70/sm75.
//
// These functions keep the portable_int8_helpers API intact for Volta/Turing
// builds without compiling the sm80-only cp.async/mma.sync helper kernels.

#include "../capi/portable_int8_helpers.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace pearl::capi::portable {

namespace {

__device__ __forceinline__ int dp4a_s32(int32_t a, int32_t b, int32_t c) {
  int32_t d;
  asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
               : "=r"(d)
               : "r"(a), "r"(b), "r"(c));
  return d;
}

__global__ void dp4a_matmul_i32_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    int M, int N, int K,
    int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int m = static_cast<int>(idx / N);
  const int n = static_cast<int>(idx - static_cast<int64_t>(m) * N);
  const int32_t* a4 = reinterpret_cast<const int32_t*>(
      A + static_cast<int64_t>(m) * K);
  const int32_t* b4 = reinterpret_cast<const int32_t*>(
      B + static_cast<int64_t>(n) * K);

  int32_t acc = 0;
  const int k4 = K >> 2;
  for (int q = 0; q < k4; ++q) {
    acc = dp4a_s32(a4[q], b4[q], acc);
  }
  C[idx] = acc;
}

__global__ void dp4a_matmul_add_clamp_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    const int8_t* __restrict__ base,
    int8_t* __restrict__ out,
    int M, int N, int K,
    int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int m = static_cast<int>(idx / N);
  const int n = static_cast<int>(idx - static_cast<int64_t>(m) * N);
  const int32_t* a4 = reinterpret_cast<const int32_t*>(
      A + static_cast<int64_t>(m) * K);
  const int32_t* b4 = reinterpret_cast<const int32_t*>(
      B + static_cast<int64_t>(n) * K);

  int32_t acc = 0;
  const int k4 = K >> 2;
  for (int q = 0; q < k4; ++q) {
    acc = dp4a_s32(a4[q], b4[q], acc);
  }

  int32_t v = static_cast<int32_t>(base[idx]) + acc;
  if (v > 127) v = 127;
  if (v < -128) v = -128;
  out[idx] = static_cast<int8_t>(v);
}

int launch_grid(int64_t total, dim3& grid, dim3& block) {
  block = dim3(256);
  int64_t blocks = (total + block.x - 1) / block.x;
  if (blocks > static_cast<int64_t>(0x7fffffff)) return -110;
  grid = dim3(static_cast<unsigned>(blocks));
  return 0;
}

}  // namespace

int int8_matmul_i32_native(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) return -201;
  if ((K & 3) != 0) return -200;
  const int64_t total = static_cast<int64_t>(M) * N;
  dim3 grid, block;
  int rc = launch_grid(total, grid, block);
  if (rc != 0) return rc;
  dp4a_matmul_i32_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, total);
  return cudaGetLastError() == cudaSuccess ? 0 : -202;
}

int int8_matmul_add_clamp_native(
    const int8_t* A, const int8_t* B,
    const int8_t* base, int8_t* out,
    int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) return -201;
  if ((K & 3) != 0) return -200;
  const int64_t total = static_cast<int64_t>(M) * N;
  dim3 grid, block;
  int rc = launch_grid(total, grid, block);
  if (rc != 0) return rc;
  dp4a_matmul_add_clamp_kernel<<<grid, block, 0, stream>>>(
      A, B, base, out, M, N, K, total);
  return cudaGetLastError() == cudaSuccess ? 0 : -202;
}

}  // namespace pearl::capi::portable
