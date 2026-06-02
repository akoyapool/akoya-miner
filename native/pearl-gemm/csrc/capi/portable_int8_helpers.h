// Torch-free portable helpers used by libpearl_gemm_capi.so when built with
// PEARL_GEMM_ARCH=portable. Replaces the at::_int_mm + ATen elementwise
// pieces of csrc/gemm/pearl_gemm_api.cpp's portable impls with native int8
// GEMM + tiny custom CUDA kernels.
//
// All matrices are row-major.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pearl::capi::portable {

// C(int32) = A(int8) @ B(int8).T   ── all row-major
//   A: (M, K)
//   B: (N, K)   (already in the layout produced by .t().contiguous())
//   C: (M, N)
// Internally invokes the native mma.sync int8 GEMM. Returns 0 on success,
// negative on error.
int int8_matmul_i32(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K, cudaStream_t stream);

// out_fp16 = (A @ B.T) / 2^divisor_log2,  cast int32 → fp16
//   A: (M, K) int8 row-major     B: (N, K) int8 row-major
//   out: (M, N) fp16 row-major
// Uses int8_matmul_i32 + a fused divide-and-cast kernel into a temporary
// int32 scratch (must be (M*N) int32s, allocated by caller).
int int8_matmul_into_fp16_div(
    const int8_t* A, const int8_t* B,
    void* out_fp16, int32_t* scratch_i32,
    int M, int N, int K, int divisor_log2,
    cudaStream_t stream);

// out_int8 = clamp(base_int8 + add_left_int8 @ add_right_int8.T, ±127)
//   base, out: (M, K) int8 row-major
//   add_left:  (M, R) int8 row-major
//   add_right: (K, R) int8 row-major  (already in the layout produced by .t().contiguous())
// Caller allocates scratch_i32 of size (M*K) int32s.
int int8_add_clamp_to_int8(
    const int8_t* base_int8,
    const int8_t* add_left_int8, const int8_t* add_right_int8,
    int8_t* out_int8, int32_t* scratch_i32,
    int M, int K, int R, cudaStream_t stream);

// In-bytes scratch for the temporaries used by int8_matmul_into_fp16_div
// and int8_add_clamp_to_int8.
inline int64_t scratch_bytes_for_matmul_i32(int M, int N) {
  return static_cast<int64_t>(M) * static_cast<int64_t>(N) * sizeof(int32_t);
}

}  // namespace pearl::capi::portable
