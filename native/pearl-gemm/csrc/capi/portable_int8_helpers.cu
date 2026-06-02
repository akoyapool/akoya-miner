// See portable_int8_helpers.h. Implementation lives in a .cu so the
// elementwise kernels live in the same TU.
#include "portable_int8_helpers.h"

#include <cuda_fp16.h>

namespace pearl::capi::portable {

// Forward decl — defined in csrc/portable/portable_int8_gemm.cu.  Native
// mma.sync int8 GEMM that replaces the cuBLASLt path for shapes where the
// (M, N, K) all divide the kernel's tile (64, 128, 64).  Returns 0 on
// success, -200/-201/-202 on precondition / launch failure.
int int8_matmul_i32_native(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K, cudaStream_t stream);

// Fused int8 matmul + base-add + clamp; eliminates the int32 scratch
// round-trip used by the two-pass int8_add_clamp_to_int8 path.  Same
// shape preconditions; returns 0 / negative.
int int8_matmul_add_clamp_native(
    const int8_t* A, const int8_t* B,
    const int8_t* base, int8_t* out,
    int M, int N, int K, cudaStream_t stream);

namespace {

// ── elementwise kernels ────────────────────────────────────────────────────

__global__ void kernel_i32_div_to_fp16(
    const int32_t* __restrict__ in, __half* __restrict__ out,
    int64_t n, float inv_divisor) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float v = static_cast<float>(in[idx]) * inv_divisor;
  out[idx] = __float2half(v);
}

__global__ void kernel_add_clamp_i8(
    const int8_t*  __restrict__ base_i8,
    const int32_t* __restrict__ prod_i32,
    int8_t*        __restrict__ out_i8,
    int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int32_t v = prod_i32[idx] + static_cast<int32_t>(base_i8[idx]);
  if (v >  127) v =  127;
  if (v < -128) v = -128;
  out_i8[idx] = static_cast<int8_t>(v);
}

}  // namespace

// ── Public API ─────────────────────────────────────────────────────────────

int int8_matmul_i32(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K, cudaStream_t stream) {
  // Try the native mma.sync kernel first — eliminates the per-call
  // GEMM library overhead. Native failures return directly; there is no
  // silent cuBLASLt fallback in production builds.
  return int8_matmul_i32_native(A, B, C, M, N, K, stream);
}

int int8_matmul_into_fp16_div(
    const int8_t* A, const int8_t* B,
    void* out_fp16, int32_t* scratch_i32,
    int M, int N, int K, int divisor_log2,
  cudaStream_t stream) {
  int rc = int8_matmul_i32_native(A, B, scratch_i32, M, N, K, stream);
  if (rc != 0) return rc;
  int64_t n = (int64_t)M * (int64_t)N;
  int block = 256;
  int64_t grid = (n + block - 1) / block;
  if (grid > (1LL << 31) - 1) return -110;
  float inv_divisor = 1.0f / static_cast<float>(int64_t{1} << divisor_log2);
  kernel_i32_div_to_fp16<<<(int)grid, block, 0, stream>>>(
      scratch_i32, reinterpret_cast<__half*>(out_fp16), n, inv_divisor);
  return cudaGetLastError() == cudaSuccess ? 0 : -111;
}

int int8_add_clamp_to_int8(
    const int8_t* base_int8,
    const int8_t* add_left_int8, const int8_t* add_right_int8,
    int8_t* out_int8, int32_t* scratch_i32,
    int M, int K, int R, cudaStream_t stream) {
  // P4: try the fused matmul+add+clamp kernel first.  Eliminates the
  // M·K·int32 scratch round-trip (256 MiB at production shape).
  int rc = int8_matmul_add_clamp_native(
      add_left_int8, add_right_int8, base_int8, out_int8,
      M, K, R, stream);
  if (rc == 0) return 0;

  // Fallback: two-pass matmul-i32 → add+clamp epilogue.  Used when the
  // shape doesn't tile-divide (M%64, K%128, R%64).
  rc = int8_matmul_i32_native(add_left_int8, add_right_int8,
                              scratch_i32, M, K, R, stream);
  if (rc != 0) return rc;
  int64_t n = (int64_t)M * (int64_t)K;
  int block = 256;
  int64_t grid = (n + block - 1) / block;
  if (grid > (1LL << 31) - 1) return -120;
  kernel_add_clamp_i8<<<(int)grid, block, 0, stream>>>(
      base_int8, scratch_i32, out_int8, n);
  return cudaGetLastError() == cudaSuccess ? 0 : -121;
}

}  // namespace pearl::capi::portable
