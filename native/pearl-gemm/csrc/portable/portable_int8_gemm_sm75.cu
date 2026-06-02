// Turing SM75 int8 helper GEMMs for portable noise_A / noise_B paths.
//
// This keeps the portable_int8_helpers API used by pearl_gemm_capi.cpp, but
// replaces the Volta/Turing DP4A fallback with exact Turing int8 tensor cores.
// Turing has no cp.async, so global-to-shared copies use ordinary vectorized
// UniversalCopy plus CTA barriers.

#include "../capi/portable_int8_helpers.h"

#include <cstdint>
#include <cuda_runtime.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/tensor.hpp>

namespace pearl::capi::portable {
namespace {

using namespace cute;

static constexpr int kBM = 64;
static constexpr int kBN = 128;
static constexpr int kBK = 64;
static constexpr int kAtomK = 16;
static constexpr int kThreads = 128;

using ElementIn = int8_t;
using ElementAcc = int32_t;

using SmemLayoutAtom = decltype(composition(
    Swizzle<2, 4, 3>{},
    Layout<Shape<_8, Int<kBK>>, Stride<Int<kBK>, _1>>{}));

using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtom{},
    make_shape(Int<kBM>{}, Int<kBK>{})));

template <int kTileN>
struct Int8GemmTraitsSm75 {
  using SmTiledMma = TiledMMA<
      MMA_Atom<SM75_8x8x16_S32S8S8S32_TN>,
      Layout<Shape<_4, _1, _1>>,
      Tile<Int<kBM>, Int<kTileN>, Int<kAtomK>>>;

  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<kTileN>{}, Int<kBK>{})));

  struct SharedStorage {
    alignas(16) ElementIn smem_A[cute::cosize_v<SmemLayoutA>];
    alignas(16) ElementIn smem_B[cute::cosize_v<SmemLayoutB>];
  };
};

template <int kTileN>
__launch_bounds__(kThreads, 2)
__global__ void int8_gemm_kernel_sm75(
    const int8_t* __restrict__ A_gmem,
    const int8_t* __restrict__ B_gmem,
    int32_t* __restrict__ C_gmem,
    int M, int N, int K) {

  extern __shared__ uint8_t smem_raw[];
  using Traits = Int8GemmTraitsSm75<kTileN>;
  typename Traits::SharedStorage& smem =
      *reinterpret_cast<typename Traits::SharedStorage*>(smem_raw);

  const int m_tile = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int tid = threadIdx.x;

  typename Traits::SmTiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);

  Tensor cD = make_identity_tensor(Shape<Int<kBM>, Int<kTileN>>{});
  Tensor tCcD = thr_mma.partition_C(cD);
  constexpr int kFragSize = decltype(size(tCcD))::value;

  Tensor tCrC = make_tensor<ElementAcc>(Shape<Int<kFragSize>>{});
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kFragSize; ++j) {
    tCrC(j) = 0;
  }

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B),
                          typename Traits::SmemLayoutB{});

  Tensor mA = make_tensor(make_gmem_ptr(A_gmem),
                          make_shape(M, K),
                          make_stride(K, _1{}));
  Tensor mB = make_tensor(make_gmem_ptr(B_gmem),
                          make_shape(N, K),
                          make_stride(K, _1{}));
  Tensor gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{},
                         make_coord(m_tile, _));
  Tensor gB = local_tile(mB, Shape<Int<kTileN>, Int<kBK>>{},
                         make_coord(n_tile, _));

  using GmemCopyAtom = Copy_Atom<UniversalCopy<cute::uint128_t>, ElementIn>;
  auto g2s_copy = make_tiled_copy(
      GmemCopyAtom{},
      Layout<Shape<_32, _4>, Stride<_4, _1>>{},
      Layout<Shape<_1, _16>>{});

  auto g2s_thr_copy = g2s_copy.get_slice(tid);
  Tensor tAgA = g2s_thr_copy.partition_S(gA);
  Tensor tAsA = g2s_thr_copy.partition_D(sA);
  Tensor tBgB = g2s_thr_copy.partition_S(gB);
  Tensor tBsB = g2s_thr_copy.partition_D(sB);

  const int K_TILES = K / kBK;
  for (int k_iter = 0; k_iter < K_TILES; ++k_iter) {
    copy(g2s_copy, tAgA(_, _, _, k_iter), tAsA);
    copy(g2s_copy, tBgB(_, _, _, k_iter), tBsB);
    __syncthreads();

    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);

    auto s2r_copy_a = make_tiled_copy_A(
        Copy_Atom<SM75_U32x1_LDSM_N, ElementIn>{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_copy_a.get_slice(tid);
    auto tXsA = s2r_thr_copy_a.partition_S(sA);
    auto tXrA = s2r_thr_copy_a.retile_D(tCrA);
    copy(s2r_copy_a, tXsA, tXrA);

    auto s2r_copy_b = make_tiled_copy_B(
        Copy_Atom<SM75_U32x1_LDSM_N, ElementIn>{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_copy_b.get_slice(tid);
    auto tXsB = s2r_thr_copy_b.partition_S(sB);
    auto tXrB = s2r_thr_copy_b.retile_D(tCrB);
    copy(s2r_copy_b, tXsB, tXrB);

    auto tCrC_view = make_tensor(tCrC.data(), thr_mma.partition_fragment_C(
        make_tensor<ElementAcc>(Shape<Int<kBM>, Int<kTileN>>{})).layout());
    gemm(tiled_mma, tCrA, tCrB, tCrC_view);
    __syncthreads();
  }

  int64_t c_base = (int64_t)m_tile * kBM * (int64_t)N
                   + (int64_t)n_tile * kTileN;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kFragSize; ++j) {
    int m = get<0>(tCcD(j));
    int n = get<1>(tCcD(j));
    C_gmem[c_base + (int64_t)m * N + n] = tCrC(j);
  }
}

template <int kTileN>
__launch_bounds__(kThreads, 2)
__global__ void int8_gemm_addclamp_kernel_sm75(
    const int8_t* __restrict__ A_gmem,
    const int8_t* __restrict__ B_gmem,
    const int8_t* __restrict__ base_gmem,
    int8_t* __restrict__ out_gmem,
    int M, int N, int K) {

  extern __shared__ uint8_t smem_raw[];
  using Traits = Int8GemmTraitsSm75<kTileN>;
  typename Traits::SharedStorage& smem =
      *reinterpret_cast<typename Traits::SharedStorage*>(smem_raw);

  const int m_tile = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int tid = threadIdx.x;

  typename Traits::SmTiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);

  Tensor cD = make_identity_tensor(Shape<Int<kBM>, Int<kTileN>>{});
  Tensor tCcD = thr_mma.partition_C(cD);
  constexpr int kFragSize = decltype(size(tCcD))::value;

  Tensor tCrC = make_tensor<ElementAcc>(Shape<Int<kFragSize>>{});
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kFragSize; ++j) {
    tCrC(j) = 0;
  }

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B),
                          typename Traits::SmemLayoutB{});

  Tensor mA = make_tensor(make_gmem_ptr(A_gmem),
                          make_shape(M, K),
                          make_stride(K, _1{}));
  Tensor mB = make_tensor(make_gmem_ptr(B_gmem),
                          make_shape(N, K),
                          make_stride(K, _1{}));
  Tensor gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{},
                         make_coord(m_tile, _));
  Tensor gB = local_tile(mB, Shape<Int<kTileN>, Int<kBK>>{},
                         make_coord(n_tile, _));

  using GmemCopyAtom = Copy_Atom<UniversalCopy<cute::uint128_t>, ElementIn>;
  auto g2s_copy = make_tiled_copy(
      GmemCopyAtom{},
      Layout<Shape<_32, _4>, Stride<_4, _1>>{},
      Layout<Shape<_1, _16>>{});

  auto g2s_thr_copy = g2s_copy.get_slice(tid);
  Tensor tAgA = g2s_thr_copy.partition_S(gA);
  Tensor tAsA = g2s_thr_copy.partition_D(sA);
  Tensor tBgB = g2s_thr_copy.partition_S(gB);
  Tensor tBsB = g2s_thr_copy.partition_D(sB);

  const int K_TILES = K / kBK;
  for (int k_iter = 0; k_iter < K_TILES; ++k_iter) {
    copy(g2s_copy, tAgA(_, _, _, k_iter), tAsA);
    copy(g2s_copy, tBgB(_, _, _, k_iter), tBsB);
    __syncthreads();

    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);

    auto s2r_copy_a = make_tiled_copy_A(
        Copy_Atom<SM75_U32x1_LDSM_N, ElementIn>{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_copy_a.get_slice(tid);
    auto tXsA = s2r_thr_copy_a.partition_S(sA);
    auto tXrA = s2r_thr_copy_a.retile_D(tCrA);
    copy(s2r_copy_a, tXsA, tXrA);

    auto s2r_copy_b = make_tiled_copy_B(
        Copy_Atom<SM75_U32x1_LDSM_N, ElementIn>{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_copy_b.get_slice(tid);
    auto tXsB = s2r_thr_copy_b.partition_S(sB);
    auto tXrB = s2r_thr_copy_b.retile_D(tCrB);
    copy(s2r_copy_b, tXsB, tXrB);

    auto tCrC_view = make_tensor(tCrC.data(), thr_mma.partition_fragment_C(
        make_tensor<ElementAcc>(Shape<Int<kBM>, Int<kTileN>>{})).layout());
    gemm(tiled_mma, tCrA, tCrB, tCrC_view);
    __syncthreads();
  }

  int64_t base_offset = (int64_t)m_tile * kBM * (int64_t)N
                        + (int64_t)n_tile * kTileN;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kFragSize; ++j) {
    int m = get<0>(tCcD(j));
    int n = get<1>(tCcD(j));
    int64_t idx = base_offset + (int64_t)m * N + n;
    int32_t v = (int32_t)base_gmem[idx] + tCrC(j);
    if (v > 127) v = 127;
    if (v < -128) v = -128;
    out_gmem[idx] = (int8_t)v;
  }
}

template <int kTileN>
int launch_int8_matmul_i32_native_sm75(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K, cudaStream_t stream) {
  if ((M % kBM) || (N % kTileN) || (K % kBK)) return -200;

  dim3 grid(static_cast<unsigned>(M / kBM),
            static_cast<unsigned>(N / kTileN),
            1u);
  dim3 block(kThreads);
  using Traits = Int8GemmTraitsSm75<kTileN>;
  int8_gemm_kernel_sm75<kTileN>
      <<<grid, block, sizeof(typename Traits::SharedStorage), stream>>>(
          A, B, C, M, N, K);
  return cudaGetLastError() == cudaSuccess ? 0 : -202;
}

template <int kTileN>
int launch_int8_matmul_add_clamp_native_sm75(
    const int8_t* A, const int8_t* B,
    const int8_t* base, int8_t* out,
    int M, int N, int K, cudaStream_t stream) {
  if ((M % kBM) || (N % kTileN) || (K % kBK)) return -200;

  dim3 grid(static_cast<unsigned>(M / kBM),
            static_cast<unsigned>(N / kTileN),
            1u);
  dim3 block(kThreads);
  using Traits = Int8GemmTraitsSm75<kTileN>;
  int8_gemm_addclamp_kernel_sm75<kTileN>
      <<<grid, block, sizeof(typename Traits::SharedStorage), stream>>>(
          A, B, base, out, M, N, K);
  return cudaGetLastError() == cudaSuccess ? 0 : -202;
}

}  // namespace

int int8_matmul_i32_native(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) return -201;
  if ((M % kBM) || (K % kBK)) return -200;
  if ((N % kBN) == 0) {
    return launch_int8_matmul_i32_native_sm75<kBN>(A, B, C, M, N, K, stream);
  }
  if ((N % 64) == 0) {
    return launch_int8_matmul_i32_native_sm75<64>(A, B, C, M, N, K, stream);
  }
  return -200;
}

int int8_matmul_add_clamp_native(
    const int8_t* A, const int8_t* B,
    const int8_t* base, int8_t* out,
    int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) return -201;
  if ((M % kBM) || (K % kBK)) return -200;
  if ((N % kBN) == 0) {
    return launch_int8_matmul_add_clamp_native_sm75<kBN>(
        A, B, base, out, M, N, K, stream);
  }
  if ((N % 64) == 0) {
    return launch_int8_matmul_add_clamp_native_sm75<64>(
        A, B, base, out, M, N, K, stream);
  }
  return -200;
}

}  // namespace pearl::capi::portable
