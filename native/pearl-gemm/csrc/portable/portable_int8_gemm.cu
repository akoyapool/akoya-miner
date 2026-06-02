// Hand-rolled int8 GEMM for the portable noise_A / noise_B projections.
//
// Native replacement for the old library-backed int8 helper, specialized for
// the two shapes that mattered to perf:
//
//   noise_A:  AxEBL projection — (M=16384, K=4096) @ (R={64,128}, K=4096).T
//             → (M=16384, R={64,128}) int32, then divided by 2^14 → fp16
//
//   noise_B:  EARxBpEB projection — same skinny pattern, σ-refresh only
//
//   ApEA / BpEB add+clamp:  also routed here via int8_matmul_i32, M wide
//
// General GEMM libraries are poor fits for these shapes: the AxEBL projection
// has N=R∈{64,128}, and the binary cost of carrying another runtime library is
// wasted on what is fundamentally a single mma.sync int8 dispatch.
//
// Byte identity vs cuBLASLt:
//   int32 accumulation wraps modulo 2^32 and is associative, so any tiling
//   produces the same int32 result.  The fp16 epilogue
//   (kernel_i32_div_to_fp16 in portable_int8_helpers.cu) stays unchanged
//   and consumes the same int32 scratch buffer.  Hence transcript bytes
//   downstream are bit-identical.
//
// Build:  compiled into libpearl_gemm_capi.so via setup.py (portable
// extension sources) and csrc/capi/Makefile.

#include "../capi/portable_int8_helpers.h"

#include <cstdint>
#include <cassert>
#include <atomic>
#include <cuda_runtime.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/mma_sm80.h>

namespace pearl::capi::portable {

namespace {

using namespace cute;

// ── Tile shape ──────────────────────────────────────────────────────────────
//   bM=64, bN={64,128}, bK=64 chosen so:
//     - bN covers R in one tile column (the tall-skinny case)
//     - bM=64  gives more CTAs (256 at M=16384) for better SM utilisation
//       on small batches than bM=128 would
//     - 4 warps × m=16 atom = bM=64 ✓
//   This keeps register pressure moderate (256 threads not needed); the
//   kernel is launch + memory bound, not compute bound, so we don't push
//   for max occupancy.
static constexpr int kBM      = 64;
static constexpr int kBN      = 128;
static constexpr int kBK      = 64;
static constexpr int kAtomK   = 32;
static constexpr int kThreads = 128;  // 4 warps
static constexpr int kStages  = 3;

using ElementIn  = int8_t;
using ElementAcc = int32_t;

// Same swizzle pattern as transcript_gemm_kernel — proven for sm_80+ LDSM.x4.
using SmemLayoutAtom = decltype(composition(
    Swizzle<2, 4, 3>{},
    Layout<Shape<_16, Int<kBK>>, Stride<Int<kBK>, _1>>{}));

using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtom{},
    make_shape(Int<kBM>{}, Int<kBK>{}, Int<kStages>{})));

template <int kTileN>
struct Int8GemmTraits {
  using SmTiledMma = TiledMMA<
      MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>,
      Layout<Shape<_4, _1, _1>>,
      Tile<Int<kBM>, Int<kTileN>, Int<kAtomK>>>;

  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<kTileN>{}, Int<kBK>{}, Int<kStages>{})));

  struct SharedStorage {
    alignas(16) ElementIn smem_A[cute::cosize_v<SmemLayoutA>];
    alignas(16) ElementIn smem_B[cute::cosize_v<SmemLayoutB>];
  };
};

// ── The kernel ──────────────────────────────────────────────────────────────
//   C(M, N) = A(M, K) @ B(N, K).T   (all int8 inputs, int32 output)
//   All strides row-major.  M, N, K must be multiples of (kBM, kTileN, kBK).
template <int kTileN>
__launch_bounds__(kThreads, 2)
__global__ void portable_int8_gemm_kernel(
    const int8_t* __restrict__ A_gmem,
    const int8_t* __restrict__ B_gmem,
    int32_t*      __restrict__ C_gmem,
    int M, int N, int K) {

  extern __shared__ uint8_t smem_raw[];
  using Traits = Int8GemmTraits<kTileN>;
  typename Traits::SharedStorage& smem =
      *reinterpret_cast<typename Traits::SharedStorage*>(smem_raw);

  const int m_tile = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int tid    = threadIdx.x;

  typename Traits::SmTiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);

  // Per-thread accumulator fragment.  cute computes its shape from the
  // TiledMma — for our case it's 32 int32 per thread (4 warps × 16 atoms_N
  // × 4 acc/atom / 32 lanes × 4 lanes-per-thread-group = 32, working out
  // to a (4, 2, 2, 2) layout per cute partition_C).
  Tensor cD = make_identity_tensor(Shape<Int<kBM>, Int<kTileN>>{});
  Tensor tCcD = thr_mma.partition_C(cD);
  constexpr int kFragSize = decltype(size(tCcD))::value;
  Tensor tCrC = make_tensor<ElementAcc>(Shape<Int<kFragSize>>{});
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kFragSize; ++j) tCrC(j) = 0;

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B),
                          typename Traits::SmemLayoutB{});

  Tensor mA = make_tensor(make_gmem_ptr(A_gmem), make_shape(M, K),
                          make_stride(K, _1{}));
  Tensor mB = make_tensor(make_gmem_ptr(B_gmem), make_shape(N, K),
                          make_stride(K, _1{}));
  Tensor gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{},
                         make_coord(m_tile, _));
  Tensor gB = local_tile(mB, Shape<Int<kTileN>, Int<kBK>>{},
                         make_coord(n_tile, _));

  const int K_TILES = K / kBK;

  // gmem → smem TiledCopy via cp.async, 16-byte granule.  128 threads × 16
  // bytes = 2 KiB per "layer".  A is 64×64=4K → 2 layers; B is 128×64=8K → 4.
  using GmemCopyAtom =
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, ElementIn>;
  auto g2s_copy_a = make_tiled_copy(
      GmemCopyAtom{},
      Layout<Shape<_32, _4>, Stride<_4, _1>>{},   // (rows, cols) per "layer"
      Layout<Shape<_1, _16>>{});                  // 16 ElementIn per cp.async
  auto g2s_copy_b = g2s_copy_a;

  auto g2s_thr_copy_a = g2s_copy_a.get_slice(tid);
  auto g2s_thr_copy_b = g2s_copy_b.get_slice(tid);
  Tensor tAgA = g2s_thr_copy_a.partition_S(gA);
  Tensor tAsA = g2s_thr_copy_a.partition_D(sA);
  Tensor tBgB = g2s_thr_copy_b.partition_S(gB);
  Tensor tBsB = g2s_thr_copy_b.partition_D(sB);

  auto issue_load = [&](int k_iter, int stg) {
    copy(g2s_copy_a, tAgA(_, _, _, k_iter), tAsA(_, _, _, stg));
    copy(g2s_copy_b, tBgB(_, _, _, k_iter), tBsB(_, _, _, stg));
    asm volatile("cp.async.commit_group;\n");
  };

  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < K_TILES) issue_load(s, s);
  }

  for (int k_iter = 0; k_iter < K_TILES; ++k_iter) {
    int stg = k_iter % kStages;

    asm volatile("cp.async.wait_group %0;\n" :: "n"(kStages - 2));
    __syncthreads();

    int next_k = k_iter + kStages - 1;
    if (next_k < K_TILES) {
      issue_load(next_k, next_k % kStages);
    } else {
      asm volatile("cp.async.commit_group;\n");
    }

    Tensor sA_stg = sA(_, _, stg);
    Tensor sB_stg = sB(_, _, stg);
    Tensor tCrA = thr_mma.partition_fragment_A(sA_stg);
    Tensor tCrB = thr_mma.partition_fragment_B(sB_stg);

    auto s2r_copy_a = make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>{}, tiled_mma);
    auto tXsA = s2r_copy_a.get_slice(tid).partition_S(sA_stg);
    auto tXrA = s2r_copy_a.get_slice(tid).retile_D(tCrA);
    copy(s2r_copy_a, tXsA, tXrA);

    auto s2r_copy_b = make_tiled_copy_B(
        Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>{}, tiled_mma);
    auto tXsB = s2r_copy_b.get_slice(tid).partition_S(sB_stg);
    auto tXrB = s2r_copy_b.get_slice(tid).retile_D(tCrB);
    copy(s2r_copy_b, tXsB, tXrB);

    auto tCrC_view = make_tensor(tCrC.data(), thr_mma.partition_fragment_C(
        make_tensor<ElementAcc>(Shape<Int<kBM>, Int<kTileN>>{})).layout());
    gemm(tiled_mma, tCrA, tCrB, tCrC_view);
  }

  // ── Write C tile to gmem (int32, M×N row-major) ────────────────────────
  // Each thread owns kFragSize accumulator slots at coords tCcD(j).
  int64_t c_base = (int64_t)m_tile * kBM * (int64_t)N
                   + (int64_t)n_tile * kTileN;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kFragSize; ++j) {
    int m = get<0>(tCcD(j));
    int n = get<1>(tCcD(j));
    C_gmem[c_base + (int64_t)m * N + n] = tCrC(j);
  }
}

// ── Fused add+clamp variant ────────────────────────────────────────────────
//   out_int8(M, N) = clamp(base_int8(M, N) + A(M, K) @ B(N, K).T, ±127)
//
// Replaces the two-pass int8_matmul_i32 + kernel_i32_add_to_int8 path
// used by int8_add_clamp_to_int8 (ApEA / BpEB computation).  Eliminates a
// (M·N·int32) DRAM scratch round-trip — 256 MiB per call at production
// shape (M=16384, N=4096).  This is the bulk of P4's win: not a transcript
// kernel rewrite, but absorbing the int32 materialisation step.
//
// Byte identity vs the two-pass path:
//   acc value is bit-identical (int32 add is associative).  base + acc is
//   computed in int32 with `__saturatef`-equivalent manual clamp before
//   the int8 cast — same operation order as the standalone epilogue.
template <int kTileN>
__launch_bounds__(kThreads, 2)
__global__ void portable_int8_gemm_addclamp_kernel(
    const int8_t* __restrict__ A_gmem,
    const int8_t* __restrict__ B_gmem,
    const int8_t* __restrict__ base_gmem,
    int8_t*       __restrict__ out_gmem,
    int M, int N, int K) {

  extern __shared__ uint8_t smem_raw[];
  using Traits = Int8GemmTraits<kTileN>;
  typename Traits::SharedStorage& smem =
      *reinterpret_cast<typename Traits::SharedStorage*>(smem_raw);

  const int m_tile = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int tid    = threadIdx.x;

  typename Traits::SmTiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);

  Tensor cD = make_identity_tensor(Shape<Int<kBM>, Int<kTileN>>{});
  Tensor tCcD = thr_mma.partition_C(cD);
  constexpr int kFragSize = decltype(size(tCcD))::value;
  Tensor tCrC = make_tensor<ElementAcc>(Shape<Int<kFragSize>>{});
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kFragSize; ++j) tCrC(j) = 0;

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B),
                          typename Traits::SmemLayoutB{});

  Tensor mA = make_tensor(make_gmem_ptr(A_gmem), make_shape(M, K),
                          make_stride(K, _1{}));
  Tensor mB = make_tensor(make_gmem_ptr(B_gmem), make_shape(N, K),
                          make_stride(K, _1{}));
  Tensor gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{},
                         make_coord(m_tile, _));
  Tensor gB = local_tile(mB, Shape<Int<kTileN>, Int<kBK>>{},
                         make_coord(n_tile, _));

  const int K_TILES = K / kBK;

  using GmemCopyAtom =
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, ElementIn>;
  auto g2s_copy_a = make_tiled_copy(
      GmemCopyAtom{},
      Layout<Shape<_32, _4>, Stride<_4, _1>>{},
      Layout<Shape<_1, _16>>{});
  auto g2s_copy_b = g2s_copy_a;

  auto g2s_thr_copy_a = g2s_copy_a.get_slice(tid);
  auto g2s_thr_copy_b = g2s_copy_b.get_slice(tid);
  Tensor tAgA = g2s_thr_copy_a.partition_S(gA);
  Tensor tAsA = g2s_thr_copy_a.partition_D(sA);
  Tensor tBgB = g2s_thr_copy_b.partition_S(gB);
  Tensor tBsB = g2s_thr_copy_b.partition_D(sB);

  auto issue_load = [&](int k_iter, int stg) {
    copy(g2s_copy_a, tAgA(_, _, _, k_iter), tAsA(_, _, _, stg));
    copy(g2s_copy_b, tBgB(_, _, _, k_iter), tBsB(_, _, _, stg));
    asm volatile("cp.async.commit_group;\n");
  };

  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < K_TILES) issue_load(s, s);
  }

  for (int k_iter = 0; k_iter < K_TILES; ++k_iter) {
    int stg = k_iter % kStages;

    asm volatile("cp.async.wait_group %0;\n" :: "n"(kStages - 2));
    __syncthreads();

    int next_k = k_iter + kStages - 1;
    if (next_k < K_TILES) {
      issue_load(next_k, next_k % kStages);
    } else {
      asm volatile("cp.async.commit_group;\n");
    }

    Tensor sA_stg = sA(_, _, stg);
    Tensor sB_stg = sB(_, _, stg);
    Tensor tCrA = thr_mma.partition_fragment_A(sA_stg);
    Tensor tCrB = thr_mma.partition_fragment_B(sB_stg);

    auto s2r_copy_a = make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>{}, tiled_mma);
    auto tXsA = s2r_copy_a.get_slice(tid).partition_S(sA_stg);
    auto tXrA = s2r_copy_a.get_slice(tid).retile_D(tCrA);
    copy(s2r_copy_a, tXsA, tXrA);

    auto s2r_copy_b = make_tiled_copy_B(
        Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>{}, tiled_mma);
    auto tXsB = s2r_copy_b.get_slice(tid).partition_S(sB_stg);
    auto tXrB = s2r_copy_b.get_slice(tid).retile_D(tCrB);
    copy(s2r_copy_b, tXsB, tXrB);

    auto tCrC_view = make_tensor(tCrC.data(), thr_mma.partition_fragment_C(
        make_tensor<ElementAcc>(Shape<Int<kBM>, Int<kTileN>>{})).layout());
    gemm(tiled_mma, tCrA, tCrB, tCrC_view);
  }

  // ── Fused epilogue: out_int8 = clamp(base_int8 + acc_int32, ±127) ──────
  int64_t base_offset = (int64_t)m_tile * kBM * (int64_t)N
                        + (int64_t)n_tile * kTileN;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kFragSize; ++j) {
    int m = get<0>(tCcD(j));
    int n = get<1>(tCcD(j));
    int64_t idx = base_offset + (int64_t)m * N + n;
    int32_t v = (int32_t)base_gmem[idx] + tCrC(j);
    if (v > 127)  v = 127;
    if (v < -128) v = -128;
    out_gmem[idx] = (int8_t)v;
  }
}

template <int kTileN>
int launch_int8_matmul_i32_native(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K, cudaStream_t stream) {
  if ((M % kBM) || (N % kTileN) || (K % kBK)) return -200;

  dim3 grid((unsigned)(M / kBM), (unsigned)(N / kTileN), 1u);
  dim3 block(kThreads);
  using Traits = Int8GemmTraits<kTileN>;
  size_t smem_bytes = sizeof(typename Traits::SharedStorage);

  static std::atomic<unsigned long long> attrs_set_mask{0};
  int dev = -1;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = -1;
  const unsigned long long bit = dev >= 0 ? (1ull << dev) : 0ull;
  if (bit == 0ull ||
      (attrs_set_mask.load(std::memory_order_acquire) & bit) == 0ull) {
    if (smem_bytes > 48 * 1024) {
      cudaFuncSetAttribute(portable_int8_gemm_kernel<kTileN>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)smem_bytes);
    }
    if (bit != 0ull) {
      attrs_set_mask.fetch_or(bit, std::memory_order_release);
    }
  }

  portable_int8_gemm_kernel<kTileN><<<grid, block, smem_bytes, stream>>>(
      A, B, C, M, N, K);
  return cudaGetLastError() == cudaSuccess ? 0 : -202;
}

template <int kTileN>
int launch_int8_matmul_add_clamp_native(
    const int8_t* A, const int8_t* B,
    const int8_t* base, int8_t* out,
    int M, int N, int K, cudaStream_t stream) {
  if ((M % kBM) || (N % kTileN) || (K % kBK)) return -200;

  dim3 grid((unsigned)(M / kBM), (unsigned)(N / kTileN), 1u);
  dim3 block(kThreads);
  using Traits = Int8GemmTraits<kTileN>;
  size_t smem_bytes = sizeof(typename Traits::SharedStorage);

  static std::atomic<unsigned long long> attrs_set_mask{0};
  int dev = -1;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = -1;
  const unsigned long long bit = dev >= 0 ? (1ull << dev) : 0ull;
  if (bit == 0ull ||
      (attrs_set_mask.load(std::memory_order_acquire) & bit) == 0ull) {
    if (smem_bytes > 48 * 1024) {
      cudaFuncSetAttribute(portable_int8_gemm_addclamp_kernel<kTileN>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)smem_bytes);
    }
    if (bit != 0ull) {
      attrs_set_mask.fetch_or(bit, std::memory_order_release);
    }
  }

  portable_int8_gemm_addclamp_kernel<kTileN><<<grid, block, smem_bytes, stream>>>(
      A, B, base, out, M, N, K);
  return cudaGetLastError() == cudaSuccess ? 0 : -202;
}

}  // namespace

// ── Public entry — called from portable_int8_helpers.cu ────────────────────
//
// Returns 0 on success, negative on shape-precondition failure.  The portable
// shapes we care about include R∈{64,128} projections and wide K/N add-clamp
// projections; they all satisfy one of the tile divisibility conditions below.
int int8_matmul_i32_native(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) return -201;
  if ((M % kBM) || (K % kBK)) return -200;
  if ((N % kBN) == 0) {
    return launch_int8_matmul_i32_native<kBN>(A, B, C, M, N, K, stream);
  }
  if ((N % 64) == 0) {
    return launch_int8_matmul_i32_native<64>(A, B, C, M, N, K, stream);
  }
  return -200;
}

// Fused int8 matmul + base-add + clamp → int8 output.
// Saves the (M·N·int32) DRAM scratch round-trip the two-pass path needed.
int int8_matmul_add_clamp_native(
    const int8_t* A, const int8_t* B,
    const int8_t* base, int8_t* out,
    int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) return -201;
  if ((M % kBM) || (K % kBK)) return -200;
  if ((N % kBN) == 0) {
    return launch_int8_matmul_add_clamp_native<kBN>(
        A, B, base, out, M, N, K, stream);
  }
  if ((N % 64) == 0) {
    return launch_int8_matmul_add_clamp_native<64>(
        A, B, base, out, M, N, K, stream);
  }
  return -200;
}

}  // namespace pearl::capi::portable
