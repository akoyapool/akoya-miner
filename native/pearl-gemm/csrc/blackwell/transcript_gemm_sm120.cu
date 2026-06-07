// Blackwell/SM120 fused int8 GEMM + transcript snapshot kernel.
//
// This source is the dedicated RTX 50-series / sm_120a mining lane. It still
// exports pearl::consumer::* so the C API ABI stays unchanged, but Blackwell
// can now evolve independently from the shared Ampere/Ada consumer path. The
// proof-critical work intentionally remains on the exact SM80 mma.sync
// m16n8k32 int8 atom; PEARL_CONSUMER_* compile-time defines remain the sweep
// interface.
//
// Byte-identity with H100 WGMMA is preserved:  probe_sm80_layout.cu
// confirmed that
//     SM80 TiledMMA( SM80_16x8x32_S32S8S8S32_TN, AtomLayout (8,1,1),
//                    Tile(128, 256, 32) )
// produces partition_C coordinates byte-identical (32768/32768 slots) to
// the H100 WGMMA m64n256k32 TiledMma at every (thread, slot index).  This
// is because both ISAs use the same Tensor-Core 16x8 sub-fragment layout;
// only the warp-tiling differs (8 warps × m=16 vs 2 warpgroups × m=64),
// and the global thread→row mapping coincidentally matches.
//
// Inputs (from the rewritten noisy_gemm_portable_impl):
//   A_int8: (M, K) row-major contiguous int8     (= ApEA)
//   B_int8: (N, K) row-major contiguous int8     (= BpEB; we transpose
//                                                  internally to use MMA TN)
// Outputs:
//   C_int32: (M, N) row-major contiguous int32   (replaces at::_int_mm result)
//   transcript: per-(m_tile, n_tile, batch, thread, slot) u32, same layout
//               as transcript_kernel.cu's transcript_buffer_elems().
//
// After this kernel, the existing launch_transcript_finalize() reads from
// transcript and writes host_signal_header — unchanged.

#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <cctype>
#include <string>
#include <atomic>
#include <cuda_runtime.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/arch/barrier.h>

#include "../blake3/blake3_constants.hpp"
#include "../gemm/pow_utils.hpp"

#include "../portable/transcript_kernel.cuh"

#ifndef PEARL_GEMM_BLACKWELL
#error "transcript_gemm_sm120.cu requires PEARL_GEMM_BLACKWELL"
#endif

namespace pearl {
namespace consumer {

using namespace cute;

// ─── Architecture traits ────────────────────────────────────────────────────
#define PEARL_CONSUMER_DEFAULT_SWIZZLE_BITS 3
#define PEARL_CONSUMER_DEFAULT_STAGES 2
#define PEARL_CONSUMER_DEFAULT_KBLOCK 128
#define PEARL_CONSUMER_DEFAULT_MIN_BLOCKS 1

#ifndef PEARL_CONSUMER_USE_TMA_EXPERIMENT
#define PEARL_CONSUMER_USE_TMA_EXPERIMENT 0
#endif
#if PEARL_CONSUMER_USE_TMA_EXPERIMENT && !defined(PEARL_GEMM_BLACKWELL)
#error "PEARL_CONSUMER_USE_TMA_EXPERIMENT is Blackwell-only"
#endif

#ifndef PEARL_CONSUMER_MANUAL_IMMA
#define PEARL_CONSUMER_MANUAL_IMMA 0
#endif
#if PEARL_CONSUMER_MANUAL_IMMA != 0 && PEARL_CONSUMER_MANUAL_IMMA != 1
#error "PEARL_CONSUMER_MANUAL_IMMA must be 0 or 1"
#endif

// ─── Shape constants (must match transcript_kernel.cu) ───────────────────────
#ifndef PEARL_CONSUMER_BM
#define PEARL_CONSUMER_BM 128
#endif
#ifndef PEARL_CONSUMER_BN
#define PEARL_CONSUMER_BN 256
#endif
#if PEARL_CONSUMER_BM != 128
#error "PEARL_CONSUMER_BM must be 128; proof row/column extraction is canonical only for 128x256"
#endif
#if PEARL_CONSUMER_BN != 256
#error "PEARL_CONSUMER_BN must be 256; proof row/column extraction is canonical only for 128x256"
#endif
static constexpr int kBM = PEARL_CONSUMER_BM;
static constexpr int kBN = PEARL_CONSUMER_BN;
static constexpr int kAtomK = 32;                     // mma.sync m16n8k32 K
#ifndef PEARL_CONSUMER_KBLOCK
#define PEARL_CONSUMER_KBLOCK PEARL_CONSUMER_DEFAULT_KBLOCK
#endif
#if PEARL_CONSUMER_KBLOCK != 64 && PEARL_CONSUMER_KBLOCK != 128
#error "PEARL_CONSUMER_KBLOCK must be 64 or 128"
#endif
static constexpr int kBK = PEARL_CONSUMER_KBLOCK;     // smem K-tile
#if PEARL_CONSUMER_MANUAL_IMMA
static constexpr int kMmaKBlocks = kBK / kAtomK;      // per-stage mma.sync slices
#endif
static constexpr int kProofThreads = 256;             // canonical transcript lanes
static_assert(kProofThreads % 32 == 0,
              "canonical transcript lanes must be whole warps");
static constexpr int kConsumerWarps = kProofThreads / 32;
#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
static constexpr int kThreads = kProofThreads + 32;   // 8 consumer warps + 1 TMA producer warp
#else
static constexpr int kThreads = kProofThreads;        // 8 warps
#endif
static constexpr int kFragSize = (kBM * kBN) / kProofThreads; // per-thread acc slots
static_assert((kBM * kBN) % kProofThreads == 0,
              "CTA tile must divide evenly across 256 threads");
static constexpr int kTranscriptSlots = 16;           // = MSG_BLOCK_SIZE_U32

using ElementIn  = int8_t;
using ElementAcc = int32_t;

using TileShape_MNK = Shape<Int<kBM>, Int<kBN>, Int<kBK>>;
using HeaderTileShape_MNK = Shape<Int<kBM>, Int<kBN>, Int<128>>;

#ifndef PEARL_CONSUMER_XOR_ACCUMS
#define PEARL_CONSUMER_XOR_ACCUMS 4
#endif
#if PEARL_CONSUMER_XOR_ACCUMS != 4 && PEARL_CONSUMER_XOR_ACCUMS != 8 && \
    PEARL_CONSUMER_XOR_ACCUMS != 16
#error "PEARL_CONSUMER_XOR_ACCUMS must be 4, 8, or 16"
#endif

template <typename TensorType>
CUTLASS_DEVICE uint32_t xor_reduction_frag128(const TensorType& input_tensor) {
  static_assert(kFragSize == 128,
                "SM120 transcript fragment XOR is specialized for 128 slots");

#if PEARL_CONSUMER_XOR_ACCUMS == 4
  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < kFragSize; i += 8) {
    a0 = pearl::xor3_lop3(a0, static_cast<uint32_t>(input_tensor[i + 0]),
                          static_cast<uint32_t>(input_tensor[i + 4]));
    a1 = pearl::xor3_lop3(a1, static_cast<uint32_t>(input_tensor[i + 1]),
                          static_cast<uint32_t>(input_tensor[i + 5]));
    a2 = pearl::xor3_lop3(a2, static_cast<uint32_t>(input_tensor[i + 2]),
                          static_cast<uint32_t>(input_tensor[i + 6]));
    a3 = pearl::xor3_lop3(a3, static_cast<uint32_t>(input_tensor[i + 3]),
                          static_cast<uint32_t>(input_tensor[i + 7]));
  }

  uint32_t r0 = pearl::xor3_lop3(a0, a1, a2);
  return pearl::xor3_lop3(r0, a3, 0);
#elif PEARL_CONSUMER_XOR_ACCUMS == 8
  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  uint32_t a4 = 0, a5 = 0, a6 = 0, a7 = 0;

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < kFragSize; i += 16) {
    a0 = pearl::xor3_lop3(a0, static_cast<uint32_t>(input_tensor[i + 0]),
                          static_cast<uint32_t>(input_tensor[i + 8]));
    a1 = pearl::xor3_lop3(a1, static_cast<uint32_t>(input_tensor[i + 1]),
                          static_cast<uint32_t>(input_tensor[i + 9]));
    a2 = pearl::xor3_lop3(a2, static_cast<uint32_t>(input_tensor[i + 2]),
                          static_cast<uint32_t>(input_tensor[i + 10]));
    a3 = pearl::xor3_lop3(a3, static_cast<uint32_t>(input_tensor[i + 3]),
                          static_cast<uint32_t>(input_tensor[i + 11]));
    a4 = pearl::xor3_lop3(a4, static_cast<uint32_t>(input_tensor[i + 4]),
                          static_cast<uint32_t>(input_tensor[i + 12]));
    a5 = pearl::xor3_lop3(a5, static_cast<uint32_t>(input_tensor[i + 5]),
                          static_cast<uint32_t>(input_tensor[i + 13]));
    a6 = pearl::xor3_lop3(a6, static_cast<uint32_t>(input_tensor[i + 6]),
                          static_cast<uint32_t>(input_tensor[i + 14]));
    a7 = pearl::xor3_lop3(a7, static_cast<uint32_t>(input_tensor[i + 7]),
                          static_cast<uint32_t>(input_tensor[i + 15]));
  }

  uint32_t r0 = pearl::xor3_lop3(a0, a1, a2);
  uint32_t r1 = pearl::xor3_lop3(a3, a4, a5);
  uint32_t r2 = pearl::xor3_lop3(a6, a7, 0);
  return pearl::xor3_lop3(r0, r1, r2);
#else
  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  uint32_t a4 = 0, a5 = 0, a6 = 0, a7 = 0;
  uint32_t a8 = 0, a9 = 0, a10 = 0, a11 = 0;
  uint32_t a12 = 0, a13 = 0, a14 = 0, a15 = 0;

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < kFragSize; i += 32) {
    a0 = pearl::xor3_lop3(a0, static_cast<uint32_t>(input_tensor[i + 0]),
                          static_cast<uint32_t>(input_tensor[i + 16]));
    a1 = pearl::xor3_lop3(a1, static_cast<uint32_t>(input_tensor[i + 1]),
                          static_cast<uint32_t>(input_tensor[i + 17]));
    a2 = pearl::xor3_lop3(a2, static_cast<uint32_t>(input_tensor[i + 2]),
                          static_cast<uint32_t>(input_tensor[i + 18]));
    a3 = pearl::xor3_lop3(a3, static_cast<uint32_t>(input_tensor[i + 3]),
                          static_cast<uint32_t>(input_tensor[i + 19]));
    a4 = pearl::xor3_lop3(a4, static_cast<uint32_t>(input_tensor[i + 4]),
                          static_cast<uint32_t>(input_tensor[i + 20]));
    a5 = pearl::xor3_lop3(a5, static_cast<uint32_t>(input_tensor[i + 5]),
                          static_cast<uint32_t>(input_tensor[i + 21]));
    a6 = pearl::xor3_lop3(a6, static_cast<uint32_t>(input_tensor[i + 6]),
                          static_cast<uint32_t>(input_tensor[i + 22]));
    a7 = pearl::xor3_lop3(a7, static_cast<uint32_t>(input_tensor[i + 7]),
                          static_cast<uint32_t>(input_tensor[i + 23]));
    a8 = pearl::xor3_lop3(a8, static_cast<uint32_t>(input_tensor[i + 8]),
                          static_cast<uint32_t>(input_tensor[i + 24]));
    a9 = pearl::xor3_lop3(a9, static_cast<uint32_t>(input_tensor[i + 9]),
                          static_cast<uint32_t>(input_tensor[i + 25]));
    a10 = pearl::xor3_lop3(a10, static_cast<uint32_t>(input_tensor[i + 10]),
                           static_cast<uint32_t>(input_tensor[i + 26]));
    a11 = pearl::xor3_lop3(a11, static_cast<uint32_t>(input_tensor[i + 11]),
                           static_cast<uint32_t>(input_tensor[i + 27]));
    a12 = pearl::xor3_lop3(a12, static_cast<uint32_t>(input_tensor[i + 12]),
                           static_cast<uint32_t>(input_tensor[i + 28]));
    a13 = pearl::xor3_lop3(a13, static_cast<uint32_t>(input_tensor[i + 13]),
                           static_cast<uint32_t>(input_tensor[i + 29]));
    a14 = pearl::xor3_lop3(a14, static_cast<uint32_t>(input_tensor[i + 14]),
                           static_cast<uint32_t>(input_tensor[i + 30]));
    a15 = pearl::xor3_lop3(a15, static_cast<uint32_t>(input_tensor[i + 15]),
                           static_cast<uint32_t>(input_tensor[i + 31]));
  }

  uint32_t r0 = pearl::xor3_lop3(a0, a1, a2);
  uint32_t r1 = pearl::xor3_lop3(a3, a4, a5);
  uint32_t r2 = pearl::xor3_lop3(a6, a7, a8);
  uint32_t r3 = pearl::xor3_lop3(a9, a10, a11);
  uint32_t r4 = pearl::xor3_lop3(a12, a13, a14);
  uint32_t r5 = pearl::xor3_lop3(a15, 0, 0);
  uint32_t s0 = pearl::xor3_lop3(r0, r1, r2);
  uint32_t s1 = pearl::xor3_lop3(r3, r4, r5);
  return pearl::xor3_lop3(s0, s1, 0);
#endif
}

// Sm80 TiledMMA — verified byte-identical partition_C with WGMMA via
// probe_sm80_layout.cu.  The Tile<> argument's K dim is the MMA *atom* K
// (= 32), NOT the smem kBK.  partition_A/_B/partition_fragment_A/_B then
// produce MMA_K = kBK / kAtomK fragments per smem stage.
using Sm80TiledMma = TiledMMA<
    MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>,
    Layout<Shape<_8, _1, _1>>,
    Tile<Int<kBM>, Int<kBN>, Int<kAtomK>>>;

// ─── SMEM layout (Swizzle<2|3,4,3> for bank-conflict-free LDSM.x4) ───────────
// kBK=64/128 bytes per row gives each ldmatrix.x4 lane-group access a clean
// bank stride.  Atom shape (16, 64) is the canonical CUTLASS sm_80 int8
// pattern (default_gemm_configuration.hpp); Swizzle<2,4,3> swaps bits
// {4,5} with {7,8} of the byte address, so consecutive matrix rows hit
// disjoint bank sets, eliminating the 4-way conflict the K-major layout
// would otherwise have.
//
// Alpha's Blackwell-native path exposes Swizzle<3,4,3> in its template names.
// RunPod RTX 5090 headless benchmark at M=8192,N=262144 confirmed a small
// win over Swizzle<2,4,3> (300.78 vs 299.19 TMAD/s), so use it by default.
#ifndef PEARL_CONSUMER_SWIZZLE_BITS
#define PEARL_CONSUMER_SWIZZLE_BITS PEARL_CONSUMER_DEFAULT_SWIZZLE_BITS
#endif
#if PEARL_CONSUMER_SWIZZLE_BITS != 2 && PEARL_CONSUMER_SWIZZLE_BITS != 3
#error "PEARL_CONSUMER_SWIZZLE_BITS must be 2 or 3"
#endif
//
// A: (kBM=128, kBK=128) int8 = 16 KiB per stage.
// B: (kBN=256, kBK=128) int8 = 32 KiB per stage.
// Total smem/block = (kBM + kBN) * kBK * kStages bytes. Stage count, tile
// shape, swizzle, and launch-bounds minBlocks are compile-time knobs because
// the fastest point differs by SKU.
#ifndef PEARL_CONSUMER_STAGES
#define PEARL_CONSUMER_STAGES PEARL_CONSUMER_DEFAULT_STAGES
#endif
#if PEARL_CONSUMER_STAGES < 2 || PEARL_CONSUMER_STAGES > 4
#error "PEARL_CONSUMER_STAGES must be 2, 3, or 4"
#endif
static constexpr int kStages = PEARL_CONSUMER_STAGES;

using SmemLayoutAtomA = decltype(composition(
    Swizzle<PEARL_CONSUMER_SWIZZLE_BITS, 4, 3>{},
    Layout<Shape<_16, Int<kBK>>, Stride<Int<kBK>, _1>>{}));
using SmemLayoutAtomB = SmemLayoutAtomA;  // same atom shape works for B

using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    make_shape(Int<kBM>{}, Int<kBK>{}, Int<kStages>{})));
using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB{},
    make_shape(Int<kBN>{}, Int<kBK>{}, Int<kStages>{})));

#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
using SmemLayoutStageA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    make_shape(Int<kBM>{}, Int<kBK>{})));
using SmemLayoutStageB = decltype(tile_to_shape(
    SmemLayoutAtomB{},
    make_shape(Int<kBN>{}, Int<kBK>{})));
using GmemLayout2D = decltype(make_layout(
    make_shape(int(0), int(0)), make_stride(int(0), _1{})));
using GmemTensor2D = decltype(make_tensor(
    make_gmem_ptr<ElementIn>(nullptr), GmemLayout2D{}));
using TmaA = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, GmemTensor2D{}, SmemLayoutStageA{}));
using TmaB = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, GmemTensor2D{}, SmemLayoutStageB{}));
static constexpr uint32_t kTmaBytes =
    (uint32_t)(kBM * kBK + kBN * kBK) * (uint32_t)sizeof(ElementIn);
#endif

struct SharedStorage {
  alignas(16) ElementIn smem_A[cute::cosize_v<SmemLayoutA>];
  alignas(16) ElementIn smem_B[cute::cosize_v<SmemLayoutB>];
#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
  alignas(16) uint64_t full_barrier[kStages];
  alignas(16) uint64_t empty_barrier[kStages];
#endif
};

// ─── The fused kernel ───────────────────────────────────────────────────────
// Default minBlocks is conservative and sweepable per architecture. The fastest
// point differs by SKU, especially GA102 vs AD102/GB202.
#ifndef PEARL_CONSUMER_MIN_BLOCKS
#define PEARL_CONSUMER_MIN_BLOCKS PEARL_CONSUMER_DEFAULT_MIN_BLOCKS
#endif
#if PEARL_CONSUMER_MIN_BLOCKS < 1
#error "PEARL_CONSUMER_MIN_BLOCKS must be >= 1"
#endif
template <bool kHeadless>
__launch_bounds__(kThreads, PEARL_CONSUMER_MIN_BLOCKS)
__global__ void transcript_gemm_kernel_consumer(
#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
    __grid_constant__ const TmaA tma_a,
    __grid_constant__ const TmaB tma_b,
#endif
    ElementIn  const* __restrict__ A_gmem,    // (M, K) row-major
    ElementIn  const* __restrict__ B_gmem,    // (N, K) row-major
    ElementAcc*       __restrict__ C_gmem,    // (M, N) row-major int32 out
    uint32_t*         __restrict__ transcript,
    int M, int N, int K, int R,
    uint32_t const*   __restrict__ pow_target,
    uint32_t const*   __restrict__ pow_key,
    HostSignalSync*               host_signal_sync,
    HostSignalHeader*             host_signal_header_pinned) {

  extern __shared__ uint8_t smem_raw[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int m_tile = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int batch  = blockIdx.z;
  const int tid    = threadIdx.x;
#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
  const bool is_tma_producer = tid >= kProofThreads;
#endif

  const int num_m_tiles = M / kBM;
  const int num_n_tiles = N / kBN;

  // SMEM tensors for A and B with multi-stage shape.
  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B), SmemLayoutB{});

#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
  if (tid == 0) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < kStages; ++s) {
      cutlass::arch::ClusterBarrier::init(&smem.full_barrier[s], 1);
      cutlass::arch::ClusterBarrier::init(&smem.empty_barrier[s],
                                          kConsumerWarps);
    }
  }
  __syncthreads();
#endif

  // gmem tile views.
  // A_gmem: (M, K) → tile (kBM, kBK) at (m_tile, k_iter)
#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
  (void)A_gmem;
  (void)B_gmem;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
#else
  Tensor mA = make_tensor(make_gmem_ptr(A_gmem),
                          make_shape(M, K),
                          make_stride(K, _1{}));
  Tensor mB = make_tensor(make_gmem_ptr(B_gmem),
                          make_shape(N, K),
                          make_stride(K, _1{}));
#endif

  Tensor gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{},
                         make_coord(m_tile, _));   // (kBM, kBK, K/kBK)
  Tensor gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{},
                         make_coord(n_tile, _));   // (kBN, kBK, K/kBK)

  const int K_TILES = K / kBK;
  const int reduce_every_k = R / kBK;       // R=128: kBK=128 → 1

  // ── gmem→smem TiledCopy via cp.async ─────────────────────────────────
  // 16-byte cp.async granule (uint128_t).  Thread layout (64,4) k-major
  // and value layout (1,16) k-major:  256 threads cooperatively load 64
  // rows × 64 cols (= 4 KiB) per "layer".  Per K-tile:
  //   A is 128×64 = 8 KiB → 2 layers per thread per K-tile (CPY=16, REST_M=2)
  //   B is 256×64 = 16 KiB → 4 layers per thread per K-tile (CPY=16, REST_M=4)
  // Routing through cute's TiledCopy ensures cp.async writes hit the same
  // swizzled smem addresses that LDSM reads from below — without that
  // consistency, the swizzle would corrupt the data.
#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
  auto cta_tma_a = tma_a.get_slice(_0{});
  auto cta_tma_b = tma_b.get_slice(_0{});
  Tensor tAgA = cta_tma_a.partition_S(gA);
  Tensor tAsA = cta_tma_a.partition_D(sA);
  Tensor tBgB = cta_tma_b.partition_S(gB);
  Tensor tBsB = cta_tma_b.partition_D(sB);

  auto issue_load = [&](int k_iter, int stg) {
    uint64_t* fb = &smem.full_barrier[stg];
    if (cute::elect_one_sync()) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
          reinterpret_cast<
              cutlass::arch::ClusterTransactionBarrier::ValueType*>(fb),
          kTmaBytes);
      copy(tma_a.with(*fb), tAgA(_, _, _, k_iter), tAsA(_, _, _, stg));
      copy(tma_b.with(*fb), tBgB(_, _, _, k_iter), tBsB(_, _, _, stg));
    }
  };
#else
#ifndef PEARL_CONSUMER_CP_ASYNC_CACHE_ALWAYS
#define PEARL_CONSUMER_CP_ASYNC_CACHE_ALWAYS 0
#endif
#ifndef PEARL_CONSUMER_A_CP_ASYNC_CACHE_ALWAYS
#define PEARL_CONSUMER_A_CP_ASYNC_CACHE_ALWAYS PEARL_CONSUMER_CP_ASYNC_CACHE_ALWAYS
#endif
#ifndef PEARL_CONSUMER_B_CP_ASYNC_CACHE_ALWAYS
#define PEARL_CONSUMER_B_CP_ASYNC_CACHE_ALWAYS PEARL_CONSUMER_CP_ASYNC_CACHE_ALWAYS
#endif
#if PEARL_CONSUMER_A_CP_ASYNC_CACHE_ALWAYS
  using GmemCopyAtomA =
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, ElementIn>;
#else
  using GmemCopyAtomA =
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, ElementIn>;
#endif
#if PEARL_CONSUMER_B_CP_ASYNC_CACHE_ALWAYS
  using GmemCopyAtomB =
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, ElementIn>;
#else
  using GmemCopyAtomB =
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, ElementIn>;
#endif
  auto g2s_copy_a = make_tiled_copy(
      GmemCopyAtomA{},
      Layout<Shape<_64, _4>, Stride<_4, _1>>{},
      Layout<Shape<_1, _16>>{});
  auto g2s_copy_b = make_tiled_copy(
      GmemCopyAtomB{},
      Layout<Shape<_64, _4>, Stride<_4, _1>>{},
      Layout<Shape<_1, _16>>{});

  auto g2s_thr_copy_a = g2s_copy_a.get_slice(tid);
  auto g2s_thr_copy_b = g2s_copy_b.get_slice(tid);
  Tensor tAgA = g2s_thr_copy_a.partition_S(gA);   // (CPY, REST_M, REST_K, K_TILES)
  Tensor tAsA = g2s_thr_copy_a.partition_D(sA);   // (CPY, REST_M, REST_K, kStages)
  Tensor tBgB = g2s_thr_copy_b.partition_S(gB);   // (CPY, REST_M, REST_K, K_TILES)
  Tensor tBsB = g2s_thr_copy_b.partition_D(sB);   // (CPY, REST_M, REST_K, kStages)

  auto issue_load = [&](int k_iter, int stg) {
    copy(g2s_copy_a, tAgA(_, _, _, k_iter), tAsA(_, _, _, stg));
    copy(g2s_copy_b, tBgB(_, _, _, k_iter), tBsB(_, _, _, stg));
    asm volatile("cp.async.commit_group;\n");
  };
#endif

#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
  if (is_tma_producer) {
    for (int k_iter = 0; k_iter < K_TILES; ++k_iter) {
      const int stg = k_iter % kStages;
      if (k_iter >= kStages) {
        cutlass::arch::ClusterBarrier::wait(
            &smem.empty_barrier[stg],
            ((k_iter / kStages) - 1) & 1);
      }
      issue_load(k_iter, stg);
    }
    return;
  }
#endif

  // Per-thread accumulator (128 int32 in registers).  In TMA mode only the
  // first 256 threads reach here; the extra warp is a pure producer.
  Sm80TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);

  // Identity-tensor partition: tells us which (m, n) of the (kBM, kBN) tile
  // each accumulator slot maps to — same as WGMMA per probe_sm80_layout.cu.
  Tensor cD   = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
  Tensor tCcD = thr_mma.partition_C(cD);
  static_assert(decltype(size(tCcD))::value == kFragSize,
                "fragment size must be 128");

  Tensor tCrC = make_tensor<ElementAcc>(
      Shape<Int<kFragSize>>{});
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kFragSize; ++j) tCrC(j) = 0;

  // Per-thread transcript (16 u32 in registers).
  uint32_t transcript_local[kTranscriptSlots];
  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < kTranscriptSlots; ++s) transcript_local[s] = 0;

  // Prologue: issue first kStages-1 loads.
#if !PEARL_CONSUMER_USE_TMA_EXPERIMENT
  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < K_TILES) issue_load(s, s);
  }
#endif

  for (int k_iter = 0; k_iter < K_TILES; ++k_iter) {
    int stg = k_iter % kStages;

    // Wait for the load of this iter's stage to land, sync all threads
    // (also a barrier between previous iter's MMA-reads-of-smem and the
    // upcoming prefetch into the same stage).  With kStages=3 prefetches
    // in flight, wait_group<1> drains 2 oldest groups, leaving the most
    // recent prefetch in flight — i.e. this iter's stage is ready.
#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
    cutlass::arch::ClusterBarrier::wait(&smem.full_barrier[stg],
                                        (k_iter / kStages) & 1);
#else
    asm volatile("cp.async.wait_group %0;\n" :: "n"(kStages - 2));
    __syncthreads();

    // Issue the next prefetch (k_iter + kStages - 1).
    int next_k = k_iter + kStages - 1;
    if (next_k < K_TILES) {
      issue_load(next_k, next_k % kStages);
    } else {
      asm volatile("cp.async.commit_group;\n");
    }
#endif

    // ── Async snapshot reduction ──────────────────────────────────────────
    // Snapshot XOR-reduce for the boundary closed at the END of the
    // PREVIOUS k-iter is issued HERE, in the shadow of the cp.async
    // commit above and before the upcoming ldmatrix.  tCrC has not been
    // touched since the previous mma, so the reduce sees identical state
    // to the original "after-mma" placement, producing byte-identical
    // transcript bytes.  Effect: the ~7-instruction lop3 dep chain runs
    // concurrently with the MIO short-scoreboard wait on ldmatrix,
    // instead of serialising after mma.
    if (k_iter > 0 && (k_iter % reduce_every_k) == 0) {
      uint32_t hash = xor_reduction_frag128(tCrC);
      int snapshot_idx = (k_iter / reduce_every_k) - 1;
      int slot = snapshot_idx % kTranscriptSlots;
      transcript_local[slot] =
          pearl::rotl_xor<pearl::HASH_ACCUMULATE_ROTATION>(
              transcript_local[slot], hash);
    }

    // Bind register fragments to SMEM stage slice.
    Tensor sA_stg = sA(_, _, stg);     // (kBM, kBK)
    Tensor sB_stg = sB(_, _, stg);     // (kBN, kBK)
#if PEARL_CONSUMER_MANUAL_IMMA
    // Experimental SM120 mainloop: stream one kAtomK=32 slice at a time so
    // the compiler can reuse A/B operand fragment registers between mma.sync
    // groups.  This keeps CUTE's verified partition_C layout and only changes
    // the per-stage A/B fragment lifetime.
    auto tCrC_view = make_tensor(tCrC.data(), thr_mma.partition_fragment_C(
        make_tensor<ElementAcc>(Shape<Int<kBM>, Int<kBN>>{})).layout());

    auto s2r_copy_a = make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_copy_a.get_slice(tid);
    auto s2r_copy_b = make_tiled_copy_B(
        Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_copy_b.get_slice(tid);

    CUTLASS_PRAGMA_UNROLL
    for (int kb = 0; kb < kMmaKBlocks; ++kb) {
      Tensor sA_k = local_tile(sA_stg, Shape<Int<kBM>, Int<kAtomK>>{},
                               make_coord(_0{}, kb));
      Tensor sB_k = local_tile(sB_stg, Shape<Int<kBN>, Int<kAtomK>>{},
                               make_coord(_0{}, kb));
      Tensor tCrA = thr_mma.partition_fragment_A(sA_k);
      Tensor tCrB = thr_mma.partition_fragment_B(sB_k);

      auto tXsA = s2r_thr_copy_a.partition_S(sA_k);
      auto tXrA = s2r_thr_copy_a.retile_D(tCrA);
      copy(s2r_copy_a, tXsA, tXrA);

      auto tXsB = s2r_thr_copy_b.partition_S(sB_k);
      auto tXrB = s2r_thr_copy_b.retile_D(tCrB);
      copy(s2r_copy_b, tXsB, tXrB);

      gemm(tiled_mma, tCrA, tCrB, tCrC_view);
    }

#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
    if ((tid & 31) == 0) {
      cutlass::arch::ClusterBarrier::arrive(&smem.empty_barrier[stg]);
    }
#endif
#else
    Tensor tCrA = thr_mma.partition_fragment_A(sA_stg);
    Tensor tCrB = thr_mma.partition_fragment_B(sB_stg);

    // smem→reg via ldmatrix.x4 (SM75_U32x4_LDSM_N).
    //
    // Each warp's ldmatrix.x4 loads 16 lanes × 16 bytes = 16×32 int8 = one
    // mma.sync m16n8k32 A operand fragment per call.  The same instruction
    // works for B because the per-thread byte layout is identical from
    // ldmatrix's perspective (it doesn't know about A vs B); cute's
    // make_tiled_copy_A / _B retile the destination to match each operand's
    // mma fragment shape.
    //
    // NOTE: smem layout is K-major with row stride = kBK. The swizzled layout
    // above is required so ldmatrix sees the same logical rows after cp.async.
    auto s2r_copy_a = make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_copy_a.get_slice(tid);
    auto tXsA = s2r_thr_copy_a.partition_S(sA_stg);
    auto tXrA = s2r_thr_copy_a.retile_D(tCrA);
    copy(s2r_copy_a, tXsA, tXrA);

    auto s2r_copy_b = make_tiled_copy_B(
        Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_copy_b.get_slice(tid);
    auto tXsB = s2r_thr_copy_b.partition_S(sB_stg);
    auto tXrB = s2r_thr_copy_b.retile_D(tCrB);
    copy(s2r_copy_b, tXsB, tXrB);

#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
    if ((tid & 31) == 0) {
      cutlass::arch::ClusterBarrier::arrive(&smem.empty_barrier[stg]);
    }
#endif

    // Issue all mma.sync ops for this k-iter.  cute::gemm dispatches to
    // the SM80_16x8x32_S32S8S8S32_TN atom for each (MMA_M, MMA_N) pair.
    // Reshape tCrC to the shape cute::gemm expects.
    auto tCrC_view = make_tensor(tCrC.data(), thr_mma.partition_fragment_C(
        make_tensor<ElementAcc>(Shape<Int<kBM>, Int<kBN>>{})).layout());
    gemm(tiled_mma, tCrA, tCrB, tCrC_view);
#endif

    // Note: no __syncthreads here.  Next iteration's wait_group + sync
    // gates the next stage's smem reuse correctly.
  }

  // ── Tail snapshot for the final boundary closed at end of last iter ──
  // The shifted-by-one snapshot scheme above never fires for the boundary
  // at k_iter == K_TILES (since the loop exits first).  Emit it here so
  // the transcript covers the full K range identically to the pre-shift
  // version.  K is consensus-fixed so K_TILES % reduce_every_k is known
  // at compile time on the host (= 32 snapshots for K=4096, R=128, kBK=128).
  if ((K_TILES % reduce_every_k) == 0) {
    uint32_t hash = xor_reduction_frag128(tCrC);
    int snapshot_idx = (K_TILES / reduce_every_k) - 1;
    int slot = snapshot_idx % kTranscriptSlots;
    transcript_local[slot] =
        pearl::rotl_xor<pearl::HASH_ACCUMULATE_ROTATION>(
            transcript_local[slot], hash);
  }

  // ── Optional in-kernel finalization ─────────────────────────────────
  //
  // Alpha's 5090 miner exposes a "headless_mine_kernel" and xored_tile
  // debugging; the important trick appears to be keeping the XOR transcript
  // in registers and checking the target here instead of spilling 16 words
  // per thread to gmem and launching transcript_finalize_kernel.
  if constexpr (kHeadless) {
    Tensor transcript_rmem = make_tensor<uint32_t>(
        Int<kTranscriptSlots>{});
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < kTranscriptSlots; ++s) {
      transcript_rmem(s) = transcript_local[s];
    }

    bool block_found = pearl::check_pow_target(
        transcript_rmem, pow_target, pow_key);
    if (block_found) {
      auto block_coord = cute::make_tuple(
          (int32_t)m_tile, (int32_t)n_tile, (int32_t)batch);
      auto problem_shape = cute::make_tuple(M, N, K, R);
      pearl::write_host_signal_header<Sm80TiledMma, HeaderTileShape_MNK>(
          host_signal_sync, host_signal_header_pinned,
          problem_shape, block_coord, tid, pow_target);
    }
  } else {
    if (pow_target != nullptr && pow_key != nullptr &&
        host_signal_sync != nullptr && host_signal_header_pinned != nullptr) {
      Tensor transcript_rmem = make_tensor<uint32_t>(
          Int<kTranscriptSlots>{});
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < kTranscriptSlots; ++s) {
        transcript_rmem(s) = transcript_local[s];
      }

      bool block_found = pearl::check_pow_target(
          transcript_rmem, pow_target, pow_key);
      if (block_found) {
        auto block_coord = cute::make_tuple(
            (int32_t)m_tile, (int32_t)n_tile, (int32_t)batch);
        auto problem_shape = cute::make_tuple(M, N, K, R);
        pearl::write_host_signal_header<Sm80TiledMma, HeaderTileShape_MNK>(
            host_signal_sync, host_signal_header_pinned,
            problem_shape, block_coord, tid, pow_target);
      }
    }
  }

  // ── Write final transcript to gmem ──────────────────────────────────
  // Layout matches transcript_kernel.cu's transcript_snapshot_kernel:
  //   base = ((batch * num_m_tiles + m_tile) * num_n_tiles + n_tile)
  //          * (kProofThreads * kTranscriptSlots)
  //   tx_idx = base + tid * kTranscriptSlots + slot
  if constexpr (!kHeadless) {
    if (transcript != nullptr) {
      int64_t base = ((int64_t)batch * num_m_tiles + m_tile)
                     * num_n_tiles + n_tile;
      int64_t tx_off = base * (int64_t)kProofThreads * kTranscriptSlots
                       + (int64_t)tid * kTranscriptSlots;
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < kTranscriptSlots; ++s) {
        transcript[tx_off + s] = transcript_local[s];
      }
    }
  }

  // ── Write final C tile to gmem (int32, M,N row-major) ──────────────
  // Each thread owns kFragSize=128 elements at coords tCcD(j).
  //
  // The pure-miner (PoW) path passes C_gmem==nullptr because the consumer
  // never reads C — the transcript is the only useful output and the C
  // store is M·N·int32 of pure waste per iter (1 GiB at the production
  // shape M=N=16384, plus the matching 1 GiB cudaMallocAsync/Free).
  // The reverse-engineered competing miner skips it; so do we when we can.
  // The torch path (pearl_gemm_api.cpp) still passes a real C_running when
  // skip_c_store=false because the ATen denoise/scale epilogue reads it.
  if constexpr (!kHeadless) {
    if (C_gmem != nullptr) {
      int64_t c_base = (int64_t)batch * M * N
                       + (int64_t)m_tile * kBM * (int64_t)N
                       + (int64_t)n_tile * kBN;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < kFragSize; ++j) {
        int m = get<0>(tCcD(j));
        int n = get<1>(tCcD(j));
        C_gmem[c_base + (int64_t)m * N + n] = tCrC(j);
      }
    }
  }

}

// ─── Host launcher ──────────────────────────────────────────────────────────
//
// Runtime knob: PEARL_GEMM_CONSUMER_CARVEOUT (or legacy
// PEARL_GEMM_BLACKWELL_CARVEOUT)
//   - unset / "default"    → driver default (typically L1-favored)
//   - "max_l1"  / "maxl1"  → cudaSharedmemCarveoutMaxL1   (smem minimised)
//   - "max_shared"/"max_smem" → cudaSharedmemCarveoutMaxShared
//   - integer 0..100       → exact percent of unified L1+smem to give to smem
//
// The driver still has to satisfy this kernel's dynamic smem request, so these
// values are advisory; it picks the smallest carveout >= requested. Useful on
// sm_120 (RTX 5090, 256 KB unified) when checking whether L1/TEX capacity or
// smem residency is the limiting factor.
static int read_carveout_env() {
  const char* env = std::getenv("PEARL_GEMM_CONSUMER_CARVEOUT");
  if (!env || !*env) env = std::getenv("PEARL_GEMM_BLACKWELL_CARVEOUT");
  if (!env || !*env) return -1;  // sentinel: don't touch
  std::string v(env);
  for (auto& c : v) c = (char)std::tolower((unsigned char)c);
  if (v == "default") return -1;
  if (v == "max_l1" || v == "maxl1" || v == "l1") {
    return cudaSharedmemCarveoutMaxL1;
  }
  if (v == "max_shared" || v == "maxshared" || v == "max_smem" ||
      v == "shared" || v == "smem") {
    return cudaSharedmemCarveoutMaxShared;
  }
  // Try integer percent.
  try {
    int pct = std::stoi(v);
    if (pct >= 0 && pct <= 100) return pct;
  } catch (...) {}
  return -1;
}

#if !PEARL_CONSUMER_USE_TMA_EXPERIMENT
// Runtime knob: PEARL_GEMM_CONSUMER_CLUSTER_M (or legacy
// PEARL_GEMM_BLACKWELL_CLUSTER_M)
//   - unset / "default" -> use the conservative tuned default below
//   - "0", "1", "off"  -> disable thread-block clustering
//   - "2" or "4"       -> cluster adjacent M tiles when the grid divides
//
// This is intentionally runtime-tunable because the 5090 trade-off is not
// obvious: clustering can improve B-tile locality, but it can also constrain
// scheduling on a launch with many independent CTAs.
static int read_cluster_m_env() {
  const char* env = std::getenv("PEARL_GEMM_CONSUMER_CLUSTER_M");
  if (!env || !*env) env = std::getenv("PEARL_GEMM_BLACKWELL_CLUSTER_M");
  if (!env || !*env) return -1;  // sentinel: use default
  std::string v(env);
  for (auto& c : v) c = (char)std::tolower((unsigned char)c);
  if (v == "default") return -1;
  if (v == "off" || v == "false" || v == "none") return 1;
  try {
    int cluster_m = std::stoi(v);
    if (cluster_m == 0) return 1;
    if (cluster_m == 1 || cluster_m == 2 || cluster_m == 4) return cluster_m;
  } catch (...) {}
  return -1;
}
#endif

template <bool kHeadless>
static cudaError_t ensure_transcript_kernel_attrs(size_t smem_bytes) {
  static std::atomic<unsigned long long> attrs_set_mask{0};

  int dev = -1;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) {
    dev = -1;
  }
  int sm_major = 0;
  if (dev >= 0) {
    (void)cudaDeviceGetAttribute(&sm_major,
                                 cudaDevAttrComputeCapabilityMajor, dev);
  }

  const unsigned long long bit = dev >= 0 ? (1ull << dev) : 0ull;
  if (bit != 0ull &&
      (attrs_set_mask.load(std::memory_order_acquire) & bit) != 0ull) {
    return cudaSuccess;
  }

  if (smem_bytes > 48 * 1024) {
    cudaError_t err = cudaFuncSetAttribute(
        transcript_gemm_kernel_consumer<kHeadless>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes);
    if (err != cudaSuccess) return err;
  }
  int carveout = read_carveout_env();
  if (carveout >= 0) {
    cudaError_t err = cudaFuncSetAttribute(
        transcript_gemm_kernel_consumer<kHeadless>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        carveout);
    if (err != cudaSuccess) return err;
  }
  // Opt into non-portable cluster sizes on sm_120 so cudaLaunchKernelEx
  // with clusterDim={2,1,1} won't fail with cudaErrorInvalidValue on
  // consumer Blackwell where default policy can reject otherwise-valid
  // cluster requests.
  if (sm_major >= 12) {
    cudaError_t err = cudaFuncSetAttribute(
        transcript_gemm_kernel_consumer<kHeadless>,
        cudaFuncAttributeNonPortableClusterSizeAllowed,
        1);
    if (err != cudaSuccess) return err;
  }

  if (bit != 0ull) {
    attrs_set_mask.fetch_or(bit, std::memory_order_release);
  }
  return cudaSuccess;
}

cudaError_t launch_transcript_gemm(
    int8_t  const* A,
    int8_t  const* B,
    int32_t*       C,
    uint32_t*      transcript,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    cudaStream_t stream) {
  assert(M % kBM == 0);
  assert(N % kBN == 0);
  assert(K % kBK == 0);
  assert(R % kBK == 0);
  assert(K % R == 0);

  dim3 grid((unsigned)(M / kBM), (unsigned)(N / kBN), (unsigned)batch);
  dim3 block(kThreads);
  size_t smem_bytes = sizeof(SharedStorage);

  cudaError_t err = ensure_transcript_kernel_attrs</*kHeadless=*/false>(smem_bytes);
  if (err != cudaSuccess) return err;

#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
  const int Mi = (int)M, Ni = (int)N, Ki = (int)K;
  Tensor mA_tma = make_tensor(make_gmem_ptr(const_cast<int8_t*>(A)),
      make_layout(make_shape(Mi, Ki), make_stride(Ki, _1{})));
  Tensor mB_tma = make_tensor(make_gmem_ptr(const_cast<int8_t*>(B)),
      make_layout(make_shape(Ni, Ki), make_stride(Ki, _1{})));
  TmaA tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA_tma, SmemLayoutStageA{});
  TmaB tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB_tma, SmemLayoutStageB{});
#endif

  (void)cudaGetLastError();
#if !PEARL_CONSUMER_USE_TMA_EXPERIMENT
  // Optional thread-block clustering. This is a scheduling-only change (no
  // DSMEM sharing yet), so the kernel runs the same code paths and preserves
  // byte identity. RunPod 5090 profiling favored the default cluster_m=1.
  int dev = -1;
  cudaGetDevice(&dev);
  int sm_major = 0;
  if (dev >= 0) {
    cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, dev);
  }
  int cluster_m = read_cluster_m_env();
  if (cluster_m < 0) cluster_m = 1;
  bool use_cluster = (sm_major >= 12) &&
                     (cluster_m > 1) &&
                     ((grid.x % (unsigned)cluster_m) == 0);

  if (use_cluster) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = (unsigned)cluster_m;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    err = cudaLaunchKernelEx(&cfg, transcript_gemm_kernel_consumer<false>,
                             A, B, C, transcript,
                             (int)M, (int)N, (int)K, (int)R,
                             nullptr, nullptr, nullptr, nullptr);
    if (err != cudaSuccess) return err;
  } else
#endif
  {
#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
    transcript_gemm_kernel_consumer<false><<<grid, block, smem_bytes, stream>>>(
        tma_a, tma_b, A, B, C, transcript, (int)M, (int)N, (int)K, (int)R,
        nullptr, nullptr, nullptr, nullptr);
#else
    transcript_gemm_kernel_consumer<false><<<grid, block, smem_bytes, stream>>>(
        A, B, C, transcript, (int)M, (int)N, (int)K, (int)R,
        nullptr, nullptr, nullptr, nullptr);
#endif
  }
  return cudaGetLastError();
}

cudaError_t launch_transcript_gemm_headless(
    int8_t  const* A,
    int8_t  const* B,
    int32_t*       C,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    uint32_t const* pow_target, uint32_t const* pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    cudaStream_t stream) {
  assert(M % kBM == 0);
  assert(N % kBN == 0);
  assert(K % kBK == 0);
  assert(R % kBK == 0);
  assert(K % R == 0);

  dim3 grid((unsigned)(M / kBM), (unsigned)(N / kBN), (unsigned)batch);
  dim3 block(kThreads);
  size_t smem_bytes = sizeof(SharedStorage);

  cudaError_t err = ensure_transcript_kernel_attrs</*kHeadless=*/true>(smem_bytes);
  if (err != cudaSuccess) return err;

#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
  const int Mi = (int)M, Ni = (int)N, Ki = (int)K;
  Tensor mA_tma = make_tensor(make_gmem_ptr(const_cast<int8_t*>(A)),
      make_layout(make_shape(Mi, Ki), make_stride(Ki, _1{})));
  Tensor mB_tma = make_tensor(make_gmem_ptr(const_cast<int8_t*>(B)),
      make_layout(make_shape(Ni, Ki), make_stride(Ki, _1{})));
  TmaA tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA_tma, SmemLayoutStageA{});
  TmaB tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB_tma, SmemLayoutStageB{});
#endif

  (void)cudaGetLastError();
#if !PEARL_CONSUMER_USE_TMA_EXPERIMENT
  int dev = -1;
  cudaGetDevice(&dev);
  int sm_major = 0;
  if (dev >= 0) {
    cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, dev);
  }
  int cluster_m = read_cluster_m_env();
  if (cluster_m < 0) cluster_m = 1;
  bool use_cluster = (sm_major >= 12) &&
                     (cluster_m > 1) &&
                     ((grid.x % (unsigned)cluster_m) == 0);

  if (use_cluster) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = (unsigned)cluster_m;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    err = cudaLaunchKernelEx(&cfg, transcript_gemm_kernel_consumer<true>,
                             A, B, C, nullptr,
                             (int)M, (int)N, (int)K, (int)R,
                             pow_target, pow_key,
                             host_signal_sync, host_signal_header_pinned);
    if (err != cudaSuccess) return err;
  } else
#endif
  {
#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
    transcript_gemm_kernel_consumer<true><<<grid, block, smem_bytes, stream>>>(
        tma_a, tma_b, A, B, C, nullptr, (int)M, (int)N, (int)K, (int)R,
        pow_target, pow_key,
        host_signal_sync, host_signal_header_pinned);
#else
    transcript_gemm_kernel_consumer<true><<<grid, block, smem_bytes, stream>>>(
        A, B, C, nullptr, (int)M, (int)N, (int)K, (int)R,
        pow_target, pow_key,
        host_signal_sync, host_signal_header_pinned);
#endif
  }
  return cudaGetLastError();
}

}  // namespace consumer
}  // namespace pearl
