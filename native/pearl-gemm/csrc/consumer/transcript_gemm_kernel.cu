// Consumer GPU fused int8 GEMM + transcript snapshot kernel.
//
// This source is compiled as the dedicated Ampere, Ada, and consumer
// Blackwell mining lane. It still uses the exact SM80 mma.sync m16n8k32 int8
// atom for proof-critical work. Architecture traits below choose the default
// tile/load policy while PEARL_CONSUMER_* compile-time defines remain the
// sweep interface.
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
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/mma_sm80.h>

#include "../blake3/blake3_constants.hpp"
#include "../gemm/pow_utils.hpp"

#include "../portable/transcript_kernel.cuh"

namespace pearl {
namespace consumer {

using namespace cute;

// ─── Architecture traits ────────────────────────────────────────────────────
#if defined(PEARL_GEMM_AMPERE)
#define PEARL_CONSUMER_DEFAULT_SWIZZLE_BITS 2
#define PEARL_CONSUMER_DEFAULT_STAGES 3
#define PEARL_CONSUMER_DEFAULT_KBLOCK 64
#define PEARL_CONSUMER_DEFAULT_MIN_BLOCKS 1
#elif defined(PEARL_GEMM_ADA)
#define PEARL_CONSUMER_DEFAULT_SWIZZLE_BITS 2
#define PEARL_CONSUMER_DEFAULT_STAGES 2
#define PEARL_CONSUMER_DEFAULT_KBLOCK 128
#define PEARL_CONSUMER_DEFAULT_MIN_BLOCKS 1
#elif defined(PEARL_GEMM_BLACKWELL)
#define PEARL_CONSUMER_DEFAULT_SWIZZLE_BITS 3
#define PEARL_CONSUMER_DEFAULT_STAGES 2
#define PEARL_CONSUMER_DEFAULT_KBLOCK 128
#define PEARL_CONSUMER_DEFAULT_MIN_BLOCKS 1
#else
#error "consumer transcript GEMM requires PEARL_GEMM_AMPERE, PEARL_GEMM_ADA, or PEARL_GEMM_BLACKWELL"
#endif

#ifndef PEARL_CONSUMER_USE_TMA_EXPERIMENT
#define PEARL_CONSUMER_USE_TMA_EXPERIMENT 0
#endif
#if PEARL_CONSUMER_USE_TMA_EXPERIMENT && !defined(PEARL_GEMM_BLACKWELL)
#error "PEARL_CONSUMER_USE_TMA_EXPERIMENT is Blackwell-only"
#endif
#if PEARL_CONSUMER_USE_TMA_EXPERIMENT
#error "Blackwell consumer TMA loader is scaffolded but not implemented; build the cp.async baseline or add the TMA mainloop before enabling this"
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
static constexpr int kThreads = 256;                  // 8 warps
static constexpr int kFragSize = (kBM * kBN) / kThreads; // per-thread acc slots
static_assert((kBM * kBN) % kThreads == 0,
              "CTA tile must divide evenly across 256 threads");
static constexpr int kTranscriptSlots = 16;           // = MSG_BLOCK_SIZE_U32

using ElementIn  = int8_t;
using ElementAcc = int32_t;

using TileShape_MNK = Shape<Int<kBM>, Int<kBN>, Int<kBK>>;
using HeaderTileShape_MNK = Shape<Int<kBM>, Int<kBN>, Int<128>>;

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

struct SharedStorage {
  alignas(16) ElementIn smem_A[cute::cosize_v<SmemLayoutA>];
  alignas(16) ElementIn smem_B[cute::cosize_v<SmemLayoutB>];
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
__launch_bounds__(kThreads, PEARL_CONSUMER_MIN_BLOCKS)
__global__ void transcript_gemm_kernel_consumer(
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

  const int num_m_tiles = M / kBM;
  const int num_n_tiles = N / kBN;

  // Per-thread accumulator (128 int32 in registers).
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

  // SMEM tensors for A and B with multi-stage shape.
  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B), SmemLayoutB{});

  // gmem tile views.
  // A_gmem: (M, K) → tile (kBM, kBK) at (m_tile, k_iter)
  Tensor mA = make_tensor(make_gmem_ptr(A_gmem),
                          make_shape(M, K),
                          make_stride(K, _1{}));
  Tensor mB = make_tensor(make_gmem_ptr(B_gmem),
                          make_shape(N, K),
                          make_stride(K, _1{}));
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

  // Prologue: issue first kStages-1 loads.
  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < K_TILES) issue_load(s, s);
  }

  for (int k_iter = 0; k_iter < K_TILES; ++k_iter) {
    int stg = k_iter % kStages;

    // Wait for the load of this iter's stage to land, sync all threads
    // (also a barrier between previous iter's MMA-reads-of-smem and the
    // upcoming prefetch into the same stage).  With kStages=3 prefetches
    // in flight, wait_group<1> drains 2 oldest groups, leaving the most
    // recent prefetch in flight — i.e. this iter's stage is ready.
    asm volatile("cp.async.wait_group %0;\n" :: "n"(kStages - 2));
    __syncthreads();

    // Issue the next prefetch (k_iter + kStages - 1).
    int next_k = k_iter + kStages - 1;
    if (next_k < K_TILES) {
      issue_load(next_k, next_k % kStages);
    } else {
      asm volatile("cp.async.commit_group;\n");
    }

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
      uint32_t hash = pearl::xor_reduction(tCrC);
      int snapshot_idx = (k_iter / reduce_every_k) - 1;
      int slot = snapshot_idx % kTranscriptSlots;
      transcript_local[slot] =
          pearl::rotl_xor<pearl::HASH_ACCUMULATE_ROTATION>(
              transcript_local[slot], hash);
    }

    // Bind register fragments to SMEM stage slice.
    Tensor sA_stg = sA(_, _, stg);     // (kBM, kBK)
    Tensor sB_stg = sB(_, _, stg);     // (kBN, kBK)
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

    // Issue all mma.sync ops for this k-iter.  cute::gemm dispatches to
    // the SM80_16x8x32_S32S8S8S32_TN atom for each (MMA_M, MMA_N) pair.
    // Reshape tCrC to the shape cute::gemm expects.
    auto tCrC_view = make_tensor(tCrC.data(), thr_mma.partition_fragment_C(
        make_tensor<ElementAcc>(Shape<Int<kBM>, Int<kBN>>{})).layout());
    gemm(tiled_mma, tCrA, tCrB, tCrC_view);

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
    uint32_t hash = pearl::xor_reduction(tCrC);
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

  // ── Write final transcript to gmem ──────────────────────────────────
  // Layout matches transcript_kernel.cu's transcript_snapshot_kernel:
  //   base = ((batch * num_m_tiles + m_tile) * num_n_tiles + n_tile)
  //          * (kThreads * kTranscriptSlots)
  //   tx_idx = base + tid * kTranscriptSlots + slot
  if (transcript != nullptr) {
    int64_t base = ((int64_t)batch * num_m_tiles + m_tile)
                   * num_n_tiles + n_tile;
    int64_t tx_off = base * (int64_t)kThreads * kTranscriptSlots
                     + (int64_t)tid * kTranscriptSlots;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < kTranscriptSlots; ++s) {
      transcript[tx_off + s] = transcript_local[s];
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

// Runtime knob: PEARL_GEMM_CONSUMER_CLUSTER_M (or legacy
// PEARL_GEMM_BLACKWELL_CLUSTER_M)
//   - unset / "default" → use the conservative tuned default below
//   - "0", "1", "off"  → disable thread-block clustering
//   - "2" or "4"       → cluster adjacent M tiles when the grid divides
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
        transcript_gemm_kernel_consumer,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes);
    if (err != cudaSuccess) return err;
  }
  int carveout = read_carveout_env();
  if (carveout >= 0) {
    cudaError_t err = cudaFuncSetAttribute(
        transcript_gemm_kernel_consumer,
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
        transcript_gemm_kernel_consumer,
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

  cudaError_t err = ensure_transcript_kernel_attrs(smem_bytes);
  if (err != cudaSuccess) return err;

  // Optional thread-block clustering. This is a scheduling-only change (no
  // DSMEM sharing yet), so the kernel runs the same code paths and preserves
  // byte identity. RunPod 5090 profiling favored the default cluster_m=1.
  int dev = -1; cudaGetDevice(&dev);
  int sm_major = 0;
  if (dev >= 0) cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, dev);
  int cluster_m = read_cluster_m_env();
  if (cluster_m < 0) cluster_m = 1;
  bool use_cluster = (sm_major >= 12) &&
                     (cluster_m > 1) &&
                     ((grid.x % (unsigned)cluster_m) == 0);

  (void)cudaGetLastError();
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
    err = cudaLaunchKernelEx(&cfg, transcript_gemm_kernel_consumer,
                             A, B, C, transcript,
                             (int)M, (int)N, (int)K, (int)R,
                             nullptr, nullptr, nullptr, nullptr);
    if (err != cudaSuccess) return err;
  } else {
    transcript_gemm_kernel_consumer<<<grid, block, smem_bytes, stream>>>(
        A, B, C, transcript, (int)M, (int)N, (int)K, (int)R,
        nullptr, nullptr, nullptr, nullptr);
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

  cudaError_t err = ensure_transcript_kernel_attrs(smem_bytes);
  if (err != cudaSuccess) return err;

  int dev = -1; cudaGetDevice(&dev);
  int sm_major = 0;
  if (dev >= 0) cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, dev);
  int cluster_m = read_cluster_m_env();
  if (cluster_m < 0) cluster_m = 1;
  bool use_cluster = (sm_major >= 12) &&
                     (cluster_m > 1) &&
                     ((grid.x % (unsigned)cluster_m) == 0);

  (void)cudaGetLastError();
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
    err = cudaLaunchKernelEx(&cfg, transcript_gemm_kernel_consumer,
                             A, B, C, nullptr,
                             (int)M, (int)N, (int)K, (int)R,
                             pow_target, pow_key,
                             host_signal_sync, host_signal_header_pinned);
    if (err != cudaSuccess) return err;
  } else {
    transcript_gemm_kernel_consumer<<<grid, block, smem_bytes, stream>>>(
        A, B, C, nullptr, (int)M, (int)N, (int)K, (int)R,
        pow_target, pow_key,
        host_signal_sync, host_signal_header_pinned);
  }
  return cudaGetLastError();
}

}  // namespace consumer
}  // namespace pearl
