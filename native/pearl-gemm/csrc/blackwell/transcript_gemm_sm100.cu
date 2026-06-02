// Single-CTA Blackwell (sm_100a) INT8 GEMM + transcript snapshot, with a
// multi-stage TMA software pipeline.
//
// Each CTA computes kTilesPerCTA output tiles (two interleaved streams), each
//   C[128,256] (int32, row-major) = A[128,K] @ B[256,K]^T
// AND, every R=128 K-columns, snapshots the running int32 accumulator into a
// 4096-u32 "transcript" buffer, bit-identical to the reference fused kernel
// csrc/portable/transcript_gemm_kernel.cu.
//
// Transcript semantics (one tile, M=128 N=256 K=2048 R=128):
//   - 256 logical WGMMA "threads" tid=0..255, 16 slots each.
//   - 16 snapshots j=0..15.  Snapshot j is the running accumulator after the
//     first (j+1)*128 K-columns.
//   - transcript[tid*16 + j%16] = rotl_xor<13>(prev, hash_j) where hash_j is
//     the XOR of all C_j[m,n] (uint32 reinterpret) for (m,n) owned by tid:
//       tid = (m/64)*128 + ((m%64)/16)*32 + (m%8)*4 + (n%8)/2
//
// Design B (cumulative MMA).  The tcgen05.mma stream
// accumulates cumulatively into ONE TMEM accumulator, so after slab gs it
// holds C_gs directly and the readback IS the snapshot — no Crun, no fold.
// 256 consumer threads each own one (row, col-half) of the 128x256 tile; per
// slab they read C_gs back, XOR-reduce to 4 column-class partials, fold rows
// m,m^8 (within-warp shfl) and the two col-halves (smem exchange), and write
// the snapshot straight to the transcript.  9 warps / 288 threads: warp 8 is
// the TMA producer, warps 0-7 the consumers, warp 0 also issues the MMA.
// Full design: see the kernel comment below.
//
// This TU exports the host launcher pearl::portable::launch_transcript_gemm_sm100
// (built into libpearl_gemm_capi.so for PEARL_GEMM_ARCH=b200).  Defining
// PEARL_SM100_VERIFY_MAIN additionally compiles the standalone verify harness.
//
// Standalone verify build (from the pearl-gemm root):
//   nvcc -O3 -std=c++20 -arch=sm_100a --expt-relaxed-constexpr \
//     --expt-extended-lambda -DPEARL_SM100_VERIFY_MAIN \
//     -I third_party/cutlass/include -I third_party/cutlass/tools/util/include \
//     -I csrc -I csrc/gemm -I csrc/blake3 -I csrc/portable \
//     csrc/blackwell/transcript_gemm_sm100.cu \
//     csrc/portable/transcript_gemm_kernel.cu csrc/portable/transcript_kernel.cu \
//     -o /tmp/verify_transcript_sm100 -lcuda
//   /tmp/verify_transcript_sm100

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <atomic>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/collective/builders/sm100_common.inl>

#include "../gemm/pow_utils.hpp"
#include "../portable/transcript_kernel.cuh"
#include "transcript_gemm_sm100.h"

using namespace cute;

// ─── Tile constants ─────────────────────────────────────────────────────────
static constexpr int kBM = 128;   // output tile rows
static constexpr int kBN = 256;   // output tile cols
static constexpr int kBK = 128;   // smem K-tile == R: one K-tile = one R-slab
static constexpr int kHalfN = kBN / 2;   // 128 cols owned per consumer thread
// 9 warps: warps 0-7 are the math/snapshot consumers (256 threads); warp 8 is
// the dedicated TMA producer.  Each consumer owns one (row, col-half) of the
// accumulator, so its cumulative running sum is 128 int32 == register-resident
// (no smem Crun).  Warp specialization lets the producer run ahead.
static constexpr int kConsumerThreads = 256;
static constexpr int kThreads = 288;
static constexpr int kTranscriptSlots = 16;
// TMEM readback chunk == the whole half-row (un-chunked).  Both streams'
// readbacks are issued before a SINGLE fence_view_async_tmem_load, so the two
// tcgen05.ld latencies overlap — one fence per slab-step, not two.  ptxas
// fits the 2x128-int32 window with no spill.  (Chunking to 2x64 gives two
// fences/step and erases the win — measured 840 vs 877 TMADs/s.)
static constexpr int kRbChunk  = kHalfN;
static constexpr int kRbChunks = kHalfN / kRbChunk;   // 1
// A/B smem-pipeline depth.  2 stages; the freed 128 KB (old Crun) leaves
// room to deepen this further.
static constexpr int kStages = 2;
// Tiles processed per CTA.  The tcgen05.mma stream runs continuously across
// all kTilesPerCTA*K_TILES slabs — tile boundaries are pure consumer-side
// bookkeeping (flush transcript, reset Crun), so there is no MMA bubble
// between tiles.  Must divide the throughput proxy's 8192-tile count.
#ifndef PEARL_TILES_PER_CTA
#define PEARL_TILES_PER_CTA 8
#endif
static constexpr int kTilesPerCTA = PEARL_TILES_PER_CTA;
// Independent tile-streams processed concurrently per CTA.  Each stream owns
// one of the two TMEM accumulators and accumulates cumulatively in place (no
// copy between them — the tiles are independent).  The consumer interleaves
// the two streams' readbacks so one stream's ~420-cycle tcgen05.ld latency is
// hidden by the other stream's work.  kTilesPerCTA must be a multiple of it.
static constexpr int kStreams = 2;
// Output tiles each stream owns within a CTA.  CTA blockIdx.x owns the
// kTilesPerCTA output tiles [blockIdx.x*kTilesPerCTA, +kTilesPerCTA); stream
// st owns the st-th contiguous run of kStreamTiles within that.
static constexpr int kStreamTiles = kTilesPerCTA / kStreams;

using ElementIn  = int8_t;
using ElementAcc = int32_t;

// ─── UMMA atom + TiledMMA ───────────────────────────────────────────────────
using S8Mma = SM100_MMA_S8_SS<int8_t, int8_t, int32_t, kBM, kBN,
                              UMMA::Major::K, UMMA::Major::K>;
using TiledMma = decltype(make_tiled_mma(S8Mma{}));

// ─── SMEM layouts (UMMA-compatible swizzle) ─────────────────────────────────
using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::
    sm100_smem_selector<UMMA::Major::K, ElementIn,
                        Int<kBM>, Int<kBK>>());
using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::
    sm100_smem_selector<UMMA::Major::K, ElementIn,
                        Int<kBN>, Int<kBK>>());

using MmaShapeA_MK = decltype(partition_shape_A(
    TiledMma{}, make_shape(Int<kBM>{}, Int<kBK>{})));
using MmaShapeB_NK = decltype(partition_shape_B(
    TiledMma{}, make_shape(Int<kBN>{}, Int<kBK>{})));

// MMA-partitioned multi-stage smem layouts the UMMA descriptors consume:
//   ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K, PIPE) with PIPE = kStages.
using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
    SmemLayoutAtomA{}, append(MmaShapeA_MK{}, Int<kStages>{}), Step<_1,_2,_3>{}));
using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
    SmemLayoutAtomB{}, append(MmaShapeB_NK{}, Int<kStages>{}), Step<_1,_2,_3>{}));

// Flat single-stage (BLK_MN, BLK_K) swizzled view — the TMA box shape.
using SmemLayoutFlatA = decltype(tile_to_shape(
    SmemLayoutAtomA{}, make_shape(Int<kBM>{}, Int<kBK>{}), Step<_1,_2>{}));
using SmemLayoutFlatB = decltype(tile_to_shape(
    SmemLayoutAtomB{}, make_shape(Int<kBN>{}, Int<kBK>{}), Step<_1,_2>{}));
// Flat multi-stage (BLK_MN, BLK_K, PIPE) view of the same buffers — the TMA
// copy destination (one stage written per K-tile).
using SmemLayoutStagedA = decltype(tile_to_shape(
    SmemLayoutAtomA{}, make_shape(Int<kBM>{}, Int<kBK>{}, Int<kStages>{}),
    Step<_1,_2,_3>{}));
using SmemLayoutStagedB = decltype(tile_to_shape(
    SmemLayoutAtomB{}, make_shape(Int<kBN>{}, Int<kBK>{}, Int<kStages>{}),
    Step<_1,_2,_3>{}));

// ─── TMEM accumulator readback op ───────────────────────────────────────────
// One copy reads back a (kBM, kRbChunk) chunk of the int32 accumulator.
static constexpr int kNumColBits = kRbChunk * 32;
using TmemLoadOp = decltype(cute::TMEM::op_repeater<SM100_TMEM_LOAD_32dp32b1x,
                                                    kNumColBits>());

struct SharedStorage {
  // Per-stream A/B pipelines — the two streams are independent output tiles.
  alignas(1024) ElementIn smem_A[kStreams][cute::cosize_v<SmemLayoutA>];
  alignas(1024) ElementIn smem_B[kStreams][cute::cosize_v<SmemLayoutB>];
  // Per-stream mbarriers:
  //   full_barrier[st][stage] : stream st's stage-`stage` TMA load landed.
  //   mma_done[st][stage]     : the UMMA into that stage retired.
  //   acc_free[st]            : stream st's accumulator readback finished.
  alignas(16)   uint64_t  full_barrier[kStreams][kStages];
  alignas(16)   uint64_t  mma_done[kStreams][kStages];
  alignas(16)   uint64_t  acc_free[kStreams];
  alignas(16)   uint32_t  tmem_base_ptr;
};

// ─── TMA copy types ─────────────────────────────────────────────────────────
// The TMA box is one pipeline stage: a (kBM,kBK) tile of A, (kBN,kBK) of B.
// The gmem operands are sized at runtime; the make_tma_copy result *type* is
// fixed and depends only on a representative gmem layout.
using GmemLayout2D = decltype(make_layout(
    make_shape(int(0), int(0)), make_stride(int(0), _1{})));
using GmemTensor2D = decltype(make_tensor(
    make_gmem_ptr<ElementIn>(nullptr), GmemLayout2D{}));

using TmaA = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, GmemTensor2D{}, SmemLayoutFlatA{}));
using TmaB = decltype(make_tma_copy(
    SM90_TMA_LOAD{}, GmemTensor2D{}, SmemLayoutFlatB{}));

// TMA transaction bytes per stage (one A tile + one B tile, int8).
static constexpr uint32_t kTmaBytes =
    (uint32_t)(kBM * kBK + kBN * kBK) * (uint32_t)sizeof(ElementIn);

// ─── The kernel ─────────────────────────────────────────────────────────────
// Two-stream cumulative-MMA transcript GEMM.
//
// The tcgen05.mma stream accumulates *cumulatively* into a TMEM accumulator;
// after slab gs the accumulator holds C_gs directly, so the snapshot readback
// IS the cumulative state — no register-resident running sum, no fold.
//
// To hide the ~420-cycle tcgen05.ld readback latency, the CTA runs TWO
// independent tile-streams concurrently — one per TMEM accumulator (TMEM's
// 512 columns hold exactly two 128x256 int32 accumulators).  The consumer
// issues both streams' readbacks before a single fence, so the two ld
// latencies overlap.  The transcript half-pair combine is a fire-and-forget
// atomicXor (no barrier).  Multi-tile + continuous MMA stream.
__global__ __launch_bounds__(kThreads)
void transcript_gemm_sm100_kernel(
    __grid_constant__ const TmaA tma_a,    // full A (M,K) int8, row-major
    __grid_constant__ const TmaB tma_b,    // full B (N,K) int8, row-major
    uint32_t*         __restrict__ transcript,  // [(M/kBM)*(N/kBN)][256][16] u32
    int M, int N, int K) {

  extern __shared__ uint8_t smem_raw[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int tid     = threadIdx.x;
  const int warp_id = tid / 32;
  const bool is_producer = (warp_id == 8);

  // ── TMEM allocation + barrier init ────────────────────────────────────────
  cute::TMEM::Allocator1Sm tmem_alloc;
  if (warp_id == 0) {
    tmem_alloc.allocate(cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns,
                        &smem.tmem_base_ptr);
  }
  if (tid == 0) {
    CUTLASS_PRAGMA_UNROLL
    for (int st = 0; st < kStreams; ++st) {
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < kStages; ++s) {
        cutlass::arch::ClusterBarrier::init(&smem.full_barrier[st][s], 1);
        cutlass::arch::ClusterBarrier::init(&smem.mma_done[st][s], 1);
      }
      cutlass::arch::ClusterBarrier::init(&smem.acc_free[st], kConsumerThreads);
    }
  }
  __syncthreads();
  const uint32_t tmem_base = smem.tmem_base_ptr;

  // ── TiledMMA + two cumulative TMEM accumulators, one per stream ───────────
  // TMEM is 512 columns; each 128x256 int32 accumulator is 256 — exactly two
  // fit.  The streams are independent output tiles, each accumulating
  // cumulatively in place into its own buffer (nothing copied between them).
  TiledMma tiled_mma;
  Tensor acc0 = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
  Tensor acc1 = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
  acc0.data() = tmem_base;
  acc1.data() = tmem_base + kBN;

  // ── TMA gmem views — full A (M,K) and B (N,K) ─────────────────────────────
  // Each output tile slices its own (m_tile,kBK) block of A and (n_tile,kBK)
  // of B at load time.  gt = m_tile*nN + n_tile is the global output-tile
  // index; it also indexes the transcript directly (reference layout).
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
  const int nN      = N / kBN;
  const int K_TILES = K / kBK;
  // Slabs per stream: each stream processes kStreamTiles tiles.
  const int kGSlabsPerStream = K_TILES * kStreamTiles;
  auto cta_tma_a = tma_a.get_slice(_0{});
  auto cta_tma_b = tma_b.get_slice(_0{});

  // ── TMEM readback tooling ─────────────────────────────────────────────────
  // Consumer thread tid owns row cm of col-half ch.  The TiledCopy is the
  // same for both accumulators — only the source tensor (acc0/acc1) differs.
  const int cm = tid & 127;
  const int ch = (tid >> 7) & 1;
  Tensor tAcc_h0 = local_tile(acc0(make_coord(_, _), _0{}, _0{}),
                              Shape<Int<kBM>, Int<kHalfN>>{},
                              make_coord(_0{}, ch));
  Tensor tAcc_h1 = local_tile(acc1(make_coord(_, _), _0{}, _0{}),
                              Shape<Int<kBM>, Int<kHalfN>>{},
                              make_coord(_0{}, ch));
  Tensor tAcc_c0 = local_tile(tAcc_h0, Shape<Int<kBM>, Int<kRbChunk>>{},
                              make_coord(_0{}, _0{}));
  auto tiled_t2r = make_tmem_copy(TmemLoadOp{}, tAcc_c0);
  auto thr_t2r   = tiled_t2r.get_slice(cm);
  Tensor cD      = make_identity_tensor(Shape<Int<kBM>, Int<kRbChunk>>{});
  Tensor tDcD    = thr_t2r.partition_D(cD);
  Tensor tDrAcc0 = make_tensor<ElementAcc>(shape(tDcD));   // stream-0 readback
  Tensor tDrAcc1 = make_tensor<ElementAcc>(shape(tDcD));   // stream-1 readback

  // ── TMA producer ──────────────────────────────────────────────────────────
  // load_stage streams K-tile k_tile of output tile (m_tile,n_tile) into
  // stream st's smem pipeline stage `stage`.
  auto load_stage = [&](int st, int m_tile, int n_tile, int k_tile,
                        int stage) {
    uint64_t* fb = &smem.full_barrier[st][stage];
    if (cute::elect_one_sync()) {
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
          reinterpret_cast<
              cutlass::arch::ClusterTransactionBarrier::ValueType*>(fb),
          kTmaBytes);
      Tensor gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{},
                             make_coord(m_tile, _));   // (kBM,kBK,K_TILES)
      Tensor gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{},
                             make_coord(n_tile, _));   // (kBN,kBK,K_TILES)
      Tensor tAgA = cta_tma_a.partition_S(gA);
      Tensor tBgB = cta_tma_b.partition_S(gB);
      Tensor sA_flat = make_tensor(make_smem_ptr(smem.smem_A[st]),
                                   SmemLayoutStagedA{});
      Tensor sB_flat = make_tensor(make_smem_ptr(smem.smem_B[st]),
                                   SmemLayoutStagedB{});
      Tensor tAsA = cta_tma_a.partition_D(sA_flat);
      Tensor tBsB = cta_tma_b.partition_D(sB_flat);
      copy(tma_a.with(*fb), tAgA(_, _, _, k_tile), tAsA(_, _, _, stage));
      copy(tma_b.with(*fb), tBgB(_, _, _, k_tile), tBsB(_, _, _, stage));
    }
  };

  if (is_producer) {
    // ── Producer warp: stream A/B for both streams, every slab-step ─────────
    for (int gs = 0; gs < kGSlabsPerStream; ++gs) {
      const int stage = gs % kStages;
      const int use_j = gs / kStages;
      const int ti    = gs / K_TILES;     // tile-in-stream
      const int kt    = gs % K_TILES;     // K-tile within that tile
      CUTLASS_PRAGMA_UNROLL
      for (int st = 0; st < kStreams; ++st) {
        if (gs >= kStages)
          cutlass::arch::ClusterBarrier::wait(
              &smem.mma_done[st][stage], (use_j - 1) & 1);
        const int gt = blockIdx.x * kTilesPerCTA + st * kStreamTiles + ti;
        load_stage(st, gt / nN, gt % nN, kt, stage);
      }
    }
  } else {
    // ── Consumer warps 0-7 — two independent tile-streams, interleaved ──────
    // warp 0 issues stream st's cumulative MMA into acc0/acc1.
    auto issue_mma = [&](int st, int s) {
      if (warp_id != 0 || s >= kGSlabsPerStream) return;
      const int stage = s % kStages;
      cutlass::arch::ClusterBarrier::wait(&smem.full_barrier[st][stage],
                                          (s / kStages) & 1);
      if (s > 0)
        cutlass::arch::ClusterBarrier::wait(&smem.acc_free[st], (s - 1) & 1);
      Tensor sA = make_tensor(make_smem_ptr(smem.smem_A[st]), SmemLayoutA{});
      Tensor sB = make_tensor(make_smem_ptr(smem.smem_B[st]), SmemLayoutB{});
      Tensor tCrA = TiledMma::make_fragment_A(sA);
      Tensor tCrB = TiledMma::make_fragment_B(sB);
      // Cumulative: fresh accumulate on a tile's first slab, add otherwise.
      tiled_mma.accumulate_ = (s % K_TILES == 0) ? UMMA::ScaleOut::Zero
                                                 : UMMA::ScaleOut::One;
      CUTLASS_PRAGMA_UNROLL
      for (int kb = 0; kb < size<2>(tCrA); ++kb) {
        cute::gemm(tiled_mma, tCrA(_, _, kb, stage), tCrB(_, _, kb, stage),
                   st == 0 ? acc0 : acc1);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&smem.mma_done[st][stage]);
    };

    // Per-slot snapshot chain — only used when a tile has > kTranscriptSlots
    // snapshots (K > 2048), where each transcript slot is hit by several
    // snapshots and the reference folds them with a rotl_xor chain.  Dynamic
    // [slot] indexing puts these in local memory; the K<=2048 fast path never
    // touches them, so it pays only a larger (unused) stack frame.
    uint32_t chain0[kTranscriptSlots][4];
    uint32_t chain1[kTranscriptSlots][4];
    issue_mma(0, 0);
    issue_mma(1, 0);
    for (int gs = 0; gs < kGSlabsPerStream; ++gs) {
      const int stage = gs % kStages;
      const uint32_t mph = (gs / kStages) & 1;
      // Wait for C_gs in BOTH streams' accumulators.
      cutlass::arch::ClusterBarrier::wait(&smem.mma_done[0][stage], mph);
      cutlass::arch::ClusterBarrier::wait(&smem.mma_done[1][stage], mph);

      // Interleaved readback: per chunk, issue stream 0's AND stream 1's
      // tcgen05.ld, then ONE fence (it waits both) — two readbacks in flight,
      // so stream 1's ~420-cycle latency hides behind stream 0's and vice
      // versa.  Each stream XOR-reduces into its own 4 column-class partials.
      uint32_t part0[4] = {0, 0, 0, 0};
      uint32_t part1[4] = {0, 0, 0, 0};
      CUTLASS_PRAGMA_UNROLL
      for (int q = 0; q < kRbChunks; ++q) {
        Tensor q0 = local_tile(tAcc_h0, Shape<Int<kBM>, Int<kRbChunk>>{},
                               make_coord(_0{}, q));
        Tensor q1 = local_tile(tAcc_h1, Shape<Int<kBM>, Int<kRbChunk>>{},
                               make_coord(_0{}, q));
        copy(tiled_t2r, thr_t2r.partition_S(q0), tDrAcc0);
        copy(tiled_t2r, thr_t2r.partition_S(q1), tDrAcc1);
        cutlass::arch::fence_view_async_tmem_load();
        CUTLASS_PRAGMA_UNROLL
        for (int jj = 0; jj < kRbChunk; ++jj) {
          int j = q * kRbChunk + jj;
          part0[(j >> 1) & 3] ^= (uint32_t)tDrAcc0(jj);
          part1[(j >> 1) & 3] ^= (uint32_t)tDrAcc1(jj);
        }
      }
      // Both accumulators fully read — free them; issue both streams' next
      // MMA (each waits its own acc_free, which all consumers just arrived).
      cutlass::arch::ClusterBarrier::arrive(&smem.acc_free[0]);
      cutlass::arch::ClusterBarrier::arrive(&smem.acc_free[1]);
      issue_mma(0, gs + 1);
      issue_mma(1, gs + 1);

      // m-pair fold (within-warp shfl), both streams.
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < 4; ++c) {
        part0[c] ^= __shfl_xor_sync(0xffffffffu, part0[c], 8);
        part1[c] ^= __shfl_xor_sync(0xffffffffu, part1[c], 8);
      }

      // Write each stream's partials into its output tile's transcript.
      // tile_in is global output tile gt, which indexes the transcript
      // directly; snapshot j folds into slot j%16.  The cross-thread combine
      // (two col-halves) is a fire-and-forget atomicXor — no barrier; the
      // host pre-zeroes the buffer.
      if ((cm & 8) == 0) {
        const int j       = gs % K_TILES;            // snapshot within tile
        const int slot    = j & (kTranscriptSlots - 1);
        const int tile_in = gs / K_TILES;
        const int base    = (cm / 64) * 128 + ((cm % 64) / 16) * 32
                          + (cm % 8) * 4;
        const int cta_tile = blockIdx.x * kTilesPerCTA + tile_in;
        const int gt0 = cta_tile + 0 * kStreamTiles;
        const int gt1 = cta_tile + 1 * kStreamTiles;
        uint32_t* tr0 = transcript + (size_t)gt0 * 256 * kTranscriptSlots;
        uint32_t* tr1 = transcript + (size_t)gt1 * 256 * kTranscriptSlots;
        if (K_TILES <= kTranscriptSlots) {
          // <=16 snapshots — one per slot; write straight through.
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < 4; ++c) {
            atomicXor(&tr0[(base + c) * kTranscriptSlots + slot], part0[c]);
            atomicXor(&tr1[(base + c) * kTranscriptSlots + slot], part1[c]);
          }
        } else {
          // >16 snapshots: the reference folds each slot's snapshots with a
          // rotl_xor chain.  rotl distributes over XOR, so each thread chains
          // its OWN partial and the cross-thread atomicXor at the slot's last
          // snapshot still combines to the reference value.
          constexpr int kRot = pearl::HASH_ACCUMULATE_ROTATION;
          const bool first = (j < kTranscriptSlots);
          const bool last  = (j >= K_TILES - kTranscriptSlots);
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < 4; ++c) {
            chain0[slot][c] = first ? part0[c]
                : pearl::rotl_xor<kRot>(chain0[slot][c], part0[c]);
            chain1[slot][c] = first ? part1[c]
                : pearl::rotl_xor<kRot>(chain1[slot][c], part1[c]);
          }
          if (last) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < 4; ++c) {
              atomicXor(&tr0[(base + c) * kTranscriptSlots + slot],
                        chain0[slot][c]);
              atomicXor(&tr1[(base + c) * kTranscriptSlots + slot],
                        chain1[slot][c]);
            }
          }
        }
      }
    }
  }

  // ── Free TMEM ─────────────────────────────────────────────────────────────
  __syncthreads();
  if (warp_id == 0) {
    tmem_alloc.free(tmem_base,
                    cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns);
  }
}

// ─── Host launcher ──────────────────────────────────────────────────────────
// Drop-in for pearl::portable::launch_transcript_gemm, minus the C output
// (Design B emits no C).  Builds the TMA descriptors over the full A (M,K) /
// B (N,K) and launches the kernel over the whole output-tile grid.  The
// caller MUST pre-zero `transcript` — the kernel combines with atomicXor.
namespace pearl {
namespace portable {

cudaError_t launch_transcript_gemm_sm100(
    int8_t const* A, int8_t const* B, uint32_t* transcript,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    cudaStream_t stream) {
  // Supported regime — see transcript_gemm_sm100.h.  R must equal the snapshot
  // cadence kBK; K any multiple of kBK (K>2048 folds >16 snapshots per slot
  // via the kernel's rotl_xor chain).  asserts are no-ops under -DNDEBUG.
  assert(M % kBM == 0 && N % kBN == 0);
  assert(R == kBK && K % kBK == 0 && batch == 1);
  (void)R; (void)batch;

  const int smem_bytes = (int)sizeof(SharedStorage);
  static std::atomic<unsigned long long> attrs_set_mask{0};
  int dev = -1;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = -1;
  const unsigned long long bit = dev >= 0 ? (1ull << dev) : 0ull;
  if (bit == 0ull ||
      (attrs_set_mask.load(std::memory_order_acquire) & bit) == 0ull) {
    cudaError_t err = cudaFuncSetAttribute(
        transcript_gemm_sm100_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes);
    if (err != cudaSuccess) return err;
    if (bit != 0ull) {
      attrs_set_mask.fetch_or(bit, std::memory_order_release);
    }
  }

  const int Mi = (int)M, Ni = (int)N, Ki = (int)K;
  Tensor mA = make_tensor(make_gmem_ptr(const_cast<int8_t*>(A)),
      make_layout(make_shape(Mi, Ki), make_stride(Ki, _1{})));
  Tensor mB = make_tensor(make_gmem_ptr(const_cast<int8_t*>(B)),
      make_layout(make_shape(Ni, Ki), make_stride(Ki, _1{})));
  TmaA tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, SmemLayoutFlatA{});
  TmaB tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, SmemLayoutFlatB{});

  const int tiles = (Mi / kBM) * (Ni / kBN);
  (void)cudaGetLastError();
  transcript_gemm_sm100_kernel<<<tiles / kTilesPerCTA, kThreads,
                                 smem_bytes, stream>>>(
      tma_a, tma_b, transcript, Mi, Ni, Ki);
  return cudaGetLastError();
}

}  // namespace portable
}  // namespace pearl

// ─── Verify harness (standalone build only) ─────────────────────────────────
#ifdef PEARL_SM100_VERIFY_MAIN
static int8_t prng_i8(uint32_t& state) {
  state ^= state << 13; state ^= state >> 17; state ^= state << 5;
  return (int8_t)((int)(state % 127) - 63);
}

// Byte-identity sweep — `trials` fresh-random A/B, reference vs the sm100
// launcher, memcmp the whole multi-tile transcript.  Returns failing trials.
static int verify_shape(int M, int N, int K, int R, int trials) {
  const int batch = 1;
  const size_t a_elems = (size_t)M * K, b_elems = (size_t)N * K;
  const size_t c_elems = (size_t)M * N;
  const int64_t tr_elems =
      pearl::portable::transcript_buffer_elems(M, N, batch);

  int8_t* hA = (int8_t*)malloc(a_elems);
  int8_t* hB = (int8_t*)malloc(b_elems);
  int8_t *dA, *dB;
  int32_t* dC_ref;
  uint32_t *dTr_ref, *dTr_b200;
  cudaMalloc(&dA, a_elems);
  cudaMalloc(&dB, b_elems);
  cudaMalloc(&dC_ref, c_elems * sizeof(int32_t));
  cudaMalloc(&dTr_ref, tr_elems * sizeof(uint32_t));
  cudaMalloc(&dTr_b200, tr_elems * sizeof(uint32_t));
  uint32_t* hr = (uint32_t*)malloc(tr_elems * sizeof(uint32_t));
  uint32_t* hb = (uint32_t*)malloc(tr_elems * sizeof(uint32_t));

  int fails = 0;
  for (int t = 0; t < trials; ++t) {
    uint32_t s = 0xC0FFEEu + t * 2654435761u + (uint32_t)K * 2246822519u;
    for (size_t i = 0; i < a_elems; ++i) hA[i] = prng_i8(s);
    for (size_t i = 0; i < b_elems; ++i) hB[i] = prng_i8(s);
    cudaMemcpy(dA, hA, a_elems, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, b_elems, cudaMemcpyHostToDevice);
    cudaMemset(dTr_ref, 0, tr_elems * sizeof(uint32_t));
    cudaMemset(dTr_b200, 0, tr_elems * sizeof(uint32_t));
    pearl::portable::launch_transcript_gemm(dA, dB, dC_ref, dTr_ref,
                                            M, N, K, R, batch, 0);
    pearl::portable::launch_transcript_gemm_sm100(dA, dB, dTr_b200,
                                                  M, N, K, R, batch, 0);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("  KERNEL ERROR (K=%d trial %d): %s\n", K, t,
             cudaGetErrorString(err));
      fails = trials;
      break;
    }
    cudaMemcpy(hr, dTr_ref, tr_elems * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(hb, dTr_b200, tr_elems * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    if (memcmp(hr, hb, tr_elems * sizeof(uint32_t)) != 0) {
      if (fails == 0) {
        int shown = 0;
        for (int64_t i = 0; i < tr_elems && shown < 6; ++i)
          if (hr[i] != hb[i]) {
            int gt  = (int)(i / (256 * kTranscriptSlots));
            int rem = (int)(i % (256 * kTranscriptSlots));
            printf("    diff gt=%d tid=%d slot=%d: ref=0x%08x b200=0x%08x\n",
                   gt, rem / kTranscriptSlots, rem % kTranscriptSlots,
                   hr[i], hb[i]);
            shown++;
          }
      }
      fails++;
    }
  }
  free(hA); free(hB); free(hr); free(hb);
  cudaFree(dA); cudaFree(dB); cudaFree(dC_ref);
  cudaFree(dTr_ref); cudaFree(dTr_b200);
  return fails;
}

// Full-grid throughput at a production shape + a one-shot byte-identity check
// against the reference.  Returns non-zero on a transcript mismatch.
static int bench_shape(int M, int N, int K, int R) {
  const int batch = 1;
  const int tiles = (M / kBM) * (N / kBN);
  const size_t a_elems = (size_t)M * K, b_elems = (size_t)N * K;
  const size_t c_elems = (size_t)M * N;
  const int64_t tr_elems =
      pearl::portable::transcript_buffer_elems(M, N, batch);

  int8_t* hA = (int8_t*)malloc(a_elems);
  int8_t* hB = (int8_t*)malloc(b_elems);
  int8_t *dA, *dB;
  int32_t* dC_ref;
  uint32_t *dTr_ref, *dTr_b200;
  cudaMalloc(&dA, a_elems);
  cudaMalloc(&dB, b_elems);
  cudaMalloc(&dC_ref, c_elems * sizeof(int32_t));
  cudaMalloc(&dTr_ref, tr_elems * sizeof(uint32_t));
  cudaMalloc(&dTr_b200, tr_elems * sizeof(uint32_t));

  uint32_t s = 0x1234567u + (uint32_t)K;
  for (size_t i = 0; i < a_elems; ++i) hA[i] = prng_i8(s);
  for (size_t i = 0; i < b_elems; ++i) hB[i] = prng_i8(s);
  cudaMemcpy(dA, hA, a_elems, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, b_elems, cudaMemcpyHostToDevice);

  cudaMemset(dTr_ref, 0, tr_elems * sizeof(uint32_t));
  cudaMemset(dTr_b200, 0, tr_elems * sizeof(uint32_t));
  pearl::portable::launch_transcript_gemm(dA, dB, dC_ref, dTr_ref,
                                          M, N, K, R, batch, 0);
  pearl::portable::launch_transcript_gemm_sm100(dA, dB, dTr_b200,
                                                M, N, K, R, batch, 0);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("[bench K=%d] KERNEL ERROR: %s\n", K, cudaGetErrorString(err));
    return 1;
  }
  uint32_t* hr = (uint32_t*)malloc(tr_elems * sizeof(uint32_t));
  uint32_t* hb = (uint32_t*)malloc(tr_elems * sizeof(uint32_t));
  cudaMemcpy(hr, dTr_ref, tr_elems * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb, dTr_b200, tr_elems * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  int bad = (memcmp(hr, hb, tr_elems * sizeof(uint32_t)) != 0);
  free(hr); free(hb);

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  const int kReps = 20;
  cudaEventRecord(e0);
  for (int r = 0; r < kReps; ++r)
    pearl::portable::launch_transcript_gemm_sm100(dA, dB, dTr_b200,
                                                  M, N, K, R, batch, 0);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float gms = 0.f;
  cudaEventElapsedTime(&gms, e0, e1);
  double per_ms = gms / kReps;
  double tmads = (double)tiles * kBM * kBN * K / (per_ms / 1e3) / 1e12;
  printf("[bench K=%d]  %d tiles  %.3f ms/grid  =>  %7.1f TMADs/s   "
         "byte-identity %s\n", K, tiles, per_ms, tmads,
         bad ? "*** FAIL ***" : "PASS");

  free(hA); free(hB);
  cudaFree(dA); cudaFree(dB); cudaFree(dC_ref);
  cudaFree(dTr_ref); cudaFree(dTr_b200);
  return bad;
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Device: %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);
  int fail = 0;

  // ── Correctness — 512x2048 (32 tiles / 4 CTAs), K=2048 and K=4096 ─────────
  // Each tile reads its own (m_tile,n_tile) A/B block; 500 fresh-random trials
  // memcmp the whole multi-tile transcript against the reference.  K=4096
  // exercises the 32-snapshot rotl_xor fold into the 16 transcript slots.
  for (int K : {2048, 4096}) {
    int f = verify_shape(512, 2048, K, 128, 500);
    printf("[verify K=%d]  byte-identity sweep: %d/500 trials PASS%s\n",
           K, 500 - f, f ? "   *** FAIL ***" : "");
    if (f) fail = 1;
  }

  // ── Throughput — production 8192x32768, K=2048 vs K=4096 (same batch) ─────
  printf("\n");
  if (bench_shape(8192, 32768, 2048, 128)) fail = 1;
  if (bench_shape(8192, 32768, 4096, 128)) fail = 1;

  printf("\n%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
  return fail;
}
#endif  // PEARL_SM100_VERIFY_MAIN
