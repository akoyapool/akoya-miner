// Fused int8 GEMM + transcript snapshot kernel.
//
// A single CUDA kernel that computes the full int8 GEMM with mma.sync
// m16n8k32 (sm_80+) and snapshots the per-thread int32 fragment into the
// transcript every R k-cols, all from registers.  No gmem read-back of C
// between snapshots.
//
// Byte-identical to the H100 WGMMA path:
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

#include "transcript_kernel.cuh"

namespace pearl {
namespace portable {

using namespace cute;

// ─── Shape constants (must match transcript_kernel.cu) ───────────────────────
static constexpr int kBM = 128;
static constexpr int kBN = 256;
static constexpr int kAtomK = 32;                     // mma.sync m16n8k32 K
static constexpr int kBK = 64;                        // smem K-tile (= 2*kAtomK)
static constexpr int kThreads = 256;                  // 8 warps
static constexpr int kFragSize = 128;                 // per-thread acc slots
static constexpr int kTranscriptSlots = 16;           // = MSG_BLOCK_SIZE_U32

using ElementIn  = int8_t;
using ElementAcc = int32_t;

using TileShape_MNK = Shape<Int<kBM>, Int<kBN>, Int<kBK>>;

// Sm80 TiledMMA — verified byte-identical partition_C with WGMMA via
// probe_sm80_layout.cu.  The Tile<> argument's K dim is the MMA *atom* K
// (= 32), NOT the smem kBK.  partition_A/_B/partition_fragment_A/_B then
// produce MMA_K = kBK / kAtomK = 2 fragments per smem stage.
using Sm80TiledMma = TiledMMA<
    MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>,
    Layout<Shape<_8, _1, _1>>,
    Tile<Int<kBM>, Int<kBN>, Int<kAtomK>>>;

// ─── SMEM layout (Swizzle<2,4,3> for bank-conflict-free LDSM.x4) ─────────────
// kBK=64 bytes per row gives each ldmatrix.x4 lane-group access a clean
// bank stride.  Atom shape (16, 64) is the canonical CUTLASS sm_80 int8
// pattern (default_gemm_configuration.hpp); Swizzle<2,4,3> swaps bits
// {4,5} with {7,8} of the byte address, so consecutive matrix rows hit
// disjoint bank sets, eliminating the 4-way conflict the K-major layout
// would otherwise have.
//
// A: (kBM=128, kBK=64) int8 = 8 KiB per stage,  3 stages = 24 KiB
// B: (kBN=256, kBK=64) int8 = 16 KiB per stage, 3 stages = 48 KiB
// Total smem/block = 72 KiB.  Fits comfortably on sm_90 (228 KB/SM,
// 2 blocks/SM = 144 KB) and sm_120 (256 KB/SM, 3 blocks/SM = 216 KB).
// On sm_86/sm_89 (100 KB/SM) only 1 block fits; that's an accepted
// trade-off given the 5090 is the deployment target.
static constexpr int kStages = 3;

using SmemLayoutAtomA = decltype(composition(
    Swizzle<2, 4, 3>{},
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
// Occupancy tuning notes (target: sm_120 / RTX 5090, also benched on sm_90):
//   - 256 threads/block, 36 KiB smem/block (3 stages of A=4K + B=8K)
//   - sm_120 hard limits: 48 warps/SM, 1536 threads/SM, 65536 regs/SM, 256 KB smem/SM
//   - sm_90  hard limits: 64 warps/SM, 2048 threads/SM, 65536 regs/SM, 228 KB smem/SM
//   - Register pressure (~128 regs/thread natural) is the binding limit on both
//     archs.  __launch_bounds__(threads, minBlocks) tells ptxas how aggressively
//     to spill so that >=minBlocks fit per SM:
//        1 block   → ~244 regs/thread, 0 B spills, 12.5% (sm_90) / 16.7% (sm_120) occ
//        2 blocks  →  128 regs/thread, ~432 B spills, 25%   / 33% occ  (default)
//        3 blocks  →   80 regs/thread, ~1200 B spills, 37.5% / 50% occ
//        4 blocks  →  ≤64 regs/thread, severe spills, 50%   / 67% occ
//   - ncu on sm_90 at 2 blocks/SM showed the kernel L1/TEX-pipe-bound at 78%;
//     since spills also hit L1, more occupancy from spilling is a lose on H100
//     (-68% measured at 3 blocks).  May be different on sm_120 (RTX 5090) where
//     the L1/TC ratio differs — override with `-DPEARL_PORTABLE_MIN_BLOCKS=N`
//     via the `PEARL_GEMM_PORTABLE_MIN_BLOCKS` env var in setup.py.
#ifndef PEARL_PORTABLE_MIN_BLOCKS
#define PEARL_PORTABLE_MIN_BLOCKS 2
#endif
__launch_bounds__(kThreads, PEARL_PORTABLE_MIN_BLOCKS)
__global__ void transcript_gemm_kernel(
    ElementIn  const* __restrict__ A_gmem,    // (M, K) row-major
    ElementIn  const* __restrict__ B_gmem,    // (N, K) row-major
    ElementAcc*       __restrict__ C_gmem,    // (M, N) row-major int32 out
    uint32_t*         __restrict__ transcript,
    int M, int N, int K, int R) {

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
  const int reduce_every_k = R / kBK;       // R=128, kBK=64 → 2

  // ── gmem→smem TiledCopy via cp.async ─────────────────────────────────
  // 16-byte cp.async granule (uint128_t).  Thread layout (64,4) k-major
  // and value layout (1,16) k-major:  256 threads cooperatively load 64
  // rows × 64 cols (= 4 KiB) per "layer".  Per K-tile:
  //   A is 128×64 = 8 KiB → 2 layers per thread per K-tile (CPY=16, REST_M=2)
  //   B is 256×64 = 16 KiB → 4 layers per thread per K-tile (CPY=16, REST_M=4)
  // Routing through cute's TiledCopy ensures cp.async writes hit the same
  // swizzled smem addresses that LDSM reads from below — without that
  // consistency, the swizzle would corrupt the data.
  using GmemCopyAtom =
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, ElementIn>;
  auto g2s_copy_a = make_tiled_copy(
      GmemCopyAtom{},
      Layout<Shape<_64, _4>, Stride<_4, _1>>{},
      Layout<Shape<_1, _16>>{});
  auto g2s_copy_b = g2s_copy_a;  // same shape works (just larger M dim)

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
    // NOTE: smem layout is K-major with row stride = kBK = 32 bytes, which
    // forces a 4-way bank conflict on each ldmatrix.x4 (8 lanes per
    // bank-cycle land in the same 32 banks).  Even with conflicts this is
    // still fewer L1 transactions than the previous auto-vectorized
    // ld.shared.b32 path.  A future Swizzle<1,4,3> + row pad would eliminate
    // the conflict; tracked separately.
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
  // at compile time on the host (= 32 snapshots for K=4096, R=128, kBK=64).
  if ((K_TILES % reduce_every_k) == 0) {
    uint32_t hash = pearl::xor_reduction(tCrC);
    int snapshot_idx = (K_TILES / reduce_every_k) - 1;
    int slot = snapshot_idx % kTranscriptSlots;
    transcript_local[slot] =
        pearl::rotl_xor<pearl::HASH_ACCUMULATE_ROTATION>(
            transcript_local[slot], hash);
  }

  // ── Write final transcript to gmem ──────────────────────────────────
  // Layout matches transcript_kernel.cu's transcript_snapshot_kernel:
  //   base = ((batch * num_m_tiles + m_tile) * num_n_tiles + n_tile)
  //          * (kThreads * kTranscriptSlots)
  //   tx_idx = base + tid * kTranscriptSlots + slot
  int64_t base = ((int64_t)batch * num_m_tiles + m_tile)
                 * num_n_tiles + n_tile;
  int64_t tx_off = base * (int64_t)kThreads * kTranscriptSlots
                   + (int64_t)tid * kTranscriptSlots;
  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < kTranscriptSlots; ++s) {
    transcript[tx_off + s] = transcript_local[s];
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
// Runtime knob: PEARL_GEMM_PORTABLE_CARVEOUT
//   - unset / "default"    → driver default (typically L1-favored)
//   - "max_l1"  / "maxl1"  → cudaSharedmemCarveoutMaxL1   (smem minimised)
//   - "max_shared"/"max_smem" → cudaSharedmemCarveoutMaxShared
//   - integer 0..100       → exact percent of unified L1+smem to give to smem
//
// The driver still has to satisfy the 72 KB/block dynamic smem request, so
// these values are advisory; the driver picks the smallest carveout >=
// requested.  Useful on sm_120 (RTX 5090, 256 KB unified) where dropping
// from 228 KB carveout (~28 KB L1) to ~144 KB carveout (~92 KB L1) is a
// big swing for an L1/TEX-bound kernel.
static int read_carveout_env() {
  const char* env = std::getenv("PEARL_GEMM_PORTABLE_CARVEOUT");
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
        transcript_gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes);
    if (err != cudaSuccess) return err;
  }
  int carveout = read_carveout_env();
  if (carveout >= 0) {
    cudaError_t err = cudaFuncSetAttribute(
        transcript_gemm_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        carveout);
    if (err != cudaSuccess) return err;
  }
  // Opt into non-portable cluster sizes on sm_90+ so cudaLaunchKernelEx
  // with clusterDim={2,1,1} won't fail with cudaErrorInvalidValue where
  // default policy can reject otherwise-valid cluster requests.
  if (sm_major >= 9) {
    cudaError_t err = cudaFuncSetAttribute(
        transcript_gemm_kernel,
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

  // P6: on sm_90+ launch with a 2x1x1 thread-block cluster.  This is a
  // scheduling-only change (no DSMEM sharing yet) — the kernel runs the
  // same code paths so byte-identity is preserved.  Pairing the (bidx.x,
  // bidx.y) and (bidx.x+1, bidx.y) blocks into a cluster lets the GPC
  // scheduler co-locate them on the same GPC, which in practice raises
  // L2 reuse on the shared B-tile column.  Gated to grid.x % 2 == 0 so
  // the cluster size always divides the grid (otherwise the launch
  // would be rejected).
  int dev = -1; cudaGetDevice(&dev);
  int sm_major = 0;
  if (dev >= 0) cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, dev);
  bool use_cluster = (sm_major >= 9) && (grid.x % 2 == 0);

  (void)cudaGetLastError();
  if (use_cluster) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    err = cudaLaunchKernelEx(&cfg, transcript_gemm_kernel,
                             A, B, C, transcript,
                             (int)M, (int)N, (int)K, (int)R);
    if (err != cudaSuccess) return err;
  } else {
    transcript_gemm_kernel<<<grid, block, smem_bytes, stream>>>(
        A, B, C, transcript, (int)M, (int)N, (int)K, (int)R);
  }
  return cudaGetLastError();
}

}  // namespace portable
}  // namespace pearl
