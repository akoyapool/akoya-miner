// Portable hash transcript + PoW check kernels.
//
// The H100 noisy_gemm kernel maintains a per-(m_tile, n_tile, thread)
// "transcript" of 16 uint32 slots in registers across the K-loop:
//
//   For each (m_tile, n_tile) tile:
//     transcript[0..15] = 0
//     C_running[bM, bN] : int32 = 0
//     For s in 0..K/R-1:
//       C_running += int8_GEMM(ApEA[m_tile, s*R:(s+1)*R],
//                              BpEB[n_tile, s*R:(s+1)*R].T)
//       hash_t = xor_reduction( per-thread fragment slots of C_running )
//       slot   = s mod 16
//       transcript[slot] = rotl_xor<13>(transcript[slot], hash_t)
//     hash256 = BLAKE3.compress(transcript, key=pow_key)
//     if hash256 <= pow_target: write_host_signal_header(...)
//
// Per-thread fragment slot ordering MUST match H100's WGMMA register layout
// byte-for-byte so the network accepts blocks mined on RTX 5090.  We get
// this for free by extracting the layout from the same KernelTraits::TiledMma
// type H100 uses, via partition_C(make_identity_tensor((bM, bN))).  This is
// the same TiledMma type the H100 mainloop instantiates, so by CUTE design
// `partition_C` returns the identical per-thread coord ordering that
// `partition_fragment_C` produces register-internally.
//
// For tiny (bM=128, bN=256, K=128, R=64): 2 snapshots, slots {0,1} active.
// For prod (bM=128, bN=256, K=4096, R=128): 32 snapshots, all 16 slots
//   receive exactly 2 rotl_xor mixings each.

#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/mma_sm90.h>

#include "../blake3/blake3.cuh"
#include "../blake3/blake3_constants.hpp"
#include "../gemm/host_signal_header.hpp"
#include "../gemm/pow_utils.hpp"

#include "transcript_canonical.cuh"
#include "transcript_kernel.cuh"

namespace pearl {
namespace portable {

using namespace cute;

// Compile-time tile parameters.  Both production-shipped tile shapes
// (R=64 and R=128) share these — see static_switch_matmul.h.
static constexpr int bM = kCanonicalTranscriptBM;
static constexpr int bN = kCanonicalTranscriptBN;
static constexpr int kNumMmaThreads = kCanonicalTranscriptThreads;

using ElementIn  = int8_t;
using ElementAcc = int32_t;

// IMPORTANT: this MUST match KernelTraits::TiledMma used in H100's
// collective_mainloop.hpp.  Both H100 and this portable kernel instantiate
// the same Atom (GMMA::ss_op_selector with the same template args) wrapped
// in the same AtomLayoutMNK, so partition_C produces identical per-thread
// coord orderings.  partition_C is pure CUTE layout math — it does NOT
// emit any wgmma SASS, so it compiles cleanly for sm_120a.
using TileShape_MNK = CanonicalTranscriptTileShape;
using PortableTiledMma = CanonicalTranscriptTiledMma;

// ─── Snapshot kernel ───────────────────────────────────────────────────────
// Grid:   (M/bM, N/bN, batch)
// Block:  kNumMmaThreads (256)
// Reads C_running[batch, M, N] (row-major, n stride = 1).
// For each (m_tile, n_tile, batch, thread):
//   - Gathers this thread's 128 int32 fragment slots from C_running using
//     partition_C(identity_tensor) on PortableTiledMma.
//   - XOR-reduces via pearl::xor_reduction (lop3 tree, identical to H100).
//   - rotl_xor<13> mixes into transcript[batch, m_tile, n_tile, thread, slot]
//     where slot = snapshot_idx mod 16.
__global__ void transcript_snapshot_kernel(
    int32_t const* __restrict__ C_running,
    int64_t M, int64_t N,
    uint32_t* __restrict__ transcript,
    int32_t snapshot_idx) {
  int m_tile = blockIdx.x;
  int n_tile = blockIdx.y;
  int batch  = blockIdx.z;
  int tid    = threadIdx.x;

  int64_t num_n_tiles = N / bN;
  int64_t num_m_tiles = M / bM;

  PortableTiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  Tensor cD   = make_identity_tensor(Shape<Int<bM>, Int<bN>>{});
  Tensor tCcD = thr_mma.partition_C(cD);
  constexpr int frag_size = decltype(size(tCcD))::value;  // 128
  static_assert(frag_size > 0, "frag_size must be positive");
  static_assert(frag_size <= MAX_NUM_REGISTERS_PER_THREAD,
                "frag_size exceeds HostSignalHeader capacity");

  // Gather per-thread fragment from C_running.
  cute::array<uint32_t, frag_size> frag;
  int64_t c_base = (int64_t)batch * M * N
                   + (int64_t)m_tile * bM * N
                   + (int64_t)n_tile * bN;

  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < frag_size; ++j) {
    int m = get<0>(tCcD(j));
    int n = get<1>(tCcD(j));
    int32_t v = C_running[c_base + (int64_t)m * N + (int64_t)n];
    frag[j] = static_cast<uint32_t>(v);
  }

  // Wrap in a CUTE tensor view so xor_reduction can compute its tree sizes.
  Tensor frag_t = make_tensor(frag.data(), Layout<Int<frag_size>>{});
  uint32_t hash = pearl::xor_reduction(frag_t);

  int slot = snapshot_idx % (int)blake3::MSG_BLOCK_SIZE_U32;
  int64_t per_tile_thread = (int64_t)kNumMmaThreads
                            * (int64_t)blake3::MSG_BLOCK_SIZE_U32;
  int64_t per_tile = (int64_t)blake3::MSG_BLOCK_SIZE_U32;
  int64_t base = ((int64_t)batch * num_m_tiles + m_tile) * num_n_tiles + n_tile;
  int64_t tx_idx = base * per_tile_thread + (int64_t)tid * per_tile + slot;

  uint32_t prev = transcript[tx_idx];
  transcript[tx_idx] = pearl::rotl_xor<pearl::HASH_ACCUMULATE_ROTATION>(
      prev, hash);
}

// ─── Finalize kernel ───────────────────────────────────────────────────────
// Grid:   (M/bM, N/bN, batch)
// Block:  kNumMmaThreads (256)
// Per (m_tile, n_tile, batch, thread):
//   - Loads its 16-u32 transcript from gmem into rmem.
//   - BLAKE3-compresses (single keyed block) with pow_key as initial chaining.
//   - Compares 256-bit hash <= pow_target (LE word order, MSW first).
//   - If hit: atomic-CAS lock on host_signal_sync, write a HostSignalHeader
//     with this thread's (tile_coord, partition_C coords, mma sizes, target).
__global__ void transcript_finalize_kernel(
    uint32_t const* __restrict__ transcript,
    int M, int N,
    uint32_t const* __restrict__ pow_target,
    uint32_t const* __restrict__ pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    int problem_m, int problem_n, int problem_k, int problem_r) {
  int m_tile = blockIdx.x;
  int n_tile = blockIdx.y;
  int batch  = blockIdx.z;
  int tid    = threadIdx.x;

  int64_t num_n_tiles = N / bN;
  int64_t num_m_tiles = M / bM;

  int64_t per_tile_thread = (int64_t)kNumMmaThreads
                            * (int64_t)blake3::MSG_BLOCK_SIZE_U32;
  int64_t per_tile = (int64_t)blake3::MSG_BLOCK_SIZE_U32;
  int64_t base = ((int64_t)batch * num_m_tiles + m_tile) * num_n_tiles
                 + n_tile;
  int64_t tx_idx = base * per_tile_thread + (int64_t)tid * per_tile;

  // Load transcript into a CUTE rmem tensor.
  Tensor transcript_rmem = make_tensor<uint32_t>(
      Int<blake3::MSG_BLOCK_SIZE_U32>{});
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < (int)blake3::MSG_BLOCK_SIZE_U32; ++i) {
    transcript_rmem(i) = transcript[tx_idx + i];
  }

  bool block_found = pearl::check_pow_target(
      transcript_rmem, pow_target, pow_key);

  if (block_found) {
    // Block coord = (m_tile, n_tile, batch).  Same convention as H100
    // tile_scheduler.
    auto block_coord = cute::make_tuple(
        (int32_t)m_tile, (int32_t)n_tile, (int32_t)batch);
    auto problem_shape = cute::make_tuple(
        problem_m, problem_n, problem_k, problem_r);
    pearl::write_host_signal_header<PortableTiledMma, TileShape_MNK>(
        host_signal_sync, host_signal_header_pinned,
        problem_shape, block_coord, tid, pow_target);
  }
}

// ─── Host launchers ────────────────────────────────────────────────────────

void launch_transcript_snapshot(
    int32_t const* C_running,
    int64_t M, int64_t N, int64_t batch,
    uint32_t* transcript,
    int32_t snapshot_idx,
    cudaStream_t stream) {
  assert(M % bM == 0);
  assert(N % bN == 0);
  dim3 grid((unsigned)(M / bM), (unsigned)(N / bN), (unsigned)batch);
  dim3 block((unsigned)kNumMmaThreads);
  transcript_snapshot_kernel<<<grid, block, 0, stream>>>(
      C_running, M, N, transcript, snapshot_idx);
}

void launch_transcript_finalize(
    uint32_t const* transcript,
    int64_t M, int64_t N, int64_t batch,
    uint32_t const* pow_target, uint32_t const* pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    int problem_m, int problem_n, int problem_k, int problem_r,
    cudaStream_t stream) {
  assert(M % bM == 0);
  assert(N % bN == 0);
  dim3 grid((unsigned)(M / bM), (unsigned)(N / bN), (unsigned)batch);
  dim3 block((unsigned)kNumMmaThreads);
  transcript_finalize_kernel<<<grid, block, 0, stream>>>(
      transcript, (int)M, (int)N,
      pow_target, pow_key,
      host_signal_sync, host_signal_header_pinned,
      problem_m, problem_n, problem_k, problem_r);
}

int64_t transcript_buffer_elems(int64_t M, int64_t N, int64_t batch) {
  assert(M % bM == 0);
  assert(N % bN == 0);
  int64_t num_m_tiles = M / bM;
  int64_t num_n_tiles = N / bN;
  return batch * num_m_tiles * num_n_tiles
         * (int64_t)kNumMmaThreads
         * (int64_t)blake3::MSG_BLOCK_SIZE_U32;
}

}  // namespace portable
}  // namespace pearl
