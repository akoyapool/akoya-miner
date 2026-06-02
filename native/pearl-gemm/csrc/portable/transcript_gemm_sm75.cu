// Turing SM75 fused int8 GEMM + headless transcript check.
//
// Turing's native int8 tensor-core atom is m8n8k16, whose 8-warp 128x256
// fragment layout is not byte-identical to the SM80/Hopper transcript layout.
// A one-warp 16x256 CTA, however, naturally maps each lane to the protocol
// proof pattern:
//   rows = {row, row + 8}
//   cols = {col, col + 1, col + 8, col + 9, ...}
// for row in 0..7 and col in {0,2,4,6}. This keeps shares verifiable without
// cross-thread regrouping while replacing the legacy DP4A hot path.

#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/tensor.hpp>

#include "../blake3/blake3_constants.hpp"
#include "../gemm/pow_utils.hpp"

#include "transcript_kernel.cuh"

namespace pearl {
namespace turing {

using namespace cute;

static constexpr int kBM = 16;
static constexpr int kBN = 256;
static constexpr int kAtomK = 16;
static constexpr int kBK = 64;
static constexpr int kThreads = 32;
static constexpr int kTranscriptSlots = blake3::MSG_BLOCK_SIZE_U32;

using ElementIn = int8_t;
using ElementAcc = int32_t;

using TileShape_MNK = Shape<Int<kBM>, Int<kBN>, Int<kBK>>;
using HeaderTileShape_MNK = Shape<Int<kBM>, Int<kBN>, Int<128>>;

using Sm75TiledMma = TiledMMA<
    MMA_Atom<SM75_8x8x16_S32S8S8S32_TN>,
    Layout<Shape<_1, _1, _1>>,
    Tile<Int<kBM>, Int<kBN>, Int<kAtomK>>>;

// SM75 ldmatrix.x1 consumes 8-row atoms. Keep the row-major K layout swizzled
// so the shared-memory reads match the tensor-core operand layout.
using SmemLayoutAtom = decltype(composition(
    Swizzle<2, 4, 3>{},
    Layout<Shape<_8, Int<kBK>>, Stride<Int<kBK>, _1>>{}));

using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtom{},
    make_shape(Int<kBM>{}, Int<kBK>{})));
using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtom{},
    make_shape(Int<kBN>{}, Int<kBK>{})));

struct SharedStorage {
  alignas(16) ElementIn smem_A[cute::cosize_v<SmemLayoutA>];
  alignas(16) ElementIn smem_B[cute::cosize_v<SmemLayoutB>];
};

__launch_bounds__(kThreads, 4)
__global__ void transcript_gemm_kernel_sm75(
    ElementIn const* __restrict__ A_gmem,
    ElementIn const* __restrict__ B_gmem,
    ElementAcc* __restrict__ C_gmem,
    uint32_t* __restrict__ transcript,
    int M, int N, int K, int R,
    uint32_t const* __restrict__ pow_target,
    uint32_t const* __restrict__ pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned) {

  extern __shared__ uint8_t smem_raw[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int m_tile = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int batch = blockIdx.z;
  const int tid = threadIdx.x;

  const int num_m_tiles = M / kBM;
  const int num_n_tiles = N / kBN;

  Sm75TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);

  Tensor cD = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
  Tensor tCcD = thr_mma.partition_C(cD);
  constexpr int kFragSize = decltype(size(tCcD))::value;
  static_assert(kFragSize == 128, "SM75 one-warp transcript fragment must be 128");
  static_assert(kFragSize <= MAX_NUM_REGISTERS_PER_THREAD,
                "fragment size exceeds HostSignalHeader capacity");

  Tensor tCrC = make_tensor<ElementAcc>(Shape<Int<kFragSize>>{});
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kFragSize; ++j) {
    tCrC(j) = 0;
  }

  uint32_t transcript_local[kTranscriptSlots];
  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < kTranscriptSlots; ++s) {
    transcript_local[s] = 0;
  }

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B), SmemLayoutB{});

  Tensor mA = make_tensor(make_gmem_ptr(A_gmem),
                          make_shape(M, K),
                          make_stride(K, _1{}));
  Tensor mB = make_tensor(make_gmem_ptr(B_gmem),
                          make_shape(N, K),
                          make_stride(K, _1{}));
  Tensor gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{},
                         make_coord(m_tile, _));
  Tensor gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{},
                         make_coord(n_tile, _));

  using GmemCopyAtom = Copy_Atom<UniversalCopy<cute::uint128_t>, ElementIn>;
  auto g2s_copy = make_tiled_copy(
      GmemCopyAtom{},
      Layout<Shape<_16, _2>, Stride<_2, _1>>{},
      Layout<Shape<_1, _16>>{});

  auto g2s_thr_copy = g2s_copy.get_slice(tid);
  Tensor tAgA = g2s_thr_copy.partition_S(gA);
  Tensor tAsA = g2s_thr_copy.partition_D(sA);
  Tensor tBgB = g2s_thr_copy.partition_S(gB);
  Tensor tBsB = g2s_thr_copy.partition_D(sB);

  const int K_TILES = K / kBK;
  const int reduce_every_k = R / kBK;

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
        make_tensor<ElementAcc>(Shape<Int<kBM>, Int<kBN>>{})).layout());
    gemm(tiled_mma, tCrA, tCrB, tCrC_view);
    __syncthreads();

    if (((k_iter + 1) % reduce_every_k) == 0) {
      uint32_t hash = pearl::xor_reduction(tCrC);
      int snapshot_idx = ((k_iter + 1) / reduce_every_k) - 1;
      int slot = snapshot_idx % kTranscriptSlots;
      transcript_local[slot] =
          pearl::rotl_xor<pearl::HASH_ACCUMULATE_ROTATION>(
              transcript_local[slot], hash);
    }
  }

  if (pow_target != nullptr && pow_key != nullptr &&
      host_signal_sync != nullptr && host_signal_header_pinned != nullptr) {
    Tensor transcript_rmem = make_tensor<uint32_t>(Int<kTranscriptSlots>{});
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < kTranscriptSlots; ++s) {
      transcript_rmem(s) = transcript_local[s];
    }

    if (pearl::check_pow_target(transcript_rmem, pow_target, pow_key)) {
      auto block_coord = cute::make_tuple(
          static_cast<int32_t>(m_tile),
          static_cast<int32_t>(n_tile),
          static_cast<int32_t>(batch));
      auto problem_shape = cute::make_tuple(M, N, K, R);
      pearl::write_host_signal_header<Sm75TiledMma, HeaderTileShape_MNK>(
          host_signal_sync, host_signal_header_pinned,
          problem_shape, block_coord, tid, pow_target);
    }
  }

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

cudaError_t launch_transcript_gemm(
    int8_t const* A,
    int8_t const* B,
    int32_t* C,
    uint32_t* transcript,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    cudaStream_t stream) {
  assert(M % kBM == 0);
  assert(N % kBN == 0);
  assert(K % kBK == 0);
  assert(R % kBK == 0);
  assert(K % R == 0);

  dim3 grid(static_cast<unsigned>(M / kBM),
            static_cast<unsigned>(N / kBN),
            static_cast<unsigned>(batch));
  dim3 block(kThreads);
  (void)cudaGetLastError();
  transcript_gemm_kernel_sm75<<<grid, block, sizeof(SharedStorage), stream>>>(
      A, B, C, transcript,
      static_cast<int>(M), static_cast<int>(N),
      static_cast<int>(K), static_cast<int>(R),
      nullptr, nullptr, nullptr, nullptr);
  return cudaGetLastError();
}

cudaError_t launch_transcript_gemm_headless(
    int8_t const* A,
    int8_t const* B,
    int32_t* C,
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

  dim3 grid(static_cast<unsigned>(M / kBM),
            static_cast<unsigned>(N / kBN),
            static_cast<unsigned>(batch));
  dim3 block(kThreads);
  (void)cudaGetLastError();
  transcript_gemm_kernel_sm75<<<grid, block, sizeof(SharedStorage), stream>>>(
      A, B, C, nullptr,
      static_cast<int>(M), static_cast<int>(N),
      static_cast<int>(K), static_cast<int>(R),
      pow_target, pow_key,
      host_signal_sync, host_signal_header_pinned);
  return cudaGetLastError();
}

}  // namespace turing
}  // namespace pearl
