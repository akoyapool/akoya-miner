// Legacy exact transcript GEMM for sm70/sm75.
//
// This backend deliberately uses DP4A and the canonical transcript
// coordinate mapper rather than the sm80 cp.async/mma.sync path. It is slower
// than the tuned Ampere/Hopper lanes, but it preserves byte-identical PoW
// transcript semantics on Volta/Turing and gives those cards a valid mining
// path without changing the pool protocol.

#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>

#include <cute/container/array.hpp>
#include <cute/tensor.hpp>

#include "../blake3/blake3_constants.hpp"
#include "../gemm/host_signal_header.hpp"
#include "../gemm/pow_utils.hpp"

#include "transcript_canonical.cuh"
#include "transcript_kernel.cuh"

namespace pearl {
namespace legacy {

using namespace cute;

static constexpr int kBM = pearl::portable::kCanonicalTranscriptBM;
static constexpr int kBN = pearl::portable::kCanonicalTranscriptBN;
static constexpr int kThreads = pearl::portable::kCanonicalTranscriptThreads;
static constexpr int kTranscriptSlots =
    pearl::portable::kCanonicalTranscriptSlots;

using CanonicalTiledMma = pearl::portable::CanonicalTranscriptTiledMma;
using CanonicalTileShape = pearl::portable::CanonicalTranscriptTileShape;

static_assert(kThreads == 256, "canonical transcript mapping expects 256 threads");

__device__ __forceinline__ int dp4a_s32(int32_t a, int32_t b, int32_t c) {
  int32_t d;
  asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
               : "=r"(d)
               : "r"(a), "r"(b), "r"(c));
  return d;
}

__launch_bounds__(kThreads, 1)
__global__ void transcript_gemm_dp4a_headless_kernel(
    int8_t const* __restrict__ A,
    int8_t const* __restrict__ B,
    int32_t* __restrict__ C,
    int M, int N, int K, int R,
    uint32_t const* __restrict__ pow_target,
    uint32_t const* __restrict__ pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned) {
  const int m_tile = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int batch = blockIdx.z;
  const int tid = threadIdx.x;

  CanonicalTiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  Tensor cD = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
  Tensor tCcD = thr_mma.partition_C(cD);
  constexpr int kFragSize = decltype(size(tCcD))::value;
  static_assert(kFragSize > 0, "fragment size must be positive");
  static_assert(kFragSize <= MAX_NUM_REGISTERS_PER_THREAD,
                "fragment size exceeds HostSignalHeader capacity");

  cute::array<int32_t, kFragSize> acc;
  cute::array<uint32_t, kFragSize> frag;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kFragSize; ++j) {
    acc[j] = 0;
    frag[j] = 0;
  }

  cute::array<uint32_t, kTranscriptSlots> transcript_local;
  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < kTranscriptSlots; ++s) {
    transcript_local[s] = 0;
  }

  const int snapshots = K / R;
  const int64_t c_base = static_cast<int64_t>(batch) * M * N
      + static_cast<int64_t>(m_tile) * kBM * N
      + static_cast<int64_t>(n_tile) * kBN;

  for (int snapshot = 0; snapshot < snapshots; ++snapshot) {
    const int k_begin = snapshot * R;
    const int k_end = k_begin + R;
    for (int kk = k_begin; kk < k_end; kk += 4) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < kFragSize; ++j) {
        const int local_m = get<0>(tCcD(j));
        const int local_n = get<1>(tCcD(j));
        const int global_m = m_tile * kBM + local_m;
        const int global_n = n_tile * kBN + local_n;
        const int32_t a4 = reinterpret_cast<int32_t const*>(
            A + static_cast<int64_t>(global_m) * K)[kk >> 2];
        const int32_t b4 = reinterpret_cast<int32_t const*>(
            B + static_cast<int64_t>(global_n) * K)[kk >> 2];
        acc[j] = dp4a_s32(a4, b4, acc[j]);
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kFragSize; ++j) {
      frag[j] = static_cast<uint32_t>(acc[j]);
    }
    Tensor frag_t = make_tensor(frag.data(), Layout<Int<kFragSize>>{});
    const uint32_t hash = pearl::xor_reduction(frag_t);
    const int slot = snapshot % kTranscriptSlots;
    transcript_local[slot] =
        pearl::rotl_xor<pearl::HASH_ACCUMULATE_ROTATION>(
            transcript_local[slot], hash);
  }

  if (C != nullptr) {
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kFragSize; ++j) {
      const int local_m = get<0>(tCcD(j));
      const int local_n = get<1>(tCcD(j));
      C[c_base + static_cast<int64_t>(local_m) * N + local_n] = acc[j];
    }
  }

  if (pow_target == nullptr || pow_key == nullptr ||
      host_signal_sync == nullptr || host_signal_header_pinned == nullptr) {
    return;
  }

  Tensor transcript_t = make_tensor(
      transcript_local.data(), Layout<Int<kTranscriptSlots>>{});
  if (pearl::check_pow_target(transcript_t, pow_target, pow_key)) {
    auto block_coord = cute::make_tuple(
        static_cast<int32_t>(m_tile),
        static_cast<int32_t>(n_tile),
        static_cast<int32_t>(batch));
    auto problem_shape = cute::make_tuple(M, N, K, R);
    pearl::write_host_signal_header<CanonicalTiledMma, CanonicalTileShape>(
        host_signal_sync,
        host_signal_header_pinned,
        problem_shape,
        block_coord,
        tid,
        pow_target);
  }
}

cudaError_t launch_transcript_gemm_dp4a_headless(
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
  assert(K % 4 == 0);
  assert(R % 4 == 0);
  assert(K % R == 0);

  dim3 grid(
      static_cast<unsigned>(M / kBM),
      static_cast<unsigned>(N / kBN),
      static_cast<unsigned>(batch));
  dim3 block(static_cast<unsigned>(kThreads));
  (void)cudaGetLastError();
  transcript_gemm_dp4a_headless_kernel<<<grid, block, 0, stream>>>(
      A, B, C,
      static_cast<int>(M), static_cast<int>(N),
      static_cast<int>(K), static_cast<int>(R),
      pow_target, pow_key,
      host_signal_sync, host_signal_header_pinned);
  return cudaGetLastError();
}

}  // namespace legacy
}  // namespace pearl
