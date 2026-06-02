// Phase 2b.1 portable hash transcript launchers — host-callable interface.
// See transcript_kernel.cu for the kernels and full documentation.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Forward-declared from gemm/host_signal_header.hpp; including the full header
// here is fine but we keep the surface area minimal.
struct HostSignalSync;
struct HostSignalHeader;

namespace pearl {
namespace portable {

// Number of uint32 elements needed for the per-(tile, thread, slot)
// transcript scratch buffer.  Caller allocates this on-device, zeroes it
// before each noisy_gemm call, and passes the pointer to the snapshot/
// finalize launchers.
int64_t transcript_buffer_elems(int64_t M, int64_t N, int64_t batch);

// Launch the snapshot kernel for a single (s = 0..K/R-1) snapshot.
// C_running must hold the running int32 accumulator [batch, M, N] row-major.
void launch_transcript_snapshot(
    int32_t const* C_running,
    int64_t M, int64_t N, int64_t batch,
    uint32_t* transcript,
    int32_t snapshot_idx,
    cudaStream_t stream);

// V2: Fused int8 GEMM + transcript snapshot.  Replaces the V1 K/R partial
// cuBLAS path entirely.  A is (M, K) int8 row-major; B is (N, K) int8
// row-major (caller must transpose ApEA-style operands accordingly).
//
// C may be nullptr.  When non-null, the full int32 GEMM result (M, N)
// row-major is written to gmem (size M·N·int32 = 1 GiB at production
// shape).  When null, the C-store is skipped entirely — used by the
// pure-miner (PoW) path where only the transcript matters and writing C
// is dead work.
//
// transcript receives per-(m_tile, n_tile, batch, thread, slot) u32 in the
// same layout as launch_transcript_snapshot, ready for
// launch_transcript_finalize.  The kernel writes every slot
// unconditionally, so callers do not need to memset transcript first.
cudaError_t launch_transcript_gemm(
    int8_t  const* A,
    int8_t  const* B,
    int32_t*       C,
    uint32_t*      transcript,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    cudaStream_t stream);

// Launch the finalize kernel.  Reads the per-thread transcript, runs keyed
// BLAKE3, compares against pow_target.  On hit, atomic-CAS-locks
// host_signal_sync and writes a HostSignalHeader to host_signal_header_pinned.
void launch_transcript_finalize(
    uint32_t const* transcript,
    int64_t M, int64_t N, int64_t batch,
    uint32_t const* pow_target, uint32_t const* pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    int problem_m, int problem_n, int problem_k, int problem_r,
    cudaStream_t stream);

}  // namespace portable

namespace consumer {

// Dedicated consumer transcript GEMM used by Ampere, Ada, and consumer
// Blackwell builds. Arch traits select sm80/sm86, sm89, or sm120a behavior
// from the same source file.
cudaError_t launch_transcript_gemm(
    int8_t  const* A,
    int8_t  const* B,
    int32_t*       C,
    uint32_t*      transcript,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    cudaStream_t stream);

// Consumer mining fast path: finalizes the per-thread transcript in the GEMM
// kernel and writes HostSignalHeader directly on a hit. This avoids the
// transcript gmem spill/read and the separate finalize kernel.
cudaError_t launch_transcript_gemm_headless(
    int8_t  const* A,
    int8_t  const* B,
    int32_t*       C,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    uint32_t const* pow_target, uint32_t const* pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    cudaStream_t stream);

}  // namespace consumer

namespace legacy {

// Exact legacy mining backend for sm70/sm75. Uses DP4A while preserving the
// canonical SM80/Hopper transcript coordinate layout. Intended for Volta and
// as a safe Turing fallback until the sm75 tensor-core backend is tuned.
cudaError_t launch_transcript_gemm_dp4a_headless(
    int8_t  const* A,
    int8_t  const* B,
    int32_t*       C,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    uint32_t const* pow_target, uint32_t const* pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    cudaStream_t stream);

}  // namespace legacy

namespace turing {

// Exact Turing mining backend using SM75 int8 tensor cores. The 16x256
// one-warp tile preserves the protocol row/column proof pattern without the
// slow DP4A fallback.
cudaError_t launch_transcript_gemm(
    int8_t  const* A,
    int8_t  const* B,
    int32_t*       C,
    uint32_t*      transcript,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    cudaStream_t stream);

cudaError_t launch_transcript_gemm_headless(
    int8_t  const* A,
    int8_t  const* B,
    int32_t*       C,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    uint32_t const* pow_target, uint32_t const* pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    cudaStream_t stream);

}  // namespace turing
}  // namespace pearl
