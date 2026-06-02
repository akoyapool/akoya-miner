// Host launcher for the Blackwell (sm_100a) tcgen05/TMEM transcript GEMM.
// Implemented in transcript_gemm_sm100.cu; built into libpearl_gemm_capi.so
// for PEARL_GEMM_ARCH=b200.  See transcript_gemm_sm100.cu for the design.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pearl {
namespace portable {

// Blackwell drop-in for launch_transcript_gemm (minus the C output — the
// cumulative-MMA "Design B" kernel never materialises C).
//
//   A          : (M, K) int8, row-major  (ApEA)
//   B          : (N, K) int8, row-major  (BpEB)
//   transcript : per-(tile, thread, slot) u32, identical layout to
//                launch_transcript_gemm / transcript_buffer_elems().  The
//                caller MUST zero it before the call — the kernel combines
//                snapshots with atomicXor.
//
// Supported regime (enforced by assert; no-op under -DNDEBUG):
//   R == 128, K any multiple of 128, M % 128 == 0, N % 256 == 0, batch == 1.
//   K > 2048 yields > 16 snapshots, folded into the 16 transcript slots with
//   the reference's rotl_xor chain (production is K = 4096).
cudaError_t launch_transcript_gemm_sm100(
    int8_t  const* A,
    int8_t  const* B,
    uint32_t*      transcript,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    cudaStream_t stream);

}  // namespace portable
}  // namespace pearl
