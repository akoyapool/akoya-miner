// Canonical transcript coordinate mapping shared by legacy and portable
// backends. "Canonical" means the byte layout accepted by the pool verifier:
// the same per-thread accumulator coordinates used by the SM80/Hopper path.
#pragma once

#include <cstdint>

#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/mma_sm90.h>

namespace pearl {
namespace portable {

using namespace cute;

static constexpr int kCanonicalTranscriptBM = 128;
static constexpr int kCanonicalTranscriptBN = 256;
static constexpr int kCanonicalTranscriptBK = 128;
static constexpr int kCanonicalTranscriptSlots = 16;
static constexpr int kCanonicalTranscriptWarpgroups =
    kCanonicalTranscriptBM / 64;
static constexpr int kCanonicalTranscriptThreads =
    kCanonicalTranscriptWarpgroups * 128;

using CanonicalTranscriptElementIn = int8_t;
using CanonicalTranscriptElementAcc = int32_t;
using CanonicalTranscriptTileShape = Shape<
    Int<kCanonicalTranscriptBM>,
    Int<kCanonicalTranscriptBN>,
    Int<kCanonicalTranscriptBK>>;
using CanonicalTranscriptAtomLayout = Layout<
    Shape<Int<kCanonicalTranscriptWarpgroups>, _1, _1>>;

// Same pure-layout TiledMMA used by transcript_kernel.cu. It is only used to
// derive partition_C coordinates and to write HostSignalHeader metadata; it
// does not emit GMMA instructions in legacy kernels.
using CanonicalTranscriptTiledMma = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<
        CanonicalTranscriptElementIn,
        CanonicalTranscriptElementIn,
        CanonicalTranscriptElementAcc,
        CanonicalTranscriptTileShape>(),
    CanonicalTranscriptAtomLayout{}));

}  // namespace portable
}  // namespace pearl
