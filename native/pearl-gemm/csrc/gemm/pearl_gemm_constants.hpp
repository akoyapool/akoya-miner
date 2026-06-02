#pragma once

namespace pearl {
static constexpr int kAxEBLScaleFactor = 1 << 14;
static constexpr int kEARxBpEBScaleFactor = 1 << 12;
// Divide int tensors by a constant factor of 1 << 12 for fp16 tensor core MMA for denoising.
static constexpr int kIntToFp16ScaleFactor = 1 << 12;
// fp16 denoise factors AxEBL, EARxBpEB were already scaled in noising kernels.
// int32 denoise factors AxEBL, EARxBpEB were already converted and scaled by denoise conversion kernel.
// fp16 AxEBL was actually divided by (1 << 14); hence we multiply EBR by 4 = (1<<14) / (1<<12)
//  in this case to adjust everything to a common 1 << 12 scaling.
// Also multiply by -1 because we will subtract when doing denoising mma.
static constexpr int kEBRScaleFactorDenoise =
    -1 * kAxEBLScaleFactor / kIntToFp16ScaleFactor;
static constexpr int kEALScaleFactorDenoise =
    -1 * kEARxBpEBScaleFactor / kIntToFp16ScaleFactor;
}  // namespace pearl
