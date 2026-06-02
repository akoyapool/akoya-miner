// Torch-shim providing `torch::kFloat16` / `torch::kInt32` enum values.
// Used by csrc/gemm/static_switch_noisingA.h and static_switch_noisingB.h
// as compile-time tags for the noising dtype dispatch (fp16 vs split-K
// int32). akoya only ever takes the fp16 path, but the comparison in
// NOISING_A_CONFIG_OPTION still references both names so they must be
// declared.
//
// Resolved via -Icsrc/capi/stub_torch instead of the real libtorch
// torch/types.h header.

#pragma once

namespace torch {

enum class ScalarType : int {
  kHalf    = 0,  // alias for kFloat16
  kFloat16 = 0,
  kBFloat16 = 1,
  kFloat   = 2,
  kFloat32 = 2,
  kInt32   = 3,
  kInt8    = 4,
  kByte    = 5,
  kUInt32  = 6,
};

inline constexpr ScalarType kFloat16  = ScalarType::kFloat16;
inline constexpr ScalarType kHalf     = ScalarType::kHalf;
inline constexpr ScalarType kBFloat16 = ScalarType::kBFloat16;
inline constexpr ScalarType kFloat    = ScalarType::kFloat;
inline constexpr ScalarType kFloat32  = ScalarType::kFloat32;
inline constexpr ScalarType kInt32    = ScalarType::kInt32;
inline constexpr ScalarType kInt8     = ScalarType::kInt8;
inline constexpr ScalarType kByte     = ScalarType::kByte;
inline constexpr ScalarType kUInt32   = ScalarType::kUInt32;

}  // namespace torch

namespace at {
using torch::ScalarType;
}
