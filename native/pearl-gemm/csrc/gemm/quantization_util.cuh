// most of this file is lifted directly from vLLM
//  * https://github.com/vllm-project/vllm/blob/main/csrc/quantization/vectorization.cuh
//  * https://github.com/vllm-project/vllm/blob/main/csrc/quantization/vectorization_utils.cuh
#pragma once
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>

namespace pearl {
namespace detail {

// Macro to apply scalar operation with correct arguments based on template params.
// This avoids repeating the if constexpr dispatch logic 4 times in vectorize_impl.
// Parameters are only evaluated when actually used due to if constexpr.
#define APPLY_SCALAR_OP(dst, src, aux) \
  do {                                 \
    if constexpr (kIsReadOnly) {       \
      if constexpr (kHasAux) {         \
        scalar_op(src, aux);           \
      } else {                         \
        scalar_op(src);                \
      }                                \
    } else {                           \
      if constexpr (kHasAux) {         \
        scalar_op(dst, src, aux);      \
      } else {                         \
        scalar_op(dst, src);           \
      }                                \
    }                                  \
  } while (0)

// Vectorized aligned vector type
template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t {
  scalar_t val[vec_size];
};

// Dummy type used when auxiliary data is not needed
struct NoAux {};

// Dummy type used for read-only operations (no output)
struct NoOut {};

// Unified vectorization implementation that handles:
// - Read-write and read-only operations (via OutT = void for read-only)
// - With or without auxiliary data (via AuxT = NoAux when not needed)
//
// Handles memory alignment by processing unaligned prefix/suffix with scalar
// operations and vectorizing the aligned middle section for optimal bandwidth.
//
// Template parameters:
// - VEC_SIZE: Number of elements per vector (must be power of 2)
// - InT: Input element type (e.g., float, half, int8_t)
// - OutT: Output element type (use void for read-only operations)
// - AuxT: Auxiliary data type (use NoAux when not needed)
// - ScaOp: Scalar operation functor type
//   - Without aux, read-only:  (const InT&) -> void
//   - Without aux, read-write: (OutT&, const InT&) -> void
//   - With aux, read-only:     (const InT&, const AuxT&) -> void
//   - With aux, read-write:    (OutT&, const InT&, const AuxT&) -> void
template <int VEC_SIZE, typename InT, typename OutT, typename AuxT,
          typename ScaOp>
CUTLASS_DEVICE void vectorize_impl(const InT* __restrict__ in,
                                   OutT* __restrict__ out,
                                   const AuxT* __restrict__ aux, int len,
                                   int tid, int stride, ScaOp&& scalar_op) {
  static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                "VEC_SIZE must be a positive power-of-two");

  constexpr bool kIsReadOnly = std::is_same_v<OutT, NoOut>;
  constexpr bool kHasAux = !std::is_same_v<AuxT, NoAux>;
  constexpr int kInWidth = VEC_SIZE * sizeof(InT);
  constexpr int kAuxWidth = kHasAux ? VEC_SIZE * sizeof(AuxT) : 0;

  using vin_t = vec_n_t<InT, VEC_SIZE>;

  uintptr_t in_addr = reinterpret_cast<uintptr_t>(in);

  // Check alignment for fast path
  bool can_vec =
      ((in_addr & (kInWidth - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);

  if constexpr (kHasAux) {
    can_vec =
        can_vec && ((reinterpret_cast<uintptr_t>(aux) & (kAuxWidth - 1)) == 0);
  }

  // === Fast path: everything aligned ===
  if (can_vec) {
    const int num_vec = len / VEC_SIZE;
    const vin_t* v_in = reinterpret_cast<const vin_t*>(in);

    for (int i = tid; i < num_vec; i += stride) {
      const vin_t src = v_in[i];

      [[maybe_unused]] vec_n_t<AuxT, VEC_SIZE> aux_vec;
      if constexpr (kHasAux) {
        using vaux_t = vec_n_t<AuxT, VEC_SIZE>;
        const vaux_t* v_aux = reinterpret_cast<const vaux_t*>(aux);
        aux_vec = v_aux[i];
      }

      [[maybe_unused]] vec_n_t<OutT, VEC_SIZE> dst;

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < VEC_SIZE; ++j) {
        APPLY_SCALAR_OP(dst.val[j], src.val[j], aux_vec.val[j]);
      }

      if constexpr (!kIsReadOnly) {
        using vout_t = vec_n_t<OutT, VEC_SIZE>;
        vout_t* v_out = reinterpret_cast<vout_t*>(out);
        v_out[i] = dst;
      }
    }
    return;
  }

  // === Slow path: handle unaligned prefix, vectorized middle, scalar tail ===

  // Calculate prefix elements needed to align input pointer
  const int prefix_elems =
      min(static_cast<int>(
              ((kInWidth - (in_addr & (kInWidth - 1))) & (kInWidth - 1)) /
              sizeof(InT)),
          len);

  // 1. Process unaligned prefix with scalar ops
  for (int i = tid; i < prefix_elems; i += stride) {
    APPLY_SCALAR_OP(out[i], in[i], aux[i]);
  }

  // Advance pointers past prefix
  const InT* in_aligned = in + prefix_elems;
  const int len_remaining = len - prefix_elems;

  // 2. Vectorize the aligned middle section
  const int num_vec = len_remaining / VEC_SIZE;
  const vin_t* v_in = reinterpret_cast<const vin_t*>(in_aligned);

  for (int i = tid; i < num_vec; i += stride) {
    const vin_t src = v_in[i];
    const int base_idx = prefix_elems + i * VEC_SIZE;

    [[maybe_unused]] AuxT aux_regs[VEC_SIZE];
    if constexpr (kHasAux) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < VEC_SIZE; ++j) {
        aux_regs[j] = aux[base_idx + j];
      }
    }

    [[maybe_unused]] vec_n_t<OutT, VEC_SIZE> dst;

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < VEC_SIZE; ++j) {
      APPLY_SCALAR_OP(dst.val[j], src.val[j], aux_regs[j]);
    }

    if constexpr (!kIsReadOnly) {
      using vout_t = vec_n_t<OutT, VEC_SIZE>;
      vout_t* v_out = reinterpret_cast<vout_t*>(out + prefix_elems);
      v_out[i] = dst;
    }
  }

  // 3. Handle tail elements with scalar ops
  const int tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len_remaining; i += stride) {
    const int global_idx = prefix_elems + i;
    APPLY_SCALAR_OP(out[global_idx], in[global_idx], aux[global_idx]);
  }
}

#undef APPLY_SCALAR_OP

}  // namespace detail

// ============================================================================
// Public API: Vectorization without auxiliary data
// ============================================================================

// Read-write vectorization with scalar operation.
// Template parameters:
// - VEC_SIZE: Number of elements per vector (must be power of 2)
// - InT: Input element type
// - OutT: Output element type
// - ScaOp: Scalar operation functor type (OutT&, const InT&) -> void
template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
CUTLASS_DEVICE void vectorize_with_alignment(const InT* __restrict__ in,
                                             OutT* __restrict__ out, int len,
                                             int tid, int stride,
                                             ScaOp&& scalar_op) {
  detail::vectorize_impl<VEC_SIZE, InT, OutT, detail::NoAux>(
      in, out, nullptr, len, tid, stride, std::forward<ScaOp>(scalar_op));
}

// Read-only vectorization with scalar operation.
// Template parameters:
// - VEC_SIZE: Number of elements per vector (must be power of 2)
// - InT: Input element type
// - ScaOp: Scalar operation functor type (const InT&) -> void
template <int VEC_SIZE, typename InT, typename ScaOp>
CUTLASS_DEVICE void vectorize_read_with_alignment(const InT* __restrict__ in,
                                                  int len, int tid, int stride,
                                                  ScaOp&& scalar_op) {
  detail::vectorize_impl<VEC_SIZE, InT, detail::NoOut, detail::NoAux>(
      in, nullptr, nullptr, len, tid, stride, std::forward<ScaOp>(scalar_op));
}

// ============================================================================
// Public API: Vectorization with auxiliary data from global memory
// ============================================================================

// Read-write vectorization with auxiliary data.
// Loads auxiliary values from global memory (relies on L1/L2 cache).
//
// Template parameters:
// - VEC_SIZE: Number of elements per vector (must be power of 2)
// - InT: Input element type
// - AuxT: Auxiliary data type (e.g., float for smooth_scale)
// - OutT: Output element type
// - ScaOp: Scalar operation functor (OutT&, const InT&, const AuxT&) -> void
template <int VEC_SIZE, typename InT, typename AuxT, typename OutT,
          typename ScaOp>
CUTLASS_DEVICE void vectorize_with_aux_global(const InT* __restrict__ in,
                                              const AuxT* __restrict__ aux,
                                              OutT* __restrict__ out, int len,
                                              int tid, int stride,
                                              ScaOp&& scalar_op) {
  detail::vectorize_impl<VEC_SIZE, InT, OutT, AuxT>(
      in, out, aux, len, tid, stride, std::forward<ScaOp>(scalar_op));
}

// Read-only vectorization with auxiliary data.
// Loads auxiliary values from global memory (relies on L1/L2 cache).
//
// Template parameters:
// - VEC_SIZE: Number of elements per vector (must be power of 2)
// - InT: Input element type
// - AuxT: Auxiliary data type (e.g., float for smooth_scale)
// - ScaOp: Scalar operation functor (const InT&, const AuxT&) -> void
template <int VEC_SIZE, typename InT, typename AuxT, typename ScaOp>
CUTLASS_DEVICE void vectorize_read_with_aux_global(const InT* __restrict__ in,
                                                   const AuxT* __restrict__ aux,
                                                   int len, int tid, int stride,
                                                   ScaOp&& scalar_op) {
  detail::vectorize_impl<VEC_SIZE, InT, detail::NoOut, AuxT>(
      in, nullptr, aux, len, tid, stride, std::forward<ScaOp>(scalar_op));
}

}  // namespace pearl
