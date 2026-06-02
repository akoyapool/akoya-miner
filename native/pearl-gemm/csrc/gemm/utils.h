#pragma once

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>

template <typename T>
inline constexpr bool always_false_v = false;

#define PRINT_LAYOUT_FROM_THREAD(tensor, tid)                       \
  do {                                                              \
    if (threadIdx.x == tid && blockIdx.x == 0 && blockIdx.y == 0 && \
        blockIdx.z == 0) {                                          \
      print(#tensor ": ");                                          \
      print(tensor);                                                \
      print("\n");                                                  \
    }                                                               \
  } while (0)

#define PRINT_LAYOUT(tensor) PRINT_LAYOUT_FROM_THREAD(tensor, 0)

#define PRINT_TENSOR_FROM_THREAD(tensor, tid, ...)                  \
  do {                                                              \
    if (threadIdx.x == tid && blockIdx.x == 0 && blockIdx.y == 0 && \
        blockIdx.z == 0) {                                          \
      print(#tensor ": ");                                          \
      print(tensor);                                                \
      print("\n");                                                  \
      pearl::pretty_print_tensor(tensor, ##__VA_ARGS__);            \
    }                                                               \
  } while (0)

#define PRINT_TENSOR(tensor, ...) \
  PRINT_TENSOR_FROM_THREAD(tensor, 0, ##__VA_ARGS__)

namespace pearl {

using namespace cute;

template <typename T, size_t N>
CUTE_DEVICE void pretty_print_tensor(const cute::array<T, N>& arr,
                                     bool full = false) {
  constexpr int size = static_cast<int>(N);

  cute::print("array([");

  for (int i = 0; i < size; i++) {
    if (full || i <= 2 || i >= size - 2) {
      cute::print("  ");
      cute::print(arr[i]);
      if (i < size - 1) {
        cute::print(", ");
      }
    } else if (i == 3) {
      cute::print("..., ");
    }
  }

  cute::print("])\n");
}

template <typename Engine, typename Layout>
requires(Layout::rank == 1) CUTE_DEVICE
    void pretty_print_tensor(const cute::Tensor<Engine, Layout>& tensor,
                             bool full = false) {

  auto shape = tensor.shape();
  int size = cute::get<0>(shape);

  cute::print("tensor([");

  for (int i = 0; i < size; i++) {
    if (full || i <= 2 || i >= size - 2) {
      cute::print("  ");
      cute::print(tensor(i));
      if (i < size - 1) {
        cute::print(", ");
      }
    } else if (i == 3) {
      cute::print("..., ");
    }
  }

  cute::print("])\n");
}

template <typename Engine, typename Layout>
requires(Layout::rank == 2) CUTE_DEVICE
    void pretty_print_tensor(const cute::Tensor<Engine, Layout>& tensor,
                             bool full = false) {

  auto shape = tensor.shape();
  int rows = cute::get<0>(shape);
  int cols = cute::get<1>(shape);

  cute::print("tensor([[");

  for (int i = 0; i < rows; i++) {
    if (i > 0) {
      if (full || i <= 2 || i >= rows - 2) {
        cute::print("\n        [");
      } else if (i == 3) {
        cute::print("\n        ...,\n        [");
      } else {
        continue;
      }
    }

    for (int j = 0; j < cols; j++) {
      if (full || j <= 2 || j >= cols - 2) {
        cute::print("  ");
        cute::print(tensor(i, j));
        if (j < cols - 1) {
          cute::print(", ");
        }
      } else if (j == 3) {
        cute::print("..., ");
      }
    }
    cute::print("]");
    if (i < rows - 1)
      cute::print(",");
  }

  cute::print("]])\n");
}

// Borrowed from FlashAttention repository:
// github.com/Dao-AILab/flash-attention/blob/main/hopper/utils.h
template <typename Engine, typename Layout, typename EngineOut>
CUTLASS_DEVICE void convert_type_out(Tensor<Engine, Layout> const& tensor,
                                     Tensor<EngineOut, Layout>& out) {
  // Somehow if we allocate out inside this function and return it, e2e is slower and the output can be wrong.
  using From_type = typename Engine::value_type;
  using To_type = typename EngineOut::value_type;
  static constexpr int FragmentSize = std::max(
      sizeof(From_type) / sizeof(To_type), sizeof(To_type) / sizeof(From_type));
  static_assert(CUTE_STATIC_V(size(tensor)) % FragmentSize == 0,
                "Fragment size does not vectorize properly");
  Tensor frag = recast<cutlass::Array<From_type, FragmentSize> const>(tensor);
  Tensor out_frg = recast<cutlass::Array<To_type, FragmentSize>>(out);
  static_assert(size(frag) == size(out_frg));
  cutlass::NumericArrayConverter<To_type, From_type, FragmentSize> convert_op;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(frag); ++i) {
    out_frg[i] = convert_op(frag[i]);
  }
}

template <class T>
CUTE_DEVICE T warp_prefix_sum(T val) {
  int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 1; i < cutlass::NumThreadsPerWarp; i <<= 1) {
    T partial_sum = __shfl_up_sync(0xffffffff, val, i);
    if (lane >= i) {
      val += partial_sum;
    }
  }
  return val;
}

template <class T>
CUTE_DEVICE T warp_uniform(T a) {
  return __shfl_sync(0xffffffff, a, 0);
}

CUTE_DEVICE int warp_idx_in_warpgroup_sync() {
  return __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
}

CUTE_DEVICE void block_wide_int8_copy(const int8_t* src, int8_t* dst, int size,
                                      int thread_idx, int num_threads) {
  // Assert alignment for vectorized access
  CUTLASS_ASSERT(reinterpret_cast<uintptr_t>(src) % sizeof(int4) == 0);
  CUTLASS_ASSERT(reinterpret_cast<uintptr_t>(dst) % sizeof(int4) == 0);

  int vector_copy_size = size / sizeof(int4);

  auto* src_vec = reinterpret_cast<const int4*>(src);
  auto* dst_vec = reinterpret_cast<int4*>(dst);

  // copy most data in a vectorized way
  for (int i = thread_idx; i < vector_copy_size; i += num_threads) {
    dst_vec[i] = src_vec[i];
  }

  int copied_size = vector_copy_size * sizeof(int4);

  // copy remaining data one by one
  for (int i = thread_idx + copied_size; i < size; i += num_threads) {
    dst[i] = src[i];
  }
}

template <int BYTES>
struct BytesToType {};

template <>
struct BytesToType<1> {
  using Type = uint8_t;
  static_assert(sizeof(Type) == 1);
};

template <>
struct BytesToType<2> {
  using Type = uint16_t;
  static_assert(sizeof(Type) == 2);
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
  static_assert(sizeof(Type) == 4);
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
  static_assert(sizeof(Type) == 8);
};

template <>
struct BytesToType<16> {
  using Type = cutlass::uint128_t;
  static_assert(sizeof(Type) == 16);
};

template <typename Fragment>
CUTLASS_DEVICE void permute_Aregs_fp8(Fragment& frag) {
  /*
  WGMMA results in int32 accumulator values being assigned to threads in each warp like this:
  row 0 | T0V0 T0V1 T1V0 T1V1 T2V0 T2V1 T3V0 T3V1 | T0V4 T0V5 T1V4 T1V5 T2V4 T2V5 T3V4 T3V5
  row 1 | T4V0 T4V1 T5V0 T5V1 T6V0 T6V1 T7V0 T7V1 | T8V4 T8V5 T9V4 T9V5 ...
  ...
  row 7 | T28V0 T28V1 ...
  ---------------------------------------------------------------------------              (1)
  row 8 | T0V2 T0V3 T1V2 T1V3 T2V2 T2V3 T3V2 T3V3 | T0V6 T0V7 T1V6 T1V7 T2V6 T2V7 T3V6 T3V7
  ...
  row 15 | T28V2 T28V3 ...

  Each thread has 8 values per 16x16 accumulator tile, in shape (2, 2). Across the 64 x tile_size_n
  MMA tile (split in the M direction between 4 warps), each thread has an accumulator tensor of
  shape ((4, 2, 2), 1, MMA_N). This is then downcast to int8.

  To efficiently load A from SMEM, we use the warp-wide ldmatrix instruction (in CUTLASS:
  SM75_U32x4_LDSM_N). We then store ApEA back to SMEM using stmatrix (CUTLASS: SM90_U32x4_STSM_N).
  Both instructions use the following thread-value assignment (for 8-bit values):

  row 0 | T0V0 T0V1 T0V2 T0V3 T1V0 T1V1 T1V2 T1V3 T2V0 T2V1 T2V2 T2V3 T3V0 T3V1 T3V2 T3V3
  row 1 | T4V0 T4V1 T4V2 T4V3 T5V0 T5V1 T5V2 T5V3 T6V0 T6V1 T6V2 T6V3 T7V0 T7V1 T7V2 T7V3
  ...
  row 7 | T28V0 T28V1 ...
  ---------------------------------------------------------------------------              (2)
  row 8 | T0V4 T0V5 T0V6 T0V7 T1V4 T1V5 T1V6 T1V7 T2V4 T2V5 T2V6 T2V7 T3V4 T3V5 T3V6 T3V7
  ...
  row 15 | T28V4 T28V5 ...

  For both (1) (downcast to int8) and (2), each set of 4 adjacent 8-bit values is packed into a
  32-bit register.

  Thus, to add the computed EA accumulator into A, we need to convert arrangement (1) into
  arrangement (2). Data only needs to be moved within each group of 4 adjacent threads (a "quad"),
  in two steps: a. Shuffle data within the quad so each thread holds the correct bytes. Per 16x16
  tile, each thread has 8 bytes which we split into a "lower" and "upper" uint32. We do 0L 1L 2L 3L
  | 0U 1U 2U 3U ==> 0L 0U 1L 1U | 2L 2U 3L 3U which can be processed as 2 shuffles per quad: 0U 1L
  2L 3U ==> 2L 0U 1L 3U, 0L 1U 2U 3L ==> 0L 2U 3L 1U

  b. Byte-permute data within each thread so that the bytes are in the correct order. For threads 0
  and 3, values are now in the order V0 V1 V4 V5 V2 V3 V6 V7 and can be untangled with two byte
  permutations. For threads 1 and 2, the two 4-byte registers are also in the wrong order, so the
  permutations have to be changed accordingly.
  */

  // frag has shape ((4, 2, 2), MMA_M, MMA_N), each element is 8 bits
  static_assert(sizeof(typename Fragment::value_type) == 1);

  int quad_idx = threadIdx.x % 4;
  bool lane_03 = quad_idx == 0 || quad_idx == 3;
  int selector_upper = lane_03 ? 0x5410 : 0x1054;
  int selector_lower = lane_03 ? 0x7632 : 0x3276;

  static constexpr int upper_map[4] = {0, 3, 1, 2};

  Tensor frag_64b = recast<uint2>(frag);  // ((1, 1, 2), MMA_M, MMA_N)
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(frag_64b); ++i) {
    uint32_t upper = frag_64b[i].x;
    uint32_t lower = frag_64b[i].y;
    uint32_t upper0 = lane_03 ? upper : lower;
    uint32_t lower0 = lane_03 ? lower : upper;
    upper0 = __shfl_sync(uint32_t(-1), upper0, upper_map[quad_idx], 4);
    lower0 = __shfl_sync(uint32_t(-1), lower0, upper_map[quad_idx] ^ 1, 4);
    frag_64b[i].x = __byte_perm(upper0, lower0, selector_upper);
    frag_64b[i].y = __byte_perm(upper0, lower0, selector_lower);
  }
}

}  // namespace pearl
