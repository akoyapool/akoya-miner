#include "blake3/blake3.cuh"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "merkle_tree_utils.hpp"
#include "tensor_hash_constants.cuh"

#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/detail/layout.hpp>

namespace pearl {

using namespace cute;

template <int kLeavesPerMTBlock, bool IsSingleBlock>
class ComputeBlakeMTKernel {
 public:
  using Element = uint32_t;
  static constexpr uint32_t MaxThreadsPerBlock = kLeavesPerMTBlock;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr int SharedStorageSize =
      kLeavesPerMTBlock * blake3::CHAINING_VALUE_SIZE;

  struct Arguments {
    Element* ptr_roots;  // pointer to GMEM
    const uint32_t
        num_blocks;  // how many blocks do we need to reduce in total?
  };

  struct Params {
    Element* ptr_roots;  // pointer to GMEM
    const uint32_t
        num_blocks;  // how many blocks do we need to reduce in total?
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return Params{args.ptr_roots, args.num_blocks};
  }

  static dim3 get_grid_shape(Params const& params) {
    return dim3((params.num_blocks + kLeavesPerMTBlock - 1) /
                kLeavesPerMTBlock);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock); }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    // Check how many leaves we need to process, and their offset.
    const u32 bid = blockIdx.x;
    const u32 tid = threadIdx.x;
    const u32 n_blocks = gridDim.x;
    const u32 remainder_block_size = params.num_blocks % kLeavesPerMTBlock;
    const bool is_remainder_block =
        (bid == n_blocks - 1) && (remainder_block_size > 0);
    const u32 num_leaves =
        is_remainder_block ? remainder_block_size : kLeavesPerMTBlock;
    const u32 offset = bid * kLeavesPerMTBlock;
    // Global is laid out as follows (each leaf's 8 words are stored contiguously):
    // [leaf0_w0][leaf0_w1]...[leaf0_w7][leaf1_w0][leaf1_w1]...[leaf1_w7]...
    // (8 words per leaf, each word is 4 bytes)
    Layout LeavesGmemLayout = make_layout(
        make_shape(Int<blake3::CHAINING_VALUE_SIZE_U32>{}, params.num_blocks),
        make_stride(Int<1>{}, Int<blake3::CHAINING_VALUE_SIZE_U32>{}));
    // SMEM is laid out as follows:
    // [0][1]...[kLeavesPerMTBlock - 1]
    // [0][1]...[kLeavesPerMTBlock - 1]
    // ...
    // [0][1]...[kLeavesPerMTBlock - 1]
    // (8 rows, each element is 4 bytes)
    // plus swizzle.
    Layout LeavesSmemLayout =
        make_layout(make_shape(Int<blake3::CHAINING_VALUE_SIZE_U32>{},
                               Int<kLeavesPerMTBlock>{}),
                    make_stride(Int<kLeavesPerMTBlock>{}, Int<1>{}));
    Tensor mLeaves = make_tensor(params.ptr_roots, LeavesGmemLayout);
    Tensor sLeaves = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(reinterpret_cast<uint32_t*>(smem_buf)),
                    LeavesSmemLayout));

    if (tid < num_leaves) {
      copy(mLeaves(_, offset + tid), sLeaves(_, tid));
    }
    __syncthreads();

    // Compute our Merkle Tree's root.
    // If we're a normal (=not remainder) block, we can use compute_perfect_mt.
    bool use_blake_mt =
        is_remainder_block && (num_leaves & (num_leaves - 1)) != 0;

    if (use_blake_mt) {
      if constexpr (IsSingleBlock) {
        merkle_tree_utils::compute_blake_mt<true>(sLeaves, num_leaves);
      } else {
        merkle_tree_utils::compute_blake_mt<false>(sLeaves, num_leaves);
      }
    } else {
      if constexpr (IsSingleBlock) {
        merkle_tree_utils::compute_perfect_mt<true>(sLeaves, num_leaves);
      } else {
        merkle_tree_utils::compute_perfect_mt<false>(sLeaves, num_leaves);
      }
    }
    __syncthreads();
    // Copy the root back from smem -> gmem.
    // Use the first warp to write back the 8 uint32_t values in parallel (coalesced)
    // Each block's 8 words are stored contiguously at offset bid * 8 words
    if (tid < blake3::CHAINING_VALUE_SIZE_U32) {
      const u32 base_offset =
          IsSingleBlock
              ? 0
              : bid * blake3::CHAINING_VALUE_SIZE_U32 * sizeof(uint32_t);

      uint8_t* roots_byte_ptr = reinterpret_cast<uint8_t*>(params.ptr_roots);
      u32* roots_u32_ptr = reinterpret_cast<u32*>(roots_byte_ptr + base_offset +
                                                  tid * sizeof(uint32_t));
      *roots_u32_ptr = sLeaves(tid, 0);
    }
  }
};
}  // namespace pearl