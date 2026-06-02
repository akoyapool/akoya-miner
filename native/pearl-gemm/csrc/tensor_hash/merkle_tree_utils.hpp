#pragma once

#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"
#include "tensor_hash_constants.cuh"

#include <cutlass/cutlass.h>

namespace pearl {
namespace merkle_tree_utils {

using namespace cute;

// ceil(log2(n)) for n >= 1, computed entirely in uint32 integer space. Avoids
// floating-point log2 — which is imprecise near powers of two and, under the
// MSVC host compiler, resolves to a __host__-only overload that cannot be
// called from device code.
CUTLASS_DEVICE int ceil_log2_u32(uint32_t n) {
  return (n <= 1u) ? 0 : 32 - __clz(static_cast<int>(n - 1u));
}

using RmemLayoutChunk = Layout<Shape<Int<blake3::CHAINING_VALUE_SIZE_U32 * 2>>>;
using RmemLayoutChainingValue =
    Layout<Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>>>;

// Compile-time checks for layout sizes
static_assert(size(RmemLayoutChainingValue{}) ==
                  blake3::CHAINING_VALUE_SIZE_U32,
              "RmemLayoutChainingValue size is not correct");
static_assert(size(RmemLayoutChunk{}) == blake3::CHAINING_VALUE_SIZE_U32 * 2,
              "RmemLayoutChunk size is not correct");

// Compute a perfect Merkle Tree with num_leaves leaves
template <bool ConsiderRoot, class SmemTensorLeaves>
CUTLASS_DEVICE void compute_perfect_mt(SmemTensorLeaves const& sLeaves,
                                       const uint32_t num_leaves) {
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  for (u32 level_size = num_leaves; level_size > 1; level_size >>= 1) {
    Tensor rChainingValue = make_tensor<uint32_t>(RmemLayoutChainingValue{});
    Tensor rChunk = make_tensor<uint32_t>(RmemLayoutChunk{});

    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rChainingValue(i) = c_key[i];
    }

    const u32 num_pairs = level_size >> 1;
    if (tid < num_pairs) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
        rChunk(i) = sLeaves(i, 2 * tid);
        rChunk(i + blake3::CHAINING_VALUE_SIZE_U32) = sLeaves(i, 2 * tid + 1);
      }
    }
    // Thread tid reads from position 2*tid, but thread (tid+1) writes to position (tid+1).
    // When tid+1 == 2*tid (i.e., tid >= 2), there's a race without this barrier.
    __syncthreads();

    if (tid < num_pairs) {
      // Compress them
      blake3::CompressParams params =
          (ConsiderRoot && tid == 0 && num_pairs == 1)
              ? blake3::COMPRESS_PARAMS_ROOT
              : blake3::COMPRESS_PARAMS_INNER_NODE;
      blake3::compress_msg_block_u32(rChunk, rChainingValue, params);

      // And store the parent in the leaves tensor
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
        sLeaves(i, tid) = rChainingValue(i);
      }
    }
    __syncthreads();
  }
}

template <bool ConsiderRoot, class SmemTensorLeaves>
CUTLASS_DEVICE void compute_blake_mt(SmemTensorLeaves const& sLeaves,
                                     const uint32_t num_leaves) {
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  // Each group of threads handles a perfect tree
  // Figure out which group we are part of
  // Offset to read from in smem for this group
  int offset = 0;
  // Number of leaves in this thread's group
  int our_num_leaves = 0;
  // Virtual thread ID within this thread's group
  int virtual_tid = 0;
  int largest_subtree = 0;
  // Compute some of the offsets and indices
  for (int i = ceil_log2_u32(num_leaves); i >= 0; --i) {
    const u32 bit_value = 1u << i;
    if (num_leaves & bit_value) {
      if (largest_subtree == 0) {
        largest_subtree = bit_value;
      }
      if (offset + bit_value > 2 * tid) {
        our_num_leaves = bit_value;
        virtual_tid = tid - (offset / 2);
        break;
      } else {
        offset += bit_value;  // we are part of this tree
      }
    }
  }
  // Work on this thread group's MT
  for (int curr_num_leaves = largest_subtree; curr_num_leaves > 1;
       curr_num_leaves >>= 1) {
    Tensor rChunk = make_tensor<uint32_t>(RmemLayoutChunk{});
    Tensor rChainingValue = make_tensor<uint32_t>(RmemLayoutChainingValue{});

    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rChainingValue(i) = c_key[i];
    }

    const int num_pairs = curr_num_leaves >> 1;
    if (curr_num_leaves <= our_num_leaves && virtual_tid < num_pairs) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
        rChunk(i) = sLeaves(i, offset + 2 * virtual_tid);
        rChunk(i + blake3::CHAINING_VALUE_SIZE_U32) =
            sLeaves(i, offset + 2 * virtual_tid + 1);
      }
    }
    // Same race condition as in compute_perfect_mt.
    __syncthreads();

    if (curr_num_leaves <= our_num_leaves && virtual_tid < num_pairs) {
      // Hash their concatenation
      blake3::CompressParams params = blake3::COMPRESS_PARAMS_INNER_NODE;

      blake3::compress_msg_block_u32(rChunk, rChainingValue, params);
      // Copy back to the tree
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
        sLeaves(i, offset + virtual_tid) = rChainingValue(i);
      }
    }
    __syncthreads();
  }
  // Have thread 0 reduce all roots
  if (tid == 0) {
    Tensor rChainingValue = make_tensor<uint32_t>(RmemLayoutChainingValue{});
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rChainingValue(i) = c_key[i];
    }
    Tensor rChunk = make_tensor<uint32_t>(RmemLayoutChunk{});
    u32 read_offset = num_leaves;
    bool written_to_chunk = false;
    // Start by writing the rightmost root to the chunk; then, hash the current aggregated root with the next roots
    for (int i = 0; i < ceil_log2_u32(num_leaves); ++i) {
      const u32 bit_mask = 1u << i;
      if (read_offset & bit_mask) {
        if (!written_to_chunk) {
          read_offset -= bit_mask;
          for (u32 j = 0; j < blake3::CHAINING_VALUE_SIZE_U32; ++j) {
            rChunk(j + blake3::CHAINING_VALUE_SIZE_U32) =
                sLeaves(j, read_offset);
          }
          written_to_chunk = true;
        } else {
          read_offset -= bit_mask;
          for (u32 j = 0; j < blake3::CHAINING_VALUE_SIZE_U32; ++j) {
            rChunk(j) = sLeaves(j, read_offset);
          }

          // Reset chaining value to c_key before each hash
          for (u32 j = 0; j < blake3::CHAINING_VALUE_SIZE_U32; ++j) {
            rChainingValue(j) = c_key[j];
          }

          blake3::CompressParams params =
              (ConsiderRoot && read_offset == 0)
                  ? blake3::COMPRESS_PARAMS_ROOT
                  : blake3::COMPRESS_PARAMS_INNER_NODE;

          // Hash their concatenation
          blake3::compress_msg_block_u32(rChunk, rChainingValue, params);

          // Copy result to right half of chunk for next iteration
          for (u32 j = 0; j < blake3::CHAINING_VALUE_SIZE_U32; ++j) {
            rChunk(j + blake3::CHAINING_VALUE_SIZE_U32) = rChainingValue(j);
          }
        }
      }
    }
    // Copy the result back to smem
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      sLeaves(i, 0) = rChainingValue(i);
    }
  }
  __syncthreads();
}

}  // namespace merkle_tree_utils
}  // namespace pearl
