// SM80-compatible MerkleTreeRootsKernel — portable replacement for the SM90
// TMA-based version.  Every thread loads its own 1024-byte chunk directly from
// global memory, runs BLAKE3 block compression in registers, then cooperates
// on the shared-memory Merkle-tree reduction via __syncthreads().
//
// No TMA, no warpgroup barriers, no PipelineTmaAsync — only SM80+ features.
#pragma once

#include "blake3/blake3.cuh"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "merkle_tree_utils.hpp"
#include "tensor_hash_constants.cuh"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_types.h>
#include <cutlass/detail/layout.hpp>
#include <cutlass/gemm/collective/builders/sm90_common.inl>  // ss_smem_selector (layout utility, portable)

namespace pearl {

using namespace cute;

// ---------------------------------------------------------------------------
// SM80 Merkle-tree roots kernel.
//
// Template parameters:
//   kNumThreads  – threads per block (= consumers; no dedicated producer
//                  warpgroup). Supported: 128, 256, 512.
// ---------------------------------------------------------------------------
template <int kNumThreads>
class MerkleTreeRootsKernelSM80 {
 public:
  using Element = uint8_t;

  static constexpr int kChunkSize = 1024;                          // bytes
  static constexpr int kWordSize  = 4;                             // bytes
  static constexpr int kNumBlocksPerChunk =
      kChunkSize / blake3::MSG_BLOCK_SIZE;                         // 16
  static constexpr int kNumWordsPerBlock =
      blake3::MSG_BLOCK_SIZE / sizeof(uint32_t);                   // 16
  static constexpr int kNumWordsPerChunk = kChunkSize / kWordSize; // 256

  static constexpr uint32_t MaxThreadsPerBlock = kNumThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // Register layouts (identical to SM90 version)
  using RmemLayoutChainingValue =
      Layout<Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>>>;
  using RmemLayoutBlock = Layout<Shape<Int<kNumWordsPerBlock>>>;

  // Shared-memory layout for the leaf chaining values.
  // [CHAINING_VALUE_SIZE_U32 (8)][kNumThreads] with bank-conflict-reducing swizzle.
  using SmemLayoutAtomLeaves = GMMA::Layout_K_SW128_Atom<uint32_t>;
  using SmemLayoutLeaves = decltype(tile_to_shape(
      SmemLayoutAtomLeaves{},
      Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>, Int<kNumThreads>>{}));

  static constexpr size_t AlignmentLeaves =
      cutlass::detail::alignment_for_swizzle(SmemLayoutLeaves{});

  struct SharedStorage : cute::aligned_struct<AlignmentLeaves> {
    cute::array_aligned<uint32_t,
                        cute::cosize_v<SmemLayoutLeaves>,
                        AlignmentLeaves>
        smem_leaves;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // ----- host-visible types --------------------------------------------------
  struct Arguments {
    const Element* ptr_data;
    const uint32_t data_len;  // bytes
    Element* ptr_roots;
    Element* ptr_leaf_cvs;
  };

  // See note in bseed_merkle_tree_roots_kernel_sm80.hpp: MSVC rejects
  // by-value parameters aligned above 16 bytes (C2719); cap on MSVC only.
#ifndef PEARL_PARAMS_ALIGN
#  if defined(_MSC_VER)
#    define PEARL_PARAMS_ALIGN 16
#  else
#    define PEARL_PARAMS_ALIGN 128
#  endif
#endif
  struct alignas(PEARL_PARAMS_ALIGN) Params {
    const Element* ptr_data;
    uint32_t data_len;
    Element* ptr_roots;
    Element* ptr_leaf_cvs;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {args.ptr_data, args.data_len, args.ptr_roots, args.ptr_leaf_cvs};
  }

  static dim3 get_grid_shape(Params const& params) {
    const size_t num_chunks =
        (params.data_len + kChunkSize - 1) / kChunkSize;
    return dim3((num_chunks + kNumThreads - 1) / kNumThreads);
  }

  static dim3 get_block_shape() { return dim3(kNumThreads); }

  // ----- device entry point --------------------------------------------------
  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage =
        *reinterpret_cast<SharedStorage*>(smem_buf);
    const int tid = threadIdx.x;

    // Smem tensor for leaf hashes
    Tensor sLeaves = as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(shared_storage.smem_leaves.data()), SmemLayoutLeaves{}));

    const size_t num_chunks =
        (params.data_len + kChunkSize - 1) / kChunkSize;
    const size_t num_grid_blocks =
        (num_chunks + kNumThreads - 1) / kNumThreads;

    // Output tensor
    Tensor mRoots = make_tensor(
        reinterpret_cast<uint32_t*>(params.ptr_roots),
        make_layout(
            make_shape(Int<blake3::CHAINING_VALUE_SIZE_U32>{}, num_grid_blocks),
            make_stride(Int<1>{}, Int<blake3::CHAINING_VALUE_SIZE_U32>{})));

    // ---- per-thread chunk hash --------------------------------------------
    const uint32_t global_chunk_idx = blockIdx.x * kNumThreads + tid;
    const bool has_chunk = (global_chunk_idx < num_chunks);

    // Initialise chaining value from the key stored in constant memory
    Tensor rChainingValue = make_tensor<uint32_t>(RmemLayoutChainingValue{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rChainingValue(i) = c_key[i];
    }

    if (has_chunk) {
      // Pointer to the start of this thread's chunk in global memory
      const uint32_t* chunk_ptr = reinterpret_cast<const uint32_t*>(
          params.ptr_data + static_cast<size_t>(global_chunk_idx) * kChunkSize);

      // How many valid bytes in this chunk?
      const uint32_t chunk_start_byte =
          static_cast<uint32_t>(global_chunk_idx) * kChunkSize;
      const uint32_t chunk_valid_bytes =
          (chunk_start_byte + kChunkSize <= params.data_len)
              ? static_cast<uint32_t>(kChunkSize)
              : (params.data_len - chunk_start_byte);

      // Process 16 blocks of 64 bytes each
      CUTLASS_PRAGMA_NO_UNROLL
      for (int blk = 0; blk < kNumBlocksPerChunk; ++blk) {
        Tensor rBlock = make_tensor<uint32_t>(RmemLayoutBlock{});

        const int word_base = blk * kNumWordsPerBlock;
        const uint32_t block_start_byte = blk * blake3::MSG_BLOCK_SIZE;

        if (block_start_byte + blake3::MSG_BLOCK_SIZE <= chunk_valid_bytes) {
          // Full block — vectorised 128-bit loads
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kNumWordsPerBlock / 4; ++i) {
            uint4 tmp = *reinterpret_cast<const uint4*>(
                chunk_ptr + word_base + i * 4);
            rBlock(i * 4 + 0) = tmp.x;
            rBlock(i * 4 + 1) = tmp.y;
            rBlock(i * 4 + 2) = tmp.z;
            rBlock(i * 4 + 3) = tmp.w;
          }
        } else {
          // Partial or fully-OOB block — load word-by-word with bounds check
          CUTLASS_PRAGMA_UNROLL
          for (int w = 0; w < kNumWordsPerBlock; ++w) {
            const uint32_t word_byte = block_start_byte + w * sizeof(uint32_t);
            if (word_byte + sizeof(uint32_t) <= chunk_valid_bytes) {
              rBlock(w) = chunk_ptr[word_base + w];
            } else if (word_byte < chunk_valid_bytes) {
              // Partial word — mask off OOB bytes
              uint32_t val = chunk_ptr[word_base + w];
              uint32_t valid = chunk_valid_bytes - word_byte;
              uint32_t mask = (1u << (valid * 8)) - 1;
              rBlock(w) = val & mask;
            } else {
              rBlock(w) = 0;
            }
          }
        }

        blake3::CompressParams cp{
            .counter   = global_chunk_idx,
            .block_len = blake3::MSG_BLOCK_SIZE,
            .flags     = blake3::KEYED_HASH};
        if (blk == 0)
          cp.flags |= blake3::CHUNK_START;
        if (blk == kNumBlocksPerChunk - 1)
          cp.flags |= blake3::CHUNK_END;

        blake3::compress_msg_block_u32(rBlock, rChainingValue, cp);
      }
    }

    // Write leaf hash to shared memory (inactive threads write the
    // initialisation value — harmless, overwritten by Merkle reduction or
    // simply ignored).
    if (has_chunk && params.ptr_leaf_cvs != nullptr) {
      uint32_t* leaf_cv_ptr =
          reinterpret_cast<uint32_t*>(params.ptr_leaf_cvs) +
          static_cast<size_t>(global_chunk_idx) * blake3::CHAINING_VALUE_SIZE_U32;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
        leaf_cv_ptr[i] = rChainingValue(i);
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      sLeaves(i, tid) = rChainingValue(i);
    }

    __syncthreads();

    // ---- Merkle-tree reduction (identical to SM90 path) -------------------
    const size_t bid = blockIdx.x;
    const bool is_last_block = (bid == num_grid_blocks - 1);

    const uint32_t num_leaves = [&]() -> uint32_t {
      if (!is_last_block)
        return static_cast<uint32_t>(kNumThreads);
      const uint32_t chunks_in_this_block = num_chunks % kNumThreads;
      const uint32_t actual =
          (chunks_in_this_block == 0) ? static_cast<uint32_t>(kNumThreads)
                                      : chunks_in_this_block;
      const uint32_t remainder_bytes = params.data_len % kChunkSize;
      const bool last_chunk_too_small =
          (remainder_bytes > 0) && (remainder_bytes < blake3::MSG_BLOCK_SIZE);
      return last_chunk_too_small ? (actual > 0 ? actual - 1 : 0) : actual;
    }();

    if (!is_last_block) {
      merkle_tree_utils::compute_perfect_mt<false>(sLeaves, kNumThreads);
    } else {
      if ((num_leaves & (num_leaves - 1)) == 0) {
        merkle_tree_utils::compute_perfect_mt<false>(sLeaves, num_leaves);
      } else {
        merkle_tree_utils::compute_blake_mt<false>(sLeaves, num_leaves);
      }
    }

    // Copy root to output (first 8 threads)
    if (tid < blake3::CHAINING_VALUE_SIZE_U32) {
      mRoots(tid, blockIdx.x) = sLeaves(tid, 0);
    }
  }
};

}  // namespace pearl
