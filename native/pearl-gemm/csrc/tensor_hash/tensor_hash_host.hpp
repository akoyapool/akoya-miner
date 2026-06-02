#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>
#include <cstring>

#include "blake3/blake3_constants.hpp"
#include "commitment_hash_from_merkle_roots_kernel.hpp"
#include "compute_blake_mt_kernel.hpp"
#include "gemm/error_check.hpp"
#include "merkle_tree_roots_kernel.hpp"
#include "reduce_roots_kernel.h"
#include "tensor_hash_constants.cuh"

#include "cute/tensor.hpp"

#include <cutlass/arch/arch.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"  // For device_kernel
#include "cutlass/kernel_hardware_info.h"

using namespace cute;

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

void set_key(const uint8_t d_key[blake3::KEY_SIZE], cudaStream_t stream) {
  // c_key is __constant__ memory bound to the σ-derived BLAKE3 keyed-hash
  // key. We unconditionally upload it — a 32-byte D2D constant-memory
  // copy is essentially free relative to the matmul/hash kernels that
  // follow it.
  //
  // No cache: a previous version of this function short-circuited when
  // d_key matched the last seen pointer, but that silently corrupted the
  // hash when callers (notably C# Akoya.Miner) reused the same device
  // buffer across σ rotations and just H2D'd new key bytes into it —
  // pointer matched, upload skipped, c_key stuck on the previous σ's key
  // → every subsequent tensor_hash digest was wrong.
  //
  // Async + stream-bound. The synchronous variant (no stream) implicitly
  // uses the legacy default stream and BLOCKS THE HOST until completion,
  // which serialises every caller's stream against the legacy default
  // stream. Async on the caller's stream is safe because c_key is
  // __constant__: subsequent kernel launches on the same stream observe
  // the new value (same-stream ordering).
  gpuErrchk(cudaMemcpyToSymbolAsync(c_key, d_key, blake3::KEY_SIZE, 0,
                                    cudaMemcpyDeviceToDevice, stream));
}

// kNumConsumerThreads: number of consumer threads for merkle_tree_roots_kernel
// kNumStages: pipeline stages for merkle_tree_roots_kernel
// kLeavesPerMTBlock: threads for compute_blake_mt_kernel
// kThreadLoadSize: bytes loaded per TMA operation (defaults to 128)
template <int kNumConsumerThreads, int kNumStages, int kLeavesPerMTBlock,
          int kThreadLoadSize = 128>
void tensor_hash_impl(const uint8_t* data, uint32_t data_size, uint8_t* out,
                       const uint8_t key[blake3::KEY_SIZE], uint32_t num_blocks,
                       uint8_t* roots, cudaDeviceProp& deviceProp,
                       cudaStream_t stream, bool upload_key = true,
                       uint8_t* leaf_cvs = nullptr) {
  if (upload_key) set_key(key, stream);
  const u32 data_len = data_size;

  using MerkleTreeRootsKernel =
      pearl::MerkleTreeRootsKernel<kNumConsumerThreads, kNumStages,
                                   kThreadLoadSize>;
  constexpr static int merkle_roots_smem_size =
      MerkleTreeRootsKernel::SharedStorageSize;
  typename MerkleTreeRootsKernel::Arguments args{
      data,
      data_len,
      reinterpret_cast<uint8_t*>(roots),
      leaf_cvs,
  };
  typename MerkleTreeRootsKernel::Params kernel_params =
      MerkleTreeRootsKernel::to_underlying_arguments(args);
  auto roots_kernel = cutlass::device_kernel<MerkleTreeRootsKernel>;
  dim3 grid = MerkleTreeRootsKernel::get_grid_shape(kernel_params);
  dim3 block = MerkleTreeRootsKernel::get_block_shape();
  if (merkle_roots_smem_size >= 48 * 1024) {
    gpuErrchk(cudaFuncSetAttribute(reinterpret_cast<const void*>(roots_kernel),
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   merkle_roots_smem_size));
  }
  roots_kernel<<<grid, block, merkle_roots_smem_size, stream>>>(kernel_params);
  gpuErrchk(cudaGetLastError());

  // 4. Compute MT as per BLAKE structure on global
  // We do this in two steps:
  // - Let each SM compute a Merkle Tree of leaves_per_mt_block leaves, where leaves_per_mt_block is a power of 2.
  // - Then, create a Merkle Tree in BLAKE3's structure over these roots.
  const int num_blocks_for_mt =
      (num_blocks + kLeavesPerMTBlock - 1) / kLeavesPerMTBlock;
  const bool is_single_block = (num_blocks_for_mt == 1);

  if (is_single_block) {
    using ComputeBlakeMTKernel =
        pearl::ComputeBlakeMTKernel<kLeavesPerMTBlock, true>;
    typename ComputeBlakeMTKernel::Arguments args2{
        reinterpret_cast<uint32_t*>(roots),
        num_blocks,
    };
    typename ComputeBlakeMTKernel::Params kernel_params2 =
        ComputeBlakeMTKernel::to_underlying_arguments(args2);
    auto blake_mt_kernel = cutlass::device_kernel<ComputeBlakeMTKernel>;
    dim3 grid2 = ComputeBlakeMTKernel::get_grid_shape(kernel_params2);
    dim3 block2 = ComputeBlakeMTKernel::get_block_shape();
    constexpr static int blake_mt_smem_size =
        ComputeBlakeMTKernel::SharedStorageSize;
    if (blake_mt_smem_size >= 48 * 1024) {
      gpuErrchk(cudaFuncSetAttribute(
          reinterpret_cast<const void*>(blake_mt_kernel),
          cudaFuncAttributeMaxDynamicSharedMemorySize, blake_mt_smem_size));
    }
    blake_mt_kernel<<<grid2, block2, blake_mt_smem_size, stream>>>(
        kernel_params2);
    gpuErrchk(cudaGetLastError());
  } else {
    using ComputeBlakeMTKernel =
        pearl::ComputeBlakeMTKernel<kLeavesPerMTBlock, false>;
    typename ComputeBlakeMTKernel::Arguments args2{
        reinterpret_cast<uint32_t*>(roots),
        num_blocks,
    };
    typename ComputeBlakeMTKernel::Params kernel_params2 =
        ComputeBlakeMTKernel::to_underlying_arguments(args2);
    auto blake_mt_kernel = cutlass::device_kernel<ComputeBlakeMTKernel>;
    dim3 grid2 = ComputeBlakeMTKernel::get_grid_shape(kernel_params2);
    dim3 block2 = ComputeBlakeMTKernel::get_block_shape();
    constexpr static int blake_mt_smem_size =
        ComputeBlakeMTKernel::SharedStorageSize;
    if (blake_mt_smem_size >= 48 * 1024) {
      gpuErrchk(cudaFuncSetAttribute(
          reinterpret_cast<const void*>(blake_mt_kernel),
          cudaFuncAttributeMaxDynamicSharedMemorySize, blake_mt_smem_size));
    }
    blake_mt_kernel<<<grid2, block2, blake_mt_smem_size, stream>>>(
        kernel_params2);
    gpuErrchk(cudaGetLastError());
  }

  // Further aggregation of roots if we have multiple blocks
  if (num_blocks_for_mt > 1) {
    using ReduceRootsKernel = pearl::ReduceRootsKernel<kNumConsumerThreads>;

    typename ReduceRootsKernel::Arguments args3{
        reinterpret_cast<uint32_t*>(roots),
        static_cast<uint32_t>(num_blocks_for_mt),
    };

    typename ReduceRootsKernel::Params kernel_params3 =
        ReduceRootsKernel::to_underlying_arguments(args3);

    auto reduce_roots_kernel = cutlass::device_kernel<ReduceRootsKernel>;

    dim3 grid3 = ReduceRootsKernel::get_grid_shape(kernel_params3);
    dim3 block3 = ReduceRootsKernel::get_block_shape();

    constexpr static int reduce_roots_smem_size =
        ReduceRootsKernel::SharedStorageSize;
    if (reduce_roots_smem_size >= 48 * 1024) {
      gpuErrchk(cudaFuncSetAttribute(
          reinterpret_cast<const void*>(reduce_roots_kernel),
          cudaFuncAttributeMaxDynamicSharedMemorySize, reduce_roots_smem_size));
    }

    reduce_roots_kernel<<<grid3, block3, reduce_roots_smem_size, stream>>>(
        kernel_params3);
    gpuErrchk(cudaGetLastError());
  }

  gpuErrchk(cudaMemcpyAsync(out, roots, blake3::CHAINING_VALUE_SIZE,
                            cudaMemcpyDeviceToDevice, stream));
}

// Dispatch helper for pipeline stages
template <int kNumConsumerThreads, int kLeavesPerMTBlock>
void dispatch_num_stages(uint32_t num_stages, const uint8_t* data,
                         uint32_t data_size, uint8_t* out,
                         const uint8_t key[blake3::KEY_SIZE],
                         uint32_t num_blocks, uint8_t* roots,
                         cudaDeviceProp& deviceProp, cudaStream_t stream,
                         bool upload_key = true,
                         uint8_t* leaf_cvs = nullptr) {
  switch (num_stages) {
    case 2:
      tensor_hash_impl<kNumConsumerThreads, 2, kLeavesPerMTBlock>(
          data, data_size, out, key, num_blocks, roots, deviceProp, stream,
          upload_key, leaf_cvs);
      break;
    case 3:
      tensor_hash_impl<kNumConsumerThreads, 3, kLeavesPerMTBlock>(
          data, data_size, out, key, num_blocks, roots, deviceProp, stream,
          upload_key, leaf_cvs);
      break;
    case 4:
      tensor_hash_impl<kNumConsumerThreads, 4, kLeavesPerMTBlock>(
          data, data_size, out, key, num_blocks, roots, deviceProp, stream,
          upload_key, leaf_cvs);
      break;
    default:
      TORCH_CHECK(false,
                  "Unsupported num_stages: " + std::to_string(num_stages) +
                      ". Supported values are: 2, 3, 4");
  }
}

// Dispatch helper for leaves per MT block (compute_blake_mt_kernel threads)
template <int kNumConsumerThreads>
void dispatch_leaves_per_mt_block(uint32_t leaves_per_mt_block,
                                  uint32_t num_stages, const uint8_t* data,
                                  uint32_t data_size, uint8_t* out,
                                  const uint8_t key[blake3::KEY_SIZE],
                                  uint32_t num_blocks, uint8_t* roots,
                                  cudaDeviceProp& deviceProp,
                                  cudaStream_t stream,
                                  bool upload_key = true,
                                  uint8_t* leaf_cvs = nullptr) {
  switch (leaves_per_mt_block) {
    case 256:
      dispatch_num_stages<kNumConsumerThreads, 256>(num_stages, data, data_size,
                                                    out, key, num_blocks, roots,
                                                    deviceProp, stream,
                                                    upload_key, leaf_cvs);
      break;
    case 512:
      dispatch_num_stages<kNumConsumerThreads, 512>(num_stages, data, data_size,
                                                    out, key, num_blocks, roots,
                                                    deviceProp, stream,
                                                    upload_key, leaf_cvs);
      break;
    case 1024:
      dispatch_num_stages<kNumConsumerThreads, 1024>(
          num_stages, data, data_size, out, key, num_blocks, roots, deviceProp,
          stream, upload_key, leaf_cvs);
      break;
    default:
      throw std::runtime_error("Unsupported leaves_per_mt_block: " +
                               std::to_string(leaves_per_mt_block) +
                               ". Supported values are: 256, 512, 1024");
  }
}

// Dispatch helper for number of threads per block
// Supports up to 256 threads due to load distribution constraints.
void dispatch_threads_per_block(
    uint32_t threads_per_block, uint32_t leaves_per_mt_block,
    uint32_t num_stages, const uint8_t* data, uint32_t data_size, uint8_t* out,
    const uint8_t key[blake3::KEY_SIZE], uint32_t num_blocks, uint8_t* roots,
    cudaDeviceProp& deviceProp, cudaStream_t stream, bool upload_key = true,
    uint8_t* leaf_cvs = nullptr) {
  switch (threads_per_block) {
    case 128:
      dispatch_leaves_per_mt_block<128>(leaves_per_mt_block, num_stages, data,
                                        data_size, out, key, num_blocks, roots,
                                        deviceProp, stream, upload_key,
                                        leaf_cvs);
      break;
    case 256:
      dispatch_leaves_per_mt_block<256>(leaves_per_mt_block, num_stages, data,
                                        data_size, out, key, num_blocks, roots,
                                        deviceProp, stream, upload_key,
                                        leaf_cvs);
      break;
    case 512:
      dispatch_leaves_per_mt_block<512>(leaves_per_mt_block, num_stages, data,
                                        data_size, out, key, num_blocks, roots,
                                        deviceProp, stream, upload_key,
                                        leaf_cvs);
      break;
    default:
      throw std::runtime_error("Unsupported threads_per_block: " +
                               std::to_string(threads_per_block) +
                               ". Supported values are: 128, 256, 512");
  }
}

// SM90+ dispatch (TMA-based pipeline) — called only for Hopper and later.
static void tensor_hash_sm90(
    const uint8_t* data, uint32_t data_size, uint8_t* out,
    const uint8_t key[32], uint32_t num_blocks,
    uint32_t threads_per_block, uint32_t num_stages,
    uint32_t leaves_per_mt_block,
    uint8_t* roots, cudaDeviceProp& deviceProp, cudaStream_t stream,
    bool upload_key = true, uint8_t* leaf_cvs = nullptr) {
  dispatch_threads_per_block(threads_per_block, leaves_per_mt_block, num_stages,
                             data, data_size, out, key, num_blocks, roots,
                             deviceProp, stream, upload_key, leaf_cvs);
}

void commitment_hash_from_merkle_roots(
    const uint8_t* A_merkle_root, const uint8_t* B_merkle_root,
    const uint8_t* key, uint8_t* A_commitment_hash, uint8_t* B_commitment_hash,
    cudaDeviceProp& deviceProp, cudaStream_t stream) {

  using CommitmentHashFromMerkleRootsKernel =
      pearl::CommitmentHashFromMerkleRootsKernel;

  typename CommitmentHashFromMerkleRootsKernel::Arguments args{
      static_cast<const uint8_t*>(A_merkle_root),
      static_cast<const uint8_t*>(B_merkle_root),
      static_cast<const uint8_t*>(key),
      static_cast<uint8_t*>(A_commitment_hash),
      static_cast<uint8_t*>(B_commitment_hash)};

  typename CommitmentHashFromMerkleRootsKernel::Params kernel_params =
      CommitmentHashFromMerkleRootsKernel::to_underlying_arguments(args);

  auto kernel = cutlass::device_kernel<CommitmentHashFromMerkleRootsKernel>;

  constexpr static int commitment_hash_smem_size =
      CommitmentHashFromMerkleRootsKernel::SharedStorageSize;

  if (commitment_hash_smem_size >= 48 * 1024) {
    gpuErrchk(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   commitment_hash_smem_size));
  }

  dim3 grid = dim3(1);
  dim3 block = dim3(1);

  kernel<<<grid, block, commitment_hash_smem_size, stream>>>(kernel_params);
}
