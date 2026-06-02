// SM80 portable host-side dispatch for tensor_hash.
//
// Replaces tensor_hash_host.hpp's MerkleTreeRootsKernel (SM90/TMA) with
// MerkleTreeRootsKernelSM80 (global-memory loads + __syncthreads__).
// The subsequent stages — ComputeBlakeMTKernel, ReduceRootsKernel,
// CommitmentHashFromMerkleRootsKernel — are already portable and reused as-is.
// NOTE: This header must be included AFTER tensor_hash_host.hpp, which pulls in
// compute_blake_mt_kernel.hpp and reduce_roots_kernel.h (they lack include guards).
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "blake3/blake3_constants.hpp"
#include "gemm/error_check.hpp"
#include "merkle_tree_roots_kernel_sm80.hpp"
#include "tensor_hash_constants.cuh"

#include "cute/tensor.hpp"
#include <cutlass/cutlass.h>
#include "cutlass/device_kernel.h"

using namespace cute;
using u8  = uint8_t;
using u32 = uint32_t;

// ── SM80 tensor_hash implementation ─────────────────────────────────────────
// kNumThreads: consumer thread count (128, 256, 512)
// kLeavesPerMTBlock: threads for ComputeBlakeMTKernel (256, 512, 1024)
template <int kNumThreads, int kLeavesPerMTBlock>
void tensor_hash_sm80_impl(const uint8_t* data, uint32_t data_size,
                           uint8_t* out, const uint8_t key[blake3::KEY_SIZE],
                           uint32_t num_blocks, uint8_t* roots,
                           cudaStream_t stream, bool upload_key = true,
                           uint8_t* leaf_cvs = nullptr) {
  if (upload_key) set_key(key, stream);

  // ── Stage 1: chunk hashing (SM80 portable kernel) ─────────────────────
  using RootsKernel = pearl::MerkleTreeRootsKernelSM80<kNumThreads>;

  typename RootsKernel::Arguments args{data, data_size, roots, leaf_cvs};
  typename RootsKernel::Params kernel_params =
      RootsKernel::to_underlying_arguments(args);

  dim3 grid  = RootsKernel::get_grid_shape(kernel_params);
  dim3 block = RootsKernel::get_block_shape();
  constexpr int smem_size = RootsKernel::SharedStorageSize;

  auto kernel = cutlass::device_kernel<RootsKernel>;
  if (smem_size >= 48 * 1024) {
    gpuErrchk(cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  kernel<<<grid, block, smem_size, stream>>>(kernel_params);
  gpuErrchk(cudaGetLastError());

  // ── Stage 2: intermediate Merkle-tree reduction ───────────────────────
  // (ComputeBlakeMTKernel — already portable, reused verbatim)
  const int num_blocks_for_mt =
      (num_blocks + kLeavesPerMTBlock - 1) / kLeavesPerMTBlock;
  const bool is_single_block = (num_blocks_for_mt == 1);

  if (is_single_block) {
    using MTKernel = pearl::ComputeBlakeMTKernel<kLeavesPerMTBlock, true>;
    typename MTKernel::Arguments args2{
        reinterpret_cast<uint32_t*>(roots), num_blocks};
    typename MTKernel::Params kp2 = MTKernel::to_underlying_arguments(args2);
    auto mt_kernel = cutlass::device_kernel<MTKernel>;
    constexpr int mt_smem = MTKernel::SharedStorageSize;
    if (mt_smem >= 48 * 1024) {
      gpuErrchk(cudaFuncSetAttribute(
          reinterpret_cast<const void*>(mt_kernel),
          cudaFuncAttributeMaxDynamicSharedMemorySize, mt_smem));
    }
    mt_kernel<<<MTKernel::get_grid_shape(kp2), MTKernel::get_block_shape(),
                mt_smem, stream>>>(kp2);
    gpuErrchk(cudaGetLastError());
  } else {
    using MTKernel = pearl::ComputeBlakeMTKernel<kLeavesPerMTBlock, false>;
    typename MTKernel::Arguments args2{
        reinterpret_cast<uint32_t*>(roots), num_blocks};
    typename MTKernel::Params kp2 = MTKernel::to_underlying_arguments(args2);
    auto mt_kernel = cutlass::device_kernel<MTKernel>;
    constexpr int mt_smem = MTKernel::SharedStorageSize;
    if (mt_smem >= 48 * 1024) {
      gpuErrchk(cudaFuncSetAttribute(
          reinterpret_cast<const void*>(mt_kernel),
          cudaFuncAttributeMaxDynamicSharedMemorySize, mt_smem));
    }
    mt_kernel<<<MTKernel::get_grid_shape(kp2), MTKernel::get_block_shape(),
                mt_smem, stream>>>(kp2);
    gpuErrchk(cudaGetLastError());
  }

  // ── Stage 3: final root aggregation (if multi-block) ──────────────────
  if (num_blocks_for_mt > 1) {
    using ReduceKernel = pearl::ReduceRootsKernel<kNumThreads>;
    typename ReduceKernel::Arguments args3{
        reinterpret_cast<uint32_t*>(roots),
        static_cast<uint32_t>(num_blocks_for_mt)};
    typename ReduceKernel::Params kp3 =
        ReduceKernel::to_underlying_arguments(args3);
    auto rk = cutlass::device_kernel<ReduceKernel>;
    constexpr int rk_smem = ReduceKernel::SharedStorageSize;
    if (rk_smem >= 48 * 1024) {
      gpuErrchk(cudaFuncSetAttribute(
          reinterpret_cast<const void*>(rk),
          cudaFuncAttributeMaxDynamicSharedMemorySize, rk_smem));
    }
    rk<<<ReduceKernel::get_grid_shape(kp3), ReduceKernel::get_block_shape(),
         rk_smem, stream>>>(kp3);
    gpuErrchk(cudaGetLastError());
  }

  // Copy final root to output
  gpuErrchk(cudaMemcpyAsync(out, roots, blake3::CHAINING_VALUE_SIZE,
                            cudaMemcpyDeviceToDevice, stream));
}

// ── Dispatch helpers (thread count × leaves-per-MT-block) ───────────────
template <int kNumThreads>
void dispatch_leaves_sm80(uint32_t leaves_per_mt_block,
                          const uint8_t* data, uint32_t data_size,
                          uint8_t* out, const uint8_t key[blake3::KEY_SIZE],
                          uint32_t num_blocks, uint8_t* roots,
                          cudaStream_t stream, bool upload_key = true,
                          uint8_t* leaf_cvs = nullptr) {
  switch (leaves_per_mt_block) {
    case 256:
      tensor_hash_sm80_impl<kNumThreads, 256>(
          data, data_size, out, key, num_blocks, roots, stream, upload_key,
          leaf_cvs);
      break;
    case 512:
      tensor_hash_sm80_impl<kNumThreads, 512>(
          data, data_size, out, key, num_blocks, roots, stream, upload_key,
          leaf_cvs);
      break;
    case 1024:
      tensor_hash_sm80_impl<kNumThreads, 1024>(
          data, data_size, out, key, num_blocks, roots, stream, upload_key,
          leaf_cvs);
      break;
    default:
      throw std::runtime_error(
          "SM80 tensor_hash: unsupported leaves_per_mt_block: " +
          std::to_string(leaves_per_mt_block));
  }
}

void tensor_hash_sm80(const uint8_t* data, uint32_t data_size, uint8_t* out,
                      const uint8_t key[blake3::KEY_SIZE], uint32_t num_blocks,
                      uint32_t threads_per_block,
                      uint32_t leaves_per_mt_block,
                      uint8_t* roots, cudaStream_t stream,
                      bool upload_key = true,
                      uint8_t* leaf_cvs = nullptr) {
  switch (threads_per_block) {
    case 128:
      dispatch_leaves_sm80<128>(leaves_per_mt_block, data, data_size, out, key,
                                num_blocks, roots, stream, upload_key,
                                leaf_cvs);
      break;
    case 256:
      dispatch_leaves_sm80<256>(leaves_per_mt_block, data, data_size, out, key,
                                num_blocks, roots, stream, upload_key,
                                leaf_cvs);
      break;
    case 512:
      dispatch_leaves_sm80<512>(leaves_per_mt_block, data, data_size, out, key,
                                num_blocks, roots, stream, upload_key,
                                leaf_cvs);
      break;
    default:
      throw std::runtime_error(
          "SM80 tensor_hash: unsupported threads_per_block: " +
          std::to_string(threads_per_block));
  }
}
