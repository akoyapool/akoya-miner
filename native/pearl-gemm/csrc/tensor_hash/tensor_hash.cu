#include "tensor_hash_host.hpp"
#include "tensor_hash_host_sm80.hpp"
#include "bseed_merkle_tree_roots_kernel_sm80.hpp"

namespace {

uint32_t load_le32_host(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) |
         (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

pearl::BSeedBlock make_bseed_block(const uint8_t bseed[32]) {
  pearl::BSeedBlock block{};
  for (int i = 0; i < 8; ++i) {
    block.word[i] = load_le32_host(bseed + i * 4);
  }
  return block;
}

template <int kNumThreads, int kLeavesPerMTBlock>
void bseed_expand_and_tensor_hash_sm80_impl(
    const uint8_t bseed[32], uint8_t* data, uint32_t data_size,
    uint8_t* out, const uint8_t key[blake3::KEY_SIZE], uint32_t num_blocks,
    uint8_t* roots, cudaStream_t stream, uint8_t* leaf_cvs = nullptr) {
  set_key(key, stream);

  using RootsKernel = pearl::BSeedMerkleTreeRootsKernelSM80<kNumThreads>;
  typename RootsKernel::Arguments args{
      make_bseed_block(bseed), data, data_size, roots, leaf_cvs};
  typename RootsKernel::Params kernel_params =
      RootsKernel::to_underlying_arguments(args);

  dim3 grid = RootsKernel::get_grid_shape(kernel_params);
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

  gpuErrchk(cudaMemcpyAsync(out, roots, blake3::CHAINING_VALUE_SIZE,
                            cudaMemcpyDeviceToDevice, stream));
}

template <int kNumThreads>
void dispatch_bseed_leaves_sm80(
    uint32_t leaves_per_mt_block, const uint8_t bseed[32], uint8_t* data,
    uint32_t data_size, uint8_t* out, const uint8_t key[blake3::KEY_SIZE],
    uint32_t num_blocks, uint8_t* roots, cudaStream_t stream,
    uint8_t* leaf_cvs = nullptr) {
  switch (leaves_per_mt_block) {
    case 256:
      bseed_expand_and_tensor_hash_sm80_impl<kNumThreads, 256>(
          bseed, data, data_size, out, key, num_blocks, roots, stream,
          leaf_cvs);
      break;
    case 512:
      bseed_expand_and_tensor_hash_sm80_impl<kNumThreads, 512>(
          bseed, data, data_size, out, key, num_blocks, roots, stream,
          leaf_cvs);
      break;
    case 1024:
      bseed_expand_and_tensor_hash_sm80_impl<kNumThreads, 1024>(
          bseed, data, data_size, out, key, num_blocks, roots, stream,
          leaf_cvs);
      break;
    default:
      throw std::runtime_error(
          "SM80 bseed_expand_and_tensor_hash: unsupported leaves_per_mt_block: " +
          std::to_string(leaves_per_mt_block));
  }
}

}  // namespace

void tensor_hash_set_key(const uint8_t key[32], cudaStream_t stream) {
  set_key(key, stream);
}

// Unified tensor_hash dispatch — routes to the SM90 TMA pipeline on Hopper
// only. All other architectures (sm_80/86/89 Ampere/Ada, sm_100 Blackwell
// datacentre, sm_120 Blackwell consumer / RTX 50-series) go through the
// portable SM80 kernel.
//
// Why exact-match on major == 9 and not "major >= 9":
//   • The SM90 kernel uses sm_90a-only intrinsics — wgmma.m64nNk32, the
//     warpgroup-barrier shape, and CUTLASS's PipelineTmaAsync. These were
//     removed in Blackwell (sm_100/sm_120) — the new tensor cores use
//     tcgen05 and a different pipeline shape. Routing major>=10 to this
//     path would either fail to load (no matching cubin in portable
//     builds) or hit an illegal instruction at launch.
//   • The portable SM80 kernel uses only direct GMEM loads + register
//     BLAKE3 + __syncthreads(), all of which work unchanged from sm_80
//     through sm_120 and forward.
//
// In a portable build (PEARL_GEMM_PORTABLE) the sm_90a cubin is not
// shipped, so the SM90 entry point is compiled out entirely — H100 still
// runs, just on the slower portable path. Users who want the fast Hopper
// path should set PEARL_GEMM_ARCH=h100 at build time.
void tensor_hash(
    const uint8_t* data, uint32_t data_size, uint8_t* out,
    const uint8_t key[32], uint32_t num_blocks,
    uint32_t threads_per_block,
    uint32_t num_stages,
    uint32_t leaves_per_mt_block,
    uint8_t* roots, cudaDeviceProp& deviceProp, cudaStream_t stream) {
#ifndef PEARL_GEMM_PORTABLE
  if (deviceProp.major == 9) {
    // Hopper (H100, H200, GH200): sm_90a TMA + warpgroup-MMA pipeline.
    tensor_hash_sm90(data, data_size, out, key, num_blocks,
                     threads_per_block, num_stages, leaves_per_mt_block,
                     roots, deviceProp, stream);
    return;
  }
#else
  (void)num_stages;  // portable kernel has no pipeline stages
#endif
  // Ampere / Ada / Blackwell (A100, RTX 3080-5090, B100/B200), or any
  // arch in a portable build: GMEM-load kernel.
  tensor_hash_sm80(data, data_size, out, key, num_blocks,
                     threads_per_block, leaves_per_mt_block, roots, stream);
}

void tensor_hash_with_leaf_cvs(
    const uint8_t* data, uint32_t data_size, uint8_t* out,
    const uint8_t key[32], uint32_t num_blocks,
    uint32_t threads_per_block,
    uint32_t num_stages,
    uint32_t leaves_per_mt_block,
    uint8_t* roots, uint8_t* leaf_cvs, cudaDeviceProp& deviceProp,
    cudaStream_t stream) {
#ifndef PEARL_GEMM_PORTABLE
  if (deviceProp.major == 9) {
    tensor_hash_sm90(data, data_size, out, key, num_blocks,
                     threads_per_block, num_stages, leaves_per_mt_block,
                     roots, deviceProp, stream, /*upload_key=*/true,
                     leaf_cvs);
    return;
  }
#else
  (void)num_stages;
  (void)deviceProp;
#endif
  tensor_hash_sm80(data, data_size, out, key, num_blocks,
                   threads_per_block, leaves_per_mt_block, roots, stream,
                   /*upload_key=*/true, leaf_cvs);
}

void tensor_hash_no_key_upload(
    const uint8_t* data, uint32_t data_size, uint8_t* out,
    const uint8_t key[32], uint32_t num_blocks,
    uint32_t threads_per_block,
    uint32_t num_stages,
    uint32_t leaves_per_mt_block,
    uint8_t* roots, cudaDeviceProp& deviceProp, cudaStream_t stream) {
#ifndef PEARL_GEMM_PORTABLE
  if (deviceProp.major == 9) {
    tensor_hash_sm90(data, data_size, out, key, num_blocks,
                     threads_per_block, num_stages, leaves_per_mt_block,
                     roots, deviceProp, stream, /*upload_key=*/false);
    return;
  }
#else
  (void)num_stages;
#endif
  tensor_hash_sm80(data, data_size, out, key, num_blocks,
                   threads_per_block, leaves_per_mt_block, roots, stream,
                   /*upload_key=*/false);
}

void bseed_expand_and_tensor_hash(
    const uint8_t bseed[32], uint8_t* data, uint32_t data_size, uint8_t* out,
    const uint8_t key[32], uint32_t num_blocks, uint32_t threads_per_block,
    uint32_t num_stages, uint32_t leaves_per_mt_block, uint8_t* roots,
    cudaDeviceProp& deviceProp, cudaStream_t stream) {
  (void)num_stages;
  (void)deviceProp;
  switch (threads_per_block) {
    case 128:
      dispatch_bseed_leaves_sm80<128>(leaves_per_mt_block, bseed, data,
                                      data_size, out, key, num_blocks, roots,
                                      stream);
      break;
    case 256:
      dispatch_bseed_leaves_sm80<256>(leaves_per_mt_block, bseed, data,
                                      data_size, out, key, num_blocks, roots,
                                      stream);
      break;
    case 512:
      dispatch_bseed_leaves_sm80<512>(leaves_per_mt_block, bseed, data,
                                      data_size, out, key, num_blocks, roots,
                                      stream);
      break;
    default:
      throw std::runtime_error(
          "SM80 bseed_expand_and_tensor_hash: unsupported threads_per_block: " +
          std::to_string(threads_per_block));
  }
}

void bseed_expand_and_tensor_hash_with_leaf_cvs(
    const uint8_t bseed[32], uint8_t* data, uint32_t data_size, uint8_t* out,
    const uint8_t key[32], uint32_t num_blocks, uint32_t threads_per_block,
    uint32_t num_stages, uint32_t leaves_per_mt_block, uint8_t* roots,
    uint8_t* leaf_cvs, cudaDeviceProp& deviceProp, cudaStream_t stream) {
  (void)num_stages;
  (void)deviceProp;
  switch (threads_per_block) {
    case 128:
      dispatch_bseed_leaves_sm80<128>(leaves_per_mt_block, bseed, data,
                                      data_size, out, key, num_blocks, roots,
                                      stream, leaf_cvs);
      break;
    case 256:
      dispatch_bseed_leaves_sm80<256>(leaves_per_mt_block, bseed, data,
                                      data_size, out, key, num_blocks, roots,
                                      stream, leaf_cvs);
      break;
    case 512:
      dispatch_bseed_leaves_sm80<512>(leaves_per_mt_block, bseed, data,
                                      data_size, out, key, num_blocks, roots,
                                      stream, leaf_cvs);
      break;
    default:
      throw std::runtime_error(
          "SM80 bseed_expand_and_tensor_hash: unsupported threads_per_block: " +
          std::to_string(threads_per_block));
  }
}
