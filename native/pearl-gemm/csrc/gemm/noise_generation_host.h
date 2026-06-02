#pragma once

#include "cute/tensor.hpp"
#include "error_check.hpp"
#include "noise_generation_kernel.h"
#include "pearl_api_params.h"

#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"  // For device_kernel

// SEED_LEN and KEY_LEN are in units of bytes
template <int R, int NumThreads>
void run_noise_generation(Noise_gen_params& params, cudaStream_t stream = 0) {
  using namespace cute;

  using NoiseGenKernel = pearl::NoiseGenerationKernel<R, NumThreads>;

  bool const generate_EAR = params.ptr_EAR_R_major || params.ptr_EAR_K_major;
  bool const generate_EBL = params.ptr_EBL_R_major || params.ptr_EBL_K_major;

  typename NoiseGenKernel::Arguments args{
      .ptr_EAL = static_cast<int8_t*>(params.ptr_EAL),
      .ptr_EAL_fp16 = static_cast<half_t*>(params.ptr_EAL_fp16),
      .ptr_EAR_R_major = static_cast<int8_t*>(params.ptr_EAR_R_major),
      .ptr_EAR_K_major = static_cast<int8_t*>(params.ptr_EAR_K_major),
      .ptr_EBL_R_major = static_cast<int8_t*>(params.ptr_EBL_R_major),
      .ptr_EBL_K_major = static_cast<int8_t*>(params.ptr_EBL_K_major),
      .ptr_EBR = static_cast<int8_t*>(params.ptr_EBR),
      .ptr_EBR_fp16 = static_cast<half_t*>(params.ptr_EBR_fp16),
      .num_rows_EAL = params.ptr_EAL ? params.m : 0,
      .length_EAR = generate_EAR ? params.k : 0,
      .length_EBL = generate_EBL ? params.k : 0,
      .num_rows_EBR = params.ptr_EBR ? params.n : 0,
      .ptr_key_A = static_cast<uint8_t const*>(params.ptr_key_A),
      .ptr_key_B = static_cast<uint8_t const*>(params.ptr_key_B),
      .ptr_aux_buffer = static_cast<uint32_t*>(params.ptr_aux_buffer),
      .aux_buffer_size = params.ptr_aux_buffer ? params.aux_buffer_size : 0};

  typename NoiseGenKernel::Params kernel_params =
      NoiseGenKernel::to_underlying_arguments(args);

  dim3 grid_dims = NoiseGenKernel::get_grid_shape(kernel_params);
  dim3 block_dims = NoiseGenKernel::get_block_shape();
  constexpr static int smem_size = NoiseGenKernel::SharedStorageSize;

  auto kernel = cutlass::device_kernel<NoiseGenKernel>;
  if (smem_size >= 48 * 1024) {
    gpuErrchk(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
  CHECK_CUDA_KERNEL_LAUNCH();
}
