#pragma once

#include "cute/tensor.hpp"
#include "denoise_converter_kernel.h"
#include "error_check.hpp"
#include "pearl_api_params.h"

#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"  // For device_kernel

template <int R>
void run_denoise_converter(PearlAPIParams& params, cudaStream_t stream = 0) {
  using namespace cute;

  using ElementDenoiseIn = int32_t;
  using ElementDenoiseOut = half_t;

  using DenoiseConverterKernel =
      pearl::DenoiseConverterKernel<R, 128 /*NumThreads*/>;

  typename DenoiseConverterKernel::Arguments args{
      .ptr_AxEBL_in =
          static_cast<ElementDenoiseIn const*>(params.ptr_AxEBL_int32),
      .ptr_EARxBpEB_in =
          static_cast<ElementDenoiseIn const*>(params.ptr_EARxBpEB_int32),
      .ptr_AxEBL_out = static_cast<ElementDenoiseOut*>(params.ptr_AxEBL_mma),
      .ptr_EARxBpEB_out =
          static_cast<ElementDenoiseOut*>(params.ptr_EARxBpEB_mma),
      .m = params.ptr_AxEBL_int32 ? params.m : 0,
      .n = params.ptr_EARxBpEB_int32 ? params.n : 0,
  };

  typename DenoiseConverterKernel::Params kernel_params =
      DenoiseConverterKernel::to_underlying_arguments(args);

  dim3 grid_dims = DenoiseConverterKernel::get_grid_shape(kernel_params);
  dim3 block_dims = DenoiseConverterKernel::get_block_shape();

  constexpr static int smem_size =
      DenoiseConverterKernel::SharedStorageSize;  // 0, no smem used
  auto kernel = cutlass::device_kernel<DenoiseConverterKernel>;
  kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);

  CHECK_CUDA_KERNEL_LAUNCH();
}
