#pragma once
#include "error_check.hpp"
#include "pearl_api_params.h"
#include "pearl_noisingB_kernel.h"

#include "cute/tensor.hpp"

#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"  // For device_kernel

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename ElementDenoise_EARxBpEB, class TileShape_NRK, int kStages,
          bool IsEvenK = false>
void run_pearl_noising_B(PearlAPIParams const& params,
                         cudaStream_t stream = 0) {
  using namespace cute;
  using Element = int8_t;
  using ElementIndex = uint8_t;
  using ElementDenoise = ElementDenoise_EARxBpEB;
  static constexpr bool NoReduction = !(cute::is_same_v<ElementDenoise, int>);
  using NoisingKernelB =
      pearl::NoisingKernelB<TileShape_NRK, 128, Element, ElementDenoise,
                            kStages, IsEvenK, NoReduction>;

  int total_k_blocks = ceil_div(params.k, get<2>(TileShape_NRK{}));
  bool no_reduce = NoReduction || params.k_blocks_per_split_noising_B <= 0;
  typename NoisingKernelB::Arguments args{
      .ptr_B = static_cast<Element const*>(params.ptr_B),
      .ptr_EBR = static_cast<Element const*>(params.ptr_EBR),
      .ptr_EAR = static_cast<Element const*>(params.ptr_EAR_K_major),
      .ptr_EBL = static_cast<Element const*>(params.ptr_EBL_R_major),
      .ptr_BpEB = static_cast<Element*>(params.ptr_BpEB),
      .ptr_EARxBpEB = static_cast<ElementDenoise*>(params.ptr_EARxBpEB),
      .n = params.n,
      .k = params.k,
      .num_k_blocks =
          no_reduce ? total_k_blocks : params.k_blocks_per_split_noising_B,
      .total_k_blocks = total_k_blocks};

  typename NoisingKernelB::Params kernel_params =
      NoisingKernelB::to_underlying_arguments(args);

  dim3 grid_dims = NoisingKernelB::get_grid_shape(kernel_params);
  dim3 block_dims = NoisingKernelB::get_block_shape();
  constexpr static int smem_size = NoisingKernelB::SharedStorageSize;

  auto kernel = cutlass::device_kernel<NoisingKernelB>;
  if (smem_size >= 48 * 1024) {
    gpuErrchk(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
  CHECK_CUDA_KERNEL_LAUNCH();
}
