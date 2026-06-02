#pragma once
#include "error_check.hpp"
#include "pearl_api_params.h"
#include "pearl_noisingA_kernel.h"

#include "cute/tensor.hpp"

#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"  // For device_kernel

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <class ElementDenoise_AxEBL, class TileShape_MRK, int kStages,
          bool IsEvenK = false>
void run_pearl_noising_A(PearlAPIParams const& params,
                         cudaStream_t stream = 0) {
  using namespace cute;
  using Element = int8_t;
  using ElementIndex = uint8_t;
  using ElementDenoise = ElementDenoise_AxEBL;
  using ElementScale = float;
  static constexpr bool NoReduction = !(cute::is_same_v<ElementDenoise, int>);
  using NoisingKernelA =
      pearl::NoisingKernelA<TileShape_MRK, /*kNumThreads=*/128, Element,
                            ElementDenoise, kStages, IsEvenK, NoReduction>;

  int total_k_blocks = ceil_div(params.k, get<2>(TileShape_MRK{}));
  bool no_reduce = NoReduction || params.k_blocks_per_split_noising_A <= 0;
  typename NoisingKernelA::Arguments args{
      .ptr_A = static_cast<Element const*>(params.ptr_A),
      .ptr_EAL = static_cast<Element const*>(params.ptr_EAL),
      .ptr_EAR = static_cast<Element const*>(params.ptr_EAR_R_major),
      .ptr_EBL = static_cast<Element const*>(params.ptr_EBL_K_major),
      .ptr_A_out = static_cast<Element*>(params.ptr_ApEA),
      .ptr_AxEBL = static_cast<ElementDenoise*>(params.ptr_AxEBL),
      .m = params.m,
      .k = params.k,
      .num_k_blocks =
          no_reduce ? total_k_blocks : params.k_blocks_per_split_noising_A,
      .total_k_blocks = total_k_blocks};

  typename NoisingKernelA::Params kernel_params =
      NoisingKernelA::to_underlying_arguments(args);

  dim3 grid_dims = NoisingKernelA::get_grid_shape(kernel_params);
  dim3 block_dims = NoisingKernelA::get_block_shape();
  constexpr static int smem_size = NoisingKernelA::SharedStorageSize;

  auto kernel = cutlass::device_kernel<NoisingKernelA>;
  if (smem_size >= 48 * 1024) {
    gpuErrchk(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
  CHECK_CUDA_KERNEL_LAUNCH();
}
