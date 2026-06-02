#pragma once

#include "cute/tensor.hpp"

#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"  // For device_kernel

#include "pearl_api_params.h"
#include "pearl_gemm_host.h"
#include "pearl_noisingA_host.h"
#include "pearl_noisingB_host.h"
#include "static_switch.h"

template <class ElementDenoise_AxEBL, int R, int bM_noising, int bK_noising,
          int kStages>
void run_pearl_noising_A_(PearlAPIParams& params, cudaStream_t stream = 0) {
  using namespace cute;
  using TileShape_MRK = Shape<Int<bM_noising>, Int<R>, Int<bK_noising>>;

  BOOL_SWITCH(params.k % get<2>(TileShape_MRK{}) == 0, IsEvenKNoising,
              run_pearl_noising_A<ElementDenoise_AxEBL, TileShape_MRK, kStages,
                                  IsEvenKNoising>(params, stream););
}

template <class ElementDenoise_EARxBpEB, int R, int bN_noising, int bK_noising,
          int kStages>
void run_pearl_noising_B_(PearlAPIParams& params, cudaStream_t stream = 0) {
  using namespace cute;
  using TileShape_NRK = Shape<Int<bN_noising>, Int<R>, Int<bK_noising>>;

  BOOL_SWITCH(params.k % get<2>(TileShape_NRK{}) == 0, IsEvenKNoising,
              run_pearl_noising_B<ElementDenoise_EARxBpEB, TileShape_NRK,
                                  kStages, IsEvenKNoising>(params, stream););
}

template <class ElementOut, int R, int bM, int bN, int bK, int kStages,
          int cM = 1, int cN = 1, bool SkipReduction = true,
          bool SkipDenoising = false, bool EnableDebug = false,
          bool SkipCStore = false>
void run_pearl_gemm_(PearlAPIParams& params, cudaStream_t stream = 0) {
  using namespace cute;
  using TileShape_MNKR = Shape<Int<bM>, Int<bN>, Int<bK>, Int<R>>;
  bool is_even_m = params.m % get<0>(TileShape_MNKR{}) == 0;
  bool is_even_n = params.n % get<1>(TileShape_MNKR{}) == 0;

  BOOL_SWITCH(
      is_even_m, IsEvenM,
      BOOL_SWITCH(
          is_even_n, IsEvenN,

          run_pearl_gemm<ElementOut, TileShape_MNKR, kStages, cM, cN, IsEvenM,
                         IsEvenN, SkipReduction, SkipDenoising, EnableDebug,
                         SkipCStore>(
              params, stream);

      ););
}
