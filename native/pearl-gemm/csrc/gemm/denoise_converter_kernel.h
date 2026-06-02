#pragma once

#include "utils.h"

#include "cute/tensor.hpp"

#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/detail/layout.hpp>

#include "pearl_gemm_constants.hpp"

namespace pearl {

using namespace cute;

template <int R, int NumThreads>
class DenoiseConverterKernel {
 public:
  using ElementDenoiseIn = int32_t;
  using ElementDenoiseOut = half_t;
  using ElementScale = float;

  struct Arguments {
    ElementDenoiseIn const* const ptr_AxEBL_in;
    ElementDenoiseIn const* const ptr_EARxBpEB_in;
    ElementDenoiseOut* const ptr_AxEBL_out;
    ElementDenoiseOut* const ptr_EARxBpEB_out;
    const int m;
    const int n;
  };

  struct Params {
    ElementDenoiseIn const* const ptr_AxEBL_in;
    ElementDenoiseIn const* const ptr_EARxBpEB_in;
    ElementDenoiseOut* const ptr_AxEBL_out;
    ElementDenoiseOut* const ptr_EARxBpEB_out;
    const int m;
    const int n;
    const int num_m_blocks;
    const int num_n_blocks;
  };

  static constexpr int SharedStorageSize = 0;  // no smem used
  static constexpr uint32_t MaxThreadsPerBlock = NumThreads;
  static constexpr int NumThreadsPerWarp = 32;
  static constexpr int NumWarps = NumThreads / NumThreadsPerWarp;

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  // each thread loads, converts, and stores 4 values, which is the maximum that fits in a single
  //  load instruction for 32-bit width
  using LoadDtype = uint128_t;
  static constexpr int ValsPerThread =
      sizeof(LoadDtype) / sizeof(ElementDenoiseIn);
  static_assert(R % ValsPerThread == 0);
  static constexpr int ThreadsPerRow = R / ValsPerThread;  // 16,32 for R=64,128

  // 8,4 for R=64,128
  static constexpr int RowsPerCTA = NumThreads / ThreadsPerRow;

  // Convert to underlying arguments
  static Params to_underlying_arguments(Arguments const& args) {
    const int num_m_blocks =
        args.ptr_AxEBL_in ? ceil_div(args.m, RowsPerCTA) : 0;
    const int num_n_blocks =
        args.ptr_EARxBpEB_in ? ceil_div(args.n, RowsPerCTA) : 0;
    return {.ptr_AxEBL_in = args.ptr_AxEBL_in,
            .ptr_EARxBpEB_in = args.ptr_EARxBpEB_in,
            .ptr_AxEBL_out = args.ptr_AxEBL_out,
            .ptr_EARxBpEB_out = args.ptr_EARxBpEB_out,
            .m = args.m,
            .n = args.n,
            .num_m_blocks = num_m_blocks,
            .num_n_blocks = num_n_blocks};
  };

  static dim3 get_grid_shape(Params const& params) {
    const int num_blocks = params.num_m_blocks + params.num_n_blocks;
    return dim3(num_blocks, 1, 1);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  using GmemCopyAtomMK =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementDenoiseIn>;
  using G2RValLayout = Layout<Shape<_1, Int<ValsPerThread>>>;
  using ThrTileShape = Shape<Int<RowsPerCTA>, Int<ThreadsPerRow>>;
  using G2RThrLayout = Layout<ThrTileShape, LayoutRight::Apply<ThrTileShape>>;
  using G2RTiledCopy = decltype(make_tiled_copy(
      GmemCopyAtomMK{}, G2RThrLayout{}, G2RValLayout{}));
  using CTAValTileShape = Shape<Int<RowsPerCTA>, Int<R>>;

  CUTLASS_DEVICE void convert_block(Params const& params, bool convert_AxEBL,
                                    int bid) {
    const int tid = threadIdx.x;
    const int num_rows = convert_AxEBL ? params.m : params.n;
    const int ScaleFactor =
        convert_AxEBL ? pearl::kAxEBLScaleFactor : pearl::kEARxBpEBScaleFactor;
    ElementDenoiseIn const* const ptr_in =
        convert_AxEBL ? params.ptr_AxEBL_in : params.ptr_EARxBpEB_in;
    ElementDenoiseOut* const ptr_out =
        convert_AxEBL ? params.ptr_AxEBL_out : params.ptr_EARxBpEB_out;

    Layout mIO_layout =
        make_layout(make_shape(num_rows, Int<R>{}), LayoutRight{});
    Tensor mI = make_tensor(make_gmem_ptr(ptr_in), mIO_layout);
    Tensor mO = make_tensor(make_gmem_ptr(ptr_out), mIO_layout);
    Tensor gI = local_tile(mI, CTAValTileShape{}, make_coord(bid, _));
    Tensor gO = local_tile(mO, CTAValTileShape{}, make_coord(bid, _));

    G2RTiledCopy g2r_tiled_copy;
    auto g2r_thr_copy = g2r_tiled_copy.get_slice(tid);
    Tensor tXgI = g2r_thr_copy.partition_S(gI);
    Tensor tXrI = make_fragment_like(tXgI);
    Tensor tXgO = g2r_thr_copy.partition_D(gO);
    Tensor tXrO = make_fragment_like(tXgO);

    const int current_row = bid * RowsPerCTA + tid / ThreadsPerRow;
    if (current_row < num_rows) {
      copy(g2r_tiled_copy, tXgI, tXrI);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tXrI); ++i) {
        tXrO(i) = static_cast<ElementDenoiseOut>(
            static_cast<ElementScale>(tXrI(i)) /
            static_cast<ElementScale>(ScaleFactor));
      }
      copy(g2r_tiled_copy, tXrO, tXgO);
    }
  }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {

    int bid = blockIdx.x;

    if (bid < params.num_m_blocks) {
      convert_block(params, true /*convert_AxEBL?*/, bid);
    } else {
      convert_block(params, false /*convert_AxEBL?*/,
                    bid - params.num_m_blocks);
    }
  }
};
}  // namespace pearl