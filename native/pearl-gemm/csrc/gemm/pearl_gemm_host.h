#pragma once

#include <cstddef>
#include <cute/tensor.hpp>

#include "error_check.hpp"
#include "host_signal_header.hpp"
#include "kernel_traits.hpp"
#include "pearl_api_params.h"
#include "static_switch.h"

#include "collective_epilogue.hpp"
#include "collective_mainloop.hpp"
#include "pearl_gemm_kernel.h"

template <class ElementOut_, typename TileShape_MNKR, int KStages_, int cM = 1,
          int cN = 1, bool Is_Even_M = true, bool Is_Even_N = true,
          bool SkipReduction = false, bool SkipDenoising = false,
          bool EnableDebug = false, bool SkipCStore = false>
void run_pearl_gemm(PearlAPIParams const& params, cudaStream_t stream = 0) {
  using namespace cute;

  static constexpr int KStages = KStages_;
  using ElementIn = int8_t;
  using ElementDenoise = cutlass::half_t;
  using ElementScale = float;
  using ElementOut = ElementOut_;

  auto problem_shape = make_shape(params.m, params.n, params.k, params.r);

  using KTraits =
      pearl::KernelTraits<ElementIn, ElementOut, ElementDenoise, ElementScale,
                          TileShape_MNKR, Is_Even_M, Is_Even_N, cM, cN,
                          SkipReduction, SkipDenoising, KStages, EnableDebug,
                          SkipCStore>;
  using CollectiveEpilogue = pearl::CollectiveEpilogue<KTraits>;
  typename CollectiveEpilogue::Arguments epilogue_args{
      .ptr_C = static_cast<ElementOut*>(params.ptr_C),
      .ptr_A_scales = static_cast<ElementScale const*>(params.ptr_A_scales),
      .ptr_B_scales = static_cast<ElementScale const*>(params.ptr_B_scales),
      .ptr_EAL = {},
      .ptr_EARxBpEB = static_cast<ElementDenoise*>(params.ptr_EARxBpEB_mma),
      .ptr_AxEBL = static_cast<ElementDenoise*>(params.ptr_AxEBL_mma),
      .ptr_EBR = {},
      .problem_shape = problem_shape};
  epilogue_args.ptr_EAL =
      static_cast<ElementDenoise const*>(params.ptr_EAL_mma);
  epilogue_args.ptr_EBR =
      static_cast<ElementDenoise const*>(params.ptr_EBR_mma);
  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments(epilogue_args);

  using CollectiveMainloop = pearl::CollectiveMainloop<KTraits>;

  using ClusterShape = typename KTraits::ClusterShape_MNK;
  // Persistent CTA scheduling is opt-in per KTraits and incompatible with
  // cluster > 1x1 (the persistent scheduler does not provide per-cluster
  // bookkeeping for tma multicast masks).
  static_assert(!KTraits::PersistentCTAs ||
                    (cute::size<0>(ClusterShape{}) == 1 &&
                     cute::size<1>(ClusterShape{}) == 1),
                "PersistentCTAs requires ClusterShape 1x1");
  // The non-SkipDenoising path embeds pipeline.producer_tail() calls inside
  // load_denoise / load_denoise_tail (collective_epilogue.hpp), which are
  // terminal pipeline ops incompatible with multi-tile reuse on persistent
  // CTAs. Restrict PersistentCTAs to the SkipDenoising path (pure miner) for
  // now; supporting the denoise path would require restructuring those tail
  // ops into per-tile-vs-final-tile variants.
  static_assert(!KTraits::PersistentCTAs || KTraits::SkipDenoising,
                "PersistentCTAs currently requires SkipDenoising");
  using Scheduler =
      cute::conditional_t<KTraits::PersistentCTAs,
                          pearl::StaticPersistentTileScheduler,
                          pearl::SingleTileScheduler>;
  int num_blocks_m = cutlass::ceil_div(params.m, KTraits::bM);
  int num_blocks_n = cutlass::ceil_div(params.n, KTraits::bN);
  // round if using clusters
  int num_clusters_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{}));
  int num_clusters_n = cutlass::ceil_div(num_blocks_n, size<1>(ClusterShape{}));
  num_blocks_m = num_clusters_m * size<0>(ClusterShape{});
  num_blocks_n = num_clusters_n * size<1>(ClusterShape{});
  // In sm90 it is more convenient to interpret swizzle in units of clusters rather than blocks
  int swizzle_divisor =
      params.swizzle_n_maj ? size<1>(ClusterShape{}) : size<0>(ClusterShape{});
  int swizzle = cutlass::ceil_div(params.swizzle, swizzle_divisor);
  typename CollectiveMainloop::Arguments mainloop_args{
      .ptr_A = static_cast<ElementIn*>(params.ptr_ApEA),
      .ptr_B = static_cast<ElementIn*>(params.ptr_BpEB),
      .host_signal_header_pinned =
          static_cast<HostSignalHeader*>(params.host_signal_header_pinned),
      .host_signal_sync = static_cast<HostSignalSync*>(params.host_signal_sync),
      .problem_shape = problem_shape,
      .inner_hash_counter = params.inner_hash_counter,
      .ptr_pow_target = static_cast<uint32_t const*>(params.ptr_pow_target),
      .ptr_pow_key = static_cast<uint32_t const*>(params.ptr_pow_key)};
  typename CollectiveMainloop::Params mainloop_params =
      CollectiveMainloop::to_underlying_arguments(mainloop_args);

  typename Scheduler::Arguments scheduler_args = {.num_blocks_m = num_blocks_m,
                                         .num_blocks_n = num_blocks_n,
                                         .num_clusters_m = num_clusters_m,
                                         .num_clusters_n = num_clusters_n,
                                         .swizzle = swizzle,
                                         .swizzle_n_maj = params.swizzle_n_maj};
  typename Scheduler::Params scheduler_params =
      Scheduler::to_underlying_arguments(scheduler_args);
  int device;
  cudaGetDevice(&device);

  void* kernel = (void*)pearl::hopper_gemm_ws<KTraits, Scheduler>;
  int smem_size = sizeof(typename KTraits::SharedStorage);
  if (smem_size >= 48 * 1024) {
    // Query device's max shared memory per block
    int max_smem_per_block;
    cudaDeviceGetAttribute(&max_smem_per_block,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    cudaError_t attr_result = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (attr_result != cudaSuccess) {
      cudaGetLastError();  // Clear error state
      TORCH_CHECK(false,
                  "Failed to set shared memory size. "
                  "Requested: ",
                  smem_size, " bytes (", smem_size / 1024,
                  " KB), "
                  "Device limit: ",
                  max_smem_per_block, " bytes (", max_smem_per_block / 1024,
                  " KB). "
                  "Error: ",
                  cudaGetErrorString(attr_result));
    }
  }

  int multiprocessor_count;
  cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount,
                         device);
  dim3 grid_dims =
      Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);

  static constexpr int ctaSize = KTraits::kNumWarps * 32;
  dim3 block_dims(ctaSize);
  dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}),
                    size<2>(ClusterShape{}));
  cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims,
                                             cluster_dims, smem_size, stream};

  cutlass::launch_kernel_on_cluster(launch_params, kernel, mainloop_params,
                                    epilogue_params, scheduler_params);
}
