#pragma once

#include "cutlass/arch/barrier.h"
#include "cutlass/fast_math.h"

namespace pearl {

struct SwizzleArgs {
  int const num_clusters_m, num_clusters_n, swizzle;
  bool const swizzle_n_maj;
};

struct SwizzleParams {
  int const num_clusters_m, num_maj_swizzle_groups;
  cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod,
      l2_minor_residual_divmod;
  bool const swizzle_n_maj;
};

template <typename SwizzleArgs>
CUTE_HOST_DEVICE static SwizzleParams make_swizzle_params(
    SwizzleArgs const swizzle_args) {
  int const num_clusters_nonmaj = swizzle_args.swizzle_n_maj
                                      ? swizzle_args.num_clusters_m
                                      : swizzle_args.num_clusters_n;
  int const num_clusters_maj = swizzle_args.swizzle_n_maj
                                   ? swizzle_args.num_clusters_n
                                   : swizzle_args.num_clusters_m;
  int const swizzle = swizzle_args.swizzle;
  int const num_maj_remainder = num_clusters_maj % swizzle;
  int const num_maj_swizzle_groups =
      num_clusters_maj / swizzle;  // not counting remainder
  return {
      .num_clusters_m = swizzle_args.num_clusters_m,
      .num_maj_swizzle_groups = num_maj_swizzle_groups,
      .l2_minor_divmod = cutlass::FastDivmod(swizzle),
      .l2_major_divmod = cutlass::FastDivmod(swizzle * num_clusters_nonmaj),
      .l2_minor_residual_divmod =
          cutlass::FastDivmod(num_maj_remainder > 0 ? num_maj_remainder : 1),
      .swizzle_n_maj = swizzle_args.swizzle_n_maj,
  };
}

template <typename SwizzleParams>
CUTE_HOST_DEVICE auto get_coords_from_linear_idx(SwizzleParams const& params,
                                                 int linear_idx) {
  // blocks are interpreted as clusters here
  int l2_mod, l2_quotient, nonmaj_cluster, maj_cluster, l2_maj_cluster;
  // divide by num_m_blocks * swizzle
  l2_quotient = params.l2_major_divmod.divmod(l2_mod, linear_idx);

  if (l2_quotient < params.num_maj_swizzle_groups) {
    // divide by swizzle
    nonmaj_cluster = params.l2_minor_divmod.divmod(l2_maj_cluster, l2_mod);
  } else {
    // divide by num_n_remainder
    nonmaj_cluster =
        params.l2_minor_residual_divmod.divmod(l2_maj_cluster, l2_mod);
  }
  maj_cluster = l2_maj_cluster + l2_quotient * params.l2_minor_divmod.divisor;

  return cute::make_tuple(nonmaj_cluster, maj_cluster);
}

///////////////////////////////////////////////////////////////////////////////

struct SingleTileScheduler {

 public:
  // Host side kernel arguments
  struct Arguments {
    int const num_blocks_m, num_blocks_n, num_clusters_m, num_clusters_n,
        swizzle;
    bool const swizzle_n_maj;
  };

  // Device side kernel params
  using Params = SwizzleParams;

  static Params to_underlying_arguments(Arguments const& args) {
    SwizzleParams swizzle_params = make_swizzle_params<Arguments>(args);
    return swizzle_params;
  }

  static dim3 get_grid_dim(Arguments const& args, int num_sm) {
    return {uint32_t(args.num_blocks_m), uint32_t(args.num_blocks_n), 1};
  }

  struct WorkTileInfo {
    int linear_cluster_idx;
    bool is_valid_tile = false;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const { return is_valid_tile; }

    template <typename ClusterShape>
    CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t> get_block_coord(
        Params const& params) const {
      // swizzle in units of clusters, preserving CTA layout inside each cluster
      auto [ctaid_in_cluster_x, ctaid_in_cluster_y, ctaid_in_cluster_z] =
          block_id_in_cluster();
      auto [nonmaj_cluster, maj_cluster] =
          get_coords_from_linear_idx(params, linear_cluster_idx);
      int m_cluster = params.swizzle_n_maj ? nonmaj_cluster : maj_cluster;
      int n_cluster = params.swizzle_n_maj ? maj_cluster : nonmaj_cluster;
      int m_block = m_cluster * size<0>(ClusterShape{}) + ctaid_in_cluster_x;
      int n_block = n_cluster * size<1>(ClusterShape{}) + ctaid_in_cluster_y;
      return {m_block, n_block, 1};
    }
  };

  CUTLASS_DEVICE
  SingleTileScheduler() {}

  CUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const& params) const {
    // calculate linear cluster index
    auto [cidx, cidy, cidz] = cluster_id_in_grid();
    int linear_cluster_idx = cidx + cidy * params.num_clusters_m;
    return {linear_cluster_idx, true};
  }

  CUTLASS_DEVICE
  void init_consumer() const {}

  CUTLASS_DEVICE
  void prefetch_next_work(Params const& params,
                          WorkTileInfo& current_work) const {}

  CUTLASS_DEVICE
  void broadcast_next_work(WorkTileInfo& current_work) const {}

  template <bool IsProducer = false>
  CUTLASS_DEVICE WorkTileInfo
  get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return {-1, false};
  }
};

///////////////////////////////////////////////////////////////////////////////

// Persistent CTA scheduler: launches `num_sm` CTAs (one per SM) and each CTA
// loops over multiple output tiles via a static stride of gridDim.x. Amortises
// per-tile launch overhead (cp.async pipeline cold-start, smem allocator) over
// many tiles per CTA and lets adjacent tiles within a CTA reuse a warm smem
// pipeline. Compatible only with cluster shape 1x1 (no multicast); for cluster
// > 1x1 the loop body's block_id_in_cluster() / tma multicast masks would need
// per-cluster bookkeeping which this scheduler does not provide.
//
// API mirrors SingleTileScheduler so the kernel template is agnostic:
//   - Arguments includes the swizzle fields (ignored here)
//   - get_initial_work takes Params (ignored — initial tile is blockIdx.x)
//   - WorkTileInfo::get_block_coord is a ClusterShape template (ignored —
//     cluster must be 1x1 so the cluster-relative coord IS the block coord).
class StaticPersistentTileScheduler {

 public:
  // Host side kernel arguments — matches SingleTileScheduler::Arguments so
  // pearl_gemm_host.h can build one Arguments struct and pass to either.
  struct Arguments {
    int const num_blocks_m, num_blocks_n, num_clusters_m, num_clusters_n,
        swizzle;
    bool const swizzle_n_maj;
  };

  // Device side kernel params
  struct Params {
    int total_blocks;
    cutlass::FastDivmod m_block_divmod;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {args.num_blocks_m * args.num_blocks_n,
            cutlass::FastDivmod(args.num_blocks_m)};
  }

  static dim3 get_grid_dim(Arguments const& args, int num_sm) {
    int total_blocks = args.num_blocks_m * args.num_blocks_n;
    // Cap grid at total_blocks for tiny problems where total_blocks < num_sm,
    // otherwise some CTAs would exit immediately on get_initial_work but still
    // pay the launch cost.
    return {uint32_t(total_blocks < num_sm ? total_blocks : num_sm)};
  }

  struct WorkTileInfo {
    int tile_idx;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      return tile_idx < params.total_blocks;
    }

    template <typename ClusterShape>
    CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t> get_block_coord(
        Params const& params) const {
      // Persistent scheduler is restricted to cluster 1x1 — see class comment.
      // Iterate m_block on the inner axis (row-major over the (m,n) grid) so
      // adjacent tile_idx values share the same B column tile, maximizing TMA
      // reuse for B in the producer pipeline.
      int m_block, n_block;
      n_block = params.m_block_divmod.divmod(m_block, tile_idx);
      // Third coord matches SingleTileScheduler's "1" sentinel (batch=1).
      return {m_block, n_block, 1};
    }
  };

  CUTLASS_DEVICE
  StaticPersistentTileScheduler(){};

  CUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const& /*params*/) const {
    return {int(blockIdx.x)};
  }

  CUTLASS_DEVICE
  void init_consumer() const {}

  CUTLASS_DEVICE
  void prefetch_next_work(Params const& params,
                          WorkTileInfo& current_work) const {}

  CUTLASS_DEVICE
  void broadcast_next_work(WorkTileInfo& current_work) const {}

  template <bool IsProducer = false>
  CUTLASS_DEVICE WorkTileInfo
  get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return {current_work.tile_idx + int(gridDim.x)};
  }
};

}  // namespace pearl
