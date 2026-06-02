/*! \file
    \brief
    Noise generation kernel generates two types of matrices:
    (1) Random int8 matrices EAL, EBR, of shape (M, R) and (N, R), with values in
        [-32, 32) (for 2 indices per column). We call these "dense".
    (2) Random sparse int8 matrices EAR and EBL of shape (K, R). Each row (each
        value of R) contains a single 1 and a single -1, with other values equal to 0.
        We call these "sparse".

        To improve noising kernel performance, we support writing them in either
        R-major or K-major format (Torch tensors of shape (K, R) or (R, K)), via the
        pointers EAR_R_major, EAR_K_major, etc. To be precise, noisingA wants
        EAR_R_major and EBL_K_major, while noisingB wants EAR_K_major and EBL_R_major.

    Can also pass int32 or uint32 aux_buffer to zero out.

    \details
    Expected dimensions are:
    - EAL (M R)
    - EAR_R_major (K R)
    - EAR_K_major (R K)
    - EBL_R_major (K R)
    - EBL_K_major (R K)
    - EBR (N R)
    with R=64, 128 currently supported.

    However, strictly speaking the kernel doesn't need the row dimensions of EAR and EBL to agree.

    To *not* construct a matrix, pass nullptr for its pointer.
    The kernel will always set the corresponding number of blocks to 0.
    For example, this can be used if we don't want to construct EAR using seed_A
    since that might be defined as a repeated identity matrix (inference situation).
    However, the kernel always uses seed_{A,B} for generation of E_{A,B}X.
    We need to adopt some other rule if we also want seed_B to be used for EAR (see below).

    Each threadwise invocation of blake3 takes as input:
    - data: 64 byte uint32 array with last 32 bytes = seed_{A,B} and rule (*) below
    - chaining_value: 32 byte uint32 array = key_{A,B}
    - uint32 flags = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT
    - uint32 counter = 0
    and returns
    - raw_hash: 32 byte uint32 array
    which is used to populate 32 contiguous values of one of the noise matrices
    (for sparse matrices, every 4 bytes of raw_hash is used to produce one pair of
    column indices).

    This matches blake3(data, key=key).digest()

    Both noise tensors for A are seeded by seed_A, and both for B are seeded by seed_B.
    To obtain distinct values for each pair of tensors, we also set data according to rule (*):
    - First clear initial segment of 32 bytes (= 8 values).
    - Let r be the global linear index of the 32 byte chunk in the tensor this
      thread will generate.
    - If EAL or EBR, data[0] = r + 1.
    - If EAR or EBL, data[1] = r + 1.
    Note that the offset by 1 is to ensure that row 0 of E_{A,B}L and E_{A,B}R differ.
    Of course, rule (*) can be adjusted to whatever we want.

    Lastly, we fuse zeroing out a auxiliary int32 or uint32 buffer to this kernel.
    For example, we can use this to prepare A_E_BL in inference situation
    since we do atomicAdd for int32 accumulation in downstream noisingA kernel.
    The only constraint is that aux_buffer_size is divisible by 4 (= 16 byte aligned).
    This is checked in the API.
*/

#pragma once

#include "blake3/blake3.cuh"
#include "blake3/blake3_constants.hpp"
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
class NoiseGenerationKernel {

 public:
  static constexpr int NOISE_ABS_MAX = 128;
  static constexpr int PERM_IDXS_PER_COL = 2;
  // random noise in range [-64, 64) if 1 index, [-32, 32) if 2 indices.
  static constexpr int NOISE_RANGE = NOISE_ABS_MAX / PERM_IDXS_PER_COL;
  static_assert(cutlass::is_pow2<NOISE_RANGE>::value);

  // Type Aliases
  static constexpr uint32_t MaxThreadsPerBlock = NumThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 3;

  static constexpr int NumWarps = NumThreads / 32;

  // Each thread produces 32 bytes = 32 vals.
  // For R = 128 and 128 threads, we'd cover 32 rows with 4 threads per row.
  static constexpr int ValsPerThread = 32;
  static_assert(R % ValsPerThread == 0);
  static constexpr int ThreadsPerRow = R / ValsPerThread;
  //the number of rows done per CTA
  static constexpr int BlockSize = NumThreads / ThreadsPerRow;

  // For sparse tensors, KValsPerThreadIdx is number of K values that each thread processes.
  // Each thread does one blake3 hash, to generate CHAINING_VALUE_SIZE_ random bytes, we then use every
  // uint32_t to produce a pair of column indices.
  static constexpr int KValsPerThreadIdx =
      blake3::CHAINING_VALUE_SIZE_U32;  // 8

  // Random matrices: (M, R) or (N, R)
  using TileShape = Shape<Int<BlockSize>, Int<R>>;
  using SmemLayoutUnswizzled = Layout<TileShape, LayoutRight::Apply<TileShape>>;
  // NOTE: bank-conflicted but consistently faster than with swizzling
  using SmemLayout = SmemLayoutUnswizzled;
  // using SmemLayout = decltype(composition(Swizzle<1, 4, 3>{}, SmemLayoutUnswizzled{}));

  // RMEM -> SMEM preparing coalesced store
  // Each thread holds 32B, so can use 16B vectorized copy
  // raw_hash is uint32_t, so adjust SMEM shape accordingly
  using R2SSmemLayout = decltype(recast_layout<int8_t, uint32_t>(SmemLayout{}));
  using R2SThrShape = Shape<Int<BlockSize>, Int<ThreadsPerRow>>;
  using R2SThrLayout = Layout<R2SThrShape, LayoutRight::Apply<R2SThrShape>>;
  using R2SValLayout = Layout<Shape<_1, Int<ValsPerThread / sizeof(uint32_t)>>>;
  using R2SCopyAtom = cute::Copy_Atom<UniversalCopy<uint128_t>, uint32_t>;
  using R2STiledCopy =
      decltype(make_tiled_copy(R2SCopyAtom{}, R2SThrLayout{}, R2SValLayout{}));

  // SMEM -> RMEM -> GMEM
  // Each thread loads 16B for 16B vectorized store
  using StoreCopyAtom = cute::Copy_Atom<UniversalCopy<uint128_t>, int8_t>;
  static constexpr int GmemElemsPerStore = sizeof(uint128_t);
  static constexpr int GmemThreadsR = R / GmemElemsPerStore;  // e.g. 128/16 = 8
  static_assert(NumThreads % GmemThreadsR == 0);
  using StoreThrShape = Shape<Int<NumThreads / GmemThreadsR>,
                              Int<GmemThreadsR>>;  // e.g. (NumWarps * 4, 8)
  using StoreThrLayout =
      Layout<StoreThrShape, LayoutRight::Apply<StoreThrShape>>;
  using StoreValLayout = Layout<Shape<_1, Int<GmemElemsPerStore>>>;  // (1, 16)
  using StoreTiledCopy = decltype(make_tiled_copy(
      StoreCopyAtom{}, StoreThrLayout{}, StoreValLayout{}));

  // RMEM -> SMEM -> GMEM for fp16
  //  we use the same thread and val layouts as the previous s2r copy for consistency
  using R2SThrShapeFP16 = StoreThrShape;
  using R2SThrLayoutFP16 =
      Layout<R2SThrShapeFP16, LayoutRight::Apply<R2SThrShapeFP16>>;
  using R2SValLayoutFP16 = StoreValLayout;
  using R2SCopyAtomFP16 = cute::Copy_Atom<UniversalCopy<uint128_t>, half_t>;
  using R2STiledCopyFP16 = decltype(make_tiled_copy(
      R2SCopyAtomFP16{}, R2SThrLayoutFP16{}, R2SValLayoutFP16{}));

  static constexpr int GmemElemsPerStoreFP16 =
      sizeof(uint128_t) / sizeof(half_t);  // 8
  static constexpr int GmemThreadsRFP16 =
      R / GmemElemsPerStoreFP16;  // 128 / 8 = 16
  using StoreCopyAtomFP16 =
      cute::Copy_Atom<UniversalCopy<uint128_t>, cutlass::half_t>;
  using StoreThrShapeFP16 = Shape<Int<NumThreads / GmemThreadsRFP16>,  // 8, 16
                                  Int<GmemThreadsRFP16>>;
  using StoreThrLayoutFP16 =
      Layout<StoreThrShapeFP16, LayoutRight::Apply<StoreThrShapeFP16>>;
  using StoreValLayoutFP16 =
      Layout<Shape<_1, Int<GmemElemsPerStoreFP16>>>;  // (1, 8)
  using StoreTiledCopyFP16 = decltype(make_tiled_copy(
      StoreCopyAtomFP16{}, StoreThrLayoutFP16{}, StoreValLayoutFP16{}));

  using StoreTiledCopySparseRMajor = StoreTiledCopy;
  // single 128B cache line; want each thread to repeat in the K direction as much as possible to reduce runtime index calc and register usage
  static constexpr int GmemThreadsK = 8;
  // KBlockSizeIdx / GmemElemsPerStore;  // e.g. 128/16 = 8
  static_assert(NumThreads % GmemThreadsK == 0);
  using StoreThrShapeSparseKMajor = Shape<Int<NumThreads / GmemThreadsK>,
                                          Int<GmemThreadsK>>;  // e.g. (16, 8)
  using StoreThrLayoutSparseKMajor =
      Layout<StoreThrShapeSparseKMajor,
             LayoutRight::Apply<StoreThrShapeSparseKMajor>>;
  using StoreValLayoutSparseKMajor =
      Layout<Shape<_1, Int<GmemElemsPerStore>>>;  // (1, 16)
  using StoreTiledCopySparseKMajor =
      decltype(make_tiled_copy(StoreCopyAtom{}, StoreThrLayoutSparseKMajor{},
                               StoreValLayoutSparseKMajor{}));

  // Sparse matrices: (K, R) / (R, K)
  static constexpr int KBlockSizeIdx = KValsPerThreadIdx * NumThreads;
  // 2D tensors for direct store to (R, K) or (K, R) sparse matrices
  using TileShapeSparse = Shape<Int<KBlockSizeIdx>, Int<R>>;
  using TileShapeSparseKMajor = Shape<Int<R>, Int<KBlockSizeIdx>>;

  // Each thread owns KValsPerThreadIdx adjacent values of K
  using ThrTileShapeSparse = Shape<Int<KValsPerThreadIdx>, Int<R>>;
  using ThrTileShapeSparseKMajor = Shape<Int<R>, Int<KValsPerThreadIdx>>;

  static constexpr int AlignmentDense =
      cutlass::detail::alignment_for_swizzle(SmemLayout{});
  static constexpr int Alignment = AlignmentDense;

  // Aux buffer is uint32
  // For equivalent out-byte traffic with other blocks, 8 values total per thread
  static constexpr int BlockSizeAux =
      size(TileShape{}) / sizeof(uint32_t);  // e.g. 8*NumThreads
  using TileShapeAux = Shape<Int<BlockSizeAux>>;

  using CopyAtomAux =
      cute::Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, uint32_t>;
  using ThrLayoutAux = Layout<Shape<Int<NumThreads>>>;
  using ValLayoutAux = Layout<Shape<_4>>;  // fix to 16 bytes
  using GmemTiledCopyAux =
      decltype(make_tiled_copy(CopyAtomAux{}, ThrLayoutAux{}, ValLayoutAux{}));

  struct SharedStorage : cute::aligned_struct<Alignment> {
    union {
      cute::array_aligned<half_t, cute::cosize_v<SmemLayout>, AlignmentDense>
          smem_fp16_dense;
      cute::array_aligned<int8_t, cute::cosize_v<SmemLayout>, AlignmentDense>
          smem_dense;
    };
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    // Pointers to E_{A,B}{L,R}
    int8_t* const ptr_EAL;
    half_t* const ptr_EAL_fp16;
    int8_t* const ptr_EAR_R_major;
    int8_t* const ptr_EAR_K_major;
    int8_t* const ptr_EBL_R_major;
    int8_t* const ptr_EBL_K_major;
    int8_t* const ptr_EBR;
    half_t* const ptr_EBR_fp16;
    int num_rows_EAL;  // m
    int length_EAR;    // k
    int length_EBL;    // k
    int num_rows_EBR;  // n
    // 32 bytes of manual key data used for controlling the generation
    uint8_t const* const ptr_key_A;
    uint8_t const* const ptr_key_B;
    // can pass optional buffer to zero initialize
    // assume 16 byte aligned
    uint32_t* const ptr_aux_buffer;
    int aux_buffer_size;
  };

  // Kernel entry point API
  struct Params {
    int8_t* const ptr_EAL;
    half_t* const ptr_EAL_fp16;
    int8_t* const ptr_EAR_R_major;
    int8_t* const ptr_EAR_K_major;
    int8_t* const ptr_EBL_R_major;
    int8_t* const ptr_EBL_K_major;
    int8_t* const ptr_EBR;
    half_t* const ptr_EBR_fp16;
    int length_EAR;
    int length_EBL;
    int num_rows_EAL;
    int num_rows_EBR;
    uint8_t const* const ptr_key_A;
    uint8_t const* const ptr_key_B;
    uint32_t* const ptr_aux_buffer;
    int aux_buffer_size;
    int num_blocks_EAL;
    int num_blocks_EAR_R_major;
    int num_blocks_EAR_K_major;
    int num_blocks_EBL_R_major;
    int num_blocks_EBL_K_major;
    int num_blocks_EBR;
    int num_blocks_zero;
  };

  // Convert to underlying arguments.
  static Params to_underlying_arguments(Arguments const& args) {
    int num_blocks_EAL = ceil_div(args.num_rows_EAL, BlockSize);
    int num_blocks_EAR = ceil_div(args.length_EAR, KBlockSizeIdx);
    int num_blocks_EBL = ceil_div(args.length_EBL, KBlockSizeIdx);
    int num_blocks_EBR = ceil_div(args.num_rows_EBR, BlockSize);
    int num_blocks_zero = ceil_div(args.aux_buffer_size, BlockSizeAux);

    int num_blocks_EAR_R_major = args.ptr_EAR_R_major ? num_blocks_EAR : 0;
    int num_blocks_EAR_K_major = args.ptr_EAR_K_major ? num_blocks_EAR : 0;
    int num_blocks_EBL_R_major = args.ptr_EBL_R_major ? num_blocks_EBL : 0;
    int num_blocks_EBL_K_major = args.ptr_EBL_K_major ? num_blocks_EBL : 0;
    return {.ptr_EAL = args.ptr_EAL,
            .ptr_EAL_fp16 = args.ptr_EAL_fp16,
            .ptr_EAR_R_major = args.ptr_EAR_R_major,
            .ptr_EAR_K_major = args.ptr_EAR_K_major,
            .ptr_EBL_R_major = args.ptr_EBL_R_major,
            .ptr_EBL_K_major = args.ptr_EBL_K_major,
            .ptr_EBR = args.ptr_EBR,
            .ptr_EBR_fp16 = args.ptr_EBR_fp16,
            .length_EAR = args.length_EAR,
            .length_EBL = args.length_EBL,
            .num_rows_EAL = args.num_rows_EAL,
            .num_rows_EBR = args.num_rows_EBR,
            .ptr_key_A = args.ptr_key_A,
            .ptr_key_B = args.ptr_key_B,
            .ptr_aux_buffer = args.ptr_aux_buffer,
            .aux_buffer_size = args.aux_buffer_size,
            .num_blocks_EAL = num_blocks_EAL,
            .num_blocks_EAR_R_major = num_blocks_EAR_R_major,
            .num_blocks_EAR_K_major = num_blocks_EAR_K_major,
            .num_blocks_EBL_R_major = num_blocks_EBL_R_major,
            .num_blocks_EBL_K_major = num_blocks_EBL_K_major,
            .num_blocks_EBR = num_blocks_EBR,
            .num_blocks_zero = num_blocks_zero};
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3 get_grid_shape(Params const& params) {
    const int num_blocks =
        params.num_blocks_EAL + params.num_blocks_EAR_R_major +
        params.num_blocks_EAR_K_major + params.num_blocks_EBL_R_major +
        params.num_blocks_EBL_K_major + params.num_blocks_EBR +
        params.num_blocks_zero;
    return dim3(num_blocks, 1, 1);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE void clear_aux_tensor(Params const& params, int bid, int tid) {
    int aux_idx_min = bid * BlockSizeAux;

    Tensor mAux = make_tensor(make_gmem_ptr(params.ptr_aux_buffer),
                              make_shape(params.aux_buffer_size));
    Tensor gAux = local_tile(mAux, TileShapeAux{}, make_coord(bid));
    Tensor cAux = make_identity_tensor(TileShapeAux{});

    GmemTiledCopyAux gmem_tiled_copy_aux;
    auto gmem_thr_copy_aux = gmem_tiled_copy_aux.get_thread_slice(tid);
    auto tXgAux = gmem_thr_copy_aux.partition_D(gAux);
    auto tXcAux = gmem_thr_copy_aux.partition_D(cAux);
    static_assert(rank(tXgAux) == 2);
    static_assert(size<0>(tXgAux) == 4);

    Tensor v = make_tensor<uint32_t>(Int<size<0>(tXgAux)>{});
    clear(v);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(tXgAux); ++i) {
      int aux_idx = get<0>(tXcAux(_0{}, i));
      if (aux_idx + aux_idx_min < params.aux_buffer_size) {
        copy(gmem_tiled_copy_aux, v, tXgAux(_, i));
      }
    }
  }

  template <typename TensorKey, typename TensorSeed, typename TensorHash>
  CUTLASS_DEVICE void generate_hash(TensorSeed const& g_seed,
                                    TensorKey const& g_key,
                                    TensorHash& raw_hash, int thread_coord,
                                    bool write_K) {
    Tensor g_seed_vec4 = recast<int4>(g_seed);
    Tensor g_key_vec4 = recast<int4>(g_key);
    Tensor data = make_tensor<uint32_t>(Int<blake3::MSG_BLOCK_SIZE_U32>{});
    Tensor data_vec4 = recast<int4>(data);
    clear(data);
    Tensor chaining_value =
        make_tensor<uint32_t>(Int<blake3::CHAINING_VALUE_SIZE_U32>{});
    Tensor chaining_value_vec4 = recast<int4>(chaining_value);
    static_assert(size(g_key_vec4) == 2);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(g_key_vec4); ++i) {
      chaining_value_vec4[i] = g_key_vec4[i];
    }

    int constexpr message_offset =
        (blake3::MSG_BLOCK_SIZE - blake3::KEY_SIZE) / sizeof(int4);
    static_assert(size(g_seed_vec4) == 2);
    static_assert(size(data_vec4) == size(g_seed_vec4) + message_offset);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(g_seed_vec4); ++i) {
      data_vec4[i + message_offset] = g_seed_vec4[i];
    }
    // use data[0] for dense tensors and data[1] for sparse tensors
    data[0] = !write_K ? thread_coord : 0;
    data[1] = write_K ? thread_coord : 0;

    // Copy chaining_value to raw_hash as blake3::compress_msg_block_u32 uses it as both input and output
    copy(chaining_value, raw_hash);
    blake3::compress_msg_block_u32(data, raw_hash,
                                   blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);
  }

  template <typename TensorHash, typename TensorIndex>
  CUTLASS_DEVICE void populate_reg_indices(Params const& params,
                                           TensorHash const& raw_hash,
                                           TensorIndex& rO_8b, int bid,
                                           int tid) {
    // populate reg-backed tensor using the blake3 hash result and algorithm where
    //  each 4 bytes u of raw_hash is used to produce a pair of indices r0 and r1
    // - r0 is the lowest log2(R) bits of u
    // - x = (1 + (high 32 bits of 64-bit result of (R-1)*u))
    // - r1 = r0 ^ x
    // It can be shown that x is nonzero, so r1 is always distinct from r0
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size(raw_hash); ++k) {
      uint32_t u = raw_hash(k);
      uint8_t r0 = u & (R - 1);
      uint8_t r1 = r0 ^ (1 + __umulhi(static_cast<uint32_t>(R - 1), u));
      rO_8b(k, _0{}) = r0;
      rO_8b(k, _1{}) = r1;
    }
  }

  template <typename TensorHash>
  CUTLASS_DEVICE void r2s_copy_out_dense(Params const& params,
                                         SharedStorage& shared_storage,
                                         TensorHash const& raw_hash, int tid) {
    Tensor sO_32b =
        make_tensor(make_smem_ptr<uint32_t>(shared_storage.smem_dense.data()),
                    R2SSmemLayout{});

    R2STiledCopy r2s_tiled_copy;
    ThrCopy r2s_thr_copy = r2s_tiled_copy.get_slice(tid);
    Tensor tR2SsO_32b = r2s_thr_copy.partition_D(sO_32b);
    Tensor tR2SrO_32b = make_tensor(raw_hash.data(), tR2SsO_32b.layout());
    copy(r2s_tiled_copy, tR2SrO_32b, tR2SsO_32b);
  }

  template <typename TensorIndex>
  CUTLASS_DEVICE void store_out_sparse(Params const& params,
                                       SharedStorage& shared_storage,
                                       TensorIndex const& rO_8b, bool write_A,
                                       bool write_sparse_R_major, int bid,
                                       int tid) {
    int const length = write_A ? params.length_EAR : params.length_EBL;
    if (write_sparse_R_major) {
      auto ptr_out = write_A ? params.ptr_EAR_R_major : params.ptr_EBL_R_major;
      Tensor mO = make_tensor(make_gmem_ptr(ptr_out),
                              make_shape(length, Int<R>{}), LayoutRight{});
      // gO: (bK, R)
      Tensor gO = local_tile(mO, TileShapeSparse{}, make_coord(bid, _0{}));
      Tensor cO = local_tile(make_identity_tensor(mO.shape()),
                             TileShapeSparse{}, make_coord(bid, _0{}));

      // tOgO: (KValsPerThreadIdx, R)
      Tensor tOgO = local_tile(gO, ThrTileShapeSparse{}, make_coord(tid, _0{}));
      Tensor tOcO = local_tile(cO, ThrTileShapeSparse{}, make_coord(tid, _0{}));

      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<0>(rO_8b); ++k) {
        int i = get<0>(tOcO(k, _0{}));
        if (i < length) {
          int r0 = static_cast<int>(rO_8b(k, _0{}));
          int r1 = static_cast<int>(rO_8b(k, _1{}));
          tOgO(k, r0) = 1;
          tOgO(k, r1) = -1;
        }
      }
    } else {
      auto ptr_out = write_A ? params.ptr_EAR_K_major : params.ptr_EBL_K_major;
      Tensor mO = make_tensor(make_gmem_ptr(ptr_out),
                              make_shape(Int<R>{}, length), LayoutRight{});

      // gO: (R, bK)
      Tensor gO =
          local_tile(mO, TileShapeSparseKMajor{}, make_coord(_0{}, bid));
      Tensor cO = local_tile(make_identity_tensor(mO.shape()),
                             TileShapeSparseKMajor{}, make_coord(_0{}, bid));

      // tOgO: (R, KValsPerThreadIdx)
      Tensor tOgO =
          local_tile(gO, ThrTileShapeSparseKMajor{}, make_coord(_0{}, tid));
      Tensor tOcO =
          local_tile(cO, ThrTileShapeSparseKMajor{}, make_coord(_0{}, tid));

      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<0>(rO_8b); ++k) {
        int i = get<1>(tOcO(_0{}, k));
        if (i < length) {
          int r0 = static_cast<int>(rO_8b(k, _0{}));
          int r1 = static_cast<int>(rO_8b(k, _1{}));
          tOgO(r0, k) = 1;
          tOgO(r1, k) = -1;
        }
      }
    }
  }

  template <int ScaleFactor, typename TensorIn, typename TensorOut,
            typename TensorPred>
  CUTLASS_DEVICE void store_out_fp16(SharedStorage& shared_storage,
                                     TensorIn const& tOrO, TensorOut& gO,
                                     TensorPred const& cO, int const tid,
                                     int const row_limit) {

    Tensor sO = make_tensor(
        make_smem_ptr<half_t>(shared_storage.smem_fp16_dense.data()),
        SmemLayout{});

    Tensor rO_fp16 = make_fragment_like<cutlass::half_t>(tOrO);

    R2STiledCopyFP16 r2s_tiled_copy_fp16;
    ThrCopy r2s_thr_copy_fp16 = r2s_tiled_copy_fp16.get_slice(tid);
    Tensor tR2SsO = r2s_thr_copy_fp16.partition_D(sO);
    Tensor tR2SrO = r2s_thr_copy_fp16.retile_S(rO_fp16);

    StoreTiledCopyFP16 store_tiled_copy_fp16;
    ThrCopy store_thr_copy_fp16 = store_tiled_copy_fp16.get_slice(tid);
    Tensor tOgO_fp16 = store_thr_copy_fp16.partition_D(gO);
    Tensor tOsO_fp16 = store_thr_copy_fp16.partition_S(sO);
    Tensor tOcO_fp16 = store_thr_copy_fp16.partition_D(cO);

    // RMEM -> RMEM convert
    for (int i = 0; i < size(tOrO); ++i) {
      rO_fp16[i] = static_cast<cutlass::half_t>(tOrO[i] * ScaleFactor);
    }

    // RMEM -> SMEM
    copy(r2s_tiled_copy_fp16, tR2SrO, tR2SsO);

    Tensor tOrO_fp16 = make_fragment_like<cutlass::half_t>(tOsO_fp16);
    __syncthreads();
    copy(store_tiled_copy_fp16, tOsO_fp16, tOrO_fp16);
    // SMEM -> GMEM
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(tOgO_fp16); ++i) {
      int row = get<0>(tOcO_fp16(_0{}, i, _0{}));
      if (row < row_limit) {
        copy(store_tiled_copy_fp16, tOrO_fp16(_, i, _), tOgO_fp16(_, i, _));
      }
    }
  }

  CUTLASS_DEVICE void store_out_dense(Params const& params,
                                      SharedStorage& shared_storage,
                                      bool write_A, int bid, int tid) {
    auto ptr_out = write_A ? params.ptr_EAL : params.ptr_EBR;
    int num_rows = write_A ? params.num_rows_EAL : params.num_rows_EBR;
    Tensor mO = make_tensor(make_gmem_ptr(ptr_out),
                            make_shape(num_rows, Int<R>{}), LayoutRight{});
    Tensor gO = local_tile(mO, TileShape{}, make_coord(bid, _0{}));
    Tensor cO = make_identity_tensor(TileShape{});
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_dense.data()),
                            SmemLayout{});

    int row_min = bid * BlockSize;

    // SMEM -> RMEM copy
    StoreTiledCopy store_tiled_copy;
    ThrCopy store_thr_copy = store_tiled_copy.get_slice(tid);
    Tensor tOsO = store_thr_copy.partition_S(sO);
    Tensor tOrO = make_fragment_like<int8_t>(tOsO);
    copy(store_tiled_copy, tOsO, tOrO);

    // Random matrices should be in range [-NOISE_RANGE/2, NOISE_RANGE/2)
    // We add NOISE_ABS_MAX first to get a nonnegative number - otherwise % could give a negative number.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tOrO); ++i) {
      tOrO(i) = static_cast<int8_t>(
          ((static_cast<int32_t>(tOrO(i)) + NOISE_ABS_MAX) % NOISE_RANGE) -
          (NOISE_RANGE / 2));
    }

    // RMEM -> GMEM copy, only need to predicate in MN direction
    Tensor tOgO = store_thr_copy.partition_D(gO);
    Tensor tOcO = store_thr_copy.partition_D(cO);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(tOgO); ++i) {
      int row = get<0>(tOcO(_0{}, i, _0{}));
      if (row < num_rows - row_min) {
        copy(store_tiled_copy, tOrO(_, i, _), tOgO(_, i, _));
      }
    }

    // Optionally write EAL/EBR in fp16 to save on conversion at denoising
    if (!write_A && params.ptr_EBR_fp16) {
      Tensor mO_fp16 =
          make_tensor(make_gmem_ptr(params.ptr_EBR_fp16),
                      make_shape(num_rows, Int<R>{}), LayoutRight{});
      Tensor gO_fp16 = local_tile(mO_fp16, TileShape{}, make_coord(bid, _0{}));
      // See pearl_gemm_constants.hpp for more info on scale factors
      static constexpr int ScaleFactor = pearl::kEBRScaleFactorDenoise;
      store_out_fp16<ScaleFactor>(shared_storage, tOrO, gO_fp16, cO, tid,
                                  num_rows - row_min);

    } else if (write_A && params.ptr_EAL_fp16) {
      Tensor mO_fp16 =
          make_tensor(make_gmem_ptr(params.ptr_EAL_fp16),
                      make_shape(num_rows, Int<R>{}), LayoutRight{});
      Tensor gO_fp16 = local_tile(mO_fp16, TileShape{}, make_coord(bid, _0{}));
      static constexpr int ScaleFactor = pearl::kEALScaleFactorDenoise;
      store_out_fp16<ScaleFactor>(shared_storage, tOrO, gO_fp16, cO, tid,
                                  num_rows - row_min);
    }
  }

  CUTLASS_DEVICE void zero_out_sparse_K(Params const& params, bool write_A,
                                        int bid, int tid) {

    int const length = write_A ? params.length_EAR : params.length_EBL;
    int const length_min = bid * KBlockSizeIdx;
    auto ptr_out = write_A ? params.ptr_EAR_K_major : params.ptr_EBL_K_major;
    Tensor mO = make_tensor(make_gmem_ptr(ptr_out),
                            make_shape(Int<R>{}, length), LayoutRight{});
    Tensor gO = local_tile(mO, TileShapeSparseKMajor{}, make_coord(_0{}, bid));
    Tensor cO = make_identity_tensor(TileShapeSparseKMajor{});

    // RMEM -> GMEM copy
    // Predicate in K direction, assume K % 16 == 0
    StoreTiledCopySparseKMajor store_tiled_copy;
    ThrCopy store_thr_copy = store_tiled_copy.get_slice(tid);
    Tensor tOgO = store_thr_copy.partition_D(gO);
    Tensor tOcO = store_thr_copy.partition_D(cO);
    Tensor tOrO = make_fragment_like<int8_t>(tOgO);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(tOgO); ++i) {
      for (int j = 0; j < size<2>(tOgO); ++j) {
        int k = get<1>(tOcO(_0{}, i, j));
        if (k < length - length_min) {
          clear(tOrO(_, i, _));
          copy(store_tiled_copy, tOrO(_, _0{}, _0{}), tOgO(_, i, j));
        }
      }
    }
  }

  CUTLASS_DEVICE void zero_out_sparse_R(Params const& params, bool write_A,
                                        int bid, int tid) {

    int const length = write_A ? params.length_EAR : params.length_EBL;
    int const length_min = bid * KBlockSizeIdx;
    auto ptr_out = write_A ? params.ptr_EAR_R_major : params.ptr_EBL_R_major;
    Tensor mO = make_tensor(make_gmem_ptr(ptr_out),
                            make_shape(length, Int<R>{}), LayoutRight{});
    Tensor gO = local_tile(mO, TileShapeSparse{}, make_coord(bid, _0{}));
    Tensor cO = make_identity_tensor(TileShapeSparse{});

    // RMEM -> GMEM copy
    // Predicate in K direction, assume K % 16 == 0
    StoreTiledCopySparseRMajor store_tiled_copy;
    ThrCopy store_thr_copy = store_tiled_copy.get_slice(tid);

    Tensor tOgO = store_thr_copy.partition_D(gO);
    Tensor tOcO = store_thr_copy.partition_D(cO);
    Tensor tOrO = make_fragment_like<int8_t>(tOgO);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(tOgO); ++i) {
      int k = get<0>(tOcO(_0{}, i, _0{}));
      if (k < length - length_min) {
        clear(tOrO(_, i, _));
        copy(store_tiled_copy, tOrO(_, i, _), tOgO(_, i, _));
      }
    }
  }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int const bid = blockIdx.x;
    int const tid = threadIdx.x;

    /* Division of labor between CTAs:
       CTAs are assigned in the order
           EAL, EAR_R_major, EAR_K_major, EBR, EBL_R_major, EBL_K_major, clear_aux.
       Each thread participating in noise generation produces 32 bytes via blake3.
       For dense matrices, these are transparently written as 32 int8 values per
       thread, so 4096 values per CTA (as a tile of shape MNxR = 32x128 or 64x64).
       For sparse matrices, these are used to generate 8 values
       of K per thread, so 1024 values of K per CTA.
       CTAs clearing the aux tensor clear 32 bytes/thread = 8 uint32 values/thread
       = 1024 uint32 values/CTA.
     */

    int bid_offset = 0;
    int num_blocks_A = params.num_blocks_EAL + params.num_blocks_EAR_R_major +
                       params.num_blocks_EAR_K_major;
    int num_blocks_B = params.num_blocks_EBR + params.num_blocks_EBL_R_major +
                       params.num_blocks_EBL_K_major;
    if (bid >= num_blocks_A + num_blocks_B) {
      bid_offset = num_blocks_A + num_blocks_B;
      int bid_in_category = bid - bid_offset;

      clear_aux_tensor(params, bid_in_category, tid);
    } else {
      bool const write_A = bid < num_blocks_A;
      bool write_dense = false;
      bool write_sparse_R_major = false;
      if (write_A) {
        write_dense = bid < params.num_blocks_EAL;
        if (!write_dense) {
          bid_offset += params.num_blocks_EAL;
        }
        write_sparse_R_major =
            !write_dense && (bid - bid_offset < params.num_blocks_EAR_R_major);
        if (!write_dense && !write_sparse_R_major) {
          bid_offset += params.num_blocks_EAR_R_major;
        }
      } else {
        bid_offset += num_blocks_A;
        write_dense = bid - bid_offset < params.num_blocks_EBR;
        if (!write_dense) {
          bid_offset += params.num_blocks_EBR;
        }
        write_sparse_R_major =
            !write_dense && (bid - bid_offset < params.num_blocks_EBL_R_major);
        if (!write_dense && !write_sparse_R_major) {
          bid_offset += params.num_blocks_EBL_R_major;
        }
      }
      bool const write_sparse = !write_dense;
      bool const write_K = write_sparse;

      int const bid_in_category = bid - bid_offset;

      constexpr uint8_t seed_A[blake3::CHAINING_VALUE_SIZE] = "A_tensor";
      constexpr uint8_t seed_B[blake3::CHAINING_VALUE_SIZE] = "B_tensor";

      auto ptr_seed = write_A ? seed_A : seed_B;
      auto ptr_key = write_A ? params.ptr_key_A : params.ptr_key_B;

      Tensor g_seed = make_tensor(make_gmem_ptr(ptr_seed),
                                  Int<blake3::CHAINING_VALUE_SIZE>{});  // uint8
      Tensor g_key = make_tensor(make_gmem_ptr(ptr_key),
                                 Int<blake3::KEY_SIZE>{});  // uint8

      // 32B output from blake3 (CHAINING_VALUE_SIZE_U32 = 8 uint32_t)
      Tensor raw_hash = make_tensor<uint32_t>(
          Layout<Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>>>{});

      // thread_coord used to give each thread a different seed
      // Starts at 1 and counts through all threads working on a given tensor
      uint32_t thread_coord = bid_in_category * NumThreads + tid + 1;
      generate_hash(g_seed, g_key, raw_hash, thread_coord, write_K);
      if (write_dense) {
        r2s_copy_out_dense(params, shared_storage, raw_hash, tid);
        __syncthreads();
        store_out_dense(params, shared_storage, write_A, bid_in_category, tid);
      } else {
        if (write_sparse_R_major) {
          zero_out_sparse_R(params, write_A, bid_in_category, tid);
        } else {
          zero_out_sparse_K(params, write_A, bid_in_category, tid);
        }
        Tensor rO_8b = make_tensor<uint8_t>(
            Layout<Shape<Int<KValsPerThreadIdx>, Int<PERM_IDXS_PER_COL>>,
                   Stride<Int<PERM_IDXS_PER_COL>, _1>>{});  // (8,2)
        populate_reg_indices(params, raw_hash, rO_8b, bid_in_category, tid);
        __syncthreads();
        store_out_sparse(params, shared_storage, rO_8b, write_A,
                         write_sparse_R_major, bid_in_category, tid);
      }
    }
  }
};

}  // namespace pearl
