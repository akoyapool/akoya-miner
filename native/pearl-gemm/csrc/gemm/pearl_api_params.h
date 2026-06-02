#pragma once

#include <cuda.h>
#include <cstdint>
#include <vector>
#include "cutlass/numeric_types.h"

struct PearlAPIParams {
  using index_t = int64_t;

  void* __restrict__ ptr_A;                  // ElementIn
  void* __restrict__ ptr_B;                  // ElementIn
  void const* __restrict__ ptr_EAL;          // ElementIn
  void const* __restrict__ ptr_EAL_mma;      // ElementDenoise
  void const* __restrict__ ptr_EAR_R_major;  // ElementIn
  void const* __restrict__ ptr_EBL_R_major;  // ElementIn
  void const* __restrict__ ptr_EAR_K_major;  // ElementIn
  void const* __restrict__ ptr_EBL_K_major;  // ElementIn
  void const* __restrict__ ptr_EBR;          // ElementIn
  void const* __restrict__ ptr_EBR_mma;      // ElementDenoise
  void* __restrict__ ptr_AxEBL_int32;        // int32 to be converted to fp16
  void* __restrict__ ptr_EARxBpEB_int32;     // int32 to be converted to fp16
  void* __restrict__ ptr_AxEBL_mma;          // ElementDenoise_AxEBL for mma
  void* __restrict__ ptr_EARxBpEB_mma;       // ElementDenoise_EARxBpEB for mma
  void* __restrict__ ptr_AxEBL;              // ElementDenoise_AxEBL
  void* __restrict__ ptr_EARxBpEB;           // ElementDenoise_EARxBpEB
  void const* __restrict__ ptr_A_scales;     // ElementScale
  void const* __restrict__ ptr_B_scales;     // ElementScale
  void* __restrict__ ptr_C;                  // ElementOut

  void* __restrict__ ptr_ApEA;  // ElementIn
  void* __restrict__ ptr_BpEB;  // ElementIn

  void* __restrict__ host_signal_header_pinned;
  void* __restrict__ host_signal_sync;

  int m, n, k, r;

  int k_blocks_per_split_noising_A;
  int k_blocks_per_split_noising_B;

  int swizzle;  // for pearl_matmul kernel
  bool swizzle_n_maj;

  // Optional counter for validating inner hash calls (nullptr to disable)
  uint64_t* inner_hash_counter;

  // PoW target and key (uint256, LE word order)
  void const* __restrict__ ptr_pow_target;  // uint32_t[8]
  void const* __restrict__ ptr_pow_key;     // uint32_t[8]
};

struct Noise_gen_params {
  using index_t = int64_t;

  void* __restrict__ ptr_EAL;
  void* __restrict__ ptr_EAL_fp16;
  void* __restrict__ ptr_EAR_R_major;
  void* __restrict__ ptr_EAR_K_major;
  void* __restrict__ ptr_EBL_R_major;
  void* __restrict__ ptr_EBL_K_major;
  void* __restrict__ ptr_EBR;
  void* __restrict__ ptr_EBR_fp16;

  int m, n, k, r;

  void* __restrict__ ptr_key_A;
  void* __restrict__ ptr_key_B;

  // can pass buffer to zero initialize
  // NOTE: option not enabled yet
  void* __restrict__ ptr_aux_buffer;

  int aux_buffer_size;
};
