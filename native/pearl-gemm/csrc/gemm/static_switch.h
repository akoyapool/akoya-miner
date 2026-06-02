#pragma once
#include <c10/util/Exception.h>
#include <cstdio>

// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  do {                                          \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      __VA_ARGS__;                              \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      __VA_ARGS__;                              \
    }                                           \
  } while (0)

#define FALSE_SWITCH(COND, CONST_NAME, ...)   \
  do {                                        \
    constexpr static bool CONST_NAME = false; \
    __VA_ARGS__;                              \
  } while (0)

#define NUM_THREADS_SWITCH(NUM_THREADS_VAR, NUM_THREADS_CONST, ...) \
  do {                                                              \
    if (NUM_THREADS_VAR == 32) {                                    \
      static constexpr int NUM_THREADS_CONST = 32;                  \
      __VA_ARGS__;                                                  \
    } else if (NUM_THREADS_VAR == 64) {                             \
      static constexpr int NUM_THREADS_CONST = 64;                  \
      __VA_ARGS__;                                                  \
    } else if (NUM_THREADS_VAR == 128) {                            \
      static constexpr int NUM_THREADS_CONST = 128;                 \
      __VA_ARGS__;                                                  \
    }                                                               \
  } while (0)

#ifndef DISABLE_SKIP_REDUCTION
#define SKIP_REDUCTION_SWITCH BOOL_SWITCH
#else
#define SKIP_REDUCTION_SWITCH FALSE_SWITCH
#endif

#ifndef DISABLE_SKIP_NOISING
#define SKIP_NOISING_SWITCH BOOL_SWITCH
#else
#define SKIP_NOISING_SWITCH FALSE_SWITCH
#endif

#ifndef DISABLE_SKIP_DENOISING
#define SKIP_DENOISING_SWITCH BOOL_SWITCH
#else
#define SKIP_DENOISING_SWITCH FALSE_SWITCH
#endif

#ifndef DISABLE_DEBUG_MODE
#define DEBUG_MODE_SWITCH BOOL_SWITCH
#else
#define DEBUG_MODE_SWITCH FALSE_SWITCH
#endif

#ifndef DISABLE_SKIP_C_STORE
#define SKIP_C_STORE_SWITCH BOOL_SWITCH
#else
#define SKIP_C_STORE_SWITCH FALSE_SWITCH
#endif

#define MATMUL_CONFIG_OPTION(BM_VAR, BN_VAR, BK_VAR, R_VAR, STAGES_VAR,     \
                             cM_VAR, cN_VAR, BM_VAL, BN_VAL, BK_VAL, R_VAL, \
                             STAGES_VAL, cM_VAL, cN_VAL, ...)               \
  if (R_VAR == R_VAL && BM_VAR == BM_VAL && BN_VAR == BN_VAL &&             \
      BK_VAR == BK_VAL && STAGES_VAR == STAGES_VAL && cM_VAR == cM_VAL &&   \
      cN_VAR == cN_VAL) {                                                   \
    static constexpr int bM_ = BM_VAL;                                      \
    static constexpr int bN_ = BN_VAL;                                      \
    static constexpr int bK_ = BK_VAL;                                      \
    static constexpr int R_ = R_VAL;                                        \
    static constexpr int stages_ = STAGES_VAL;                              \
    static constexpr int cM_ = cM_VAL;                                      \
    static constexpr int cN_ = cN_VAL;                                      \
    __VA_ARGS__;                                                            \
  }

#define NOISING_A_CONFIG_OPTION(                                              \
    BM_VAR, BK_VAR, R_VAR, STAGES_VAR, AxEBL_TYPE_VAR, BM_VAL, BK_VAL, R_VAL, \
    STAGES_VAL, AxEBL_TYPE_VAL, AxEBL_TYPE_CONST, ...)                        \
  if (R_VAR == R_VAL && BM_VAR == BM_VAL && BK_VAR == BK_VAL &&               \
      STAGES_VAR == STAGES_VAL && AxEBL_TYPE_VAR == AxEBL_TYPE_VAL) {         \
    static constexpr int bM_ = BM_VAL;                                        \
    static constexpr int bK_ = BK_VAL;                                        \
    static constexpr int stages_ = STAGES_VAL;                                \
    static constexpr int R_ = R_VAL;                                          \
    using ElementDenoise_AxEBL = AxEBL_TYPE_CONST;                            \
    __VA_ARGS__;                                                              \
  }

#define NOISING_B_CONFIG_OPTION(                                            \
    BN_VAR, BK_VAR, R_VAR, STAGES_VAR, EARxBpEB_TYPE_VAR, BN_VAL, BK_VAL,   \
    R_VAL, STAGES_VAL, EARxBpEB_TYPE_VAL, EARxBpEB_TYPE_CONST, ...)         \
  if (R_VAR == R_VAL && BN_VAR == BN_VAL && BK_VAR == BK_VAL &&             \
      STAGES_VAR == STAGES_VAL && EARxBpEB_TYPE_VAR == EARxBpEB_TYPE_VAL) { \
    static constexpr int bN_ = BN_VAL;                                      \
    static constexpr int bK_ = BK_VAL;                                      \
    static constexpr int stages_ = STAGES_VAL;                              \
    static constexpr int R_ = R_VAL;                                        \
    using ElementDenoise_EARxBpEB = EARxBpEB_TYPE_CONST;                    \
    __VA_ARGS__;                                                            \
  }

#define SCALAR_TYPE_SWITCH(DTYPE_VAR, TYPE_NAME, ...)               \
  do {                                                              \
    if (DTYPE_VAR == at::ScalarType::Half) {                        \
      using TYPE_NAME = cutlass::half_t;                            \
      __VA_ARGS__;                                                  \
    } else if (DTYPE_VAR == at::ScalarType::BFloat16) {             \
      using TYPE_NAME = cutlass::bfloat16_t;                        \
      __VA_ARGS__;                                                  \
    } else {                                                        \
      TORCH_CHECK(false,                                            \
                  "Unsupported input dtype for quantization. Only " \
                  "float16 and bfloat16 are supported.");           \
    }                                                               \
  } while (0)

#define QUANTIZE_CONFIG_OPTION(FAST_MATH_VAR, USE_SMOOTH_SCALE_VAR,    \
                               MAX_VAL_VAR, FAST_MATH_VAL,             \
                               USE_SMOOTH_SCALE_VAL, MAX_VAL_VAL, ...) \
  if (FAST_MATH_VAR == FAST_MATH_VAL &&                                \
      USE_SMOOTH_SCALE_VAR == USE_SMOOTH_SCALE_VAL &&                  \
      MAX_VAL_VAR == MAX_VAL_VAL) {                                    \
    static constexpr bool FastMath_ = FAST_MATH_VAL;                   \
    static constexpr bool UseSmoothScale_ = USE_SMOOTH_SCALE_VAL;      \
    static constexpr int MaxVal_ = MAX_VAL_VAL;                        \
    __VA_ARGS__;                                                       \
  }

#define QUANTIZE_CONFIG_SWITCH(FAST_MATH_VAR, USE_SMOOTH_SCALE_VAR,        \
                               MAX_VAL_VAR, ...)                           \
  QUANTIZE_CONFIG_OPTION(FAST_MATH_VAR, USE_SMOOTH_SCALE_VAR, MAX_VAL_VAR, \
                         false, false, 63, __VA_ARGS__)                    \
  QUANTIZE_CONFIG_OPTION(FAST_MATH_VAR, USE_SMOOTH_SCALE_VAR, MAX_VAL_VAR, \
                         false, false, 127, __VA_ARGS__)                   \
  QUANTIZE_CONFIG_OPTION(FAST_MATH_VAR, USE_SMOOTH_SCALE_VAR, MAX_VAL_VAR, \
                         false, true, 63, __VA_ARGS__)                     \
  QUANTIZE_CONFIG_OPTION(FAST_MATH_VAR, USE_SMOOTH_SCALE_VAR, MAX_VAL_VAR, \
                         false, true, 127, __VA_ARGS__)                    \
  QUANTIZE_CONFIG_OPTION(FAST_MATH_VAR, USE_SMOOTH_SCALE_VAR, MAX_VAL_VAR, \
                         true, false, 63, __VA_ARGS__)                     \
  QUANTIZE_CONFIG_OPTION(FAST_MATH_VAR, USE_SMOOTH_SCALE_VAR, MAX_VAL_VAR, \
                         true, false, 127, __VA_ARGS__)                    \
  QUANTIZE_CONFIG_OPTION(FAST_MATH_VAR, USE_SMOOTH_SCALE_VAR, MAX_VAL_VAR, \
                         true, true, 63, __VA_ARGS__)                      \
  QUANTIZE_CONFIG_OPTION(FAST_MATH_VAR, USE_SMOOTH_SCALE_VAR, MAX_VAL_VAR, \
                         true, true, 127, __VA_ARGS__)

#define SMOOTH_SCALE_TYPE_SWITCH(USE_SMOOTH_SCALE, SMOOTH_SCALE_DTYPE, \
                                 INPUT_DTYPE, SMOOTH_SCALE_TYPE, ...)  \
  do {                                                                 \
    if (!(USE_SMOOTH_SCALE)) {                                         \
      using SMOOTH_SCALE_TYPE = float;                                 \
      __VA_ARGS__;                                                     \
    } else if (SMOOTH_SCALE_DTYPE == at::ScalarType::Float) {          \
      using SMOOTH_SCALE_TYPE = float;                                 \
      __VA_ARGS__;                                                     \
    } else if (SMOOTH_SCALE_DTYPE == at::ScalarType::Half) {           \
      using SMOOTH_SCALE_TYPE = cutlass::half_t;                       \
      __VA_ARGS__;                                                     \
    } else if (SMOOTH_SCALE_DTYPE == at::ScalarType::BFloat16) {       \
      using SMOOTH_SCALE_TYPE = cutlass::bfloat16_t;                   \
      __VA_ARGS__;                                                     \
    }                                                                  \
  } while (0)

#include "static_switch_matmul.h"
#include "static_switch_noisingA.h"
#include "static_switch_noisingB.h"
