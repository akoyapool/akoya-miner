/*! \file
    \brief
    Dynamic quantization kernel that quantizes floating-point tensors to int8 values with per-token dynamic
    scaling. Supports 7-bit ([-63, 63]) and 8-bit ([-127, 127]) quantization ranges, with optional smooth scale
    pre-processing.

    \details
    This kernel performs dynamic quantization on 2D tensors with shape (num_tokens, hidden_size).  Each token (row) is
    quantized independently using its own scale factor computed as:
        scale = max(abs(token_values)) / max_val

    When smooth_scale is provided, the input is first multiplied by smooth_scale before quantization.
    NOTE: smooth_scale is expected to contain reciprocal values (1/original_scale), so multiplication
    achieves the same effect as division but with better performance:
        scaled_input = input * smooth_scale
        scale = max(abs(scaled_input)) / max_val
        quantized_value = round(scaled_input / scale)

    The quantized values are computed as:
        quantized_value = round(original_value / scale)

    This ensures all quantized values fall within the specified range.

    Template Parameters:
    - scalar_t: Input data type (cutlass::half_t or cutlass::bfloat16_t)
    - FastMath: When true, uses fast division; when false, uses IEEE-754 compliant division
                for bit-exact reproducibility
    - UseSmoothScale: When true, applies smooth_scale multiplication before quantization
    - MaxVal: Maximum quantization value (63 for 7-bit, 127 for 8-bit)

    Memory Layout:
    - ptr_data: Input tensor of shape (num_tokens, hidden_size) in row-major layout.
                Each row represents one token's hidden state values.
                Supported types: cutlass::half_t (float16), cutlass::bfloat16_t (bfloat16)

    - ptr_smooth_scale: Optional smooth scale tensor of shape (hidden_size,).
                        When provided (UseSmoothScale=true), input is multiplied by this
                        before quantization. Must be float32.

    - ptr_x_q:  Output quantized tensor of shape (num_tokens, hidden_size) in row-major layout.
                Each element is an int8 value in range [-max_val, max_val].
                Memory layout matches input tensor exactly.

    - ptr_x_s:  Output scale tensor of shape (num_tokens, 1) in row-major layout.
                Each element is a float32 scale factor for the corresponding token.
                scale[i] = max(abs(input[i, :] * smooth_scale[:])) / max_val

    Thread/Block Organization:
    - Grid dimension: (num_tokens, 1, 1) - one block per token
    - Block dimension: (256, 1, 1) - fixed at compile time (COMPILE_TIME_STRIDE)
    - Each block processes exactly one token (one row of the input tensor)
    - Threads within a block cooperatively:
      1. Find the maximum absolute value across the token's hidden_size elements
      2. Compute the scale factor using CUB block reduction
      3. Apply quantization to all elements in the token

    Memory Access Pattern:
    - Each thread processes multiple elements of a token using vectorized loads/stores (32-byte vectors)
    - Threads stride through the hidden_size dimension with stride = blockDim.x
    - Thread tid processes elements at indices: tid, tid + blockDim.x, tid + 2*blockDim.x, ...
    - Uses shared memory for CUB block reduction to find the maximum value
    - When UseSmoothScale=true, smooth_scale is read from global memory (relies on L1/L2 cache)
*/
#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime.h>

// Maximum quantization values for 7-bit and 8-bit quantization
constexpr int MAX_VAL_7BIT = 63;
constexpr int MAX_VAL_8BIT = 127;

template <typename scalar_t, typename SmoothScaleT, bool FastMath,
          bool UseSmoothScale, int MaxVal>
void run_quantize_kernel(const scalar_t* ptr_data,
                         const SmoothScaleT* ptr_smooth_scale, int8_t* ptr_x_q,
                         float* ptr_x_s, int num_tokens, int hidden_size,
                         cudaStream_t stream);
