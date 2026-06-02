#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Host function to launch the inner_hash kernel
void launch_inner_hash_kernel(uint32_t* input_buffer, int input_size,
                              uint32_t* output_hash, int64_t iterations,
                              cudaStream_t stream);
