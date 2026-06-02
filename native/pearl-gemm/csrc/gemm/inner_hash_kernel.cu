#include <cuda_runtime.h>
#include <cassert>
#include <cute/container/array.hpp>
#include <cute/pointer.hpp>
#include <cute/tensor.hpp>
#include "pow_utils.hpp"

using namespace cute;

template <size_t NUM_ITER>
__global__ void inner_hash_kernel(uint32_t* input_buffer, int input_size,
                                  uint32_t* output_hash) {

  // Only process with a single thread
  if (blockIdx.x == 0 && threadIdx.x == 0) {

    uint32_t hash_result = 0;
    auto hash_with_size = [&](auto SIZE_INT) {
      auto input_tensor = make_tensor(make_gmem_ptr(input_buffer), SIZE_INT);
      auto register_tensor = make_fragment_like(input_tensor);
      copy(input_tensor, register_tensor);
      for (int i = 0; i < NUM_ITER; i++) {
        hash_result += pearl::xor_reduction(register_tensor);
      }
    };

    if (input_size == 64) {
      hash_with_size(Int<64>{});
    } else if (input_size == 96) {
      hash_with_size(Int<96>{});
    } else if (input_size == 128) {
      hash_with_size(Int<128>{});
    } else if (input_size == 192) {
      hash_with_size(Int<192>{});
    } else if (input_size == 256) {
      hash_with_size(Int<256>{});
    }

    *output_hash = hash_result;
  }
}

// Host function to launch the kernel
void launch_inner_hash_kernel(uint32_t* input_buffer, int input_size,
                              uint32_t* output_hash, int64_t iterations,
                              cudaStream_t stream) {

  // iterations == 1 is for correctness testing, 100000 is for benchmarking
  if (iterations == 1) {
    inner_hash_kernel<1>
        <<<1, 1, 0, stream>>>(input_buffer, input_size, output_hash);
  } else if (iterations == 100000) {
    inner_hash_kernel<100000>
        <<<1, 1, 0, stream>>>(input_buffer, input_size, output_hash);
  } else {
    assert(false);
  }
}
