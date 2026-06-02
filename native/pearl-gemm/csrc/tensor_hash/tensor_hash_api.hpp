// Include these 2 headers instead of torch/extension.h since we don't need all
// of the torch headers.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/python.h>
#include <cstddef>
// Include only the host function declaration
#include "blake3/blake3_constants.hpp"
#include "tensor_hash_decl.hpp"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Default values for kernel parameters
constexpr size_t DEFAULT_THREADS_PER_BLOCK = 128;    // merkle_tree_roots_kernel
constexpr size_t DEFAULT_NUM_STAGES = 2;             // merkle_tree_roots_kernel
constexpr size_t DEFAULT_LEAVES_PER_MT_BLOCK = 512;  // compute_blake_mt_kernel

size_t get_required_scratchpad_bytes(
    size_t matrix_bytes, size_t threads_per_block = DEFAULT_THREADS_PER_BLOCK) {
  size_t bytes_per_block = threads_per_block * blake3::CHUNK_SIZE;
  size_t required_blocks =
      (matrix_bytes + bytes_per_block - 1) / bytes_per_block;
  return required_blocks * blake3::CHAINING_VALUE_SIZE;
}

// Tensor hash with configurable kernel parameters:
//   threads_per_block: Threads per block for merkle_tree_roots_kernel (128, 256, 512)
//   num_stages: Pipeline stages for merkle_tree_roots_kernel (2)
//   leaves_per_mt_block: Threads for compute_blake_mt_kernel (256, 512, 1024)
void run_tensor_hash(
    at::Tensor& data,  // input data tensor
    at::Tensor& key, at::Tensor& out, at::Tensor& roots,
    int64_t threads_per_block = DEFAULT_THREADS_PER_BLOCK,
    int64_t num_stages = DEFAULT_NUM_STAGES,
    int64_t leaves_per_mt_block = DEFAULT_LEAVES_PER_MT_BLOCK) {
  CHECK_DEVICE(data);
  CHECK_DEVICE(key);
  CHECK_DEVICE(out);
  CHECK_DEVICE(roots);
  CHECK_CONTIGUOUS(data);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(roots);

  TORCH_CHECK(key.dtype() == at::kByte, "key must be uint8");
  TORCH_CHECK(out.dtype() == at::kByte, "out must be uint8");
  TORCH_CHECK(roots.dtype() == at::kByte, "roots must be uint8");
  TORCH_CHECK(data.dim() == 2, "data must be 2D tensor");
  TORCH_CHECK(key.numel() == blake3::KEY_SIZE, "key must have exactly",
              blake3::KEY_SIZE, "bytes");
  TORCH_CHECK(out.numel() == blake3::CHAINING_VALUE_SIZE,
              "out must have exactly", blake3::CHAINING_VALUE_SIZE, "bytes");
  TORCH_CHECK(roots.numel() % blake3::CHAINING_VALUE_SIZE == 0,
              "roots must have a multiple of", blake3::CHAINING_VALUE_SIZE,
              "bytes");

  // Validate threads_per_block (merkle_tree_roots_kernel)
  TORCH_CHECK(threads_per_block == 128 || threads_per_block == 256 ||
                  threads_per_block == 512,
              "threads_per_block must be 128, 256, or 512");

  // Validate num_stages
  TORCH_CHECK(num_stages == 2 || num_stages == 3 || num_stages == 4,
              "num_stages must be 2, 3, or 4");

  // Validate leaves_per_mt_block (compute_blake_mt_kernel)
  TORCH_CHECK(leaves_per_mt_block == 256 || leaves_per_mt_block == 512 ||
                  leaves_per_mt_block == 1024,
              "leaves_per_mt_block must be 256, 512, or 1024");

  constexpr size_t chunk_size = 1024;

  // We split data into chunks of size C (chunk_size)
  size_t num_chunks = (data.numel() + chunk_size - 1) / chunk_size;
  // We split chunks into blocks based on threads_per_block
  size_t num_blocks = (num_chunks + threads_per_block - 1) / threads_per_block;

  TORCH_INTERNAL_ASSERT(
      num_blocks * blake3::CHAINING_VALUE_SIZE ==
          get_required_scratchpad_bytes(data.numel(), threads_per_block),
      "num_blocks=", num_blocks, " get_required_scratchpad_bytes=",
      get_required_scratchpad_bytes(data.numel(), threads_per_block));
  TORCH_CHECK((size_t)roots.numel() >= get_required_scratchpad_bytes(
                                           data.numel(), threads_per_block),
              "roots must have at least ", num_blocks, " * ",
              blake3::CHAINING_VALUE_SIZE, "bytes");
  TORCH_CHECK((size_t)data.numel() > (1u << 17),
              "data must have more than 2^17 = 131072 bytes, got ",
              data.numel());

  auto stream = at::cuda::getCurrentCUDAStream();
  auto dprops = at::cuda::getCurrentDeviceProperties();

  tensor_hash(data.data_ptr<uint8_t>(), data.numel(), out.data_ptr<uint8_t>(),
              key.data_ptr<uint8_t>(), num_blocks,
              static_cast<uint32_t>(threads_per_block),
              static_cast<uint32_t>(num_stages),
              static_cast<uint32_t>(leaves_per_mt_block),
              roots.data_ptr<uint8_t>(), *dprops, stream);
}

// Computes both A and B commitment hashes from their merkle roots
// Should be the same as commitment_hash_from_merkle_roots from Commitment_hash.py
void run_commitment_hash_from_merkle_roots(at::Tensor& A_merkle_root,
                                           at::Tensor& B_merkle_root,
                                           at::Tensor& key,
                                           at::Tensor& A_commitment_hash,
                                           at::Tensor& B_commitment_hash) {
  CHECK_DEVICE(A_merkle_root);
  CHECK_DEVICE(B_merkle_root);
  CHECK_DEVICE(key);
  CHECK_CONTIGUOUS(A_merkle_root);
  CHECK_CONTIGUOUS(B_merkle_root);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(A_commitment_hash);
  CHECK_CONTIGUOUS(B_commitment_hash);

  TORCH_CHECK(A_merkle_root.dtype() == at::kByte,
              "A_merkle_root must be uint8");
  TORCH_CHECK(B_merkle_root.dtype() == at::kByte,
              "B_merkle_root must be uint8");
  TORCH_CHECK(key.dtype() == at::kByte, "key must be uint8");
  TORCH_CHECK(A_commitment_hash.dtype() == at::kByte,
              "A_commitment_hash must be uint8");
  TORCH_CHECK(B_commitment_hash.dtype() == at::kByte,
              "B_commitment_hash must be uint8");

  TORCH_CHECK(A_merkle_root.numel() == blake3::CHAINING_VALUE_SIZE,
              "A_merkle_root must have exactly", blake3::CHAINING_VALUE_SIZE,
              "bytes");
  TORCH_CHECK(B_merkle_root.numel() == blake3::CHAINING_VALUE_SIZE,
              "B_merkle_root must have exactly", blake3::CHAINING_VALUE_SIZE,
              "bytes");
  TORCH_CHECK(key.numel() == blake3::KEY_SIZE, "key must have exactly",
              blake3::KEY_SIZE, "bytes");
  TORCH_CHECK(A_commitment_hash.numel() == blake3::CHAINING_VALUE_SIZE,
              "A_commitment_hash must have exactly",
              blake3::CHAINING_VALUE_SIZE, "bytes");
  TORCH_CHECK(B_commitment_hash.numel() == blake3::CHAINING_VALUE_SIZE,
              "B_commitment_hash must have exactly",
              blake3::CHAINING_VALUE_SIZE, "bytes");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto dprops = at::cuda::getCurrentDeviceProperties();

  commitment_hash_from_merkle_roots(
      A_merkle_root.data_ptr<uint8_t>(), B_merkle_root.data_ptr<uint8_t>(),
      key.data_ptr<uint8_t>(), A_commitment_hash.data_ptr<uint8_t>(),
      B_commitment_hash.data_ptr<uint8_t>(), *dprops, stream);
}

#undef CHECK_DEVICE
#undef CHECK_CONTIGUOUS
