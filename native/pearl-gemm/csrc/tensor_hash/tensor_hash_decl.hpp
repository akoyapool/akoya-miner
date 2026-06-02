#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

// Forward declarations for tensor hash functions
// These can be included from .cpp files

// Tensor hash with configurable kernel parameters:
//   threads_per_block: Number of threads per block for merkle_tree_roots_kernel (128, 256, 512)
//   num_stages: Pipeline stages for merkle_tree_roots_kernel (2, 3, 4)
//   leaves_per_mt_block: Threads for compute_blake_mt_kernel (256, 512, 1024)
void tensor_hash(
    const uint8_t* data, uint32_t data_size, uint8_t* out,
    const uint8_t key[32], uint32_t num_blocks,
    uint32_t threads_per_block,    // merkle_tree_roots_kernel threads
    uint32_t num_stages,           // merkle_tree_roots_kernel pipeline stages
    uint32_t leaves_per_mt_block,  // compute_blake_mt_kernel threads
    uint8_t* roots, cudaDeviceProp& deviceProp, cudaStream_t stream);

// Upload the tensor_hash key into device constant memory on `stream`.
// The no-key-upload variant below assumes this has already been called.
void tensor_hash_set_key(const uint8_t key[32], cudaStream_t stream);

void tensor_hash_no_key_upload(
    const uint8_t* data, uint32_t data_size, uint8_t* out,
    const uint8_t key[32], uint32_t num_blocks,
    uint32_t threads_per_block,
    uint32_t num_stages,
    uint32_t leaves_per_mt_block,
    uint8_t* roots, cudaDeviceProp& deviceProp, cudaStream_t stream);

void tensor_hash_with_leaf_cvs(
    const uint8_t* data, uint32_t data_size, uint8_t* out,
    const uint8_t key[32], uint32_t num_blocks,
    uint32_t threads_per_block,
    uint32_t num_stages,
    uint32_t leaves_per_mt_block,
    uint8_t* roots, uint8_t* leaf_cvs, cudaDeviceProp& deviceProp,
    cudaStream_t stream);

// Fused BSeed expansion + tensor_hash stage for sigma rotation. `bseed` is a
// host pointer to the 32-byte seed. `data`, `out`, `key`, and `roots` are
// device pointers. The generated B bytes are written to `data` before hashing.
void bseed_expand_and_tensor_hash(
    const uint8_t bseed[32], uint8_t* data, uint32_t data_size, uint8_t* out,
    const uint8_t key[32], uint32_t num_blocks, uint32_t threads_per_block,
    uint32_t num_stages, uint32_t leaves_per_mt_block, uint8_t* roots,
    cudaDeviceProp& deviceProp, cudaStream_t stream);

void bseed_expand_and_tensor_hash_with_leaf_cvs(
    const uint8_t bseed[32], uint8_t* data, uint32_t data_size, uint8_t* out,
    const uint8_t key[32], uint32_t num_blocks, uint32_t threads_per_block,
    uint32_t num_stages, uint32_t leaves_per_mt_block, uint8_t* roots,
    uint8_t* leaf_cvs, cudaDeviceProp& deviceProp, cudaStream_t stream);

void commitment_hash_from_merkle_roots(
    const uint8_t* A_merkle_root, const uint8_t* B_merkle_root,
    const uint8_t* key, uint8_t* A_commitment_hash, uint8_t* B_commitment_hash,
    cudaDeviceProp& deviceProp, cudaStream_t stream);
