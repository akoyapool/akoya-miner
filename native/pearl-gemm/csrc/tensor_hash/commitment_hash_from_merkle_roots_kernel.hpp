#include "blake3/blake3.cuh"

namespace pearl {

class CommitmentHashFromMerkleRootsKernel {
 public:
  using Element = uint8_t;
  static constexpr uint32_t MaxThreadsPerBlock = 1;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr int SharedStorageSize = 0;

  using RmemChainingValueLayout =
      Layout<Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>>>;
  using RmemBlockLayout = Layout<Shape<Int<blake3::MSG_BLOCK_SIZE_U32>>>;

  // Device side arguments
  struct Arguments {
    Element const* const ptr_A_merkle_root;
    Element const* const ptr_B_merkle_root;
    Element const* const ptr_key;
    Element* const ptr_A_commitment_hash;
    Element* const ptr_B_commitment_hash;
  };

  // Kernel entry point API
  struct Params {
    Element const* const ptr_A_merkle_root;
    Element const* const ptr_B_merkle_root;
    Element const* const ptr_key;
    Element* const ptr_A_commitment_hash;
    Element* const ptr_B_commitment_hash;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {args.ptr_A_merkle_root, args.ptr_B_merkle_root, args.ptr_key,
            args.ptr_A_commitment_hash, args.ptr_B_commitment_hash};
  }

  static dim3 get_grid_shape(Params const& params) { return dim3(1); }

  static dim3 get_block_shape() { return dim3(1); }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {

    // First calculate B commitment hash: BLAKE3(key || B_merkle_root)
    Tensor rBlockB = make_tensor<uint32_t>(RmemBlockLayout{});
    Tensor rChainingValueB = make_tensor<uint32_t>(RmemChainingValueLayout{});

    uint32_t const* key_u32 = (uint32_t const*)params.ptr_key;
    uint32_t const* B_merkle_root_u32 =
        (uint32_t const*)params.ptr_B_merkle_root;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rBlockB(i) = key_u32[i];
      rBlockB(i + blake3::CHAINING_VALUE_SIZE_U32) = B_merkle_root_u32[i];
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rChainingValueB(i) = blake3::IV[i];
    }

    // We do one BLAKE3 block as we have exactly blake3::MSG_BLOCK_SIZE data
    static constexpr blake3::CompressParams single_block_params = {
        .counter = 0,
        .block_len = blake3::MSG_BLOCK_SIZE,
        .flags = blake3::CHUNK_START | blake3::CHUNK_END | blake3::ROOT,
    };
    blake3::compress_msg_block_u32(rBlockB, rChainingValueB,
                                   single_block_params);

    // Now calculate A commitment hash: BLAKE3(B_commitment_hash || A_merkle_root)
    Tensor rBlockA = make_tensor<uint32_t>(RmemBlockLayout{});
    Tensor rChainingValueA = make_tensor<uint32_t>(RmemChainingValueLayout{});

    uint32_t const* A_merkle_root_u32 =
        (uint32_t const*)params.ptr_A_merkle_root;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rBlockA(i) = rChainingValueB(i);  // Use B's result, not the key!
      rBlockA(i + blake3::CHAINING_VALUE_SIZE_U32) = A_merkle_root_u32[i];
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rChainingValueA(i) = blake3::IV[i];
    }

    blake3::compress_msg_block_u32(rBlockA, rChainingValueA,
                                   single_block_params);

    // Copy the result back to global memory
    uint32_t* A_commitment_hash_u32 = (uint32_t*)params.ptr_A_commitment_hash;
    uint32_t* B_commitment_hash_u32 = (uint32_t*)params.ptr_B_commitment_hash;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      A_commitment_hash_u32[i] = rChainingValueA(i);
      B_commitment_hash_u32[i] = rChainingValueB(i);
    }
  }
};  // class CommitmentHashFromMerkleRootsKernel
}  // namespace pearl