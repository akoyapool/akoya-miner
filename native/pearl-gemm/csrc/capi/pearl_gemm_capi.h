// pearl_gemm_capi.h — public C ABI for pearl-gemm kernels.
//
// Stable C-language interface — narrow to what the miner actually uses.
// Compiled into the torch-free libpearl_gemm_capi.so.
//
// Status code convention:
//   0   success
//   <0  failure
//
// "Default" behaviour notes (baked into shim, NOT exposed):
//   * noise_gen:   num_threads = 64, aux_buffer disabled
//   * noise_B:     defaults for tile_n / tile_k / pipeline_stages,
//                  no split-K, EARxBpEB is fp16 (never int32)
//   * noisy_gemm:  pipeline_stages = 3, swizzle = auto, swizzle_n_maj = true,
//                  noising tile sizes / pipeline_stages_noising_* default,
//                  k_blocks_per_split_noising_* auto,
//                  run_noising_A = true, run_noising_B = false,
//                  skip_reduction = false, skip_denoising = true,
//                  skip_c_store = true,
//                  AxEBL / EARxBpEB always fp16 (never int32 split-K),
//                  no inner_hash_counter, enable_debug = false.
// If any of those need to vary, add the field back rather than threading
// the full pybind surface through.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ABI version. Bumped on incompatible signature changes.
int pearl_capi_abi_version(void);

// Native build profile reporting. Used by miner startup to fail fast when a
// loaded architecture-specific .so cannot run on the selected GPU.
const char* pearl_capi_build_profile(void);
int pearl_capi_supports_sm(int major, int minor);

// Sizes (bytes).
int     pearl_capi_get_host_signal_sync_size(void);
int     pearl_capi_get_host_signal_header_size(void);
int64_t pearl_capi_get_required_scratchpad_bytes(int64_t matrix_bytes,
                                                 int     threads_per_block);

// tensor_hash on a flat byte buffer. All pointers are device pointers.
int pearl_capi_tensor_hash(const uint8_t* data,
                           uint32_t        data_size,
                           uint8_t*        out,
                           const uint8_t*  key,
                           uint32_t        num_blocks,
                           uint32_t        threads_per_block,
                           uint32_t        num_stages,
                           uint32_t        leaves_per_mt_block,
                           uint8_t*        roots,
                           int             device_id,
                           void*           stream /* cudaStream_t */);

// tensor_hash plus the per-1024-byte leaf chaining values. All pointers are
// device pointers. `leaf_cvs` must have ceil(data_size / 1024) * 32 bytes.
int pearl_capi_tensor_hash_leaf_cvs(const uint8_t* data,
                                    uint32_t       data_size,
                                    uint8_t*       out,
                                    const uint8_t* key,
                                    uint32_t       num_blocks,
                                    uint32_t       threads_per_block,
                                    uint32_t       num_stages,
                                    uint32_t       leaves_per_mt_block,
                                    uint8_t*       roots,
                                    uint8_t*       leaf_cvs,
                                    int            device_id,
                                    void*          stream);

// Fused BSeed expansion + tensor_hash for resident B install. `bseed` is a
// host pointer to the 32-byte BSeed. All other pointers are device pointers.
// Writes generated B into `data` and the Merkle root into `out`.
int pearl_capi_bseed_expand_and_tensor_hash(const uint8_t* bseed,
                                            uint8_t*       data,
                                            uint32_t       data_size,
                                            uint8_t*       out,
                                            const uint8_t* key,
                                            uint32_t       num_blocks,
                                            uint32_t       threads_per_block,
                                            uint32_t       num_stages,
                                            uint32_t       leaves_per_mt_block,
                                            uint8_t*       roots,
                                            int            device_id,
                                            void*          stream);

int pearl_capi_bseed_expand_and_tensor_hash_leaf_cvs(
                                            const uint8_t* bseed,
                                            uint8_t*       data,
                                            uint32_t       data_size,
                                            uint8_t*       out,
                                            const uint8_t* key,
                                            uint32_t       num_blocks,
                                            uint32_t       threads_per_block,
                                            uint32_t       num_stages,
                                            uint32_t       leaves_per_mt_block,
                                            uint8_t*       roots,
                                            uint8_t*       leaf_cvs,
                                            int            device_id,
                                            void*          stream);

int pearl_capi_commitment_hash_from_merkle_roots(const uint8_t* A_merkle_root,
                                                 const uint8_t* B_merkle_root,
                                                 const uint8_t* key,
                                                 uint8_t*       A_commitment_hash,
                                                 uint8_t*       B_commitment_hash,
                                                 int            device_id,
                                                 void*          stream);

// noise_gen. Any nullptr device pointer means the caller does not want
// that noise matrix populated. R must be 64 or 128. num_threads is fixed
// at 64 inside the shim (matches pure-miner Python default).
int pearl_capi_noise_gen(int R,
                         int m, int n, int k,
                         void* EAL, void* EAL_fp16,
                         void* EAR_R_major, void* EAR_K_major,
                         void* EBL_R_major, void* EBL_K_major,
                         void* EBR, void* EBR_fp16,
                         const uint8_t* key_A, const uint8_t* key_B,
                         void* stream);

// noise_B (σ-refresh only — runs once per σ to compute BpEB = B + EBL·EBR).
struct PearlCapiNoiseBParams {
    int32_t n, k, r;
    void* B;            // n x k  int8
    void* EAR_K_major;  // r x k  int8
    void* EBL_R_major;  // k x r  int8
    void* EBR;          // n x r  int8
    void* EARxBpEB;     // n x r  fp16  (write-only, value discarded)
    void* BpEB;         // n x k  int8

    // Optional pre-allocated workspace (PearlCapiWorkspace*). When non-null,
    // noise_B uses the workspace's noise_B scratchpad instead of
    // cudaMallocAsync-ing a fresh buffer per call. When null, falls back to
    // the per-call alloc/free path. (ABI v2.)
    void* workspace;
};
int pearl_capi_noise_B(const struct PearlCapiNoiseBParams* p, void* stream);

// Full native B-side σ install. The caller must have already computed AHash
// for the throwaway A seed. If `expand_bseed != 0`, `bseed` must point to the
// 32-byte host BSeed and resident `B` is regenerated before hashing. If
// `expand_bseed == 0`, resident `B` is reused and only rehashed with `key`.
// Internally does:
//   optional BSeed -> B + BHash, else tensor_hash(B) -> BHash
//   commitment_hash(AHash, BHash) -> CommitA/CommitB
//   noise_gen(B-side only)
//   noise_B(B, EBR, EBL, EAR) -> BpEB/EARxBpEB
struct PearlCapiInstallBParams {
    int32_t m, n, k, r;
    int32_t expand_bseed;
    uint32_t th_num_blocks;
    uint32_t th_threads;
    uint32_t th_stages;
    uint32_t th_leaves;
    int32_t device_id;

    const uint8_t* bseed;  // host pointer, nullable when expand_bseed == 0
    void* B;
    void* BHash;
    void* Key;
    void* Roots;
    void* AHash;
    void* CommitA;
    void* CommitB;
    void* EAR_K_major;
    void* EBL_R_major;
    void* EBL_K_major;
    void* EBR;
    void* EBR_fp16;
    void* EARxBpEB;
    void* BpEB;
    void* workspace;
    void* LeafCvs;  // optional total_leaves x 32B device buffer
};
int pearl_capi_install_B(const struct PearlCapiInstallBParams* p,
                         void* stream);

// noisy_gemm — the hot inner-loop kernel.
struct PearlCapiNoisyGemmParams {
    // Dimensions.
    int32_t m, n, k, r;

    // Matmul tiling / cluster — must match a built kernel instantiation.
    int32_t bM, bN, bK, cM, cN;

    // Device pointers.
    void* A;             // m x k  int8
    void* B;             // n x k  int8
    void* EAL;           // m x r  int8
    void* EAL_fp16;      // m x r  fp16
    void* EBR;           // n x r  int8
    void* EBR_fp16;      // n x r  fp16
    void* EAR_R_major;   // k x r  int8
    void* EBL_R_major;   // k x r  int8
    void* EAR_K_major;   // r x k  int8
    void* EBL_K_major;   // r x k  int8
    void* AxEBL_fp16;    // m x r  fp16
    void* EARxBpEB_fp16; // n x r  fp16
    void* ApEA;          // m x k  int8
    void* BpEB;          // n x k  int8
    void* A_scales;      // m      fp32
    void* B_scales;      // n      fp32
    void* C;             // m x n  bf16
    void* host_signal_header_pinned; // host pinned, int8
    void* host_signal_sync;          // device, int8
    void* pow_target;                // device, uint32 [8]
    void* pow_key;                   // device, uint32 [8]

    // Optional pre-allocated workspace (PearlCapiWorkspace*). When non-null,
    // noisy_gemm uses the workspace's noiseA scratchpad + transcript buffer
    // instead of cudaMallocAsync-ing per call. When null, falls back to the
    // per-call alloc/free path. Single workspace can be shared across iters
    // for a given (m, n, k, r) — typically allocated once per σ-refresh.
    // (ABI v2.)
    void* workspace;
};
int pearl_capi_noisy_gemm(const struct PearlCapiNoisyGemmParams* p, void* stream);

// Per-σ workspace pool.  Holds the noise_A / noise_B int32 scratch buffers
// and the transcript buffer used by noisy_gemm.  Allocate once per σ-refresh
// (the buffers' sizes are determined by m, n, k, r which are stable across
// nonces within a σ), pass the handle in every noise_B / noisy_gemm call,
// then free on σ-rotation.  This removes per-iter cudaMallocAsync overhead
// — the repo's perf notes attribute ~30 % of portable wall time to this.
//
// `with_noise_A` and `with_noise_B` toggle whether each scratchpad is
// allocated; in pure-miner mode noise_B runs only once per σ inside the
// σ-refresh path while noise_A runs every iter inside noisy_gemm.  Setting
// `with_noise_B=0` after the σ-refresh is permitted (the noise_B scratch is
// still kept around so a re-refresh can reuse the same workspace).
//
// All allocations are stream-ordered (cudaMallocAsync / cudaFreeAsync) on
// the supplied stream.
int pearl_capi_workspace_alloc(int32_t m, int32_t n, int32_t k, int32_t r,
                               int with_noise_A, int with_noise_B,
                               void** out_workspace, void* stream);
int pearl_capi_workspace_free(void* workspace, void* stream);

// Per-σ constant parameter cache — eliminates per-iter argument marshalling.
//
// Call pearl_capi_workspace_install_params() ONCE after workspace_alloc and
// after all device pointers / seeds are stable (i.e. at σ-install time).
// The workspace then caches ALL constants so the per-iter hot path can use
// the minimal pearl_capi_iter() call (4 args instead of 40 across 5 calls).
//
// Fields must remain valid for the lifetime of the workspace (device
// pointers do not need to be stable host-side; only the pointer values
// themselves are stored).
struct PearlCapiWorkspaceParams {
    // Dimensions (constant for σ lifetime).
    int32_t m, n, k, r;

    // Matmul tiling — must match a built kernel instantiation.
    int32_t bM, bN, bK, cM, cN;

    // TensorHash constants (equal to TENSOR_HASH_THREADS/STAGES/LEAVES).
    uint32_t th_num_blocks;   // = ceil(m*k / (th_threads * 1024))
    uint32_t th_threads;      // = 128
    uint32_t th_stages;       // = 2
    uint32_t th_leaves;       // = 512

    // seed_hi for lcg_int7_fill (= σ seed, constant within σ lifetime).
    uint64_t sigma_seed;

    // Device pointers — content changes per-iter but pointer values are const.
    void* A;              // m×k int8   (overwritten by lcg_int7 each iter)
    void* B;              // n×k int8
    void* AHash;          // tensor_hash output (overwritten each iter)
    void* BHash;          // tensor_hash output for B (computed at σ-install)
    void* Key;            // blake3 job key
    void* Roots;          // merkle roots intermediate (overwritten each iter)
    void* CommitA;        // commitment hash A output (overwritten each iter)
    void* CommitB;        // commitment hash B output
    void* EAL;            // m×r int8
    void* EAL_fp16;       // m×r fp16
    void* EBR;            // n×r int8
    void* EBR_fp16;       // n×r fp16
    void* EAR_R_major;    // k×r int8
    void* EBL_R_major;    // k×r int8
    void* EAR_K_major;    // r×k int8
    void* EBL_K_major;    // r×k int8
    void* AxEBL_fp16;     // m×r fp16
    void* EARxBpEB_fp16;  // n×r fp16
    void* ApEA;           // m×k int8
    void* BpEB;           // n×k int8
    void* A_scales;       // m   fp32
    void* B_scales;       // n   fp32
    void* C;              // m×n bf16
    void* host_signal_sync;   // device int8 — dSync coordination block
    void* pow_target;         // device uint32[8]
    void* pow_key;            // device uint32[8]
};

// Install constant params into the workspace.  Must be called before the
// first pearl_capi_iter() call.  Safe to call again on σ-rotation (the
// workspace's scratch memory is reused; only the param cache is updated).
int pearl_capi_workspace_install_params(void* workspace,
                                        const struct PearlCapiWorkspaceParams* p);

// Per-iteration hot path — replaces 5 separate CAPI calls per iter.
// Internally does: lcg_int7_fill → tensor_hash → commitment_hash →
// noise_gen_A → noisy_gemm, reading all constants from the installed params.
// Only seed_lo (the nonce counter) and host_signal_header_pinned (the
// pinned host buffer for this slot) change between iterations.
//
// Must call pearl_capi_workspace_install_params() before the first use.
// All kernel launches are enqueued on `stream` in order.
int pearl_capi_iter(void*    workspace,
                    uint64_t seed_lo,
                    void*    host_signal_header_pinned,
                    void*    stream);

// Batched variant of pearl_capi_iter() — runs `count` consecutive nonces:
//   seed_lo_start + [0..count-1]
// against host_signal_header_pinned_batch[i] for slot i.
// Reduces managed/native transition overhead in the miner hot path by doing
// one C-ABI call per batch instead of one call per iter.
int pearl_capi_iter_batch(void*         workspace,
                          uint64_t      seed_lo_start,
                          void* const*  host_signal_header_pinned_batch,
                          int32_t       count,
                          void*         stream);

// CUDA graph variant of pearl_capi_iter_batch(). Prepare captures a fixed
// batch shape and fixed host header slots into the workspace. Launch replays
// the graph for a new consecutive seed range. If `count` differs from the
// prepared batch count, callers should use pearl_capi_iter_batch instead.
int pearl_capi_iter_batch_graph_prepare(
    void*         workspace,
    void* const*  host_signal_header_pinned_batch,
    int32_t       count,
    void*         stream);

int pearl_capi_iter_batch_graph_launch(
    void*     workspace,
    uint64_t  seed_lo_start,
    void*     stream);

// Deterministic int7 fill of a device buffer. Each output byte is in
// [-63, +63] (matches torch.randint(-63, 64) range used by pure-miner's
// matrix_factory.fresh_A). Output is a deterministic function of
// (seed_lo, seed_hi) — caller can re-run the same algorithm on the host
// to recover the exact A used by any iteration without keeping per-iter
// snapshot buffers. dst is a device pointer.
int pearl_capi_lcg_int7_fill(void* dst,
                             int64_t n,
                             uint64_t seed_lo,
                             uint64_t seed_hi,
                             void*    stream);

int pearl_capi_lcg_int7_fill_indirect(void* dst,
                                      int64_t n,
                                      const void* seed_lo_base,
                                      uint64_t seed_lo_offset,
                                      uint64_t seed_hi,
                                      void* stream);

// BSeed XOF expansion directly into a device buffer. `bseed` is a host
// pointer to the 32-byte seed; `dst` is a device pointer. Output bytes are
// mapped to signed int7 in [-63, +63], matching the Rust/C# BSeed expander.
int pearl_capi_bseed_expand_raw_device(const uint8_t* bseed,
                                       void* dst,
                                       int64_t n,
                                       void* stream);

int pearl_capi_bseed_expand_range_raw_device(const uint8_t* bseed,
                                             uint64_t byte_offset,
                                             void* dst,
                                             int64_t n,
                                             void* stream);

#ifdef __cplusplus
}  // extern "C"
#endif
