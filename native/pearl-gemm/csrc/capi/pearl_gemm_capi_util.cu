// pearl_gemm_capi_util.cu — small utility kernels for the C-ABI shim.
//
// Currently:
//   pearl_capi_lcg_int7_fill — deterministic int7 ([-63, +63]) fill of a
//                              device buffer keyed by (seed_lo, seed_hi).
//                              Used by akoya-miner to (re)generate per-iter
//                              A matrices on-device. Both seeds are passed
//                              through SplitMix64 mixing so caller can
//                              freely use ``seed_hi=σ_first8`` and
//                              ``seed_lo=iter_idx`` (or any pair) without
//                              worrying about correlation.
//
//                              The same algorithm can be replayed on the
//                              host (C#) when assembling a proof to recover
//                              the exact A used at any iteration without
//                              keeping per-iter snapshot buffers.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace {

constexpr uint32_t B3_CHUNK_START = 1u << 0;
constexpr uint32_t B3_CHUNK_END = 1u << 1;
constexpr uint32_t B3_ROOT = 1u << 3;
constexpr uint32_t B3_IV0 = 0x6A09E667u;
constexpr uint32_t B3_IV1 = 0xBB67AE85u;
constexpr uint32_t B3_IV2 = 0x3C6EF372u;
constexpr uint32_t B3_IV3 = 0xA54FF53Au;
constexpr uint32_t B3_IV4 = 0x510E527Fu;
constexpr uint32_t B3_IV5 = 0x9B05688Cu;
constexpr uint32_t B3_IV6 = 0x1F83D9ABu;
constexpr uint32_t B3_IV7 = 0x5BE0CD19u;

struct BSeedBlock {
  uint32_t word[16];
};

static inline uint32_t load_le32_host(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) |
         (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ uint32_t b3_rotr32(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ void b3_g(
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
    uint32_t mx, uint32_t my) {
  a = a + b + mx;
  d = b3_rotr32(d ^ a, 16);
  c = c + d;
  b = b3_rotr32(b ^ c, 12);
  a = a + b + my;
  d = b3_rotr32(d ^ a, 8);
  c = c + d;
  b = b3_rotr32(b ^ c, 7);
}

__device__ __forceinline__ void b3_round(uint32_t s[16], const uint32_t m[16]) {
  b3_g(s[0], s[4], s[8],  s[12], m[0],  m[1]);
  b3_g(s[1], s[5], s[9],  s[13], m[2],  m[3]);
  b3_g(s[2], s[6], s[10], s[14], m[4],  m[5]);
  b3_g(s[3], s[7], s[11], s[15], m[6],  m[7]);
  b3_g(s[0], s[5], s[10], s[15], m[8],  m[9]);
  b3_g(s[1], s[6], s[11], s[12], m[10], m[11]);
  b3_g(s[2], s[7], s[8],  s[13], m[12], m[13]);
  b3_g(s[3], s[4], s[9],  s[14], m[14], m[15]);
}

__device__ __forceinline__ void b3_permute(uint32_t m[16]) {
  uint32_t t0 = m[0],  t1 = m[1],  t2 = m[2],  t3 = m[3];
  uint32_t t4 = m[4],  t5 = m[5],  t6 = m[6],  t7 = m[7];
  uint32_t t8 = m[8],  t9 = m[9],  t10 = m[10], t11 = m[11];
  uint32_t t12 = m[12], t13 = m[13], t14 = m[14], t15 = m[15];
  m[0] = t2;   m[1] = t6;   m[2] = t3;   m[3] = t10;
  m[4] = t7;   m[5] = t0;   m[6] = t4;   m[7] = t13;
  m[8] = t1;   m[9] = t11;  m[10] = t12; m[11] = t5;
  m[12] = t9;  m[13] = t14; m[14] = t15; m[15] = t8;
}

__device__ __forceinline__ void bseed_xof_compress(
    const BSeedBlock& seed_block,
    uint64_t counter,
    uint32_t out[16]) {
  uint32_t s[16];
  uint32_t m[16];

  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    m[i] = seed_block.word[i];
  }

  s[0] = B3_IV0; s[1] = B3_IV1; s[2] = B3_IV2; s[3] = B3_IV3;
  s[4] = B3_IV4; s[5] = B3_IV5; s[6] = B3_IV6; s[7] = B3_IV7;
  s[8] = B3_IV0; s[9] = B3_IV1; s[10] = B3_IV2; s[11] = B3_IV3;
  s[12] = static_cast<uint32_t>(counter);
  s[13] = static_cast<uint32_t>(counter >> 32);
  s[14] = 32u;
  s[15] = B3_CHUNK_START | B3_CHUNK_END | B3_ROOT;

  #pragma unroll
  for (int round = 0; round < 6; ++round) {
    b3_round(s, m);
    b3_permute(m);
  }
  b3_round(s, m);

  out[0] = s[0] ^ s[8];
  out[1] = s[1] ^ s[9];
  out[2] = s[2] ^ s[10];
  out[3] = s[3] ^ s[11];
  out[4] = s[4] ^ s[12];
  out[5] = s[5] ^ s[13];
  out[6] = s[6] ^ s[14];
  out[7] = s[7] ^ s[15];
  out[8] = s[8] ^ B3_IV0;
  out[9] = s[9] ^ B3_IV1;
  out[10] = s[10] ^ B3_IV2;
  out[11] = s[11] ^ B3_IV3;
  out[12] = s[12] ^ B3_IV4;
  out[13] = s[13] ^ B3_IV5;
  out[14] = s[14] ^ B3_IV6;
  out[15] = s[15] ^ B3_IV7;
}

__device__ __forceinline__ uint8_t bseed_map_byte(uint32_t x) {
  return static_cast<uint8_t>(static_cast<int32_t>(x % 127u) - 63);
}

__device__ __forceinline__ uint32_t bseed_pack_mapped_word(uint32_t x) {
  uint32_t b0 = bseed_map_byte(x & 0xFFu);
  uint32_t b1 = bseed_map_byte((x >> 8) & 0xFFu);
  uint32_t b2 = bseed_map_byte((x >> 16) & 0xFFu);
  uint32_t b3 = bseed_map_byte((x >> 24) & 0xFFu);
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

__device__ __forceinline__ uint8_t bseed_xof_byte(const uint32_t out[16], int index) {
  uint32_t word = out[index >> 2];
  return static_cast<uint8_t>((word >> ((index & 3) * 8)) & 0xFFu);
}

__global__ void bseed_expand_int7_kernel(
    int8_t* __restrict__ dst,
    size_t n,
    uint64_t byte_offset,
    BSeedBlock seed_block) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t chunks = (n + 63u) / 64u;

  for (size_t chunk = tid; chunk < chunks; chunk += stride) {
    size_t dst_off = chunk * 64u;
    size_t remaining = n - dst_off;
    size_t take = remaining < 64u ? remaining : 64u;
    uint64_t xof_off = byte_offset + static_cast<uint64_t>(dst_off);
    uint64_t counter = xof_off >> 6;
    int skip = static_cast<int>(xof_off & 63u);

    uint32_t out[16];
    bseed_xof_compress(seed_block, counter, out);

    if (skip == 0 && take == 64u) {
      uint32_t* out32 = reinterpret_cast<uint32_t*>(dst + dst_off);
      #pragma unroll
      for (int w = 0; w < 16; ++w) {
        out32[w] = bseed_pack_mapped_word(out[w]);
      }
      continue;
    }

    size_t written = 0;
    int idx = skip;
    while (written < take && idx < 64) {
      dst[dst_off + written] =
          static_cast<int8_t>(bseed_map_byte(bseed_xof_byte(out, idx)));
      ++written;
      ++idx;
    }
    if (written < take) {
      bseed_xof_compress(seed_block, counter + 1, out);
      idx = 0;
      while (written < take) {
        dst[dst_off + written] =
            static_cast<int8_t>(bseed_map_byte(bseed_xof_byte(out, idx)));
        ++written;
        ++idx;
      }
    }
  }
}

// SplitMix64 — single-step, used both for thread/element seeding and
// per-byte expansion. Known good distribution; trivially replayable on
// host.
__device__ __forceinline__ uint64_t splitmix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

__global__ void lcg_int7_fill_kernel(int8_t* __restrict__ dst,
                                     size_t n,
                                     uint64_t seed_lo,
                                     uint64_t seed_hi) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  // Mix seeds once up front so every output is a function of both.
  uint64_t base = splitmix64(seed_lo ^ splitmix64(seed_hi));

  // Process 8 bytes per step: one splitmix64 → 8 int7 values. This keeps
  // arithmetic intensity high and matches how the host replay walks 8
  // bytes per step.
  size_t n8 = n / 8;
  for (size_t i = tid; i < n8; i += stride) {
    uint64_t z = splitmix64(base + i);
    int8_t out[8];
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
      uint32_t v = static_cast<uint32_t>((z >> (b * 8)) & 0xFFu);
      // 256 → [-63, +63] (127 values) via mod-127, well-distributed: bias
      // is < 1 part in ~64M, immaterial for proof contents.
      uint32_t r = v % 127u;
      out[b] = static_cast<int8_t>(static_cast<int32_t>(r) - 63);
    }
    // 8-byte aligned store when n8 alignment permits.
    *reinterpret_cast<int2*>(dst + i * 8) =
        *reinterpret_cast<const int2*>(out);
  }

  // Trailing 0..7 bytes: use a separate splitmix call so host replay
  // matches.
  size_t tail_off = n8 * 8;
  size_t tail_len = n - tail_off;
  if (tail_len > 0 && tid == 0) {
    uint64_t z = splitmix64(base + n8);
    for (size_t b = 0; b < tail_len; ++b) {
      uint32_t v = static_cast<uint32_t>((z >> (b * 8)) & 0xFFu);
      uint32_t r = v % 127u;
      dst[tail_off + b] = static_cast<int8_t>(static_cast<int32_t>(r) - 63);
    }
  }
}

__global__ void lcg_int7_fill_indirect_kernel(
    int8_t* __restrict__ dst,
    size_t n,
    const uint64_t* __restrict__ seed_lo_base,
    uint64_t seed_lo_offset,
    uint64_t seed_hi) {
  uint64_t seed_lo = *seed_lo_base + seed_lo_offset;
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  uint64_t base = splitmix64(seed_lo ^ splitmix64(seed_hi));

  size_t n8 = n / 8;
  for (size_t i = tid; i < n8; i += stride) {
    uint64_t z = splitmix64(base + i);
    int8_t out[8];
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
      uint32_t v = static_cast<uint32_t>((z >> (b * 8)) & 0xFFu);
      uint32_t r = v % 127u;
      out[b] = static_cast<int8_t>(static_cast<int32_t>(r) - 63);
    }
    *reinterpret_cast<int2*>(dst + i * 8) =
        *reinterpret_cast<const int2*>(out);
  }

  size_t tail_off = n8 * 8;
  size_t tail_len = n - tail_off;
  if (tail_len > 0 && tid == 0) {
    uint64_t z = splitmix64(base + n8);
    for (size_t b = 0; b < tail_len; ++b) {
      uint32_t v = static_cast<uint32_t>((z >> (b * 8)) & 0xFFu);
      uint32_t r = v % 127u;
      dst[tail_off + b] = static_cast<int8_t>(static_cast<int32_t>(r) - 63);
    }
  }
}

}  // namespace

// Symbol export: on Windows the DLL's exports are generated by CMake
// (WINDOWS_EXPORT_ALL_SYMBOLS), so this is a no-op there; elsewhere the build
// compiles with -fvisibility=hidden, so the attribute makes the C ABI visible.
#if defined(_WIN32)
#  define PEARL_CAPI_EXPORT
#else
#  define PEARL_CAPI_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

PEARL_CAPI_EXPORT
int pearl_capi_lcg_int7_fill(void* dst,
                             int64_t n,
                             uint64_t seed_lo,
                             uint64_t seed_hi,
                             void* stream) {
  if (dst == nullptr) return -1;
  if (n <= 0) return 0;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  constexpr int kThreads = 256;
  size_t n8 = static_cast<size_t>(n) / 8;
  if (n8 == 0) n8 = 1;
  size_t needed = (n8 + kThreads - 1) / kThreads;
  // Cap blocks so we don't oversubscribe — H100 has 132 SMs.
  int blocks = static_cast<int>(std::min<size_t>(needed, 1024));
  if (blocks < 1) blocks = 1;
  lcg_int7_fill_kernel<<<blocks, kThreads, 0, s>>>(
      static_cast<int8_t*>(dst),
      static_cast<size_t>(n),
      seed_lo,
      seed_hi);
  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? 0 : -static_cast<int>(err);
}

PEARL_CAPI_EXPORT
int pearl_capi_lcg_int7_fill_indirect(void* dst,
                                      int64_t n,
                                      const void* seed_lo_base,
                                      uint64_t seed_lo_offset,
                                      uint64_t seed_hi,
                                      void* stream) {
  if (dst == nullptr || seed_lo_base == nullptr) return -1;
  if (n <= 0) return 0;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  constexpr int kThreads = 256;
  size_t n8 = static_cast<size_t>(n) / 8;
  if (n8 == 0) n8 = 1;
  size_t needed = (n8 + kThreads - 1) / kThreads;
  int blocks = static_cast<int>(std::min<size_t>(needed, 1024));
  if (blocks < 1) blocks = 1;
  lcg_int7_fill_indirect_kernel<<<blocks, kThreads, 0, s>>>(
      static_cast<int8_t*>(dst),
      static_cast<size_t>(n),
      static_cast<const uint64_t*>(seed_lo_base),
      seed_lo_offset,
      seed_hi);
  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? 0 : -static_cast<int>(err);
}

PEARL_CAPI_EXPORT
int pearl_capi_bseed_expand_range_raw_device(const uint8_t* bseed,
                                             uint64_t byte_offset,
                                             void* dst,
                                             int64_t n,
                                             void* stream) {
  if (bseed == nullptr || dst == nullptr) return -1;
  if (n <= 0) return 0;

  BSeedBlock seed_block{};
  for (int i = 0; i < 8; ++i) {
    seed_block.word[i] = load_le32_host(bseed + i * 4);
  }

  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  constexpr int kThreads = 256;
  size_t chunks = (static_cast<size_t>(n) + 63u) / 64u;
  size_t needed = (chunks + kThreads - 1) / kThreads;
  int blocks = static_cast<int>(std::min<size_t>(needed, 4096));
  if (blocks < 1) blocks = 1;

  bseed_expand_int7_kernel<<<blocks, kThreads, 0, s>>>(
      static_cast<int8_t*>(dst),
      static_cast<size_t>(n),
      byte_offset,
      seed_block);
  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? 0 : -static_cast<int>(err);
}

PEARL_CAPI_EXPORT
int pearl_capi_bseed_expand_raw_device(const uint8_t* bseed,
                                       void* dst,
                                       int64_t n,
                                       void* stream) {
  return pearl_capi_bseed_expand_range_raw_device(
      bseed, 0, dst, n, stream);
}

}  // extern "C"
