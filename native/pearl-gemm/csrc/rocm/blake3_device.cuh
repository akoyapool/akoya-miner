// blake3_device.cuh — shared CuTe-free HIP BLAKE3 core (gfx942/gfx950).
//
// Arithmetic byte-identical to csrc/blake3/blake3.cuh (round/permute macros
// lifted verbatim). Provides the compress primitive plus BLAKE3 framing
// helpers: single-chunk hash, per-chunk leaf hash (counter = chunk index),
// and parent compression — i.e. standard keyed BLAKE3, which is what the
// Pearl tensor_hash computes for length-multiple-of-1024 inputs.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

using u32 = uint32_t;
using u64 = uint64_t;

namespace b3 {
__device__ constexpr u32 IV0=0x6A09E667,IV1=0xBB67AE85,IV2=0x3C6EF372,IV3=0xA54FF53A,
                         IV4=0x510e527f,IV5=0x9b05688c,IV6=0x1f83d9ab,IV7=0x5be0cd19;
constexpr u32 CHUNK_START=1<<0, CHUNK_END=1<<1, PARENT=1<<2, ROOT=1<<3, KEYED_HASH=1<<4;
constexpr u32 MSG_BLOCK_SIZE=64, CHUNK_SIZE=1024, BLOCKS_PER_CHUNK=16;

__device__ __forceinline__ u32 add32(u32 x,u32 y){ return x+y; }
__device__ __forceinline__ u32 rightrotate32(u32 x,u32 n){ return (x<<(32-n))|(x>>n); }

#define rState(i)     state[i]
#define rBlock(i)     block[i]
#define rOrigBlock(i) origblock[i]
#include "blake3_rounds.inc"

// Compress one 64-byte message block (block[16] words) into cv[8] (in/out).
__device__ inline void compress(u32 cv[8], const u32 block_in[16], u64 counter,
                                u32 block_len, u32 flags) {
  u32 state[16], block[16], origblock[16];
  #pragma unroll
  for (int i=0;i<16;++i) block[i]=block_in[i];
  #pragma unroll
  for (int i=0;i<8;++i)  state[i]=cv[i];
  state[8]=IV0; state[9]=IV1; state[10]=IV2; state[11]=IV3;
  state[12]=(u32)counter; state[13]=(u32)(counter>>32);
  state[14]=block_len; state[15]=flags;
  #pragma unroll
  for (int i=0;i<6;++i){ BLAKE3_ROUND(); BLAKE3_PERMUTE(); }
  BLAKE3_ROUND();
  #pragma unroll
  for (int i=0;i<8;++i) cv[i]=state[i]^state[i+8];
}

// Full 16-word output of one compression (for XOF / extendable output):
//   out[0..7]  = state[i] ^ state[i+8]
//   out[8..15] = state[i+8] ^ cv_in[i]
__device__ inline void compress_full(const u32 cv_in[8], const u32 block_in[16],
                                     u64 counter, u32 block_len, u32 flags, u32 out16[16]){
  u32 state[16], block[16], origblock[16];
  #pragma unroll
  for (int i=0;i<16;++i) block[i]=block_in[i];
  #pragma unroll
  for (int i=0;i<8;++i)  state[i]=cv_in[i];
  state[8]=IV0; state[9]=IV1; state[10]=IV2; state[11]=IV3;
  state[12]=(u32)counter; state[13]=(u32)(counter>>32);
  state[14]=block_len; state[15]=flags;
  #pragma unroll
  for (int i=0;i<6;++i){ BLAKE3_ROUND(); BLAKE3_PERMUTE(); }
  BLAKE3_ROUND();
  #pragma unroll
  for (int i=0;i<8;++i){ out16[i]=state[i]^state[i+8]; out16[i+8]=state[i+8]^cv_in[i]; }
}

__device__ __forceinline__ u32 load_le32(const uint8_t* p){
  return (u32)p[0] | ((u32)p[1]<<8) | ((u32)p[2]<<16) | ((u32)p[3]<<24);
}
__device__ __forceinline__ void init_cv(u32 cv[8], const u32* key){
  if (key){ for(int i=0;i<8;++i) cv[i]=key[i]; }
  else { cv[0]=IV0;cv[1]=IV1;cv[2]=IV2;cv[3]=IV3;cv[4]=IV4;cv[5]=IV5;cv[6]=IV6;cv[7]=IV7; }
}
__device__ __forceinline__ void cv_to_bytes(const u32 cv[8], uint8_t out[32]){
  #pragma unroll
  for(int i=0;i<8;++i){ out[i*4]=cv[i]&0xff; out[i*4+1]=(cv[i]>>8)&0xff;
                        out[i*4+2]=(cv[i]>>16)&0xff; out[i*4+3]=(cv[i]>>24)&0xff; }
}

// Single-chunk (<=1024 byte) keyed/unkeyed BLAKE3 ROOT hash of raw bytes.
// key==nullptr → unkeyed. block_len of last block = actual remaining bytes.
__device__ inline void hash_small(const uint8_t* data, int len, const u32* key, uint8_t out[32]){
  u32 cv[8]; init_cv(cv, key);
  u32 base = key ? KEYED_HASH : 0;
  int nblocks=(len+63)/64; if(nblocks==0) nblocks=1;
  for(int b=0;b<nblocks;++b){
    u32 block[16]; int boff=b*64; int blen=len-boff; if(blen<0)blen=0; if(blen>64)blen=64;
    uint8_t buf[64]; for(int i=0;i<64;++i) buf[i]=(boff+i<len)?data[boff+i]:0;
    #pragma unroll
    for(int i=0;i<16;++i) block[i]=load_le32(buf+i*4);
    u32 f=base; if(b==0)f|=CHUNK_START; if(b==nblocks-1)f|=CHUNK_END|ROOT;
    compress(cv, block, 0, (u32)blen, f);
  }
  cv_to_bytes(cv,out);
}

// Hash one full 1024-byte chunk (16 full blocks) at chunk index `idx`.
// is_root=true only when this chunk is the entire input (single-chunk file).
__device__ inline void chunk_cv(const uint8_t* chunk, u64 idx, const u32* key,
                                bool is_root, u32 out_cv[8]){
  u32 cv[8]; init_cv(cv, key);
  u32 base = (key?KEYED_HASH:0);
  for(int b=0;b<16;++b){
    u32 block[16];
    #pragma unroll
    for(int i=0;i<16;++i) block[i]=load_le32(chunk+b*64+i*4);
    u32 f=base; if(b==0)f|=CHUNK_START; if(b==15){ f|=CHUNK_END; if(is_root)f|=ROOT; }
    compress(cv, block, idx, MSG_BLOCK_SIZE, f);
  }
  #pragma unroll
  for(int i=0;i<8;++i) out_cv[i]=cv[i];
}

// Parent node: compress(left_cv ++ right_cv) with PARENT (+ROOT at the top).
__device__ inline void parent_cv(const u32 l[8], const u32 r[8], const u32* key,
                                 bool is_root, u32 out_cv[8]){
  u32 cv[8]; init_cv(cv, key);
  u32 block[16];
  #pragma unroll
  for(int i=0;i<8;++i){ block[i]=l[i]; block[i+8]=r[i]; }
  u32 f=(key?KEYED_HASH:0)|PARENT; if(is_root)f|=ROOT;
  compress(cv, block, 0, MSG_BLOCK_SIZE, f);
  #pragma unroll
  for(int i=0;i<8;++i) out_cv[i]=cv[i];
}
} // namespace b3
