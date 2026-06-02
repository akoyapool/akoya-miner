// blake3_hip.hip — BLAKE3 core + tensor_hash + commitment_hash on gfx942.
//
// CuTe-free HIP port of csrc/blake3/blake3.cuh. The BLAKE3 round/permute macros
// are reused VERBATIM (only the rState/rBlock accessors are redefined to plain
// arrays) so the arithmetic is byte-identical to the CUDA reference.
//
// Validates the full inline golden chain from
//   miner/pearl-pure-miner/tests/test_kernel_goldens.py
// entirely on the MI300X, reproducing GOLDEN_COMMITMENT_HASH_A/B.
//
// Build: hipcc --offload-arch=gfx942 -O3 blake3_hip.hip -o blake3_hip
// Run:   ./blake3_hip

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

using u32 = uint32_t;
using u64 = uint64_t;

#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  printf("HIP error %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__); return 1;} }while(0)

namespace b3 {
__device__ constexpr u32 IV0=0x6A09E667,IV1=0xBB67AE85,IV2=0x3C6EF372,IV3=0xA54FF53A,
                         IV4=0x510e527f,IV5=0x9b05688c,IV6=0x1f83d9ab,IV7=0x5be0cd19;
constexpr u32 CHUNK_START=1<<0, CHUNK_END=1<<1, PARENT=1<<2, ROOT=1<<3, KEYED_HASH=1<<4;
constexpr u32 MSG_BLOCK_SIZE=64, CHUNK_SIZE=1024;

__device__ __forceinline__ u32 add32(u32 x,u32 y){ return x+y; }
__device__ __forceinline__ u32 rightrotate32(u32 x,u32 n){ return (x<<(32-n))|(x>>n); }

// Accessor shims so the verbatim CUDA macros operate on plain register arrays.
#define rState(i)     state[i]
#define rBlock(i)     block[i]
#define rOrigBlock(i) origblock[i]

#include "blake3_rounds.inc"   // BLAKE3_ROUND / BLAKE3_PERMUTE, lifted verbatim

// Compress one 64-byte message block. cv[8] in/out; block[16] message words.
__device__ void compress(u32 cv[8], const u32 block_in[16], u64 counter,
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

__device__ __forceinline__ u32 load_le32(const uint8_t* p){
  return (u32)p[0] | ((u32)p[1]<<8) | ((u32)p[2]<<16) | ((u32)p[3]<<24);
}

// Single-chunk (<=1024 byte) keyed/unkeyed BLAKE3, ROOT output (32 bytes).
// key==nullptr → unkeyed (CV starts at IV, no KEYED_HASH flag).
__device__ void blake3_hash(const uint8_t* data, int len, const u32* key, uint8_t out[32]) {
  u32 cv[8];
  u32 base_flags = 0;
  if (key) { for (int i=0;i<8;++i) cv[i]=key[i]; base_flags = KEYED_HASH; }
  else     { cv[0]=IV0;cv[1]=IV1;cv[2]=IV2;cv[3]=IV3;cv[4]=IV4;cv[5]=IV5;cv[6]=IV6;cv[7]=IV7; }

  int nblocks = (len + 63) / 64; if (nblocks == 0) nblocks = 1;
  for (int b=0;b<nblocks;++b){
    u32 block[16];
    int boff = b*64;
    int blen = len - boff; if (blen < 0) blen = 0; if (blen > 64) blen = 64;
    uint8_t buf[64];
    for (int i=0;i<64;++i) buf[i] = (boff+i < len) ? data[boff+i] : 0;
    #pragma unroll
    for (int i=0;i<16;++i) block[i]=load_le32(buf+i*4);
    u32 flags = base_flags;
    if (b==0)          flags |= CHUNK_START;
    if (b==nblocks-1)  flags |= CHUNK_END | ROOT;
    compress(cv, block, /*counter*/0, (u32)blen, flags);
  }
  #pragma unroll
  for (int i=0;i<8;++i){ out[i*4]=cv[i]&0xff; out[i*4+1]=(cv[i]>>8)&0xff;
                         out[i*4+2]=(cv[i]>>16)&0xff; out[i*4+3]=(cv[i]>>24)&0xff; }
}
} // namespace b3

// Run the full inline-golden chain on-device (single thread).
__global__ void chain_kernel(const uint8_t* header,int hlen,const uint8_t* cfg,int clen,
                             const uint8_t* A,int alen,const uint8_t* B,int blen,
                             uint8_t* commitA,uint8_t* commitB){
  if (blockIdx.x|threadIdx.x) return;
  // key = BLAKE3(header || config)  (unkeyed)
  uint8_t keyin[256]; int kn=0;
  for(int i=0;i<hlen;++i) keyin[kn++]=header[i];
  for(int i=0;i<clen;++i) keyin[kn++]=cfg[i];
  uint8_t keyb[32]; b3::blake3_hash(keyin, kn, nullptr, keyb);
  u32 key[8]; for(int i=0;i<8;++i) key[i]=(u32)keyb[i*4]|((u32)keyb[i*4+1]<<8)|((u32)keyb[i*4+2]<<16)|((u32)keyb[i*4+3]<<24);

  // A_root, B_root = keyed BLAKE3 of raw bytes
  uint8_t A_root[32], B_root[32];
  b3::blake3_hash(A, alen, key, A_root);
  b3::blake3_hash(B, blen, key, B_root);

  // commitment_B = BLAKE3(keyb || B_root); commitment_A = BLAKE3(commitment_B || A_root)
  uint8_t buf[64];
  for(int i=0;i<32;++i) buf[i]=keyb[i];
  for(int i=0;i<32;++i) buf[32+i]=B_root[i];
  b3::blake3_hash(buf, 64, nullptr, commitB);
  for(int i=0;i<32;++i) buf[i]=commitB[i];
  for(int i=0;i<32;++i) buf[32+i]=A_root[i];
  b3::blake3_hash(buf, 64, nullptr, commitA);
}

static void hexcmp(const char* name,const uint8_t* got,const char* expect){
  char hex[65]; for(int i=0;i<32;++i) sprintf(hex+i*2,"%02x",got[i]); hex[64]=0;
  bool ok = (strcmp(hex,expect)==0);
  printf("  %-10s = %s  %s\n", name, hex, ok?"MATCH":"*** MISMATCH");
  printf("  %-10s   %s (expected)\n","", expect);
}

int main(){
  // Inline golden literals (test_kernel_goldens.py)
  uint8_t header[76]; for(int i=0;i<76;++i) header[i]=(uint8_t)i;
  uint8_t cfg[52]={0}; cfg[0]=0x08; cfg[4]=0x04;   // common_dim=8 (LE u32), rank=4 (LE u16), rest 0
  int8_t Av[32]={-63,-31,-1,0,1,7,31,63, 62,30,2,-2,-8,-32,-62,0, 5,-5,11,-11,17,-17,23,-23, 42,-42,13,-13,29,-29,3,-3};
  int8_t Bv[32]={3,1,-1,-3,5,-5,7,-7, -9,9,-11,11,-13,13,-15,15, 17,19,23,29,31,37,41,43, -17,-19,-23,-29,-31,-37,-41,-43};
  uint8_t A[32],B[32]; for(int i=0;i<32;++i){A[i]=(uint8_t)Av[i]; B[i]=(uint8_t)Bv[i];}
  const char* GA="8aa77a86172a5c981c79668575cdfccc1fbea42bda2c280b9e31e02b558ae4f0";
  const char* GB="a7e69996a00b2e5954f1df70ec70b9ae020817a281bae6c51e12548fbd275e68";

  uint8_t *dh,*dc,*dA,*dB,*dcA,*dcB;
  CK(hipMalloc(&dh,76)); CK(hipMalloc(&dc,52)); CK(hipMalloc(&dA,32)); CK(hipMalloc(&dB,32));
  CK(hipMalloc(&dcA,32)); CK(hipMalloc(&dcB,32));
  CK(hipMemcpy(dh,header,76,hipMemcpyHostToDevice));
  CK(hipMemcpy(dc,cfg,52,hipMemcpyHostToDevice));
  CK(hipMemcpy(dA,A,32,hipMemcpyHostToDevice));
  CK(hipMemcpy(dB,B,32,hipMemcpyHostToDevice));

  hipLaunchKernelGGL(chain_kernel,dim3(1),dim3(1),0,0,dh,76,dc,52,dA,32,dB,32,dcA,dcB);
  CK(hipDeviceSynchronize());

  uint8_t cA[32],cB[32];
  CK(hipMemcpy(cA,dcA,32,hipMemcpyDeviceToHost));
  CK(hipMemcpy(cB,dcB,32,hipMemcpyDeviceToHost));

  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p,0));
  printf("device: %s (%s)\n", p.name, p.gcnArchName);
  printf("BLAKE3 tensor_hash + commitment_hash inline-golden chain:\n");
  hexcmp("commit_A", cA, GA);
  hexcmp("commit_B", cB, GB);
  bool ok = true; { char h[65]; for(int i=0;i<32;++i) sprintf(h+i*2,"%02x",cA[i]); ok&=!strcmp(h,GA);
                    for(int i=0;i<32;++i) sprintf(h+i*2,"%02x",cB[i]); ok&=!strcmp(h,GB); }
  printf("\n%s\n", ok?"PASS — BLAKE3 byte-identical on MI300X.":"FAIL");
  return ok?0:1;
}
