// bseed_expand_hip.hip — BSeed → int7 B expansion on gfx942.
//
// Reproduces pearl_capi_bseed_expand_raw_device / Akoya.Crypto.BSeedExpander:
//   stream = BLAKE3_XOF(bSeed)               (unkeyed, 32-byte seed)
//   B[i]   = (int8)((stream[i] % 127) - 63)  in [-63,+63], row-major
//
// XOF: the 32-byte seed is one ≤64-byte chunk → its single ROOT block is the
// output node. Output block j re-runs that compression with output counter = j,
// emitting the full 64-byte state (compress_full). Matches blake3(seed).digest(n).
//
// Build: hipcc --offload-arch=gfx942 -O3 bseed_expand_hip.hip -o bseed_expand_hip
// Use:   ./bseed_expand_hip <bseed.bin(32)> <n_bytes> <out.bin>

#include "blake3_device.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  printf("HIP error %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__); exit(1);} }while(0)

// One thread per 64-byte XOF output block.
__global__ void k_bseed(const u32* seed_words, int8_t* out, long n){
  long j = (long)blockIdx.x*blockDim.x + threadIdx.x;     // output block index
  long nblk = (n + 63) / 64;
  if (j >= nblk) return;

  u32 cv_in[8]; b3::init_cv(cv_in, nullptr);              // unkeyed → IV
  u32 msg[16];
  #pragma unroll
  for (int i=0;i<8;++i)  msg[i]=seed_words[i];            // 32-byte seed
  #pragma unroll
  for (int i=8;i<16;++i) msg[i]=0;

  u32 o[16];
  b3::compress_full(cv_in, msg, /*counter=*/(u64)j, /*block_len=*/32,
                    b3::CHUNK_START | b3::CHUNK_END | b3::ROOT, o);

  long base = j*64;
  #pragma unroll
  for (int w=0; w<16; ++w){
    u32 v=o[w];
    #pragma unroll
    for (int b=0;b<4;++b){
      long idx = base + w*4 + b;
      if (idx < n){ u32 byte=(v>>(b*8))&0xFF; out[idx]=(int8_t)((int)(byte%127)-63); }
    }
  }
}

int main(int argc,char**argv){
  if(argc<4){printf("usage: %s bseed.bin n_bytes out.bin\n",argv[0]);return 2;}
  FILE* f=fopen(argv[1],"rb"); if(!f){printf("open %s\n",argv[1]);return 2;}
  uint8_t seed[32]; if(fread(seed,1,32,f)!=32){return 2;} fclose(f);
  long n=atol(argv[2]);
  u32 sw[8]; for(int i=0;i<8;++i) sw[i]=(u32)seed[i*4]|((u32)seed[i*4+1]<<8)|((u32)seed[i*4+2]<<16)|((u32)seed[i*4+3]<<24);
  u32* dsw; CK(hipMalloc(&dsw,32)); CK(hipMemcpy(dsw,sw,32,hipMemcpyHostToDevice));
  int8_t* d; CK(hipMalloc(&d,n?n:1));
  long nblk=(n+63)/64; hipLaunchKernelGGL(k_bseed,dim3((nblk+63)/64),dim3(64),0,0,dsw,d,n);
  CK(hipDeviceSynchronize());
  std::vector<int8_t> h(n); CK(hipMemcpy(h.data(),d,n,hipMemcpyDeviceToHost));
  FILE* o=fopen(argv[3],"wb"); fwrite(h.data(),1,n,o); fclose(o);
  return 0;
}
