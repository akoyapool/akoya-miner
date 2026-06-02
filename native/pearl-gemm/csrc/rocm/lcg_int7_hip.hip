// lcg_int7_hip.hip — deterministic int7 A-fill on gfx942.
//
// Reproduces pearl_capi_lcg_int7_fill (per-iter A generation). Each output byte
// is in [-63,+63], a deterministic function of (seed_lo, seed_hi) via SplitMix64.
// Host-replayable — matches akoya-miner/scripts/lcg_int7_ref.py byte-for-byte.
//
// Build: hipcc --offload-arch=gfx942 -O3 lcg_int7_hip.hip -o lcg_int7_hip
// Use:   ./lcg_int7_hip <n> <seed_lo> <seed_hi> <out.bin>

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
using u64=uint64_t; using u32=uint32_t;
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  printf("HIP error %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__); exit(1);} }while(0)

__device__ __forceinline__ u64 splitmix64(u64 z){
  z = z + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// One thread per 8-byte group; thread 0 also does the (<8 byte) tail.
__global__ void k_lcg(int8_t* out, long n, u64 seed_lo, u64 seed_hi){
  u64 base = splitmix64(seed_lo ^ splitmix64(seed_hi));
  long n8 = n / 8;
  long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n8){
    u64 z = splitmix64(base + (u64)i);
    #pragma unroll
    for(int b=0;b<8;++b){ u32 v=(u32)((z>>(b*8))&0xFF); out[i*8+b]=(int8_t)((int)(v%127)-63); }
  }
  if (i==0 && (n%8)){
    u64 z = splitmix64(base + (u64)n8);
    long tail=n-n8*8;
    for(int b=0;b<tail;++b){ u32 v=(u32)((z>>(b*8))&0xFF); out[n8*8+b]=(int8_t)((int)(v%127)-63); }
  }
}

int main(int argc,char**argv){
  if(argc<5){ printf("usage: %s n seed_lo seed_hi out.bin\n",argv[0]); return 2; }
  long n=atol(argv[1]); u64 lo=strtoull(argv[2],0,0), hi=strtoull(argv[3],0,0);
  int8_t* d; CK(hipMalloc(&d, n?n:1));
  int n8=(int)(n/8); int blk=(n8+63)/64; if(blk<1)blk=1;
  hipLaunchKernelGGL(k_lcg,dim3(blk),dim3(64),0,0,d,n,lo,hi);
  CK(hipDeviceSynchronize());
  std::vector<int8_t> h(n); CK(hipMemcpy(h.data(),d,n,hipMemcpyDeviceToHost));
  FILE* f=fopen(argv[4],"wb"); fwrite(h.data(),1,n,f); fclose(f);
  return 0;
}
