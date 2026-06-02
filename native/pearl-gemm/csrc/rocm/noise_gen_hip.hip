// noise_gen_hip.hip — BLAKE3-keyed noise generation on gfx942.
//
// Reproduces pearl_capi_noise_gen / miner_base.NoiseGenerator:
//   - dense  (EAL[m,r], EBR[n,r]): byte -> (byte & 63) - 32   (noise_range=128)
//     fp16 variant = -value
//   - sparse (EAR, EBL): per line, +1 at first_idx, -1 at second_idx where
//     first = u & (R-1); second = first ^ (1 + mulhi(R-1, u)); R-major and
//     K-major layouts both written.
// Each hash = blake3(int32[8]{prepend=1+index} ‖ seed(32), key=key) — exactly
// the single-block keyed BLAKE3 from blake3_device.cuh (hash_small, len=64).
//
// Build: hipcc --offload-arch=gfx942 -O3 noise_gen_hip.hip -o noise_gen_hip
// Use:   ./noise_gen_hip <key_A.bin> <key_B.bin> <out_dir> [m n k r]

#include "blake3_device.cuh"
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  printf("HIP error %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__); exit(1);} }while(0)

__device__ inline void build_msg(uint8_t msg[64], int prepend_idx, int value,
                                 const uint8_t* seed){
  #pragma unroll
  for(int j=0;j<64;++j) msg[j]=0;
  msg[prepend_idx*4+0]=value&0xff; msg[prepend_idx*4+1]=(value>>8)&0xff;
  msg[prepend_idx*4+2]=(value>>16)&0xff; msg[prepend_idx*4+3]=(value>>24)&0xff;
  for(int j=0;j<32;++j) msg[32+j]=seed[j];
}

// Denoise fp16 scale factors (pearl_gemm_constants.hpp): EAL=-1, EBR=-(1<<14)/(1<<12)=-4.
static constexpr int kEALScaleFactorDenoise = -1 * (1<<12) / (1<<12);   // -1
static constexpr int kEBRScaleFactorDenoise = -1 * (1<<14) / (1<<12);   // -4

// Dense uniform random matrix [rows, R], row-major; fp16 out = value * scale.
__global__ void k_uniform(int8_t* out, __half* outscaled, int rows, int R, int scale,
                          const uint8_t* seed, const u32* key){
  int nbytes = rows*R, nhash=(nbytes+31)/32;
  int i = blockIdx.x*blockDim.x+threadIdx.x; if(i>=nhash) return;
  uint8_t msg[64]; build_msg(msg, 0, 1+i, seed);
  uint8_t h[32]; b3::hash_small(msg, 64, key, h);
  #pragma unroll
  for(int j=0;j<32;++j){ int idx=i*32+j; if(idx<nbytes){
    int8_t v=(int8_t)(((int)(h[j]&63))-32);
    out[idx]=v; outscaled[idx]=__float2half((float)((int)v * scale)); } }
}

// Sparse permutation matrix: req lines of length R; writes both layouts.
//   K_major [R, req]  (element [idx, line]),  R_major [req, R] (element [line, idx]).
__global__ void k_perm(int8_t* Km, int8_t* Rm, int req, int R,
                       const uint8_t* seed, const u32* key){
  int draws=(req*4+31)/32;
  int i = blockIdx.x*blockDim.x+threadIdx.x; if(i>=draws) return;
  uint8_t msg[64]; build_msg(msg, 1, 1+i, seed);
  uint8_t h[32]; b3::hash_small(msg, 64, key, h);
  #pragma unroll
  for(int kk=0;kk<8;++kk){
    int L=i*8+kk; if(L>=req) break;
    u32 u=b3::load_le32(h+kk*4);
    int fi=u & (R-1);
    int si=fi ^ (1 + (int)__umulhi((u32)(R-1), u));
    Km[(size_t)fi*req+L]=1;  Km[(size_t)si*req+L]=-1;
    Rm[(size_t)L*R+fi]=1;    Rm[(size_t)L*R+si]=-1;
  }
}

static std::vector<uint8_t> readfile(const char* p){
  FILE* f=fopen(p,"rb"); if(!f){printf("open %s\n",p);exit(2);} fseek(f,0,SEEK_END);
  long n=ftell(f); fseek(f,0,SEEK_SET); std::vector<uint8_t> v(n);
  if(fread(v.data(),1,n,f)!=(size_t)n){exit(2);} fclose(f); return v;
}
static void writefile(const std::string& p, const void* d, size_t n){
  FILE* f=fopen(p.c_str(),"wb"); fwrite(d,1,n,f); fclose(f);
}
static u32* upload_key(const std::vector<uint8_t>& kb){
  u32 hk[8]; for(int i=0;i<8;++i) hk[i]=(u32)kb[i*4]|((u32)kb[i*4+1]<<8)|((u32)kb[i*4+2]<<16)|((u32)kb[i*4+3]<<24);
  u32* d; CK(hipMalloc(&d,32)); CK(hipMemcpy(d,hk,32,hipMemcpyHostToDevice)); return d;
}

int main(int argc, char** argv){
  if(argc<4){ printf("usage: %s key_A.bin key_B.bin out_dir [m n k r]\n",argv[0]); return 2; }
  std::string od=argv[3];
  int m=128,n=256,k=128,r=64;
  if(argc>=8){ m=atoi(argv[4]); n=atoi(argv[5]); k=atoi(argv[6]); r=atoi(argv[7]); }

  auto kA=readfile(argv[1]); auto kB=readfile(argv[2]);
  u32 *dkA=upload_key(kA), *dkB=upload_key(kB);
  uint8_t sA[32]={0}, sB[32]={0}; memcpy(sA,"A_tensor",8); memcpy(sB,"B_tensor",8);
  uint8_t *dsA,*dsB; CK(hipMalloc(&dsA,32)); CK(hipMalloc(&dsB,32));
  CK(hipMemcpy(dsA,sA,32,hipMemcpyHostToDevice)); CK(hipMemcpy(dsB,sB,32,hipMemcpyHostToDevice));

  auto run_uniform=[&](int rows,const char* nm,const char* nmfp,int scale,u32* key,uint8_t* seed){
    int8_t* d; __half* df; CK(hipMalloc(&d,(size_t)rows*r)); CK(hipMalloc(&df,(size_t)rows*r*2));
    int nh=((rows*r)+31)/32; hipLaunchKernelGGL(k_uniform,dim3((nh+63)/64),dim3(64),0,0,d,df,rows,r,scale,seed,key);
    CK(hipDeviceSynchronize());
    std::vector<int8_t> h(rows*r); std::vector<uint16_t> hf(rows*r);
    CK(hipMemcpy(h.data(),d,(size_t)rows*r,hipMemcpyDeviceToHost));
    CK(hipMemcpy(hf.data(),df,(size_t)rows*r*2,hipMemcpyDeviceToHost));
    writefile(od+"/"+nm, h.data(), h.size()); writefile(od+"/"+nmfp, hf.data(), hf.size()*2);
    CK(hipFree(d)); CK(hipFree(df));
  };
  auto run_perm=[&](int req,const char* nmK,const char* nmR,u32* key,uint8_t* seed){
    int8_t *dK,*dR; CK(hipMalloc(&dK,(size_t)r*req)); CK(hipMalloc(&dR,(size_t)req*r));
    CK(hipMemset(dK,0,(size_t)r*req)); CK(hipMemset(dR,0,(size_t)req*r));
    int draws=(req*4+31)/32; hipLaunchKernelGGL(k_perm,dim3((draws+63)/64),dim3(64),0,0,dK,dR,req,r,seed,key);
    CK(hipDeviceSynchronize());
    std::vector<int8_t> hK(r*req), hR(req*r);
    CK(hipMemcpy(hK.data(),dK,(size_t)r*req,hipMemcpyDeviceToHost));
    CK(hipMemcpy(hR.data(),dR,(size_t)req*r,hipMemcpyDeviceToHost));
    writefile(od+"/"+nmK, hK.data(), hK.size()); writefile(od+"/"+nmR, hR.data(), hR.size());
    CK(hipFree(dK)); CK(hipFree(dR));
  };

  run_uniform(m, "EAL.bin", "EAL_fp16.bin", kEALScaleFactorDenoise, dkA, dsA);
  run_uniform(n, "EBR.bin", "EBR_fp16.bin", kEBRScaleFactorDenoise, dkB, dsB);
  run_perm(k, "EAR_K_major.bin", "EAR_R_major.bin", dkA, dsA);
  run_perm(k, "EBL_K_major.bin", "EBL_R_major.bin", dkB, dsB);
  printf("wrote 8 noise tensors to %s\n", od.c_str());
  return 0;
}
