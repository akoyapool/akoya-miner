// noisy_gemm_hip.hip — production noisy-GEMM numerics on gfx942 (rocWMMA).
//
// PRODUCTION path only (mirrors the hot path: skip_denoising / skip_c_store).
// Computes exactly what mining needs:
//   ApEA[m,k]      = A + int8(EAL · EAR_K)                (noise_A)
//   BpEB[n,k]      = (Bᵀ + int8(EBL_R · EBRᵀ))ᵀ           (B install)
//   C_noised[m,n]  = ApEA · BpEBᵀ_as[k,n]  (int32)        (the GEMM → transcript)
// NO denoise, NO bf16 C store, NO AxEBL/EARxBpEB fp16 terms — those are
// validation-only. Transposes here exist purely to fit layouts.
//
// All matmuls are NN row-major [M,K]·[K,N]→int32 via one rocWMMA helper.
//
// Build: hipcc --offload-arch=gfx942 -O3 noisy_gemm_hip.hip -o noisy_gemm_hip
// Use:   ./noisy_gemm_hip <indir> <outdir> [m n k r]   (indir holds the input .bin files)

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using namespace rocwmma;
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  printf("HIP error %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__); exit(1);} }while(0)

// C[M,N] = A[M,K] @ B[K,N], all row-major int8 -> int32. One warp per 16x16 tile.
__global__ void wmma_gemm(const int8_t* A,const int8_t* B,int32_t* C,int M,int N,int K){
  int tM=blockIdx.y, tN=blockIdx.x;
  fragment<matrix_a,16,16,16,int8_t,row_major> a;
  fragment<matrix_b,16,16,16,int8_t,row_major> b;
  fragment<accumulator,16,16,16,int32_t> c;
  fill_fragment(c,0);
  for(int kk=0;kk<K;kk+=16){
    load_matrix_sync(a, A + (tM*16)*K + kk, K);
    load_matrix_sync(b, B + kk*N + tN*16, N);
    mma_sync(c,a,b,c);
  }
  store_matrix_sync(C + (tM*16)*N + tN*16, c, N, mem_row_major);
}
static void gemm(const int8_t* A,const int8_t* B,int32_t* C,int M,int N,int K){
  dim3 g(N/16,M/16); hipLaunchKernelGGL(wmma_gemm,g,dim3(64),0,0,A,B,C,M,N,K);
}

__global__ void k_transpose_i8(const int8_t* in,int8_t* out,int rows,int cols){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*cols)return;
  int r=i/cols,c=i%cols; out[(size_t)c*rows+r]=in[i];           // out[cols,rows]
}
// ApEA[m,k] = A + int8(E_A[m,k])
__global__ void k_add_i8(const int8_t* A,const int32_t* E,int8_t* out,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  out[i]=(int8_t)((int)A[i] + (int)(int8_t)E[i]);
}

static std::vector<uint8_t> rd(const std::string& p){
  FILE* f=fopen(p.c_str(),"rb"); if(!f){printf("open %s\n",p.c_str());exit(2);}
  fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);std::vector<uint8_t> v(n);
  if(fread(v.data(),1,n,f)!=(size_t)n)exit(2); fclose(f); return v;
}
static void wr(const std::string& p,const void* d,size_t n){FILE* f=fopen(p.c_str(),"wb");fwrite(d,1,n,f);fclose(f);}
template<class T> static T* up(const std::vector<uint8_t>& h){T* d;CK(hipMalloc(&d,h.size()));CK(hipMemcpy(d,h.data(),h.size(),hipMemcpyHostToDevice));return d;}

int main(int argc,char**argv){
  if(argc<3){printf("usage: %s indir outdir [m n k r]\n",argv[0]);return 2;}
  std::string in=argv[1], out=argv[2];
  int m=128,n=256,k=128,r=64;
  if(argc>=7){m=atoi(argv[3]);n=atoi(argv[4]);k=atoi(argv[5]);r=atoi(argv[6]);}

  // Inputs (row-major): A[m,k], B[n,k], EAL[m,r], EBR[n,r], EAR_K[r,k], EBL_R[k,r]
  int8_t* dA   = up<int8_t>(rd(in+"/A.bin"));
  int8_t* dB   = up<int8_t>(rd(in+"/B.bin"));         // [n,k]
  int8_t* dEAL = up<int8_t>(rd(in+"/EAL.bin"));       // [m,r]
  int8_t* dEBRn= up<int8_t>(rd(in+"/EBR.bin"));       // [n,r]
  int8_t* dEARk= up<int8_t>(rd(in+"/EAR_K.bin"));     // [r,k]
  int8_t* dEBLr= up<int8_t>(rd(in+"/EBL_R.bin"));     // [k,r]
  auto alloc=[&](size_t nb){void*p;CK(hipMalloc(&p,nb));return p;};
  int32_t* dEA =(int32_t*)alloc((size_t)m*k*4);
  int8_t*  dApEA=(int8_t*)alloc((size_t)m*k);
  int8_t*  dEBR =(int8_t*)alloc((size_t)r*n);   // E_BR[r,n] = transpose(EBR[n,r])
  int32_t* dEB =(int32_t*)alloc((size_t)k*n*4);
  int8_t*  dBkn=(int8_t*)alloc((size_t)k*n);    // Bᵀ[k,n]
  int8_t*  dBnoi=(int8_t*)alloc((size_t)k*n);   // Bnoised[k,n]
  int8_t*  dBpEB=(int8_t*)alloc((size_t)n*k);   // BpEB[n,k]
  int32_t* dC  =(int32_t*)alloc((size_t)m*n*4);
  auto E=[&](int t){return dim3((t+255)/256);};

  // noise_A: E_A = EAL[m,r] @ EAR_K[r,k]; ApEA = A + int8(E_A)
  gemm(dEAL,dEARk,dEA,m,k,r);
  hipLaunchKernelGGL(k_add_i8,E(m*k),dim3(256),0,0,dA,dEA,dApEA,m*k); CK(hipDeviceSynchronize());

  // B install: E_BR = transpose(EBR[n,r])->[r,n]; E_B = EBL_R[k,r] @ E_BR[r,n]
  hipLaunchKernelGGL(k_transpose_i8,E(n*r),dim3(256),0,0,dEBRn,dEBR,n,r); CK(hipDeviceSynchronize());
  gemm(dEBLr,dEBR,dEB,k,n,r);
  // Bkn = transpose(B[n,k])->[k,n]; Bnoised = Bkn + int8(E_B); BpEB = transpose(Bnoised)->[n,k]
  hipLaunchKernelGGL(k_transpose_i8,E(n*k),dim3(256),0,0,dB,dBkn,n,k); CK(hipDeviceSynchronize());
  hipLaunchKernelGGL(k_add_i8,E(k*n),dim3(256),0,0,dBkn,dEB,dBnoi,k*n); CK(hipDeviceSynchronize());
  hipLaunchKernelGGL(k_transpose_i8,E(k*n),dim3(256),0,0,dBnoi,dBpEB,k,n); CK(hipDeviceSynchronize());

  // main GEMM (production): C_noised = ApEA[m,k] @ Bnoised[k,n]
  gemm(dApEA,dBnoi,dC,m,n,k); CK(hipDeviceSynchronize());

  std::vector<int8_t> hApEA(m*k),hBpEB(n*k); std::vector<int32_t> hC(m*n);
  CK(hipMemcpy(hApEA.data(),dApEA,(size_t)m*k,hipMemcpyDeviceToHost));
  CK(hipMemcpy(hBpEB.data(),dBpEB,(size_t)n*k,hipMemcpyDeviceToHost));
  CK(hipMemcpy(hC.data(),dC,(size_t)m*n*4,hipMemcpyDeviceToHost));
  wr(out+"/ApEA.bin",hApEA.data(),hApEA.size());
  wr(out+"/BpEB.bin",hBpEB.data(),hBpEB.size());
  wr(out+"/C_noised.bin",hC.data(),hC.size()*4);
  printf("wrote ApEA, BpEB, C_noised to %s\n",out.c_str());
  return 0;
}
