// pearl_iter_hip.hip — fused per-nonce iteration on gfx942.
//
// Mirrors pearl_capi_iter's hot path exactly:
//   σ-install (once): BHash=tensor_hash(B,Key); CommitB=BLAKE3(Key‖BHash);
//                     B-side noise keyed by CommitB → Bnoised[k,n].
//   per nonce:        A=lcg(seed_lo,σ) → AHash=tensor_hash(A,Key) →
//                     CommitA=BLAKE3(CommitB‖AHash) → A-side noise keyed by
//                     CommitA → ApEA → transcript(ApEA·Bnoised) → keyed-BLAKE3
//                     vs pow_target.
// Reuses the validated kernels in pearl_kernels.cuh. Validated against an
// identical numpy chain (verify_iter.py).
//
// Build: hipcc --offload-arch=gfx942 -O3 pearl_iter_hip.hip -o pearl_iter_hip
// Use:   ./pearl_iter_hip <indir> <outdir> <nonce_start> <count> [m n k r]
//   indir: Key.bin(32), B.bin(n*k i8), pow_key.bin(32), pow_target.bin(32)

#include "pearl_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  printf("HIP %s @%d\n",hipGetErrorString(e),__LINE__); exit(1);} }while(0)
static std::vector<uint8_t> rd(const std::string&p){FILE*f=fopen(p.c_str(),"rb");if(!f){printf("open %s\n",p.c_str());exit(2);}
  fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);std::vector<uint8_t>v(n);if(fread(v.data(),1,n,f)!=(size_t)n)exit(2);fclose(f);return v;}
static void wr(const std::string&p,const void*d,size_t n){FILE*f=fopen(p.c_str(),"wb");fwrite(d,1,n,f);fclose(f);}
template<class T> static T* dev(size_t nb){T*p;CK(hipMalloc(&p,nb?nb:1));return p;}
static u32* keywords(const uint8_t* kb){u32 h[8];for(int i=0;i<8;++i)h[i]=(u32)kb[i*4]|((u32)kb[i*4+1]<<8)|((u32)kb[i*4+2]<<16)|((u32)kb[i*4+3]<<24);
  u32*d;CK(hipMalloc(&d,32));CK(hipMemcpy(d,h,32,hipMemcpyHostToDevice));return d;}
static dim3 E(int t){return dim3((t+255)/256);}

int main(int argc,char**argv){
  if(argc<5){printf("usage: %s indir outdir nonce_start count [m n k r]\n",argv[0]);return 2;}
  std::string in=argv[1],out=argv[2]; u64 nonce0=strtoull(argv[3],0,0); int count=atoi(argv[4]);
  int m=128,n=256,k=128,r=64; if(argc>=9){m=atoi(argv[5]);n=atoi(argv[6]);k=atoi(argv[7]);r=atoi(argv[8]);}
  const int EALscale=-1; // not used in transcript path

  auto Key=rd(in+"/Key.bin"); auto Bh=rd(in+"/B.bin");
  uint8_t* dKey=dev<uint8_t>(32); CK(hipMemcpy(dKey,Key.data(),32,hipMemcpyHostToDevice));
  u32* dKeyW=keywords(Key.data());  // Key as 8 little-endian words for tensor_hash
  int8_t* dB=dev<int8_t>((size_t)n*k); CK(hipMemcpy(dB,Bh.data(),(size_t)n*k,hipMemcpyHostToDevice));
  u32* dpowkey=keywords(rd(in+"/pow_key.bin").data());
  u32* dpowtgt=keywords(rd(in+"/pow_target.bin").data());
  uint8_t sA[32]={0},sB[32]={0}; memcpy(sA,"A_tensor",8); memcpy(sB,"B_tensor",8);
  uint8_t* dsA=dev<uint8_t>(32),*dsB=dev<uint8_t>(32);
  CK(hipMemcpy(dsA,sA,32,hipMemcpyHostToDevice)); CK(hipMemcpy(dsB,sB,32,hipMemcpyHostToDevice));

  // scratch
  u32* dcvs=dev<u32>((size_t)((n*k+1023)/1024 + 16)*8);
  uint8_t* dBHash=dev<uint8_t>(32), *dCommitA=dev<uint8_t>(32), *dCommitB=dev<uint8_t>(32), *dAHash=dev<uint8_t>(32);
  u32* dCommitBw=dev<u32>(32); // CommitB as key words for B-side noise (32 bytes)
  u32* dCommitAw=dev<u32>(32);

  // ── σ-install ──────────────────────────────────────────────────────────
  hipLaunchKernelGGL(pk::k_tensor_hash,1,1,0,0,(const uint8_t*)dB,(long)n*k,dKeyW,1,dcvs,dBHash); CK(hipDeviceSynchronize());
  // CommitB = BLAKE3(Key‖BHash); reuse commitment kernel with AHash=garbage (CommitA unused here)
  hipLaunchKernelGGL(pk::k_commitment,1,1,0,0,dBHash,dBHash,dKey,dCommitA,dCommitB); CK(hipDeviceSynchronize());
  // pack CommitB into key words (host round-trip)
  uint8_t hCommitB[32]; CK(hipMemcpy(hCommitB,dCommitB,32,hipMemcpyDeviceToHost));
  u32 cbw[8]; for(int i=0;i<8;++i)cbw[i]=(u32)hCommitB[i*4]|((u32)hCommitB[i*4+1]<<8)|((u32)hCommitB[i*4+2]<<16)|((u32)hCommitB[i*4+3]<<24);
  CK(hipMemcpy(dCommitBw,cbw,32,hipMemcpyHostToDevice));

  // B-side noise keyed by CommitB: EBR[n,r] dense (seed_B), EBL_R[k,r] sparse (seed_B)
  int8_t* dEBR=dev<int8_t>((size_t)n*r); __half* dEBRf=dev<__half>((size_t)n*r);
  int8_t* dEBLk=dev<int8_t>((size_t)r*k); int8_t* dEBLr=dev<int8_t>((size_t)k*r);
  { int nh=((n*r)+31)/32; hipLaunchKernelGGL(pk::k_uniform,E(nh),256,0,0,dEBR,dEBRf,n,r,EALscale,dsB,dCommitBw);}
  { int draws=(k*4+31)/32; CK(hipMemset(dEBLk,0,(size_t)r*k)); CK(hipMemset(dEBLr,0,(size_t)k*r));
    hipLaunchKernelGGL(pk::k_perm,E(draws),256,0,0,dEBLk,dEBLr,k,r,dsB,dCommitBw);} CK(hipDeviceSynchronize());
  // E_B[k,n] = EBL_R[k,r] @ EBR^T[r,n]; Bnoised[k,n] = B^T + int8(E_B)
  int8_t* dEBRt=dev<int8_t>((size_t)r*n);
  hipLaunchKernelGGL(pk::k_transpose_i8,E(n*r),256,0,0,dEBR,dEBRt,n,r); CK(hipDeviceSynchronize());
  int32_t* dEB=dev<int32_t>((size_t)k*n*4); { dim3 g(n/16,k/16); hipLaunchKernelGGL(pk::k_wmma_gemm,g,64,0,0,dEBLr,dEBRt,dEB,k,n,r);} CK(hipDeviceSynchronize());
  int8_t* dBkn=dev<int8_t>((size_t)k*n); int8_t* dBnoi=dev<int8_t>((size_t)k*n);
  hipLaunchKernelGGL(pk::k_transpose_i8,E(n*k),256,0,0,dB,dBkn,n,k); CK(hipDeviceSynchronize());
  hipLaunchKernelGGL(pk::k_add_i8,E(k*n),256,0,0,dBkn,dEB,dBnoi,k*n); CK(hipDeviceSynchronize());

  // per-iter buffers
  int8_t* dA=dev<int8_t>((size_t)m*k);
  int8_t* dEAL=dev<int8_t>((size_t)m*r); __half* dEALf=dev<__half>((size_t)m*r);
  int8_t* dEARk=dev<int8_t>((size_t)r*k); int8_t* dEARr=dev<int8_t>((size_t)k*r);
  int32_t* dEA=dev<int32_t>((size_t)m*k*4); int8_t* dApEA=dev<int8_t>((size_t)m*k);
  int ntiles=(m/16)*(n/16);
  u32* dtr=dev<u32>((size_t)ntiles*16*4); uint8_t* dph=dev<uint8_t>((size_t)ntiles*32); int32_t* dfnd=dev<int32_t>((size_t)ntiles*4);

  std::vector<uint8_t> outAHash(count*32), outCommitA(count*32);
  std::vector<int32_t> outFound(count);

  for(int it=0; it<count; ++it){
    u64 seed_lo=nonce0+(u64)it;
    hipLaunchKernelGGL(pk::k_lcg,E((int)(m*k/8)+1),256,0,0,dA,(long)m*k,seed_lo,(u64)0/*sigma_seed*/);
    hipLaunchKernelGGL(pk::k_tensor_hash,1,1,0,0,(const uint8_t*)dA,(long)m*k,dKeyW,1,dcvs,dAHash);
    hipLaunchKernelGGL(pk::k_commitment,1,1,0,0,dAHash,dBHash,dKey,dCommitA,dCommitB);
    CK(hipDeviceSynchronize());
    uint8_t hCA[32]; CK(hipMemcpy(hCA,dCommitA,32,hipMemcpyDeviceToHost));
    u32 caw[8]; for(int i=0;i<8;++i)caw[i]=(u32)hCA[i*4]|((u32)hCA[i*4+1]<<8)|((u32)hCA[i*4+2]<<16)|((u32)hCA[i*4+3]<<24);
    CK(hipMemcpy(dCommitAw,caw,32,hipMemcpyHostToDevice));
    // A-side noise keyed by CommitA
    { int nh=((m*r)+31)/32; hipLaunchKernelGGL(pk::k_uniform,E(nh),256,0,0,dEAL,dEALf,m,r,EALscale,dsA,dCommitAw);}
    { int draws=(k*4+31)/32; CK(hipMemset(dEARk,0,(size_t)r*k)); CK(hipMemset(dEARr,0,(size_t)k*r));
      hipLaunchKernelGGL(pk::k_perm,E(draws),256,0,0,dEARk,dEARr,k,r,dsA,dCommitAw);} CK(hipDeviceSynchronize());
    // E_A[m,k]=EAL[m,r]@EAR_K[r,k]; ApEA=A+int8(E_A)
    { dim3 g(k/16,m/16); hipLaunchKernelGGL(pk::k_wmma_gemm,g,64,0,0,dEAL,dEARk,dEA,m,k,r);} CK(hipDeviceSynchronize());
    hipLaunchKernelGGL(pk::k_add_i8,E(m*k),256,0,0,dA,dEA,dApEA,m*k); CK(hipDeviceSynchronize());
    // transcript + pow
    hipLaunchKernelGGL(pk::k_transcript,E(ntiles),256,0,0,dApEA,dBnoi,m,n,k,r,dpowkey,dpowtgt,dtr,dph,dfnd);
    CK(hipDeviceSynchronize());
    std::vector<int32_t> hf(ntiles); CK(hipMemcpy(hf.data(),dfnd,(size_t)ntiles*4,hipMemcpyDeviceToHost));
    int fc=0; for(int t=0;t<ntiles;++t) fc+=hf[t];
    CK(hipMemcpy(&outAHash[it*32],dAHash,32,hipMemcpyDeviceToHost));
    memcpy(&outCommitA[it*32],hCA,32);
    outFound[it]=fc;
  }
  wr(out+"/AHash.bin",outAHash.data(),outAHash.size());
  wr(out+"/CommitA.bin",outCommitA.data(),outCommitA.size());
  wr(out+"/CommitB.bin",hCommitB,32);
  wr(out+"/found.bin",outFound.data(),outFound.size()*4);
  printf("ran %d nonces from %llu\n",count,(unsigned long long)nonce0);
  return 0;
}
