// transcript_hip.hip — PoW transcript + keyed-BLAKE3 target check on gfx942.
//
// Per 16x16 hash tile, accumulate the int32 product in R-sized k-chunks; at each
// chunk snapshot the FULL-TILE XOR (canonical inner hash) and fold it into the
// transcript via rotl(.,13)^h at slot (step % 16). Then keyed BLAKE3 of the
// 64-byte transcript vs the uint256 pow_target.
//
// CONSENSUS NOTE: the inner hash is a pure XOR of all 256 tile elements and the
// fold uses rotl, which is linear over XOR. The result is therefore independent
// of which lane/thread holds which element — so the wave-64 / MFMA fragment
// layout is irrelevant here (Strategy A is satisfied by computing the full-tile
// XOR directly). Matches miner_base/inner_hash.py + pow_utils.hpp byte-for-byte.
//
// Build: hipcc --offload-arch=gfx942 -O3 transcript_hip.hip -o transcript_hip
// Use:   ./transcript_hip <indir> <outdir> [m n k r]
//   indir: ApEA.bin[m,k] int8, Bnoised.bin[k,n] int8, pow_key.bin[32], pow_target.bin[32]
//   outdir: transcripts.bin (num_tiles*16 u32), powhash.bin (num_tiles*32 B), found.bin (num_tiles i32)

#include "blake3_device.cuh"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  printf("HIP error %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__); exit(1);} }while(0)

__device__ __forceinline__ u32 rotl32(u32 x,int n){ return (x<<n)|(x>>(32-n)); }

// One thread per 16x16 hash tile.
__global__ void k_transcript(const int8_t* ApEA, const int8_t* Bn,
                             int m, int n, int k, int R,
                             const u32* pow_key, const u32* pow_target,
                             u32* transcripts, uint8_t* powhash, int32_t* found){
  int tilesN = n/16;
  int t = blockIdx.x*blockDim.x + threadIdx.x;
  int ntiles = (m/16)*tilesN;
  if (t >= ntiles) return;
  int ti = t / tilesN, tj = t % tilesN;
  int m0 = ti*16, n0 = tj*16;

  int32_t acc[16][16];
  #pragma unroll
  for(int a=0;a<16;++a) for(int b=0;b<16;++b) acc[a][b]=0;

  u32 tr[16];
  #pragma unroll
  for(int i=0;i<16;++i) tr[i]=0;

  int nsteps = k/R;
  for(int s=0;s<nsteps;++s){
    // accumulate this R-chunk into acc
    for(int a=0;a<16;++a){
      const int8_t* Arow = ApEA + (size_t)(m0+a)*k + s*R;
      for(int b=0;b<16;++b){
        int32_t sum=0;
        for(int kk=0;kk<R;++kk) sum += (int)Arow[kk] * (int)Bn[(size_t)(s*R+kk)*n + n0 + b];
        acc[a][b] += sum;
      }
    }
    // full-tile XOR of the running accumulator (canonical inner hash)
    u32 h=0;
    #pragma unroll
    for(int a=0;a<16;++a) for(int b=0;b<16;++b) h ^= (u32)acc[a][b];
    int idx = s % 16;
    tr[idx] = rotl32(tr[idx],13) ^ h;
  }

  // store transcript
  #pragma unroll
  for(int i=0;i<16;++i) transcripts[(size_t)t*16 + i] = tr[i];

  // keyed BLAKE3 of the 64-byte transcript
  uint8_t tbytes[64];
  #pragma unroll
  for(int i=0;i<16;++i){ tbytes[i*4]=tr[i]&0xff; tbytes[i*4+1]=(tr[i]>>8)&0xff;
                         tbytes[i*4+2]=(tr[i]>>16)&0xff; tbytes[i*4+3]=(tr[i]>>24)&0xff; }
  uint8_t hh[32]; b3::hash_small(tbytes, 64, pow_key, hh);
  #pragma unroll
  for(int i=0;i<32;++i) powhash[(size_t)t*32+i]=hh[i];

  // uint256 compare hash <= target (LE: word 7 = MSW)
  u32 hw[8];
  #pragma unroll
  for(int i=0;i<8;++i) hw[i]=(u32)hh[i*4]|((u32)hh[i*4+1]<<8)|((u32)hh[i*4+2]<<16)|((u32)hh[i*4+3]<<24);
  int fnd=1;
  for(int i=7;i>=0;--i){ if(hw[i]>pow_target[i]){fnd=0;break;} if(hw[i]<pow_target[i]) break; }
  found[t]=fnd;
}

static std::vector<uint8_t> rd(const std::string&p){FILE*f=fopen(p.c_str(),"rb");if(!f){printf("open %s\n",p.c_str());exit(2);}
  fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);std::vector<uint8_t>v(n);if(fread(v.data(),1,n,f)!=(size_t)n)exit(2);fclose(f);return v;}
static void wr(const std::string&p,const void*d,size_t n){FILE*f=fopen(p.c_str(),"wb");fwrite(d,1,n,f);fclose(f);}
static u32* upkey(const std::vector<uint8_t>&kb){u32 h[8];for(int i=0;i<8;++i)h[i]=(u32)kb[i*4]|((u32)kb[i*4+1]<<8)|((u32)kb[i*4+2]<<16)|((u32)kb[i*4+3]<<24);
  u32*d;CK(hipMalloc(&d,32));CK(hipMemcpy(d,h,32,hipMemcpyHostToDevice));return d;}

int main(int argc,char**argv){
  if(argc<3){printf("usage: %s indir outdir [m n k r]\n",argv[0]);return 2;}
  std::string in=argv[1],out=argv[2]; int m=128,n=256,k=128,r=64;
  if(argc>=7){m=atoi(argv[3]);n=atoi(argv[4]);k=atoi(argv[5]);r=atoi(argv[6]);}
  int8_t* dA=(int8_t*)0; { auto v=rd(in+"/ApEA.bin"); CK(hipMalloc(&dA,v.size())); CK(hipMemcpy(dA,v.data(),v.size(),hipMemcpyHostToDevice)); }
  int8_t* dB=(int8_t*)0; { auto v=rd(in+"/Bnoised.bin"); CK(hipMalloc(&dB,v.size())); CK(hipMemcpy(dB,v.data(),v.size(),hipMemcpyHostToDevice)); }
  u32* dkey=upkey(rd(in+"/pow_key.bin"));
  u32* dtgt=upkey(rd(in+"/pow_target.bin"));
  int ntiles=(m/16)*(n/16);
  u32* dtr; uint8_t* dph; int32_t* dfnd;
  CK(hipMalloc(&dtr,(size_t)ntiles*16*4)); CK(hipMalloc(&dph,(size_t)ntiles*32)); CK(hipMalloc(&dfnd,(size_t)ntiles*4));
  hipLaunchKernelGGL(k_transcript,dim3((ntiles+63)/64),dim3(64),0,0,dA,dB,m,n,k,r,dkey,dtgt,dtr,dph,dfnd);
  CK(hipDeviceSynchronize());
  std::vector<u32> htr(ntiles*16); std::vector<uint8_t> hph(ntiles*32); std::vector<int32_t> hf(ntiles);
  CK(hipMemcpy(htr.data(),dtr,(size_t)ntiles*16*4,hipMemcpyDeviceToHost));
  CK(hipMemcpy(hph.data(),dph,(size_t)ntiles*32,hipMemcpyDeviceToHost));
  CK(hipMemcpy(hf.data(),dfnd,(size_t)ntiles*4,hipMemcpyDeviceToHost));
  wr(out+"/transcripts.bin",htr.data(),htr.size()*4);
  wr(out+"/powhash.bin",hph.data(),hph.size());
  wr(out+"/found.bin",hf.data(),hf.size()*4);
  printf("wrote %d tiles\n",ntiles);
  return 0;
}
