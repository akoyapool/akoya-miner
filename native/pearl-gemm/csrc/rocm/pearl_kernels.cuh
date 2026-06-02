// pearl_kernels.cuh — consolidated, validated HIP device kernels for the Pearl
// miner (gfx942/gfx950). Each kernel here was proven byte-identical in its own
// test harness; this header is the single source reused by the fused iter
// and the C-ABI backend.
#pragma once
#include "blake3_device.cuh"
#include <rocwmma/rocwmma.hpp>
#include <hip/hip_fp16.h>

namespace pk {

// ── lcg_int7: per-iter A fill ─────────────────────────────────────
__device__ __forceinline__ u64 splitmix64(u64 z){
  z += 0x9E3779B97F4A7C15ULL; z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
  z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31);
}
__global__ void k_lcg(int8_t* out,long n,u64 seed_lo,u64 seed_hi){
  u64 base=splitmix64(seed_lo ^ splitmix64(seed_hi)); long n8=n/8;
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n8){u64 z=splitmix64(base+(u64)i);
    #pragma unroll
    for(int b=0;b<8;++b){u32 v=(u32)((z>>(b*8))&0xFF); out[i*8+b]=(int8_t)((int)(v%127)-63);}}
  if(i==0&&(n%8)){u64 z=splitmix64(base+(u64)n8); long t=n-n8*8;
    for(int b=0;b<t;++b){u32 v=(u32)((z>>(b*8))&0xFF); out[n8*8+b]=(int8_t)((int)(v%127)-63);}}
}

// ── tensor_hash: multi-chunk Merkle, single-thread (correctness) ──
__global__ void k_tensor_hash(const uint8_t* data,long len,const u32* key,int has_key,
                              u32* cvs,uint8_t* out){
  if(blockIdx.x|threadIdx.x) return;
  const u32* kp=has_key?key:nullptr;
  long nchunks=(len+b3::CHUNK_SIZE-1)/b3::CHUNK_SIZE;
  if(nchunks<=1){ b3::hash_small(data,(int)len,kp,out); return; }
  for(long i=0;i<nchunks;++i){
    long off=i*b3::CHUNK_SIZE, rem=len-off;
    if(rem>=(long)b3::CHUNK_SIZE) b3::chunk_cv(data+off,(u64)i,kp,false,cvs+i*8);
    else { u32 cv[8]; b3::init_cv(cv,kp); int nb=(int)((rem+63)/64); if(!nb)nb=1;
      for(int b=0;b<nb;++b){u32 bl[16];uint8_t bf[64];
        for(int j=0;j<64;++j) bf[j]=(off+b*64+j<len)?data[off+b*64+j]:0;
        for(int j=0;j<16;++j) bl[j]=b3::load_le32(bf+j*4);
        int blen=(int)rem-b*64; if(blen>64)blen=64; if(blen<0)blen=0;
        u32 f=(kp?b3::KEYED_HASH:0); if(b==0)f|=b3::CHUNK_START; if(b==nb-1)f|=b3::CHUNK_END;
        b3::compress(cv,bl,(u64)i,(u32)blen,f);} for(int j=0;j<8;++j)cvs[i*8+j]=cv[j]; } }
  long level=nchunks;
  while(level>1){long half=level>>1;
    for(long i=0;i<half;++i){bool root=(half==1); b3::parent_cv(cvs+(2*i)*8,cvs+(2*i+1)*8,kp,root,cvs+i*8);}
    level=half;}
  b3::cv_to_bytes(cvs,out);
}

// ── commitment_hash: CommitB=BLAKE3(Key‖BHash); CommitA=BLAKE3(CommitB‖AHash) ──
__global__ void k_commitment(const uint8_t* AHash,const uint8_t* BHash,const uint8_t* Key,
                             uint8_t* CommitA,uint8_t* CommitB){
  if(blockIdx.x|threadIdx.x) return;
  uint8_t buf[64];
  for(int i=0;i<32;++i){buf[i]=Key[i];buf[32+i]=BHash[i];}
  b3::hash_small(buf,64,nullptr,CommitB);
  for(int i=0;i<32;++i){buf[i]=CommitB[i];buf[32+i]=AHash[i];}
  b3::hash_small(buf,64,nullptr,CommitA);
}

// ── noise_gen: dense uniform + sparse permutation ──────────────────
__device__ inline void build_msg(uint8_t m[64],int pp,int v,const uint8_t* seed){
  #pragma unroll
  for(int j=0;j<64;++j)m[j]=0;
  m[pp*4]=v&0xff;m[pp*4+1]=(v>>8)&0xff;m[pp*4+2]=(v>>16)&0xff;m[pp*4+3]=(v>>24)&0xff;
  for(int j=0;j<32;++j)m[32+j]=seed[j];
}
__global__ void k_uniform(int8_t* out,__half* outscaled,int rows,int R,int scale,
                          const uint8_t* seed,const u32* key){
  int nb=rows*R,nh=(nb+31)/32,i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nh)return;
  uint8_t m[64]; build_msg(m,0,1+i,seed); uint8_t h[32]; b3::hash_small(m,64,key,h);
  #pragma unroll
  for(int j=0;j<32;++j){int idx=i*32+j; if(idx<nb){int8_t v=(int8_t)(((int)(h[j]&63))-32);
    out[idx]=v; outscaled[idx]=__float2half((float)((int)v*scale));}}
}
__global__ void k_perm(int8_t* Km,int8_t* Rm,int req,int R,const uint8_t* seed,const u32* key){
  int draws=(req*4+31)/32,i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=draws)return;
  uint8_t m[64]; build_msg(m,1,1+i,seed); uint8_t h[32]; b3::hash_small(m,64,key,h);
  #pragma unroll
  for(int kk=0;kk<8;++kk){int L=i*8+kk; if(L>=req)break; u32 u=b3::load_le32(h+kk*4);
    int fi=u&(R-1); int si=fi^(1+(int)__umulhi((u32)(R-1),u));
    if(Km){ Km[(size_t)fi*req+L]=1; Km[(size_t)si*req+L]=-1; }
    if(Rm){ Rm[(size_t)L*R+fi]=1; Rm[(size_t)L*R+si]=-1; }}
}

// ── int8 GEMM: rocWMMA, C[M,N]=A[M,K]@B[K,N] row-major → int32 ──────
__global__ void k_wmma_gemm(const int8_t* A,const int8_t* B,int32_t* C,int M,int N,int K){
  using namespace rocwmma; int tM=blockIdx.y,tN=blockIdx.x;
  fragment<matrix_a,16,16,16,int8_t,row_major> a;
  fragment<matrix_b,16,16,16,int8_t,row_major> b;
  fragment<accumulator,16,16,16,int32_t> c; fill_fragment(c,0);
  for(int kk=0;kk<K;kk+=16){ load_matrix_sync(a,A+(tM*16)*K+kk,K);
    load_matrix_sync(b,B+kk*N+tN*16,N); mma_sync(c,a,b,c); }
  store_matrix_sync(C+(tM*16)*N+tN*16,c,N,mem_row_major);
}
__global__ void k_add_i8(const int8_t* A,const int32_t* E,int8_t* o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=(int8_t)((int)A[i]+(int)(int8_t)E[i]);
}
__global__ void k_transpose_i8(const int8_t* in,int8_t* out,int rows,int cols){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<rows*cols){int r=i/cols,c=i%cols; out[(size_t)c*rows+r]=in[i];}
}

// ── transcript + PoW: per 16x16 tile, XOR inner hash, rotl fold ─────
__device__ __forceinline__ u32 rotl32(u32 x,int n){ return (x<<n)|(x>>(32-n)); }
// Full 64-lane XOR-reduce for the transcript fold. The naive __shfl_xor ladder lowers to 6x
// ds_bpermute_b32 (general address-gather LDS path); since XOR is assoc/commutative we regroup into
// the cheaper fixed-pattern ops: o=1,2 via DPP quad_perm (VALU, off the LDS issue port), o=4,8,16 via
// ds_swizzle_b32 (no address operand), o=32 via one bpermute. Byte-identical result. (gfx942; validated
// vs a scalar reference, SWZRED=2: fused 368->423 TOPS.)
__device__ __forceinline__ u32 xorreduce64(u32 v){
  v^=__builtin_amdgcn_mov_dpp(v,0xB1,0xF,0xF,false);   // lane^1   ([1,0,3,2])
  v^=__builtin_amdgcn_mov_dpp(v,0x4E,0xF,0xF,false);   // lane^2   ([2,3,0,1])
  v^=__builtin_amdgcn_ds_swizzle(v,(4<<10)|0x1F);
  v^=__builtin_amdgcn_ds_swizzle(v,(8<<10)|0x1F);
  v^=__builtin_amdgcn_ds_swizzle(v,(16<<10)|0x1F);
  v^=__shfl_xor(v,32);
  return v;
}
__global__ void k_transcript(const int8_t* ApEA,const int8_t* Bn,int m,int n,int k,int R,
                             const u32* pow_key,const u32* pow_target,
                             u32* transcripts,uint8_t* powhash,int32_t* found){
  int tilesN=n/16,t=blockIdx.x*blockDim.x+threadIdx.x,nt=(m/16)*tilesN; if(t>=nt)return;
  int ti=t/tilesN,tj=t%tilesN,m0=ti*16,n0=tj*16;
  int32_t acc[16][16];
  #pragma unroll
  for(int a=0;a<16;++a) for(int b=0;b<16;++b) acc[a][b]=0;
  u32 tr[16];
  #pragma unroll
  for(int i=0;i<16;++i) tr[i]=0;
  for(int s=0;s<k/R;++s){
    for(int a=0;a<16;++a){const int8_t* Ar=ApEA+(size_t)(m0+a)*k+s*R;
      for(int b=0;b<16;++b){int32_t sum=0;
        for(int kk=0;kk<R;++kk) sum+=(int)Ar[kk]*(int)Bn[(size_t)(s*R+kk)*n+n0+b]; acc[a][b]+=sum;}}
    u32 h=0;
    #pragma unroll
    for(int a=0;a<16;++a) for(int b=0;b<16;++b) h^=(u32)acc[a][b];
    tr[s%16]=rotl32(tr[s%16],13)^h;
  }
  #pragma unroll
  for(int i=0;i<16;++i) transcripts[(size_t)t*16+i]=tr[i];
  uint8_t tb[64];
  #pragma unroll
  for(int i=0;i<16;++i){tb[i*4]=tr[i]&0xff;tb[i*4+1]=(tr[i]>>8)&0xff;tb[i*4+2]=(tr[i]>>16)&0xff;tb[i*4+3]=(tr[i]>>24)&0xff;}
  uint8_t hh[32]; b3::hash_small(tb,64,pow_key,hh);
  #pragma unroll
  for(int i=0;i<32;++i) powhash[(size_t)t*32+i]=hh[i];
  u32 hw[8];
  #pragma unroll
  for(int i=0;i<8;++i) hw[i]=(u32)hh[i*4]|((u32)hh[i*4+1]<<8)|((u32)hh[i*4+2]<<16)|((u32)hh[i*4+3]<<24);
  int fnd=1; for(int i=7;i>=0;--i){if(hw[i]>pow_target[i]){fnd=0;break;} if(hw[i]<pow_target[i])break;}
  found[t]=fnd;
}

// ── bseed_expand: BLAKE3 XOF of 32-byte seed → int7 ───────────────
__global__ void k_bseed(const u32* seed_words,int8_t* out,long n){
  long j=(long)blockIdx.x*blockDim.x+threadIdx.x, nblk=(n+63)/64; if(j>=nblk)return;
  u32 cv[8]; b3::init_cv(cv,nullptr); u32 msg[16];
  #pragma unroll
  for(int i=0;i<8;++i) msg[i]=seed_words[i];
  #pragma unroll
  for(int i=8;i<16;++i) msg[i]=0;
  u32 o[16]; b3::compress_full(cv,msg,(u64)j,32,b3::CHUNK_START|b3::CHUNK_END|b3::ROOT,o);
  long base=j*64;
  #pragma unroll
  for(int w=0;w<16;++w){u32 v=o[w];
    #pragma unroll
    for(int b=0;b<4;++b){long idx=base+w*4+b; if(idx<n){u32 by=(v>>(b*8))&0xFF; out[idx]=(int8_t)((int)(by%127)-63);}}}
}

// ── fast transcript-GEMM + fused distributed PoW (rocWMMA v6) ───────
// C_noised = ApEA[m,k]·Bn[k,n] on MFMA; per 16×16 tile XOR transcript (every R),
// deferred cross-lane reduce, then keyed-BLAKE3 PoW distributed across lanes 0-3;
// on a hit set host_signal[0]=1. ~170 TOPS vs the scalar k_transcript's ~22.
template<int BM,int BN,int BK>
__global__ void k_tgemm_pow(const int8_t* __restrict__ A,const int8_t* __restrict__ Bn,
                            int m,int n,int k,int R,
                            const u32* __restrict__ pow_key,const u32* __restrict__ pow_target,
                            int* __restrict__ host_signal){
  using namespace rocwmma;
  __shared__ int8_t As[2][BM*BK];
  __shared__ int8_t Bs[2][BK*BN];
  const int tid=threadIdx.x, warp=tid>>6, lane=tid&63;
  const int wN=warp%(BN/32), wM=warp/(BN/32);
  const int blockM=blockIdx.y*BM, blockN=blockIdx.x*BN;
  const int AK4=BK/16, BN4=BN/16;
  fragment<accumulator,16,16,32,int32_t> c[2][2];
  #pragma unroll
  for(int i=0;i<2;++i) for(int j=0;j<2;++j) fill_fragment(c[i][j],0);
  u32 tr[2][2][16];
  #pragma unroll
  for(int i=0;i<2;++i) for(int j=0;j<2;++j) for(int e=0;e<16;++e) tr[i][j][e]=0;
  auto load=[&](int buf,int k0){
    int4* As4=(int4*)As[buf]; int4* Bs4=(int4*)Bs[buf];
    for(int u=tid; u<BM*AK4; u+=blockDim.x){ int rr=u/AK4,c4=u%AK4;
      As4[u]=*(const int4*)&A[(size_t)(blockM+rr)*k + k0 + c4*16]; }
    for(int u=tid; u<BK*BN4; u+=blockDim.x){ int rr=u/BN4,c4=u%BN4;
      Bs4[u]=*(const int4*)&Bn[(size_t)(k0+rr)*n + blockN + c4*16]; }
  };
  load(0,0); __syncthreads();
  int snap=0;
  for(int k0=0;k0<k;k0+=BK){
    int buf=(k0/BK)&1;
    if(k0+BK<k) load(buf^1,k0+BK);
    #pragma unroll
    for(int kk=0;kk<BK;kk+=32){
      fragment<matrix_a,16,16,32,int8_t,row_major> a[2];
      fragment<matrix_b,16,16,32,int8_t,row_major> b[2];
      #pragma unroll
      for(int i=0;i<2;++i) load_matrix_sync(a[i], &As[buf][(wM*32+i*16)*BK+kk], BK);
      #pragma unroll
      for(int j=0;j<2;++j) load_matrix_sync(b[j], &Bs[buf][kk*BN+wN*32+j*16], BN);
      #pragma unroll
      for(int i=0;i<2;++i) for(int j=0;j<2;++j) mma_sync(c[i][j],a[i],b[j],c[i][j]);
    }
    if((k0+BK)%R==0){
      #pragma unroll
      for(int i=0;i<2;++i) for(int j=0;j<2;++j){
        u32 pp=0;
        #pragma unroll
        for(int e=0;e<c[i][j].num_elements;++e) pp^=(u32)c[i][j].x[e];
        tr[i][j][snap%16]=rotl32(tr[i][j][snap%16],13)^pp;
      }
      ++snap;
    }
    __syncthreads();
  }
  // per tile: cross-lane reduce (all lanes get t16) → distribute BLAKE3 PoW to lanes 0..3
  #pragma unroll
  for(int i=0;i<2;++i) for(int j=0;j<2;++j){
    u32 t16[16];
    #pragma unroll
    for(int e=0;e<16;++e){ t16[e]=xorreduce64(tr[i][j][e]); }
    if(lane==i*2+j){
      uint8_t tb[64];
      #pragma unroll
      for(int e=0;e<16;++e){ tb[e*4]=t16[e]&0xff; tb[e*4+1]=(t16[e]>>8)&0xff; tb[e*4+2]=(t16[e]>>16)&0xff; tb[e*4+3]=(t16[e]>>24)&0xff; }
      uint8_t hh[32]; b3::hash_small(tb,64,pow_key,hh);
      u32 hw[8];
      #pragma unroll
      for(int e=0;e<8;++e) hw[e]=(u32)hh[e*4]|((u32)hh[e*4+1]<<8)|((u32)hh[e*4+2]<<16)|((u32)hh[e*4+3]<<24);
      int fnd=1; for(int e=7;e>=0;--e){ if(hw[e]>pow_target[e]){fnd=0;break;} if(hw[e]<pow_target[e])break; }
      if(fnd) atomicExch(host_signal,1);
    }
  }
}

// ── hand-MFMA segment-ping-pong transcript-GEMM + fused PoW ─
// Replaces the rocWMMA k_tgemm_pow. Kills the mid-K accumulator drain via fresh
// per-R-segment accumulators (ping-pong) + running-sum/XOR fold off the critical
// path; transcript staged in LDS; LDS XOR swizzle removes ds_read_b64 bank
// conflicts; the prev segment's fold is SW-pipelined across the next segment's MFMA
// steps. ~375 TOPS @256×128 vs rocWMMA's ~170 (byte-identical to k_transcript).
// B is passed in [n,k] layout (= BpEB directly, K-contiguous per column).
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
template<int BM,int BN,int BK,int TM,int TN,int NSEG,int SEGLEN>
__global__ void __launch_bounds__((BM/(TM*16))*(BN/(TN*16))*64) k_tgemm_pow_pp(
    const int8_t* __restrict__ A,const int8_t* __restrict__ Bnk,int m,int n,int k,int R,
    const u32* __restrict__ pow_key,const u32* __restrict__ pow_target,int* __restrict__ host_signal,
    uint8_t* __restrict__ hdr,u32* __restrict__ transcripts){
  // PoW split out to k_pow_check (fully-parallel kernel): this kernel flushes the per-tile transcript
  // to `transcripts` global; the BLAKE3+target check + hdr write happens in k_pow_check. Keeping the
  // BLAKE3 in here was a low-parallelism (8/64-lane) serial tail = ~32% of the GEMM. (pow_key/target/
  // host_signal/hdr kept in the signature for ABI stability but unused here now.)
  (void)pow_key;(void)pow_target;(void)host_signal;(void)hdr;
  constexpr int NWARP=(BM/(TM*16))*(BN/(TN*16)), NT=TM*TN;
  __shared__ int8_t As[2][BM*BK];
  __shared__ int8_t Bs[2][BN*BK];
  __shared__ u32 trS[NWARP][TM][TN][16];
  const int tid=threadIdx.x, warp=tid>>6, lane=tid&63;
  const int WN=BN/(TN*16), wN=warp%WN, wM=warp/WN;
  const int blockM=blockIdx.y*BM, blockN=blockIdx.x*BN;
  const int wm0=wM*TM*16, wn0=wN*TN*16;
  const int li=lane&15, lg=lane>>4;
  const int AK4=BK/16, BN4=BK/16;
  i32x4 seg[2][TM][TN]; i32x4 run[TM][TN];
  #pragma unroll
  for(int i=0;i<TM;++i) for(int j=0;j<TN;++j) run[i][j]=i32x4{0,0,0,0};
  for(int idx=tid; idx<NWARP*TM*TN*16; idx+=blockDim.x) ((u32*)trS)[idx]=0;
  // LDS XOR swizzle: permute each row's four 8-byte K-chunks by s=(row>>2)&3.
  auto load=[&](int buf,int k0){
    for(int u=tid; u<BM*AK4; u+=blockDim.x){ int rr=u/AK4,c4=u%AK4, s=(rr>>2)&3;
      const long* gp=(const long*)&A[(size_t)(blockM+rr)*k + k0 + c4*16];
      *(long*)&As[buf][rr*BK + (((2*c4)  ^s)<<3)]=gp[0];
      *(long*)&As[buf][rr*BK + (((2*c4+1)^s)<<3)]=gp[1]; }
    for(int u=tid; u<BN*BN4; u+=blockDim.x){ int rr=u/BN4,c4=u%BN4, s=(rr>>2)&3;
      const long* gp=(const long*)&Bnk[(size_t)(blockN+rr)*k + k0 + c4*16];
      *(long*)&Bs[buf][rr*BK + (((2*c4)  ^s)<<3)]=gp[0];
      *(long*)&Bs[buf][rr*BK + (((2*c4+1)^s)<<3)]=gp[1]; }
  };
  auto consume1=[&](int t,int prev,int sidx){
    const int i=t/TN, j=t%TN;
    run[i][j]=run[i][j]+seg[prev][i][j];
    u32 v=(u32)run[i][j][0]^(u32)run[i][j][1]^(u32)run[i][j][2]^(u32)run[i][j][3];
    v=xorreduce64(v);
    if(lane==0) trS[warp][i][j][sidx]=v;
  };
  load(0,0); __syncthreads();
  #pragma unroll
  for(int sg=0;sg<NSEG;++sg){
    const int cur=sg&1, prev=(sg-1)&1, psidx=sg-1;
    #pragma unroll
    for(int st=0;st<SEGLEN;++st){
      const int g=sg*SEGLEN+st, buf=g&1;
      if(g+1<NSEG*SEGLEN) load(buf^1,(g+1)*BK);
      long al[TM], bl[TN];
      #pragma unroll
      for(int i=0;i<TM;++i){ int row=wm0+i*16+li, s=(row>>2)&3;
        al[i]=*(const long*)&As[buf][row*BK + ((lg^s)<<3)]; }
      #pragma unroll
      for(int j=0;j<TN;++j){ int row=wn0+j*16+li, s=(row>>2)&3;
        bl[j]=*(const long*)&Bs[buf][row*BK + ((lg^s)<<3)]; }
      #pragma unroll
      for(int i=0;i<TM;++i) for(int j=0;j<TN;++j)
        seg[cur][i][j]= st==0
          ? __builtin_amdgcn_mfma_i32_16x16x32_i8(al[i],bl[j],i32x4{0,0,0,0},0,0,0)
          : __builtin_amdgcn_mfma_i32_16x16x32_i8(al[i],bl[j],seg[cur][i][j],0,0,0);
      if(sg>0){
        #pragma unroll
        for(int t=0;t<NT;++t) if(t%SEGLEN==st) consume1(t,prev,psidx);
      }
      __syncthreads();
    }
  }
  { const int prev=(NSEG-1)&1, sidx=NSEG-1;
    #pragma unroll
    for(int t=0;t<NT;++t) consume1(t,prev,sidx); }
  __syncthreads();
  // Flush the 16-word transcript per tile to global (coalesced, lane<16). PoW runs in k_pow_check.
  #pragma unroll
  for(int i=0;i<TM;++i) for(int j=0;j<TN;++j){
    int gti=(blockM+wm0+i*16)>>4, gtj=(blockN+wn0+j*16)>>4, t=gti*(n/16)+gtj;
    if(lane<16) transcripts[(size_t)t*16+lane]=trS[warp][i][j][lane];
  }
}

// ── PoW check (split out of the GEMM): one thread per 16x16 tile, FULLY parallel across the grid.
// Keyed-BLAKE3 of the tile's 16-word transcript; on the first hit, emit the winning tile into the
// HostSignalHeader (same layout as the old fused tail). This replaces the 8/64-lane serial BLAKE3
// tail that cost ~32% of the GEMM kernel.
__global__ void k_pow_check(const u32* __restrict__ transcripts,int ntiles,int tilesN,
                            const u32* __restrict__ pow_key,const u32* __restrict__ pow_target,
                            int* __restrict__ host_signal,uint8_t* __restrict__ hdr){
  int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=ntiles) return;
  uint8_t tb[64];
  #pragma unroll
  for(int e=0;e<16;++e){ u32 w=transcripts[(size_t)t*16+e];
    tb[e*4]=w&0xff; tb[e*4+1]=(w>>8)&0xff; tb[e*4+2]=(w>>16)&0xff; tb[e*4+3]=(w>>24)&0xff; }
  uint8_t hh[32]; b3::hash_small(tb,64,pow_key,hh);
  u32 hw[8];
  #pragma unroll
  for(int e=0;e<8;++e) hw[e]=(u32)hh[e*4]|((u32)hh[e*4+1]<<8)|((u32)hh[e*4+2]<<16)|((u32)hh[e*4+3]<<24);
  int fnd=1; for(int e=7;e>=0;--e){ if(hw[e]>pow_target[e]){fnd=0;break;} if(hw[e]<pow_target[e])break; }
  if(fnd && atomicCAS(host_signal,0,1)==0){
    int gti=t/tilesN, gtj=t%tilesN;
    *(int*)(hdr+0)=1;                                  // status = kSignalTriggered
    ((unsigned*)(hdr+40))[0]=(unsigned)gti; ((unsigned*)(hdr+40))[1]=(unsigned)gtj; ((unsigned*)(hdr+40))[2]=0;
    *(unsigned short*)(hdr+64)=16;                     // num_registers_per_thread
    #pragma unroll
    for(int e=0;e<16;++e){ hdr[66+e]=(unsigned char)e; hdr[322+e]=(unsigned char)e; }
    ((int*)(hdr+592))[0]=16; ((int*)(hdr+592))[1]=16; ((int*)(hdr+592))[2]=0;  // mma_tile_size
    __threadfence();
  }
}

// ── parallel tensor_hash (Merkle). One thread per 1024-B chunk → leaf
// CV; then log levels of parent reduction. Replaces single-thread k_tensor_hash.
__global__ void k_blake_leaves(const uint8_t* __restrict__ data,long len,long nchunks,
                               const u32* __restrict__ key,u32* __restrict__ cvs){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=nchunks) return;
  long off=i*b3::CHUNK_SIZE, rem=len-off;
  u32 cv[8];
  if(rem>=(long)b3::CHUNK_SIZE){ b3::chunk_cv(data+off,(u64)i,key,/*root*/nchunks==1,cv); }
  else { b3::init_cv(cv,key); int nb=(int)((rem+63)/64); if(nb<1)nb=1;
    for(int b=0;b<nb;++b){ u32 bl[16]; uint8_t bf[64];
      for(int j=0;j<64;++j) bf[j]=(off+b*64+j<len)?data[off+b*64+j]:0;
      for(int j=0;j<16;++j) bl[j]=b3::load_le32(bf+j*4);
      int blen=(int)rem-b*64; if(blen>64)blen=64; if(blen<0)blen=0;
      u32 f=b3::KEYED_HASH; if(b==0)f|=b3::CHUNK_START; if(b==nb-1)f|=b3::CHUNK_END;
      b3::compress(cv,bl,(u64)i,(u32)blen,f); } }
  #pragma unroll
  for(int j=0;j<8;++j) cvs[i*8+j]=cv[j];
}
// One parent-reduction level: out[i] = parent(in[2i], in[2i+1]); top level → ROOT.
__global__ void k_blake_reduce(const u32* __restrict__ in,u32* __restrict__ out,
                               long npairs,const u32* __restrict__ key,int is_root){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=npairs) return;
  u32 o[8]; b3::parent_cv(in+(2*i)*8,in+(2*i+1)*8,key,is_root&&npairs==1,o);
  #pragma unroll
  for(int j=0;j<8;++j) out[i*8+j]=o[j];
}

} // namespace pk
