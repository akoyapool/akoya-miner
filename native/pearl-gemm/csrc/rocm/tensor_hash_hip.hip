// tensor_hash_hip.hip — multi-chunk Merkle tensor_hash on gfx942.
//
// Standard keyed BLAKE3 over a byte buffer: 1024-byte leaves (counter = chunk
// index), perfect binary-tree PARENT reduction, ROOT at the top. For inputs
// whose length is a multiple of 1024 with a power-of-2 chunk count — i.e. every
// real Pearl A/B matrix — this equals what csrc/tensor_hash/* computes.
//
// Validated against the reference `blake3` Python library (see
// verify_tensor_hash.py), the authoritative standard-BLAKE3 oracle.
//
// Build: hipcc --offload-arch=gfx942 -O3 tensor_hash_hip.hip -o tensor_hash_hip
// Use:   ./tensor_hash_hip <datafile> [key_hex_64|none]   # prints root hex

#include "blake3_device.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  printf("HIP error %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__); exit(1);} }while(0)

// Single-thread reference-correctness kernel (perf comes later).
// cvs: device scratch for up to nchunks*8 u32.
__global__ void tensor_hash_kernel(const uint8_t* data, long len,
                                   const u32* key, int has_key,
                                   u32* cvs, uint8_t* out){
  if (blockIdx.x | threadIdx.x) return;
  const u32* k = has_key ? key : nullptr;
  long nchunks = (len + b3::CHUNK_SIZE - 1) / b3::CHUNK_SIZE;
  if (nchunks <= 1) { b3::hash_small(data, (int)len, k, out); return; }

  // Leaves: per-chunk CV (not root).
  for (long i=0;i<nchunks;++i){
    long off = i * b3::CHUNK_SIZE;
    long rem = len - off;
    if (rem >= (long)b3::CHUNK_SIZE) {
      b3::chunk_cv(data + off, (u64)i, k, /*is_root*/false, cvs + i*8);
    } else {
      // Partial final chunk (not exercised by real multiple-of-1024 shapes).
      u32 cv[8]; b3::init_cv(cv, k);
      int nb=(int)((rem+63)/64); if(nb==0)nb=1;
      for(int b=0;b<nb;++b){
        u32 block[16]; uint8_t buf[64];
        for(int j=0;j<64;++j) buf[j]=(off+b*64+j<len)?data[off+b*64+j]:0;
        for(int j=0;j<16;++j) block[j]=b3::load_le32(buf+j*4);
        int blen=(int)rem-b*64; if(blen>64)blen=64; if(blen<0)blen=0;
        u32 f=(k?b3::KEYED_HASH:0); if(b==0)f|=b3::CHUNK_START; if(b==nb-1)f|=b3::CHUNK_END;
        b3::compress(cv, block, (u64)i, (u32)blen, f);
      }
      for(int j=0;j<8;++j) cvs[i*8+j]=cv[j];
    }
  }

  // Perfect binary-tree reduction (power-of-2 nchunks → matches BLAKE3 tree).
  long level = nchunks;
  while (level > 1){
    long half = level >> 1;
    for (long i=0;i<half;++i){
      bool is_root = (half == 1);
      b3::parent_cv(cvs + (2*i)*8, cvs + (2*i+1)*8, k, is_root, cvs + i*8);
    }
    level = half;
  }
  b3::cv_to_bytes(cvs, out);
}

int main(int argc, char** argv){
  if (argc < 2){ printf("usage: %s <datafile> [key_hex_64|none]\n", argv[0]); return 2; }
  FILE* f=fopen(argv[1],"rb"); if(!f){ printf("cannot open %s\n",argv[1]); return 2; }
  fseek(f,0,SEEK_END); long len=ftell(f); fseek(f,0,SEEK_SET);
  std::vector<uint8_t> host(len); if(fread(host.data(),1,len,f)!=(size_t)len){return 2;} fclose(f);

  int has_key=0; u32 key[8]={0};
  if (argc>2 && strcmp(argv[2],"none")!=0){
    has_key=1; const char* h=argv[2];
    uint8_t kb[32]; for(int i=0;i<32;++i){ unsigned v; sscanf(h+i*2,"%2x",&v); kb[i]=(uint8_t)v; }
    for(int i=0;i<8;++i) key[i]=(u32)kb[i*4]|((u32)kb[i*4+1]<<8)|((u32)kb[i*4+2]<<16)|((u32)kb[i*4+3]<<24);
  }

  long nchunks=(len + 1023)/1024; if(nchunks<1) nchunks=1;
  uint8_t *ddata,*dout; u32 *dkey,*dcvs;
  CK(hipMalloc(&ddata, len?len:1)); CK(hipMalloc(&dout,32));
  CK(hipMalloc(&dkey,32)); CK(hipMalloc(&dcvs, (size_t)nchunks*8*sizeof(u32)));
  CK(hipMemcpy(ddata,host.data(),len,hipMemcpyHostToDevice));
  CK(hipMemcpy(dkey,key,32,hipMemcpyHostToDevice));

  hipLaunchKernelGGL(tensor_hash_kernel,dim3(1),dim3(1),0,0,ddata,len,dkey,has_key,dcvs,dout);
  CK(hipDeviceSynchronize());

  uint8_t out[32]; CK(hipMemcpy(out,dout,32,hipMemcpyDeviceToHost));
  for(int i=0;i<32;++i) printf("%02x",out[i]); printf("\n");
  return 0;
}
