#include "noise_generation_host.h"
#include "pearl_api_params.h"

template void run_noise_generation<128, 32>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<128, 64>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<128, 128>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<64, 32>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<64, 64>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<64, 128>(Noise_gen_params&, cudaStream_t);
