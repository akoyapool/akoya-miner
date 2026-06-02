#include <cuda.h>
#include "pearl_api_params.h"

template <class ElementOut, int R, int bM, int bN, int bK, int kStages, int cM,
          int cN, bool SkipReduction, bool SkipDenoising, bool EnableDebug,
          bool SkipCStore>
void run_pearl_gemm_(PearlAPIParams& params, cudaStream_t stream);

template <class ElementDenoise_AxEBL, int R, int bM_noising, int bK_noising,
          int kStages>
void run_pearl_noising_A_(PearlAPIParams& params, cudaStream_t stream);
template <class ElementDenoise_EARxBpEB, int R, int bN_noising, int bK_noising,
          int kStages>
void run_pearl_noising_B_(PearlAPIParams& params, cudaStream_t stream);

template <int R, int NumThreads>
void run_noise_generation(Noise_gen_params& params, cudaStream_t stream);

template <int R>
void run_denoise_converter(PearlAPIParams& params, cudaStream_t stream);
