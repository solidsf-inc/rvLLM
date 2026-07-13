// Additional split-KV combine instantiation for Dao-AILab/flash-attention
// commit 1233b73b6c95340c65c9edfe929611838354fc6e (BSD-3-Clause).
// Upstream hopper/flash_fwd_combine.cu instantiates only 64 and 128.
#include "flash_fwd_combine_launch_template.h"

template void run_mha_fwd_combine_<cutlass::half_t, float, 256>(Flash_fwd_params &params,
                                                                cudaStream_t stream,
                                                                bool enable_pdl);
template void run_mha_fwd_combine_<cutlass::bfloat16_t, float, 256>(Flash_fwd_params &params,
                                                                    cudaStream_t stream,
                                                                    bool enable_pdl);
