// Multiply each F16 residual row in place by one BF16 layer scalar.
//
// Grid:  (num_tokens, 1, 1)
// Block: (min(hidden, 1024), 1, 1)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

extern "C"
__global__ void residual_scale_bf16s_f16_kernel(
    __half*              __restrict__ residual,  // [num_tokens, hidden], in-place f16
    const __nv_bfloat16* __restrict__ scalar,    // [1] bf16 per-layer scale
    int hidden
) {
    const int row = blockIdx.x;
    if (!residual || !scalar || hidden <= 0 || blockDim.x == 0 ||
        blockDim.x > 1024 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1) {
        return;
    }
    const float s = __bfloat162float(scalar[0]);

    __half* row_ptr = residual + (long long)row * hidden;

    if (!isfinite(s)) {
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            row_ptr[i] = __float2half(nanf(""));
        }
        return;
    }

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = __half2float(row_ptr[i]);
        row_ptr[i] = __float2half(v * s);
    }
}
