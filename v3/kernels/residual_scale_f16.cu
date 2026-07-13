// Per-layer residual scaling: residual *= scalar
//
// Gemma 4 applies a learnable per-layer scalar [1] (stored as f16)
// to the residual stream after each sub-block's post-norm.
// This kernel multiplies every element in the residual by that scalar.
//
// Grid:  (num_tokens, 1, 1)
// Block: (min(hidden, 1024), 1, 1)

#include <cuda_fp16.h>

extern "C"
__global__ void residual_scale_f16_kernel(
    __half* __restrict__ residual,      // [num_tokens, hidden], in-place
    const __half* __restrict__ scalar,   // [1], per-layer scale
    int hidden
) {
    const int row = blockIdx.x;
    const float s = __half2float(scalar[0]);

    __half* row_ptr = residual + row * hidden;

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = __half2float(row_ptr[i]);
        row_ptr[i] = __float2half(v * s);
    }
}
