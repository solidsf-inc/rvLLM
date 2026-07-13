// Half-precision element-wise bias addition and tensor add kernels.
// Reads/writes f16, computes in f32 for precision.
//
// Launch config (add_bias_f16):
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(dim, 1024), 1, 1)
//   Shared memory: none
//
// Launch config (add_f16, add_inplace_f16):
//   Grid:  (ceil(n / 256), 1, 1)
//   Block: (256, 1, 1)
//   Shared memory: none

#include <cuda_fp16.h>

extern "C"
__global__ void add_bias_f16_kernel(
    __half* __restrict__ tensor,       // [num_tokens, dim] -- modified in-place
    const __half* __restrict__ bias,   // [dim]
    int dim
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int offset = token_idx * dim;

    for (int i = tid; i < dim; i += stride) {
        float val = __half2float(tensor[offset + i]) + __half2float(bias[i]);
        tensor[offset + i] = __float2half(val);
    }
}

extern "C"
__global__ void add_f16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }
}

extern "C"
__global__ void add_inplace_f16_kernel(
    __half* __restrict__ a,
    const __half* __restrict__ b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }
}
