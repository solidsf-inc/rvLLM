// dst[i] += src[i] for f16 vectors of length n.
#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(1024)
vector_add_f16_kernel(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __hadd(dst[idx], src[idx]);
    }
}
