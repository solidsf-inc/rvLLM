// f32 -> bf16 conversion (lossless for values within bf16 range).
#include <cuda_bf16.h>

extern "C" __launch_bounds__(1024) __global__ void f32_to_bf16_kernel(
    __nv_bfloat16* __restrict__ dst,
    const float* __restrict__ src,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __float2bfloat16(src[idx]);
}
