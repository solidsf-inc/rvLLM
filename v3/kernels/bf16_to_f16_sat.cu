// bf16 -> f16 conversion with saturation clamp to f16 range.
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __launch_bounds__(1024) __global__ void bf16_to_f16_sat_kernel(
    __half* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int n
) {
    if (dst == nullptr || src == nullptr || n <= 0) return;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __bfloat162float(src[idx]);
    if (!isnan(v)) v = fminf(fmaxf(v, -65504.0f), 65504.0f);
    dst[idx] = __float2half(v);
}
