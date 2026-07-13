// f32 -> f16 conversion with saturation clamp to f16 range.
#include <cuda_fp16.h>

extern "C" __launch_bounds__(1024) __global__ void f32_to_f16_sat_kernel(
    __half* __restrict__ dst,
    const float* __restrict__ src,
    int n
) {
    if (dst == nullptr || src == nullptr || n <= 0) return;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = src[idx];
    if (!isnan(v)) v = fminf(fmaxf(v, -65504.0f), 65504.0f);
    dst[idx] = __float2half(v);
}
