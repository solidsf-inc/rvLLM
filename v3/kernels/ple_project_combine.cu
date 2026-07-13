#include <cuda_fp16.h>
#include <math.h>

#define WARPS_MAX 32

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

extern "C" __global__ void __launch_bounds__(256)
ple_project_combine_f16_kernel(
    __half* __restrict__ out,
    const __half* __restrict__ projection,
    const __half* __restrict__ embeds,
    const __half* __restrict__ norm_gamma,
    int num_layers,
    int ple_dim,
    float projection_scale,
    float combine_scale,
    float eps
) {
    const int token = blockIdx.x;
    const int layer = blockIdx.y;
    const int tid = threadIdx.x;
    if (out == nullptr || projection == nullptr || norm_gamma == nullptr ||
        num_layers <= 0 || ple_dim <= 0 || layer >= num_layers ||
        blockDim.x < 32 || blockDim.x > 256 || blockDim.x % 32 != 0 ||
        blockDim.y != 1 || blockDim.z != 1 || gridDim.z != 1 ||
        !isfinite(projection_scale) || !isfinite(combine_scale) ||
        !isfinite(eps) || eps <= 0.0f) return;
    const long long base = ((long long)token * num_layers + layer) * ple_dim;

    __shared__ float smem[WARPS_MAX];

    float ss = 0.0f;
    for (int i = tid; i < ple_dim; i += blockDim.x) {
        float v = __half2float(projection[base + i]) * projection_scale;
        ss += v * v;
    }
    ss = warp_sum(ss);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) smem[warp] = ss;
    __syncthreads();

    float total = 0.0f;
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        total = (lane < nwarps) ? smem[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) smem[0] = rsqrtf(total / (float)ple_dim + eps);
    }
    __syncthreads();
    const float inv_rms = smem[0];

    for (int i = tid; i < ple_dim; i += blockDim.x) {
        float p = __half2float(projection[base + i]) * projection_scale;
        p = p * inv_rms * __half2float(norm_gamma[i]);
        float e = embeds ? __half2float(embeds[base + i]) : 0.0f;
        out[base + i] = __float2half((p + e) * combine_scale);
    }
}
