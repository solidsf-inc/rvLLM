// Fused: f32->bf16 + rmsnorm + add-to-residual(f16) + optional layer_scalar
//
// Grid: (num_tokens), Block: (min(hidden, 1024))
// Shared memory: static warp partials only.

#include <cuda_fp16.h>

extern "C" __global__ void fused_norm_add_residual_kernel(
    const float* __restrict__ gemm_out,
    const half*  __restrict__ gamma,
    half*        __restrict__ residual,
    const half*  __restrict__ layer_scalar,
    int hidden,
    float eps
) {
    int token = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (gemm_out == nullptr || gamma == nullptr || residual == nullptr || hidden <= 0 ||
        !isfinite(eps) || eps <= 0.0f || blockDim.x < 32 || blockDim.x > 1024 ||
        blockDim.x % 32 != 0 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1) return;
    const float* row = gemm_out + (size_t)token * hidden;
    half* res = residual + (size_t)token * hidden;

    float local_ss = 0.0f;
    for (int i = tid; i < hidden; i += stride) {
        float v = row[i];
        local_ss += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_ss += __shfl_xor_sync(0xffffffff, local_ss, offset);

    __shared__ float warp_ss[32];
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) warp_ss[warp_id] = local_ss;
    __syncthreads();

    if (tid == 0) {
        int nw = (stride + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int w = 0; w < nw; w++) total += warp_ss[w];
        warp_ss[0] = total;
    }
    __syncthreads();
    float rms_inv = rsqrtf(warp_ss[0] / (float)hidden + eps);

    float ls = layer_scalar ? __half2float(*layer_scalar) : 1.0f;
    for (int i = tid; i < hidden; i += stride) {
        float normed = row[i] * rms_inv * __half2float(gamma[i]);
        float r = __half2float(res[i]) + normed;
        res[i] = __float2half(r * ls);
    }
}
