// Native sm_90 (H100) FP8 E4M3 M=1 GEMV — hardware-cvt decode + smem activation.
// out[N] = (act[K] · W[N,K]^T) * wscale[N] * act_scale.
//
// Weights use CUDA's native E4M3-to-half conversion. The activation is decoded
// once per block into shared memory and reused by all warps.
//
// Launch: grid=(ceil(N/warps_per_block),1,1) block=(warps_per_block*32,1,1)
//         dynamic smem = K * sizeof(__half). The caller must verify the device's
//         dynamic-shared-memory limit before launch.
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <math.h>

namespace {
constexpr int kMaxDefaultDynamicSmemHalfs = 24576;  // 48 KiB
}

// 2 packed FP8 e4m3 bytes -> half2, one hardware cvt instruction on sm_90.
__device__ __forceinline__ __half2 decode2_e4m3(uint16_t packed) {
    return __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)packed, __NV_E4M3));
}

extern "C" __global__ void fp8_e4m3_gemv_kernel(
    float* __restrict__ out,             // [N]
    const uint8_t* __restrict__ weight,  // [N, K] fp8 e4m3, row-major
    const float* __restrict__ wscale,    // [N] per-row f32 descale
    const uint8_t* __restrict__ act,     // [K] fp8 e4m3 activation
    const float* __restrict__ act_scale, // scalar f32 (pointer for graph capture)
    int N, int K)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
    return;
#endif
    if (!out || !weight || !wscale || !act || !act_scale ||
        N <= 0 || K <= 0 || (K & 15) != 0 ||
        K > kMaxDefaultDynamicSmemHalfs ||
        blockDim.y != 1 || blockDim.z != 1 || blockDim.x == 0 ||
        blockDim.x > 1024 || (blockDim.x & 31) != 0 ||
        gridDim.y != 1 || gridDim.z != 1 ||
        (reinterpret_cast<uintptr_t>(weight) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(act) & 15u) != 0 ||
        !isfinite(*act_scale) || *act_scale <= 0.0f) {
        return;
    }
    extern __shared__ __half xs[]; // K decoded activations, shared by all warps
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        xs[i] = __half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)act[i], __NV_E4M3));
    }
    __syncthreads();

    const int warps_per_block = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * warps_per_block + warp;
    if (n >= N) return;

    const uint8_t* w_row = weight + (long long)n * K;
    float acc = 0.f;

    // 16 fp8 weights per uint4 load; warp covers 32*16 = 512 elements per iter.
    for (int k = lane * 16; k + 15 < K; k += 512) {
        uint4 wv = __ldg(reinterpret_cast<const uint4*>(w_row + k));
        uint32_t wp[4] = {wv.x, wv.y, wv.z, wv.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 lo = __half22float2(decode2_e4m3((uint16_t)(wp[j] & 0xFFFFu)));
            float2 hi = __half22float2(decode2_e4m3((uint16_t)(wp[j] >> 16)));
            int kk = k + j * 4;
            acc += lo.x * __half2float(xs[kk + 0]);
            acc += lo.y * __half2float(xs[kk + 1]);
            acc += hi.x * __half2float(xs[kk + 2]);
            acc += hi.y * __half2float(xs[kk + 3]);
        }
    }
    for (int kr = (K & ~15) + lane; kr < K; kr += 32) {
        float w = __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)w_row[kr], __NV_E4M3)));
        acc += w * __half2float(xs[kr]);
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        const float ws = wscale[n];
        out[n] = (isfinite(ws) && ws > 0.0f) ? acc * ws * (*act_scale) : nanf("");
    }
}
