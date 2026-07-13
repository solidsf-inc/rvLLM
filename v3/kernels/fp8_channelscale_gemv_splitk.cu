// Native sm_90 (H100) FP8 E4M3 M=1 channelscale GEMV — SPLIT-K.
// out[n] = (Σ_k dec(act[k]) · dec(W[n,k])) * act_scale * wscale[n].
//
// SPLIT blocks cooperate per output row, each reducing an aligned K slice and
// atomically accumulating its partial result.
//
// Scale folding: wscale[n]*act_scale is the SAME constant c for every K-slice of
// row n, so each block applies c to its partial before the atomicAdd -- the sum
// stays correct: Σ_s (partial_s · c) = c · Σ_s partial_s. out[] MUST be verified
// zeroed before launch. Floating-point atomic accumulation is not deterministic;
// callers must use a documented numerical tolerance for parity checks.
//
// Grid: (ceil(N/warps_per_block), SPLIT, 1). Block: (warps_per_block*32,1,1).
// No dynamic smem. Requires K % 16 == 0.
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <math.h>

__device__ __forceinline__ __half2 dec2_e4m3(uint16_t packed) {
    return __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)packed, __NV_E4M3));
}

extern "C" __global__ void fp8_channelscale_gemv_splitk_kernel(
    float* __restrict__ out,             // [N], pre-zeroed
    const uint8_t* __restrict__ weight,  // [N, K] fp8 e4m3, row-major
    const float* __restrict__ wscale,    // [N] per-channel f32 descale
    const uint8_t* __restrict__ act,     // [K] fp8 e4m3 activation
    const float* __restrict__ act_scale, // scalar f32
    int N, int K, int SPLIT)
{
    if (!out || !weight || !wscale || !act || !act_scale ||
        N <= 0 || K <= 0 || SPLIT <= 0 || (K & 15) != 0 ||
        blockDim.y != 1 || blockDim.z != 1 || blockDim.x == 0 ||
        blockDim.x > 1024 || (blockDim.x & 31) != 0 ||
        gridDim.y != static_cast<unsigned int>(SPLIT) || gridDim.z != 1 ||
        (reinterpret_cast<uintptr_t>(weight) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(act) & 15u) != 0 ||
        !isfinite(*act_scale) || *act_scale <= 0.0f) {
        return;
    }
    const int wpb  = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n    = blockIdx.x * wpb + warp;
    if (n >= N) return;

    // K-slice for this block's split index (aligned to 16 so uint4 loads stay in-slice).
    long long kchunk = (((long long)K + SPLIT - 1) / SPLIT + 15) & ~15LL;
    long long k_begin = (long long)blockIdx.y * kchunk;
    if (k_begin >= K) return;
    long long k_end = min((long long)K, k_begin + kchunk);

    const uint8_t* wr = weight + (long long)n * K;
    float acc = 0.f;

    // 16 fp8 per uint4; warp covers 32*16 = 512 elements/step over its K-slice.
    long long k = k_begin + (long long)lane * 16;
    for (; k + 15 < k_end; k += 512) {
        uint4 wv = __ldg(reinterpret_cast<const uint4*>(wr + k));
        uint4 av = __ldg(reinterpret_cast<const uint4*>(act + k));
        uint32_t wp[4] = {wv.x, wv.y, wv.z, wv.w};
        uint32_t ap[4] = {av.x, av.y, av.z, av.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 wl = __half22float2(dec2_e4m3((uint16_t)(wp[j] & 0xFFFFu)));
            float2 wh = __half22float2(dec2_e4m3((uint16_t)(wp[j] >> 16)));
            float2 xl = __half22float2(dec2_e4m3((uint16_t)(ap[j] & 0xFFFFu)));
            float2 xh = __half22float2(dec2_e4m3((uint16_t)(ap[j] >> 16)));
            acc += wl.x * xl.x + wl.y * xl.y + wh.x * xh.x + wh.y * xh.y;
        }
    }
    // scalar tail (dead when slice length % 16 == 0).
    for (long long kr = ((k_end) & ~15LL) + lane; kr < k_end; kr += 32) {
        if (kr < k_begin) continue;
        float w = __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)wr[kr], __NV_E4M3)));
        float x = __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)act[kr], __NV_E4M3)));
        acc += w * x;
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        const float ws = wscale[n];
        if (isfinite(ws) && ws > 0.0f) {
            atomicAdd(&out[n], acc * ws * (*act_scale));
        } else {
            out[n] = nanf("");
        }
    }
}
