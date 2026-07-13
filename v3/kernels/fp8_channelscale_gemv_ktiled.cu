// Native sm_90 (H100) FP8 E4M3 M=1 channelscale GEMV — K-TILED.
// out[n] = (Σ_k dec(act[k]) · dec(W[n,k])) * act_scale * wscale[n].
//
// This variant bounds activation shared memory to one K tile and accumulates over
// tiles. It is intended for large-K shapes where caching the complete activation
// vector would exceed the launcher's configured dynamic-shared-memory budget.
//
// TILE_K is a runtime arg so the host can sweep it without recompiling; dynamic
// smem = TILE_K * sizeof(__half). Numerically bit-identical to fp8_e4m3_gemv
// (same HW-cvt decode, same F32 accumulate, same scale order) — only the smem
// footprint and loop structure differ. The caller must verify the device's dynamic
// shared-memory limit and allocate exactly TILE_K * sizeof(__half) bytes.
//
// Launch: grid=(ceil(N/warps_per_block),1,1) block=(warps_per_block*32,1,1)
//         dynamic smem = TILE_K * 2.  Requires TILE_K % 16 == 0 and K % 16 == 0.
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <math.h>

namespace {
constexpr int kMaxDefaultDynamicSmemHalfs = 24576;  // 48 KiB
}

__device__ __forceinline__ __half2 decode2_e4m3(uint16_t packed) {
    return __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)packed, __NV_E4M3));
}

extern "C" __global__ void fp8_channelscale_gemv_ktiled_kernel(
    float* __restrict__ out,             // [N]
    const uint8_t* __restrict__ weight,  // [N, K] fp8 e4m3, row-major
    const float* __restrict__ wscale,    // [N] per-channel (per-row) f32 descale
    const uint8_t* __restrict__ act,     // [K] fp8 e4m3 activation
    const float* __restrict__ act_scale, // scalar f32 (pointer for graph capture)
    int N, int K, int TILE_K)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
    return;
#endif
    if (!out || !weight || !wscale || !act || !act_scale ||
        N <= 0 || K <= 0 || TILE_K <= 0 ||
        (K & 15) != 0 || (TILE_K & 15) != 0 ||
        TILE_K > kMaxDefaultDynamicSmemHalfs || TILE_K > K ||
        blockDim.y != 1 || blockDim.z != 1 || blockDim.x == 0 ||
        blockDim.x > 1024 || (blockDim.x & 31) != 0 ||
        gridDim.y != 1 || gridDim.z != 1 ||
        (reinterpret_cast<uintptr_t>(weight) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(act) & 15u) != 0 ||
        !isfinite(*act_scale) || *act_scale <= 0.0f) {
        return;
    }
    extern __shared__ __half xs[]; // TILE_K decoded activations, shared by all warps

    const int wpb  = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n    = blockIdx.x * wpb + warp; // this warp's output row (may be >= N)

    float acc = 0.f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const int tile = min(TILE_K, K - k0);

        // Cooperatively decode this K-tile of the activation into smem (all threads).
        for (int i = threadIdx.x; i < tile; i += blockDim.x) {
            xs[i] = __half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)act[k0 + i], __NV_E4M3));
        }
        __syncthreads();

        if (n < N) {
            const uint8_t* wr = weight + (long long)n * K + k0;
            // 16 fp8 weights per uint4; warp covers 32*16 = 512 elements per step.
            for (int k = lane * 16; k + 15 < tile; k += 512) {
                uint4 wv = __ldg(reinterpret_cast<const uint4*>(wr + k));
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
            // scalar tail for a non-16-multiple tile end (dead code when tile%16==0).
            for (int kr = (tile & ~15) + lane; kr < tile; kr += 32) {
                float w = __half2float(__half(
                    __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)wr[kr], __NV_E4M3)));
                acc += w * __half2float(xs[kr]);
            }
        }
        __syncthreads(); // all threads before overwriting xs with the next tile
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0 && n < N) {
        const float ws = wscale[n];
        out[n] = (isfinite(ws) && ws > 0.0f) ? acc * ws * (*act_scale) : nanf("");
    }
}
