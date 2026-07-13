// Fused GELU(tanh)*up + per-token FP8 E4M3 quantization for SM90.
// Input layout: [num_tokens, 2 * intermediate_size] as [gate | up] per row.
// Output: [num_tokens, intermediate_size] in FP8 with per-row scales.
// Compile: nvcc -ptx -arch=sm_90 -O3 --use_fast_math
//
// Vectorized: 128-bit loads (8 halves), register-cached intermediates
// (eliminates second global memory pass), 64-bit FP8 stores.
//
// GELU(tanh)(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Used by Gemma 4 (gelu_pytorch_tanh) instead of SiLU.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <math_constants.h>

#define FP8_E4M3_MAX 448.0f
#define WARPS_MAX 32
#define VEC_SIZE 8
#define MAX_VECS_PER_THREAD 4  // supports intermediate_size up to 32768

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    val = warp_reduce_max(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
    if (warp_id == 0) val = warp_reduce_max(val);
    return val;
}

__device__ __forceinline__ float gelu_tanh(float x) {
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// GELU(gate) * up + per-token FP8 quantization.
// grid=(num_tokens), block=(min(intermediate_size, 1024))
// shared mem: WARPS_MAX * sizeof(float)
// Requires: intermediate_size % 8 == 0 (true for all transformer models)
extern "C" __global__ void __launch_bounds__(1024)
fused_gelu_mul_fp8_quant_kernel(
    __nv_fp8_storage_t* __restrict__ output_fp8,
    float*              __restrict__ output_scales,
    const __half*       __restrict__ gate_up,
    int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    if (output_fp8 == nullptr || output_scales == nullptr || gate_up == nullptr ||
        intermediate_size <= 0 || intermediate_size % VEC_SIZE != 0 ||
        blockDim.x < 32 || blockDim.x > 1024 || blockDim.x % 32 != 0 ||
        blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1 ||
        reinterpret_cast<uintptr_t>(gate_up) % alignof(uint4) != 0 ||
        reinterpret_cast<uintptr_t>(output_fp8) % alignof(uint2) != 0) return;
    const int n_vecs = intermediate_size / VEC_SIZE;
    if (n_vecs > blockDim.x * MAX_VECS_PER_THREAD) return;

    // 128-bit vectorized pointers for gate and up halves of this row
    const uint4* gate_vec = reinterpret_cast<const uint4*>(
        gate_up + (long long)row * 2 * intermediate_size);
    const uint4* up_vec = reinterpret_cast<const uint4*>(
        gate_up + (long long)row * 2 * intermediate_size + intermediate_size);

    __shared__ float smem[WARPS_MAX];
    __shared__ int invalid;
    if (tid == 0) invalid = 0;
    __syncthreads();

    // Register cache: store gelu(g)*u to avoid reloading in pass 2
    float cached[MAX_VECS_PER_THREAD * VEC_SIZE];

    // Pass 1: vectorized 128-bit loads, compute GELU(gate)*up, find absmax
    float local_max = 0.0f;
    int vec_idx = 0;
    for (int i = tid; i < n_vecs; i += stride, vec_idx++) {
        uint4 gv = gate_vec[i];
        uint4 uv = up_vec[i];
        const __half* g = reinterpret_cast<const __half*>(&gv);
        const __half* u = reinterpret_cast<const __half*>(&uv);

        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            float gf = __half2float(g[j]);
            float uf = __half2float(u[j]);
            float gelu_g = gelu_tanh(gf);
            float v = gelu_g * uf;
            if (!isfinite(v)) atomicExch(&invalid, 1);
            cached[vec_idx * VEC_SIZE + j] = v;
            local_max = fmaxf(local_max, fabsf(v));
        }
    }

    float absmax = block_reduce_max(local_max, smem);
    if (threadIdx.x == 0) smem[0] = absmax;
    __syncthreads();
    absmax = smem[0];

    float scale = invalid ? CUDART_NAN_F : fmaxf(absmax / FP8_E4M3_MAX, 1e-12f);
    if (tid == 0) output_scales[row] = scale;
    float inv_scale = 1.0f / scale;

    // Pass 2: quantize from register cache, 64-bit vectorized FP8 store
    uint2* out_vec = reinterpret_cast<uint2*>(output_fp8 + (long long)row * intermediate_size);
    vec_idx = 0;
    for (int i = tid; i < n_vecs; i += stride, vec_idx++) {
        __nv_fp8_storage_t fp8[VEC_SIZE];
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            fp8[j] = __nv_cvt_float_to_fp8(
                cached[vec_idx * VEC_SIZE + j] * inv_scale,
                __NV_SATFINITE, __NV_E4M3);
        }
        out_vec[i] = *reinterpret_cast<const uint2*>(fp8);
    }
}
