// Gemma 4 per-layer-embedding (PLE) gate injection.
//
// Two kernels:
//
//  1. gemma4_ple_gate_kernel — the per-layer hot path. Runs once per
//     decoder layer, after its attention and MLP residual
//     blocks and BEFORE the per-layer scalar. Fuses:
//        residual = h
//        gate = gelu_tanh(per_layer_input_gate @ h)   # [hidden]->[h_ple]
//        gate = gate * per_layer_input                # [h_ple]
//        contrib = per_layer_projection @ gate        # [h_ple]->[hidden]
//        contrib = post_per_layer_input_norm(contrib) # RMSNorm [hidden]
//        residual += contrib
//     One CUDA block processes each token with a shared-memory-staged input.
//
//     Dense weights are bf16 (gate_w [h_ple, hidden] row-major,
//     proj_w [hidden, h_ple] row-major) — the loader dequantizes the
//     quantized per_layer_input_gate / per_layer_projection tensors to
//     f16 before launch. The residual stream is f16.
//
//  2. gemma4_ple_projection_combine_kernel — run ONCE at model input,
//     across all layers. Implements mlx-lm `_project_per_layer_inputs`:
//        proj = per_layer_model_projection(h_scaled)  # external GEMM,
//                                                      # raw output in
//        proj[l] *= hidden^-0.5
//        proj[l] = per_layer_projection_norm(proj[l]) # RMSNorm [h_ple]
//        out[l]  = (proj[l] + per_layer_inputs[l]) * per_layer_input_scale
//     One block per (token, layer). per_layer_inputs is the gathered +
//     scale-folded PLE embed table slice for this token.
//
// The fixed block-reduction order is deterministic for a fixed launch shape.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

namespace {

__device__ __forceinline__ float gelu_tanh(float x) {
    const float k = 0.7978845608f; // sqrt(2/pi)
    float x3 = x * x * x;
    float inner = k * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Deterministic block sum of a per-thread partial into thread 0, then
// broadcast through smem. Uses a fixed warp-tree + fixed serial combine
// over warps (same order every launch) to keep greedy argmax identity.
__device__ __forceinline__ float block_reduce_sum(float v, float* warp_scratch) {
    for (int off = warpSize / 2; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    int warp_id = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    if (lane == 0) warp_scratch[warp_id] = v;
    __syncthreads();
    float total = 0.0f;
    int nwarps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x == 0) {
        for (int w = 0; w < nwarps; w++) total += warp_scratch[w];
        warp_scratch[0] = total;
    }
    __syncthreads();
    return warp_scratch[0];
}

} // namespace

// Per-layer PLE gate injection.
//
// Grid: (num_tokens), Block: (one or more complete warps), smem: layout below.
// The launcher must provide (hidden + h_ple + 32) * sizeof(float) bytes.
//
// smem layout (floats):
//   [0 .. hidden)          : h staged (f16->f32)
//   [hidden .. hidden+h_ple): gated vector (post gate*gelu*per_layer_input)
//   [.. + 32]              : warp reduction scratch
extern "C" __global__ void gemma4_ple_gate_kernel(
    __half*              __restrict__ residual,        // [num_tokens, hidden] in/out
    const __half*        __restrict__ gate_w,          // [h_ple, hidden] (f16, dequantized)
    const __half*        __restrict__ proj_w,          // [hidden, h_ple] (f16, dequantized)
    const __half*        __restrict__ per_layer_input, // [num_tokens, h_ple]
    const __nv_bfloat16* __restrict__ post_norm_gamma, // [hidden] (bf16)
    int hidden,
    int h_ple,
    int pli_stride,   // per-token stride of [T,L,h_ple] = num_layers*h_ple
    float eps
) {
    if (!residual || !gate_w || !proj_w || !per_layer_input ||
        !post_norm_gamma || hidden <= 0 || h_ple <= 0 ||
        pli_stride < h_ple || !isfinite(eps) || eps <= 0.0f ||
        blockDim.x == 0 || blockDim.x > 1024 || (blockDim.x & 31) != 0 ||
        blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1) {
        return;
    }
    extern __shared__ float smem[];
    float* sh_h    = smem;                 // [hidden]
    float* sh_gate = smem + hidden;        // [h_ple]
    float* warp_sc = smem + hidden + h_ple;// [32]

    int token = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    __half* res_row = residual + (size_t)token * hidden;
    const __half* pli = per_layer_input + (size_t)token * (size_t)pli_stride;

    // Stage h (== current residual stream) into smem as f32.
    for (int i = tid; i < hidden; i += stride)
        sh_h[i] = __half2float(res_row[i]);
    __syncthreads();

    // gate[o] = gelu_tanh( sum_i gate_w[o,i] * h[i] ) * per_layer_input[o]
    // One output per thread loop-strided over h_ple.
    for (int o = tid; o < h_ple; o += stride) {
        const __half* wrow = gate_w + (size_t)o * hidden;
        float acc = 0.0f;
        for (int i = 0; i < hidden; i++) {
            const float w = __half2float(wrow[i]);
            acc += w * sh_h[i];
        }
        const float input = __half2float(pli[o]);
        sh_gate[o] = isfinite(acc) && isfinite(input)
            ? gelu_tanh(acc) * input
            : nanf("");
    }
    __syncthreads();

    // contrib[o] = sum_j proj_w[o,j] * gate[j]   (o over hidden)
    // Accumulate RMS sum-of-squares deterministically across the block.
    float local_ss = 0.0f;
    // First pass: compute contrib into a register-per-output is not
    // possible (hidden > blockDim), so recompute in the write pass after
    // we have rms_inv. Here we only need sum(contrib^2) for the norm.
    for (int o = tid; o < hidden; o += stride) {
        const __half* wrow = proj_w + (size_t)o * h_ple;
        float acc = 0.0f;
        for (int j = 0; j < h_ple; j++)
            acc += __half2float(wrow[j]) * sh_gate[j];
        local_ss += acc * acc;
    }
    float ss = block_reduce_sum(local_ss, warp_sc);
    float rms_inv = rsqrtf(ss / (float)hidden + eps);

    // Second pass: recompute contrib, apply norm gamma, add to residual.
    for (int o = tid; o < hidden; o += stride) {
        const __half* wrow = proj_w + (size_t)o * h_ple;
        float acc = 0.0f;
        for (int j = 0; j < h_ple; j++)
            acc += __half2float(wrow[j]) * sh_gate[j];
        const float gamma = __bfloat162float(post_norm_gamma[o]);
        float normed = isfinite(gamma) ? acc * rms_inv * gamma : nanf("");
        res_row[o] = __float2half(__half2float(res_row[o]) + normed);
    }
}

// PLE model-projection + combine. Run once at model input.
//
// Grid: (num_tokens * num_layers), Block: (>= h_ple rounded, multiple of
// warp), smem: (h_ple + 32) * sizeof(float) bytes.
//
// proj_in : [num_tokens, num_layers, h_ple]  raw per_layer_model_projection
//           GEMM output (BEFORE hidden^-0.5 scale).
// per_layer_inputs : [num_tokens, num_layers, h_ple]  gathered + folded.
// out     : [num_tokens, num_layers, h_ple]  combined per-layer inputs.
extern "C" __global__ void gemma4_ple_projection_combine_kernel(
    const __half*        __restrict__ proj_in,
    const __half*        __restrict__ per_layer_inputs,
    const __half*        __restrict__ proj_norm_gamma, // [h_ple]
    __half*              __restrict__ out,
    int num_layers,
    int h_ple,
    int hidden,
    float eps
) {
    if (!proj_in || !per_layer_inputs || !proj_norm_gamma || !out ||
        num_layers <= 0 || h_ple <= 0 || hidden <= 0 ||
        !isfinite(eps) || eps <= 0.0f ||
        blockDim.x == 0 || blockDim.x > 1024 || (blockDim.x & 31) != 0 ||
        blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.x == 0 || gridDim.x % static_cast<unsigned int>(num_layers) != 0 ||
        gridDim.y != 1 || gridDim.z != 1) {
        return;
    }
    extern __shared__ float smem[];
    float* sh_scaled = smem;         // [h_ple]
    float* warp_sc   = smem + h_ple; // [32]

    int row = blockIdx.x; // (token * num_layers + layer)
    int tid = threadIdx.x;
    int stride = blockDim.x;
    const __half* pin = proj_in + (size_t)row * h_ple;
    const __half* pli = per_layer_inputs + (size_t)row * h_ple;
    __half* orow = out + (size_t)row * h_ple;

    float proj_scale = rsqrtf((float)hidden); // hidden^-0.5
    const float input_scale = 0.70710678118f; // 2^-0.5

    float local_ss = 0.0f;
    for (int i = tid; i < h_ple; i += stride) {
        float v = __half2float(pin[i]) * proj_scale;
        sh_scaled[i] = v;
        local_ss += v * v;
    }
    float ss = block_reduce_sum(local_ss, warp_sc);
    float rms_inv = rsqrtf(ss / (float)h_ple + eps);

    for (int i = tid; i < h_ple; i += stride) {
        const float gamma = __half2float(proj_norm_gamma[i]);
        const float input = __half2float(pli[i]);
        float normed = isfinite(gamma) ? sh_scaled[i] * rms_inv * gamma : nanf("");
        float combined = isfinite(input) ? (normed + input) * input_scale : nanf("");
        orow[i] = __float2half(combined);
    }
}
