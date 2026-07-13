// fp8_decode_v2.cu — split-KV + GQA-grouped FP8 paged decode (sm_90a).
//
// ABI-compatible replacement for paged_attention_sm89.cu. The optimized FP8
// decode is checked against the retained scalar reference by
// fp8_decode_v2_parity_bench.cu.
//
// Build (.so):
//   nvcc -shared -o libfa_sm89_kernels_v2.so fp8_decode_v2.cu \
//        -arch=sm_90a -O3 --use_fast_math -Xcompiler -fPIC
//
// New FP8 decode design:
//   Grid (num_kv_heads, NUM_CHUNKS, batch), 256 threads/block.
//   Each block loads its kv-head's chunk of K/V ONCE and serves all
//   group = num_heads/num_kv_heads q-heads of that kv head.
//   A "unit" = HEAD_DIM/16 lanes owns one K/V row per iteration
//   (uint4 = 16 fp8 loads, hardware cvt.rn.f16x2.e4m3x2 dequant which
//   is bit-exact vs fp8e4m3_to_float for every non-NaN encoding —
//   verified exhaustively in parity_bench.cu). Units run independent
//   online softmax over a strided token subset (zero __syncthreads in
//   the token loop), then a log2(NUM_UNITS) smem tree merge produces
//   one (m, l, acc[head_dim]) partial per (q_head, chunk).
//   NUM_CHUNKS == 1: epilogue acc/l written straight to output (no
//   combine pass). NUM_CHUNKS > 1: partials go to `workspace` and
//   paged_decode_fp8_combine_kernel does the standard LSE merge.
//
// Workspace contract (fa_sm89_paged_decode_fp8):
//   batch * num_heads * NUM_CHUNKS * (head_dim + 2) f32, where
//   NUM_CHUNKS = clamp(128 / (num_kv_heads * batch), 1, 32).
//   fa_sm89_decode_workspace_size() reports the exact byte count for the
//   launch shape. The wrapper rejects a null or undersized workspace before
//   launching whenever NUM_CHUNKS > 1.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>

// FP8 E4M3 -> float conversion (manual, no cuda_fp8.h dependency).
// E4M3: 1 sign, 4 exp (bias=7), 3 mantissa. Range [-448, 448].
__device__ __forceinline__ float fp8e4m3_to_float(uint8_t x) {
    uint32_t s = (x >> 7) & 1;
    uint32_t e = (x >> 3) & 0xF;
    uint32_t m = x & 0x7;
    if (e == 0) {
        if (m == 0) return 0.0f;
        float val = (float)m * 1.953125e-3f; // m * 2^-9
        return s ? -val : val;
    }
    if (e == 15 && m == 7) {
        return __int_as_float(0x7FC00000); // NaN
    }
    uint32_t f32 = (s << 31) | ((e + 120u) << 23) | (m << 20);
    return __int_as_float(f32);
}

// Hardware FP8 dequant: two E4M3 bytes -> two f32. E4M3 -> f16 is an
// exact widening (subnormals land in f16 normal range, max 448 <<
// 65504), so this matches fp8e4m3_to_float bit-for-bit on all 254
// non-NaN encodings (NaN sign bit may differ; output is NaN either
// way). parity_bench.cu asserts this exhaustively.
__device__ __forceinline__ float2 fp8x2_to_float2(uint32_t pair) {
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;"
        : "=r"(h2)
        : "h"(static_cast<unsigned short>(pair)));
    return __half22float2(*reinterpret_cast<const __half2*>(&h2));
}

__device__ __forceinline__ void fp8x4_to_float4(uint32_t u, float* f) {
    float2 lo = fp8x2_to_float2(u & 0xFFFFu);
    float2 hi = fp8x2_to_float2(u >> 16);
    f[0] = lo.x; f[1] = lo.y; f[2] = hi.x; f[3] = hi.y;
}

__device__ __forceinline__ void fp8x16_to_float16(uint4 v, float* f) {
    fp8x4_to_float4(v.x, f + 0);
    fp8x4_to_float4(v.y, f + 4);
    fp8x4_to_float4(v.z, f + 8);
    fp8x4_to_float4(v.w, f + 12);
}

// -------------------------------------------------------------------
// Paged decode f16 (UNCHANGED from paged_attention_sm89.cu).
// -------------------------------------------------------------------

template<int HEAD_DIM>
__global__ void paged_decode_f16_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half* __restrict__ output,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    float scale,
    int num_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq,
    int num_blocks_total,
    int window_size_left
) {
    const int bid = blockIdx.x;
    const int batch_idx = bid / num_heads;
    const int head_idx  = bid % num_heads;
    const int tid = threadIdx.x;
    const int kv_head = head_idx * num_kv_heads / num_heads;

    const int ctx_len = context_lens[batch_idx];
    const long long capacity = (long long)block_size * max_blocks_per_seq;
    if (ctx_len < 0 || (window_size_left < 0 && (long long)ctx_len > capacity)) {
        output[(long long)bid * HEAD_DIM + tid] = __float2half(nanf(""));
        return;
    }
    if (ctx_len == 0) {
        output[(long long)bid * HEAD_DIM + tid] = __float2half(0.0f);
        return;
    }

    float q_val = __half2float(q[(long long)bid * HEAD_DIM + tid]);

    constexpr int NUM_WARPS = HEAD_DIM / 32;
    __shared__ float s_warp_sums[NUM_WARPS];
    __shared__ float s_qk;

    float acc = 0.0f;
    float m_val = -1e20f;
    float l_val = 0.0f;

    const int warp_id = tid / 32;
    const int lane = tid % 32;
    int attend_start = 0;
    int attend_len = ctx_len;
    if (window_size_left >= 0 && attend_len > window_size_left + 1) {
        attend_len = window_size_left + 1;
        attend_start = ctx_len - attend_len;
    }
    const int attend_end = attend_start + attend_len;
    const long long start_page = (long long)attend_start / block_size;
    const long long end_page = ((long long)attend_end + block_size - 1) / block_size;

    for (long long p = start_page; p < end_page; p++) {
        const int table_page = window_size_left >= 0 ? (int)(p % max_blocks_per_seq) : (int)p;
        int phys = block_tables[(long long)batch_idx * max_blocks_per_seq + table_page];
        if (phys < 0 || phys >= num_blocks_total) {
            output[(long long)bid * HEAD_DIM + tid] = __float2half(nanf(""));
            return;
        }
        const long long page_start = p * block_size;
        const int t0 = attend_start > page_start ? (int)(attend_start - page_start) : 0;
        const int t1 = attend_end - page_start < block_size ? (int)(attend_end - page_start) : block_size;

        for (int t = t0; t < t1; t++) {
            const long long slot = (long long)phys * block_size + t;
            const long long kv_idx = (slot * num_kv_heads + kv_head) * HEAD_DIM + tid;
            float k_val = __half2float(k_cache[kv_idx]);

            float dot = q_val * k_val;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);
            if (lane == 0) s_warp_sums[warp_id] = dot;
            __syncthreads();

            if (warp_id == 0 && lane == 0) {
                float total = 0.0f;
                #pragma unroll
                for (int w = 0; w < NUM_WARPS; w++) total += s_warp_sums[w];
                s_qk = total * scale;
            }
            __syncthreads();

            float qk_s = s_qk;
            if (!isfinite(qk_s)) {
                output[(long long)bid * HEAD_DIM + tid] = __float2half(nanf(""));
                return;
            }
            float m_new = fmaxf(m_val, qk_s);
            float exp_diff = __expf(m_val - m_new);
            float exp_qk  = __expf(qk_s - m_new);

            float v_val = __half2float(v_cache[kv_idx]);
            acc = acc * exp_diff + exp_qk * v_val;
            l_val = l_val * exp_diff + exp_qk;
            m_val = m_new;
        }
    }

    if (l_val > 0.0f) acc /= l_val;
    output[(long long)bid * HEAD_DIM + tid] = __float2half(acc);
}

// -------------------------------------------------------------------
// Scalar FP8 reference kernel for the parity harness. Not exported.
// -------------------------------------------------------------------

template<int HEAD_DIM>
__global__ void paged_decode_fp8_kernel_reference(
    const uint8_t* __restrict__ q_fp8,
    const uint8_t* __restrict__ k_cache_fp8,
    const uint8_t* __restrict__ v_cache_fp8,
    __half* __restrict__ output,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const float* __restrict__ k_scale_cache,
    const float* __restrict__ v_scale_cache,
    const float* __restrict__ q_scale_cache,
    const float* __restrict__ q_descale_ptr,
    const float* __restrict__ k_descale_ptr,
    const float* __restrict__ v_descale_ptr,
    float scale,
    int num_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq,
    int num_blocks_total,
    int window_size_left
) {
    const int bid = blockIdx.x;
    const int batch_idx = bid / num_heads;
    const int head_idx  = bid % num_heads;
    const int tid = threadIdx.x;
    const int kv_head = head_idx * num_kv_heads / num_heads;

    const int ctx_len = context_lens[batch_idx];
    const long long capacity = (long long)block_size * max_blocks_per_seq;
    if (ctx_len < 0 || (window_size_left < 0 && (long long)ctx_len > capacity)) {
        output[(long long)bid * HEAD_DIM + tid] = __float2half(nanf(""));
        return;
    }
    if (ctx_len == 0) {
        output[(long long)bid * HEAD_DIM + tid] = __float2half(0.0f);
        return;
    }

    const float q_ds = (q_scale_cache != nullptr)
        ? q_scale_cache[batch_idx * num_heads + head_idx]
        : *q_descale_ptr;
    const bool k_perslot = (k_scale_cache != nullptr);
    const bool v_perslot = (v_scale_cache != nullptr);
    const float k_ds_scalar = k_perslot ? 0.0f : *k_descale_ptr;
    const float v_ds_scalar = v_perslot ? 0.0f : *v_descale_ptr;

    float q_val = fp8e4m3_to_float(q_fp8[(long long)bid * HEAD_DIM + tid]) * q_ds;

    constexpr int NUM_WARPS = HEAD_DIM / 32;
    __shared__ float s_warp_sums[NUM_WARPS];
    __shared__ float s_qk;

    float acc = 0.0f;
    float m_val = -1e20f;
    float l_val = 0.0f;

    const int warp_id = tid / 32;
    const int lane = tid % 32;
    int attend_start = 0;
    int attend_len = ctx_len;
    if (window_size_left >= 0 && attend_len > window_size_left + 1) {
        attend_len = window_size_left + 1;
        attend_start = ctx_len - attend_len;
    }
    const int attend_end = attend_start + attend_len;
    const long long start_page = (long long)attend_start / block_size;
    const long long end_page = ((long long)attend_end + block_size - 1) / block_size;

    for (long long p = start_page; p < end_page; p++) {
        const int table_page = window_size_left >= 0 ? (int)(p % max_blocks_per_seq) : (int)p;
        int phys = block_tables[(long long)batch_idx * max_blocks_per_seq + table_page];
        if (phys < 0 || phys >= num_blocks_total) {
            output[(long long)bid * HEAD_DIM + tid] = __float2half(nanf(""));
            return;
        }
        const long long page_start = p * block_size;
        const int t0 = attend_start > page_start ? (int)(attend_start - page_start) : 0;
        const int t1 = attend_end - page_start < block_size ? (int)(attend_end - page_start) : block_size;

        for (int t = t0; t < t1; t++) {
            long long slot = (long long)phys * block_size + t;
            long long kv_idx = (slot * num_kv_heads + kv_head) * HEAD_DIM + tid;
            float k_ds = k_perslot
                ? k_scale_cache[slot * num_kv_heads + kv_head]
                : k_ds_scalar;
            float v_ds = v_perslot
                ? v_scale_cache[slot * num_kv_heads + kv_head]
                : v_ds_scalar;
            float k_val = fp8e4m3_to_float(k_cache_fp8[kv_idx]) * k_ds;

            float dot = q_val * k_val;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);
            if (lane == 0) s_warp_sums[warp_id] = dot;
            __syncthreads();

            if (warp_id == 0 && lane == 0) {
                float total = 0.0f;
                #pragma unroll
                for (int w = 0; w < NUM_WARPS; w++) total += s_warp_sums[w];
                s_qk = total * scale;
            }
            __syncthreads();

            float qk_s = s_qk;
            if (!isfinite(q_ds) || q_ds <= 0.0f || !isfinite(k_ds) ||
                k_ds <= 0.0f || !isfinite(v_ds) || v_ds <= 0.0f ||
                !isfinite(qk_s)) {
                output[(long long)bid * HEAD_DIM + tid] = __float2half(nanf(""));
                return;
            }
            float m_new = fmaxf(m_val, qk_s);
            float exp_diff = __expf(m_val - m_new);
            float exp_qk  = __expf(qk_s - m_new);

            float v_val = fp8e4m3_to_float(v_cache_fp8[kv_idx]) * v_ds;
            acc = acc * exp_diff + exp_qk * v_val;
            l_val = l_val * exp_diff + exp_qk;
            m_val = m_new;
        }
    }

    if (l_val > 0.0f) acc /= l_val;
    output[(long long)bid * HEAD_DIM + tid] = __float2half(acc);
}

// -------------------------------------------------------------------
// NEW FP8 decode: split-KV main kernel + combine kernel.
//
// Math semantics match the reference exactly up to fp32 reduction
// order: per-element q = fp8(q)*q_ds; qk = (sum_d q_d*fp8(k_d)) *
// k_ds * scale (k_ds is a per-(token,kv_head) scalar so it commutes
// with the sum); online softmax with m init -1e20, l init 0, __expf;
// acc += exp_qk*v_ds*fp8(v_d); epilogue acc/l when l>0; zero output
// on an empty context. Sliding-window launches partition only the attended
// logical interval and map its pages through the circular block table.
// -------------------------------------------------------------------

template<int HEAD_DIM, int GROUP_TILE>
__global__ void __launch_bounds__(256)
paged_decode_fp8_splitkv_kernel(
    const uint8_t* __restrict__ q_fp8,
    const uint8_t* __restrict__ k_cache_fp8,
    const uint8_t* __restrict__ v_cache_fp8,
    __half* __restrict__ output,        // written only when gridDim.y == 1
    float* __restrict__ workspace,      // [B][H][C][(HD+2)] when gridDim.y > 1
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const float* __restrict__ k_scale_cache,
    const float* __restrict__ v_scale_cache,
    const float* __restrict__ q_scale_cache,
    const float* __restrict__ q_descale_ptr,
    const float* __restrict__ k_descale_ptr,
    const float* __restrict__ v_descale_ptr,
    float scale,
    int num_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq,
    int num_blocks_total,
    int window_size_left
) {
    constexpr int THREADS    = 256;
    constexpr int UNIT_LANES = HEAD_DIM / 16;        // lanes that cover one K/V row
    constexpr int NUM_UNITS  = THREADS / UNIT_LANES; // independent softmax units

    const int kv_head  = blockIdx.x;
    const int chunk    = blockIdx.y;
    const int C        = gridDim.y;
    const int batch    = blockIdx.z;
    const int tid       = threadIdx.x;
    const int unit      = tid / UNIT_LANES;
    const int unit_lane = tid % UNIT_LANES;
    // Member mask of THIS unit's lanes within its warp. Units in the
    // same warp have different token-loop trip counts, so the shuffle
    // reduction must name only the unit's own lanes (a full-warp mask
    // deadlocks on sm_90 when the sibling unit has already exited).
    const unsigned unit_mask = (UNIT_LANES == 32)
        ? 0xFFFFFFFFu
        : (((1u << UNIT_LANES) - 1u) << (((tid & 31) / UNIT_LANES) * UNIT_LANES));

    const int group   = num_heads / num_kv_heads;
    const int ctx_len = context_lens[batch];
    const long long capacity = (long long)block_size * max_blocks_per_seq;

    extern __shared__ float smem[];
    float* s_acc = smem;                                    // [NUM_UNITS][GROUP_TILE][HEAD_DIM]
    float* s_ml  = smem + NUM_UNITS * GROUP_TILE * HEAD_DIM; // [NUM_UNITS][GROUP_TILE][2]

    if (ctx_len < 0 || (window_size_left < 0 && (long long)ctx_len > capacity)) {
        if (C == 1) {
            for (int j = tid; j < group * HEAD_DIM; j += THREADS) {
                const int qh = kv_head * group + j / HEAD_DIM;
                output[((long long)batch * num_heads + qh) * HEAD_DIM + (j % HEAD_DIM)] =
                    __float2half(nanf(""));
            }
        } else {
            for (int j = tid; j < group * HEAD_DIM; j += THREADS) {
                const int qh = kv_head * group + j / HEAD_DIM;
                float* w = workspace + ((size_t)(batch * num_heads + qh) * C + chunk) *
                                       (HEAD_DIM + 2);
                w[j % HEAD_DIM] = nanf("");
            }
            if (tid < group) {
                const int qh = kv_head * group + tid;
                float* w = workspace + ((size_t)(batch * num_heads + qh) * C + chunk) *
                                       (HEAD_DIM + 2);
                w[HEAD_DIM] = nanf("");
                w[HEAD_DIM + 1] = nanf("");
            }
        }
        return;
    }
    if (ctx_len == 0) {
        // Reference writes zeros. With C > 1 the combine kernel does it;
        // with C == 1 there is no combine, so this block zeroes its rows.
        if (C == 1) {
            for (int j = tid; j < group * HEAD_DIM; j += THREADS) {
                int qh = kv_head * group + j / HEAD_DIM;
                output[(size_t)(batch * num_heads + qh) * HEAD_DIM + (j % HEAD_DIM)] =
                    __float2half(0.0f);
            }
        }
        return;
    }

    int attend_start = 0;
    int attend_len = ctx_len;
    if (window_size_left >= 0 && attend_len > window_size_left + 1) {
        attend_len = window_size_left + 1;
        attend_start = ctx_len - attend_len;
    }
    const int chunk_size = (int)(((long long)attend_len + C - 1) / C);
    const long long t_begin = (long long)attend_start + (long long)chunk * chunk_size;
    const long long attend_end = (long long)attend_start + attend_len;
    const long long t_end = min(t_begin + chunk_size, attend_end);
    if (t_begin >= t_end) return; // empty chunk (only when C > 1); combine skips it

    const bool k_perslot = (k_scale_cache != nullptr);
    const bool v_perslot = (v_scale_cache != nullptr);
    const float k_ds_scalar = k_perslot ? 0.0f : *k_descale_ptr;
    const float v_ds_scalar = v_perslot ? 0.0f : *v_descale_ptr;

    for (int tile = 0; tile < group; tile += GROUP_TILE) {
        const int jn = min(GROUP_TILE, group - tile); // valid q-heads in this tile

        // Dequantized Q for this unit's 16 dims, per tile head.
        float qreg[GROUP_TILE][16];
        #pragma unroll
        for (int j = 0; j < GROUP_TILE; j++) {
            if (j < jn) {
                const int qh = kv_head * group + tile + j;
                const float q_ds = (q_scale_cache != nullptr)
                    ? q_scale_cache[batch * num_heads + qh]
                    : *q_descale_ptr;
                uint4 qv = *reinterpret_cast<const uint4*>(
                    q_fp8 + (size_t)(batch * num_heads + qh) * HEAD_DIM + unit_lane * 16);
                float qf[16];
                fp8x16_to_float16(qv, qf);
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    qreg[j][i] = isfinite(q_ds) && q_ds > 0.0f ? qf[i] * q_ds : nanf("");
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) qreg[j][i] = 0.0f;
            }
        }

        float acc[GROUP_TILE][16];
        float m_j[GROUP_TILE], l_j[GROUP_TILE];
        #pragma unroll
        for (int j = 0; j < GROUP_TILE; j++) {
            m_j[j] = -1e20f;
            l_j[j] = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; i++) acc[j][i] = 0.0f;
        }

        // Token loop: zero barriers. Each unit owns tokens
        // t_begin+unit, +NUM_UNITS, ...
        for (long long t = t_begin + unit; t < t_end; t += NUM_UNITS) {
            const long long page = t / block_size;
            const int table_page = window_size_left >= 0
                ? (int)(page % max_blocks_per_seq) : (int)page;
            const int phys = block_tables[(long long)batch * max_blocks_per_seq + table_page];
            if (phys < 0 || phys >= num_blocks_total) {
                #pragma unroll
                for (int j = 0; j < GROUP_TILE; j++) {
                    m_j[j] = nanf("");
                    l_j[j] = nanf("");
                    #pragma unroll
                    for (int i = 0; i < 16; i++) acc[j][i] = nanf("");
                }
                continue;
            }
            const long long slot = (long long)phys * block_size + (t - page * block_size);
            const long long sk = slot * num_kv_heads + kv_head;

            const float k_ds = k_perslot ? k_scale_cache[sk] : k_ds_scalar;
            const float v_ds = v_perslot ? v_scale_cache[sk] : v_ds_scalar;

            const uint4 kv = *reinterpret_cast<const uint4*>(
                k_cache_fp8 + sk * HEAD_DIM + unit_lane * 16);
            const uint4 vv = *reinterpret_cast<const uint4*>(
                v_cache_fp8 + sk * HEAD_DIM + unit_lane * 16);

            float kf[16];
            fp8x16_to_float16(kv, kf);

            float dot[GROUP_TILE];
            #pragma unroll
            for (int j = 0; j < GROUP_TILE; j++) {
                float d = 0.0f;
                #pragma unroll
                for (int i = 0; i < 16; i++) d = fmaf(qreg[j][i], kf[i], d);
                dot[j] = d;
            }
            #pragma unroll
            for (int off = UNIT_LANES / 2; off > 0; off >>= 1) {
                #pragma unroll
                for (int j = 0; j < GROUP_TILE; j++)
                    dot[j] += __shfl_xor_sync(unit_mask, dot[j], off);
            }

            float vf[16];
            fp8x16_to_float16(vv, vf);

            const float ks = k_ds * scale;
            #pragma unroll
            for (int j = 0; j < GROUP_TILE; j++) {
                const float qk    = dot[j] * ks;
                if (!isfinite(k_ds) || k_ds <= 0.0f || !isfinite(qk) ||
                    !isfinite(v_ds) || v_ds <= 0.0f) {
                    m_j[j] = nanf("");
                    l_j[j] = nanf("");
                    #pragma unroll
                    for (int i = 0; i < 16; i++) acc[j][i] = nanf("");
                    continue;
                }
                const float m_new = fmaxf(m_j[j], qk);
                const float ed    = __expf(m_j[j] - m_new);
                const float eq    = __expf(qk - m_new);
                const float ev    = eq * v_ds;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    acc[j][i] = acc[j][i] * ed + ev * vf[i];
                l_j[j] = l_j[j] * ed + eq;
                m_j[j] = m_new;
            }
        }

        // Block-level merge of NUM_UNITS partials -> one partial per
        // (q_head, chunk). Empty units carry (m=-1e20, l=0, acc=0)
        // which is the neutral element of the LSE merge.
        #pragma unroll
        for (int j = 0; j < GROUP_TILE; j++) {
            #pragma unroll
            for (int i = 0; i < 16; i++)
                s_acc[(size_t)(unit * GROUP_TILE + j) * HEAD_DIM + unit_lane * 16 + i] = acc[j][i];
            if (unit_lane == 0) {
                s_ml[(unit * GROUP_TILE + j) * 2 + 0] = m_j[j];
                s_ml[(unit * GROUP_TILE + j) * 2 + 1] = l_j[j];
            }
        }
        __syncthreads();

        for (int s = NUM_UNITS / 2; s > 0; s >>= 1) {
            if (unit < s) {
                #pragma unroll
                for (int j = 0; j < GROUP_TILE; j++) {
                    const int ra = (unit * GROUP_TILE + j);
                    const int rb = ((unit + s) * GROUP_TILE + j);
                    const float ma = s_ml[ra * 2 + 0], la = s_ml[ra * 2 + 1];
                    const float mb = s_ml[rb * 2 + 0], lb = s_ml[rb * 2 + 1];
                    const float M  = (isfinite(ma) && isfinite(mb) &&
                                      isfinite(la) && isfinite(lb))
                        ? fmaxf(ma, mb) : nanf("");
                    const float fa = __expf(ma - M);
                    const float fb = __expf(mb - M);
                    __syncwarp(unit_mask);
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        const int d = unit_lane * 16 + i;
                        s_acc[(size_t)ra * HEAD_DIM + d] =
                            s_acc[(size_t)ra * HEAD_DIM + d] * fa +
                            s_acc[(size_t)rb * HEAD_DIM + d] * fb;
                    }
                    if (unit_lane == 0) {
                        s_ml[ra * 2 + 0] = M;
                        s_ml[ra * 2 + 1] = la * fa + lb * fb;
                    }
                }
            }
            __syncthreads();
        }

        // Unit 0's rows now hold the merged partial for this chunk.
        if (C == 1) {
            for (int j = 0; j < jn; j++) {
                const int qh = kv_head * group + tile + j;
                const float l = s_ml[j * 2 + 1];
                for (int d = tid; d < HEAD_DIM; d += THREADS) {
                    float a = s_acc[(size_t)j * HEAD_DIM + d];
                    if (l > 0.0f) a /= l;
                    output[(size_t)(batch * num_heads + qh) * HEAD_DIM + d] = __float2half(a);
                }
            }
        } else {
            for (int j = 0; j < jn; j++) {
                const int qh  = kv_head * group + tile + j;
                const size_t row = ((size_t)(batch * num_heads + qh) * C + chunk);
                float* w = workspace + row * (HEAD_DIM + 2);
                for (int d = tid; d < HEAD_DIM; d += THREADS)
                    w[d] = s_acc[(size_t)j * HEAD_DIM + d];
                if (tid == 0) {
                    w[HEAD_DIM + 0] = s_ml[j * 2 + 0];
                    w[HEAD_DIM + 1] = s_ml[j * 2 + 1];
                }
            }
        }
        __syncthreads(); // protect smem reuse across group tiles
    }
}

// Combine: one block per (batch, q_head), HEAD_DIM threads. Standard
// log-sum-exp merge over the valid chunks.
template<int HEAD_DIM>
__global__ void paged_decode_fp8_combine_kernel(
    const float* __restrict__ workspace,
    __half* __restrict__ output,
    const int* __restrict__ context_lens,
    int num_heads,
    int num_chunks,
    int window_size_left
) {
    const int bid   = blockIdx.x;           // batch * num_heads + head
    const int batch = bid / num_heads;
    const int tid   = threadIdx.x;

    const int ctx_len = context_lens[batch];
    if (ctx_len < 0) {
        output[(size_t)bid * HEAD_DIM + tid] = __float2half(nanf(""));
        return;
    }
    if (ctx_len == 0) {
        output[(size_t)bid * HEAD_DIM + tid] = __float2half(0.0f);
        return;
    }

    int attend_len = ctx_len;
    if (window_size_left >= 0 && attend_len > window_size_left + 1)
        attend_len = window_size_left + 1;
    const int chunk_size = (int)(((long long)attend_len + num_chunks - 1) / num_chunks);
    float M = -1e20f, L = 0.0f, out = 0.0f;
    for (int c = 0; c < num_chunks; c++) {
        if ((long long)c * chunk_size >= attend_len) break; // trailing empty chunks
        const float* w = workspace + ((size_t)bid * num_chunks + c) * (HEAD_DIM + 2);
        const float m = w[HEAD_DIM + 0];
        const float l = w[HEAD_DIM + 1];
        const float a = w[tid];
        const float Mn = (isfinite(M) && isfinite(L) && isfinite(m) && isfinite(l))
            ? fmaxf(M, m) : nanf("");
        const float fo = __expf(M - Mn);
        const float fc = __expf(m - Mn);
        out = out * fo + a * fc;
        L   = L * fo + l * fc;
        M   = Mn;
    }
    output[(size_t)bid * HEAD_DIM + tid] = __float2half(L > 0.0f ? out / L : out);
}

// -------------------------------------------------------------------
// Paged prefill FP8 (UNCHANGED from paged_attention_sm89.cu).
// -------------------------------------------------------------------

template<int HEAD_DIM>
__global__ void paged_prefill_fp8_kernel(
    const uint8_t* __restrict__ q_fp8,       // [total_q, num_heads, head_dim]
    const uint8_t* __restrict__ k_cache_fp8, // paged
    const uint8_t* __restrict__ v_cache_fp8, // paged
    __half* __restrict__ output,             // [total_q, num_heads, head_dim]
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const int* __restrict__ cu_seqlens_q,    // [batch+1]
    const float* __restrict__ k_scale_cache, // nullable [num_slots, num_kv_heads]
    const float* __restrict__ v_scale_cache, // nullable [num_slots, num_kv_heads]
    const float* __restrict__ q_scale_cache, // nullable [total_q, num_heads]
    const float* __restrict__ q_descale_ptr,
    const float* __restrict__ k_descale_ptr,
    const float* __restrict__ v_descale_ptr,
    float scale,
    int total_q,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq,
    int num_blocks_total,
    int max_seqlen_q,
    int window_size_left
) {
    const int gid = blockIdx.x;
    const int q_token_idx = gid / num_heads;
    const int head_idx    = gid % num_heads;
    const int tid = threadIdx.x;
    const int kv_head = head_idx * num_kv_heads / num_heads;

    if (q_token_idx >= total_q) return;
    if (cu_seqlens_q[0] != 0 || cu_seqlens_q[batch_size] != total_q) {
        output[((long long)q_token_idx * num_heads + head_idx) * HEAD_DIM + tid] =
            __float2half(nanf(""));
        return;
    }

    // Find which sequence this Q token belongs to (linear scan; batch is small).
    int seq_idx = -1;
    for (int s = 0; s < batch_size; s++) {
        if (q_token_idx < cu_seqlens_q[s + 1]) { seq_idx = s; break; }
    }
    if (seq_idx < 0) {
        output[((long long)q_token_idx * num_heads + head_idx) * HEAD_DIM + tid] =
            __float2half(nanf(""));
        return;
    }
    int ctx_len = context_lens[seq_idx];
    int q_seq_start = cu_seqlens_q[seq_idx];
    int q_seq_end = cu_seqlens_q[seq_idx + 1];
    int q_pos_in_seq = q_token_idx - q_seq_start;
    int q_len = max(0, q_seq_end - q_seq_start);
    const long long capacity = (long long)block_size * max_blocks_per_seq;
    if (q_seq_start < 0 || q_seq_end < q_seq_start || q_len > max_seqlen_q ||
        q_token_idx < q_seq_start || q_token_idx >= q_seq_end || ctx_len < q_len ||
        (window_size_left < 0 && (long long)ctx_len > capacity)) {
        output[((long long)q_token_idx * num_heads + head_idx) * HEAD_DIM + tid] =
            __float2half(nanf(""));
        return;
    }
    int prefix_len = max(0, ctx_len - q_len);

    // Causal: for chunked prefill, the Q batch is the tail of an
    // existing context. Attend to prefix plus Q positions [0, q_pos].
    int causal_len = prefix_len + q_pos_in_seq + 1;
    int attend_len = min(causal_len, ctx_len);

    const float q_ds = (q_scale_cache != nullptr)
        ? q_scale_cache[(long long)q_token_idx * num_heads + head_idx]
        : *q_descale_ptr;
    const bool k_perslot = (k_scale_cache != nullptr);
    const bool v_perslot = (v_scale_cache != nullptr);
    const float k_ds_scalar = k_perslot ? 0.0f : *k_descale_ptr;
    const float v_ds_scalar = v_perslot ? 0.0f : *v_descale_ptr;

    long long q_offset = ((long long)q_token_idx * num_heads + head_idx) * HEAD_DIM + tid;
    float q_val = fp8e4m3_to_float(q_fp8[q_offset]) * q_ds;

    constexpr int NUM_WARPS = HEAD_DIM / 32;
    __shared__ float s_warp_sums[NUM_WARPS];
    __shared__ float s_qk;

    float acc = 0.0f;
    float m_val = -1e20f;
    float l_val = 0.0f;

    const int warp_id = tid / 32;
    const int lane = tid % 32;
    int attend_start = 0;
    if (window_size_left >= 0) {
        int window_len = window_size_left + 1;
        if (attend_len > window_len) {
            attend_start = attend_len - window_len;
            attend_len = window_len;
        }
    }
    const int attend_end = attend_start + attend_len;
    const long long start_page = (long long)attend_start / block_size;
    const long long end_page = ((long long)attend_end + block_size - 1) / block_size;

    for (long long p = start_page; p < end_page; p++) {
        int table_p = (window_size_left >= 0) ? (int)(p % max_blocks_per_seq) : (int)p;
        int phys = block_tables[(long long)seq_idx * max_blocks_per_seq + table_p];
        if (phys < 0 || phys >= num_blocks_total) {
            output[q_offset] = __float2half(nanf(""));
            return;
        }
        const long long page_start = p * block_size;
        int t0 = attend_start > page_start ? (int)(attend_start - page_start) : 0;
        int t1 = attend_end - page_start < block_size ? (int)(attend_end - page_start) : block_size;

        for (int t = t0; t < t1; t++) {
            long long slot = (long long)phys * block_size + t;
            long long kv_idx = (slot * num_kv_heads + kv_head) * HEAD_DIM + tid;
            float k_ds = k_perslot
                ? k_scale_cache[slot * num_kv_heads + kv_head]
                : k_ds_scalar;
            float v_ds = v_perslot
                ? v_scale_cache[slot * num_kv_heads + kv_head]
                : v_ds_scalar;
            float k_val = fp8e4m3_to_float(k_cache_fp8[kv_idx]) * k_ds;

            float dot = q_val * k_val;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);
            if (lane == 0) s_warp_sums[warp_id] = dot;
            __syncthreads();

            if (warp_id == 0 && lane == 0) {
                float total = 0.0f;
                #pragma unroll
                for (int w = 0; w < NUM_WARPS; w++) total += s_warp_sums[w];
                s_qk = total * scale;
            }
            __syncthreads();

            float qk_s = s_qk;
            if (!isfinite(q_ds) || q_ds <= 0.0f || !isfinite(k_ds) ||
                k_ds <= 0.0f || !isfinite(v_ds) || v_ds <= 0.0f ||
                !isfinite(qk_s)) {
                output[q_offset] = __float2half(nanf(""));
                return;
            }
            float m_new = fmaxf(m_val, qk_s);
            float exp_diff = __expf(m_val - m_new);
            float exp_qk  = __expf(qk_s - m_new);

            float v_val = fp8e4m3_to_float(v_cache_fp8[kv_idx]) * v_ds;
            acc = acc * exp_diff + exp_qk * v_val;
            l_val = l_val * exp_diff + exp_qk;
            m_val = m_new;
        }
    }

    if (l_val > 0.0f) acc /= l_val;
    long long out_idx = ((long long)q_token_idx * num_heads + head_idx) * HEAD_DIM + tid;
    output[out_idx] = __float2half(acc);
}


// -------------------------------------------------------------------
// Launch helpers for the new FP8 decode path.
// -------------------------------------------------------------------

static inline int fp8_decode_pick_chunks(int batch_size, int num_kv_heads,
                                         int num_heads, int head_dim) {
    (void)num_heads;
    (void)head_dim;
    const long long denominator = (long long)num_kv_heads * batch_size;
    if (denominator <= 0) return 0;
    int c = (int)(128 / denominator);
    if (c < 1) c = 1;
    if (c > 32) c = 32;
    return c;
}

template<int HEAD_DIM, int GROUP_TILE>
static int launch_fp8_decode_splitkv(
    const uint8_t* q, const uint8_t* k, const uint8_t* v, __half* out,
    float* workspace, const int* bt, const int* cl,
    const float* ks, const float* vs, const float* qs,
    const float* qd, const float* kd, const float* vd,
    float scale, int batch, int num_heads, int num_kv_heads,
    int block_size, int mbps, int num_blocks_total, int window_size_left,
    int num_chunks, cudaStream_t stream
) {
    constexpr int THREADS    = 256;
    constexpr int UNIT_LANES = HEAD_DIM / 16;
    constexpr int NUM_UNITS  = THREADS / UNIT_LANES;
    constexpr size_t SMEM =
        (size_t)NUM_UNITS * GROUP_TILE * HEAD_DIM * sizeof(float) +
        (size_t)NUM_UNITS * GROUP_TILE * 2 * sizeof(float);

    if (SMEM > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(
            paged_decode_fp8_splitkv_kernel<HEAD_DIM, GROUP_TILE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM);
        if (e != cudaSuccess) return -1;
    }

    dim3 grid(num_kv_heads, num_chunks, batch);
    paged_decode_fp8_splitkv_kernel<HEAD_DIM, GROUP_TILE>
        <<<grid, THREADS, SMEM, stream>>>(
            q, k, v, out, workspace, bt, cl, ks, vs, qs, qd, kd, vd,
            scale, num_heads, num_kv_heads, block_size, mbps,
            num_blocks_total, window_size_left);
    if (cudaPeekAtLastError() != cudaSuccess) return -1;

    if (num_chunks > 1) {
        paged_decode_fp8_combine_kernel<HEAD_DIM>
            <<<batch * num_heads, HEAD_DIM, 0, stream>>>(
                workspace, out, cl, num_heads, num_chunks, window_size_left);
        if (cudaPeekAtLastError() != cudaSuccess) return -1;
    }
    return 0;
}

// -------------------------------------------------------------------
// C ABI wrappers. Symbol names: fa_sm89_* (unchanged set).
// -------------------------------------------------------------------

static bool checked_workspace_bytes(
    int batch_size, int num_heads, int num_chunks, int head_dim, long long* bytes
) {
    if (batch_size <= 0 || num_heads <= 0 || num_chunks <= 0 || head_dim <= 0) return false;
    long long value = batch_size;
    if (value > LLONG_MAX / num_heads) return false;
    value *= num_heads;
    if (value > LLONG_MAX / num_chunks) return false;
    value *= num_chunks;
    if (value > LLONG_MAX / (head_dim + 2)) return false;
    value *= head_dim + 2;
    if (value > LLONG_MAX / (long long)sizeof(float)) return false;
    *bytes = value * (long long)sizeof(float);
    return true;
}

static bool valid_attention_shape(
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, int num_blocks_total,
    int window_size_left, float scale, long long query_tokens
) {
    if (batch_size <= 0 || num_heads <= 0 || num_kv_heads <= 0 ||
        num_heads % num_kv_heads != 0 ||
        (head_dim != 128 && head_dim != 256 && head_dim != 512) ||
        block_size <= 0 || max_blocks_per_seq <= 0 || num_blocks_total <= 0 ||
        window_size_left < -1 || window_size_left == INT_MAX ||
        !isfinite(scale) || scale <= 0.0f || query_tokens <= 0 ||
        query_tokens > INT_MAX / num_heads) {
        return false;
    }
    const long long capacity = (long long)block_size * max_blocks_per_seq;
    if (window_size_left >= 0 && (long long)window_size_left + 1 > capacity) return false;
    long long elements = (long long)num_blocks_total * block_size;
    if (elements > LLONG_MAX / num_kv_heads) return false;
    elements *= num_kv_heads;
    return elements <= LLONG_MAX / head_dim;
}

extern "C" {

int rvllm_fa_sm89_abi_version() { return 2; }
int fa_sm89_fp8_output_dtype() { return 1; }
int fa_sm89_fp8_output_element_size() { return sizeof(__half); }

uint64_t fa_sm89_decode_workspace_size(
    int batch_size, int num_heads, int num_kv_heads, int head_dim
) {
    if (batch_size <= 0 || num_heads <= 0 || num_kv_heads <= 0 ||
        num_heads % num_kv_heads != 0 || num_heads / num_kv_heads > 32 ||
        (head_dim != 128 && head_dim != 256 && head_dim != 512)) return UINT64_MAX;
    const int chunks = fp8_decode_pick_chunks(
        batch_size, num_kv_heads, num_heads, head_dim);
    if (chunks <= 1) return 0;
    long long need = 0;
    if (!checked_workspace_bytes(batch_size, num_heads, chunks, head_dim, &need)) return UINT64_MAX;
    return static_cast<uint64_t>(need);
}

int fa_sm89_workspace_size(int batch_size, int num_heads, int max_num_splits) {
    if (max_num_splits <= 0) return -1;
    const int chunks = min(max_num_splits, 32);
    long long need = 0;
    if (!checked_workspace_bytes(batch_size, num_heads, chunks, 512, &need) ||
        need > INT_MAX) return -1;
    return (int)need;
}

int fa_sm89_paged_decode(
    void* q, void* k_cache, void* v_cache, void* output,
    void* block_tables, void* context_lens, void* workspace,
    size_t workspace_bytes,
    float scale,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, int num_blocks_total,
    int window_size_left,
    void* stream_ptr
) {
    (void)workspace;
    (void)workspace_bytes;
    if (!q || !k_cache || !v_cache || !output || !block_tables || !context_lens ||
        !valid_attention_shape(batch_size, num_heads, num_kv_heads, head_dim,
                               block_size, max_blocks_per_seq, num_blocks_total,
                               window_size_left, scale, batch_size)) return -1;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int grid = batch_size * num_heads;

    #define LAUNCH_F16(HD) \
        paged_decode_f16_kernel<HD><<<grid, HD, 0, stream>>>( \
            (const __half*)q, (const __half*)k_cache, (const __half*)v_cache, \
            (__half*)output, (const int*)block_tables, (const int*)context_lens, \
            scale, num_heads, num_kv_heads, block_size, max_blocks_per_seq, \
            num_blocks_total, window_size_left)

    if      (head_dim == 128) { LAUNCH_F16(128); }
    else if (head_dim == 256) { LAUNCH_F16(256); }
    else if (head_dim == 512) { LAUNCH_F16(512); }
    else { return -1; }
    #undef LAUNCH_F16

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int fa_sm89_paged_decode_fp8(
    void* q_fp8, void* k_cache_fp8, void* v_cache_fp8, void* output,
    void* block_tables, void* context_lens, void* workspace,
    size_t workspace_bytes,
    void* k_scale_cache, void* v_scale_cache, void* q_scale_cache,
    float* q_descale, float* k_descale, float* v_descale,
    float scale,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, int num_blocks_total,
    int window_size_left,
    void* stream_ptr
) {
    if (!q_fp8 || !k_cache_fp8 || !v_cache_fp8 || !output || !block_tables ||
        !context_lens || (!q_scale_cache && !q_descale) ||
        (!k_scale_cache && !k_descale) || (!v_scale_cache && !v_descale) ||
        (reinterpret_cast<uintptr_t>(q_fp8) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(k_cache_fp8) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(v_cache_fp8) & 15u) != 0 ||
        !valid_attention_shape(batch_size, num_heads, num_kv_heads, head_dim,
                               block_size, max_blocks_per_seq, num_blocks_total,
                               window_size_left, scale, batch_size)) return -1;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    const int group = num_heads / num_kv_heads;
    if (group > 32) return -1;

    const int num_chunks =
        fp8_decode_pick_chunks(batch_size, num_kv_heads, num_heads, head_dim);
    long long required_workspace = 0;
    if (num_chunks <= 0 ||
        !checked_workspace_bytes(batch_size, num_heads, num_chunks, head_dim,
                                 &required_workspace)) return -1;
    if (num_chunks > 1 &&
        (workspace == nullptr ||
         reinterpret_cast<uintptr_t>(workspace) % alignof(float) != 0 ||
         static_cast<unsigned long long>(required_workspace) > workspace_bytes)) return -1;

    #define LAUNCH_FP8_V2(HD, GT) \
        launch_fp8_decode_splitkv<HD, GT>( \
            (const uint8_t*)q_fp8, (const uint8_t*)k_cache_fp8, \
            (const uint8_t*)v_cache_fp8, (__half*)output, \
            (float*)workspace, (const int*)block_tables, \
            (const int*)context_lens, \
            (const float*)k_scale_cache, (const float*)v_scale_cache, \
            (const float*)q_scale_cache, \
            (const float*)q_descale, (const float*)k_descale, \
            (const float*)v_descale, \
            scale, batch_size, num_heads, num_kv_heads, \
            block_size, max_blocks_per_seq, num_blocks_total, window_size_left, \
            num_chunks, stream)

    int rc;
    const bool wide = (group >= 3); // GROUP_TILE 4 for group >= 3, else 2
    if      (head_dim == 128) { rc = wide ? LAUNCH_FP8_V2(128, 4) : LAUNCH_FP8_V2(128, 2); }
    else if (head_dim == 256) { rc = wide ? LAUNCH_FP8_V2(256, 4) : LAUNCH_FP8_V2(256, 2); }
    else if (head_dim == 512) { rc = wide ? LAUNCH_FP8_V2(512, 4) : LAUNCH_FP8_V2(512, 2); }
    else { return -1; }
    #undef LAUNCH_FP8_V2
    if (rc != 0) return rc;

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int fa_sm89_paged_prefill_fp8(
    void* q_fp8, void* k_cache_fp8, void* v_cache_fp8, void* output,
    void* block_tables, void* context_lens, void* cu_seqlens_q,
    void* workspace,
    size_t workspace_bytes,
    void* k_scale_cache, void* v_scale_cache, void* q_scale_cache,
    float* q_descale, float* k_descale, float* v_descale,
    float scale,
    int total_q, int max_seqlen_q,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, int num_blocks_total,
    int window_size_left,
    void* stream_ptr
) {
    (void)workspace;
    (void)workspace_bytes;
    if (!q_fp8 || !k_cache_fp8 || !v_cache_fp8 || !output || !block_tables ||
        !context_lens || !cu_seqlens_q || (!q_scale_cache && !q_descale) ||
        (!k_scale_cache && !k_descale) || (!v_scale_cache && !v_descale) ||
        total_q <= 0 || max_seqlen_q <= 0 || max_seqlen_q > total_q ||
        !valid_attention_shape(batch_size, num_heads, num_kv_heads, head_dim,
                               block_size, max_blocks_per_seq, num_blocks_total,
                               window_size_left, scale, total_q)) return -1;
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int grid = total_q * num_heads;

    #define LAUNCH_PREFILL(HD) \
        paged_prefill_fp8_kernel<HD><<<grid, HD, 0, stream>>>( \
            (const uint8_t*)q_fp8, (const uint8_t*)k_cache_fp8, \
            (const uint8_t*)v_cache_fp8, (__half*)output, \
            (const int*)block_tables, (const int*)context_lens, \
            (const int*)cu_seqlens_q, \
            (const float*)k_scale_cache, (const float*)v_scale_cache, \
            (const float*)q_scale_cache, \
            (const float*)q_descale, (const float*)k_descale, \
            (const float*)v_descale, \
            scale, total_q, batch_size, num_heads, num_kv_heads, \
            block_size, max_blocks_per_seq, num_blocks_total, max_seqlen_q, \
            window_size_left)

    if      (head_dim == 128) { LAUNCH_PREFILL(128); }
    else if (head_dim == 256) { LAUNCH_PREFILL(256); }
    else if (head_dim == 512) { LAUNCH_PREFILL(512); }
    else { return -1; }
    #undef LAUNCH_PREFILL

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

} // extern "C"
