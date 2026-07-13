// Paged attention kernels for SM89 (Ada Lovelace).
//
// Independent implementation of paged scaled-dot-product attention using the
// public CUDA runtime API. No FlashAttention source is copied into this file.
// The exported C ABI is consumed by the rvLLM runtime.
//
// Build:
//   nvcc -shared -o libfa_sm89_kernels.so paged_attention_sm89.cu \
//        -arch=sm_89 -O3 --use_fast_math -Xcompiler -fPIC
//
// Kernels: paged decode (f16 and fp8) + paged prefill (fp8).
// Single-split design: one thread block per (batch, q_head). Iterates
// sequentially over KV pages. Online softmax avoids two-pass.

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

// -------------------------------------------------------------------
// Paged decode: one Q token per sequence (batch decode).
//
// Grid:  (batch_size * num_heads, 1, 1)
// Block: (HEAD_DIM, 1, 1)   e.g. 256 threads for head_dim=256
//
// Each thread owns dimension `tid` of Q and accumulates dimension
// `tid` of the output via online softmax over all KV tokens.
// -------------------------------------------------------------------

template<int HEAD_DIM>
__global__ void paged_decode_f16_kernel(
    const __half* __restrict__ q,          // [batch, num_heads, head_dim]
    const __half* __restrict__ k_cache,    // [num_blocks_total, block_size, num_kv_heads, head_dim]
    const __half* __restrict__ v_cache,    // [num_blocks_total, block_size, num_kv_heads, head_dim]
    __half* __restrict__ output,           // [batch, num_heads, head_dim]
    const int* __restrict__ block_tables,  // [batch, max_blocks_per_seq]
    const int* __restrict__ context_lens,  // [batch]
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
    const long long cache_capacity = (long long)block_size * max_blocks_per_seq;
    if (ctx_len < 0 || (window_size_left < 0 && (long long)ctx_len > cache_capacity)) {
        output[(long long)bid * HEAD_DIM + tid] = __float2half(nanf(""));
        return;
    }
    if (ctx_len <= 0) {
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
        int phys = block_tables[(long long)batch_idx * max_blocks_per_seq + table_p];
        if (phys < 0 || phys >= num_blocks_total) {
            output[(long long)bid * HEAD_DIM + tid] = __float2half(nanf(""));
            return;
        }
        long long page_start = p * block_size;
        int t0 = attend_start > page_start ? (int)(attend_start - page_start) : 0;
        int t1 = attend_end - page_start < block_size ? (int)(attend_end - page_start) : block_size;

        for (int t = t0; t < t1; t++) {
            long long slot = (long long)phys * block_size + t;
            long long kv_idx = (slot * num_kv_heads + kv_head) * HEAD_DIM + tid;
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

// FP8 variant: Q/K/V are FP8 E4M3 with per-tensor descales. Output f16.
template<int HEAD_DIM>
__global__ void paged_decode_fp8_kernel(
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
    const long long cache_capacity = (long long)block_size * max_blocks_per_seq;
    if (ctx_len < 0 || (window_size_left < 0 && (long long)ctx_len > cache_capacity)) {
        output[(long long)bid * HEAD_DIM + tid] = __float2half(nanf(""));
        return;
    }
    if (ctx_len <= 0) {
        output[(long long)bid * HEAD_DIM + tid] = __float2half(0.0f);
        return;
    }

    const float q_ds = (q_scale_cache != nullptr)
        ? q_scale_cache[(long long)batch_idx * num_heads + head_idx]
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
        int phys = block_tables[(long long)batch_idx * max_blocks_per_seq + table_p];
        if (phys < 0 || phys >= num_blocks_total) {
            output[(long long)bid * HEAD_DIM + tid] = __float2half(nanf(""));
            return;
        }
        long long page_start = p * block_size;
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
            if (!isfinite(qk_s)) {
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
// Paged prefill FP8: multiple Q tokens per sequence with causal mask.
//
// Grid:  (total_q * num_heads, 1, 1)
// Block: (HEAD_DIM, 1, 1)
//
// Each block handles one (q_token_in_batch, head) pair. The q token's
// sequence index and position within the sequence are derived from
// cu_seqlens_q. Causal masking: q at position p attends to KV [0..p].
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
    const long long cache_capacity = (long long)block_size * max_blocks_per_seq;
    if (q_seq_start < 0 || q_seq_end < q_seq_start || q_len > max_seqlen_q ||
        q_token_idx < q_seq_start || q_token_idx >= q_seq_end || ctx_len < q_len ||
        (window_size_left < 0 && (long long)ctx_len > cache_capacity)) {
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
        long long page_start = p * block_size;
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
            if (!isfinite(qk_s)) {
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
// C ABI wrappers. Symbol names: fa_sm89_*
// -------------------------------------------------------------------

extern "C" {

int rvllm_fa_sm89_abi_version() { return 2; }
int fa_sm89_fp8_output_dtype() { return 1; }
int fa_sm89_fp8_output_element_size() { return sizeof(__half); }

uint64_t fa_sm89_decode_workspace_size(int, int, int, int) { return 0; }

int fa_sm89_workspace_size(int batch_size, int num_heads, int max_num_splits) {
    if (batch_size <= 0 || num_heads <= 0 || max_num_splits <= 0) return -1;
    return 0; // single-split, no workspace needed
}

static bool fa_sm89_valid_shape(
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, int num_blocks_total,
    int window_size_left, float scale, long long query_tokens
) {
    if (batch_size <= 0 || num_heads <= 0 || num_kv_heads <= 0 ||
        num_heads % num_kv_heads != 0 ||
        (head_dim != 128 && head_dim != 256 && head_dim != 512) ||
        block_size <= 0 || max_blocks_per_seq <= 0 || num_blocks_total <= 0 ||
        window_size_left < -1 || window_size_left == INT_MAX ||
        !isfinite(scale) || scale <= 0.0f || query_tokens <= 0) {
        return false;
    }
    const long long capacity = (long long)block_size * max_blocks_per_seq;
    if (window_size_left >= 0 && (long long)window_size_left + 1 > capacity) return false;
    if (query_tokens > INT_MAX / num_heads) return false;
    long long elements = (long long)num_blocks_total * block_size;
    if (elements > LLONG_MAX / num_kv_heads) return false;
    elements *= num_kv_heads;
    if (elements > LLONG_MAX / head_dim) return false;
    return true;
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
        !fa_sm89_valid_shape(batch_size, num_heads, num_kv_heads, head_dim,
                            block_size, max_blocks_per_seq, num_blocks_total,
                            window_size_left, scale, batch_size)) {
        return -1;
    }
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
    (void)workspace;
    (void)workspace_bytes;
    if (!q_fp8 || !k_cache_fp8 || !v_cache_fp8 || !output || !block_tables ||
        !context_lens || (!q_scale_cache && !q_descale) ||
        (!k_scale_cache && !k_descale) || (!v_scale_cache && !v_descale) ||
        !fa_sm89_valid_shape(batch_size, num_heads, num_kv_heads, head_dim,
                            block_size, max_blocks_per_seq, num_blocks_total,
                            window_size_left, scale, batch_size)) {
        return -1;
    }
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int grid = batch_size * num_heads;

    #define LAUNCH_FP8(HD) \
        paged_decode_fp8_kernel<HD><<<grid, HD, 0, stream>>>( \
            (const uint8_t*)q_fp8, (const uint8_t*)k_cache_fp8, \
            (const uint8_t*)v_cache_fp8, (__half*)output, \
            (const int*)block_tables, (const int*)context_lens, \
            (const float*)k_scale_cache, (const float*)v_scale_cache, \
            (const float*)q_scale_cache, \
            (const float*)q_descale, (const float*)k_descale, \
            (const float*)v_descale, \
            scale, num_heads, num_kv_heads, block_size, max_blocks_per_seq, \
            num_blocks_total, window_size_left)

    if      (head_dim == 128) { LAUNCH_FP8(128); }
    else if (head_dim == 256) { LAUNCH_FP8(256); }
    else if (head_dim == 512) { LAUNCH_FP8(512); }
    else { return -1; }
    #undef LAUNCH_FP8

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
        !fa_sm89_valid_shape(batch_size, num_heads, num_kv_heads, head_dim,
                            block_size, max_blocks_per_seq, num_blocks_total,
                            window_size_left, scale, total_q)) {
        return -1;
    }
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
