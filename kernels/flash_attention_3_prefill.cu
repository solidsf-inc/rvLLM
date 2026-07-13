// FlashAttention-3 prefill kernel with f16 shared memory.
//
// KV tiles are stored as f16 in shared memory.
//
// Launch: grid(num_seqs, num_heads, 1), block(256)

#include <float.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#define PF_BC 64
#define PF_THREADS 256
#define PF_WARPS 8

__device__ __forceinline__ float pf_warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

extern "C"
__global__ void __launch_bounds__(PF_THREADS, 2)
flash_attention_3_prefill_f16io_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ q,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const int* __restrict__ seq_start_pos,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    int num_tokens,
    int causal,
    int num_blocks_total
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane_id  = tid % 32;

    if (output == nullptr || q == nullptr || key_cache == nullptr || value_cache == nullptr ||
        block_tables == nullptr || context_lens == nullptr || seq_start_pos == nullptr ||
        !isfinite(scale) || scale <= 0.0f || num_heads <= 0 || num_kv_heads <= 0 ||
        num_heads % num_kv_heads != 0 || head_dim <= 0 || head_dim > 1024 ||
        head_dim % 2 != 0 || block_size <= 0 || max_context_len <= 0 ||
        max_blocks_per_seq <= 0 || num_tokens <= 0 || (causal != 0 && causal != 1) ||
        num_blocks_total <= 0 || head_idx >= num_heads || blockDim.x != PF_THREADS ||
        blockDim.y != 1 || blockDim.z != 1 || gridDim.z != 1 ||
        (long long)max_blocks_per_seq * block_size < max_context_len) return;

    const int context_len = context_lens[seq_idx];
    if (context_len <= 0 || context_len > max_context_len) return;

    const int kv_head_idx = (num_kv_heads == num_heads)
        ? head_idx
        : (head_idx / (num_heads / num_kv_heads));

    const int q_start = seq_start_pos[seq_idx];
    const int q_end   = seq_start_pos[seq_idx + 1];
    const int q_len   = q_end - q_start;
    if (q_start < 0 || q_end < q_start || q_end > num_tokens || q_len == 0 ||
        (causal && q_len > context_len)) return;

    const int num_tiles = (int)(((long long)context_len + PF_BC - 1) / PF_BC);
    const int dims_per_thread = (head_dim + PF_THREADS - 1) / PF_THREADS;

    // Shared memory: f16 KV tile + f32 scores + f32 warp scratch
    extern __shared__ char smem_raw[];
    __half* s_kv   = (__half*)smem_raw;                              // [BC * head_dim]
    float* s_score = (float*)(s_kv + PF_BC * head_dim);             // [BC]
    float* s_warp  = s_score + PF_BC;                                // [WARPS]

    for (int qi = 0; qi < q_len; qi++) {
        const int q_token_idx = q_start + qi;

        const int causal_limit = causal
            ? (context_len - q_len + qi)
            : (context_len - 1);

        // Load Q into registers (f16 -> f32, pre-scaled)
        float q_reg[4];
        const long long q_base = ((long long)q_token_idx * num_heads + head_idx) * head_dim;
        #pragma unroll
        for (int r = 0; r < dims_per_thread && r < 4; r++) {
            int d = tid + r * PF_THREADS;
            q_reg[r] = (d < head_dim) ? (__half2float(q[q_base + d]) * scale) : 0.0f;
        }

        float row_max = -FLT_MAX;
        float row_sum = 0.0f;
        float acc[4];
        #pragma unroll
        for (int r = 0; r < 4; r++) acc[r] = 0.0f;

        for (int tile = 0; tile < num_tiles; tile++) {
            const int tile_start = tile * PF_BC;
            const int tile_end_raw = min(tile_start + PF_BC, context_len);

            if (causal && tile_start > causal_limit) break;

            const int tile_len = tile_end_raw - tile_start;

            // ---- Load K tile (half2 vectorized, f16 in smem) ----
            {
                const int total_h2 = (tile_len * head_dim) / 2;
                for (int idx = tid; idx < total_h2; idx += PF_THREADS) {
                    int elem = idx * 2;
                    int t = elem / head_dim;
                    int d = elem % head_dim;
                    int kv_pos = tile_start + t;
                    int page_idx = kv_pos / block_size;
                    int page_off = kv_pos % block_size;
                    int phys_block = block_tables[(long long)seq_idx * max_blocks_per_seq + page_idx];
                    if (phys_block < 0 || phys_block >= num_blocks_total) {
                        s_kv[t * head_dim + d] = __float2half(CUDART_NAN_F);
                        s_kv[t * head_dim + d + 1] = __float2half(CUDART_NAN_F);
                        continue;
                    }
                    long long base = (((long long)phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d;
                    __half2 h2 = *reinterpret_cast<const __half2*>(&key_cache[base]);
                    s_kv[t * head_dim + d]     = h2.x;
                    s_kv[t * head_dim + d + 1] = h2.y;
                }
                int total_elems = tile_len * head_dim;
                if ((total_elems & 1) && tid == 0) {
                    int e = total_elems - 1;
                    int t = e / head_dim, d = e % head_dim;
                    int kv_pos = tile_start + t;
                    int pi = kv_pos / block_size, po = kv_pos % block_size;
                    int pb = block_tables[(long long)seq_idx * max_blocks_per_seq + pi];
                    s_kv[t * head_dim + d] = (pb >= 0 && pb < num_blocks_total)
                        ? key_cache[(((long long)pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d]
                        : __float2half(CUDART_NAN_F);
                }
            }
            __syncthreads();

            // ---- Q @ K^T with causal masking (f16 K from smem) ----
            for (int t = 0; t < tile_len; t++) {
                int kv_pos = tile_start + t;

                float dot = 0.0f;
                #pragma unroll
                for (int r = 0; r < dims_per_thread && r < 4; r++) {
                    int d = tid + r * PF_THREADS;
                    if (d < head_dim) dot += q_reg[r] * __half2float(s_kv[t * head_dim + d]);
                }
                dot = pf_warp_sum(dot);
                if (lane_id == 0) s_warp[warp_id] = dot;
                __syncthreads();
                if (tid == 0) {
                    float total = 0.0f;
                    for (int w = 0; w < PF_WARPS; w++) total += s_warp[w];
                    s_score[t] = (causal && kv_pos > causal_limit) ? -FLT_MAX : total;
                }
                __syncthreads();
            }

            // ---- Online softmax ----
            float tile_max = -FLT_MAX;
            if (tid == 0) {
                for (int t = 0; t < tile_len; t++) {
                    if (!isfinite(s_score[t])) { tile_max = CUDART_NAN_F; break; }
                    tile_max = fmaxf(tile_max, s_score[t]);
                }
                s_warp[0] = tile_max;
            }
            __syncthreads();
            tile_max = s_warp[0];
            __syncthreads();

            if (!isfinite(tile_max)) {
                row_sum = CUDART_NAN_F;
                break;
            }

            if (tile_max <= -FLT_MAX + 1.0f) {
                __syncthreads();
                continue;
            }

            float prev_max = row_max;
            float new_max = fmaxf(row_max, tile_max);
            if (new_max > prev_max && prev_max > -FLT_MAX) {
                float correction = expf(prev_max - new_max);
                #pragma unroll
                for (int r = 0; r < dims_per_thread && r < 4; r++) acc[r] *= correction;
                row_sum *= correction;
            }
            row_max = new_max;

            if (tid == 0) {
                float tsum = 0.0f;
                for (int t = 0; t < tile_len; t++) {
                    float v = (s_score[t] > -FLT_MAX + 1.0f) ? expf(s_score[t] - row_max) : 0.0f;
                    s_score[t] = v;
                    tsum += v;
                }
                s_warp[0] = tsum;
            }
            __syncthreads();
            row_sum += s_warp[0];
            __syncthreads();

            // ---- Load V tile (reuse s_kv, K consumed) ----
            {
                const int total_h2 = (tile_len * head_dim) / 2;
                for (int idx = tid; idx < total_h2; idx += PF_THREADS) {
                    int elem = idx * 2;
                    int t = elem / head_dim;
                    int d = elem % head_dim;
                    int kv_pos = tile_start + t;
                    int page_idx = kv_pos / block_size;
                    int page_off = kv_pos % block_size;
                    int phys_block = block_tables[(long long)seq_idx * max_blocks_per_seq + page_idx];
                    if (phys_block < 0 || phys_block >= num_blocks_total) {
                        s_kv[t * head_dim + d] = __float2half(CUDART_NAN_F);
                        s_kv[t * head_dim + d + 1] = __float2half(CUDART_NAN_F);
                        continue;
                    }
                    long long base = (((long long)phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d;
                    __half2 h2 = *reinterpret_cast<const __half2*>(&value_cache[base]);
                    s_kv[t * head_dim + d]     = h2.x;
                    s_kv[t * head_dim + d + 1] = h2.y;
                }
                int total_elems = tile_len * head_dim;
                if ((total_elems & 1) && tid == 0) {
                    int e = total_elems - 1;
                    int t = e / head_dim, d = e % head_dim;
                    int kv_pos = tile_start + t;
                    int pi = kv_pos / block_size, po = kv_pos % block_size;
                    int pb = block_tables[(long long)seq_idx * max_blocks_per_seq + pi];
                    s_kv[t * head_dim + d] = (pb >= 0 && pb < num_blocks_total)
                        ? value_cache[(((long long)pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d]
                        : __float2half(CUDART_NAN_F);
                }
            }
            __syncthreads();

            // ---- Accumulate P @ V (f16 V from smem) ----
            #pragma unroll
            for (int r = 0; r < dims_per_thread && r < 4; r++) {
                int d = tid + r * PF_THREADS;
                if (d < head_dim) {
                    float v_acc = 0.0f;
                    for (int t = 0; t < tile_len; t++)
                        v_acc += s_score[t] * __half2float(s_kv[t * head_dim + d]);
                    acc[r] += v_acc;
                }
            }
            __syncthreads();
        }

        // ---- Write output ----
        float inv_sum = isfinite(row_sum) && row_sum > 0.0f
            ? (1.0f / row_sum)
            : CUDART_NAN_F;
        long long out_base = ((long long)q_token_idx * num_heads + head_idx) * head_dim;
        #pragma unroll
        for (int r = 0; r < dims_per_thread && r < 4; r++) {
            int d = tid + r * PF_THREADS;
            if (d < head_dim)
                output[out_base + d] = __float2half(acc[r] * inv_sum);
        }
    }
}
