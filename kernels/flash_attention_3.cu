// Decode attention prototype with f16 shared-memory KV tiles.
// Supported head dimensions are positive even values up to 128.
//
// Shared memory layout:
//   non-GQA: s_kv[BC*(D+2)] half, s_score[BC] float, s_warp[8] float
//   GQA:     s_kv[BC*(D+2)] half, s_scores[HPG*(BC+1)] float, s_warp[8] float
//
// Launch: non-GQA: grid(num_seqs, num_heads), block(256)
//         GQA:     grid(num_seqs, num_kv_heads), block(256)

#include <float.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#define FA3_BC 64
#define FA3_THREADS 256
#define FA3_WARPS 8
#define FA3_GQA_MAX_HPG 8
#define FA3_SCORE_STRIDE (FA3_BC + 1)

__device__ __forceinline__ float fa3_warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float fa3_warp_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, val, offset);
        val = isfinite(val) && isfinite(other) ? fmaxf(val, other) : CUDART_NAN_F;
    }
    return val;
}

__device__ __forceinline__ float fa3_block_reduce_max(
    float val, int tid, int lane_id, int warp_id, float* s_warp
) {
    val = fa3_warp_max(val);
    if (lane_id == 0) s_warp[warp_id] = val;
    __syncthreads();
    if (tid == 0) {
        float m = s_warp[0];
        for (int w = 1; w < FA3_WARPS; w++)
            m = isfinite(m) && isfinite(s_warp[w]) ? fmaxf(m, s_warp[w]) : CUDART_NAN_F;
        s_warp[0] = m;
    }
    __syncthreads();
    return s_warp[0];
}

__device__ __forceinline__ bool fa3_valid_block_table(
    const int* block_tables, int seq_idx, int context_len, int block_size,
    int max_blocks_per_seq, int num_blocks_total, int tid, int* invalid
) {
    if (tid == 0) *invalid = 0;
    __syncthreads();
    const int pages = (int)(((long long)context_len + block_size - 1) / block_size);
    for (int page = tid; page < pages; page += FA3_THREADS) {
        int physical = block_tables[(long long)seq_idx * max_blocks_per_seq + page];
        if (physical < 0 || physical >= num_blocks_total) atomicExch(invalid, 1);
    }
    __syncthreads();
    return *invalid == 0;
}

__device__ __forceinline__ float fa3_block_reduce_sum(
    float val, int tid, int lane_id, int warp_id, float* s_warp
) {
    val = fa3_warp_sum(val);
    if (lane_id == 0) s_warp[warp_id] = val;
    __syncthreads();
    if (tid == 0) {
        float s = s_warp[0];
        for (int w = 1; w < FA3_WARPS; w++) s += s_warp[w];
        s_warp[0] = s;
    }
    __syncthreads();
    return s_warp[0];
}

// Helper: load a KV tile from paged cache into padded shared memory
__device__ __forceinline__ void fa3_load_kv_tile(
    __half* s_kv, const __half* cache,
    const int* block_tables, int seq_idx, int max_blocks_per_seq,
    int tile_start, int tile_len, int num_kv_heads, int kv_head_idx,
    int head_dim, int kv_stride, int block_size, int tid
) {
    const int total_h2 = (tile_len * head_dim) / 2;
    for (int idx = tid; idx < total_h2; idx += FA3_THREADS) {
        int elem = idx * 2;
        int t = elem / head_dim;
        int d = elem % head_dim;
        int kv_pos = tile_start + t;
        int page_idx = kv_pos / block_size;
        int page_off = kv_pos % block_size;
        int phys_block = block_tables[(long long)seq_idx * max_blocks_per_seq + page_idx];
        long long base = (((long long)phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d;
        __half2 h2 = *reinterpret_cast<const __half2*>(&cache[base]);
        s_kv[t * kv_stride + d]     = h2.x;
        s_kv[t * kv_stride + d + 1] = h2.y;
    }
    int total_elems = tile_len * head_dim;
    if ((total_elems & 1) && tid == 0) {
        int e = total_elems - 1;
        int t = e / head_dim, d = e % head_dim;
        int kv_pos = tile_start + t;
        int pi = kv_pos / block_size, po = kv_pos % block_size;
        int pb = block_tables[(long long)seq_idx * max_blocks_per_seq + pi];
        s_kv[t * kv_stride + d] = cache[(((long long)pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d];
    }
}

// ======================================================================
// Non-GQA decode kernel
// ======================================================================

extern "C"
__global__ void __launch_bounds__(FA3_THREADS, 2)
flash_attention_3_decode_f16io_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ query,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int max_context_len,
    int num_blocks_total
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane_id  = tid % 32;

    if (output == nullptr || query == nullptr || key_cache == nullptr || value_cache == nullptr ||
        block_tables == nullptr || context_lens == nullptr || !isfinite(scale) || scale <= 0.0f ||
        num_heads <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0 ||
        head_dim <= 0 || head_dim > 128 || head_dim % 2 != 0 || block_size <= 0 ||
        max_blocks_per_seq <= 0 || max_context_len <= 0 || num_blocks_total <= 0 ||
        head_idx >= num_heads || blockDim.x != FA3_THREADS || blockDim.y != 1 ||
        blockDim.z != 1 || gridDim.y != (unsigned)num_heads || gridDim.z != 1 ||
        (long long)max_blocks_per_seq * block_size < max_context_len) return;
    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;
    if (context_len < 0 || context_len > max_context_len) return;
    __shared__ int invalid_table;
    if (!fa3_valid_block_table(block_tables, seq_idx, context_len, block_size,
                               max_blocks_per_seq, num_blocks_total, tid, &invalid_table)) {
        long long base = ((long long)seq_idx * num_heads + head_idx) * head_dim;
        for (int d = tid; d < head_dim; d += FA3_THREADS)
            output[base + d] = __float2half(CUDART_NAN_F);
        return;
    }

    const int kv_head_idx = (num_kv_heads == num_heads)
        ? head_idx
        : (head_idx / (num_heads / num_kv_heads));

    const int num_tiles = (int)(((long long)context_len + FA3_BC - 1) / FA3_BC);
    const int kv_stride = head_dim + 2;
    const int half2_iters = (head_dim + 63) / 64;
    const int acc_dims = (head_dim + FA3_THREADS - 1) / FA3_THREADS;

    extern __shared__ char smem_raw[];
    __half* s_kv   = (__half*)smem_raw;
    float* s_score = (float*)(s_kv + FA3_BC * kv_stride);
    float* s_warp  = s_score + FA3_BC;

    // Q in registers: lane-based mapping for warp-parallel dot products.
    // lane_id handles dims: lane_id*2+0, lane_id*2+1, lane_id*2+64, lane_id*2+65 (for hd=128)
    const long long q_base = ((long long)seq_idx * num_heads + head_idx) * head_dim;
    float q_reg[4];
    #pragma unroll
    for (int r = 0; r < half2_iters && r < 2; r++) {
        int d = lane_id * 2 + r * 64;
        if (d + 1 < head_dim) {
            q_reg[r*2]   = __half2float(query[q_base + d]) * scale;
            q_reg[r*2+1] = __half2float(query[q_base + d + 1]) * scale;
        } else if (d < head_dim) {
            q_reg[r*2]   = __half2float(query[q_base + d]) * scale;
            q_reg[r*2+1] = 0.0f;
        } else {
            q_reg[r*2] = 0.0f;
            q_reg[r*2+1] = 0.0f;
        }
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) acc[r] = 0.0f;

    for (int tile = 0; tile < num_tiles; tile++) {
        const int tile_start = tile * FA3_BC;
        const int tile_len = min(FA3_BC, context_len - tile_start);

        // Load K tile
        fa3_load_kv_tile(s_kv, key_cache, block_tables, seq_idx, max_blocks_per_seq,
                          tile_start, tile_len, num_kv_heads, kv_head_idx,
                          head_dim, kv_stride, block_size, tid);
        __syncthreads();

        // Warp-parallel QK^T: each warp handles one position
        for (int base_t = 0; base_t < tile_len; base_t += FA3_WARPS) {
            int t = base_t + warp_id;
            if (t < tile_len) {
                float dot = 0.0f;
                #pragma unroll
                for (int r = 0; r < half2_iters && r < 2; r++) {
                    int d = lane_id * 2 + r * 64;
                    if (d + 1 < head_dim) {
                        dot += q_reg[r*2]   * __half2float(s_kv[t * kv_stride + d]);
                        dot += q_reg[r*2+1] * __half2float(s_kv[t * kv_stride + d + 1]);
                    } else if (d < head_dim) {
                        dot += q_reg[r*2] * __half2float(s_kv[t * kv_stride + d]);
                    }
                }
                dot = fa3_warp_sum(dot);
                if (lane_id == 0) s_score[t] = dot;
            }
        }
        __syncthreads();

        // Online softmax -- parallel reductions
        float tile_max = fa3_block_reduce_max(
            (tid < tile_len) ? s_score[tid] : -FLT_MAX,
            tid, lane_id, warp_id, s_warp);

        float prev_max = row_max;
        float new_max = isfinite(row_max) && isfinite(tile_max)
            ? fmaxf(row_max, tile_max)
            : CUDART_NAN_F;
        if (new_max > prev_max && prev_max > -FLT_MAX) {
            float correction = expf(prev_max - new_max);
            #pragma unroll
            for (int r = 0; r < acc_dims && r < 4; r++) acc[r] *= correction;
            row_sum *= correction;
        }
        row_max = new_max;

        float my_exp = (tid < tile_len) ? expf(s_score[tid] - new_max) : 0.0f;
        if (tid < tile_len) s_score[tid] = my_exp;
        row_sum += fa3_block_reduce_sum(my_exp, tid, lane_id, warp_id, s_warp);

        // Load V tile (reuses s_kv)
        fa3_load_kv_tile(s_kv, value_cache, block_tables, seq_idx, max_blocks_per_seq,
                          tile_start, tile_len, num_kv_heads, kv_head_idx,
                          head_dim, kv_stride, block_size, tid);
        __syncthreads();

        // P @ V accumulation
        #pragma unroll
        for (int r = 0; r < acc_dims && r < 4; r++) {
            int d = tid + r * FA3_THREADS;
            if (d < head_dim) {
                float v_acc = 0.0f;
                for (int t = 0; t < tile_len; t++)
                    v_acc += s_score[t] * __half2float(s_kv[t * kv_stride + d]);
                acc[r] += v_acc;
            }
        }
        __syncthreads();
    }

    // Write output
    float inv_sum = isfinite(row_sum) && row_sum > 0.0f
        ? (1.0f / row_sum)
        : CUDART_NAN_F;
    long long out_base = ((long long)seq_idx * num_heads + head_idx) * head_dim;
    #pragma unroll
    for (int r = 0; r < acc_dims && r < 4; r++) {
        int d = tid + r * FA3_THREADS;
        if (d < head_dim)
            output[out_base + d] = __float2half(acc[r] * inv_sum);
    }
}

// ======================================================================
// GQA-optimized decode kernel
// One block per KV head, all query heads in the group share KV loads.
// ======================================================================

extern "C"
__global__ void __launch_bounds__(FA3_THREADS, 2)
flash_attention_3_decode_gqa_f16io_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ query,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    int num_blocks_total
) {
    const int seq_idx     = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int tid         = threadIdx.x;
    const int warp_id     = tid / 32;
    const int lane_id     = tid % 32;

    if (output == nullptr || query == nullptr || key_cache == nullptr || value_cache == nullptr ||
        block_tables == nullptr || context_lens == nullptr || !isfinite(scale) || scale <= 0.0f ||
        num_heads <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0 ||
        num_heads / num_kv_heads > FA3_GQA_MAX_HPG || head_dim <= 0 || head_dim > 128 ||
        head_dim % 2 != 0 || block_size <= 0 || max_context_len <= 0 ||
        max_blocks_per_seq <= 0 || num_blocks_total <= 0 || kv_head_idx >= num_kv_heads ||
        blockDim.x != FA3_THREADS || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != (unsigned)num_kv_heads || gridDim.z != 1 ||
        (long long)max_blocks_per_seq * block_size < max_context_len) return;
    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;
    if (context_len < 0 || context_len > max_context_len) return;
    __shared__ int invalid_table;
    if (!fa3_valid_block_table(block_tables, seq_idx, context_len, block_size,
                               max_blocks_per_seq, num_blocks_total, tid, &invalid_table)) {
        const int heads_per_group = num_heads / num_kv_heads;
        for (int g = 0; g < heads_per_group; g++) {
            int g_head = kv_head_idx * heads_per_group + g;
            long long base = ((long long)seq_idx * num_heads + g_head) * head_dim;
            for (int d = tid; d < head_dim; d += FA3_THREADS)
                output[base + d] = __float2half(CUDART_NAN_F);
        }
        return;
    }

    const int heads_per_group = num_heads / num_kv_heads;
    const int num_tiles = (int)(((long long)context_len + FA3_BC - 1) / FA3_BC);
    const int kv_stride = head_dim + 2;
    const int half2_iters = (head_dim + 63) / 64;
    const int acc_dims = (head_dim + FA3_THREADS - 1) / FA3_THREADS;

    extern __shared__ char smem_raw[];
    __half* s_kv    = (__half*)smem_raw;
    float* s_scores = (float*)(s_kv + FA3_BC * kv_stride);
    float* s_warp   = s_scores + FA3_GQA_MAX_HPG * FA3_SCORE_STRIDE;

    // Pre-load ALL Q vectors for the group into registers (lane-based mapping).
    // Eliminates redundant global reads: 1 load per head at start vs per-tile.
    float q_regs[FA3_GQA_MAX_HPG][4];
    float head_row_max[FA3_GQA_MAX_HPG];
    float head_row_sum[FA3_GQA_MAX_HPG];
    float head_acc[FA3_GQA_MAX_HPG][4];

    for (int g = 0; g < heads_per_group && g < FA3_GQA_MAX_HPG; g++) {
        int g_head = kv_head_idx * heads_per_group + g;
        long long q_base = ((long long)seq_idx * num_heads + g_head) * head_dim;
        #pragma unroll
        for (int r = 0; r < half2_iters && r < 2; r++) {
            int d = lane_id * 2 + r * 64;
            if (d + 1 < head_dim) {
                q_regs[g][r*2]   = __half2float(query[q_base + d]) * scale;
                q_regs[g][r*2+1] = __half2float(query[q_base + d + 1]) * scale;
            } else if (d < head_dim) {
                q_regs[g][r*2]   = __half2float(query[q_base + d]) * scale;
                q_regs[g][r*2+1] = 0.0f;
            } else {
                q_regs[g][r*2]   = 0.0f;
                q_regs[g][r*2+1] = 0.0f;
            }
        }
        head_row_max[g] = -FLT_MAX;
        head_row_sum[g] = 0.0f;
        #pragma unroll
        for (int r = 0; r < 4; r++) head_acc[g][r] = 0.0f;
    }

    for (int tile = 0; tile < num_tiles; tile++) {
        const int tile_start = tile * FA3_BC;
        const int tile_len = min(FA3_BC, context_len - tile_start);

        // Load K tile ONCE for all heads
        fa3_load_kv_tile(s_kv, key_cache, block_tables, seq_idx, max_blocks_per_seq,
                          tile_start, tile_len, num_kv_heads, kv_head_idx,
                          head_dim, kv_stride, block_size, tid);
        __syncthreads();

        // Per-head: warp-parallel QK^T + online softmax
        for (int g = 0; g < heads_per_group && g < FA3_GQA_MAX_HPG; g++) {
            float* g_scores = s_scores + g * FA3_SCORE_STRIDE;

            // Warp-parallel QK^T: 8 positions per round
            for (int base_t = 0; base_t < tile_len; base_t += FA3_WARPS) {
                int t = base_t + warp_id;
                if (t < tile_len) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int r = 0; r < half2_iters && r < 2; r++) {
                        int d = lane_id * 2 + r * 64;
                        if (d + 1 < head_dim) {
                            dot += q_regs[g][r*2]   * __half2float(s_kv[t * kv_stride + d]);
                            dot += q_regs[g][r*2+1] * __half2float(s_kv[t * kv_stride + d + 1]);
                        } else if (d < head_dim) {
                            dot += q_regs[g][r*2] * __half2float(s_kv[t * kv_stride + d]);
                        }
                    }
                    dot = fa3_warp_sum(dot);
                    if (lane_id == 0) g_scores[t] = dot;
                }
            }
            __syncthreads();

            // Online softmax -- parallel block reductions
            float tile_max = fa3_block_reduce_max(
                (tid < tile_len) ? g_scores[tid] : -FLT_MAX,
                tid, lane_id, warp_id, s_warp);

            float prev_max = head_row_max[g];
            float new_max = isfinite(prev_max) && isfinite(tile_max)
                ? fmaxf(prev_max, tile_max)
                : CUDART_NAN_F;
            if (new_max > prev_max && prev_max > -FLT_MAX) {
                float correction = expf(prev_max - new_max);
                #pragma unroll
                for (int r = 0; r < acc_dims && r < 4; r++) head_acc[g][r] *= correction;
                head_row_sum[g] *= correction;
            }
            head_row_max[g] = new_max;

            float my_exp = (tid < tile_len) ? expf(g_scores[tid] - new_max) : 0.0f;
            if (tid < tile_len) g_scores[tid] = my_exp;
            head_row_sum[g] += fa3_block_reduce_sum(my_exp, tid, lane_id, warp_id, s_warp);
        }

        // Load V tile ONCE (reuses s_kv, K consumed)
        fa3_load_kv_tile(s_kv, value_cache, block_tables, seq_idx, max_blocks_per_seq,
                          tile_start, tile_len, num_kv_heads, kv_head_idx,
                          head_dim, kv_stride, block_size, tid);
        __syncthreads();

        // P@V with V reuse across all heads: load V[t][d] once, multiply by
        // each head's score. Saves (HPG-1) * tile_len smem reads of V.
        #pragma unroll
        for (int r = 0; r < acc_dims && r < 4; r++) {
            int d = tid + r * FA3_THREADS;
            if (d < head_dim) {
                for (int t = 0; t < tile_len; t++) {
                    float v = __half2float(s_kv[t * kv_stride + d]);
                    for (int g = 0; g < heads_per_group && g < FA3_GQA_MAX_HPG; g++)
                        head_acc[g][r] += s_scores[g * FA3_SCORE_STRIDE + t] * v;
                }
            }
        }
        __syncthreads();
    }

    // Write output for all heads in the group
    for (int g = 0; g < heads_per_group && g < FA3_GQA_MAX_HPG; g++) {
        int g_head = kv_head_idx * heads_per_group + g;
        float inv = isfinite(head_row_sum[g]) && head_row_sum[g] > 0.0f
            ? (1.0f / head_row_sum[g])
            : CUDART_NAN_F;
        long long out_base = ((long long)seq_idx * num_heads + g_head) * head_dim;
        #pragma unroll
        for (int r = 0; r < acc_dims && r < 4; r++) {
            int d = tid + r * FA3_THREADS;
            if (d < head_dim)
                output[out_base + d] = __float2half(head_acc[g][r] * inv);
        }
    }
}
