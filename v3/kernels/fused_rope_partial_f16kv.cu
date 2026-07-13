// Partial RoPE + F16 paged-KV-cache write (Gemma 4, F16 mode).
// Same as fused_rope_partial_fp8kv but outputs F16 Q and F16 KV cache.
// No FP8 quantization -- full F16 precision throughout.

#include <cuda_fp16.h>

extern "C"
__global__ void fused_rope_partial_f16kv_kernel(
    const __half* __restrict__ q_in,
    const __half* __restrict__ k_in,
    const __half* __restrict__ v_in,
    __half* __restrict__ q_out,
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    const int* __restrict__ positions,
    const int* __restrict__ slot_mapping,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    int max_positions,
    int num_cache_slots
) {
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    if (q_in == nullptr || q_out == nullptr || positions == nullptr ||
        num_tokens <= 0 || num_heads <= 0 || num_kv_heads < 0 ||
        head_dim <= 0 || head_dim % 2 != 0 || rotary_dim < 0 ||
        rotary_dim > head_dim || rotary_dim % 2 != 0 || max_positions <= 0 ||
        (rotary_dim > 0 && (cos_table == nullptr || sin_table == nullptr)) ||
        (num_kv_heads > 0 && (k_in == nullptr || v_in == nullptr || key_cache == nullptr ||
         value_cache == nullptr || slot_mapping == nullptr || num_cache_slots <= 0)) ||
        (long long)num_heads * head_dim > 2147483647LL ||
        (long long)num_kv_heads * head_dim > 2147483647LL ||
        token_idx >= num_tokens ||
        head_idx >= (num_heads > num_kv_heads ? num_heads : num_kv_heads) ||
        blockDim.x < (unsigned)((head_dim + 1) / 2) || blockDim.x > 1024 ||
        blockDim.y != 1 || blockDim.z != 1 || gridDim.z != 1) return;
    const int half_rotary = rotary_dim / 2;
    const int half_head   = head_dim / 2;
    const int tid         = threadIdx.x;
    if (tid >= half_head) return;

    const int pos = positions[token_idx];
    if (pos < 0 || pos >= max_positions) return;

    // Q head: split-half RoPE, output as F16
    if (head_idx < num_heads) {
        long long q_base = ((long long)token_idx * num_heads + head_idx) * head_dim;

        if (tid < half_rotary) {
            float cos_val = __half2float(cos_table[(long long)pos * half_rotary + tid]);
            float sin_val = __half2float(sin_table[(long long)pos * half_rotary + tid]);
            float q_lo = __half2float(q_in[q_base + tid]);
            float q_hi = __half2float(q_in[q_base + tid + half_head]);
            q_out[q_base + tid]             = __float2half(q_lo * cos_val - q_hi * sin_val);
            q_out[q_base + tid + half_head] = __float2half(q_lo * sin_val + q_hi * cos_val);
        } else {
            q_out[q_base + tid]             = q_in[q_base + tid];
            q_out[q_base + tid + half_head] = q_in[q_base + tid + half_head];
        }
    }

    // K head: split-half RoPE + F16 cache write. V: direct F16 cache write.
    if (head_idx < num_kv_heads) {
        long long k_base = ((long long)token_idx * num_kv_heads + head_idx) * head_dim;
        int slot = slot_mapping[token_idx];

        if (slot >= 0 && slot < num_cache_slots) {
            long long cache_offset = ((long long)slot * num_kv_heads + head_idx) * head_dim;

            if (tid < half_rotary) {
                float cos_val = __half2float(cos_table[(long long)pos * half_rotary + tid]);
                float sin_val = __half2float(sin_table[(long long)pos * half_rotary + tid]);
                float k_lo = __half2float(k_in[k_base + tid]);
                float k_hi = __half2float(k_in[k_base + tid + half_head]);
                key_cache[cache_offset + tid]             = __float2half(k_lo * cos_val - k_hi * sin_val);
                key_cache[cache_offset + tid + half_head] = __float2half(k_lo * sin_val + k_hi * cos_val);
            } else {
                key_cache[cache_offset + tid]             = k_in[k_base + tid];
                key_cache[cache_offset + tid + half_head] = k_in[k_base + tid + half_head];
            }

            // V: no rotation, direct F16 copy
            long long v_base = ((long long)token_idx * num_kv_heads + head_idx) * head_dim;
            value_cache[cache_offset + tid]             = v_in[v_base + tid];
            value_cache[cache_offset + tid + half_head] = v_in[v_base + tid + half_head];
        }
    }
}
