// Half-precision embedding gather kernel: gathers embedding rows on GPU.
// Embedding table is f16, output is f16. No f32 conversion needed -- pure copy.
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(hidden_size, 1024), 1, 1)
//   Shared memory: none

#include <cuda_fp16.h>

extern "C"
__global__ void embedding_gather_f16_kernel(
    __half* __restrict__ output,            // [num_tokens, hidden_size]
    const __half* __restrict__ embed_table, // [vocab_size, hidden_size]
    const int* __restrict__ token_ids,      // [num_tokens]
    int hidden_size,
    int vocab_size
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const int token_id = token_ids[token_idx];
    const long long out_offset = (long long)token_idx * (long long)hidden_size;

    // Bounds check: out-of-range tokens get zeros
    if (token_id < 0 || token_id >= vocab_size) {
        __half zero = __float2half(0.0f);
        for (int i = tid; i < hidden_size; i += stride) {
            output[out_offset + i] = zero;
        }
        return;
    }

    const long long embed_offset = (long long)token_id * (long long)hidden_size;
    for (int i = tid; i < hidden_size; i += stride) {
        output[out_offset + i] = embed_table[embed_offset + i];
    }
}
