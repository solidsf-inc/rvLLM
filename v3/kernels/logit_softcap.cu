// Logit softcapping: logits = cap * tanh(logits / cap)
//
// Applied in-place on f16 logits before argmax sampling.
// Gemma 3/4 uses cap=30.0.
//
// Grid:  (num_tokens, 1, 1)
// Block: (min(vocab, 1024), 1, 1)

#include <cuda_fp16.h>

extern "C"
__global__ void logit_softcap_kernel(
    __half* __restrict__ logits,    // [num_tokens, vocab]
    int vocab,
    float cap
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (logits == nullptr || vocab <= 0 || !isfinite(cap) || cap <= 0.0f ||
        blockDim.x == 0 || blockDim.x > 1024 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1) return;
    const float inv_cap = 1.0f / cap;

    __half* row_ptr = logits + (long long)row * vocab;

    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = __half2float(row_ptr[i]);
        float capped = cap * tanhf(v * inv_cap);
        row_ptr[i] = __float2half(capped);
    }
}
