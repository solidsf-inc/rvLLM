// GPU-side argmax kernel: finds the token ID with maximum logit per row.
// Eliminates full logits DtoH copy for greedy (temperature=0) decoding.
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(vocab_size, 1024), 1, 1)
//   Shared memory: none (uses static shared arrays)
//
// Each block finds the argmax of one token's logits row via shared memory reduction,
// then writes the winning token ID to output_token[row].

#include <float.h>
#include <cuda_fp16.h>
#include <math_constants.h>

__device__ __forceinline__ bool argmax_pair_better(
    float cand_val,
    int cand_idx,
    float best_val,
    int best_idx
) {
    if (cand_idx < 0) return false;
    if (best_idx < 0) return true;
    return cand_val > best_val || (cand_val == best_val && cand_idx < best_idx);
}

__device__ __forceinline__ bool valid_argmax_launch(int vocab_size) {
    const int n = blockDim.x;
    return vocab_size > 0 && n > 0 && n <= 1024 && (n & (n - 1)) == 0 &&
           blockDim.y == 1 && blockDim.z == 1;
}

extern "C"
__global__ void argmax_f16_kernel(
    const __half* __restrict__ logits,
    int* __restrict__ output_token,
    int vocab_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int n = blockDim.x;

    if (logits == nullptr || output_token == nullptr || !valid_argmax_launch(vocab_size)) {
        if (output_token != nullptr && tid == 0) output_token[row] = -1;
        return;
    }
    const __half* x = logits + (long long)row * vocab_size;

    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];

    float local_max = -CUDART_INF_F;
    int   local_idx = -1;
    for (int i = tid; i < vocab_size; i += stride) {
        float v = __half2float(x[i]);
        if (!isnan(v) && argmax_pair_better(v, i, local_max, local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (int s = n / 2; s > 0; s >>= 1) {
        if (tid < s &&
            argmax_pair_better(s_val[tid + s], s_idx[tid + s], s_val[tid], s_idx[tid])) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output_token[row] = s_idx[0];
    }
}

extern "C"
__global__ void argmax_kernel(
    const float* __restrict__ logits,
    int* __restrict__ output_token,
    int vocab_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int n = blockDim.x;

    if (logits == nullptr || output_token == nullptr || !valid_argmax_launch(vocab_size)) {
        if (output_token != nullptr && tid == 0) output_token[row] = -1;
        return;
    }
    const float* x = logits + (long long)row * vocab_size;

    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];

    // Pass 1: thread-local max across strided elements
    float local_max = -CUDART_INF_F;
    int   local_idx = -1;
    for (int i = tid; i < vocab_size; i += stride) {
        float v = x[i];
        if (!isnan(v) && argmax_pair_better(v, i, local_max, local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Power-of-two tree reduction. Ties select the lowest token ID.
    for (int s = n / 2; s > 0; s >>= 1) {
        if (tid < s &&
            argmax_pair_better(s_val[tid + s], s_idx[tid + s], s_val[tid], s_idx[tid])) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the result
    if (tid == 0) {
        output_token[row] = s_idx[0];
    }
}

// output_token must be initialized to -1 before launching a multi-part grid.
extern "C"
__global__ void argmax_grid_f32_kernel(
    const float* __restrict__ logits,
    int* __restrict__ output_token,
    int vocab_size
) {
    const int row = blockIdx.y;
    const int part = blockIdx.x;
    const int parts = gridDim.x;
    const int tid = threadIdx.x;
    if (logits == nullptr || output_token == nullptr || !valid_argmax_launch(vocab_size)) {
        return;
    }
    const long long chunk = ((long long)vocab_size + parts - 1) / parts;
    const long long begin = (long long)part * chunk;
    const long long end = begin + chunk < vocab_size ? begin + chunk : vocab_size;
    const float* x = logits + (long long)row * vocab_size;

    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];

    float local_max = -CUDART_INF_F;
    int local_idx = -1;
    for (long long i = begin + tid; i < end; i += blockDim.x) {
        float v = x[i];
        const int idx = (int)i;
        if (!isnan(v) && argmax_pair_better(v, idx, local_max, local_idx)) {
            local_max = v;
            local_idx = idx;
        }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s &&
            argmax_pair_better(s_val[tid + s], s_idx[tid + s], s_val[tid], s_idx[tid])) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && s_idx[0] >= 0) {
        int* out = output_token + row;
        int cur = *out;
        while (true) {
            if (cur < 0 || cur >= vocab_size) {
                int old = atomicCAS(out, cur, s_idx[0]);
                if (old == cur) break;
                cur = old;
                continue;
            }
            float cur_val = x[cur];
            bool take = (cur_val != cur_val) ||
                s_val[0] > cur_val ||
                (s_val[0] == cur_val && s_idx[0] < cur);
            if (!take) {
                break;
            }
            int old = atomicCAS(out, cur, s_idx[0]);
            if (old == cur) {
                break;
            }
            cur = old;
        }
    }
}
