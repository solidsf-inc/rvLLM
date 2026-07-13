// Temperature scale + deterministic top-K' partial selection over one f32
// logits row. ONE launch per sampled decode step:
//   1. selection runs on raw logit bits (order is temperature-invariant for
//      T > 0); the 1/T scale is applied at output-write time only,
//   2. radix-selects the K' largest elements under the unique 64-bit key
//        key(i) = mono(bits(logits[i])) << 32 | (0xFFFFFFFF - i)
//      where mono is the sign-flip transform that makes f32 compare as
//      unsigned. The index tie-break makes the selected SET deterministic
//      across runs even when logits tie exactly at the K' boundary —
//      required by the same-seed-identical-stream contract,
//   3. compacts (logit * inv_temp, index) pairs, arrival order arbitrary,
//      for the host sampling tail (sort + softmax + top-p + draw over
//      <= 1024 floats).
//
// Launch: grid (1,1,1), block (1024,1,1), no dynamic smem.
// Single block on purpose: 8 byte-wide histogram passes + 1 compaction pass
// re-read the row. Non-finite logits fail closed with out_count=-1.

#include <math.h>

extern "C" __global__ void sample_topk_f32_kernel(
    const float* __restrict__ logits, // [vocab]
    float inv_temp,                   // 1/T, T > 0
    int vocab,
    int k_select,                     // 1..=1024 and <= vocab
    float* __restrict__ out_vals,     // [k_select] logit * inv_temp
    int* __restrict__ out_idx,        // [k_select]
    int* __restrict__ out_count       // [1]; == k_select on exit
) {
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    __shared__ unsigned int hist[256];
    __shared__ unsigned long long prefix; // selected high bytes so far
    __shared__ unsigned int need;         // ranks left to resolve inside prefix
    __shared__ unsigned int out_pos;
    __shared__ int invalid;
    __shared__ int overflow;

    if (tid == 0) {
        prefix = 0ull;
        need = (unsigned int)k_select;
        out_pos = 0u;
        invalid = 0;
        overflow = 0;
    }
    __syncthreads();
    if (logits == nullptr || out_vals == nullptr || out_idx == nullptr || out_count == nullptr ||
        gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != 1024 || blockDim.y != 1 || blockDim.z != 1 ||
        vocab <= 0 || k_select <= 0 || k_select > 1024 || k_select > vocab ||
        !isfinite(inv_temp) || inv_temp <= 0.0f) {
        if (out_count != nullptr && tid == 0) out_count[0] = -1;
        return;
    }
    for (int i = tid; i < vocab; i += nthreads) {
        if (!isfinite(logits[i])) atomicExch(&invalid, 1);
    }
    __syncthreads();
    if (invalid) {
        if (tid == 0) out_count[0] = -1;
        return;
    }

    // 8 radix passes, high byte -> low byte. After pass b the top (8-b)
    // bytes of the k_select-th largest key are known; `need` counts the
    // ranks still to place among keys sharing that prefix.
    for (int pass = 7; pass >= 0; --pass) {
        for (int b = tid; b < 256; b += nthreads) {
            hist[b] = 0u;
        }
        __syncthreads();
        for (int i = tid; i < vocab; i += nthreads) {
            unsigned int u = __float_as_uint(logits[i]);
            unsigned int mono = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
            unsigned long long key =
                ((unsigned long long)mono << 32) | (unsigned int)(0xFFFFFFFFu - i);
            if (pass == 7 || (key >> (8 * (pass + 1))) == prefix) {
                atomicAdd(&hist[(unsigned int)(key >> (8 * pass)) & 0xFFu], 1u);
            }
        }
        __syncthreads();
        if (tid == 0) {
            unsigned int cum = 0;
            int chosen = 0;
            for (int b = 255; b >= 0; --b) {
                cum += hist[b];
                if (cum >= need) {
                    chosen = b;
                    need -= cum - hist[b]; // ranks consumed by higher bins
                    break;
                }
            }
            prefix = (prefix << 8) | (unsigned long long)(unsigned int)chosen;
        }
        __syncthreads();
    }

    // prefix is now the exact k_select-th largest key; keys are unique, so
    // { i : key(i) >= prefix } has exactly k_select members.
    const unsigned long long threshold = prefix;

    for (int i = tid; i < vocab; i += nthreads) {
        unsigned int u = __float_as_uint(logits[i]);
        unsigned int mono = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        unsigned long long key =
            ((unsigned long long)mono << 32) | (unsigned int)(0xFFFFFFFFu - i);
        if (key >= threshold) {
            unsigned int pos = atomicAdd(&out_pos, 1u);
            if (pos < (unsigned int)k_select) {
                out_vals[pos] = logits[i] * inv_temp;
                out_idx[pos] = i;
            } else {
                atomicExch(&overflow, 1);
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        out_count[0] = (!overflow && out_pos == (unsigned int)k_select) ? (int)out_pos : -1;
    }
}
