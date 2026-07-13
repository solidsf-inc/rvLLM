// Pruned lm-head greedy tail: INT4 pack-quantized GEMV over the KEPT vocab
// rows -> bf16-rounded scores -> grid argmax (left tie-break) -> remap winning
// local row to the global token id via keep_ids.
//
// Scores are bf16-rounded before the reduction. Ties break to the smallest
// local row index; the remap
// keep_ids[local] -> global preserves that because keep_ids is the kept-row
// order from the prune step.
//
// Weight layout (compressed-tensors pack-quantized, channel strategy for
// lm_head -- group_size == K so one scale per row):
//   weight_packed [K_rows, hidden/8] int32  -- 8 signed int4 per int32,
//                                               LSB-first, row-major over rows.
//   weight_scale  [K_rows, num_groups] f16  -- per (row, group) scale; for the
//                                               channel-strategy lm_head
//                                               num_groups == 1 (whole row).
//   keep_ids      [K_rows] int32             -- local row -> global token id.
//
// Kernels:
//   K1 lmhead_int4_gemv_bf16score_kernel : warp-per-row int4 GEMV, f32 accum,
//        per-group dequant scale, then bf16-ROUND the row score. Writes
//        scores_bf16score_as_f32 [M, K_rows] (still f32 storage, value is the
//        bf16-rounded score so downstream argmax is identity).
//   K2 argmax_remap_kernel : grid-strided block argmax over K_rows with left
//        tie-break, then output_global_token[row] = keep_ids[local_arg].
//   (optional) K3 scatter_full_vocab_neginf_kernel : behind a flag, scatter the
//        pruned scores into a [M, full_vocab] -inf buffer for logprobs only.
//
// No softcap here: argmax is softcap-invariant (30*tanh(x/30) is monotonic).
// The Python greedy path likewise argmaxes pre-softcap.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#include <math_constants.h>

#ifndef LMH_WARPS_PER_BLOCK
#define LMH_WARPS_PER_BLOCK 8   // 256 threads/block -> 8 rows/block
#endif

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum_lmh(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// Round-to-nearest-even f32 -> bf16 -> f32. Matches torch bf16 cast used by the
// reference lm-head GEMM (scores live in bf16 before the fp32 argmax upcast).
__device__ __forceinline__ float bf16_round(float x) {
    __nv_bfloat16 b = __float2bfloat16(x);
    return __bfloat162float(b);
}

// Unpack one compressed-tensors int4 lane. The pack stores OFFSET-BINARY
// (zero-point 8): the stored nibble n in [0,15] maps to value n-8 in [-8,7].
// This is not two's-complement sign extension.
__device__ __forceinline__ int s4(unsigned int nib) {
    return (int)(nib & 0xF) - 8;
}

// ---------------------------------------------------------------------------
// K1: INT4 pack-quantized lm-head GEMV with per-group scale + bf16-round score.
//   residual : [M, hidden] f16 (final-normed activations)
//   packed   : [K_rows, hidden/8] int32 (8 signed int4, LSB-first)
//   scale    : [K_rows, num_groups] f16  (group_size = hidden/num_groups)
//   scores   : [M, K_rows] f32 (each value is the bf16-rounded row score)
// One warp owns one (m_row, vocab_row) pair. Residual cached in smem as f32.
// ---------------------------------------------------------------------------
extern "C" __global__ void
lmhead_int4_gemv_bf16score_kernel(
    const __half* __restrict__ residual,   // [M, hidden]
    const int*    __restrict__ packed,      // [K_rows, hidden/8]
    const __half* __restrict__ scale,       // [K_rows, num_groups]
    float*        __restrict__ scores,       // [M, K_rows]
    int M,
    int K_rows,
    int hidden,
    int group_size
) {
    const int m = blockIdx.y;
    if (!residual || !packed || !scale || !scores ||
        M <= 0 || K_rows <= 0 || hidden <= 0 || group_size <= 0 ||
        (hidden & 7) != 0 || hidden % group_size != 0 ||
        group_size < 8 || (group_size & 7) != 0 ||
        blockDim.x != LMH_WARPS_PER_BLOCK * 32 ||
        blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.z != 1 || m >= M) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row  = blockIdx.x * LMH_WARPS_PER_BLOCK + warp;  // vocab row
    // The launcher must provide hidden * sizeof(float) dynamic shared memory.
    extern __shared__ float s_res[];
    for (int i = threadIdx.x; i < hidden; i += blockDim.x)
        s_res[i] = __half2float(residual[(long long)m * hidden + i]);
    __syncthreads();

    if (row >= K_rows) return;

    const int packed_cols = hidden >> 3;            // int32 lanes per row
    const int num_groups  = hidden / group_size;    // 1 for channel-strategy
    const int* prow = packed + (long long)row * packed_cols;
    const __half* srow = scale + (long long)row * num_groups;

    float acc = 0.0f;
    // each lane strides the packed int32 lanes; lane l handles lanes l, l+32,...
    for (int p = lane; p < packed_cols; p += 32) {
        unsigned int w = (unsigned int)prow[p];
        int base = p << 3;                          // logical col of nibble 0
        // group index is constant across the 8 nibbles only when group_size>=8
        // and 8-aligned (group_size=128 here), so compute once.
        int g = base / group_size;
        float sc = __half2float(srow[g]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int q = s4(w >> (4 * j));
            acc += isfinite(sc) ? (float)q * sc * s_res[base + j] : nanf("");
        }
    }
    acc = warp_reduce_sum_lmh(acc);
    if (lane == 0) {
        // bf16-round the score to match the reference GEMM output dtype, then
        // store as f32 (exact monotonic upcast) so argmax is identity.
        scores[(long long)m * K_rows + row] = bf16_round(acc);
    }
}

// ---------------------------------------------------------------------------
// K2: grid argmax over K_rows (left tie-break) + remap local -> global id.
//   scores       : [M, K_rows] f32
//   keep_ids     : [K_rows] int32   (local row -> global token id)
//   out_token    : [M] int32        (global token id of the winner)
// One block per (m). Block-strided scan; ties keep the smaller local index.
// ---------------------------------------------------------------------------
#ifndef LMH_ARGMAX_BLOCK
#define LMH_ARGMAX_BLOCK 1024
#endif

extern "C" __global__ void
lmhead_argmax_remap_kernel(
    const float* __restrict__ scores,    // [M, K_rows]
    const int*   __restrict__ keep_ids,  // [K_rows]
    int*         __restrict__ out_token, // [M]
    int M,
    int K_rows
) {
    const int m = blockIdx.x;
    if (!scores || !keep_ids || !out_token || M <= 0 || K_rows <= 0 ||
        m >= M || blockDim.x == 0 || blockDim.x > LMH_ARGMAX_BLOCK ||
        (blockDim.x & (blockDim.x - 1)) != 0 ||
        blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1) {
        return;
    }
    const int tid = threadIdx.x;
    const int n = blockDim.x;
    const float* x = scores + (long long)m * K_rows;

    __shared__ float s_val[LMH_ARGMAX_BLOCK];
    __shared__ int   s_idx[LMH_ARGMAX_BLOCK];

    float local_max = -CUDART_INF_F;
    int   local_idx = -1;
    for (int i = tid; i < K_rows; i += n) {
        float v = x[i];
        if (!isnan(v) && (local_idx < 0 || v > local_max)) {
            local_max = v;
            local_idx = i;
        }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (int s = n / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // On ties prefer the smaller stored index.
            if (s_idx[tid + s] >= 0 &&
                (s_idx[tid] < 0 || s_val[tid + s] > s_val[tid] ||
                 (s_val[tid + s] == s_val[tid] && s_idx[tid + s] < s_idx[tid]))) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        const int local = s_idx[0];
        const int global = local >= 0 ? keep_ids[local] : -1;
        out_token[m] = global >= 0 ? global : -1;
    }
}

// ---------------------------------------------------------------------------
// K3 (optional logprobs path): scatter pruned scores into
// a [M, full_vocab] buffer pre-filled with -inf. Non-kept columns stay -inf.
// out_full must be pre-initialized to -inf by the caller (persistent template).
// ---------------------------------------------------------------------------
extern "C" __global__ void
lmhead_scatter_full_vocab_kernel(
    const float* __restrict__ scores,    // [M, K_rows]
    const int*   __restrict__ keep_ids,  // [K_rows]
    float*       __restrict__ out_full,  // [M, full_vocab], pre-filled -inf
    int M,
    int K_rows,
    int full_vocab
) {
    const int m = blockIdx.x;
    if (!scores || !keep_ids || !out_full || M <= 0 || K_rows <= 0 ||
        full_vocab <= 0 || m >= M || blockDim.x == 0 || blockDim.x > 1024 ||
        blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1) {
        return;
    }
    float* row = out_full + (long long)m * full_vocab;
    for (int gid = threadIdx.x; gid < full_vocab; gid += blockDim.x) {
        row[gid] = -CUDART_INF_F;
    }
    __syncthreads();
    for (int local = threadIdx.x; local < K_rows; local += blockDim.x) {
        const int gid = keep_ids[local];
        if (gid >= 0 && gid < full_vocab) {
            row[gid] = scores[(long long)m * K_rows + local];
        }
    }
}

// ---------------------------------------------------------------------------
// dequant_pack_to_f16: compressed-tensors pack-quantized [N, K] -> FP16 [N, K]
// row-major. Used offline at load to feed W4a8Lib::encode_fp16 (the pack ->
// w4a8 reorder goes through the canonical encoder, not a bespoke shuffle).
//   packed : [N, K/8] int32 (8 signed int4 per lane, LSB-first)
//   scale  : [N, num_groups] f16  (num_groups = K/group_size)
//   out    : [N, K] f16
// One block per output row N; threads stride the K columns.
// ---------------------------------------------------------------------------
extern "C" __global__ void
dequant_pack_to_f16_kernel(
    const int*    __restrict__ packed,   // [N, K/8]
    const __half* __restrict__ scale,    // [N, num_groups]
    __half*       __restrict__ out_f16,  // [N, K]
    int N,
    int K,
    int group_size
) {
    const int row = blockIdx.x;
    if (!packed || !scale || !out_f16 || N <= 0 || K <= 0 ||
        group_size <= 0 || (K & 7) != 0 || K % group_size != 0 ||
        group_size < 8 || (group_size & 7) != 0 || row >= N ||
        blockDim.x == 0 || blockDim.x > 1024 ||
        blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1) {
        return;
    }
    const int packed_cols = K >> 3;
    const int num_groups  = K / group_size;
    const int* prow = packed + (long long)row * packed_cols;
    const __half* srow = scale + (long long)row * num_groups;
    __half* orow = out_f16 + (long long)row * K;

    for (int col = threadIdx.x; col < K; col += blockDim.x) {
        int p = col >> 3;
        int j = col & 7;
        unsigned int w = (unsigned int)prow[p];
        int q = s4(w >> (4 * j));
        int g = col / group_size;
        const float sc = __half2float(srow[g]);
        float v = isfinite(sc) ? (float)q * sc : nanf("");
        orow[col] = __float2half(v);
    }
}

// ===========================================================================
// TEST: int4 GEMV + bf16-round + argmax-remap parity vs host f32 reference.
// ===========================================================================
#ifdef LMH_TEST_MAIN
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

static unsigned rng=0xC0FFEEu;
static float frand(){ rng=rng*1664525u+1013904223u; return ((rng>>8)&0xFFFFFF)/16777216.0f*2.f-1.f; }
static float host_bf16_round(float x){
    unsigned u; memcpy(&u,&x,4);
    unsigned lsb=(u>>16)&1u; u += 0x7FFFu + lsb; u &= 0xFFFF0000u;
    float r; memcpy(&r,&u,4); return r;
}

int main(){
    const int HIDDEN=2560, GROUP=2560 /*channel strategy*/, M=2;
    const int K_ROWS=4096;
    const int FULL_VOCAB=262144;
    const int PACKED_COLS=HIDDEN/8;
    const int NGROUPS=HIDDEN/GROUP;

    std::vector<__half> h_res(M*HIDDEN);
    for(auto&v:h_res) v=__float2half(frand());
    std::vector<int> h_packed((size_t)K_ROWS*PACKED_COLS);
    for(auto&w:h_packed){ unsigned acc=0; for(int j=0;j<8;j++){ unsigned q=(unsigned)(rng>>j)&0xF; rng=rng*1103515245u+12345u; acc|=(q&0xF)<<(4*j);} w=(int)acc; }
    std::vector<__half> h_scale((size_t)K_ROWS*NGROUPS);
    for(auto&s:h_scale) s=__float2half(0.01f+0.02f*((frand()+1)*0.5f));
    std::vector<int> h_keep(K_ROWS);
    // keep_ids: a strictly increasing-ish scattered subset of full_vocab
    { int g=0; for(int i=0;i<K_ROWS;i++){ g += 1 + (int)((frand()+1)*30); if(g>=FULL_VOCAB) g=FULL_VOCAB-1; h_keep[i]=g; } }

    // ---- host reference ----
    std::vector<float> ref_scores((size_t)M*K_ROWS);
    for(int m=0;m<M;m++) for(int r=0;r<K_ROWS;r++){
        float acc=0.f;
        for(int p=0;p<PACKED_COLS;p++){
            unsigned w=(unsigned)h_packed[(size_t)r*PACKED_COLS+p];
            int base=p*8; int g=base/GROUP; float sc=__half2float(h_scale[(size_t)r*NGROUPS+g]);
            for(int j=0;j<8;j++){ int q=(int)((w>>(4*j))&0xF); q-=8;
                acc += (float)q*sc*__half2float(h_res[(size_t)m*HIDDEN+base+j]); }
        }
        ref_scores[(size_t)m*K_ROWS+r]=host_bf16_round(acc);
    }
    std::vector<int> ref_tok(M);
    for(int m=0;m<M;m++){ float best=-FLT_MAX; int arg=0;
        for(int r=0;r<K_ROWS;r++){ float v=ref_scores[(size_t)m*K_ROWS+r]; if(v>best){best=v;arg=r;} }
        ref_tok[m]=h_keep[arg]; }

    // ---- device ----
    __half *d_res,*d_scale; int *d_packed,*d_keep,*d_tok; float* d_scores;
    CK(cudaMalloc(&d_res,h_res.size()*2)); CK(cudaMalloc(&d_scale,h_scale.size()*2));
    CK(cudaMalloc(&d_packed,h_packed.size()*4)); CK(cudaMalloc(&d_keep,K_ROWS*4));
    CK(cudaMalloc(&d_tok,M*4)); CK(cudaMalloc(&d_scores,(size_t)M*K_ROWS*4));
    CK(cudaMemcpy(d_res,h_res.data(),h_res.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_scale,h_scale.data(),h_scale.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_packed,h_packed.data(),h_packed.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_keep,h_keep.data(),K_ROWS*4,cudaMemcpyHostToDevice));

    dim3 g1((K_ROWS+LMH_WARPS_PER_BLOCK-1)/LMH_WARPS_PER_BLOCK, M);
    size_t shm=HIDDEN*sizeof(float);
    lmhead_int4_gemv_bf16score_kernel<<<g1,LMH_WARPS_PER_BLOCK*32,shm>>>(d_res,d_packed,d_scale,d_scores,M,K_ROWS,HIDDEN,GROUP);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    lmhead_argmax_remap_kernel<<<M,LMH_ARGMAX_BLOCK>>>(d_scores,d_keep,d_tok,M,K_ROWS);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    std::vector<int> dev_tok(M);
    CK(cudaMemcpy(dev_tok.data(),d_tok,M*4,cudaMemcpyDeviceToHost));
    std::vector<float> dev_scores((size_t)M*K_ROWS);
    CK(cudaMemcpy(dev_scores.data(),d_scores,(size_t)M*K_ROWS*4,cudaMemcpyDeviceToHost));

    double maxabs=0; for(size_t i=0;i<dev_scores.size();i++) maxabs=fmax(maxabs,fabs(dev_scores[i]-ref_scores[i]));
    int ok=1; for(int m=0;m<M;m++){ printf("m=%d host_tok=%d dev_tok=%d %s\n",m,ref_tok[m],dev_tok[m],ref_tok[m]==dev_tok[m]?"OK":"MISMATCH"); if(ref_tok[m]!=dev_tok[m]) ok=0; }
    {
        const float tie_scores[4] = {1.0f, 3.0f, 3.0f, 2.0f};
        const int tie_keep[4] = {10, 20, 30, 40};
        CK(cudaMemcpy(d_scores,tie_scores,sizeof(tie_scores),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_keep,tie_keep,sizeof(tie_keep),cudaMemcpyHostToDevice));
        lmhead_argmax_remap_kernel<<<1,32>>>(d_scores,d_keep,d_tok,1,4);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        int tie_token=-1; CK(cudaMemcpy(&tie_token,d_tok,sizeof(tie_token),cudaMemcpyDeviceToHost));
        if(tie_token!=20){ printf("left-tie parity FAIL: got %d expected 20\n",tie_token); ok=0; }
    }
    printf("score max_abs_diff=%.3e  ARGMAX %s\n",maxabs, ok?"PASS":"FAIL");
    return ok?0:1;
}
#endif
