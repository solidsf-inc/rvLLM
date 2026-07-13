// FA3 SM90 wrapper around Dao-AILab/flash-attention commit
// 1233b73b6c95340c65c9edfe929611838354fc6e (BSD-3-Clause).
// Flash_fwd_params, dispatch templates, scheduler preparation, and split-KV
// combine calls follow hopper/flash.h, hopper/flash_fwd_launch_template.h,
// hopper/flash_prepare_scheduler.cu, and hopper/flash_fwd_combine.cu at that
// exact revision. The build verifies the checkout before compiling this file.
// Compiled on H100 with CUTLASS headers; no ATen/PyTorch dependency.
//
// Provides paged-KV decode attention for SM90 using WGMMA/TMA.
// KV cache layout: [num_blocks, block_size, num_kv_heads, head_dim] (matches rvLLM).

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include "flash.h"
#include "heuristics.h"
#include "tile_size.h"

// Forward declarations of the instantiated templates we link against.
// Paged, non-split, PackGQA=true, fp16
template<> void run_mha_fwd_<90, cutlass::half_t, 128, 128, false, true, false, true>(
    Flash_fwd_params &params, cudaStream_t stream);
template<> void run_mha_fwd_<90, cutlass::half_t, 256, 256, false, true, false, true>(
    Flash_fwd_params &params, cudaStream_t stream);

// Paged, split, PackGQA=true (for low-batch high-seqlen), fp16
template<> void run_mha_fwd_<90, cutlass::half_t, 128, 128, true, true, false, true>(
    Flash_fwd_params &params, cudaStream_t stream);
template<> void run_mha_fwd_<90, cutlass::half_t, 256, 256, true, true, false, true>(
    Flash_fwd_params &params, cudaStream_t stream);

// Paged, non-split, PackGQA=true, e4m3 (FP8 KV)
template<> void run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, false, true, false, true>(
    Flash_fwd_params &params, cudaStream_t stream);
template<> void run_mha_fwd_<90, cutlass::float_e4m3_t, 256, 256, false, true, false, true>(
    Flash_fwd_params &params, cudaStream_t stream);

// Paged, split, PackGQA=true, e4m3 (FP8 KV)
template<> void run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, true, true, false, true>(
    Flash_fwd_params &params, cudaStream_t stream);
template<> void run_mha_fwd_<90, cutlass::float_e4m3_t, 256, 256, true, true, false, true>(
    Flash_fwd_params &params, cudaStream_t stream);

// Split-KV combine output type must match run_mha_fwd_'s output type.
template<> void run_mha_fwd_combine_<cutlass::half_t, float, 128>(
    Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);
template<> void run_mha_fwd_combine_<cutlass::half_t, float, 256>(
    Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);
template<> void run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(
    Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);
template<> void run_mha_fwd_combine_<cutlass::bfloat16_t, float, 256>(
    Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);

// prepare_varlen_num_blocks from flash_prepare_scheduler.cu
void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t stream,
                               bool packgqa, int blockM, int blockN, bool enable_pdl);

// Block sizes for hdim=128, fp16, PagedKVNonTMA=true, non-causal
static constexpr int kBlockM = 128;
static constexpr int kBlockN = 128;
#ifndef RVLLM_FA3_ABI_VERSION
#error "RVLLM_FA3_ABI_VERSION must be supplied by build_fa3.sh"
#endif
static_assert(RVLLM_FA3_ABI_VERSION == 2, "unsupported rvLLM FA3 ABI");
static constexpr int kRvllmFa3AbiVersion = RVLLM_FA3_ABI_VERSION;
static constexpr const char* kFa3UpstreamRevision = "1233b73b6c95340c65c9edfe929611838354fc6e";
static constexpr int kFa3Fp8OutputF16 = 1;
static constexpr int kFa3Fp8OutputElementBytes = 2;
static_assert(sizeof(cutlass::bfloat16_t) == kFa3Fp8OutputElementBytes);
static_assert(sizeof(cutlass::half_t) == kFa3Fp8OutputElementBytes);

// Pinned FA3 writes BF16 for E4M3 input. The rvLLM C ABI promises F16, so the
// wrapper converts each contiguous output element in place on the same stream.
// A single uint16_t pointer avoids restrict/type-alias assumptions: every
// thread loads one complete BF16 bit pattern before replacing that same element.
__global__ void fa3_sm90_bf16_to_f16_inplace(unsigned short* data, size_t elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    data[index] = __half_as_ushort(
        __float2half_rn(__bfloat162float(__ushort_as_bfloat16(data[index]))));
}

static inline int ceil_div_positive(int x, int divisor) {
    return 1 + (x - 1) / divisor;
}

static inline int round_multiple(int x, int multiple) {
    return ceil_div_positive(x, multiple) * multiple;
}

static inline bool supported_head_dim(int head_dim) {
    return head_dim == 128 || head_dim == 256;
}

struct Fa3WorkspaceLayout {
    size_t metadata_bytes;
    size_t lse_bytes;
    size_t oaccum_bytes;
    size_t lseaccum_bytes;
    size_t total_bytes;
    int num_splits;
    int num_sm;
};

static bool checked_mul_size(size_t lhs, size_t rhs, size_t* out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) return false;
    *out = lhs * rhs;
    return true;
}

static bool checked_add_size(size_t lhs, size_t rhs, size_t* out) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) return false;
    *out = lhs + rhs;
    return true;
}

static bool checked_align_size(size_t value, size_t alignment, size_t* out) {
    size_t with_padding = 0;
    if (alignment == 0 || !checked_add_size(value, alignment - 1, &with_padding)) return false;
    *out = with_padding / alignment * alignment;
    return true;
}

static bool fa3_sm90_pick_num_splits(
    int max_seqlen_q,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    bool is_fp8,
    bool is_prefill,
    int window_size_left,
    int* num_splits,
    int* num_sm
) {
    if (max_seqlen_q <= 0 || batch_size <= 0 || num_heads <= 0 ||
        num_kv_heads <= 0 || num_heads % num_kv_heads != 0 ||
        !supported_head_dim(head_dim) || block_size <= 0 ||
        max_blocks_per_seq <= 0 || window_size_left < -1 ||
        batch_size > INT_MAX - 3 || max_seqlen_q > INT_MAX - (kBlockM - 1)) return false;

    int64_t seqlen_k_wide = static_cast<int64_t>(block_size) * max_blocks_per_seq;
    if (seqlen_k_wide <= 0 || seqlen_k_wide > INT_MAX - (kBlockN - 1)) return false;
    if (window_size_left >= 0 &&
        static_cast<int64_t>(window_size_left) + 1 > seqlen_k_wide) return false;
    int seqlen_k = static_cast<int>(seqlen_k_wide);
    const bool is_causal = is_prefill && window_size_left < 0;
    const bool is_local = window_size_left >= 0;
    auto tile = tile_size_fwd_sm90(
        head_dim, head_dim, is_causal, is_local,
        is_fp8 ? 1 : 2, false, true, false);
    int block_m = std::get<0>(tile);
    int block_n = std::get<1>(tile);
    if (block_m <= 0 || block_n <= 0 ||
        max_seqlen_q > INT_MAX - (block_m - 1) ||
        seqlen_k > INT_MAX - (block_n - 1)) return false;
    int qhead_per_khead = num_heads / num_kv_heads;
    int64_t packed_q = static_cast<int64_t>(max_seqlen_q) * qhead_per_khead;
    if (packed_q <= 0 || packed_q > INT_MAX - (block_m - 1)) return false;
    int num_m_blocks = ceil_div_positive(static_cast<int>(packed_q), block_m);
    int64_t local_span = static_cast<int64_t>(window_size_left) + 1 + block_m;
    int seqlen_k_loaded = window_size_left < 0
        ? seqlen_k
        : static_cast<int>(std::min<int64_t>(seqlen_k, local_span));
    int num_n_blocks = ceil_div_positive(seqlen_k_loaded, block_n);
    // seqused_k makes both decode and prefill varlen. Upstream's dynamic
    // split upper-bound models one long sequence rather than multiplying by B.
    int64_t total_mblocks_wide = static_cast<int64_t>(num_kv_heads) * num_m_blocks;
    int kv_bytes_per_elem = is_fp8 ? 1 : 2;
    int64_t size_one_kv_head_wide = static_cast<int64_t>(seqlen_k) * head_dim * 2 * kv_bytes_per_elem;
    if (total_mblocks_wide <= 0 || total_mblocks_wide > INT_MAX ||
        size_one_kv_head_wide <= 0 || size_one_kv_head_wide > INT_MAX) return false;

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    cudaDeviceProp props = {};
    if (cudaGetDeviceProperties(&props, device) != cudaSuccess ||
        props.major * 10 + props.minor != 90 || props.multiProcessorCount <= 0) return false;

    int splits = num_splits_heuristic(
        static_cast<int>(total_mblocks_wide), props.multiProcessorCount,
        num_n_blocks, num_m_blocks, static_cast<int>(size_one_kv_head_wide),
        is_causal || is_local, 128);
    if (is_prefill && head_dim >= 256) splits = 1;
    if (splits <= 0 || splits > 128) return false;
    *num_splits = splits;
    *num_sm = props.multiProcessorCount;
    return true;
}

static bool fa3_sm90_workspace_layout(
    int total_q,
    int max_seqlen_q,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    bool is_fp8,
    bool is_prefill,
    int window_size_left,
    Fa3WorkspaceLayout* layout
) {
    if (!layout || total_q <= 0 || (is_prefill && total_q < batch_size) ||
        (!is_prefill && (total_q != batch_size || max_seqlen_q != 1)) ||
        max_seqlen_q > total_q) return false;

    const size_t query_rows = static_cast<size_t>(is_prefill ? total_q : batch_size);
    const size_t heads = static_cast<size_t>(num_heads);
    const size_t dim = static_cast<size_t>(head_dim);
    size_t index_stride = 0;
    const size_t max_index = static_cast<size_t>(std::numeric_limits<int64_t>::max());
    if (!checked_mul_size(query_rows, heads, &index_stride) ||
        !checked_mul_size(index_stride, dim, &index_stride) ||
        index_stride > max_index ||
        !checked_mul_size(heads, dim, &index_stride) ||
        index_stride > max_index ||
        !checked_mul_size(static_cast<size_t>(num_kv_heads),
                          dim, &index_stride) ||
        index_stride > max_index ||
        !checked_mul_size(index_stride, static_cast<size_t>(block_size), &index_stride) ||
        index_stride > max_index) return false;

    int splits = 0;
    int num_sm = 0;
    if (!fa3_sm90_pick_num_splits(
            max_seqlen_q, batch_size, num_heads, num_kv_heads, head_dim,
            block_size, max_blocks_per_seq, is_fp8, is_prefill,
            window_size_left, &splits, &num_sm)) return false;

    const size_t split_count = static_cast<size_t>(splits);
    const size_t b_rounded = static_cast<size_t>(round_multiple(batch_size, 4));
    const bool varlen_sort_batches = window_size_left < 0;
    const bool head_swizzle = is_prefill || window_size_left >= 0;
    const size_t metadata_vectors = 2 +
        static_cast<size_t>(varlen_sort_batches) +
        static_cast<size_t>(head_swizzle);
    size_t value = 0;

    if (!checked_mul_size(b_rounded, metadata_vectors, &value) ||
        !checked_add_size(value, 1, &value) ||
        value > static_cast<size_t>(INT_MAX) ||
        !checked_mul_size(value, sizeof(int), &value) ||
        !checked_align_size(value, 256, &layout->metadata_bytes)) return false;

    if (!checked_mul_size(query_rows, heads, &value) ||
        !checked_mul_size(value, sizeof(float), &value) ||
        !checked_align_size(value, 256, &layout->lse_bytes)) return false;

    layout->oaccum_bytes = 0;
    layout->lseaccum_bytes = 0;
    if (splits > 1) {
        if (!checked_mul_size(query_rows, split_count, &value) ||
            !checked_mul_size(value, heads, &value) ||
            !checked_mul_size(value, dim, &value) ||
            !checked_mul_size(value, sizeof(float), &value) ||
            !checked_align_size(value, 256, &layout->oaccum_bytes)) return false;
        if (!checked_mul_size(query_rows, split_count, &value) ||
            !checked_mul_size(value, heads, &value) ||
            !checked_mul_size(value, sizeof(float), &value) ||
            !checked_align_size(value, 256, &layout->lseaccum_bytes)) return false;
    }

    if (!checked_add_size(layout->metadata_bytes, layout->lse_bytes, &value) ||
        !checked_add_size(value, layout->oaccum_bytes, &value) ||
        !checked_add_size(value, layout->lseaccum_bytes, &layout->total_bytes)) return false;
    layout->num_splits = splits;
    layout->num_sm = num_sm;
    return layout->total_bytes != 0;
}

extern "C" uint64_t fa3_sm90_decode_workspace_size(
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, int is_fp8, int window_size_left
) {
    Fa3WorkspaceLayout layout = {};
    return fa3_sm90_workspace_layout(
        batch_size, 1, batch_size, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks_per_seq, is_fp8 != 0, false,
        window_size_left, &layout) ? static_cast<uint64_t>(layout.total_bytes) : 0;
}

extern "C" uint64_t fa3_sm90_prefill_workspace_size(
    int total_q, int max_seqlen_q, int batch_size, int num_heads,
    int num_kv_heads, int head_dim, int block_size, int max_blocks_per_seq,
    int is_fp8, int window_size_left
) {
    Fa3WorkspaceLayout layout = {};
    return fa3_sm90_workspace_layout(
        total_q, max_seqlen_q, batch_size, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks_per_seq, is_fp8 != 0, true,
        window_size_left, &layout) ? static_cast<uint64_t>(layout.total_bytes) : 0;
}

extern "C" int rvllm_fa3_abi_version() { return kRvllmFa3AbiVersion; }
extern "C" const char* rvllm_fa3_upstream_revision() { return kFa3UpstreamRevision; }
extern "C" int fa3_sm90_fp8_output_dtype() { return kFa3Fp8OutputF16; }
extern "C" int fa3_sm90_fp8_output_element_size() { return kFa3Fp8OutputElementBytes; }

// Internal dispatcher: both fp16 KV and fp8 (e4m3) KV paths share param setup.
// Decode path: cu_seqlens_q == nullptr, max_seqlen_q == 1, total_q == batch_size.
// Prefill path: cu_seqlens_q is the per-seq Q offset prefix sum of length
// batch+1; max_seqlen_q is the longest seq's Q length; total_q is the
// sum; is_causal_prefill == true applies a causal mask so query t only
// sees K positions 0..=t within its own seq.
static int fa3_sm90_paged_decode_impl(
    void* q_ptr,
    void* k_cache_ptr,
    void* v_cache_ptr,
    void* o_ptr,
    int*  block_tables_ptr,
    int*  context_lens_ptr,
    int*  cu_seqlens_q_ptr,   // nullptr for decode
    int   max_seqlen_q,       // 1 for decode
    int   total_q,            // batch_size for decode, sum of Q lens for prefill
    void* workspace_ptr,
    size_t workspace_bytes,
    float scale,
    int   batch_size,
    int   num_heads,
    int   num_kv_heads,
    int   head_dim,
    int   block_size,
    int   max_blocks_per_seq,
    int   num_blocks_total,
    bool  is_fp8,
    bool  is_causal_prefill,
    float* q_descale_ptr,
    float* k_descale_ptr,
    float* v_descale_ptr,
    int   window_size_left,  // -1 = full attention, >= 0 = sliding window
    cudaStream_t stream
) {
    if (!q_ptr || !k_cache_ptr || !v_cache_ptr || !o_ptr || !block_tables_ptr ||
        !context_lens_ptr || batch_size <= 0 || num_heads <= 0 ||
        num_kv_heads <= 0 || num_heads % num_kv_heads != 0 || block_size <= 0 ||
        max_blocks_per_seq <= 0 || num_blocks_total <= 0 ||
        max_blocks_per_seq > num_blocks_total || total_q <= 0 ||
        max_seqlen_q <= 0 || max_seqlen_q > total_q || window_size_left < -1 ||
        !std::isfinite(scale) || scale <= 0.0f) return -2;
    if (cu_seqlens_q_ptr == nullptr && (max_seqlen_q != 1 || total_q != batch_size)) return -3;
    if (cu_seqlens_q_ptr != nullptr && total_q < batch_size) return -4;
    if (is_fp8 && (!q_descale_ptr || !k_descale_ptr || !v_descale_ptr)) return -5;
    if (!supported_head_dim(head_dim)) {
        fprintf(stderr, "fa3_sm90_paged_decode: only head_dim=128 or 256 supported, got %d\n", head_dim);
        return -1;
    }

    const bool is_prefill = cu_seqlens_q_ptr != nullptr;
    Fa3WorkspaceLayout workspace_layout = {};
    if (!fa3_sm90_workspace_layout(
            total_q, max_seqlen_q, batch_size, num_heads, num_kv_heads,
            head_dim, block_size, max_blocks_per_seq, is_fp8, is_prefill,
            window_size_left, &workspace_layout)) return -6;
    if (!workspace_ptr ||
        reinterpret_cast<uintptr_t>(workspace_ptr) % 256 != 0 ||
        workspace_bytes < workspace_layout.total_bytes) return -11;
    if (reinterpret_cast<uintptr_t>(o_ptr) % kFa3Fp8OutputElementBytes != 0) return -13;
    size_t cache_elements = 0;
    const size_t max_index = static_cast<size_t>(std::numeric_limits<int64_t>::max());
    if (!checked_mul_size(static_cast<size_t>(num_blocks_total),
                          static_cast<size_t>(block_size), &cache_elements) ||
        !checked_mul_size(cache_elements, static_cast<size_t>(num_kv_heads),
                          &cache_elements) ||
        !checked_mul_size(cache_elements, static_cast<size_t>(head_dim),
                          &cache_elements) ||
        cache_elements > max_index) return -12;
    const int arch = 90;
    const int num_sm = workspace_layout.num_sm;
    const int b_rounded = round_multiple(batch_size, 4);

    char* ws = (char*)workspace_ptr;
    int* metadata_ptr = (int*)ws;
    float* lse_ptr = (float*)(ws + workspace_layout.metadata_bytes);
    float* oaccum_ptr = (float*)(
        ws + workspace_layout.metadata_bytes + workspace_layout.lse_bytes);
    float* lseaccum_ptr = (float*)(
        ws + workspace_layout.metadata_bytes + workspace_layout.lse_bytes +
        workspace_layout.oaccum_bytes);

    // Zero the metadata region (semaphore must start at 0)
    if (cudaMemsetAsync(metadata_ptr, 0, workspace_layout.metadata_bytes, stream) != cudaSuccess) return -9;

    // Populate Flash_fwd_params
    Flash_fwd_params params = {};

    params.is_bf16 = false;
    params.is_fp32 = false;
    params.is_e4m3 = is_fp8;
    // Per-tensor FP8 descale: scalar broadcast (strides all zero).
    params.q_descale_ptr = is_fp8 ? q_descale_ptr : nullptr;
    params.k_descale_ptr = is_fp8 ? k_descale_ptr : nullptr;
    params.v_descale_ptr = is_fp8 ? v_descale_ptr : nullptr;
    params.q_descale_batch_stride = 0;
    params.q_descale_head_stride  = 0;
    params.k_descale_batch_stride = 0;
    params.k_descale_head_stride  = 0;
    params.v_descale_batch_stride = 0;
    params.v_descale_head_stride  = 0;

    // Q: decode is [batch, num_heads, head_dim] (1 row per seq).
    // Prefill is [total_q, num_heads, head_dim] indexed via cu_seqlens_q.
    params.q_ptr = q_ptr;
    params.q_batch_stride = static_cast<int64_t>(num_heads) * head_dim;
    params.q_row_stride = static_cast<int64_t>(num_heads) * head_dim;
    params.q_head_stride = head_dim;

    // K: [num_blocks_total, block_size, num_kv_heads, head_dim]
    params.k_ptr = k_cache_ptr;
    params.k_batch_stride = static_cast<int64_t>(block_size) * num_kv_heads * head_dim;  // stride between pages
    params.k_row_stride = static_cast<int64_t>(num_kv_heads) * head_dim;  // stride between tokens in page
    params.k_head_stride = head_dim;

    // V: same layout as K
    params.v_ptr = v_cache_ptr;
    params.v_batch_stride = static_cast<int64_t>(block_size) * num_kv_heads * head_dim;
    params.v_row_stride = static_cast<int64_t>(num_kv_heads) * head_dim;
    params.v_head_stride = head_dim;
    params.v_dim_stride = 1;  // contiguous in head_dim

    // O: [batch, num_heads, head_dim] treated as [batch, 1, num_heads, head_dim]
    params.o_ptr = o_ptr;
    params.o_batch_stride = static_cast<int64_t>(num_heads) * head_dim;
    params.o_row_stride = static_cast<int64_t>(num_heads) * head_dim;
    params.o_head_stride = head_dim;

    // Dimensions
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_kv_heads;
    params.d = head_dim;
    params.d_rounded = head_dim;
    params.dv = head_dim;
    params.dv_rounded = head_dim;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_blocks_per_seq * block_size;
    params.seqlen_q_rounded = round_multiple(max_seqlen_q, kBlockM);
    params.seqlen_k_rounded = round_multiple(params.seqlen_k, kBlockN);
    params.total_q = total_q;  // decode: batch; prefill: sum of per-seq Q lens

    // Paged KV
    params.page_table = block_tables_ptr;
    params.page_table_batch_stride = max_blocks_per_seq;
    params.page_size = block_size;
    params.num_pages = num_blocks_total;
    params.pagedkv_tma = false;

    // Varlen via seqused_k (actual context lengths per sequence)
    params.seqused_k = context_lens_ptr;
    params.cu_seqlens_q = cu_seqlens_q_ptr;   // nullptr for decode
    params.cu_seqlens_k = nullptr;
    params.cu_seqlens_knew = nullptr;
    params.seqused_q = nullptr;
    params.leftpad_k = nullptr;

    // No KV append
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.seqlen_knew = 0;
    params.total_knew = 0;

    // No QV
    params.qv_ptr = nullptr;

    // No rotary (applied before attention in rvLLM)
    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.seqlens_rotary = nullptr;
    params.rotary_dim = 0;
    params.is_rotary_interleaved = false;

    // No KV batch indexing
    params.kv_batch_idx = nullptr;
    params.b_k = batch_size;

    // Softmax
    params.scale_softmax = scale;
    params.softcap = 0.0f;

    // No dropout
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;

    // Decode: seqlen_q == 1, non-causal (query sees all prior KV).
    // Prefill: causal self-attention over Q-then-KV, query t sees K 0..=t.
    params.is_causal = is_causal_prefill && window_size_left < 0;
    params.is_local = window_size_left >= 0;
    params.window_size_left = (window_size_left >= 0) ? window_size_left : (params.seqlen_k - 1);
    params.window_size_right = 0;
    params.attention_chunk = 0;

    // Architecture
    params.arch = arch;
    params.num_sm = num_sm;

    // LSE output
    params.softmax_lse_ptr = lse_ptr;

    // PackGQA = true (matches template)
    params.pack_gqa = true;

    // The workspace query and launch share this exact split decision.
    int ns = workspace_layout.num_splits;
    params.num_splits = ns;
    bool use_split = ns > 1;


    // Scheduler metadata setup — matches upstream hopper/flash_api.cpp:
    //   varlen_sort_batches = !is_local
    //   head_swizzle        = is_causal || is_local
    params.varlen_sort_batches = !params.is_local;
    params.head_swizzle        = params.is_causal || params.is_local;
    int num_vectors = 2;  // num_splits_dynamic + num_m_blocks (always for prepare_varlen)
    if (params.varlen_sort_batches) num_vectors += 1;  // varlen_batch_idx
    if (params.head_swizzle) num_vectors += 1;          // num_nheads_in_l2

    int head_swizzle_offset = b_rounded * (params.varlen_sort_batches ? 3 : 2);
    int semaphore_offset = b_rounded * num_vectors;

    params.num_splits_dynamic_ptr = metadata_ptr;
    params.num_m_blocks_ptr = metadata_ptr + b_rounded;
    params.varlen_batch_idx_ptr = params.varlen_sort_batches ? metadata_ptr + b_rounded * 2 : nullptr;
    params.num_nheads_in_l2_ptr = params.head_swizzle ? metadata_ptr + head_swizzle_offset : nullptr;
    params.tile_count_semaphore = metadata_ptr + semaphore_offset;
    params.tile_count_semaphore_offset = semaphore_offset;

    params.skip_scheduler_metadata_computation = false;
    params.prepare_varlen_pdl = false;

    // Split-KV output buffers
    if (use_split) {
        const int64_t rows = is_prefill ? total_q : batch_size;
        const int64_t o_split_stride = rows * num_heads * head_dim;
        const int64_t o_head_stride = is_prefill ? rows * head_dim : head_dim;
        const int64_t lse_split_stride = rows * num_heads;
        if (o_split_stride > std::numeric_limits<decltype(params.oaccum_split_stride)>::max() ||
            o_head_stride > std::numeric_limits<decltype(params.oaccum_head_stride)>::max() ||
            lse_split_stride > std::numeric_limits<decltype(params.lseaccum_split_stride)>::max()) return -12;

        // Pinned upstream layouts:
        // decode  [split, batch, head, row=1, dim]
        // prefill [split, head, total_q, dim]
        params.oaccum_ptr = oaccum_ptr;
        params.oaccum_split_stride = o_split_stride;
        params.oaccum_batch_stride = static_cast<int64_t>(num_heads) * head_dim;
        params.oaccum_row_stride = head_dim;
        params.oaccum_head_stride = o_head_stride;

        // LSE partial is row-major in Q: [split, head, query_row].
        params.softmax_lseaccum_ptr = lseaccum_ptr;
        params.lseaccum_split_stride = lse_split_stride;
        params.lseaccum_batch_stride = num_heads;
        params.lseaccum_head_stride = is_prefill ? total_q : 1;
    } else {
        params.oaccum_ptr = nullptr;
        params.softmax_lseaccum_ptr = nullptr;
    }

    params.rng_state = nullptr;

    if (is_fp8) {
        if (use_split) {
            if (head_dim == 128) {
                run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, true, true, false, true>(params, stream);

                run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(params, stream, false);
            } else {
                run_mha_fwd_<90, cutlass::float_e4m3_t, 256, 256, true, true, false, true>(params, stream);

                run_mha_fwd_combine_<cutlass::bfloat16_t, float, 256>(params, stream, false);
            }
        } else {
            if (head_dim == 128) {
                run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, false, true, false, true>(params, stream);
            } else {
                run_mha_fwd_<90, cutlass::float_e4m3_t, 256, 256, false, true, false, true>(params, stream);
            }
        }
    } else {
        if (use_split) {
            if (head_dim == 128) {
                run_mha_fwd_<90, cutlass::half_t, 128, 128, true, true, false, true>(params, stream);
                run_mha_fwd_combine_<cutlass::half_t, float, 128>(params, stream, false);
            } else {
                run_mha_fwd_<90, cutlass::half_t, 256, 256, true, true, false, true>(params, stream);
                run_mha_fwd_combine_<cutlass::half_t, float, 256>(params, stream, false);
            }
        } else {
            if (head_dim == 128) {
                run_mha_fwd_<90, cutlass::half_t, 128, 128, false, true, false, true>(params, stream);
            } else {
                run_mha_fwd_<90, cutlass::half_t, 256, 256, false, true, false, true>(params, stream);
            }
        }
    }

    if (cudaPeekAtLastError() != cudaSuccess) return -10;
    if (is_fp8) {
        size_t output_elements = 0;
        size_t rows_and_heads = 0;
        if (!checked_mul_size(static_cast<size_t>(total_q), static_cast<size_t>(num_heads),
                              &rows_and_heads) ||
            !checked_mul_size(rows_and_heads, static_cast<size_t>(head_dim), &output_elements)) {
            return -12;
        }
        constexpr size_t threads = 256;
        const size_t blocks = 1 + (output_elements - 1) / threads;
        if (blocks > static_cast<size_t>(INT_MAX)) return -12;
        fa3_sm90_bf16_to_f16_inplace<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            static_cast<unsigned short*>(o_ptr), output_elements);
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -10;
}

extern "C" {

// FP16 KV path: calls the shared impl with is_fp8=false.
int fa3_sm90_paged_decode(
    void* q_ptr,
    void* k_cache_ptr,
    void* v_cache_ptr,
    void* o_ptr,
    int*  block_tables_ptr,
    int*  context_lens_ptr,
    void* workspace_ptr,
    size_t workspace_bytes,
    float scale,
    int   batch_size,
    int   num_heads,
    int   num_kv_heads,
    int   head_dim,
    int   block_size,
    int   max_blocks_per_seq,
    int   num_blocks_total,
    int   window_size_left,
    cudaStream_t stream
) {
    return fa3_sm90_paged_decode_impl(
        q_ptr, k_cache_ptr, v_cache_ptr, o_ptr,
        block_tables_ptr, context_lens_ptr,
        /*cu_seqlens_q=*/nullptr, /*max_seqlen_q=*/1, /*total_q=*/batch_size,
        workspace_ptr, workspace_bytes,
        scale, batch_size, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks_per_seq, num_blocks_total,
        /*is_fp8=*/false, /*is_causal_prefill=*/false,
        /*q_descale=*/nullptr, /*k_descale=*/nullptr, /*v_descale=*/nullptr,
        window_size_left,
        stream);
}

// FP8 E4M3 KV path. Q / K cache / V cache are FP8 (1 byte/elem).
// q_descale / k_descale / v_descale point at single-scalar f32 scales on device.
// Pinned FA3 writes BF16 for E4M3 input; this ABI converts it to F16 in place.
int fa3_sm90_paged_decode_fp8(
    void* q_fp8_ptr,
    void* k_cache_fp8_ptr,
    void* v_cache_fp8_ptr,
    void* o_f16_ptr,
    int*  block_tables_ptr,
    int*  context_lens_ptr,
    void* workspace_ptr,
    size_t workspace_bytes,
    void* k_scale_cache_ptr,
    void* v_scale_cache_ptr,
    void* q_scale_cache_ptr,
    float* q_descale_ptr,
    float* k_descale_ptr,
    float* v_descale_ptr,
    float scale,
    int   batch_size,
    int   num_heads,
    int   num_kv_heads,
    int   head_dim,
    int   block_size,
    int   max_blocks_per_seq,
    int   num_blocks_total,
    int   window_size_left,
    cudaStream_t stream
) {
    (void)k_scale_cache_ptr;
    (void)v_scale_cache_ptr;
    (void)q_scale_cache_ptr;
    return fa3_sm90_paged_decode_impl(
        q_fp8_ptr, k_cache_fp8_ptr, v_cache_fp8_ptr, o_f16_ptr,
        block_tables_ptr, context_lens_ptr,
        /*cu_seqlens_q=*/nullptr, /*max_seqlen_q=*/1, /*total_q=*/batch_size,
        workspace_ptr, workspace_bytes,
        scale, batch_size, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks_per_seq, num_blocks_total,
        /*is_fp8=*/true, /*is_causal_prefill=*/false,
        q_descale_ptr, k_descale_ptr, v_descale_ptr,
        window_size_left,
        stream);
}

// FP8 E4M3 paged PREFILL: Q / K cache / V cache are FP8. cu_seqlens_q
// gives per-seq offsets in a varlen [total_q, num_heads, head_dim] Q
// tensor; max_seqlen_q is the longest per-seq Q length. Causal
// self-attention (query t only sees K 0..=t within its own seq).
int fa3_sm90_paged_prefill_fp8(
    void* q_fp8_ptr,
    void* k_cache_fp8_ptr,
    void* v_cache_fp8_ptr,
    void* o_f16_ptr,
    int*  block_tables_ptr,
    int*  context_lens_ptr,
    int*  cu_seqlens_q_ptr,
    void* workspace_ptr,
    size_t workspace_bytes,
    void* k_scale_cache_ptr,
    void* v_scale_cache_ptr,
    void* q_scale_cache_ptr,
    float* q_descale_ptr,
    float* k_descale_ptr,
    float* v_descale_ptr,
    float scale,
    int   total_q,
    int   max_seqlen_q,
    int   batch_size,
    int   num_heads,
    int   num_kv_heads,
    int   head_dim,
    int   block_size,
    int   max_blocks_per_seq,
    int   num_blocks_total,
    int   window_size_left,
    cudaStream_t stream
) {
    (void)k_scale_cache_ptr;
    (void)v_scale_cache_ptr;
    (void)q_scale_cache_ptr;
    return fa3_sm90_paged_decode_impl(
        q_fp8_ptr, k_cache_fp8_ptr, v_cache_fp8_ptr, o_f16_ptr,
        block_tables_ptr, context_lens_ptr,
        cu_seqlens_q_ptr, max_seqlen_q, total_q,
        workspace_ptr, workspace_bytes,
        scale, batch_size, num_heads, num_kv_heads, head_dim,
        block_size, max_blocks_per_seq, num_blocks_total,
        /*is_fp8=*/true, /*is_causal_prefill=*/true,
        q_descale_ptr, k_descale_ptr, v_descale_ptr,
        window_size_left,
        stream);
}

}  // extern "C"
