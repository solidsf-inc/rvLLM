// Derived from NVIDIA CUTLASS 4.4.2 commit
// da5e086dab31d63815acafdac9a9c5893b1c69e2,
// examples/55_hopper_mixed_dtype_gemm (BSD-3-Clause).
//
// Exposes one entry point: rvllm_w4a8_gemm_run(...) that runs
//   D (f16)  =  alpha * A_fp8 * B_int4_quant  +  beta * C (f16)
// where B is int4 with per-group FP8 scales (group_size = 128) packed as
// cutlass::Array<e4m3, 8> (LUT trick from example 55).
//
// A is RowMajor [M, K] FP8 E4M3.
// B is ColMajor [K, N] int4 in the LayoutB_Reordered physical layout below.
// Scales are MN-major [N, K/group_size] as packed e4m3x8 LUT blocks.
// D is RowMajor [M, N] fp16.
//
// We swap A <-> B and transpose inside CUTLASS to keep the narrow type in
// registers AND use TMA epilogues (example 55's approach). Caller still
// sees the logical A*B^T semantics.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/mixed_dtype_utils.hpp"  // compute_memory_reordering_atom

using namespace cute;

#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
#error "SM90 MMA required"
#endif

// =========================================================================
// Types (match example 55 int4_fp8 config)
// =========================================================================
using MmaType    = cutlass::float_e4m3_t;
using QuantType  = cutlass::int4b_t;

constexpr int kGroupSize = 128;

// Tile: 128 M x 128 N x 128 K (one WGMMA-friendly tile)
constexpr int TileShapeK = 128 * 8 / cutlass::sizeof_bits<MmaType>::value; // = 128 for FP8

using ElementA       = MmaType;
using LayoutA        = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB       = QuantType;
using LayoutB        = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using LayoutA_T = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_T = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;

using LayoutAtomQuant    = decltype(cutlass::compute_memory_reordering_atom<MmaType>());
using LayoutB_Reordered  = decltype(cute::tile_to_shape(LayoutAtomQuant{}, Layout<Shape<int,int,int>, StrideB>{}));

using ElementScale = MmaType; // scales are FP8 E4M3 (example 55 convention)
using LayoutScale  = cutlass::layout::RowMajor;

using ElementC       = cutlass::half_t;
using LayoutC        = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD       = ElementC;
using LayoutD        = LayoutC;
constexpr int AlignmentD = AlignmentC;

using ElementAccumulator = float;
using ElementCompute     = float;
using ArchTag            = cutlass::arch::Sm90;
using OperatorClass      = cutlass::arch::OpClassTensorOp;

using TileShape      = Shape<_128, _128, cute::Int<TileShapeK>>;
using ClusterShape   = Shape<_1, _1, _1>;
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSched  = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpilogueTileT  = cutlass::epilogue::collective::EpilogueTileAuto;

// Epilogue: linear-combo D = alpha * accum + beta * C (transposed layout
// for the explicit-swap convention).
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileT,
    ElementAccumulator, ElementAccumulator,
    ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type, AlignmentC,
    ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD,
    EpilogueSched
>::CollectiveOp;

// Mainloop: INT4 (B, swapped to operand A internally) with packed e4m3 LUT
// scales (Array<e4m3, 8>). B is shuffle-reordered for contiguous thread reads.
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementB, cutlass::Array<ElementScale, 8>>, LayoutB_Reordered, AlignmentB,
    ElementA, LayoutA_T, AlignmentA,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// =========================================================================
// C ABI entry point
// =========================================================================
extern "C" int rvllm_w4a8_gemm_run(
    const void* a_fp8,              // [M, K] RowMajor E4M3
    const void* b_int4_reordered,   // [K, N] INT4 ColMajor, AWQ-shuffled offline
    const void* b_scales_packed,    // [N, K/group_size] as Array<e4m3, 8> LUT blocks
    const void* c_f16,              // [M, N] RowMajor (may be nullptr; used only if beta != 0)
    void*       d_f16,              // [M, N] RowMajor output
    int         m,
    int         n,
    int         k,
    int         group_size,         // must equal kGroupSize (128)
    float       alpha,
    float       beta,
    void*       workspace,
    size_t      workspace_bytes,
    cudaStream_t stream
) {
    if (!a_fp8 || !b_int4_reordered || !b_scales_packed || !d_f16 ||
        m <= 0 || n <= 0 || k <= 0 || !std::isfinite(alpha) || !std::isfinite(beta)) return -3;
    if (beta != 0.0f && !c_f16) return -4;
    if (group_size != kGroupSize) {
        fprintf(stderr, "rvllm_w4a8: group_size must be %d, got %d\n", kGroupSize, group_size);
        return -1;
    }
    if (k % kGroupSize != 0) {
        fprintf(stderr, "rvllm_w4a8: K (%d) must be divisible by group_size (%d)\n", k, kGroupSize);
        return -2;
    }

    const int scale_k = k / kGroupSize;

    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using StrideS = typename CollectiveMainloop::StrideScale;

    // Explicit swap+transpose: CUTLASS call is (B, A) -> D^T.
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(n, m, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, 1));
    auto stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, scale_k, 1));

    // Reordered B layout — follow example 55 line 399: shape-only overload.
    auto shape_B = cute::make_shape(n, k, 1);
    LayoutB_Reordered layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);

    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        { n, m, k, /*batch*/ 1 },
        {
            reinterpret_cast<const ElementB*>(b_int4_reordered), layout_B_reordered,
            reinterpret_cast<const ElementA*>(a_fp8), stride_A,
            reinterpret_cast<const cutlass::Array<ElementScale, 8>*>(b_scales_packed), stride_S,
            kGroupSize,
        },
        {
            { alpha, beta },
            reinterpret_cast<const ElementC*>(c_f16), stride_C,
            reinterpret_cast<ElementD*>(d_f16), stride_D,
        }
    };

    Gemm gemm;
    cutlass::Status s = gemm.can_implement(arguments);
    if (s != cutlass::Status::kSuccess) {
        fprintf(stderr, "rvllm_w4a8: can_implement failed: %d\n", (int)s);
        return -10;
    }
    size_t need = Gemm::get_workspace_size(arguments);
    if (need > workspace_bytes || (need && !workspace)) {
        fprintf(stderr, "rvllm_w4a8: workspace too small: need %zu, got %zu\n", need, workspace_bytes);
        return -11;
    }
    s = gemm.initialize(arguments, workspace, stream);
    if (s != cutlass::Status::kSuccess) {
        fprintf(stderr, "rvllm_w4a8: initialize failed: %d\n", (int)s);
        return -12;
    }
    s = gemm.run(stream);
    if (s != cutlass::Status::kSuccess) {
        fprintf(stderr, "rvllm_w4a8: run failed: %d\n", (int)s);
        return -13;
    }
    return 0;
}

// Workspace size probe.
extern "C" size_t rvllm_w4a8_gemm_workspace_size(int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0 || (k % kGroupSize) != 0) return 0;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using StrideS = typename CollectiveMainloop::StrideScale;
    const int scale_k = k / kGroupSize;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(n, m, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, 1));
    auto stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, scale_k, 1));

    auto shape_B = cute::make_shape(n, k, 1);
    LayoutB_Reordered layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);

    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        { n, m, k, 1 },
        {
            nullptr, layout_B_reordered,
            nullptr, stride_A,
            nullptr, stride_S,
            kGroupSize,
        },
        {
            { 1.0f, 0.0f },
            nullptr, stride_C,
            nullptr, stride_D,
        }
    };
    return Gemm::get_workspace_size(arguments);
}

// =========================================================================
// Weight encoder: FP16 weights -> (reordered INT4 + LUT-packed FP8 scales).
//
// Simple symmetric per-group (g=128) quantization. For each [N, group] block
// of K-contiguous weights:
//    scale_fp32 = max(|w|) / 7
//    w_int4     = round(w / scale_fp32)  clamped to [-8, 7]
// Then the INT4 positive encoding is "unified" with the negative encoding
// (example 55 convention) and the scale is packed as Array<e4m3, 8> holding
//    {scale * -8, scale * -7, ..., scale * -1}
// Finally the INT4 tensor is memory-reordered via the LayoutAtomQuant atom
// so each thread reads 8 contiguous elements in one load.
//
// This is symmetric group quantization. Its exact nibble and scale layouts are
// part of the public ABI and must be parity-tested against a scalar decoder.
// =========================================================================

#include "cutlass/util/device_memory.h"
#include <cuda_fp16.h>

// Quantize FP16 weights to INT4 with per-group FP32 scales, both on device.
// Writes unified-encoded INT4 (positive encoding == negative encoding except
// sign bit) into w_int4_raw and per-group f32 scales into scales_f32.
static __global__ void quantize_sym_group_kernel(
    const __half* __restrict__ w_fp16,  // [N, K] row-major (N rows)
    int* __restrict__ w_int4_raw,        // packed int4: [N, K/8] (8 per int32)
    float* __restrict__ scales_f32,       // [N, K/group]
    int n, int k, int group
) {
    int row = blockIdx.y;
    int grp = blockIdx.x;
    int tid = threadIdx.x;  // 0..31
    if (row >= n || grp * group >= k) return;

    const __half* w_row = w_fp16 + (size_t)row * k + grp * group;

    // Pass 1: max |w| in group, reduce across the 32-thread warp.
    float local_max = 0.0f;
    for (int i = tid; i < group; i += 32) {
        float v = __half2float(w_row[i]);
        local_max = fmaxf(local_max, fabsf(v));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));

    float scale = local_max / 7.0f;
    if (scale == 0.0f) scale = 1e-9f;  // avoid div0
    if (tid == 0) scales_f32[row * (k / group) + grp] = scale;

    float inv_scale = 1.0f / scale;

    // Pass 2: quantize + pack 8 int4 into one int32.
    // Each int4 value is 4 bits; we write 8-wide chunks.
    // Encode the nibble convention used by CUTLASS example 55.
    int* row_int4 = w_int4_raw + row * (k / 8);
    for (int i = tid; i < group; i += 32) {
        float v = __half2float(w_row[i]) * inv_scale;
        int q = __float2int_rn(v);
        if (q > 7) q = 7;
        if (q < -8) q = -8;
        // CUTLASS "unified" int4 encoding (mixed_dtype_utils.hpp
        // unified_encode_int4b): the kernel's PRMT converter + the TC LUT
        // {-8s..-1s} expect POSITIVE q in [1,7] stored as bit pattern (8-q),
        // while 0 and negative q (two's complement) are stored unchanged. The
        // plain `q & 0xF` we stored before scrambled every positive weight.
        unsigned int nib = (q > 0) ? (unsigned int)(8 - q) : (unsigned int)(q & 0xF);

        int word_idx = (grp * group + i) / 8;
        int slot_idx = (grp * group + i) % 8;
        atomicOr(&row_int4[word_idx], (int)(nib << (slot_idx * 4)));
    }
}

// Build packed LUT scales (Array<e4m3, 8>) from per-group f32 scales.
// For each group, the LUT contains {scale * -8, scale * -7, ..., scale * -1}
// stored as 8 packed e4m3 values (8 bytes total per group).
static __global__ void build_packed_scales_kernel(
    const float* __restrict__ scales_f32,  // [N, scale_k]
    __nv_fp8_storage_t* __restrict__ scales_packed,  // [N, scale_k, 8] e4m3
    int n, int scale_k
) {
    int row = blockIdx.y;
    int grp = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= n || grp >= scale_k || tid >= 8) return;

    float s = scales_f32[row * scale_k + grp];
    // i in [0..7] maps to lut value (i - 8) * s
    float lut_val = (float)((int)tid - 8) * s;
    __nv_fp8_storage_t fp8 = __nv_cvt_float_to_fp8(lut_val, __NV_SATFINITE, __NV_E4M3);
    scales_packed[(row * scale_k + grp) * 8 + tid] = fp8;
}

// Host entry: quantize + pack weights + reorder into the kernel-expected
// layout. Expects:
//   w_fp16        [N, K] row-major device ptr (input; will be read)
//   w_int4_out    [N, K/2] bytes device ptr (output; zeroed by caller)
//   scales_out    [N, K/group, 8] bytes device ptr (output; e4m3 LUT)
//   workspace     temporary f32 scales buffer, >= N*K/group*4 bytes
//   shuffle       if 1, apply CUTLASS memory-reordering atom; else leave raw
extern "C" int rvllm_w4a8_encode_weight_fp16(
    const void* w_fp16,
    int         n,
    int         k,
    int         group_size,
    void*       w_int4_out,
    void*       scales_packed_out,
    void*       scales_f32_workspace,
    int         shuffle,
    cudaStream_t stream
) {
    if (!w_fp16 || !w_int4_out || !scales_packed_out || !scales_f32_workspace || n <= 0 || k <= 0) return -4;
    if (group_size != kGroupSize) return -1;
    if (k % group_size != 0)      return -2;
    if (k % 8 != 0)               return -3;
    if (shuffle != 1) return -5;
    const int scale_k = k / group_size;

    // 1) Quantize + build f32 scales.
    cudaError_t error = cudaMemsetAsync(w_int4_out, 0, (size_t)n * (k / 2), stream);
    if (error != cudaSuccess) return -10;
    dim3 grid_q(scale_k, n, 1);
    dim3 block_q(32, 1, 1);
    quantize_sym_group_kernel<<<grid_q, block_q, 0, stream>>>(
        (const __half*)w_fp16,
        (int*)w_int4_out,
        (float*)scales_f32_workspace,
        n, k, group_size
    );
    if (cudaPeekAtLastError() != cudaSuccess) return -11;

    // 2) Build packed e4m3 LUT scales.
    dim3 grid_s(scale_k, n, 1);
    dim3 block_s(8, 1, 1);
    build_packed_scales_kernel<<<grid_s, block_s, 0, stream>>>(
        (const float*)scales_f32_workspace,
        (__nv_fp8_storage_t*)scales_packed_out,
        n, scale_k
    );
    if (cudaPeekAtLastError() != cudaSuccess) return -12;

    // 3) Reorder B into LayoutB_Reordered — the GEMM ALWAYS reads B via
    // tile_to_shape(LayoutAtomQuant{}, shape_B) (the reordering atom), so the
    // weights must be physically reordered to match (example 55 line 400).
    // quantize_sym_group_kernel wrote w_int4_out in the natural col-major-K
    // layout (int4b_t [N,K] row-major == [K,N] col-major); reorder it in place.
    {
        auto shape_B = cute::make_shape(n, k, 1);
        auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
        auto layout_B = cute::make_layout(shape_B, stride_B);
        LayoutB_Reordered layout_B_reordered =
            cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
        cutlass::reorder_tensor(
            reinterpret_cast<ElementB*>(w_int4_out), layout_B, layout_B_reordered);
    }
    if (cudaPeekAtLastError() != cudaSuccess) return -13;

    return 0;
}
