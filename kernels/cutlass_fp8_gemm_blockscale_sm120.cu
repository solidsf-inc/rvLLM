// CUTLASS 4.4.2 Blackwell-Geforce (SM120) FP8 GEMM with blockwise weight scale
// and per-token activation scale.
//
//     D_f16[m, n] = sum_k (A_fp8[m, k] * B_fp8[n, k])
//                    * a_scale[m]                       <- per-token (M-vector)
//                    * b_block_scale[n / 128, k / 128]  <- 128×128 weight block-scale
//                    -> f16
//
// Derived from NVIDIA CUTLASS 4.4.2 commit
// da5e086dab31d63815acafdac9a9c5893b1c69e2,
// examples/87_blackwell_geforce_gemm_blockwise/87a (BSD-3-Clause), adapted to:
//   * f16 output (ElementC = half_t, not bfloat16_t)
//   * per-M a_scale vector fused into epilogue (87a has a simple linear
//     combination epilogue with alpha/beta; we replace it with a
//     ColBroadcast fusion EVT so the per-token scale lands in the
//     fused epilogue rather than a separate post-pass).
//   * an extern-C entry point for row activation scales and 128x128 weight
//     block scales. It is intentionally not ABI-compatible with a per-column
//     scale GEMM despite the similar operation name.
//
// A layout:  RowMajor   [M, K]
// B layout:  ColumnMajor[N, K]  (stored RowMajor by our loader but
//                                read as ColumnMajor here because
//                                CUTLASS expects the MMA B operand
//                                K-major, which is our RowMajor.
//                                CUTLASS's "ColumnMajor B" means the
//                                logical [K, N] operand is packed
//                                with K as fastest dim, which matches
//                                a RowMajor [N, K] byte stream.)

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>
#include <climits>
#include <cstdint>

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

using namespace cute;

using ElementA           = cutlass::float_e4m3_t;
using ElementB           = cutlass::float_e4m3_t;
using ElementD           = cutlass::half_t;            // f16 output
using ElementAccum       = float;
using ElementCompute     = float;
using ElementScalar      = float;

using LayoutA            = cutlass::layout::RowMajor;
using LayoutB            = cutlass::layout::ColumnMajor;
using LayoutD            = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8

// Tile shape can be overridden at build time via -D{TILE_M,TILE_N,TILE_K}=N.
// Must stay a multiple of 128 per dim to line up with the blockwise scale
// granularity (sm120_trivial_blockwise_scale_config asserts this).
#ifndef TILE_M
#define TILE_M 128
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 128
#endif
using MmaTileShape_MNK = Shape<cute::Int<TILE_M>, cute::Int<TILE_N>, cute::Int<TILE_K>>;
using ClusterShape_MNK = Shape<_1, _1, _1>;        // SM120 does not support cluster multicast

// Explicit scale-vector granularity:
//   SFVecSizeM = 1    → per-row activation scale (per-token)
//   SFVecSizeN = 128  → per-128-channel weight scale (N-block)
//   SFVecSizeK = 128  → per-128-channel K-block scale
// This explicit configuration is required: the convenience helper uses
// tile-granular SFA, while rvLLM's contract is one activation scale per row
// and K block.
using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128>;
using LayoutSFA   = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB   = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    void, LayoutD, AlignmentD,           // no C
    ElementD, LayoutD, AlignmentD,        // D
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccum,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void   // default CLC tile scheduler
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideD = typename Gemm::GemmKernel::StrideD;

static inline int ceil_div_positive(int value, int divisor) {
    return 1 + (value - 1) / divisor;
}

extern "C" {

/// Launch the SM120 blockwise FP8 GEMM.
///
/// `a_scale`  : `[M, ceil(K/128)]` f32 in CUTLASS MN-major SFA layout.
/// `b_scale`  : `[ceil(N/128), ceil(K/128)]` f32 in CUTLASS MN-major SFB layout.
///
/// Returns 0 on success, negative for CUTLASS / launch failures.
int cutlass_fp8_gemm_blockscale_sm120(
    void* output,              // [M, N] f16
    const void* a,             // [M, K] fp8_e4m3
    const void* b,             // [N, K] fp8_e4m3, interpreted as ColumnMajor K-major
    const void* a_scale,       // [M, K/128] f32 (SFA)
    const void* b_scale,       // [N/128, K/128] f32 (SFB)
    int m, int n, int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
) {
    if (!output || !a || !b || !a_scale || !b_scale || m <= 0 || n <= 0 || k <= 0) return -10;
    if ((k % 128) != 0 || (n % 128) != 0) return -11;
    if ((reinterpret_cast<uintptr_t>(a) % AlignmentA) != 0 ||
        (reinterpret_cast<uintptr_t>(b) % AlignmentB) != 0 ||
        (reinterpret_cast<uintptr_t>(output) % (AlignmentD * sizeof(ElementD))) != 0) return -12;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

    auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {
            reinterpret_cast<const ElementA*>(a), stride_A,
            reinterpret_cast<const ElementB*>(b), stride_B,
            reinterpret_cast<const ElementAccum*>(a_scale), layout_SFA,
            reinterpret_cast<const ElementAccum*>(b_scale), layout_SFB,
        },
        {
            {},                          // epilogue.thread (alpha/beta) — defaults to alpha=1, beta=0
            nullptr, stride_D,           // no C
            reinterpret_cast<ElementD*>(output), stride_D,
        }
    };
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta  = 0.0f;

    Gemm gemm_op;
    size_t required_workspace = gemm_op.get_workspace_size(args);
    if (required_workspace > workspace_size || (required_workspace && !workspace)) return -13;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -3;

    return 0;
}

/// SFA is contiguous in row with K block as the outer stride:
/// `sfa[row + k_block * M]`.
///
/// SFB physical layout (majorSFB=MN): `sfb[n_tile + k_block * ceil(N/128)]`.
/// The input weight-scale layout is row-major `[N/128, K/128]`; the
/// preparation kernel transposes it into CUTLASS SFB layout.
///
/// Bytes needed by each tensor at a given problem shape.
size_t cutlass_fp8_gemm_blockscale_sm120_sfa_bytes(int m, int k) {
    if (m <= 0 || k <= 0) return 0;
    // SFVecSizeM = 1 → one SFA entry per (row, k_block).
    int kb = ceil_div_positive(k, 128);
    return (size_t)m * (size_t)kb * sizeof(float);
}

size_t cutlass_fp8_gemm_blockscale_sm120_sfb_bytes(int n, int k) {
    if (n <= 0 || k <= 0) return 0;
    int nb = ceil_div_positive(n, 128);
    int kb = ceil_div_positive(k, 128);
    return (size_t)nb * (size_t)kb * sizeof(float);
}

} // extern "C"

// Replicate each per-row activation scale over the K blocks in SFA layout.
__global__ void fill_sfa_from_a_scale_sm120(
    const float* __restrict__ a_scale,   // [M]
    float*       __restrict__ sfa,       // [m * k_blocks], CUTLASS MN-major
    int m,
    int /*m_blocks_unused*/,
    int k_blocks
) {
    int row     = blockIdx.y * blockDim.x + threadIdx.x;
    int k_block = blockIdx.x;
    if (row >= m || k_block >= k_blocks) return;
    sfa[row + k_block * m] = a_scale[row];
}

// SFB transpose kernel — read row-major b_chscale[n_tile, k_block] and
// store at CUTLASS SFB[n_tile + k_block * n_blocks]. Pure per-element
// transpose, no reduction.
__global__ void fill_sfb_from_b_chscale_sm120(
    const float* __restrict__ b_chscale, // row-major [n_blocks, k_blocks]
    float*       __restrict__ sfb,       // [n_blocks * k_blocks], CUTLASS layout
    int n_blocks,
    int k_blocks
) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_blocks * k_blocks;
    if (idx >= total) return;
    // b_chscale is row-major [n_blocks, k_blocks]; SFB is CUTLASS
    // MN-major [n_blocks, k_blocks] — transpose.
    int n_tile  = idx / k_blocks;
    int k_block = idx - n_tile * k_blocks;
    sfb[n_tile + k_block * n_blocks] = b_chscale[n_tile * k_blocks + k_block];
}

extern "C" {

int cutlass_fp8_gemm_blockscale_sm120_prep_sfa(
    const void* a_scale,
    void*       sfa,
    int m, int k,
    cudaStream_t stream
) {
    if (!a_scale || !sfa || m <= 0 || k <= 0) return -2;
    if ((k % 128) != 0) return -3;
    if ((reinterpret_cast<uintptr_t>(a_scale) % alignof(float)) != 0 ||
        (reinterpret_cast<uintptr_t>(sfa) % alignof(float)) != 0) return -4;
    // Per-row SFA: grid = (k_blocks, m_tiles_of_128_threads).
    int kb = k / 128;
    int threads = 128;
    int m_tiles = ceil_div_positive(m, threads);
    if (m_tiles > 65535) return -5;
    dim3 grid(kb, m_tiles);
    dim3 block(threads);
    fill_sfa_from_a_scale_sm120<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(a_scale),
        reinterpret_cast<float*>(sfa),
        m, m_tiles, kb
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cutlass_fp8_gemm_blockscale_sm120_prep_sfb(
    const void* b_chscale,
    void*       sfb,
    int n, int k,
    cudaStream_t stream
) {
    if (!b_chscale || !sfb || n <= 0 || k <= 0) return -2;
    if ((n % 128) != 0 || (k % 128) != 0) return -3;
    if ((reinterpret_cast<uintptr_t>(b_chscale) % alignof(float)) != 0 ||
        (reinterpret_cast<uintptr_t>(sfb) % alignof(float)) != 0) return -4;
    int nb = n / 128;
    int kb = k / 128;
    int64_t total_wide = static_cast<int64_t>(nb) * static_cast<int64_t>(kb);
    if (total_wide <= 0 || total_wide > INT_MAX) return -5;
    int total = static_cast<int>(total_wide);
    int bs = 256;
    int gs = ceil_div_positive(total, bs);
    fill_sfb_from_b_chscale_sm120<<<gs, bs, 0, stream>>>(
        reinterpret_cast<const float*>(b_chscale),
        reinterpret_cast<float*>(sfb),
        nb, kb
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

/// Query the workspace size required for a given problem shape.
size_t cutlass_fp8_gemm_blockscale_sm120_workspace(int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0 || (n % 128) != 0 || (k % 128) != 0) return 0;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
    auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {nullptr, stride_A, nullptr, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
        {{}, nullptr, stride_D, nullptr, stride_D}
    };

    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

} // extern "C"

#else

// sm < 120: the symbols are not built. Operators shouldn't link this
// object on pre-Blackwell-Geforce targets — the build script only
// compiles it under `-arch=sm_120a` / `-arch=sm_121a`.

extern "C" {

int cutlass_fp8_gemm_blockscale_sm120(
    void*, const void*, const void*, const void*, const void*,
    int, int, int, void*, size_t, void*
) {
    return -100;  // unsupported arch
}

size_t cutlass_fp8_gemm_blockscale_sm120_workspace(int, int, int) {
    return 0;
}

size_t cutlass_fp8_gemm_blockscale_sm120_sfa_bytes(int, int) { return 0; }
size_t cutlass_fp8_gemm_blockscale_sm120_sfb_bytes(int, int) { return 0; }

int cutlass_fp8_gemm_blockscale_sm120_prep_sfa(
    const void*, void*, int, int, void*
) { return -100; }

int cutlass_fp8_gemm_blockscale_sm120_prep_sfb(
    const void*, void*, int, int, void*
) { return -100; }

} // extern "C"

#endif // CUTLASS_ARCH_MMA_SM120_SUPPORTED || SM121_SUPPORTED
