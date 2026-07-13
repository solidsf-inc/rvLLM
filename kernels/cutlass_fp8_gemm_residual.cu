// CUTLASS 3.x SM90 FP8 GEMM + residual add in epilogue.
//
// D[m,n] = cast<f16>(a_scales[m] * b_scale * sum_k(A_fp8[m,k] * B_fp8[k,n]) + residual[m,n])
//
// Fuses the residual add into the GEMM epilogue via EVT, eliminating a
// separate HBM round-trip for the intermediate GEMM output buffer.
//
// Used for:
//   - O-proj: D = FP8_GEMM(attn_out, W_oproj) + residual
//   - Down-proj: D = FP8_GEMM(silu_out, W_down) + residual
//
// A=[M,K] RowMajor FP8 E4M3, B=[N,K] RowMajor (ColumnMajor to CUTLASS) FP8 E4M3
// C=[M,N] RowMajor F16 (residual), D=[M,N] RowMajor F16 (output = scaled_gemm + residual)

#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp>
#include <cute/tensor.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cuda_fp16.h>

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::half_t;       // residual input
using ElementD = cutlass::half_t;       // output (scaled GEMM + residual)
using ElementAccum = float;
using ElementCompute = float;
using ElementScalar = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

// ---------------------------------------------------------------------------
// EVT epilogue: D[m,n] = cast<f16>(row_scale[m] * col_scale * Acc[m,n] + residual[m,n])
//
// Tree:
//   Compute<plus, f16, f32>           -- add scaled acc + residual, cast to f16
//   ├── Compute<multiply, f32, f32>   -- row_scale * (col_scale * acc)
//   │   ├── RowBroadcast(row_scale)
//   │   └── Compute<multiply, f32, f32>  -- col_scale * acc
//   │       ├── ScalarBroadcast(col_scale)
//   │       └── AccFetch
//   └── SrcFetch                      -- residual (C tensor)
// ---------------------------------------------------------------------------

template<typename TileShape_>
using ScaledAccT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>,
    cutlass::epilogue::fusion::Sm90AccFetch
>;

template<typename TileShape_>
using RowScaledAccT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90RowBroadcast<
        0, TileShape_, ElementScalar>,
    ScaledAccT<TileShape_>
>;

template<typename TileShape_>
using ResidualEVT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::plus, ElementD, ElementCompute, RoundStyle>,
    RowScaledAccT<TileShape_>,
    cutlass::epilogue::fusion::Sm90SrcFetch<ElementC>
>;

// ---------------------------------------------------------------------------
// Template: FP8 GEMM + residual with fused scaling
// ---------------------------------------------------------------------------

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
struct Fp8GemmResidualType {
    using EVT = ResidualEVT<TileShape_>;

    using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum, ElementCompute,
        ElementC, LayoutC, 8,       // C = residual (f16)
        ElementD, LayoutD, 8,
        cutlass::epilogue::TmaWarpSpecializedCooperative,
        EVT
    >::CollectiveOp;

    using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, 16,
        ElementB, LayoutB, 16,
        ElementAccum,
        TileShape_,
        ClusterShape_,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename EpilogueOp::SharedStorage))>,
        KernelSchedule_
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        MainloopOp,
        EpilogueOp
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// ---------------------------------------------------------------------------
// Dispatch template
// ---------------------------------------------------------------------------

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
int fp8_gemm_residual_dispatch(
    void* output, const void* a, const void* b,
    const void* a_scales, const void* b_scale,
    const void* residual,
    int M, int N, int K,
    void* workspace, size_t workspace_size,
    cudaStream_t stream)
{
    using G = Fp8GemmResidualType<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename G::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {
            reinterpret_cast<const ElementA*>(a), stride_A,
            reinterpret_cast<const ElementB*>(b), stride_B,
        },
        {
            {},
            reinterpret_cast<const ElementC*>(residual), stride_C,
            reinterpret_cast<ElementD*>(output), stride_D,
        }
    };

    // EVT thread params: same structure as the non-residual version
    // Tree: plus(multiply(RowBroadcast, multiply(ScalarBroadcast, AccFetch)), SrcFetch)
    args.epilogue.thread = {
        // RowScaledAccT params:
        {
            // RowBroadcast (row_scale)
            {reinterpret_cast<const ElementScalar*>(a_scales), ElementScalar(0), {}},
            // ScaledAccT params:
            {
                // ScalarBroadcast (col_scale)
                {{ElementScalar(0)}, {reinterpret_cast<const ElementScalar*>(b_scale)}, {}},
                // AccFetch (no params)
                {},
                // Compute<multiply> (no params)
                {}
            },
            // Compute<multiply> (no params)
            {}
        },
        // SrcFetch (no params)
        {},
        // Compute<plus> (no params)
        {}
    };

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -3;

    return 0;
}

// ---------------------------------------------------------------------------
// Workspace-size template
// ---------------------------------------------------------------------------

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
size_t fp8_gemm_residual_ws_dispatch(int M, int N, int K)
{
    using G = Fp8GemmResidualType<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename G::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{}, nullptr, stride_C, nullptr, stride_D}
    };

    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

// ---------------------------------------------------------------------------
// Schedule aliases
// ---------------------------------------------------------------------------

using WS      = cutlass::gemm::KernelTmaWarpSpecialized;
using Coop    = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using FP8WS   = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
using FP8Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
using FP8PP   = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;

// ---------------------------------------------------------------------------
// Macro for entry points
// ---------------------------------------------------------------------------

#define FP8_GEMM_RESIDUAL_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SCHED) \
extern "C" int cutlass_fp8_gemm_residual_v##ID(                                         \
    void* o, const void* a, const void* b,                                               \
    const void* a_scales, const void* b_scale,                                           \
    const void* residual,                                                                \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {                       \
    return fp8_gemm_residual_dispatch<                                                   \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                         \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED>(                                      \
            o, a, b, a_scales, b_scale, residual, M, N, K, ws, ws_sz, s);              \
}                                                                                        \
extern "C" size_t cutlass_fp8_gemm_residual_v##ID##_workspace_size(int M, int N, int K) {\
    return fp8_gemm_residual_ws_dispatch<                                                \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                         \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED>(M, N, K);                            \
}

// ---------------------------------------------------------------------------
// Variants: baseline (v0-v3) + FP8FastAccum (v4-v9)
// ---------------------------------------------------------------------------

// Baseline schedules
FP8_GEMM_RESIDUAL_VARIANT(0,  64, 128, 128, 1,1,1, WS)
FP8_GEMM_RESIDUAL_VARIANT(1, 128, 128, 128, 1,1,1, Coop)
FP8_GEMM_RESIDUAL_VARIANT(2, 128, 128, 128, 1,1,1, WS)
FP8_GEMM_RESIDUAL_VARIANT(3, 128, 256, 128, 1,1,1, Coop)

// FP8FastAccum -- decode (M<=64)
FP8_GEMM_RESIDUAL_VARIANT(4,  64, 128, 128, 1,1,1, FP8WS)
FP8_GEMM_RESIDUAL_VARIANT(5,  64, 128, 256, 1,1,1, FP8WS)    // deep K for down-proj
FP8_GEMM_RESIDUAL_VARIANT(6,  64, 256, 128, 1,1,1, FP8WS)

// FP8FastAccum -- medium/large batch
FP8_GEMM_RESIDUAL_VARIANT(7, 128, 128, 128, 1,1,1, FP8Coop)
FP8_GEMM_RESIDUAL_VARIANT(8, 128, 128, 256, 1,1,1, FP8Coop)  // deep K
FP8_GEMM_RESIDUAL_VARIANT(9, 128, 256, 128, 1,1,1, FP8Coop)
