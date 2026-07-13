// CUTLASS 3.x SM90 FP8 GEMM with per-row + per-column scale epilogue.
//
// D_f16[m,n] = cast_f16_sat(acc_f32[m,n] * row_scale[m] * col_scale[n])
//
// row_scale = per-token activation scale (from fused_rmsnorm_fp8_quant)
// col_scale = per-channel weight scale (channelscale from FP8-Dynamic checkpoint)
//
// This replaces the 3-kernel chain: fp8_gemm_f32 + scale_cols_f32 + f32_to_f16_sat

#include <cutlass/cutlass.h>
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

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementD = cutlass::half_t;
using ElementAccum = float;
using ElementCompute = float;
using ElementScalar = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

// EVT epilogue: D[m,n] = cast<f16>(row_scale[m] * col_scale[n] * Acc[m,n])
//
// Tree: multiply(ColBroadcast(act_scale[m]), multiply(RowBroadcast(ch_scale[n]), AccFetch))
// CUTLASS naming: RowBroadcast = per-N (stride<0,1,0>), ColBroadcast = per-M (stride<1,0,0>)

// Inner: channelscale[n] * Acc (RowBroadcast = per-N)
using ChScaledAcc = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90RowBroadcast<
        0, TileShape, ElementScalar>,
    cutlass::epilogue::fusion::Sm90AccFetch
>;

// Outer: act_scale[m] * (channelscale[n] * Acc) (ColBroadcast = per-M)
using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementD, ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90ColBroadcast<
        0, TileShape, ElementScalar>,
    ChScaledAcc
>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    void, LayoutD, 8,
    ElementD, LayoutD, 8,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    EpilogueEVT
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 16,
    ElementB, LayoutB, 16,
    ElementAccum,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

extern "C" {

int cutlass_fp8_gemm_channelscale(
    void* output,             // [M, N] f16
    const void* a,            // [M, K] fp8_e4m3
    const void* b,            // [N, K] fp8_e4m3
    const void* row_scale,    // [M] f32 per-token activation scale
    const void* col_scale,    // [N] f32 per-channel weight scale
    int M, int N, int K,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
) {
    auto prob_shape = cute::make_shape(M, N, K, 1);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
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
            {},  // thread args -- filled below
            nullptr, {}, // no C
            reinterpret_cast<ElementD*>(output), stride_D,
        }
    };

    // EVT thread args:
    // Tree: multiply(RowBroadcast(row_scale), multiply(ColBroadcast(col_scale), AccFetch))
    args.epilogue.thread = {
        // ColBroadcast(act_scale) -- per-M (per-token activation scale)
        {reinterpret_cast<const ElementScalar*>(row_scale), ElementScalar(1), {}},
        // multiply(RowBroadcast(ch_scale), AccFetch)
        {
            // RowBroadcast -- per-N (per-channel weight scale)
            {reinterpret_cast<const ElementScalar*>(col_scale), ElementScalar(1), {}},
            // AccFetch
            {},
            // multiply op
            {}
        },
        // outer multiply op
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

size_t cutlass_fp8_gemm_channelscale_workspace(int M, int N, int K) {
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{}, nullptr, {}, nullptr, stride_D}
    };

    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

} // extern "C"
