// CUTLASS 3.x SM90 fused O-projection GEMM + residual add -- autotuned variants.
//
// D[M,N] = A[M,K] @ B[K,N]^T + residual[M,N]
// (alpha=1, beta=1, C=residual)
//
// 31 tile/cluster/schedule variants for runtime autotuning.

#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cute/tensor.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cuda_fp16.h>

using namespace cute;

using ElementA     = cutlass::half_t;
using ElementB     = cutlass::half_t;
using ElementC     = cutlass::half_t;  // residual
using ElementD     = cutlass::half_t;  // output
using ElementAccum = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int Alignment = 8;

// ---------------------------------------------------------------------------
// Template: build Gemm type from tile/cluster/schedule
// ---------------------------------------------------------------------------

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
struct OprojResidualGemm {
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum, ElementAccum,
        ElementC, LayoutC, Alignment,
        ElementD, LayoutD, Alignment,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, Alignment,
        ElementB, LayoutB, Alignment,
        ElementAccum,
        TileShape_, ClusterShape_,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule_
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
static int oproj_residual_dispatch(
    void* output, const void* input, const void* weight, const void* residual,
    int M, int N, int K,
    void* workspace, size_t workspace_size, cudaStream_t stream)
{
    using G = OprojResidualGemm<TileShape_, ClusterShape_, KernelSchedule_>;
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
            reinterpret_cast<const ElementA*>(input),  stride_A,
            reinterpret_cast<const ElementB*>(weight), stride_B,
        },
        {
            {ElementAccum(1.0f), ElementAccum(1.0f)},
            reinterpret_cast<const ElementC*>(residual), stride_C,
            reinterpret_cast<ElementD*>(output),         stride_D,
        }
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

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
static size_t oproj_residual_ws_dispatch(int M, int N, int K)
{
    using G = OprojResidualGemm<TileShape_, ClusterShape_, KernelSchedule_>;
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
        {{ElementAccum(1.0f), ElementAccum(1.0f)}, nullptr, stride_C, nullptr, stride_D}
    };

    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

// ---------------------------------------------------------------------------
// Schedule aliases
// ---------------------------------------------------------------------------

using WS   = cutlass::gemm::KernelTmaWarpSpecialized;
using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using PP   = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

// ---------------------------------------------------------------------------
// Macro to stamp out extern "C" entry points
// ---------------------------------------------------------------------------

#define OPROJ_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SCHED)    \
extern "C" int cutlass_oproj_residual_v##ID(                                     \
    void* o, const void* i, const void* w, const void* r,                        \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {               \
    return oproj_residual_dispatch<                                              \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                 \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED>(o, i, w, r, M, N, K, ws, ws_sz, s); \
}                                                                                \
extern "C" size_t cutlass_oproj_residual_v##ID##_workspace_size(int M, int N, int K) { \
    return oproj_residual_ws_dispatch<                                           \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                 \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED>(M, N, K);                     \
}

// ---------------------------------------------------------------------------
// 10 variants -- O-proj shape is (M, 3584, 3584), square-ish
// ---------------------------------------------------------------------------
//
// ID  Tile MxNxK     Cluster  Schedule   Notes
//  0  64x128x64      1x1x1   WS         small M baseline
//  1  64x128x64      1x1x1   Coop       small M cooperative
//  2  64x256x64      1x1x1   WS         small M wide N
//  3  128x128x64     1x1x1   WS         balanced baseline
//  4  128x128x64     1x1x1   Coop       balanced cooperative
//  5  128x128x128    1x1x1   WS         K=128 for tall-K (3584)
//  6  128x256x64     1x1x1   WS         wide N
//  7  128x256x64     1x2x1   WS         wide N, N-clustered
//  8  128x256x64     1x2x1   PP         wide N, N-clustered, pingpong
//  9  256x128x64     2x1x1   WS         tall M, M-clustered

OPROJ_VARIANT(0,  64, 128, 64, 1,1,1, WS)
OPROJ_VARIANT(1, 128, 128,128, 1,1,1, Coop)  // K=128 Coop (pair of v5 WS)
OPROJ_VARIANT(2,  64, 256, 64, 1,1,1, WS)
OPROJ_VARIANT(3, 128, 128, 64, 1,1,1, WS)
OPROJ_VARIANT(4, 128, 128, 64, 1,1,1, Coop)
OPROJ_VARIANT(5, 128, 128,128, 1,1,1, WS)
OPROJ_VARIANT(6, 128, 256, 64, 1,1,1, WS)
OPROJ_VARIANT(7, 128, 256, 64, 1,2,1, WS)
OPROJ_VARIANT(8, 128, 256, 64, 1,2,1, PP)
OPROJ_VARIANT(9, 256, 128, 64, 2,1,1, WS)
OPROJ_VARIANT(10, 128, 128, 64, 1,2,1, WS)    // N-clustered balanced
OPROJ_VARIANT(11, 128, 256, 64, 2,2,1, WS)    // 4-SM cluster
// PingPong mirrors
OPROJ_VARIANT(12,  64, 128, 64, 1,1,1, PP)    // PP mirror of v1
OPROJ_VARIANT(13, 128, 128, 64, 1,1,1, PP)    // PP mirror of v4

// ===========================================================================
// SM count helper (for persistent/stream-K/swizzle schedulers)
// ===========================================================================

static int get_sm_count() {
    static int sm_count = 0;
    if (sm_count == 0) {
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    }
    return sm_count;
}

// ===========================================================================
// EXPLICIT PIPELINE STAGES: fixed stage count instead of auto-carveout
// ===========================================================================

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int NumStages>
struct OprojResidualStaged {
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum, ElementAccum,
        ElementC, LayoutC, Alignment,
        ElementD, LayoutD, Alignment,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, Alignment,
        ElementB, LayoutB, Alignment,
        ElementAccum,
        TileShape_, ClusterShape_,
        cutlass::gemm::collective::StageCount<NumStages>,
        KernelSchedule_
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int NumStages>
static int oproj_residual_staged_dispatch(
    void* output, const void* input, const void* weight, const void* residual,
    int M, int N, int K, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    using G = OprojResidualStaged<TileShape_, ClusterShape_, KernelSchedule_, NumStages>;
    using Gemm = typename G::Gemm;
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, prob_shape,
        {reinterpret_cast<const ElementA*>(input), stride_A,
         reinterpret_cast<const ElementB*>(weight), stride_B},
        {{ElementAccum(1.0f), ElementAccum(1.0f)},
         reinterpret_cast<const ElementC*>(residual), stride_C,
         reinterpret_cast<ElementD*>(output), stride_D}
    };
    Gemm gemm_op;
    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;
    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;
    status = gemm_op(stream);
    return status == cutlass::Status::kSuccess ? 0 : -3;
}

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int NumStages>
static size_t oproj_residual_staged_ws(int M, int N, int K) {
    using G = OprojResidualStaged<TileShape_, ClusterShape_, KernelSchedule_, NumStages>;
    using Gemm = typename G::Gemm;
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(1.0f)}, nullptr, stride_C, nullptr, stride_D}
    };
    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

#define OPROJ_STAGED_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SCHED, STAGES) \
extern "C" int cutlass_oproj_residual_v##ID(                                                 \
    void* o, const void* i, const void* w, const void* r,                                    \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {                          \
    return oproj_residual_staged_dispatch<                                                    \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                             \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED, STAGES>(o, i, w, r, M, N, K, ws, ws_sz, s); \
}                                                                                             \
extern "C" size_t cutlass_oproj_residual_v##ID##_workspace_size(int M, int N, int K) {       \
    return oproj_residual_staged_ws<                                                          \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                             \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED, STAGES>(M, N, K);                         \
}

// ===========================================================================
// SWIZZLE VARIANTS: rasterization order for L2 locality
// ===========================================================================

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int MaxSwizzle>
static int oproj_residual_swizzle_dispatch(
    void* output, const void* input, const void* weight, const void* residual,
    int M, int N, int K, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    using G = OprojResidualGemm<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename G::Gemm;
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, prob_shape,
        {reinterpret_cast<const ElementA*>(input), stride_A,
         reinterpret_cast<const ElementB*>(weight), stride_B},
        {{ElementAccum(1.0f), ElementAccum(1.0f)},
         reinterpret_cast<const ElementC*>(residual), stride_C,
         reinterpret_cast<ElementD*>(output), stride_D},
        hw_info, {MaxSwizzle}
    };
    Gemm gemm_op;
    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;
    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;
    status = gemm_op(stream);
    return status == cutlass::Status::kSuccess ? 0 : -3;
}

template <typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int MaxSwizzle>
static size_t oproj_residual_swizzle_ws(int M, int N, int K) {
    using G = OprojResidualGemm<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename G::Gemm;
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(1.0f)}, nullptr, stride_C, nullptr, stride_D},
        hw_info, {MaxSwizzle}
    };
    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

#define OPROJ_SWIZZLE_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SCHED, SWIZZLE) \
extern "C" int cutlass_oproj_residual_v##ID(                                                   \
    void* o, const void* i, const void* w, const void* r,                                      \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {                            \
    return oproj_residual_swizzle_dispatch<                                                     \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                               \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED, SWIZZLE>(o, i, w, r, M, N, K, ws, ws_sz, s); \
}                                                                                               \
extern "C" size_t cutlass_oproj_residual_v##ID##_workspace_size(int M, int N, int K) {         \
    return oproj_residual_swizzle_ws<                                                           \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                               \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED, SWIZZLE>(M, N, K);                          \
}

// ===========================================================================
// SPLIT-K + STREAM-K: StreamKScheduler on Cooperative kernels
// Only Cooperative schedule supports StreamKScheduler in CUTLASS 3.x SM90.
// ===========================================================================

template <typename TileShape_, typename ClusterShape_>
struct OprojResidualStreamK {
    using KSched = cutlass::gemm::KernelTmaWarpSpecializedCooperative;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum, ElementAccum,
        ElementC, LayoutC, Alignment,
        ElementD, LayoutD, Alignment,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, Alignment,
        ElementB, LayoutB, Alignment,
        ElementAccum,
        TileShape_, ClusterShape_,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KSched
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
        cutlass::gemm::StreamKScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename TileShape_, typename ClusterShape_, int SplitK>
static int oproj_residual_splitk_dispatch(
    void* output, const void* input, const void* weight, const void* residual,
    int M, int N, int K, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    using G = OprojResidualStreamK<TileShape_, ClusterShape_>;
    using Gemm = typename G::Gemm;
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();
    using SchedArgs = typename Gemm::GemmKernel::TileScheduler::Arguments;
    SchedArgs sched_args;
    sched_args.splits = SplitK;
    sched_args.max_swizzle_size = 1;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, prob_shape,
        {reinterpret_cast<const ElementA*>(input), stride_A,
         reinterpret_cast<const ElementB*>(weight), stride_B},
        {{ElementAccum(1.0f), ElementAccum(1.0f)},
         reinterpret_cast<const ElementC*>(residual), stride_C,
         reinterpret_cast<ElementD*>(output), stride_D},
        hw_info, sched_args
    };
    Gemm gemm_op;
    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;
    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;
    status = gemm_op(stream);
    return status == cutlass::Status::kSuccess ? 0 : -3;
}

template <typename TileShape_, typename ClusterShape_, int SplitK>
static size_t oproj_residual_splitk_ws(int M, int N, int K) {
    using G = OprojResidualStreamK<TileShape_, ClusterShape_>;
    using Gemm = typename G::Gemm;
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();
    using SchedArgs = typename Gemm::GemmKernel::TileScheduler::Arguments;
    SchedArgs sched_args;
    sched_args.splits = SplitK;
    sched_args.max_swizzle_size = 1;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(1.0f)}, nullptr, stride_C, nullptr, stride_D},
        hw_info, sched_args
    };
    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

template <typename TileShape_, typename ClusterShape_>
static int oproj_residual_streamk_dispatch(
    void* output, const void* input, const void* weight, const void* residual,
    int M, int N, int K, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    using G = OprojResidualStreamK<TileShape_, ClusterShape_>;
    using Gemm = typename G::Gemm;
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, prob_shape,
        {reinterpret_cast<const ElementA*>(input), stride_A,
         reinterpret_cast<const ElementB*>(weight), stride_B},
        {{ElementAccum(1.0f), ElementAccum(1.0f)},
         reinterpret_cast<const ElementC*>(residual), stride_C,
         reinterpret_cast<ElementD*>(output), stride_D},
        hw_info
    };
    Gemm gemm_op;
    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;
    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;
    status = gemm_op(stream);
    return status == cutlass::Status::kSuccess ? 0 : -3;
}

template <typename TileShape_, typename ClusterShape_>
static size_t oproj_residual_streamk_ws(int M, int N, int K) {
    using G = OprojResidualStreamK<TileShape_, ClusterShape_>;
    using Gemm = typename G::Gemm;
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(1.0f)}, nullptr, stride_C, nullptr, stride_D},
        hw_info
    };
    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

#define OPROJ_SPLITK_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SPLITK) \
extern "C" int cutlass_oproj_residual_v##ID(                                          \
    void* o, const void* i, const void* w, const void* r,                             \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {                   \
    return oproj_residual_splitk_dispatch<                                             \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                      \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SPLITK>(o, i, w, r, M, N, K, ws, ws_sz, s); \
}                                                                                      \
extern "C" size_t cutlass_oproj_residual_v##ID##_workspace_size(int M, int N, int K) {\
    return oproj_residual_splitk_ws<                                                   \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                      \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SPLITK>(M, N, K);                         \
}

#define OPROJ_STREAMK_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K)  \
extern "C" int cutlass_oproj_residual_v##ID(                                    \
    void* o, const void* i, const void* w, const void* r,                       \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {             \
    return oproj_residual_streamk_dispatch<                                     \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                               \
        Shape<_##CL_M, _##CL_N, _##CL_K>>(o, i, w, r, M, N, K, ws, ws_sz, s); \
}                                                                               \
extern "C" size_t cutlass_oproj_residual_v##ID##_workspace_size(int M, int N, int K) { \
    return oproj_residual_streamk_ws<                                           \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                               \
        Shape<_##CL_M, _##CL_N, _##CL_K>>(M, N, K);                          \
}

// ---------------------------------------------------------------------------
// Explicit pipeline stage variants (v14-v17)
// ---------------------------------------------------------------------------
OPROJ_STAGED_VARIANT(14, 128, 128, 64, 1,1,1, WS, 2)
OPROJ_STAGED_VARIANT(15, 128, 128, 64, 1,1,1, WS, 4)
OPROJ_STAGED_VARIANT(16,  64, 128, 64, 1,1,1, WS, 2)
OPROJ_STAGED_VARIANT(17,  64, 128, 64, 1,1,1, WS, 4)

// ---------------------------------------------------------------------------
// Swizzle rasterization variants (v18-v21)
// ---------------------------------------------------------------------------
OPROJ_SWIZZLE_VARIANT(18, 128, 256, 64, 1,2,1, WS, 2)
OPROJ_SWIZZLE_VARIANT(19, 128, 256, 64, 1,2,1, WS, 4)
OPROJ_SWIZZLE_VARIANT(20, 128, 128, 64, 1,1,1, WS, 2)
OPROJ_SWIZZLE_VARIANT(21, 128, 128, 64, 1,1,1, WS, 4)

// ---------------------------------------------------------------------------
// Split-K variants (v22-v27) -- Cooperative only
// ---------------------------------------------------------------------------
OPROJ_SPLITK_VARIANT(22, 128, 256, 64, 1,1,1, 2)
OPROJ_SPLITK_VARIANT(23, 128, 256, 64, 1,1,1, 4)
OPROJ_SPLITK_VARIANT(24, 128, 256, 64, 1,1,1, 8)
OPROJ_SPLITK_VARIANT(25, 128, 128, 64, 1,1,1, 2)
OPROJ_SPLITK_VARIANT(26, 128, 128, 64, 1,1,1, 4)
OPROJ_SPLITK_VARIANT(27, 128, 128, 64, 1,1,1, 8)

// ---------------------------------------------------------------------------
// Stream-K variants (v28-v30) -- Cooperative only
// ---------------------------------------------------------------------------
OPROJ_STREAMK_VARIANT(28, 256, 128, 64, 1,1,1)
OPROJ_STREAMK_VARIANT(29, 128, 128, 64, 1,1,1)
OPROJ_STREAMK_VARIANT(30, 128, 256, 64, 1,1,1)
