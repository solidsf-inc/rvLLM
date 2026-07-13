// CUTLASS 3.x SM90 autotuned FP8 GEMM variants for rvLLM.
//
// 15 tile/cluster/schedule configurations compiled as separate extern "C"
// entry points. The Rust autotune engine benchmarks all variants per
// (M,N,K) shape and caches the winner.
//
// D[m,n] = cast_to_f16(a_scales[m] * b_scale[0] * sum_k(A_fp8[m,k] * B_fp8[k,n]))
//
// Uses CUTLASS 3.x EVT to fuse per-row/per-tensor scaling into the epilogue.
// No post-kernel scale application needed.
//
// A=[M,K] RowMajor FP8 E4M3, B=[N,K] RowMajor (ColumnMajor to CUTLASS) FP8 E4M3
// D=[M,N] RowMajor F16

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
using ElementD = cutlass::half_t;
using ElementAccum = float;
using ElementCompute = float;
using ElementScalar = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

// ---------------------------------------------------------------------------
// EVT epilogue: D[m,n] = cast<f16>(row_scale[m] * col_scale * Acc[m,n])
// ---------------------------------------------------------------------------

template<typename TileShape_>
using ScaledAccT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>,
    cutlass::epilogue::fusion::Sm90AccFetch
>;

template<typename TileShape_>
using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementD, ElementCompute, RoundStyle>,
    cutlass::epilogue::fusion::Sm90RowBroadcast<
        0, TileShape_, ElementScalar>,
    ScaledAccT<TileShape_>
>;

// ---------------------------------------------------------------------------
// Template: build a full CUTLASS 3.x FP8 GEMM type with fused scaling
// ---------------------------------------------------------------------------

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
struct Fp8GemmType {
    using EVT = EpilogueEVT<TileShape_>;

    using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum, ElementCompute,
        void, LayoutD, 8,
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
// Dispatch template: run an FP8 GEMM variant with fused scaling
// ---------------------------------------------------------------------------

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
int fp8_gemm_dispatch(
    void* output, const void* a, const void* b,
    const void* a_scales, const void* b_scale,
    int M, int N, int K,
    void* workspace, size_t workspace_size,
    cudaStream_t stream)
{
    using G = Fp8GemmType<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename G::Gemm;

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
            {},
            nullptr, {},
            reinterpret_cast<ElementD*>(output), stride_D,
        }
    };

    args.epilogue.thread = {
        {reinterpret_cast<const ElementScalar*>(a_scales), ElementScalar(0), {}},
        {
            {{ElementScalar(0)}, {reinterpret_cast<const ElementScalar*>(b_scale)}, {}},
            {},
            {}
        },
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
size_t fp8_gemm_ws_dispatch(int M, int N, int K)
{
    using G = Fp8GemmType<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename G::Gemm;

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

// ---------------------------------------------------------------------------
// Schedule aliases
// ---------------------------------------------------------------------------

using WS   = cutlass::gemm::KernelTmaWarpSpecialized;
using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using PP   = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using FP8WS   = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
using FP8Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
using FP8PP   = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;

// ---------------------------------------------------------------------------
// Macro to stamp out extern "C" entry points
// ---------------------------------------------------------------------------

#define FP8_GEMM_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SCHED) \
extern "C" int cutlass_fp8_gemm_v##ID(                                          \
    void* o, const void* a, const void* b,                                      \
    const void* a_scales, const void* b_scale,                                  \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {              \
    return fp8_gemm_dispatch<                                                   \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED>(                             \
            o, a, b, a_scales, b_scale, M, N, K, ws, ws_sz, s);               \
}                                                                               \
extern "C" size_t cutlass_fp8_gemm_v##ID##_workspace_size(int M, int N, int K) {\
    return fp8_gemm_ws_dispatch<                                                \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED>(M, N, K);                   \
}

// ---------------------------------------------------------------------------
// 15 variants
// ---------------------------------------------------------------------------

FP8_GEMM_VARIANT( 0,  64, 128, 128, 1,1,1, WS)
FP8_GEMM_VARIANT( 1, 128, 128, 256, 1,1,1, Coop)
FP8_GEMM_VARIANT( 2,  64, 256, 128, 1,1,1, WS)
FP8_GEMM_VARIANT( 3,  64, 256, 128, 1,2,1, WS)
FP8_GEMM_VARIANT( 4, 128, 128, 128, 1,1,1, WS)
FP8_GEMM_VARIANT( 5, 128, 128, 128, 1,1,1, Coop)
FP8_GEMM_VARIANT( 6, 128, 256, 128, 1,1,1, WS)
FP8_GEMM_VARIANT( 7, 128, 256, 128, 1,1,1, Coop)
FP8_GEMM_VARIANT( 8, 128, 256, 128, 1,2,1, WS)
FP8_GEMM_VARIANT( 9, 128, 256, 128, 1,2,1, Coop)
FP8_GEMM_VARIANT(10, 128, 256, 128, 1,2,1, PP)
FP8_GEMM_VARIANT(11, 256, 128, 128, 2,1,1, WS)
FP8_GEMM_VARIANT(12, 128, 256, 128, 2,2,1, WS)
FP8_GEMM_VARIANT(13, 128, 128, 256, 1,1,1, WS)
FP8_GEMM_VARIANT(14, 128, 256, 128, 1,4,1, WS)

// ---------------------------------------------------------------------------
// FP8FastAccum variants (v15-v24): reduced-precision accumulation for higher
// throughput. Enables K=256 tiles rejected by standard schedules.
// ---------------------------------------------------------------------------

// Decode-optimized (tile_M=64, FP8FastAccum)
FP8_GEMM_VARIANT(15,  64, 128, 128, 1,1,1, FP8WS)
FP8_GEMM_VARIANT(16,  64, 128, 256, 1,1,1, FP8WS)   // deep K -- halves K-iterations
FP8_GEMM_VARIANT(17,  64, 256, 128, 1,1,1, FP8WS)
FP8_GEMM_VARIANT(18,  64, 256, 128, 1,2,1, FP8WS)
FP8_GEMM_VARIANT(19,  64, 256, 256, 1,1,1, FP8WS)   // wide N + deep K

// Prefill-optimized (tile_M=128, FP8FastAccum)
FP8_GEMM_VARIANT(20, 128, 128, 256, 1,1,1, FP8Coop)
FP8_GEMM_VARIANT(21, 128, 128, 128, 1,1,1, FP8Coop)
FP8_GEMM_VARIANT(22, 128, 256, 128, 1,1,1, FP8Coop)
FP8_GEMM_VARIANT(23, 128, 256, 128, 1,2,1, FP8Coop)
FP8_GEMM_VARIANT(24, 128, 256, 128, 1,2,1, FP8PP)

// ===========================================================================
// SM count helper (for stream-K / split-K schedulers)
// ===========================================================================

static int get_sm_count() {
    static int sm_count = 0;
    if (sm_count == 0) {
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    }
    return sm_count;
}

// ===========================================================================
// STREAM-K / SPLIT-K: StreamKScheduler on Cooperative kernels with EVT
// Only Cooperative schedule supports StreamKScheduler in CUTLASS 3.x SM90.
// ===========================================================================

template<typename TileShape_, typename ClusterShape_,
         typename KernelSchedule_ = cutlass::gemm::KernelTmaWarpSpecializedCooperative>
struct Fp8GemmStreamK {
    using EVT = EpilogueEVT<TileShape_>;

    using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum, ElementCompute,
        void, LayoutD, 8,
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
        Shape<int, int, int, int>, MainloopOp, EpilogueOp,
        cutlass::gemm::StreamKScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// ---------------------------------------------------------------------------
// Stream-K dispatch: heuristic decomposition for load balancing
// ---------------------------------------------------------------------------

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
int fp8_gemm_streamk_dispatch(
    void* output, const void* a, const void* b,
    const void* a_scales, const void* b_scale,
    int M, int N, int K,
    void* workspace, size_t workspace_size,
    cudaStream_t stream)
{
    using G = Fp8GemmStreamK<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename G::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {
            reinterpret_cast<const ElementA*>(a), stride_A,
            reinterpret_cast<const ElementB*>(b), stride_B,
        },
        {
            {},
            nullptr, {},
            reinterpret_cast<ElementD*>(output), stride_D,
        },
        hw_info
    };

    args.epilogue.thread = {
        {reinterpret_cast<const ElementScalar*>(a_scales), ElementScalar(0), {}},
        {
            {{ElementScalar(0)}, {reinterpret_cast<const ElementScalar*>(b_scale)}, {}},
            {},
            {}
        },
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

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
size_t fp8_gemm_streamk_ws(int M, int N, int K)
{
    using G = Fp8GemmStreamK<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename G::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{}, nullptr, {}, nullptr, stride_D},
        hw_info
    };

    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

// ---------------------------------------------------------------------------
// Split-K dispatch: explicit K decomposition across threadblocks
// ---------------------------------------------------------------------------

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int SplitK>
int fp8_gemm_splitk_dispatch(
    void* output, const void* a, const void* b,
    const void* a_scales, const void* b_scale,
    int M, int N, int K,
    void* workspace, size_t workspace_size,
    cudaStream_t stream)
{
    using G = Fp8GemmStreamK<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename G::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();

    using SchedArgs = typename Gemm::GemmKernel::TileScheduler::Arguments;
    SchedArgs sched_args;
    sched_args.splits = SplitK;
    sched_args.max_swizzle_size = 1;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {
            reinterpret_cast<const ElementA*>(a), stride_A,
            reinterpret_cast<const ElementB*>(b), stride_B,
        },
        {
            {},
            nullptr, {},
            reinterpret_cast<ElementD*>(output), stride_D,
        },
        hw_info, sched_args
    };

    args.epilogue.thread = {
        {reinterpret_cast<const ElementScalar*>(a_scales), ElementScalar(0), {}},
        {
            {{ElementScalar(0)}, {reinterpret_cast<const ElementScalar*>(b_scale)}, {}},
            {},
            {}
        },
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

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int SplitK>
size_t fp8_gemm_splitk_ws(int M, int N, int K)
{
    using G = Fp8GemmStreamK<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename G::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();

    using SchedArgs = typename Gemm::GemmKernel::TileScheduler::Arguments;
    SchedArgs sched_args;
    sched_args.splits = SplitK;
    sched_args.max_swizzle_size = 1;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{}, nullptr, {}, nullptr, stride_D},
        hw_info, sched_args
    };

    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

// ---------------------------------------------------------------------------
// Macros for stream-K and split-K entry points
// ---------------------------------------------------------------------------

#define FP8_GEMM_STREAMK_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SCHED) \
extern "C" int cutlass_fp8_gemm_v##ID(                                          \
    void* o, const void* a, const void* b,                                      \
    const void* a_scales, const void* b_scale,                                  \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {              \
    return fp8_gemm_streamk_dispatch<                                            \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED>(                             \
            o, a, b, a_scales, b_scale, M, N, K, ws, ws_sz, s);               \
}                                                                               \
extern "C" size_t cutlass_fp8_gemm_v##ID##_workspace_size(int M, int N, int K) {\
    return fp8_gemm_streamk_ws<                                                 \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED>(M, N, K);                   \
}

#define FP8_GEMM_SPLITK_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SCHED, SPLITK) \
extern "C" int cutlass_fp8_gemm_v##ID(                                          \
    void* o, const void* a, const void* b,                                      \
    const void* a_scales, const void* b_scale,                                  \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {              \
    return fp8_gemm_splitk_dispatch<                                             \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED, SPLITK>(                     \
            o, a, b, a_scales, b_scale, M, N, K, ws, ws_sz, s);               \
}                                                                               \
extern "C" size_t cutlass_fp8_gemm_v##ID##_workspace_size(int M, int N, int K) {\
    return fp8_gemm_splitk_ws<                                                  \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED, SPLITK>(M, N, K);           \
}

// ---------------------------------------------------------------------------
// Stream-K variants (v25-v27): heuristic K-decomposition for SM load balancing
// Cooperative requires tile_M >= 128; M=64 decode uses 128-tile with M-padding
// but stream-K still helps by splitting K-work across more SMs.
// Target: O-proj (M<=128,N=3584,K=3584) and Down (M<=128,N=3584,K=18944)
// ---------------------------------------------------------------------------

FP8_GEMM_STREAMK_VARIANT(25, 128, 128, 128, 1,1,1, Coop)     // auto-balance
FP8_GEMM_STREAMK_VARIANT(26, 128, 256, 128, 1,1,1, Coop)     // wide N
FP8_GEMM_STREAMK_VARIANT(27, 128, 128, 128, 1,1,1, FP8Coop)  // FP8FastAccum

// ---------------------------------------------------------------------------
// Split-K variants (v28-v39): explicit K-decomposition
// Down proj: M<=128, N=3584, K=18944 -> 28 output tiles on 132 SMs
//   split-K=2 -> 56 (42%), split-K=3 -> 84 (64%), split-K=4 -> 112 (85%)
//   split-K=5 -> 140 (106%), split-K=6 -> 168 (127%), split-K=8 -> 224 (170%)
// ---------------------------------------------------------------------------

// Cooperative (full-precision accumulation)
FP8_GEMM_SPLITK_VARIANT(28, 128, 128, 128, 1,1,1, Coop, 2)
FP8_GEMM_SPLITK_VARIANT(29, 128, 128, 128, 1,1,1, Coop, 4)
FP8_GEMM_SPLITK_VARIANT(30, 128, 256, 128, 1,1,1, Coop, 2)
FP8_GEMM_SPLITK_VARIANT(31, 128, 256, 128, 1,1,1, Coop, 4)
FP8_GEMM_SPLITK_VARIANT(32, 128, 128, 128, 1,1,1, Coop, 3)
FP8_GEMM_SPLITK_VARIANT(33, 128, 128, 128, 1,1,1, Coop, 5)
FP8_GEMM_SPLITK_VARIANT(34, 128, 128, 128, 1,1,1, Coop, 6)
FP8_GEMM_SPLITK_VARIANT(35, 128, 128, 128, 1,1,1, Coop, 8)

// FP8FastAccum split-K (higher throughput, reduced precision)
FP8_GEMM_SPLITK_VARIANT(36, 128, 128, 128, 1,1,1, FP8Coop, 4)
FP8_GEMM_SPLITK_VARIANT(37, 128, 128, 128, 1,1,1, FP8Coop, 5)
FP8_GEMM_SPLITK_VARIANT(38, 128, 256, 128, 1,1,1, Coop, 5)
FP8_GEMM_SPLITK_VARIANT(39, 128, 256, 128, 1,1,1, Coop, 6)
