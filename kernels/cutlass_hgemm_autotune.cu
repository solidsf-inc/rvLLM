// CUTLASS 3.x SM90 autotuned HGEMM variants for rvLLM.
//
// 20 tile/cluster/schedule configurations compiled as separate extern "C"
// entry points. The Rust autotune engine benchmarks all variants per
// (M,N,K) shape and caches the winner.
//
// D = A @ B^T, alpha=1, beta=0
// A=[M,K] RowMajor, B=[N,K] RowMajor (ColumnMajor to CUTLASS), D=[M,N] RowMajor

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

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccum = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// ---------------------------------------------------------------------------
// Template: build a full CUTLASS 3.x GEMM type from tile/cluster/schedule
// ---------------------------------------------------------------------------

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
struct HgemmType {
    using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum, ElementAccum,
        ElementC, LayoutC, 8,
        ElementC, LayoutC, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, 8,
        ElementB, LayoutB, 8,
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
// Dispatch template: run a GEMM variant
// ---------------------------------------------------------------------------

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
int cutlass_hgemm_dispatch(
    void* output, const void* input, const void* weight,
    int M, int N, int K,
    void* workspace, size_t workspace_size,
    cudaStream_t stream)
{
    using G = HgemmType<TileShape_, ClusterShape_, KernelSchedule_>;
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
            reinterpret_cast<const ElementA*>(input), stride_A,
            reinterpret_cast<const ElementB*>(weight), stride_B,
        },
        {
            {ElementAccum(1.0f), ElementAccum(0.0f)},
            reinterpret_cast<const ElementC*>(output), stride_C,
            reinterpret_cast<ElementC*>(output), stride_D,
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

// ---------------------------------------------------------------------------
// Workspace-size template
// ---------------------------------------------------------------------------

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
size_t cutlass_hgemm_ws_dispatch(int M, int N, int K)
{
    using G = HgemmType<TileShape_, ClusterShape_, KernelSchedule_>;
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
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D}
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

#define HGEMM_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SCHED)   \
extern "C" int cutlass_hgemm_v##ID(                                             \
    void* o, const void* i, const void* w,                                      \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {              \
    return cutlass_hgemm_dispatch<                                              \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED>(o, i, w, M, N, K, ws, ws_sz, s); \
}                                                                               \
extern "C" size_t cutlass_hgemm_v##ID##_workspace_size(int M, int N, int K) {   \
    return cutlass_hgemm_ws_dispatch<                                           \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED>(M, N, K);                    \
}

// ---------------------------------------------------------------------------
// 30 variants: WS/Coop/PP x tile/cluster combos
// ---------------------------------------------------------------------------
//
// ID  Tile MxNxK     Cluster  Schedule   Notes
//  0  64x64x64       1x1x1   WS         tiny M (M=1-4)
//  1  64x128x64      1x1x1   WS         small M baseline
//  2  64x128x64      1x1x1   Coop       small M cooperative
//  3  64x256x64      1x1x1   WS         small M, wide N
//  4  64x256x64      1x1x1   Coop       small M, wide N, cooperative
//  5  64x256x64      1x2x1   WS         small M, wide N, N-clustered
//  6  128x128x64     1x1x1   WS         balanced baseline
//  7  128x128x64     1x1x1   Coop       balanced cooperative
//  8  128x128x64     2x1x1   WS         balanced M-clustered
//  9  128x128x128    1x1x1   WS         K=128 for tall-K (down_proj K=18944)
// 10  128x256x64     1x1x1   WS         wide N baseline
// 11  128x256x64     1x1x1   Coop       wide N cooperative
// 12  128x256x64     1x2x1   WS         wide N, N-clustered
// 13  128x256x64     1x2x1   Coop       wide N, N-clustered, cooperative
// 14  128x256x64     1x2x1   PP         wide N, N-clustered, pingpong
// 15  128x256x128    1x2x1   WS         K=128, N-clustered
// 16  256x128x64     1x1x1   WS         tall M, no cluster
// 17  256x128x64     2x1x1   WS         tall M, M-clustered
// 18  256x128x64     2x1x1   Coop       tall M, M-clustered, cooperative
// 19  256x256x64     1x1x1   WS         huge tile
// 20  128x256x64     2x2x1   WS         4-SM, balanced cluster
// 21  128x256x64     2x2x1   Coop       4-SM, cooperative
// 22  64x256x64      1x4x1   WS         4-SM along N, small M
// 23  128x256x64     1x4x1   WS         4-SM along N, medium M
// -- PingPong mirrors (double-buffer producer/consumer overlap) --
// 24  64x128x64      1x1x1   PP         PP mirror of v2
// 25  64x256x64      1x1x1   PP         PP mirror of v4
// 26  128x128x64     1x1x1   PP         PP mirror of v7
// 27  128x256x64     1x1x1   PP         PP mirror of v11
// 28  256x128x64     2x1x1   PP         PP mirror of v18
// 29  128x256x64     2x2x1   PP         PP mirror of v21

HGEMM_VARIANT( 0,  64,  64, 64, 1,1,1, WS)
HGEMM_VARIANT( 1,  64, 128, 64, 1,1,1, WS)
HGEMM_VARIANT( 2, 128, 128,128, 1,1,1, Coop)  // K=128 Coop (pair of v9 WS)
HGEMM_VARIANT( 3,  64, 256, 64, 1,1,1, WS)
HGEMM_VARIANT( 4, 256, 256, 64, 1,1,1, Coop)  // big tile Coop (pair of v19 WS)
HGEMM_VARIANT( 5,  64, 256, 64, 1,2,1, WS)
HGEMM_VARIANT( 6, 128, 128, 64, 1,1,1, WS)
HGEMM_VARIANT( 7, 128, 128, 64, 1,1,1, Coop)
HGEMM_VARIANT( 8, 128, 128, 64, 2,1,1, WS)
HGEMM_VARIANT( 9, 128, 128,128, 1,1,1, WS)
HGEMM_VARIANT(10, 128, 256, 64, 1,1,1, WS)
HGEMM_VARIANT(11, 128, 256, 64, 1,1,1, Coop)
HGEMM_VARIANT(12, 128, 256, 64, 1,2,1, WS)
HGEMM_VARIANT(13, 128, 256, 64, 1,2,1, Coop)
HGEMM_VARIANT(14, 128, 256, 64, 1,2,1, PP)
HGEMM_VARIANT(15, 128, 256,128, 1,2,1, WS)
HGEMM_VARIANT(16, 256, 128, 64, 1,1,1, WS)
HGEMM_VARIANT(17, 256, 128, 64, 2,1,1, WS)
HGEMM_VARIANT(18, 256, 128, 64, 2,1,1, Coop)
HGEMM_VARIANT(19, 256, 256, 64, 1,1,1, WS)
// 4-SM cluster variants for huge N (LM head N=152064, gate_up N=37888)
HGEMM_VARIANT(20, 128, 256, 64, 2,2,1, WS)    // 4-SM, balanced cluster
HGEMM_VARIANT(21, 128, 256, 64, 2,2,1, Coop)   // 4-SM, cooperative
HGEMM_VARIANT(22,  64, 256, 64, 1,4,1, WS)     // 4-SM along N, small M
HGEMM_VARIANT(23, 128, 256, 64, 1,4,1, WS)     // 4-SM along N, medium M
// PingPong mirrors: PP on every tile/cluster where Coop exists
HGEMM_VARIANT(24,  64, 128, 64, 1,1,1, PP)     // PP mirror of v2
HGEMM_VARIANT(25,  64, 256, 64, 1,1,1, PP)     // PP mirror of v4
HGEMM_VARIANT(26, 128, 128, 64, 1,1,1, PP)     // PP mirror of v7
HGEMM_VARIANT(27, 128, 256, 64, 1,1,1, PP)     // PP mirror of v11
HGEMM_VARIANT(28, 256, 128, 64, 2,1,1, PP)     // PP mirror of v18
HGEMM_VARIANT(29, 128, 256, 64, 2,2,1, PP)     // PP mirror of v21

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

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int NumStages>
struct HgemmStaged {
    using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum, ElementAccum,
        ElementC, LayoutC, 8,
        ElementC, LayoutC, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, 8,
        ElementB, LayoutB, 8,
        ElementAccum,
        TileShape_,
        ClusterShape_,
        cutlass::gemm::collective::StageCount<NumStages>,
        KernelSchedule_
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, MainloopOp, EpilogueOp>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int NumStages>
int cutlass_hgemm_staged_dispatch(
    void* output, const void* input, const void* weight,
    int M, int N, int K, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    using G = HgemmStaged<TileShape_, ClusterShape_, KernelSchedule_, NumStages>;
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
        {{ElementAccum(1.0f), ElementAccum(0.0f)},
         reinterpret_cast<const ElementC*>(output), stride_C,
         reinterpret_cast<ElementC*>(output), stride_D}
    };
    Gemm gemm_op;
    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;
    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;
    status = gemm_op(stream);
    return status == cutlass::Status::kSuccess ? 0 : -3;
}

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int NumStages>
size_t cutlass_hgemm_staged_ws(int M, int N, int K) {
    using G = HgemmStaged<TileShape_, ClusterShape_, KernelSchedule_, NumStages>;
    using Gemm = typename G::Gemm;
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D}
    };
    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

#define HGEMM_STAGED_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SCHED, STAGES) \
extern "C" int cutlass_hgemm_v##ID(                                                          \
    void* o, const void* i, const void* w,                                                    \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {                           \
    return cutlass_hgemm_staged_dispatch<                                                     \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                              \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED, STAGES>(o, i, w, M, N, K, ws, ws_sz, s);  \
}                                                                                             \
extern "C" size_t cutlass_hgemm_v##ID##_workspace_size(int M, int N, int K) {                \
    return cutlass_hgemm_staged_ws<                                                           \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                              \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED, STAGES>(M, N, K);                          \
}

// ===========================================================================
// SWIZZLE VARIANTS: rasterization order for L2 locality
// ===========================================================================

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int MaxSwizzle>
int cutlass_hgemm_swizzle_dispatch(
    void* output, const void* input, const void* weight,
    int M, int N, int K, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    using G = HgemmType<TileShape_, ClusterShape_, KernelSchedule_>;
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
        {{ElementAccum(1.0f), ElementAccum(0.0f)},
         reinterpret_cast<const ElementC*>(output), stride_C,
         reinterpret_cast<ElementC*>(output), stride_D},
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

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_, int MaxSwizzle>
size_t cutlass_hgemm_swizzle_ws(int M, int N, int K) {
    using G = HgemmType<TileShape_, ClusterShape_, KernelSchedule_>;
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
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D},
        hw_info, {MaxSwizzle}
    };
    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

#define HGEMM_SWIZZLE_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SCHED, SWIZZLE) \
extern "C" int cutlass_hgemm_v##ID(                                                            \
    void* o, const void* i, const void* w,                                                      \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {                             \
    return cutlass_hgemm_swizzle_dispatch<                                                      \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED, SWIZZLE>(o, i, w, M, N, K, ws, ws_sz, s);   \
}                                                                                               \
extern "C" size_t cutlass_hgemm_v##ID##_workspace_size(int M, int N, int K) {                  \
    return cutlass_hgemm_swizzle_ws<                                                            \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED, SWIZZLE>(M, N, K);                           \
}

// ===========================================================================
// SPLIT-K + STREAM-K: StreamKScheduler on Cooperative kernels
// Only Cooperative schedule supports StreamKScheduler in CUTLASS 3.x SM90.
// ===========================================================================

template<typename TileShape_, typename ClusterShape_>
struct HgemmStreamK {
    using KSched = cutlass::gemm::KernelTmaWarpSpecializedCooperative;

    using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum, ElementAccum,
        ElementC, LayoutC, 8,
        ElementC, LayoutC, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, 8,
        ElementB, LayoutB, 8,
        ElementAccum,
        TileShape_,
        ClusterShape_,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename EpilogueOp::SharedStorage))>,
        KSched
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, MainloopOp, EpilogueOp,
        cutlass::gemm::StreamKScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Split-K dispatch: decomposes K across threadblocks
template<typename TileShape_, typename ClusterShape_, int SplitK>
int cutlass_hgemm_splitk_dispatch(
    void* output, const void* input, const void* weight,
    int M, int N, int K, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    using G = HgemmStreamK<TileShape_, ClusterShape_>;
    using Gemm = typename G::Gemm;
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();
    // Build scheduler args: force split-K with given factor
    using SchedArgs = typename Gemm::GemmKernel::TileScheduler::Arguments;
    SchedArgs sched_args;
    sched_args.splits = SplitK;
    sched_args.max_swizzle_size = 1;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, prob_shape,
        {reinterpret_cast<const ElementA*>(input), stride_A,
         reinterpret_cast<const ElementB*>(weight), stride_B},
        {{ElementAccum(1.0f), ElementAccum(0.0f)},
         reinterpret_cast<const ElementC*>(output), stride_C,
         reinterpret_cast<ElementC*>(output), stride_D},
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

template<typename TileShape_, typename ClusterShape_, int SplitK>
size_t cutlass_hgemm_splitk_ws(int M, int N, int K) {
    using G = HgemmStreamK<TileShape_, ClusterShape_>;
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
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D},
        hw_info, sched_args
    };
    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

// Stream-K dispatch: heuristic decomposition for load balancing
template<typename TileShape_, typename ClusterShape_>
int cutlass_hgemm_streamk_dispatch(
    void* output, const void* input, const void* weight,
    int M, int N, int K, void* workspace, size_t workspace_size, cudaStream_t stream)
{
    using G = HgemmStreamK<TileShape_, ClusterShape_>;
    using Gemm = typename G::Gemm;
    auto prob_shape = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = get_sm_count();
    // Default scheduler args: heuristic decides data-parallel vs stream-K
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, prob_shape,
        {reinterpret_cast<const ElementA*>(input), stride_A,
         reinterpret_cast<const ElementB*>(weight), stride_B},
        {{ElementAccum(1.0f), ElementAccum(0.0f)},
         reinterpret_cast<const ElementC*>(output), stride_C,
         reinterpret_cast<ElementC*>(output), stride_D},
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

template<typename TileShape_, typename ClusterShape_>
size_t cutlass_hgemm_streamk_ws(int M, int N, int K) {
    using G = HgemmStreamK<TileShape_, ClusterShape_>;
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
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D},
        hw_info
    };
    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

#define HGEMM_SPLITK_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SPLITK) \
extern "C" int cutlass_hgemm_v##ID(                                                    \
    void* o, const void* i, const void* w,                                              \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {                     \
    return cutlass_hgemm_splitk_dispatch<                                               \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                        \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SPLITK>(o, i, w, M, N, K, ws, ws_sz, s);   \
}                                                                                       \
extern "C" size_t cutlass_hgemm_v##ID##_workspace_size(int M, int N, int K) {          \
    return cutlass_hgemm_splitk_ws<                                                     \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                        \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SPLITK>(M, N, K);                           \
}

#define HGEMM_STREAMK_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K)  \
extern "C" int cutlass_hgemm_v##ID(                                             \
    void* o, const void* i, const void* w,                                      \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {             \
    return cutlass_hgemm_streamk_dispatch<                                      \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>>(o, i, w, M, N, K, ws, ws_sz, s);   \
}                                                                               \
extern "C" size_t cutlass_hgemm_v##ID##_workspace_size(int M, int N, int K) {  \
    return cutlass_hgemm_streamk_ws<                                            \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>>(M, N, K);                           \
}

// ---------------------------------------------------------------------------
// Explicit pipeline stage variants (v30-v35)
// ---------------------------------------------------------------------------
HGEMM_STAGED_VARIANT(30, 128, 256, 64, 1,2,1, WS, 2)
HGEMM_STAGED_VARIANT(31, 128, 256, 64, 1,2,1, WS, 4)
HGEMM_STAGED_VARIANT(32, 128, 128, 64, 1,1,1, WS, 2)
HGEMM_STAGED_VARIANT(33, 128, 128, 64, 1,1,1, WS, 4)
HGEMM_STAGED_VARIANT(34,  64, 256, 64, 1,2,1, WS, 2)
HGEMM_STAGED_VARIANT(35,  64, 256, 64, 1,2,1, WS, 4)

// ---------------------------------------------------------------------------
// Swizzle rasterization variants (v36-v41)
// ---------------------------------------------------------------------------
HGEMM_SWIZZLE_VARIANT(36, 128, 256, 64, 1,2,1, WS, 2)
HGEMM_SWIZZLE_VARIANT(37, 128, 256, 64, 1,2,1, WS, 4)
HGEMM_SWIZZLE_VARIANT(38, 128, 256, 64, 1,1,1, WS, 2)
HGEMM_SWIZZLE_VARIANT(39, 128, 256, 64, 1,1,1, WS, 4)
HGEMM_SWIZZLE_VARIANT(40,  64, 256, 64, 1,2,1, WS, 2)
HGEMM_SWIZZLE_VARIANT(41,  64, 256, 64, 1,2,1, WS, 4)

// ---------------------------------------------------------------------------
// Split-K variants for M>=128 decode (v42-v50) -- Cooperative only
// ---------------------------------------------------------------------------
HGEMM_SPLITK_VARIANT(42, 128, 256, 64, 1,1,1, 2)
HGEMM_SPLITK_VARIANT(43, 128, 256, 64, 1,1,1, 4)
HGEMM_SPLITK_VARIANT(44, 128, 256, 64, 1,1,1, 8)
HGEMM_SPLITK_VARIANT(45, 256, 128, 64, 1,1,1, 2)
HGEMM_SPLITK_VARIANT(46, 256, 128, 64, 1,1,1, 4)
HGEMM_SPLITK_VARIANT(47, 256, 128, 64, 1,1,1, 8)
HGEMM_SPLITK_VARIANT(48, 128, 128, 64, 1,1,1, 2)
HGEMM_SPLITK_VARIANT(49, 128, 128, 64, 1,1,1, 4)
HGEMM_SPLITK_VARIANT(50, 128, 128, 64, 1,1,1, 8)

// ---------------------------------------------------------------------------
// Stream-K variants for irregular grid shapes (v51-v54) -- Cooperative only
// ---------------------------------------------------------------------------
HGEMM_STREAMK_VARIANT(51, 256, 128, 64, 1,1,1)
HGEMM_STREAMK_VARIANT(52, 128, 128, 64, 1,1,1)
HGEMM_STREAMK_VARIANT(53, 128, 256, 64, 1,1,1)
HGEMM_STREAMK_VARIANT(54, 256, 256, 64, 1,1,1)

// ---------------------------------------------------------------------------
// M=64 WGMMA additional WS/PP variants (v55-v62)
// Cooperative/split-K/stream-K require M>=128, so M=64 is WS/PP only.
// Strategy: maximize N-tile variety + stage/swizzle combos for the autotune
// to find optimal configs per shape.
// ---------------------------------------------------------------------------

// Wide-K variants for down_proj (K=18944) -- more K per tile = fewer K-iterations
HGEMM_STAGED_VARIANT(55,  64, 256, 64, 1,1,1, WS, 2)   // M=64, wide N, 2 stages
HGEMM_STAGED_VARIANT(56,  64, 256, 64, 1,1,1, WS, 4)   // M=64, wide N, 4 stages
HGEMM_STAGED_VARIANT(57,  64, 128, 64, 1,1,1, WS, 2)   // M=64, medium N, 2 stages
HGEMM_STAGED_VARIANT(58,  64, 128, 64, 1,1,1, WS, 4)   // M=64, medium N, 4 stages
// PingPong with clusters -- overlap producer/consumer + N-parallelism
HGEMM_VARIANT(59,  64, 256, 64, 1,2,1, PP)              // PP + N-clustered
HGEMM_VARIANT(60,  64, 128, 64, 1,2,1, PP)              // PP + N-clustered, medium N
// Swizzle for M=64 without cluster
HGEMM_SWIZZLE_VARIANT(61,  64, 256, 64, 1,1,1, WS, 2)  // swizzle 2, no cluster
HGEMM_SWIZZLE_VARIANT(62,  64, 256, 64, 1,1,1, WS, 4)  // swizzle 4, no cluster

// ---------------------------------------------------------------------------
// 128xN split-K for M=32 decode (v63-v68)
// M=32 rounds to M=128 tile (75% M-waste) but split-K fills all 132 SMs.
// WGMMA 4x throughput + TMA pipeline may compensate for the M-waste.
// The autotune benchmarks them head-to-head with cuBLAS SM80 tiles.
// ---------------------------------------------------------------------------
HGEMM_SPLITK_VARIANT(63, 128, 128, 64, 1,1,1, 16)
HGEMM_SPLITK_VARIANT(64, 128, 256, 64, 1,1,1, 16)
HGEMM_STREAMK_VARIANT(65, 128, 256, 64, 1,2,1)   // stream-K + 2-SM N-cluster
HGEMM_STREAMK_VARIANT(66, 128, 128, 64, 1,2,1)   // stream-K + 2-SM N-cluster
