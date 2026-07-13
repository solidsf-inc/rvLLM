// CUTLASS 3.x SM90 fused O-projection GEMM + residual add.
//
// D[M,N] = A[M,K] @ B[K,N]^T + residual[M,N]
//
// Standard CUTLASS LinearCombination: D = alpha*A@B + beta*C
// With alpha=1, beta=1, C=residual => D = GEMM_output + residual
//
// Replaces separate O-proj GEMM + elementwise residual add with one pass.
// RMSNorm stays separate (requires row reduction).
//
// Build: nvcc -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
//        -I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/tools/util/include \
//        -O3 -o libcutlass_oproj_residual.so --shared -Xcompiler -fPIC \
//        cutlass_oproj_residual.cu

#include <cutlass/cutlass.h>
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

// Types: fp16 compute, fp32 accumulator
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;  // residual
using ElementD = cutlass::half_t;  // output
using ElementAccum = float;

// A[M,K] row-major, B[N,K] row-major stored as col-major (B^T), C/D[M,N] row-major
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// Tile 128x128x64 for Hopper WGMMA
using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;

// Epilogue: D = 1.0 * AccumTile + 1.0 * C (residual)
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementAccum,
    ElementC, LayoutC, 8,
    ElementD, LayoutD, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Mainloop with stage count auto-carved for epilogue shared memory
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8,
    ElementB, LayoutB, 8,
    ElementAccum,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

extern "C" {

int cutlass_oproj_residual_gemm(
    void* output,            // [M, N] = gemm_result + residual
    const void* input,       // [M, K] (attention output)
    const void* weight,      // [N, K] (row-major, treated as col-major for B^T)
    const void* residual,    // [M, N]
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
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    // alpha=1, beta=1: D = 1*AccumTile + 1*residual
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

size_t cutlass_oproj_residual_workspace_size(int M, int N, int K) {
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

} // extern "C"
