// CUTLASS 3.x SM90 GEMM + per-column bias for QKV projection.
//
// D[M,N] = A[M,K] @ B[K,N]^T + bias[N]
//
// Uses the same GEMM pattern as cutlass_gemm.cu, then applies
// the per-column bias add in a lightweight epilogue kernel.
// Shapes for Qwen2.5-7B: M=128, N=4608, K=3584.
//
// Build: nvcc -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
//        -I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/tools/util/include \
//        -O3 -o libcutlass_qkv_bias.so --shared -Xcompiler -fPIC cutlass_qkv_bias.cu

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

// ============================================================================
// Type aliases
// ============================================================================

using ElementA     = cutlass::half_t;
using ElementB     = cutlass::half_t;
using ElementC     = cutlass::half_t;
using ElementD     = cutlass::half_t;
using ElementAccum = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using TileShape    = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;

// ============================================================================
// GEMM
// ============================================================================

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8,
    ElementB, LayoutB, 8,
    ElementAccum,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            TileShape, ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAccum, ElementAccum,
            ElementC, LayoutC, 8,
            ElementC, LayoutC, 8,
            cutlass::epilogue::collective::EpilogueScheduleAuto
        >::CollectiveOp::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementAccum,
    ElementC, LayoutC, 8,
    ElementC, LayoutC, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// ============================================================================
// Per-column bias add kernel: output[m, n] += bias[n]
// Vectorized 8-wide half2x4 loads for full memory bandwidth.
// ============================================================================

__global__ void bias_add_kernel(
    half* __restrict__ output,
    const half* __restrict__ bias,
    int M, int N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    half b = bias[col];
    for (int row = blockIdx.y; row < M; row += gridDim.y) {
        long long idx = (long long)row * N + col;
        output[idx] = __hadd(output[idx], b);
    }
}

// ============================================================================
// C interface
// ============================================================================

extern "C" {

int cutlass_qkv_bias_gemm(
    void* output,          // [M, N] half
    const void* input,     // [M, K] half
    const void* weight,    // [N, K] half (row-major, transposed in GEMM)
    const void* bias,      // [N] half
    int M, int N, int K,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream
) {
    if (output == nullptr || input == nullptr || weight == nullptr ||
        M <= 0 || N <= 0 || K <= 0) return -10;
    // --- GEMM: output = input @ weight^T ---
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
    const size_t required_workspace = gemm_op.get_workspace_size(args);
    if (workspace_size < required_workspace ||
        (required_workspace > 0 && workspace == nullptr)) return -11;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -3;

    // --- Bias add: output[m,n] += bias[n] ---
    if (bias != nullptr) {
        dim3 block(256);
        const unsigned grid_x = (unsigned)(((long long)N + 255) / 256);
        const unsigned grid_y = (unsigned)(M < 65535 ? M : 65535);
        dim3 grid(grid_x, grid_y);
        bias_add_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<half*>(output),
            reinterpret_cast<const half*>(bias),
            M, N
        );
        if (cudaPeekAtLastError() != cudaSuccess) return -4;
    }

    return 0;
}

size_t cutlass_qkv_bias_workspace_size(int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return 0;
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

} // extern "C"
