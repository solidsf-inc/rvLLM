// Post-w4a8-GEMM per-token ACTIVATION dequant for the INT4 decoder GEMMs.
//
// The w4a8 kernel computes  D = alpha * (A_fp8 @ B_dequant)  with a SCALAR
// `alpha` and applies only the WEIGHT group scales internally; it never sees
// the per-token ACTIVATION scale. The FP8 cuBLASLt path applies that scale via
// the A_SCALE pointer, but w4a8 has no per-token scale input (only scalar
// alpha, which cannot represent M distinct per-token scales for M>1). So each
// w4a8 output row m must be multiplied by the activation's per-token
// `scale[m]` to undo the FP8 quantization of the activation
// (a_fp8 = round(a_true / scale[m])). Without it the decoder GEMM outputs are
// ~1/scale (~100x) too large and overflow f16 -> NaN.
//
// Grid:  (num_rows, 1, 1)
// Block: (min(n, 1024), 1, 1)

#include <cuda_fp16.h>
#include <math_constants.h>

extern "C"
__global__ void scale_rows_f16_pertoken_kernel(
    __half* __restrict__ data,       // [num_rows, n] row-major, in-place
    const float* __restrict__ scale, // [num_rows] per-token f32 scale
    int n
) {
    const int row = blockIdx.x;
    if (data == nullptr || scale == nullptr || n <= 0 || blockDim.x == 0 ||
        blockDim.x > 1024 || blockDim.y != 1 || blockDim.z != 1 ||
        gridDim.y != 1 || gridDim.z != 1) return;
    const float s = scale[row];

    __half* row_ptr = data + (long long)row * (long long)n;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = __half2float(row_ptr[i]);
        row_ptr[i] = __float2half(isfinite(s) && s > 0.0f ? v * s : CUDART_NAN_F);
    }
}
