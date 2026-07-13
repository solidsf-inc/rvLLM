// Post-GEMM per-channel scale: data[m, n] *= scale[n] for an M x N f32 matrix.
// Used when cuBLASLt OUTER_VEC_32F is unavailable (CUDA < 12.8).

extern "C" __launch_bounds__(256) __global__ void scale_cols_f32_kernel(
    float* __restrict__ data,        // [M, N] row-major, in-place
    const float* __restrict__ scale, // [N]
    int M,
    int N
) {
    if (data == nullptr || scale == nullptr || M <= 0 || N <= 0) return;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)M * N;
    if (idx < total) {
        int n = (int)(idx % N);
        data[idx] *= scale[n];
    }
}
