// Non-production conformance probe for the shared-memory fragment packers.
// The host loads A / B
// as flat row-major / col-major FP8 tiles into smem, the kernel
// calls the packing helpers on each lane, issues one MMA, and writes
// the D tile back to gmem via the smem unpacker.
//
// Results must be checked against an independent FP64 reference.

#include <cstdint>
#include <cmath>

#include "fp8_mma_frag_pack.cuh"

extern "C"
__global__ void fp8_mma_smem_probe_kernel(
    const unsigned char* __restrict__ a_tile,  // [16][32] row-major
    const unsigned char* __restrict__ b_tile,  // [8][32]  col-major ([N][K])
    float*               __restrict__ d_out    // [16][8]  row-major
) {
    if (d_out == nullptr) return;
    const bool valid_launch = a_tile != nullptr && b_tile != nullptr &&
        blockDim.x == 32 && blockDim.y == 1 && blockDim.z == 1 &&
        gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1 &&
        reinterpret_cast<uintptr_t>(a_tile) % alignof(uint32_t) == 0 &&
        reinterpret_cast<uintptr_t>(b_tile) % alignof(uint32_t) == 0 &&
        reinterpret_cast<uintptr_t>(d_out) % alignof(float) == 0;
    if (!valid_launch) {
        for (int i = threadIdx.x; i < 16 * 8; i += blockDim.x) d_out[i] = nanf("");
        return;
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    constexpr int M = 16, N = 8, K = 32;

    extern __shared__ unsigned char smem_raw[];
    unsigned char* s_a = smem_raw;                     // [M][K] = 512 B
    unsigned char* s_b = s_a + M * K;                  // [N][K] = 256 B
    float*         s_d = reinterpret_cast<float*>(s_b + N * K); // [M][N]

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    constexpr int THREADS = 32; // one warp

    // Cooperative load A / B into smem. Using u32 copies to match the
    // alignment the packers will reuse.
    for (int i = tid; i < (M * K) / 4; i += THREADS) {
        reinterpret_cast<uint32_t*>(s_a)[i] =
            reinterpret_cast<const uint32_t*>(a_tile)[i];
    }
    for (int i = tid; i < (N * K) / 4; i += THREADS) {
        reinterpret_cast<uint32_t*>(s_b)[i] =
            reinterpret_cast<const uint32_t*>(b_tile)[i];
    }
    __syncthreads();

    // Pack per-lane fragments straight out of smem.
    uint32_t a_frag[4];
    uint32_t b_frag[2];
    float    d_frag[4]; rvllm::zero_mma_d_frag(d_frag);
    rvllm::pack_a_frag_row_major_m16k32(s_a, /*stride=*/K, a_frag, lane);
    rvllm::pack_b_frag_col_major_n8k32 (s_b, /*stride=*/K, b_frag, lane);

    // One MMA, accumulate into d_frag.
    rvllm::mma_m16n8k32_e4m3_e4m3_f32(d_frag, a_frag, b_frag);

    // Unpack D back into a row-major [M][N] tile in smem, then dump.
    rvllm::unpack_d_frag_to_smem_m16n8(
        s_d, /*stride_bytes=*/N * sizeof(float), d_frag, lane);
    __syncthreads();

    for (int i = tid; i < M * N; i += THREADS) {
        d_out[i] = s_d[i];
    }
#else
    (void)a_tile; (void)b_tile;
    for (int i = threadIdx.x; i < 16 * 8; i += blockDim.x) d_out[i] = nanf("");
#endif
}
