// GELU(tanh)(gate) * up -> F16 output (no FP8 quantization).
// Input:  gate_up [num_tokens, 2 * intermediate] f16 (gate || up interleaved)
// Output: out_f16 [num_tokens, intermediate] f16
#include <cuda_fp16.h>
#include <math.h>

extern "C" __global__ void __launch_bounds__(1024)
fused_gelu_mul_f16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ gate_up,
    int intermediate
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int gate_offset = row * 2 * intermediate;
    const int up_offset = gate_offset + intermediate;
    const int out_offset = row * intermediate;

    for (int i = tid; i < intermediate; i += stride) {
        float g = __half2float(gate_up[gate_offset + i]);
        float u = __half2float(gate_up[up_offset + i]);
        // GELU(tanh) approximation
        float g3 = g * g * g;
        float inner = 0.7978845608f * (g + 0.044715f * g3);
        float gelu = 0.5f * g * (1.0f + tanhf(inner));
        output[out_offset + i] = __float2half(gelu * u);
    }
}
