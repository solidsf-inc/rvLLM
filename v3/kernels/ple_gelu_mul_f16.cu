#include <cuda_fp16.h>
#include <math.h>

extern "C" __global__ void __launch_bounds__(256)
ple_gelu_mul_f16_kernel(
    __half* __restrict__ gate,
    const __half* __restrict__ per_layer_inputs,
    int layer_idx,
    int num_layers,
    int ple_dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int gate_base = row * ple_dim;
    const int ple_base = (row * num_layers + layer_idx) * ple_dim;

    for (int i = tid; i < ple_dim; i += blockDim.x) {
        float g = __half2float(gate[gate_base + i]);
        float g3 = g * g * g;
        float inner = 0.7978845608f * (g + 0.044715f * g3);
        float gelu = 0.5f * g * (1.0f + tanhf(inner));
        float p = __half2float(per_layer_inputs[ple_base + i]);
        gate[gate_base + i] = __float2half(gelu * p);
    }
}
