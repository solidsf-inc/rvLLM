// Adapted from mistral.rs revision 31c13eb4587d3e4a5204870c98b70c05a1e5c943:
// mistralrs-quant/src/metal_kernels/scalar_fp8.metal
// License: MIT (mistralrs) + original Apache-2.0 (Apple/MLX where applicable)
// Modifications: common include and explicit per-channel scaling.

#include "common.metal"
#include "float8.metal"
#include <metal_stdlib>

using namespace metal;

// ============================================================================
// FP8 E4M3 to other dtypes (per-element conversion)
// ============================================================================

template <typename OutT>
kernel void fp8_to_dtype_kernel(device const uchar *input [[buffer(0)]],
                                device OutT *output [[buffer(1)]],
                                constant uint &num_elements [[buffer(2)]],
                                uint idx [[thread_position_in_grid]]) {
  if (idx >= num_elements)
    return;
  float val = fp8_e4m3_to_float(input[idx]);
  output[idx] = OutT(val);
}

// ============================================================================
// Other dtypes to FP8 E4M3 (per-element conversion with clamping)
// ============================================================================

template <typename InT>
kernel void dtype_to_fp8_kernel(device const InT *input [[buffer(0)]],
                                device uchar *output [[buffer(1)]],
                                constant uint &num_elements [[buffer(2)]],
                                uint idx [[thread_position_in_grid]]) {
  if (idx >= num_elements)
    return;
  float val = float(input[idx]);
  // Clamp to FP8 E4M3 range: [-448, 448]
  val = clamp(val, -448.0f, 448.0f);
  output[idx] = float_to_fp8_e4m3(val);
}

// ============================================================================
// Per-channel FP8 dequantization: output = fp8_weight * scale_inv[row]
// ============================================================================

template <typename OutT>
kernel void
fp8_perchannel_dequant_kernel(device const uchar *weight [[buffer(0)]],
                              device const float *scale_inv [[buffer(1)]],
                              device OutT *output [[buffer(2)]],
                              constant uint &num_elements [[buffer(3)]],
                              constant uint &row_stride [[buffer(4)]],
                              constant uint &num_channels [[buffer(5)]],
                              uint idx [[thread_position_in_grid]]) {
  if (idx >= num_elements)
    return;
  float w_val = fp8_e4m3_to_float(weight[idx]);
  uint channel = idx / row_stride;
  if (channel >= num_channels)
    return;
  float scaled = w_val * scale_inv[channel];
  output[idx] = OutT(scaled);
}

// ============================================================================
// Instantiate kernels for all supported output types
// ============================================================================

#define instantiate_fp8_to_dtype(type)                                         \
  template [[host_name("fp8_to_dtype_" #type)]] [[kernel]] void                \
  fp8_to_dtype_kernel<type>(device const uchar *input [[buffer(0)]],           \
                            device type *output [[buffer(1)]],                 \
                            constant uint &num_elements [[buffer(2)]],         \
                            uint idx [[thread_position_in_grid]]);

instantiate_fp8_to_dtype(float);
instantiate_fp8_to_dtype(half);
instantiate_fp8_to_dtype(bfloat16_t);

#define instantiate_dtype_to_fp8(type)                                         \
  template [[host_name("dtype_to_fp8_" #type)]] [[kernel]] void                \
  dtype_to_fp8_kernel<type>(device const type *input [[buffer(0)]],            \
                            device uchar *output [[buffer(1)]],                \
                            constant uint &num_elements [[buffer(2)]],         \
                            uint idx [[thread_position_in_grid]]);

instantiate_dtype_to_fp8(float);
instantiate_dtype_to_fp8(half);
instantiate_dtype_to_fp8(bfloat16_t);

#define instantiate_fp8_perchannel_dequant(type)                               \
  template [[host_name("fp8_perchannel_dequant_" #type)]] [[kernel]] void      \
  fp8_perchannel_dequant_kernel<type>(                                         \
      device const uchar *weight [[buffer(0)]],                                \
      device const float *scale_inv [[buffer(1)]],                             \
      device type *output [[buffer(2)]],                                       \
      constant uint &num_elements [[buffer(3)]],                               \
      constant uint &row_stride [[buffer(4)]],                                 \
      constant uint &num_channels [[buffer(5)]],                               \
      uint idx [[thread_position_in_grid]]);

instantiate_fp8_perchannel_dequant(float);
instantiate_fp8_perchannel_dequant(half);
instantiate_fp8_perchannel_dequant(bfloat16_t);

// ============================================================================
// Vector FP8 dequantization: output[i] = fp8_weight[i] * scale[i / VECTOR_SIZE]
// Each group of 128 elements shares one scale
// ============================================================================

#define VECTOR_SIZE 128

template <typename OutT>
kernel void fp8_vector_dequant_kernel(device const uchar *weight [[buffer(0)]],
                                      device const float *scale [[buffer(1)]],
                                      device OutT *output [[buffer(2)]],
                                      constant uint &num_elements [[buffer(3)]],
                                      uint idx [[thread_position_in_grid]]) {
  if (idx >= num_elements)
    return;
  uint vector_idx = idx / VECTOR_SIZE;
  float w_val = fp8_e4m3_to_float(weight[idx]);
  float scaled = w_val * scale[vector_idx];
  output[idx] = OutT(scaled);
}

#define instantiate_fp8_vector_dequant(type)                                   \
  template [[host_name("fp8_vector_dequant_" #type)]] [[kernel]] void          \
  fp8_vector_dequant_kernel<type>(device const uchar *weight [[buffer(0)]],    \
                                  device const float *scale [[buffer(1)]],     \
                                  device type *output [[buffer(2)]],           \
                                  constant uint &num_elements [[buffer(3)]],   \
                                  uint idx [[thread_position_in_grid]]);

instantiate_fp8_vector_dequant(float);
instantiate_fp8_vector_dequant(half);
instantiate_fp8_vector_dequant(bfloat16_t);
