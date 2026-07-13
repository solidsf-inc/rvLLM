// Copyright 2026 m0at
// SPDX-License-Identifier: Apache-2.0

#include "common.metal"
#include "float8.metal"

using namespace metal;

constant uint RVLLM_GEMV_THREADS = 256;
constant uint RVLLM_GEMV_SIMD_SIZE = 32;
constant uint RVLLM_GEMV_SIMDGROUPS = 8;

struct RvllmGemvParams {
  uint rows;
  uint cols;
  uint scale_layout;
  uint scale_stride;
};

struct RvllmBf16ArgmaxGemvParams {
  uint rows;
  uint cols;
  uint rows_per_group;
  uint partial_count;
};

struct RvllmLmHeadNllParams {
  uint rows;
  float softcap;
  float pad0;
  float pad1;
};

struct RvllmGeluMulParams {
  uint rows;
  uint pad0;
  uint pad1;
  uint pad2;
};

struct RvllmManyGemvSpecParams {
  uint rows;
  uint cols;
  uint row_offset;
  uint scale_layout;
  uint scale_stride;
  uint pad0;
  uint pad1;
  uint pad2;
};

struct RvllmManyGemvParams {
  uint total_rows;
  uint spec_count;
  uint pad0;
  uint pad1;
  RvllmManyGemvSpecParams specs[3];
};

struct RvllmArgmaxResult {
  uint index;
  float score;
};

struct RvllmHostAttentionParams {
  uint num_heads;
  uint num_kv_heads;
  uint head_dim;
  uint kv_dim;
  uint len;
  float scale;
  uint pad1;
  uint pad2;
};

inline uint rvllm_scale_index(uint row, uint layout, uint stride) {
  uint idx = 0;
  idx = select(idx, row * stride, layout == 0);
  idx = select(idx, (row / 128) * stride, layout == 1);
  return idx;
}

inline bool rvllm_argmax_better(float score, uint index, float best_score,
                                uint best_index) {
  return (score > best_score) || ((score == best_score) && (index < best_index));
}

inline float rvllm_softcap_logit(float x, float softcap) {
  const bool enabled = softcap > 0.0f;
  const float denom = select(1.0f, softcap, enabled);
  const float capped = softcap * tanh(x / denom);
  return select(x, capped, enabled);
}

inline float rvllm_threadgroup_sum(float value, threadgroup float *scratch,
                                   uint tid, uint simd_gid, uint simd_lid) {
  for (uint offset = RVLLM_GEMV_SIMD_SIZE >> 1; offset > 0; offset >>= 1) {
    value += simd_shuffle_xor(value, offset);
  }
  if (simd_lid == 0) {
    scratch[simd_gid] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < RVLLM_GEMV_SIMD_SIZE) {
    value = (tid < RVLLM_GEMV_SIMDGROUPS) ? scratch[tid] : 0.0f;
    for (uint offset = RVLLM_GEMV_SIMD_SIZE >> 1; offset > 0; offset >>= 1) {
      value += simd_shuffle_xor(value, offset);
    }
    if (tid == 0) {
      scratch[0] = value;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch[0];
}

inline float rvllm_threadgroup_max(float value, threadgroup float *scratch,
                                   uint tid, uint simd_gid, uint simd_lid) {
  for (uint offset = RVLLM_GEMV_SIMD_SIZE >> 1; offset > 0; offset >>= 1) {
    value = max(value, simd_shuffle_xor(value, offset));
  }
  if (simd_lid == 0) {
    scratch[simd_gid] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < RVLLM_GEMV_SIMD_SIZE) {
    value = (tid < RVLLM_GEMV_SIMDGROUPS) ? scratch[tid] : -INFINITY;
    for (uint offset = RVLLM_GEMV_SIMD_SIZE >> 1; offset > 0; offset >>= 1) {
      value = max(value, simd_shuffle_xor(value, offset));
    }
    if (tid == 0) {
      scratch[0] = value;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch[0];
}

inline RvllmArgmaxResult
rvllm_threadgroup_argmax(float score, uint index, threadgroup float *scores,
                         threadgroup uint *indices, uint tid, uint simd_gid,
                         uint simd_lid) {
  for (uint offset = RVLLM_GEMV_SIMD_SIZE >> 1; offset > 0; offset >>= 1) {
    const float other_score = simd_shuffle_xor(score, offset);
    const uint other_index = simd_shuffle_xor(index, offset);
    const bool take = rvllm_argmax_better(other_score, other_index, score, index);
    score = select(score, other_score, take);
    index = select(index, other_index, take);
  }
  if (simd_lid == 0) {
    scores[simd_gid] = score;
    indices[simd_gid] = index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < RVLLM_GEMV_SIMD_SIZE) {
    score = (tid < RVLLM_GEMV_SIMDGROUPS) ? scores[tid] : -INFINITY;
    index = (tid < RVLLM_GEMV_SIMDGROUPS) ? indices[tid] : 0;
    for (uint offset = RVLLM_GEMV_SIMD_SIZE >> 1; offset > 0; offset >>= 1) {
      const float other_score = simd_shuffle_xor(score, offset);
      const uint other_index = simd_shuffle_xor(index, offset);
      const bool take =
          rvllm_argmax_better(other_score, other_index, score, index);
      score = select(score, other_score, take);
      index = select(index, other_index, take);
    }
    if (tid == 0) {
      scores[0] = score;
      indices[0] = index;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  RvllmArgmaxResult result;
  result.index = indices[0];
  result.score = scores[0];
  return result;
}

template <typename ScaleT>
inline void rvllm_fp8_many_gemv_row(
    device const uchar *weight,
    device const ScaleT *scale,
    device const float *x,
    device float *out,
    constant RvllmManyGemvSpecParams &spec,
    uint global_row,
    uint local_row,
    uint tid,
    uint simd_gid,
    uint simd_lid,
    threadgroup float *partial) {
  float acc = 0.0f;
  const uint base = local_row * spec.cols;
  for (uint c = tid; c < spec.cols; c += RVLLM_GEMV_THREADS) {
    acc += fp8_e4m3_to_float(weight[base + c]) * x[c];
  }
  acc = rvllm_threadgroup_sum(acc, partial, tid, simd_gid, simd_lid);

  if (tid == 0) {
    const uint sidx =
        rvllm_scale_index(local_row, spec.scale_layout, spec.scale_stride);
    out[global_row] = acc * float(scale[sidx]);
  }
}

template <typename ScaleT>
inline void rvllm_fp8_many_gemv(
    device const uchar *weight0,
    device const ScaleT *scale0,
    device const uchar *weight1,
    device const ScaleT *scale1,
    device const uchar *weight2,
    device const ScaleT *scale2,
    device const float *x,
    device float *out,
    constant RvllmManyGemvParams &p,
    uint row,
    uint tid,
    uint simd_gid,
    uint simd_lid,
    threadgroup float *partial) {
  const uint row1 = p.specs[1].row_offset;
  const uint row2 = p.specs[2].row_offset;
  if (row < row1) {
    rvllm_fp8_many_gemv_row(weight0, scale0, x, out, p.specs[0], row, row,
                            tid, simd_gid, simd_lid, partial);
  } else if (row < row2) {
    rvllm_fp8_many_gemv_row(weight1, scale1, x, out, p.specs[1], row,
                            row - row1, tid, simd_gid, simd_lid, partial);
  } else {
    rvllm_fp8_many_gemv_row(weight2, scale2, x, out, p.specs[2], row,
                            row - row2, tid, simd_gid, simd_lid, partial);
  }
}

[[host_name("rvllm_fp8_gemv_bf16scale_f32")]] kernel void
rvllm_fp8_gemv_bf16scale_f32(
    device const uchar *weight [[buffer(0)]],
    device const bfloat16_t *scale [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant RvllmGemvParams &p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint row = tg.x;
  if (row >= p.rows) {
    return;
  }

  threadgroup float partial[RVLLM_GEMV_SIMDGROUPS];
  float acc = 0.0f;
  const uint base = row * p.cols;
  for (uint c = tid; c < p.cols; c += RVLLM_GEMV_THREADS) {
    acc += fp8_e4m3_to_float(weight[base + c]) * x[c];
  }
  acc = rvllm_threadgroup_sum(acc, partial, tid, simd_gid, simd_lid);

  if (tid == 0) {
    const uint sidx = rvllm_scale_index(row, p.scale_layout, p.scale_stride);
    out[row] = acc * float(scale[sidx]);
  }
}

[[host_name("rvllm_fp8_gemv_f32scale_f32")]] kernel void
rvllm_fp8_gemv_f32scale_f32(
    device const uchar *weight [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant RvllmGemvParams &p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint row = tg.x;
  if (row >= p.rows) {
    return;
  }

  threadgroup float partial[RVLLM_GEMV_SIMDGROUPS];
  float acc = 0.0f;
  const uint base = row * p.cols;
  for (uint c = tid; c < p.cols; c += RVLLM_GEMV_THREADS) {
    acc += fp8_e4m3_to_float(weight[base + c]) * x[c];
  }
  acc = rvllm_threadgroup_sum(acc, partial, tid, simd_gid, simd_lid);

  if (tid == 0) {
    const uint sidx = rvllm_scale_index(row, p.scale_layout, p.scale_stride);
    out[row] = acc * scale[sidx];
  }
}

[[host_name("rvllm_fp8_many_gemv_bf16scale_f32")]] kernel void
rvllm_fp8_many_gemv_bf16scale_f32(
    device const uchar *weight0 [[buffer(0)]],
    device const bfloat16_t *scale0 [[buffer(1)]],
    device const uchar *weight1 [[buffer(2)]],
    device const bfloat16_t *scale1 [[buffer(3)]],
    device const uchar *weight2 [[buffer(4)]],
    device const bfloat16_t *scale2 [[buffer(5)]],
    device const float *x [[buffer(6)]],
    device float *out [[buffer(7)]],
    constant RvllmManyGemvParams &p [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint row = tg.x;
  if (row >= p.total_rows) {
    return;
  }

  threadgroup float partial[RVLLM_GEMV_SIMDGROUPS];
  rvllm_fp8_many_gemv(weight0, scale0, weight1, scale1, weight2, scale2, x,
                      out, p, row, tid, simd_gid, simd_lid, partial);
}

[[host_name("rvllm_fp8_many_gemv_f32scale_f32")]] kernel void
rvllm_fp8_many_gemv_f32scale_f32(
    device const uchar *weight0 [[buffer(0)]],
    device const float *scale0 [[buffer(1)]],
    device const uchar *weight1 [[buffer(2)]],
    device const float *scale1 [[buffer(3)]],
    device const uchar *weight2 [[buffer(4)]],
    device const float *scale2 [[buffer(5)]],
    device const float *x [[buffer(6)]],
    device float *out [[buffer(7)]],
    constant RvllmManyGemvParams &p [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint row = tg.x;
  if (row >= p.total_rows) {
    return;
  }

  threadgroup float partial[RVLLM_GEMV_SIMDGROUPS];
  rvllm_fp8_many_gemv(weight0, scale0, weight1, scale1, weight2, scale2, x,
                      out, p, row, tid, simd_gid, simd_lid, partial);
}

[[host_name("rvllm_bf16_gemv_f32")]] kernel void rvllm_bf16_gemv_f32(
    device const bfloat16_t *weight [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant RvllmGemvParams &p [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint row = tg.x;
  if (row >= p.rows) {
    return;
  }

  threadgroup float partial[RVLLM_GEMV_SIMDGROUPS];
  float acc = 0.0f;
  const uint base = row * p.cols;
  for (uint c = tid; c < p.cols; c += RVLLM_GEMV_THREADS) {
    acc += float(weight[base + c]) * x[c];
  }
  acc = rvllm_threadgroup_sum(acc, partial, tid, simd_gid, simd_lid);

  if (tid == 0) {
    out[row] = acc;
  }
}

[[host_name("rvllm_gelu_tanh_mul_f32")]] kernel void rvllm_gelu_tanh_mul_f32(
    device float *gate_up [[buffer(0)]],
    constant RvllmGeluMulParams &p [[buffer(1)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint i = tg.x * RVLLM_GEMV_THREADS + tid;
  if (i >= p.rows) {
    return;
  }
  const float g = gate_up[i];
  const float u = gate_up[p.rows + i];
  const float x3 = g * g * g;
  const float y = 0.7978846f * (g + 0.044715f * x3);
  float gelu = 0.5f * g * (1.0f + tanh(y));
  gelu = select(gelu, g, g > 10.0f);
  gelu = select(gelu, 0.0f, g < -10.0f);
  gate_up[i] = gelu * u;
}

template <typename ScaleT>
inline void rvllm_fp8_gelu_down_row(device const uchar *weight,
                                    device const ScaleT *scale,
                                    device const float *gate_up,
                                    device float *out,
                                    constant RvllmGemvParams &p, uint row,
                                    uint tid, uint simd_gid, uint simd_lid,
                                    threadgroup float *partial) {
  float acc = 0.0f;
  const uint base = row * p.cols;
  for (uint c = tid; c < p.cols; c += RVLLM_GEMV_THREADS) {
    const float g = gate_up[c];
    const float u = gate_up[p.cols + c];
    const float x3 = g * g * g;
    const float y = 0.7978846f * (g + 0.044715f * x3);
    float gelu = 0.5f * g * (1.0f + tanh(y));
    gelu = select(gelu, g, g > 10.0f);
    gelu = select(gelu, 0.0f, g < -10.0f);
    acc += fp8_e4m3_to_float(weight[base + c]) * gelu * u;
  }
  acc = rvllm_threadgroup_sum(acc, partial, tid, simd_gid, simd_lid);

  if (tid == 0) {
    const uint sidx = rvllm_scale_index(row, p.scale_layout, p.scale_stride);
    out[row] = acc * float(scale[sidx]);
  }
}

[[host_name("rvllm_fp8_gelu_down_bf16scale_f32")]] kernel void
rvllm_fp8_gelu_down_bf16scale_f32(
    device const uchar *weight [[buffer(0)]],
    device const bfloat16_t *scale [[buffer(1)]],
    device const float *gate_up [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant RvllmGemvParams &p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint row = tg.x;
  if (row >= p.rows) {
    return;
  }

  threadgroup float partial[RVLLM_GEMV_SIMDGROUPS];
  rvllm_fp8_gelu_down_row(weight, scale, gate_up, out, p, row, tid, simd_gid,
                          simd_lid, partial);
}

[[host_name("rvllm_fp8_gelu_down_f32scale_f32")]] kernel void
rvllm_fp8_gelu_down_f32scale_f32(
    device const uchar *weight [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    device const float *gate_up [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant RvllmGemvParams &p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint row = tg.x;
  if (row >= p.rows) {
    return;
  }

  threadgroup float partial[RVLLM_GEMV_SIMDGROUPS];
  rvllm_fp8_gelu_down_row(weight, scale, gate_up, out, p, row, tid, simd_gid,
                          simd_lid, partial);
}

[[host_name("rvllm_host_f32_attention")]] kernel void rvllm_host_f32_attention(
    device const float *q [[buffer(0)]],
    device const float *k_cache [[buffer(1)]],
    device const float *v_cache [[buffer(2)]],
    device const uint *slots [[buffer(3)]], device float *out [[buffer(4)]],
    constant RvllmHostAttentionParams &p [[buffer(5)]],
    threadgroup float *scores [[threadgroup(0)]],
    uint head_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint group = p.num_heads / p.num_kv_heads;
  const uint kv_head = head_idx / group;
  const uint q_base = head_idx * p.head_dim;
  const uint red_base = p.len;

  float local_max = -INFINITY;
  for (uint j = tid; j < p.len; j += RVLLM_GEMV_THREADS) {
    const uint slot = slots[j];
    const uint k_base = slot * p.kv_dim + kv_head * p.head_dim;
    float acc = 0.0f;
    for (uint i = 0; i < p.head_dim; ++i) {
      acc += q[q_base + i] * k_cache[k_base + i];
    }
    acc *= p.scale;
    scores[j] = acc;
    local_max = max(local_max, acc);
  }

  const float max_score = rvllm_threadgroup_max(
      local_max, scores + red_base, tid, simd_gid, simd_lid);

  float local_sum = 0.0f;
  for (uint j = tid; j < p.len; j += RVLLM_GEMV_THREADS) {
    const float e = exp(scores[j] - max_score);
    scores[j] = e;
    local_sum += e;
  }
  const float denom =
      rvllm_threadgroup_sum(local_sum, scores + red_base, tid, simd_gid, simd_lid);

  for (uint i = tid; i < p.head_dim; i += RVLLM_GEMV_THREADS) {
    float acc = 0.0f;
    for (uint j = 0; j < p.len; ++j) {
      const uint slot = slots[j];
      const uint v_base = slot * p.kv_dim + kv_head * p.head_dim;
      acc += (scores[j] / denom) * v_cache[v_base + i];
    }
    out[q_base + i] = acc;
  }
}

[[host_name("rvllm_bf16_lm_head_argmax_gemv")]] kernel void
rvllm_bf16_lm_head_argmax_gemv(
    device const bfloat16_t *weight [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device RvllmArgmaxResult *partials [[buffer(2)]],
    constant RvllmBf16ArgmaxGemvParams &p [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint group = tg.x;
  const uint row_start = group * p.rows_per_group;
  if (row_start >= p.rows) {
    return;
  }
  const uint row_end = min(row_start + p.rows_per_group, p.rows);

  threadgroup float partial[RVLLM_GEMV_THREADS];
  float best_score = -INFINITY;
  uint best_index = 0;

  for (uint row = row_start; row < row_end; ++row) {
    float acc = 0.0f;
    const uint base = row * p.cols;
    for (uint c = tid; c < p.cols; c += RVLLM_GEMV_THREADS) {
      acc += float(weight[base + c]) * x[c];
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = RVLLM_GEMV_THREADS >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        partial[tid] += partial[tid + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
      const float score = partial[0];
      const bool take = rvllm_argmax_better(score, row, best_score, best_index);
      best_score = select(best_score, score, take);
      best_index = select(best_index, row, take);
    }
  }

  if (tid == 0) {
    partials[group].index = best_index;
    partials[group].score = best_score;
  }
}

[[host_name("rvllm_lm_head_logsumexp_f32")]] kernel void
rvllm_lm_head_logsumexp_f32(
    device const float *logits [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant RvllmLmHeadNllParams &p [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[RVLLM_GEMV_THREADS];

  float local_max = -INFINITY;
  for (uint row = tid; row < p.rows; row += RVLLM_GEMV_THREADS) {
    local_max = max(local_max, rvllm_softcap_logit(logits[row], p.softcap));
  }
  scratch[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = RVLLM_GEMV_THREADS >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] = max(scratch[tid], scratch[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float global_max = scratch[0];

  float local_sum = 0.0f;
  for (uint row = tid; row < p.rows; row += RVLLM_GEMV_THREADS) {
    local_sum += exp(rvllm_softcap_logit(logits[row], p.softcap) - global_max);
  }
  scratch[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = RVLLM_GEMV_THREADS >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    out[0] = global_max + log(scratch[0]);
  }
}

[[host_name("rvllm_lm_head_argmax_reduce")]] kernel void
rvllm_lm_head_argmax_reduce(
    device const RvllmArgmaxResult *partials [[buffer(0)]],
    device RvllmArgmaxResult *out [[buffer(1)]],
    constant RvllmBf16ArgmaxGemvParams &p [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scores[RVLLM_GEMV_THREADS];
  threadgroup uint indices[RVLLM_GEMV_THREADS];

  float best_score = -INFINITY;
  uint best_index = 0;
  for (uint i = tid; i < p.partial_count; i += RVLLM_GEMV_THREADS) {
    const float score = partials[i].score;
    const uint index = partials[i].index;
    const bool take = rvllm_argmax_better(score, index, best_score, best_index);
    best_score = select(best_score, score, take);
    best_index = select(best_index, index, take);
  }

  scores[tid] = best_score;
  indices[tid] = best_index;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = RVLLM_GEMV_THREADS >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float score = scores[tid + stride];
      const uint index = indices[tid + stride];
      const bool take =
          rvllm_argmax_better(score, index, scores[tid], indices[tid]);
      scores[tid] = select(scores[tid], score, take);
      indices[tid] = select(indices[tid], index, take);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    out[0].index = indices[0];
    out[0].score = scores[0];
  }
}
