// Copyright 2026 m0at <47344131+m0at@users.noreply.github.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Portions of this file are adapted from mistral.rs revision
// 31c13eb4587d3e4a5204870c98b70c05a1e5c943:
// mistralrs-paged-attn/src/metal/kernels/mod.rs::call_paged_attention_v1
// licensed under the MIT License, Copyright (c) 2024 Eric Buehler. See
// LICENSES/MIT-mistralrs at the workspace root.

//! Rust wrapper for the paged-attention V1 (single-pass) Metal kernel.
//!
//! Dispatches `paged_attention_<T>_cache_<CT>_hs<HD>_bs<BS>_nt256_nsl32_ps0`
//! from `pagedattention.metal`. Specialization is
//! done by encoding the head_size / block_size / dtype combo into the
//! function name; the four optional feature flags
//! (`use_partitioning`, `use_alibi`, `use_fp8_scales`, `use_sinks`) are
//! function constants at indices 10/20/30/40.
//! V1 fails before dispatch when its dynamic shared-memory requirement exceeds
//! the selected device. The partitioned V2 path is not exposed yet.

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::ffi::c_void;

use metal::{Buffer, ComputeCommandEncoderRef, FunctionConstantValues, MTLDataType, MTLSize};

use rvllm_core::DType;

use crate::device::MetalKernelError;
use crate::kernels::MetalKernels;

const NUM_THREADS: u32 = 256;
const NUM_SIMD_LANES: u32 = 32;

/// Map an `rvllm-core` dtype to the metal-side type token used in
/// kernel host names (matches the `instantiate_paged_attention_v1`
/// macro invocations at the bottom of `pagedattention.metal`).
fn dtype_token(dt: DType) -> Result<&'static str, MetalKernelError> {
    match dt {
        DType::F32 => Ok("float"),
        DType::F16 => Ok("half"),
        DType::Bf16 => Ok("bfloat16_t"),
        other => Err(MetalKernelError::FeatureNotAvailable(match other {
            DType::Fp8E4M3 => "paged-attention: FP8 cache scales are not exposed by this wrapper",
            DType::Fp8E5M2 => "paged-attention: Fp8E5M2 not instantiated in pagedattention.metal",
            DType::F64 => "paged-attention: F64 unsupported",
            DType::I32 => "paged-attention: I32 unsupported",
            DType::I64 => "paged-attention: I64 unsupported",
            DType::U32 => "paged-attention: U32 unsupported",
            DType::U8 => "paged-attention: U8 unsupported",
            _ => "paged-attention: unsupported dtype",
        })),
    }
}

/// Validate the head-size against the set of compile-time instantiated
/// variants in `pagedattention.metal`.
fn validate_head_size(head_size: u32) -> Result<(), MetalKernelError> {
    match head_size {
        64 | 80 | 96 | 112 | 128 | 192 | 256 | 512 => Ok(()),
        _ => Err(MetalKernelError::InvalidShape(format!(
            "paged-attention: head_size must be one of 64/80/96/112/128/192/256/512, got {head_size}"
        ))),
    }
}

/// Validate the block size against the compile-time instantiations.
fn validate_block_size(block_size: u32) -> Result<(), MetalKernelError> {
    match block_size {
        8 | 16 | 32 => Ok(()),
        _ => Err(MetalKernelError::InvalidShape(format!(
            "paged-attention: block_size must be 8, 16, or 32; got {block_size}"
        ))),
    }
}

/// Build a `FunctionConstantValues` populated with the four optional
/// feature flags. For V1 the values are
/// `use_partitioning=false, use_alibi=false, use_fp8_scales=false,
/// use_sinks=false` — this wrapper does not expose those optionals
/// (alibi/sinks/fp8 will be added in a follow-up slice).
fn v1_constants() -> FunctionConstantValues {
    let constants = FunctionConstantValues::new();
    let f: bool = false;
    let pf = &f as *const bool as *const c_void;
    // Indices match the declarations in pagedattention.metal.
    constants.set_constant_value_at_index(pf, MTLDataType::Bool, 10u64); // use_partitioning
    constants.set_constant_value_at_index(pf, MTLDataType::Bool, 20u64); // use_alibi
    constants.set_constant_value_at_index(pf, MTLDataType::Bool, 30u64); // use_fp8_scales
    constants.set_constant_value_at_index(pf, MTLDataType::Bool, 40u64); // use_sinks
    constants
}

/// Compile (or look up) a pipeline state for `name` specialized with
/// the V1 function-constant set. Bypasses `MetalKernels::pipeline`
/// because that helper does not accept constants.
fn pipeline_with_v1_constants(
    kernels: &MetalKernels,
    name: &str,
) -> Result<metal::ComputePipelineState, MetalKernelError> {
    let constants = v1_constants();
    let function = kernels
        .library()
        .get_function(name, Some(constants))
        .map_err(|e| MetalKernelError::KernelLoadFailed(format!("function `{name}`: {e}")))?;
    // Reach for the underlying device through the kernels' library —
    // metal::Library doesn't expose the device, so we round-trip via the
    // function's `device()` accessor.
    let device = function.device();
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| MetalKernelError::KernelLoadFailed(format!("pipeline `{name}`: {e}")))
}

/// Encode a paged-attention V1 dispatch onto `encoder`.
///
/// Buffer indices and threadgroup memory sizing mirror
/// `mistralrs-paged-attn::call_paged_attention_v1`. Stride scalars
/// (`q_stride`, `kv_block_stride`, `kv_head_stride`,
/// `max_num_blocks_per_seq`) are computed from the surrounding shapes
/// by the caller — this wrapper does not introspect buffer layouts.
///
/// # Errors
/// Returns `MetalKernelError::FeatureNotAvailable` if the requested
/// `output_dtype`/`cache_dtype` combination has not been instantiated
/// in `pagedattention.metal`, or `KernelLoadFailed` if the metallib
/// does not contain the specialized function.
#[allow(clippy::too_many_arguments)]
pub fn call_paged_attention_metal(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    output_dtype: DType,
    cache_dtype: DType,
    head_size: u32,
    block_size: u32,
    q_buf: &Buffer,
    k_cache_buf: &Buffer,
    v_cache_buf: &Buffer,
    out_buf: &Buffer,
    block_tables_buf: &Buffer,
    context_lens_buf: &Buffer,
    num_seqs: u32,
    num_kv_heads: u32,
    num_heads: u32,
    softmax_scale: f32,
    softcapping: Option<f32>,
    max_context_len: u32,
) -> Result<(), MetalKernelError> {
    validate_head_size(head_size)?;
    validate_block_size(block_size)?;
    if num_seqs == 0 || num_heads == 0 || num_kv_heads == 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "paged-attention: num_seqs/num_heads/num_kv_heads must all be > 0 \
             (got num_seqs={num_seqs}, num_heads={num_heads}, num_kv_heads={num_kv_heads})"
        )));
    }
    if num_heads % num_kv_heads != 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "paged-attention: num_heads ({num_heads}) must be a multiple of \
             num_kv_heads ({num_kv_heads}) for GQA"
        )));
    }
    if max_context_len == 0 {
        return Err(MetalKernelError::InvalidShape(
            "paged-attention: max_context_len must be > 0".into(),
        ));
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(MetalKernelError::InvalidShape(
            "paged-attention: softmax_scale must be finite and > 0".into(),
        ));
    }
    if let Some(value) = softcapping {
        if !value.is_finite() || value <= 0.0 {
            return Err(MetalKernelError::InvalidShape(
                "paged-attention: softcapping must be finite and > 0".into(),
            ));
        }
    }
    if cache_dtype != output_dtype || cache_dtype.needs_scale() {
        return Err(MetalKernelError::FeatureNotAvailable(
            "paged-attention V1 requires matching unquantized output/cache dtypes",
        ));
    }

    let t_tok = dtype_token(output_dtype)?;
    let ct_tok = dtype_token(cache_dtype)?;

    // ps0 selects the single-pass specialization; nt256/nsl32 are compiled
    // into the kernel family.
    let name = format!(
        "paged_attention_{t_tok}_cache_{ct_tok}_hs{head_size}_bs{block_size}_nt{NUM_THREADS}_nsl{NUM_SIMD_LANES}_ps0"
    );

    let pipeline = pipeline_with_v1_constants(kernels, &name)?;
    encoder.set_compute_pipeline_state(&pipeline);

    // Threadgroup memory: max(logits_size, outputs_size). Matches
    // mistralrs' formula exactly so behavior is bit-identical for the
    // same shape inputs.
    let num_simds = NUM_THREADS / NUM_SIMD_LANES;
    let max_num_blocks_per_seq = max_context_len.div_ceil(block_size);
    let padded_max_context_len = max_num_blocks_per_seq
        .checked_mul(block_size)
        .ok_or_else(|| MetalKernelError::InvalidShape("padded context overflow".into()))?;
    let f32_bytes = std::mem::size_of::<f32>() as u64;
    let logits_size = u64::from(padded_max_context_len)
        .checked_mul(f32_bytes)
        .ok_or_else(|| MetalKernelError::InvalidShape("logits workspace overflow".into()))?;
    let outputs_size = u64::from(num_simds / 2)
        .checked_mul(u64::from(head_size))
        .and_then(|value| value.checked_mul(f32_bytes))
        .ok_or_else(|| MetalKernelError::InvalidShape("output workspace overflow".into()))?;
    let shared_mem_size = logits_size.max(outputs_size);
    let total_threadgroup_memory = shared_mem_size
        .checked_add(pipeline.static_threadgroup_memory_length() as u64)
        .ok_or_else(|| MetalKernelError::InvalidShape("threadgroup memory overflow".into()))?;
    let max_threadgroup_memory = kernels.library().device().max_threadgroup_memory_length() as u64;
    if total_threadgroup_memory > max_threadgroup_memory {
        return Err(MetalKernelError::FeatureNotAvailable(
            "paged-attention V1 context exceeds device threadgroup memory; V2 is required",
        ));
    }
    if u64::from(NUM_THREADS) > pipeline.max_total_threads_per_threadgroup() as u64 {
        return Err(MetalKernelError::DispatchFailed(format!(
            "paged-attention needs {NUM_THREADS} threads, pipeline supports {}",
            pipeline.max_total_threads_per_threadgroup()
        )));
    }
    encoder.set_threadgroup_memory_length(0u64, shared_mem_size);

    // Scalar params, converted to the i32/f32 ABI the kernel expects.
    let num_kv_heads_i = i32::try_from(num_kv_heads)
        .map_err(|_| MetalKernelError::InvalidShape("num_kv_heads exceeds i32".into()))?;
    let scale_f = softmax_scale;
    // softcapping==None -> identity. Upstream mistralrs uses 0.0 as the
    // "off" sentinel for the tanh(logit / softcap) * softcap branch;
    // we follow the same convention so the kernel disables softcap.
    let softcap_f = softcapping.unwrap_or(0.0);
    // Default contiguous strides; the caller-provided shapes determine
    // these — for the v1 wrapper we accept the typical layout
    // `q: [num_seqs, num_heads, head_size]`,
    // `k_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]`,
    // with `x = 16 / sizeof(cache_dtype)`.
    let x: u32 = 16 / cache_dtype.bytes() as u32;
    if head_size % x != 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "head_size {head_size} is not divisible by cache vector width {x}"
        )));
    }
    let q_stride = num_heads
        .checked_mul(head_size)
        .ok_or_else(|| MetalKernelError::InvalidShape("Q stride overflow".into()))?;
    let kv_head_stride = (head_size / x)
        .checked_mul(block_size)
        .and_then(|value| value.checked_mul(x))
        .ok_or_else(|| MetalKernelError::InvalidShape("KV head stride overflow".into()))?;
    let kv_block_stride = num_kv_heads
        .checked_mul(kv_head_stride)
        .ok_or_else(|| MetalKernelError::InvalidShape("KV block stride overflow".into()))?;
    let q_stride_i = i32::try_from(q_stride)
        .map_err(|_| MetalKernelError::InvalidShape("Q stride exceeds i32".into()))?;
    let kv_block_stride_i = i32::try_from(kv_block_stride)
        .map_err(|_| MetalKernelError::InvalidShape("KV block stride exceeds i32".into()))?;
    let kv_head_stride_i = i32::try_from(kv_head_stride)
        .map_err(|_| MetalKernelError::InvalidShape("KV head stride exceeds i32".into()))?;
    let max_num_blocks_per_seq_i = i32::try_from(max_num_blocks_per_seq)
        .map_err(|_| MetalKernelError::InvalidShape("block-table width exceeds i32".into()))?;

    let output_elements = u64::from(num_seqs)
        .checked_mul(u64::from(q_stride))
        .ok_or_else(|| MetalKernelError::InvalidShape("Q/output extent overflow".into()))?;
    let output_bytes = output_elements
        .checked_mul(output_dtype.bytes() as u64)
        .ok_or_else(|| MetalKernelError::InvalidShape("Q/output byte extent overflow".into()))?;
    require_buffer(q_buf, output_bytes, "paged-attention query")?;
    require_buffer(out_buf, output_bytes, "paged-attention output")?;
    let table_bytes = u64::from(num_seqs)
        .checked_mul(u64::from(max_num_blocks_per_seq))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| MetalKernelError::InvalidShape("block-table extent overflow".into()))?;
    require_buffer(block_tables_buf, table_bytes, "paged-attention block table")?;
    require_buffer(
        context_lens_buf,
        u64::from(num_seqs) * 4,
        "paged-attention context lengths",
    )?;

    let cache_block_bytes = u64::from(kv_block_stride)
        .checked_mul(cache_dtype.bytes() as u64)
        .ok_or_else(|| MetalKernelError::InvalidShape("cache block byte extent overflow".into()))?;
    if cache_block_bytes == 0
        || k_cache_buf.length() == 0
        || k_cache_buf.length() % cache_block_bytes != 0
        || v_cache_buf.length() % cache_block_bytes != 0
        || k_cache_buf.length() / cache_block_bytes != v_cache_buf.length() / cache_block_bytes
    {
        return Err(MetalKernelError::InvalidShape(
            "K/V cache buffers must contain the same whole number of cache blocks".into(),
        ));
    }
    let num_cache_blocks = u32::try_from(k_cache_buf.length() / cache_block_bytes)
        .map_err(|_| MetalKernelError::InvalidShape("cache block count exceeds u32".into()))?;
    if num_cache_blocks == 0 {
        return Err(MetalKernelError::InvalidShape(
            "paged-attention cache contains no blocks".into(),
        ));
    }

    // Buffer / scalar layout matches
    // mistralrs-paged-attn/src/metal/kernels/mod.rs::call_paged_attention_v1
    // (indices 0..18). Slots 0/1 (exp_sums/max_logits) and 14 (alibi),
    // 6/7 (fp8 scales), 18 (sinks) are unused in this V1 wrapper.
    encoder.set_buffer(2, Some(out_buf), 0u64);
    encoder.set_buffer(3, Some(q_buf), 0u64);
    encoder.set_buffer(4, Some(k_cache_buf), 0u64);
    encoder.set_buffer(5, Some(v_cache_buf), 0u64);
    set_bytes(encoder, 8, &num_kv_heads_i);
    set_bytes(encoder, 9, &scale_f);
    set_bytes(encoder, 10, &softcap_f);
    encoder.set_buffer(11, Some(block_tables_buf), 0u64);
    encoder.set_buffer(12, Some(context_lens_buf), 0u64);
    set_bytes(encoder, 13, &max_num_blocks_per_seq_i);
    set_bytes(encoder, 15, &q_stride_i);
    set_bytes(encoder, 16, &kv_block_stride_i);
    set_bytes(encoder, 17, &kv_head_stride_i);
    set_bytes(encoder, 19, &num_cache_blocks);
    set_bytes(encoder, 20, &max_context_len);

    let thread_groups = MTLSize {
        width: num_heads as u64,
        height: num_seqs as u64,
        depth: 1,
    };
    let threads_per_group = MTLSize {
        width: NUM_THREADS as u64,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups, threads_per_group);
    Ok(())
}

/// Helper that wraps `set_bytes` for a single POD scalar value.
fn set_bytes<T: Copy>(encoder: &ComputeCommandEncoderRef, index: u64, value: &T) {
    let size = std::mem::size_of::<T>() as u64;
    encoder.set_bytes(index, size, value as *const T as *const c_void);
}

/// V2 (partitioned long-context) dispatch.
///
/// The two-pass kernel needs caller-owned partition workspaces and is not part
/// of this API yet.
#[allow(clippy::too_many_arguments)]
pub fn call_paged_attention_metal_v2(
    _kernels: &MetalKernels,
    _encoder: &ComputeCommandEncoderRef,
    _output_dtype: DType,
    _cache_dtype: DType,
    _head_size: u32,
    _block_size: u32,
    _q_buf: &Buffer,
    _k_cache_buf: &Buffer,
    _v_cache_buf: &Buffer,
    _out_buf: &Buffer,
    _block_tables_buf: &Buffer,
    _context_lens_buf: &Buffer,
    _num_seqs: u32,
    _num_kv_heads: u32,
    _num_heads: u32,
    _softmax_scale: f32,
    _softcapping: Option<f32>,
    _max_context_len: u32,
) -> Result<(), MetalKernelError> {
    Err(MetalKernelError::FeatureNotAvailable(
        "paged-attention V2 partition workspaces are not wired",
    ))
}

fn require_buffer(buffer: &Buffer, required: u64, label: &str) -> Result<(), MetalKernelError> {
    if buffer.length() < required {
        return Err(MetalKernelError::InvalidShape(format!(
            "{label} buffer has {} bytes, needs {required}",
            buffer.length()
        )));
    }
    Ok(())
}
