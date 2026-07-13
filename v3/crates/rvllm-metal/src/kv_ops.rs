// Copyright 2026 m0at
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
// Rust wrappers for the two paged-KV-cache Metal kernels
// (`reshape_and_cache`, `gather_kv_cache`). Buffer and dispatch
// setup is adapted from mistral.rs revision
// 31c13eb4587d3e4a5204870c98b70c05a1e5c943:
//   mistralrs-paged-attn/src/metal/kernels/mod.rs
//     :: call_reshape_and_cache
//     :: call_gather_kv_cache
// (https://github.com/EricLBuehler/mistral.rs, MIT).
//
// FP8 (`use_fp8_scales`) is wired off via the
// function-constant gate and the scale-buffer slots are left unbound.

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::ffi::c_void;

use metal::{Buffer, ComputeCommandEncoderRef, FunctionConstantValues, MTLDataType, MTLSize};

use rvllm_core::DType;

use crate::device::MetalKernelError;
use crate::kernels::MetalKernels;

/// Function-constant index for `use_fp8_scales` — must match both
/// `reshape_and_cache.metal` and `gather_kv_cache.metal`.
const FC_USE_FP8_SCALES: u64 = 10;

/// Cap on threads-per-threadgroup to match mistralrs's dispatch.
const THREADGROUP_MAX_THREADS: usize = 512;

/// Translate the rvllm `DType` into the mistralrs Metal type token used
/// to instantiate `reshape_and_cache_kv_<kv>_cache_<cache>` and
/// `gather_kv_cache_cache_<cache>_out_<out>`. Returns the literal token
/// (e.g. `"bfloat16_t"`) — must match the `instantiate_*` macros.
fn dtype_token(dtype: DType) -> Result<&'static str, MetalKernelError> {
    match dtype {
        DType::Bf16 => Ok("bfloat16_t"),
        DType::F16 => Ok("half"),
        DType::F32 => Ok("float"),
        other => Err(MetalKernelError::FeatureNotAvailable(match other {
            DType::Fp8E4M3 => "fp8 KV path (v1 is bf16-only)",
            _ => "unsupported KV dtype",
        })),
    }
}

/// Build `FunctionConstantValues` with `use_fp8_scales = false`. The
/// constant gates both the optional scale-buffer slots and the divide
/// path inside the kernel body.
fn fp8_disabled_constants() -> FunctionConstantValues {
    let constants = FunctionConstantValues::new();
    let v: bool = false;
    constants.set_constant_value_at_index(
        &v as *const bool as *const c_void,
        MTLDataType::Bool,
        FC_USE_FP8_SCALES,
    );
    constants
}

/// Build a pipeline for `name` with `use_fp8_scales = false`.
fn pipeline_no_fp8(
    kernels: &MetalKernels,
    name: &str,
) -> Result<metal::ComputePipelineState, MetalKernelError> {
    let constants = fp8_disabled_constants();
    let function = kernels
        .library()
        .get_function(name, Some(constants))
        .map_err(|e| MetalKernelError::KernelLoadFailed(format!("function `{name}`: {e}")))?;
    // The library is owned by `MetalKernels`, so reuse its underlying
    // device for pipeline creation by querying through the Function.
    function
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| MetalKernelError::KernelLoadFailed(format!("pipeline `{name}`: {e}")))
}

#[inline]
fn set_bytes_i32(encoder: &ComputeCommandEncoderRef, index: u64, value: i32) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<i32>() as u64,
        &value as *const i32 as *const c_void,
    );
}

/// Write freshly-projected K/V into the paged KV cache.
///
/// Kernel layout (one threadgroup per token):
/// - grid:  `(num_tokens, 1, 1)`
/// - block: `(min(num_kv_heads * head_size, 512), 1, 1)`
///
/// FP8 cache types are gated off. The K-cache vector width is 16 bytes divided
/// by the element size (8 for F16/BF16, 4 for F32).
#[allow(clippy::too_many_arguments)]
pub fn call_reshape_and_cache(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    k_buf: &Buffer,
    v_buf: &Buffer,
    k_cache_buf: &Buffer,
    v_cache_buf: &Buffer,
    slot_mapping_buf: &Buffer,
    num_tokens: u32,
    num_kv_heads: u32,
    head_size: u32,
    block_size: u32,
    dtype: DType,
) -> Result<(), MetalKernelError> {
    if num_tokens == 0 {
        return Ok(());
    }
    if num_kv_heads == 0 || head_size == 0 || block_size == 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "reshape_and_cache: zero dim (num_kv_heads={num_kv_heads}, \
             head_size={head_size}, block_size={block_size})"
        )));
    }
    let kv_token = dtype_token(dtype)?;
    let x = 16u32 / dtype.bytes() as u32;
    if head_size % x != 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "reshape_and_cache: head_size {head_size} is not divisible by x={x}"
        )));
    }
    let token_elements = num_kv_heads
        .checked_mul(head_size)
        .ok_or_else(|| MetalKernelError::InvalidShape("KV token extent overflow".into()))?;
    let input_elements = u64::from(num_tokens)
        .checked_mul(u64::from(token_elements))
        .ok_or_else(|| MetalKernelError::InvalidShape("KV input extent overflow".into()))?;
    let input_bytes = input_elements
        .checked_mul(dtype.bytes() as u64)
        .ok_or_else(|| MetalKernelError::InvalidShape("KV input byte extent overflow".into()))?;
    require_buffer(k_buf, input_bytes, "reshape K input")?;
    require_buffer(v_buf, input_bytes, "reshape V input")?;
    require_buffer(
        slot_mapping_buf,
        u64::from(num_tokens) * std::mem::size_of::<i64>() as u64,
        "reshape slot mapping",
    )?;
    let cache_block_bytes = u64::from(token_elements)
        .checked_mul(u64::from(block_size))
        .and_then(|value| value.checked_mul(dtype.bytes() as u64))
        .ok_or_else(|| MetalKernelError::InvalidShape("KV cache block extent overflow".into()))?;
    let num_cache_blocks = matching_cache_blocks(k_cache_buf, v_cache_buf, cache_block_bytes)?;
    // v1: cache dtype mirrors the KV dtype (no FP8 path).
    let cache_token = kv_token;
    let name = format!("reshape_and_cache_kv_{kv_token}_cache_{cache_token}");

    let pipeline = pipeline_no_fp8(kernels, &name)?;
    encoder.set_compute_pipeline_state(&pipeline);

    // Flat K/V are contiguous `[num_tokens, num_kv_heads, head_size]`,
    // so the per-token stride is `num_kv_heads * head_size` elements.
    let kv_stride = i32::try_from(token_elements)
        .map_err(|_| MetalKernelError::InvalidShape("KV stride exceeds i32".into()))?;
    let num_heads_i32 = i32::try_from(num_kv_heads)
        .map_err(|_| MetalKernelError::InvalidShape("KV head count exceeds i32".into()))?;
    let head_size_i32 = i32::try_from(head_size)
        .map_err(|_| MetalKernelError::InvalidShape("head_size exceeds i32".into()))?;
    let block_size_i32 = i32::try_from(block_size)
        .map_err(|_| MetalKernelError::InvalidShape("block_size exceeds i32".into()))?;
    let x_i32 =
        i32::try_from(x).map_err(|_| MetalKernelError::InvalidShape("x exceeds i32".into()))?;

    // Buffers (matches mistralrs binding indices 0..=4; 5/6 are FP8
    // scale buffers, gated off via `use_fp8_scales = false`).
    encoder.set_buffer(0, Some(k_buf), 0);
    encoder.set_buffer(1, Some(v_buf), 0);
    encoder.set_buffer(2, Some(k_cache_buf), 0);
    encoder.set_buffer(3, Some(v_cache_buf), 0);
    encoder.set_buffer(4, Some(slot_mapping_buf), 0);
    // Scalar params (indices 7..=12).
    set_bytes_i32(encoder, 7, kv_stride);
    set_bytes_i32(encoder, 8, kv_stride);
    set_bytes_i32(encoder, 9, num_heads_i32);
    set_bytes_i32(encoder, 10, head_size_i32);
    set_bytes_i32(encoder, 11, block_size_i32);
    set_bytes_i32(encoder, 12, x_i32);
    set_bytes_u32(encoder, 13, num_cache_blocks);

    let thread_groups_count = MTLSize {
        width: num_tokens as u64,
        height: 1,
        depth: 1,
    };
    let thread_width = u64::from(token_elements).min(THREADGROUP_MAX_THREADS as u64);
    if thread_width > pipeline.max_total_threads_per_threadgroup() as u64 {
        return Err(MetalKernelError::DispatchFailed(format!(
            "reshape_and_cache needs {thread_width} threads, pipeline supports {}",
            pipeline.max_total_threads_per_threadgroup()
        )));
    }
    let threads_per_threadgroup = MTLSize {
        width: thread_width,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Unpack the paged KV cache into contiguous `(num_tokens, kv_heads,
/// head_size)` K/V tensors (used by the SDPA prefill path).
///
/// Kernel layout (one threadgroup per output token):
/// - grid:  `(num_tokens, 1, 1)`
/// - block: `(min(num_kv_heads * head_size, 512), 1, 1)`
///
/// Cache and output dtypes match because FP8 is gated off.
#[allow(clippy::too_many_arguments)]
pub fn call_gather_kv_cache(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    k_cache_buf: &Buffer,
    v_cache_buf: &Buffer,
    block_table_buf: &Buffer,
    cu_seq_lens_buf: &Buffer,
    k_out_buf: &Buffer,
    v_out_buf: &Buffer,
    num_tokens: u32,
    num_seqs: u32,
    block_size: u32,
    block_table_stride: u32,
    num_kv_heads: u32,
    head_size: u32,
    x: u32,
    out_dtype: DType,
) -> Result<(), MetalKernelError> {
    if num_tokens == 0 {
        return Ok(());
    }
    if num_seqs == 0 || num_kv_heads == 0 || head_size == 0 || block_size == 0 || x == 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "gather_kv_cache: zero dim (num_seqs={num_seqs}, \
             num_kv_heads={num_kv_heads}, head_size={head_size}, \
             block_size={block_size}, x={x})"
        )));
    }
    if block_table_stride == 0 {
        return Err(MetalKernelError::InvalidShape(
            "gather_kv_cache: block_table_stride must be > 0".into(),
        ));
    }
    let out_token = dtype_token(out_dtype)?;
    let expected_x = 16u32 / out_dtype.bytes() as u32;
    if x != expected_x || head_size % x != 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "gather_kv_cache: expected x={expected_x} dividing head_size {head_size}, got {x}"
        )));
    }
    let token_elements = num_kv_heads
        .checked_mul(head_size)
        .ok_or_else(|| MetalKernelError::InvalidShape("gather token extent overflow".into()))?;
    let output_bytes = u64::from(num_tokens)
        .checked_mul(u64::from(token_elements))
        .and_then(|value| value.checked_mul(out_dtype.bytes() as u64))
        .ok_or_else(|| MetalKernelError::InvalidShape("gather output extent overflow".into()))?;
    require_buffer(k_out_buf, output_bytes, "gather K output")?;
    require_buffer(v_out_buf, output_bytes, "gather V output")?;
    require_buffer(
        block_table_buf,
        u64::from(num_seqs)
            .checked_mul(u64::from(block_table_stride))
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| MetalKernelError::InvalidShape("block-table extent overflow".into()))?,
        "gather block table",
    )?;
    require_buffer(
        cu_seq_lens_buf,
        (u64::from(num_seqs) + 1) * 4,
        "gather cumulative sequence lengths",
    )?;
    let cache_block_bytes = u64::from(token_elements)
        .checked_mul(u64::from(block_size))
        .and_then(|value| value.checked_mul(out_dtype.bytes() as u64))
        .ok_or_else(|| MetalKernelError::InvalidShape("gather cache extent overflow".into()))?;
    let num_cache_blocks = matching_cache_blocks(k_cache_buf, v_cache_buf, cache_block_bytes)?;
    // v1: cache dtype matches the output dtype (no FP8 path).
    let cache_token = out_token;
    let name = format!("gather_kv_cache_cache_{cache_token}_out_{out_token}");

    let pipeline = pipeline_no_fp8(kernels, &name)?;
    encoder.set_compute_pipeline_state(&pipeline);

    let num_tokens_i32 = to_i32(num_tokens, "num_tokens")?;
    let num_seqs_i32 = to_i32(num_seqs, "num_seqs")?;
    let block_size_i32 = to_i32(block_size, "block_size")?;
    let block_table_stride_i32 = to_i32(block_table_stride, "block_table_stride")?;
    let num_kv_heads_i32 = to_i32(num_kv_heads, "num_kv_heads")?;
    let head_size_i32 = to_i32(head_size, "head_size")?;
    let x_i32 = to_i32(x, "x")?;

    // Buffers (matches mistralrs binding indices 0..=3, 6..=7; 4/5 are
    // FP8 scale buffers, gated off via `use_fp8_scales = false`).
    encoder.set_buffer(0, Some(k_cache_buf), 0);
    encoder.set_buffer(1, Some(v_cache_buf), 0);
    encoder.set_buffer(2, Some(k_out_buf), 0);
    encoder.set_buffer(3, Some(v_out_buf), 0);
    encoder.set_buffer(6, Some(block_table_buf), 0);
    encoder.set_buffer(7, Some(cu_seq_lens_buf), 0);
    // Scalar params (indices 8..=14).
    set_bytes_i32(encoder, 8, num_tokens_i32);
    set_bytes_i32(encoder, 9, num_seqs_i32);
    set_bytes_i32(encoder, 10, block_size_i32);
    set_bytes_i32(encoder, 11, block_table_stride_i32);
    set_bytes_i32(encoder, 12, num_kv_heads_i32);
    set_bytes_i32(encoder, 13, head_size_i32);
    set_bytes_i32(encoder, 14, x_i32);
    set_bytes_u32(encoder, 15, num_cache_blocks);

    let thread_groups_count = MTLSize {
        width: num_tokens as u64,
        height: 1,
        depth: 1,
    };
    let thread_width = u64::from(token_elements).min(THREADGROUP_MAX_THREADS as u64);
    if thread_width > pipeline.max_total_threads_per_threadgroup() as u64 {
        return Err(MetalKernelError::DispatchFailed(format!(
            "gather_kv_cache needs {thread_width} threads, pipeline supports {}",
            pipeline.max_total_threads_per_threadgroup()
        )));
    }
    let threads_per_threadgroup = MTLSize {
        width: thread_width,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

#[inline]
fn set_bytes_u32(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<u32>() as u64,
        (&value as *const u32).cast(),
    );
}

fn to_i32(value: u32, name: &str) -> Result<i32, MetalKernelError> {
    i32::try_from(value).map_err(|_| MetalKernelError::InvalidShape(format!("{name} exceeds i32")))
}

fn matching_cache_blocks(
    key: &Buffer,
    value: &Buffer,
    block_bytes: u64,
) -> Result<u32, MetalKernelError> {
    if block_bytes == 0
        || key.length() == 0
        || key.length() % block_bytes != 0
        || value.length() % block_bytes != 0
        || key.length() / block_bytes != value.length() / block_bytes
    {
        return Err(MetalKernelError::InvalidShape(
            "K/V cache buffers must contain the same whole number of blocks".into(),
        ));
    }
    u32::try_from(key.length() / block_bytes)
        .map_err(|_| MetalKernelError::InvalidShape("cache block count exceeds u32".into()))
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
