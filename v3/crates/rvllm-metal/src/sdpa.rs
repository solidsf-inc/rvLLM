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
// Portions of the dispatch logic are adapted from mistral.rs revision
// 31c13eb4587d3e4a5204870c98b70c05a1e5c943:
// https://github.com/EricLBuehler/mistral.rs
// licensed under the MIT License, Copyright (c) 2024 Eric Buehler.
// See LICENSES/MIT-mistralrs for the full upstream license text.

//! Fused softmax and SDPA dispatch. Function constant 50 selects sink-aware
//! math. `None` compiles the true no-sink path, rather than adding a zero-valued
//! virtual token to the denominator.

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use metal::{Buffer, ComputeCommandEncoderRef, MTLResourceOptions, MTLSize};

use rvllm_core::DType;

use crate::device::MetalKernelError;
use crate::kernels::MetalKernels;

const FC_USE_SINKS: u64 = 50;

/// Map `rvllm_core::DType` to the type suffix used by the lifted
/// softmax-with-sinks kernel's `[[host_name(...)]]`.
fn softmax_type_suffix(dtype: DType) -> Result<&'static str, MetalKernelError> {
    match dtype {
        DType::F32 => Ok("float"),
        DType::F16 => Ok("half"),
        DType::Bf16 => Ok("bfloat"),
        _ => Err(MetalKernelError::FeatureNotAvailable(
            "softmax_with_sinks: only F32/F16/Bf16 supported",
        )),
    }
}

/// Map `rvllm_core::DType` to the type suffix used by the lifted
/// SDPA-with-sinks kernels. The bf16 suffix follows mistralrs's
/// convention (`bfloat16_t`) for the SDPA family, which differs from
/// the softmax family's `bfloat` suffix.
fn sdpa_type_suffix(dtype: DType) -> Result<&'static str, MetalKernelError> {
    match dtype {
        DType::F32 => Ok("float"),
        DType::F16 => Ok("half"),
        DType::Bf16 => Ok("bfloat16_t"),
        _ => Err(MetalKernelError::FeatureNotAvailable(
            "sdpa_with_sinks: only F32/F16/Bf16 supported",
        )),
    }
}

/// Pick the BC (KV tile width) used by the lifted prefill kernel for a
/// given `head_dim`. Mirrors the `instantiate_flash_attn_sinks_heads`
/// macro in `sdpa_with_sinks.metal`.
fn prefill_bc_for_head_dim(head_dim: u32) -> Result<u32, MetalKernelError> {
    match head_dim {
        64 => Ok(64),
        80 | 96 | 112 | 128 => Ok(32),
        192 | 256 => Ok(16),
        _ => Err(MetalKernelError::FeatureNotAvailable(
            "sdpa_with_sinks prefill: unsupported head_dim (expected 64/80/96/112/128/192/256)",
        )),
    }
}

/// The ABI keeps the sink buffer slot even when function constant 50 is false.
/// That specialization never reads the one-element placeholder.
fn placeholder_sinks_buffer(kernels: &MetalKernels) -> Result<Buffer, MetalKernelError> {
    let device = kernels.library().device();
    let size_bytes = std::mem::size_of::<f32>() as u64;
    let buf = device.new_buffer(size_bytes, MTLResourceOptions::StorageModePrivate);
    if buf.length() < size_bytes {
        return Err(MetalKernelError::DispatchFailed(format!(
            "failed to allocate {size_bytes}-byte placeholder sink buffer"
        )));
    }
    Ok(buf)
}

/// Bind a `u32` scalar at the given buffer index using `setBytes`.
#[inline]
fn set_u32(encoder: &ComputeCommandEncoderRef, index: u64, val: u32) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<u32>() as u64,
        &val as *const u32 as *const std::ffi::c_void,
    );
}

/// Bind an `i32` scalar at the given buffer index using `setBytes`.
#[inline]
fn set_i32(encoder: &ComputeCommandEncoderRef, index: u64, val: i32) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<i32>() as u64,
        &val as *const i32 as *const std::ffi::c_void,
    );
}

/// Bind an `f32` scalar at the given buffer index using `setBytes`.
#[inline]
fn set_f32(encoder: &ComputeCommandEncoderRef, index: u64, val: f32) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<f32>() as u64,
        &val as *const f32 as *const std::ffi::c_void,
    );
}

/// Bind a `u64` (`size_t` in Metal) scalar at the given buffer index.
#[inline]
fn set_u64(encoder: &ComputeCommandEncoderRef, index: u64, val: u64) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<u64>() as u64,
        &val as *const u64 as *const std::ffi::c_void,
    );
}

/// Dispatch `softmax_with_sinks_<type>` from
/// `src/metal_kernels/softmax_sdpa.metal`.
///
/// * `input_buf`  — logits, shape `[n_axis, n]` (row-major, last dim = `n`).
/// * `output_buf` — softmax probabilities, same shape as input.
/// * `sinks_buf`  — one F32 sink per row, or `None` for exact softmax.
/// * `n`          — last-dim length (`k_len` in the kernel).
/// * `n_axis`     — number of rows to softmax (`total_rows =
///   batch * num_heads * q_len`). The kernel computes
///   `head_idx = (row / q_len) % num_heads`; with sinks=None the
///   `num_heads`/`q_len` split is irrelevant (the bound sinks buffer
///   is zero for every head), so we pass `num_heads = n_axis`,
///   `q_len = 1`. When sinks are real, callers must use
///   `call_softmax_with_sinks_full` if they need a non-flat row layout
///   (not exposed in this slice — punt to a follow-up).
pub fn call_softmax_with_sinks(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    input_buf: &Buffer,
    output_buf: &Buffer,
    sinks_buf: Option<&Buffer>,
    n: u32,
    n_axis: u32,
    dtype: DType,
) -> Result<(), MetalKernelError> {
    if n == 0 || n_axis == 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "softmax_with_sinks: zero-sized dispatch n={n} n_axis={n_axis}"
        )));
    }
    let elements = u64::from(n)
        .checked_mul(u64::from(n_axis))
        .ok_or_else(|| MetalKernelError::InvalidShape("softmax extent overflow".into()))?;
    let bytes = elements
        .checked_mul(dtype.bytes() as u64)
        .ok_or_else(|| MetalKernelError::InvalidShape("softmax byte extent overflow".into()))?;
    require_buffer(input_buf, bytes, "softmax input")?;
    require_buffer(output_buf, bytes, "softmax output")?;
    if let Some(sinks) = sinks_buf {
        require_buffer(sinks, u64::from(n_axis) * 4, "softmax sinks")?;
    }

    let suffix = softmax_type_suffix(dtype)?;
    let name = format!("softmax_with_sinks_{suffix}");
    let pipeline = kernels.pipeline_with_bool_constant(&name, FC_USE_SINKS, sinks_buf.is_some())?;
    encoder.set_compute_pipeline_state(&pipeline);

    // Either a caller-supplied sinks buffer or a transient zero buffer.
    // The owned `Buffer` keeps the allocation alive until function
    // return; the encoder retains it once bound for the command-buffer
    // lifetime.
    let placeholder: Option<Buffer> = match sinks_buf {
        Some(_) => None,
        None => Some(placeholder_sinks_buffer(kernels)?),
    };
    let sinks_bound: &Buffer = sinks_buf.unwrap_or_else(|| placeholder.as_ref().unwrap());

    encoder.set_buffer(0, Some(input_buf), 0);
    encoder.set_buffer(1, Some(sinks_bound), 0);
    encoder.set_buffer(2, Some(output_buf), 0);
    // The lifted kernel signature is (num_heads, q_len, k_len). With
    // sinks=None we collapse heads/q_len; with sinks=Some the caller
    // is expected to have arranged a layout where every row's head
    // index resolves correctly (heads == n_axis works for the common
    // per-row sinks case).
    set_u32(encoder, 3, n_axis); // num_heads (effective)
    set_u32(encoder, 4, 1); //     q_len
    set_u32(encoder, 5, n); //     k_len

    // Threadgroup sizing matches mistralrs (`metal_kernels/mod.rs`):
    let threads_per_group: u32 = if n <= 64 {
        64
    } else if n <= 128 {
        128
    } else if n <= 256 {
        256
    } else {
        512
    };
    if u64::from(threads_per_group) > pipeline.max_total_threads_per_threadgroup() as u64 {
        return Err(MetalKernelError::DispatchFailed(format!(
            "softmax requires {threads_per_group} threads, pipeline supports {}",
            pipeline.max_total_threads_per_threadgroup()
        )));
    }

    // Shared memory: s_max(1) + s_sum(1) + warp_scratch(threads/32).
    let num_simdgroups = (threads_per_group + 31) / 32;
    let shared_mem_bytes = ((2 + num_simdgroups) as u64) * std::mem::size_of::<f32>() as u64;
    require_threadgroup_memory(kernels, &pipeline, shared_mem_bytes)?;
    encoder.set_threadgroup_memory_length(0, shared_mem_bytes);

    let grid = MTLSize {
        width: n_axis as u64,
        height: 1,
        depth: 1,
    };
    let group = MTLSize {
        width: threads_per_group as u64,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid, group);

    // `owned_zero` drops here. By this point Metal has retained any
    // bound buffer for the command buffer's lifetime, so the
    // allocation outlives this function regardless of our local drop.
    let _ = placeholder;
    Ok(())
}

/// Dispatch `sdpa_with_sinks` — auto-selects between decode
/// (`sdpa_vector_with_sinks_<type>_<head_dim>`, when `seq_len_q == 1`)
/// and prefill (`flash_attn_sinks_<type>_hd<head_dim>_br8_bc<bc>`,
/// when `seq_len_q > 1`).
///
/// * `q_buf` / `k_buf` / `v_buf` — Q `[batch, num_heads, seq_q,
///   head_dim]`, K/V `[batch, num_kv_heads, seq_kv, head_dim]`.
/// * `sinks_buf` — per-head `f32` sinks. `None` selects exact no-sink math.
/// * `out_buf` — `[batch, num_heads, seq_q, head_dim]`.
/// * `softmax_scale` — usually `1.0 / sqrt(head_dim)`; passed through
///   verbatim to the kernel.
///
/// Caller is responsible for binding `batch_size == 1`; multi-batch
/// dispatch should be done by the consumer crate looping over the
/// batch dimension (matches the rvllm decode loop). The grid we emit
/// uses `b = num_heads` for decode and `(num_heads, 1, ceil(seq_q/BR))`
/// for prefill, which assumes batch=1 — sufficient for Gemma 4's
/// initial bring-up.
pub fn call_sdpa_with_sinks(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    sinks_buf: Option<&Buffer>,
    out_buf: &Buffer,
    seq_len_q: u32,
    seq_len_kv: u32,
    head_dim: u32,
    num_heads: u32,
    num_kv_heads: u32,
    softmax_scale: f32,
    dtype: DType,
) -> Result<(), MetalKernelError> {
    if seq_len_q == 0 || seq_len_kv == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "sdpa_with_sinks: zero head dim/count head_dim={head_dim} \
             num_heads={num_heads} num_kv_heads={num_kv_heads}"
        )));
    }
    if num_heads % num_kv_heads != 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "sdpa_with_sinks: num_heads ({num_heads}) must be a \
             multiple of num_kv_heads ({num_kv_heads})"
        )));
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(MetalKernelError::InvalidShape(
            "softmax_scale must be finite and > 0".into(),
        ));
    }
    let _ = i32::try_from(seq_len_q)
        .and_then(|_| i32::try_from(seq_len_kv))
        .and_then(|_| i32::try_from(num_heads))
        .and_then(|_| i32::try_from(num_kv_heads))
        .map_err(|_| MetalKernelError::InvalidShape("SDPA dimensions exceed i32".into()))?;

    let suffix = sdpa_type_suffix(dtype)?;
    let dtype_bytes = match dtype {
        DType::F32 => 4u64,
        DType::F16 | DType::Bf16 => 2u64,
        _ => unreachable!("guarded by sdpa_type_suffix"),
    };

    let q_elements = u64::from(num_heads)
        .checked_mul(u64::from(seq_len_q))
        .and_then(|value| value.checked_mul(u64::from(head_dim)))
        .ok_or_else(|| MetalKernelError::InvalidShape("Q extent overflow".into()))?;
    let kv_elements = u64::from(num_kv_heads)
        .checked_mul(u64::from(seq_len_kv))
        .and_then(|value| value.checked_mul(u64::from(head_dim)))
        .ok_or_else(|| MetalKernelError::InvalidShape("KV extent overflow".into()))?;
    let q_bytes = q_elements
        .checked_mul(dtype_bytes)
        .ok_or_else(|| MetalKernelError::InvalidShape("Q byte extent overflow".into()))?;
    let kv_bytes = kv_elements
        .checked_mul(dtype_bytes)
        .ok_or_else(|| MetalKernelError::InvalidShape("KV byte extent overflow".into()))?;
    require_buffer(q_buf, q_bytes, "SDPA queries")?;
    require_buffer(k_buf, kv_bytes, "SDPA keys")?;
    require_buffer(v_buf, kv_bytes, "SDPA values")?;
    require_buffer(out_buf, q_bytes, "SDPA output")?;
    if let Some(sinks) = sinks_buf {
        require_buffer(sinks, u64::from(num_heads) * 4, "SDPA sinks")?;
    }

    let placeholder: Option<Buffer> = match sinks_buf {
        Some(_) => None,
        None => Some(placeholder_sinks_buffer(kernels)?),
    };
    let sinks_bound: &Buffer = sinks_buf.unwrap_or_else(|| placeholder.as_ref().unwrap());

    if seq_len_q == 1 {
        // ---- Decode path: sdpa_vector_with_sinks_<type>_<head_dim> ----
        if !matches!(head_dim, 64 | 80 | 96 | 128 | 256) {
            return Err(MetalKernelError::FeatureNotAvailable(
                "sdpa_with_sinks decode: unsupported head_dim (expected 64/80/96/128/256)",
            ));
        }

        let name = format!("sdpa_vector_with_sinks_{suffix}_{head_dim}");
        let pipeline =
            kernels.pipeline_with_bool_constant(&name, FC_USE_SINKS, sinks_buf.is_some())?;
        encoder.set_compute_pipeline_state(&pipeline);

        let gqa_factor = i32::try_from(num_heads / num_kv_heads)
            .map_err(|_| MetalKernelError::InvalidShape("GQA factor exceeds i32".into()))?;
        let n_kv = i32::try_from(seq_len_kv)
            .map_err(|_| MetalKernelError::InvalidShape("KV length exceeds i32".into()))?;
        // Strides between consecutive KV positions in elements (row-major
        // `[num_kv_heads, seq_kv, head_dim]`): k_stride = v_stride =
        // head_dim. Express in *bytes* for size_t binding.
        let k_stride: u64 = (head_dim as u64) * dtype_bytes;
        let v_stride: u64 = k_stride;

        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_buf), 0);
        encoder.set_buffer(2, Some(v_buf), 0);
        encoder.set_buffer(3, Some(sinks_bound), 0);
        encoder.set_buffer(4, Some(out_buf), 0);
        set_i32(encoder, 5, gqa_factor);
        set_i32(encoder, 6, n_kv);
        set_u64(encoder, 7, k_stride);
        set_u64(encoder, 8, v_stride);
        set_f32(encoder, 9, softmax_scale);

        let grid = MTLSize {
            width: 1,
            height: num_heads as u64,
            depth: 1,
        };
        let group = MTLSize {
            width: 1024, // 32 simdgroups * 32 threads
            height: 1,
            depth: 1,
        };
        if group.width > pipeline.max_total_threads_per_threadgroup() as u64 {
            return Err(MetalKernelError::DispatchFailed(format!(
                "decode SDPA requires {} threads, pipeline supports {}",
                group.width,
                pipeline.max_total_threads_per_threadgroup()
            )));
        }
        encoder.dispatch_thread_groups(grid, group);
    } else {
        // ---- Prefill path: flash_attn_sinks_<type>_hd<HD>_br8_bc<BC> ----
        let br: u32 = 8;
        let bc: u32 = prefill_bc_for_head_dim(head_dim)?;

        let name = format!("flash_attn_sinks_{suffix}_hd{head_dim}_br{br}_bc{bc}");
        let pipeline =
            kernels.pipeline_with_bool_constant(&name, FC_USE_SINKS, sinks_buf.is_some())?;
        encoder.set_compute_pipeline_state(&pipeline);

        // Shared memory: K + V tiles, each `bc * d_pad` float32.
        let d_pad = ((head_dim + 31) / 32) * 32;
        let shared_mem_bytes = 2u64 * bc as u64 * d_pad as u64 * std::mem::size_of::<f32>() as u64;
        require_threadgroup_memory(kernels, &pipeline, shared_mem_bytes)?;
        encoder.set_threadgroup_memory_length(0, shared_mem_bytes);

        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(k_buf), 0);
        encoder.set_buffer(2, Some(v_buf), 0);
        encoder.set_buffer(3, Some(sinks_bound), 0);
        encoder.set_buffer(4, Some(out_buf), 0);
        set_f32(encoder, 5, softmax_scale);
        set_i32(encoder, 6, seq_len_q as i32);
        set_i32(encoder, 7, seq_len_kv as i32);
        set_i32(encoder, 8, num_heads as i32);
        set_i32(encoder, 9, num_kv_heads as i32);
        // window_size = 0 → no sliding-window mask (full causal).
        set_i32(encoder, 10, 0);

        let grid = MTLSize {
            width: num_heads as u64,
            height: 1, // batch_size; consumer loops if > 1
            depth: ((seq_len_q + br - 1) / br) as u64,
        };
        let group = MTLSize {
            width: (br * 32) as u64,
            height: 1,
            depth: 1,
        };
        if group.width > pipeline.max_total_threads_per_threadgroup() as u64 {
            return Err(MetalKernelError::DispatchFailed(format!(
                "prefill SDPA requires {} threads, pipeline supports {}",
                group.width,
                pipeline.max_total_threads_per_threadgroup()
            )));
        }
        encoder.dispatch_thread_groups(grid, group);
    }

    // See `call_softmax_with_sinks` — Metal retains the binding for
    // the encoder's command buffer lifetime; our local drop is safe.
    let _ = placeholder;
    Ok(())
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

fn require_threadgroup_memory(
    kernels: &MetalKernels,
    pipeline: &metal::ComputePipelineStateRef,
    dynamic: u64,
) -> Result<(), MetalKernelError> {
    let total = dynamic
        .checked_add(pipeline.static_threadgroup_memory_length() as u64)
        .ok_or_else(|| MetalKernelError::InvalidShape("threadgroup memory overflow".into()))?;
    let maximum = kernels.library().device().max_threadgroup_memory_length() as u64;
    if total > maximum {
        return Err(MetalKernelError::DispatchFailed(format!(
            "threadgroup memory requires {total} bytes, device supports {maximum}"
        )));
    }
    Ok(())
}
