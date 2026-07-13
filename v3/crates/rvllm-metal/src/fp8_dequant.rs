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

//! FP8 (E4M3) dequantization wrappers.
//!
//! Adapted from mistral.rs revision
//! `31c13eb4587d3e4a5204870c98b70c05a1e5c943`. Per-channel and blockwise
//! E4M3 paths validate every buffer extent before encoding a dispatch.

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

use crate::{device::MetalKernelError, kernels::MetalKernels};

/// Per-channel FP8 (E4M3) dequantization.
///
/// `weight_buf` holds `num_elements` packed FP8 bytes (E4M3) laid out
/// row-major as `[num_channels, num_elements / num_channels]`.
/// `scale_inv_buf` holds one f32 per output channel (`num_channels`
/// entries). `output_buf` receives `num_elements` values in `out_dtype`.
///
/// The kernel name follows the `fp8_perchannel_dequant_*` exports of
/// `scalar_fp8.metal`.
///
/// Threadgroup config: linear 1D split, one thread per element, with
/// `width = min(pipeline.max_total_threads_per_threadgroup(), num_elements)`
/// — identical to mistralrs's `linear_split` helper.
#[allow(clippy::too_many_arguments)]
pub fn call_dequant_scalar_fp8(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    out_dtype: rvllm_core::DType,
    weight_buf: &Buffer,
    scale_inv_buf: &Buffer,
    output_buf: &Buffer,
    num_elements: u32,
    num_channels: u32,
) -> Result<(), MetalKernelError> {
    if num_channels == 0 {
        return Err(MetalKernelError::InvalidShape(
            "num_channels must be > 0".into(),
        ));
    }
    if num_elements == 0 {
        return Err(MetalKernelError::InvalidShape(
            "num_elements must be > 0".into(),
        ));
    }
    if num_elements % num_channels != 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "num_elements ({num_elements}) must be a multiple of num_channels ({num_channels})"
        )));
    }
    let row_stride = num_elements / num_channels;
    require_buffer(weight_buf, u64::from(num_elements), "FP8 weight")?;
    require_buffer(
        scale_inv_buf,
        u64::from(num_channels) * std::mem::size_of::<f32>() as u64,
        "FP8 channel scales",
    )?;
    require_buffer(
        output_buf,
        u64::from(num_elements) * out_dtype.bytes() as u64,
        "dequantized output",
    )?;

    let kernel_name = match out_dtype {
        rvllm_core::DType::F32 => "fp8_perchannel_dequant_float",
        rvllm_core::DType::F16 => "fp8_perchannel_dequant_half",
        rvllm_core::DType::Bf16 => "fp8_perchannel_dequant_bfloat16_t",
        other => {
            return Err(MetalKernelError::FeatureNotAvailable(match other {
                rvllm_core::DType::F64 => "scalar fp8 dequant: F64 output",
                rvllm_core::DType::I32 => "scalar fp8 dequant: I32 output",
                rvllm_core::DType::I64 => "scalar fp8 dequant: I64 output",
                rvllm_core::DType::U32 => "scalar fp8 dequant: U32 output",
                rvllm_core::DType::U8 => "scalar fp8 dequant: U8 output",
                rvllm_core::DType::Fp8E4M3 => "scalar fp8 dequant: FP8E4M3 output",
                rvllm_core::DType::Fp8E5M2 => "scalar fp8 dequant: FP8E5M2 output",
                _ => "scalar fp8 dequant: unsupported output dtype",
            }));
        }
    };

    let pipeline = kernels.pipeline(kernel_name)?;
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(weight_buf), 0);
    encoder.set_buffer(1, Some(scale_inv_buf), 0);
    encoder.set_buffer(2, Some(output_buf), 0);

    let n = num_elements;
    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        (&n as *const u32) as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        (&row_stride as *const u32).cast(),
    );
    encoder.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        (&num_channels as *const u32).cast(),
    );

    // linear_split: one thread per element, capped at the pipeline's max
    // threads per threadgroup. Identical to mistralrs's helper in
    // `metal_kernels/utils.rs`.
    let max_tg = pipeline.max_total_threads_per_threadgroup() as u64;
    let elems = num_elements as u64;
    let width = std::cmp::min(max_tg, elems);
    // ceil_div
    let group_count = elems.div_ceil(width);

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let thread_group_count = MTLSize {
        width: group_count,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Blockwise FP8 dequantization with one F32 scale per logical tile.
#[allow(clippy::too_many_arguments)]
pub fn call_dequant_blockwise_fp8(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    out_dtype: rvllm_core::DType,
    weight_buf: &Buffer,
    scale_buf: &Buffer,
    output_buf: &Buffer,
    weight_height: u32,
    weight_width: u32,
    weight_row_stride: u32,
    scale_stride: u32,
    block_size_y: u32,
    block_size_x: u32,
) -> Result<(), MetalKernelError> {
    if weight_height == 0
        || weight_width == 0
        || block_size_y == 0
        || block_size_x == 0
        || weight_row_stride < weight_width
    {
        return Err(MetalKernelError::InvalidShape(
            "blockwise FP8 dimensions, block sizes, and row stride are invalid".into(),
        ));
    }
    let scale_cols = weight_width.div_ceil(block_size_x);
    let scale_rows = weight_height.div_ceil(block_size_y);
    if scale_stride < scale_cols {
        return Err(MetalKernelError::InvalidShape(format!(
            "scale_stride {scale_stride} is smaller than {scale_cols} tile columns"
        )));
    }
    let weight_elements = u64::from(weight_height - 1)
        .checked_mul(u64::from(weight_row_stride))
        .and_then(|value| value.checked_add(u64::from(weight_width)))
        .ok_or_else(|| MetalKernelError::InvalidShape("weight extent overflow".into()))?;
    let scale_elements = u64::from(scale_rows - 1)
        .checked_mul(u64::from(scale_stride))
        .and_then(|value| value.checked_add(u64::from(scale_cols)))
        .ok_or_else(|| MetalKernelError::InvalidShape("scale extent overflow".into()))?;
    require_buffer(weight_buf, weight_elements, "blockwise FP8 weight")?;
    require_buffer(
        scale_buf,
        scale_elements
            .checked_mul(4)
            .ok_or_else(|| MetalKernelError::InvalidShape("scale byte extent overflow".into()))?,
        "blockwise FP8 scales",
    )?;
    require_buffer(
        output_buf,
        weight_elements
            .checked_mul(out_dtype.bytes() as u64)
            .ok_or_else(|| MetalKernelError::InvalidShape("output byte extent overflow".into()))?,
        "blockwise FP8 output",
    )?;

    let kernel_name = match out_dtype {
        rvllm_core::DType::F32 => "dequant_fp8_blockwise_float",
        rvllm_core::DType::F16 => "dequant_fp8_blockwise_half",
        rvllm_core::DType::Bf16 => "dequant_fp8_blockwise_bfloat16_t",
        _ => {
            return Err(MetalKernelError::FeatureNotAvailable(
                "blockwise FP8 output dtype",
            ))
        }
    };
    #[repr(C)]
    struct DequantParams {
        weight_height: u32,
        weight_width: u32,
        weight_row_stride: u32,
        scale_stride: u32,
        block_size_y: u32,
        block_size_x: u32,
    }
    let params = DequantParams {
        weight_height,
        weight_width,
        weight_row_stride,
        scale_stride,
        block_size_y,
        block_size_x,
    };
    let pipeline = kernels.pipeline(kernel_name)?;
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(weight_buf), 0);
    encoder.set_buffer(1, Some(scale_buf), 0);
    encoder.set_buffer(2, Some(output_buf), 0);
    encoder.set_bytes(
        3,
        std::mem::size_of::<DequantParams>() as u64,
        (&params as *const DequantParams).cast(),
    );

    let max_threads = pipeline.max_total_threads_per_threadgroup() as u64;
    let threads_x = u64::from(block_size_x).min(32).min(max_threads);
    let threads_y = u64::from(block_size_y).min(max_threads / threads_x).max(1);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: u64::from(scale_cols),
            height: u64::from(scale_rows),
            depth: 1,
        },
        MTLSize {
            width: threads_x,
            height: threads_y,
            depth: 1,
        },
    );
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
