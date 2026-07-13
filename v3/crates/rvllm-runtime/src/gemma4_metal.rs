// Copyright 2026 m0at
// Licensed under the Apache License, Version 2.0.

//! Fail-closed Gemma 4 Metal integration surface.
//!
//! The component kernels remain available through `rvllm-metal`. The layer
//! forward is not advertised as runnable until every stage and its full-layer
//! parity gate are implemented.

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::sync::Arc;

use rvllm_attention::AttentionBackend;
use rvllm_core::{CutlassCtx, CutlassError, Result, RvllmError};
use rvllm_cutlass::{CublasLt, CutlassBackend};
use rvllm_loader::metal_loader::MetalWeightCache;
use rvllm_mem::MetalKvAllocator;
use rvllm_metal::device::MetalDevice;
use rvllm_metal::kernels::MetalKernels;

use crate::gemma4_layer_exec::{Gemma4LayerDims, Gemma4LayerScratch, Gemma4MetadataPtrs};

pub enum ComputeDevice {
    Cuda {
        cublaslt: Arc<CublasLt>,
        cutlass: Arc<CutlassBackend>,
        sliding_attention: Arc<AttentionBackend>,
        global_attention: Arc<AttentionBackend>,
    },
    Metal {
        device: Arc<MetalDevice>,
        kernels: Arc<MetalKernels>,
        kv_alloc: Arc<MetalKvAllocator>,
        weight_cache: Arc<MetalWeightCache>,
    },
}

pub struct Gemma4LayerMetalBuffers<'a> {
    pub attn_norm_gamma: &'a metal::Buffer,
    pub post_attn_norm_gamma: &'a metal::Buffer,
    pub pre_ff_norm_gamma: &'a metal::Buffer,
    pub post_ff_norm_gamma: &'a metal::Buffer,
    pub q_norm_gamma: &'a metal::Buffer,
    pub k_norm_gamma: &'a metal::Buffer,
    pub qkv_fp8: &'a metal::Buffer,
    pub qkv_scale: &'a metal::Buffer,
    pub o_fp8: &'a metal::Buffer,
    pub o_scale: &'a metal::Buffer,
    pub gate_up_fp8: &'a metal::Buffer,
    pub gate_up_scale: &'a metal::Buffer,
    pub down_fp8: &'a metal::Buffer,
    pub down_scale: &'a metal::Buffer,
    pub layer_scalar: &'a metal::Buffer,
    pub qkv_f16: &'a metal::Buffer,
    pub o_f16: &'a metal::Buffer,
    pub gate_up_f16: &'a metal::Buffer,
    pub down_f16: &'a metal::Buffer,
    pub hidden_fp8: &'a metal::Buffer,
    pub hidden_scale: &'a metal::Buffer,
    pub q_out: &'a metal::Buffer,
    pub k_out: &'a metal::Buffer,
    pub v_out: &'a metal::Buffer,
    pub q_normed: &'a metal::Buffer,
    pub k_normed: &'a metal::Buffer,
    pub v_normed: &'a metal::Buffer,
    pub q_fp8: &'a metal::Buffer,
    pub k_cache: &'a metal::Buffer,
    pub v_cache: &'a metal::Buffer,
    pub q_scale_ptr: &'a metal::Buffer,
    pub kv_scale_ptr: &'a metal::Buffer,
    pub k_scale_cache: &'a metal::Buffer,
    pub v_scale_cache: &'a metal::Buffer,
    pub q_scale_cache: &'a metal::Buffer,
    pub attn_out: &'a metal::Buffer,
    pub attn_out_fp8: &'a metal::Buffer,
    pub attn_out_scale: &'a metal::Buffer,
    pub delta_f16: &'a metal::Buffer,
    pub gate_up_out: &'a metal::Buffer,
    pub gate_up_out_fp8: &'a metal::Buffer,
    pub gate_up_out_scale: &'a metal::Buffer,
    pub mlp_out_fp8: &'a metal::Buffer,
    pub mlp_out_scale: &'a metal::Buffer,
    pub residual: &'a metal::Buffer,
    pub positions: &'a metal::Buffer,
    pub slot_mapping: &'a metal::Buffer,
    pub cos: &'a metal::Buffer,
    pub sin: &'a metal::Buffer,
    pub block_tables: &'a metal::Buffer,
    pub context_lens: &'a metal::Buffer,
}

/// This turns true only after all layer stages and end-to-end parity are
/// implemented. Callers must not select Metal while it is false.
#[must_use]
pub const fn gemma4_metal_end_to_end_available() -> bool {
    false
}

fn metal_error(op: &'static str, unavailable: bool) -> RvllmError {
    let err = if unavailable {
        CutlassError::FeatureNotAvailable { op }
    } else {
        CutlassError::KernelLaunchFailed {
            variant: 0,
            cuda: rvllm_core::CudaErrorKind::Other,
        }
    };
    RvllmError::cutlass(
        err,
        CutlassCtx {
            kernel: op,
            stream: 0,
        },
    )
}

fn checked_elements(values: &[u32]) -> Result<u64> {
    values.iter().try_fold(1u64, |product, value| {
        product
            .checked_mul(u64::from(*value))
            .ok_or_else(|| metal_error("gemma4_metal.shape_overflow", false))
    })
}

fn checked_bytes(elements: u64, element_bytes: u64) -> Result<u64> {
    elements
        .checked_mul(element_bytes)
        .ok_or_else(|| metal_error("gemma4_metal.byte_span_overflow", false))
}

fn require_buffer(name: &'static str, buffer: &metal::Buffer, minimum: u64) -> Result<()> {
    if minimum == 0 || buffer.length() < minimum {
        return Err(metal_error(name, false));
    }
    Ok(())
}

pub fn validate_metal_buffers(
    dims: Gemma4LayerDims,
    buffers: &Gemma4LayerMetalBuffers<'_>,
) -> Result<()> {
    if dims.num_tokens == 0
        || dims.hidden == 0
        || dims.num_heads == 0
        || dims.head_dim == 0
        || dims.intermediate == 0
        || dims.block_size == 0
        || dims.num_blocks_total == 0
        || dims.max_blocks_per_seq == 0
        || dims.num_kv_heads == 0
        || dims.rotary_dim == 0
        || dims.rotary_dim > dims.head_dim
        || !dims.rotary_dim.is_multiple_of(2)
        || !dims.attn_scale.is_finite()
        || !dims.rms_eps.is_finite()
        || dims.rms_eps < 0.0
    {
        return Err(metal_error("gemma4_metal.invalid_dimensions", false));
    }

    let q_width = dims
        .num_heads
        .checked_mul(dims.head_dim)
        .ok_or_else(|| metal_error("gemma4_metal.q_width", false))?;
    let kv_width = dims
        .num_kv_heads
        .checked_mul(dims.head_dim)
        .ok_or_else(|| metal_error("gemma4_metal.kv_width", false))?;
    let qkv_width = q_width
        .checked_add(
            kv_width
                .checked_mul(2)
                .ok_or_else(|| metal_error("gemma4_metal.qkv_width", false))?,
        )
        .ok_or_else(|| metal_error("gemma4_metal.qkv_width", false))?;
    let tokens_hidden = checked_elements(&[dims.num_tokens, dims.hidden])?;
    let tokens_q = checked_elements(&[dims.num_tokens, q_width])?;
    let tokens_kv = checked_elements(&[dims.num_tokens, kv_width])?;
    let cache_elements = checked_elements(&[
        dims.num_blocks_total,
        dims.block_size,
        dims.num_kv_heads,
        dims.head_dim,
    ])?;
    let cache_scale_elements =
        checked_elements(&[dims.num_blocks_total, dims.block_size, dims.num_kv_heads])?;
    let kv_element_bytes = if dims.f16_kv { 2 } else { 1 };

    for (name, buffer) in [
        ("gemma4_metal.qkv_scale", buffers.qkv_scale),
        ("gemma4_metal.o_scale", buffers.o_scale),
        ("gemma4_metal.gate_up_scale", buffers.gate_up_scale),
        ("gemma4_metal.down_scale", buffers.down_scale),
        ("gemma4_metal.q_scale_ptr", buffers.q_scale_ptr),
        ("gemma4_metal.kv_scale_ptr", buffers.kv_scale_ptr),
        ("gemma4_metal.cos", buffers.cos),
        ("gemma4_metal.sin", buffers.sin),
        ("gemma4_metal.context_lens", buffers.context_lens),
    ] {
        require_buffer(name, buffer, 4)?;
    }

    let hidden_f16 = checked_bytes(u64::from(dims.hidden), 2)?;
    let head_f16 = checked_bytes(u64::from(dims.head_dim), 2)?;
    require_buffer(
        "gemma4_metal.attn_norm_gamma",
        buffers.attn_norm_gamma,
        hidden_f16,
    )?;
    require_buffer(
        "gemma4_metal.post_attn_norm_gamma",
        buffers.post_attn_norm_gamma,
        hidden_f16,
    )?;
    require_buffer(
        "gemma4_metal.pre_ff_norm_gamma",
        buffers.pre_ff_norm_gamma,
        hidden_f16,
    )?;
    require_buffer(
        "gemma4_metal.post_ff_norm_gamma",
        buffers.post_ff_norm_gamma,
        hidden_f16,
    )?;
    require_buffer("gemma4_metal.q_norm_gamma", buffers.q_norm_gamma, head_f16)?;
    require_buffer("gemma4_metal.k_norm_gamma", buffers.k_norm_gamma, head_f16)?;
    require_buffer("gemma4_metal.layer_scalar", buffers.layer_scalar, 2)?;

    let qkv_weights = checked_elements(&[qkv_width, dims.hidden])?;
    let o_weights = checked_elements(&[dims.hidden, q_width])?;
    let gate_up_weights = checked_elements(&[2, dims.intermediate, dims.hidden])?;
    let down_weights = checked_elements(&[dims.hidden, dims.intermediate])?;
    require_buffer("gemma4_metal.qkv_fp8", buffers.qkv_fp8, qkv_weights)?;
    require_buffer("gemma4_metal.o_fp8", buffers.o_fp8, o_weights)?;
    require_buffer(
        "gemma4_metal.gate_up_fp8",
        buffers.gate_up_fp8,
        gate_up_weights,
    )?;
    require_buffer("gemma4_metal.down_fp8", buffers.down_fp8, down_weights)?;
    require_buffer(
        "gemma4_metal.qkv_f16",
        buffers.qkv_f16,
        checked_bytes(qkv_weights, 2)?,
    )?;
    require_buffer(
        "gemma4_metal.o_f16",
        buffers.o_f16,
        checked_bytes(o_weights, 2)?,
    )?;
    require_buffer(
        "gemma4_metal.gate_up_f16",
        buffers.gate_up_f16,
        checked_bytes(gate_up_weights, 2)?,
    )?;
    require_buffer(
        "gemma4_metal.down_f16",
        buffers.down_f16,
        checked_bytes(down_weights, 2)?,
    )?;

    require_buffer("gemma4_metal.hidden_fp8", buffers.hidden_fp8, tokens_hidden)?;
    require_buffer(
        "gemma4_metal.hidden_scale",
        buffers.hidden_scale,
        checked_bytes(u64::from(dims.num_tokens), 4)?,
    )?;
    for (name, buffer, elements) in [
        ("gemma4_metal.q_out", buffers.q_out, tokens_q),
        ("gemma4_metal.q_normed", buffers.q_normed, tokens_q),
        ("gemma4_metal.k_out", buffers.k_out, tokens_kv),
        ("gemma4_metal.v_out", buffers.v_out, tokens_kv),
        ("gemma4_metal.k_normed", buffers.k_normed, tokens_kv),
        ("gemma4_metal.v_normed", buffers.v_normed, tokens_kv),
        ("gemma4_metal.attn_out", buffers.attn_out, tokens_q),
        ("gemma4_metal.delta_f16", buffers.delta_f16, tokens_hidden),
        ("gemma4_metal.residual", buffers.residual, tokens_hidden),
    ] {
        require_buffer(name, buffer, checked_bytes(elements, 2)?)?;
    }
    require_buffer("gemma4_metal.q_fp8", buffers.q_fp8, tokens_q)?;
    require_buffer(
        "gemma4_metal.k_cache",
        buffers.k_cache,
        checked_bytes(cache_elements, kv_element_bytes)?,
    )?;
    require_buffer(
        "gemma4_metal.v_cache",
        buffers.v_cache,
        checked_bytes(cache_elements, kv_element_bytes)?,
    )?;
    require_buffer(
        "gemma4_metal.k_scale_cache",
        buffers.k_scale_cache,
        checked_bytes(cache_scale_elements, 4)?,
    )?;
    require_buffer(
        "gemma4_metal.v_scale_cache",
        buffers.v_scale_cache,
        checked_bytes(cache_scale_elements, 4)?,
    )?;
    require_buffer(
        "gemma4_metal.q_scale_cache",
        buffers.q_scale_cache,
        checked_bytes(checked_elements(&[dims.num_tokens, dims.num_heads])?, 4)?,
    )?;
    require_buffer("gemma4_metal.attn_out_fp8", buffers.attn_out_fp8, tokens_q)?;
    require_buffer(
        "gemma4_metal.attn_out_scale",
        buffers.attn_out_scale,
        checked_bytes(u64::from(dims.num_tokens), 4)?,
    )?;
    require_buffer(
        "gemma4_metal.gate_up_out",
        buffers.gate_up_out,
        checked_bytes(
            checked_elements(&[dims.num_tokens, 2, dims.intermediate])?,
            2,
        )?,
    )?;
    require_buffer(
        "gemma4_metal.gate_up_out_fp8",
        buffers.gate_up_out_fp8,
        checked_elements(&[dims.num_tokens, dims.intermediate])?,
    )?;
    require_buffer(
        "gemma4_metal.gate_up_out_scale",
        buffers.gate_up_out_scale,
        checked_bytes(u64::from(dims.num_tokens), 4)?,
    )?;
    require_buffer(
        "gemma4_metal.mlp_out_fp8",
        buffers.mlp_out_fp8,
        tokens_hidden,
    )?;
    require_buffer(
        "gemma4_metal.mlp_out_scale",
        buffers.mlp_out_scale,
        checked_bytes(u64::from(dims.num_tokens), 4)?,
    )?;
    require_buffer(
        "gemma4_metal.positions",
        buffers.positions,
        checked_bytes(u64::from(dims.num_tokens), 4)?,
    )?;
    require_buffer(
        "gemma4_metal.slot_mapping",
        buffers.slot_mapping,
        checked_bytes(u64::from(dims.num_tokens), 4)?,
    )?;
    require_buffer(
        "gemma4_metal.block_tables",
        buffers.block_tables,
        checked_bytes(u64::from(dims.max_blocks_per_seq), 4)?,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemma4_forward_metal_impl(
    dims: Gemma4LayerDims,
    _layer_idx: u32,
    _scratch: &Gemma4LayerScratch,
    _meta: &Gemma4MetadataPtrs,
    _residual: u64,
    _device: &MetalDevice,
    _kernels: &MetalKernels,
    _weight_cache: &MetalWeightCache,
    buffers: &Gemma4LayerMetalBuffers<'_>,
) -> Result<()> {
    validate_metal_buffers(dims, buffers)?;
    if !gemma4_metal_end_to_end_available() {
        return Err(metal_error(
            "gemma4_metal.end_to_end_parity_not_established",
            true,
        ));
    }
    Err(metal_error(
        "gemma4_metal.forward_dispatch_not_implemented",
        true,
    ))
}
