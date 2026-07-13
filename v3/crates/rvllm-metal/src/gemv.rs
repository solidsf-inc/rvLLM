// Copyright 2026 m0at
//
// Licensed under the Apache License, Version 2.0.

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::ffi::c_void;
use std::mem;
use std::sync::Arc;

use metal::{
    Buffer, CommandBufferRef, ComputeCommandEncoderRef, MTLCommandBufferStatus, MTLResourceOptions,
    MTLSize,
};
use rvllm_core::DType;

use crate::device::{MetalDevice, MetalKernelError};
use crate::kernels::MetalKernels;

const GEMV_THREADS: u64 = 256;
const MANY_GEMV_MAX_SPECS: usize = 3;
const LM_HEAD_ARGMAX_MAX_PARTIALS: u32 = 8192;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleLayout {
    PerRow = 0,
    BlockRow128 = 1,
    Single = 2,
}

#[derive(Clone, Copy)]
pub struct Fp8GemvInput<'a> {
    pub weight: &'a Buffer,
    pub weight_offset: u64,
    pub scale: &'a Buffer,
    pub scale_offset: u64,
    pub scale_dtype: DType,
    pub scale_layout: ScaleLayout,
    pub scale_stride: u32,
    pub rows: usize,
    pub cols: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GemvParams {
    rows: u32,
    cols: u32,
    scale_layout: u32,
    scale_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Bf16ArgmaxGemvParams {
    rows: u32,
    cols: u32,
    rows_per_group: u32,
    partial_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LmHeadNllParams {
    rows: u32,
    softcap: f32,
    _pad0: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GeluMulParams {
    rows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HostF32AttentionParams {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    len: u32,
    scale: f32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ManyGemvSpecParams {
    rows: u32,
    cols: u32,
    row_offset: u32,
    scale_layout: u32,
    scale_stride: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ManyGemvParams {
    total_rows: u32,
    spec_count: u32,
    _pad0: u32,
    _pad1: u32,
    specs: [ManyGemvSpecParams; MANY_GEMV_MAX_SPECS],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ArgmaxResult {
    index: u32,
    score: f32,
}

pub struct MetalGemv {
    device: Arc<MetalDevice>,
    kernels: Arc<MetalKernels>,
    input: Option<Buffer>,
    input_bytes: usize,
    output: Option<Buffer>,
    output_bytes: usize,
    scratch: Option<Buffer>,
    scratch_bytes: usize,
    argmax_partials: Option<Buffer>,
    argmax_partials_bytes: usize,
    argmax_result: Option<Buffer>,
    argmax_result_bytes: usize,
    nll_result: Option<Buffer>,
    nll_result_bytes: usize,
}

impl MetalGemv {
    pub fn new(device: Arc<MetalDevice>, kernels: Arc<MetalKernels>) -> Self {
        Self {
            device,
            kernels,
            input: None,
            input_bytes: 0,
            output: None,
            output_bytes: 0,
            scratch: None,
            scratch_bytes: 0,
            argmax_partials: None,
            argmax_partials_bytes: 0,
            argmax_result: None,
            argmax_result_bytes: 0,
            nll_result: None,
            nll_result_bytes: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fp8_row_scaled_f32(
        &mut self,
        weight: &Buffer,
        weight_offset: u64,
        scale: &Buffer,
        scale_offset: u64,
        scale_dtype: DType,
        scale_layout: ScaleLayout,
        scale_stride: u32,
        x: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, MetalKernelError> {
        let mut out = Vec::new();
        self.fp8_row_scaled_f32_into(
            weight,
            weight_offset,
            scale,
            scale_offset,
            scale_dtype,
            scale_layout,
            scale_stride,
            x,
            rows,
            cols,
            &mut out,
        )?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fp8_row_scaled_f32_into(
        &mut self,
        weight: &Buffer,
        weight_offset: u64,
        scale: &Buffer,
        scale_offset: u64,
        scale_dtype: DType,
        scale_layout: ScaleLayout,
        scale_stride: u32,
        x: &[f32],
        rows: usize,
        cols: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), MetalKernelError> {
        validate_dims(rows, cols, x.len(), "fp8_row_scaled_f32")?;
        let input = self.ensure_input(x.len())?;
        let output = self.ensure_output(rows)?;
        copy_f32_to_buffer(&input, x)?;

        let cmd = self.device.queue().new_command_buffer().to_owned();
        let enc = cmd.new_compute_command_encoder();
        call_fp8_row_scaled_gemv_f32(
            &self.kernels,
            enc,
            weight,
            weight_offset,
            scale,
            scale_offset,
            scale_dtype,
            scale_layout,
            scale_stride,
            &input,
            &output,
            rows as u32,
            cols as u32,
        )?;
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        ensure_command_completed(&cmd)?;
        read_f32_from_buffer_into(&output, rows, out)
    }

    pub fn bf16_f32(
        &mut self,
        weight: &Buffer,
        weight_offset: u64,
        x: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, MetalKernelError> {
        let mut out = Vec::new();
        self.bf16_f32_into(weight, weight_offset, x, rows, cols, &mut out)?;
        Ok(out)
    }

    pub fn bf16_f32_into(
        &mut self,
        weight: &Buffer,
        weight_offset: u64,
        x: &[f32],
        rows: usize,
        cols: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), MetalKernelError> {
        validate_dims(rows, cols, x.len(), "bf16_f32")?;
        let input = self.ensure_input(x.len())?;
        let output = self.ensure_output(rows)?;
        copy_f32_to_buffer(&input, x)?;

        let cmd = self.device.queue().new_command_buffer().to_owned();
        let enc = cmd.new_compute_command_encoder();
        call_bf16_gemv_f32(
            &self.kernels,
            enc,
            weight,
            weight_offset,
            &input,
            &output,
            rows as u32,
            cols as u32,
        )?;
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        ensure_command_completed(&cmd)?;
        read_f32_from_buffer_into(&output, rows, out)
    }

    pub fn bf16_argmax_f32(
        &mut self,
        weight: &Buffer,
        weight_offset: u64,
        x: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<u32, MetalKernelError> {
        validate_dims(rows, cols, x.len(), "bf16_argmax_f32")?;
        let rows_u32 = rows as u32;
        let rows_per_group = div_ceil_u32(rows_u32, LM_HEAD_ARGMAX_MAX_PARTIALS).max(1);
        let partial_count = div_ceil_u32(rows_u32, rows_per_group);
        let input = self.ensure_input(x.len())?;
        let result = self.ensure_argmax_result()?;
        let partials = if partial_count == 1 {
            result.clone()
        } else {
            self.ensure_argmax_partials(partial_count as usize)?
        };
        copy_f32_to_buffer(&input, x)?;

        let cmd = self.device.queue().new_command_buffer().to_owned();
        let enc = cmd.new_compute_command_encoder();
        call_bf16_lm_head_argmax_gemv(
            &self.kernels,
            enc,
            weight,
            weight_offset,
            &input,
            &partials,
            &result,
            rows_u32,
            cols as u32,
            rows_per_group,
            partial_count,
        )?;
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        ensure_command_completed(&cmd)?;
        Ok(read_argmax_result(&result)?.0)
    }

    pub fn bf16_target_nll_f32(
        &mut self,
        weight: &Buffer,
        weight_offset: u64,
        x: &[f32],
        rows: usize,
        cols: usize,
        target: u32,
        softcap: f32,
    ) -> Result<f64, MetalKernelError> {
        validate_dims(rows, cols, x.len(), "bf16_target_nll_f32")?;
        if target as usize >= rows {
            return Err(MetalKernelError::InvalidShape(format!(
                "bf16_target_nll_f32 target {target} >= rows {rows}"
            )));
        }
        let rows_u32 = rows as u32;
        let input = self.ensure_input(x.len())?;
        let logits = self.ensure_output(rows)?;
        let result = self.ensure_nll_result()?;
        copy_f32_to_buffer(&input, x)?;

        let cmd = self.device.queue().new_command_buffer().to_owned();
        let enc = cmd.new_compute_command_encoder();
        call_bf16_gemv_f32(
            &self.kernels,
            enc,
            weight,
            weight_offset,
            &input,
            &logits,
            rows_u32,
            cols as u32,
        )?;
        call_lm_head_logsumexp_f32(&self.kernels, enc, &logits, &result, rows_u32, softcap)?;
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        ensure_command_completed(&cmd)?;
        let lse = read_f32_scalar(&result)? as f64;
        let target_logit = read_f32_at(&logits, target as usize)?;
        let target_score = softcapped_logit_f64(target_logit, softcap);
        Ok(lse - target_score)
    }

    pub fn fp8_many_row_scaled_f32(
        &mut self,
        specs: &[Fp8GemvInput<'_>],
        x: &[f32],
    ) -> Result<Vec<Vec<f32>>, MetalKernelError> {
        let mut out = Vec::new();
        self.fp8_many_row_scaled_f32_into(specs, x, &mut out)?;
        Ok(out)
    }

    pub fn fp8_many_row_scaled_f32_into(
        &mut self,
        specs: &[Fp8GemvInput<'_>],
        x: &[f32],
        out: &mut Vec<Vec<f32>>,
    ) -> Result<(), MetalKernelError> {
        if specs.is_empty() {
            out.clear();
            return Ok(());
        }
        let mut total_rows = 0usize;
        for spec in specs {
            validate_dims(spec.rows, spec.cols, x.len(), "fp8_many_row_scaled_f32")?;
            total_rows = total_rows.checked_add(spec.rows).ok_or_else(|| {
                MetalKernelError::InvalidShape("fp8_many output row overflow".into())
            })?;
        }

        let input = self.ensure_input(x.len())?;
        let output = self.ensure_output(total_rows)?;
        copy_f32_to_buffer(&input, x)?;

        let cmd = self.device.queue().new_command_buffer().to_owned();
        let enc = cmd.new_compute_command_encoder();
        if !try_call_fp8_many_row_scaled_gemv_f32(
            &self.kernels,
            enc,
            specs,
            &input,
            &output,
            0,
            total_rows,
        )? {
            let mut row_offset = 0usize;
            for spec in specs {
                call_fp8_row_scaled_gemv_f32_at(
                    &self.kernels,
                    enc,
                    spec.weight,
                    spec.weight_offset,
                    spec.scale,
                    spec.scale_offset,
                    spec.scale_dtype,
                    spec.scale_layout,
                    spec.scale_stride,
                    &input,
                    &output,
                    (row_offset * mem::size_of::<f32>()) as u64,
                    spec.rows as u32,
                    spec.cols as u32,
                )?;
                row_offset += spec.rows;
            }
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        ensure_command_completed(&cmd)?;

        read_f32_chunks_from_buffer_into(&output, specs, total_rows, out)
    }

    pub fn fp8_many_row_scaled_f32_into_outputs(
        &mut self,
        specs: &[Fp8GemvInput<'_>],
        x: &[f32],
        out: &mut [&mut Vec<f32>],
    ) -> Result<(), MetalKernelError> {
        if specs.is_empty() {
            for dst in out.iter_mut() {
                dst.clear();
            }
            return Ok(());
        }
        if out.len() != specs.len() {
            return Err(MetalKernelError::InvalidShape(format!(
                "fp8_many output count {} != spec count {}",
                out.len(),
                specs.len()
            )));
        }
        let mut total_rows = 0usize;
        for spec in specs {
            validate_dims(spec.rows, spec.cols, x.len(), "fp8_many_row_scaled_f32")?;
            total_rows = total_rows.checked_add(spec.rows).ok_or_else(|| {
                MetalKernelError::InvalidShape("fp8_many output row overflow".into())
            })?;
        }

        let input = self.ensure_input(x.len())?;
        let output = self.ensure_output(total_rows)?;
        copy_f32_to_buffer(&input, x)?;

        let cmd = self.device.queue().new_command_buffer().to_owned();
        let enc = cmd.new_compute_command_encoder();
        if !try_call_fp8_many_row_scaled_gemv_f32(
            &self.kernels,
            enc,
            specs,
            &input,
            &output,
            0,
            total_rows,
        )? {
            let mut row_offset = 0usize;
            for spec in specs {
                call_fp8_row_scaled_gemv_f32_at(
                    &self.kernels,
                    enc,
                    spec.weight,
                    spec.weight_offset,
                    spec.scale,
                    spec.scale_offset,
                    spec.scale_dtype,
                    spec.scale_layout,
                    spec.scale_stride,
                    &input,
                    &output,
                    (row_offset * mem::size_of::<f32>()) as u64,
                    spec.rows as u32,
                    spec.cols as u32,
                )?;
                row_offset += spec.rows;
            }
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        ensure_command_completed(&cmd)?;

        read_f32_chunks_from_buffer_into_outputs(&output, specs, total_rows, out)
    }

    pub fn fp8_gelu_down_f32(
        &mut self,
        gate: &Fp8GemvInput<'_>,
        up: &Fp8GemvInput<'_>,
        down: &Fp8GemvInput<'_>,
        x: &[f32],
    ) -> Result<Vec<f32>, MetalKernelError> {
        let mut out = Vec::new();
        self.fp8_gelu_down_f32_into(gate, up, down, x, &mut out)?;
        Ok(out)
    }

    pub fn fp8_gelu_down_f32_into(
        &mut self,
        gate: &Fp8GemvInput<'_>,
        up: &Fp8GemvInput<'_>,
        down: &Fp8GemvInput<'_>,
        x: &[f32],
        out: &mut Vec<f32>,
    ) -> Result<(), MetalKernelError> {
        validate_dims(gate.rows, gate.cols, x.len(), "fp8_gelu_down_f32 gate")?;
        validate_dims(up.rows, up.cols, x.len(), "fp8_gelu_down_f32 up")?;
        if gate.rows != up.rows {
            return Err(MetalKernelError::InvalidShape(format!(
                "fp8_gelu_down_f32 gate/up rows {} != {}",
                gate.rows, up.rows
            )));
        }
        validate_dims(down.rows, down.cols, gate.rows, "fp8_gelu_down_f32 down")?;

        let input = self.ensure_input(x.len())?;
        let gate_up = self.ensure_output(gate.rows + up.rows)?;
        let down_out = self.ensure_scratch(down.rows)?;
        copy_f32_to_buffer(&input, x)?;

        let cmd = self.device.queue().new_command_buffer().to_owned();
        let enc = cmd.new_compute_command_encoder();
        call_fp8_many_row_scaled_gemv_f32(&self.kernels, enc, &[*gate, *up], &input, &gate_up, 0)?;
        call_fp8_gelu_down_gemv_f32(
            &self.kernels,
            enc,
            down.weight,
            down.weight_offset,
            down.scale,
            down.scale_offset,
            down.scale_dtype,
            down.scale_layout,
            down.scale_stride,
            &gate_up,
            &down_out,
            down.rows as u32,
            down.cols as u32,
        )?;
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        ensure_command_completed(&cmd)?;
        read_f32_from_buffer_into(&down_out, down.rows, out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn host_f32_attention(
        &mut self,
        q: &[f32],
        k_cache: &Buffer,
        v_cache: &Buffer,
        slots: &[u32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
    ) -> Result<Vec<f32>, MetalKernelError> {
        let mut out = Vec::new();
        self.host_f32_attention_into(
            q,
            k_cache,
            v_cache,
            slots,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_dim,
            &mut out,
        )?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn host_f32_attention_into(
        &mut self,
        q: &[f32],
        k_cache: &Buffer,
        v_cache: &Buffer,
        slots: &[u32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), MetalKernelError> {
        if slots.is_empty() {
            return Err(MetalKernelError::InvalidShape(
                "host_f32_attention: empty slot list".into(),
            ));
        }
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 || kv_dim == 0 {
            return Err(MetalKernelError::InvalidShape(format!(
                "host_f32_attention: invalid dims heads={num_heads} kv_heads={num_kv_heads} head_dim={head_dim} kv_dim={kv_dim}"
            )));
        }
        if num_heads % num_kv_heads != 0 {
            return Err(MetalKernelError::InvalidShape(format!(
                "host_f32_attention: num_heads {num_heads} not divisible by num_kv_heads {num_kv_heads}"
            )));
        }
        let q_len = num_heads.checked_mul(head_dim).ok_or_else(|| {
            MetalKernelError::InvalidShape("host attention Q extent overflow".into())
        })?;
        if q.len() != q_len {
            return Err(MetalKernelError::InvalidShape(format!(
                "host_f32_attention: q len {} != {}",
                q.len(),
                q_len
            )));
        }
        let expected_kv_dim = num_kv_heads.checked_mul(head_dim).ok_or_else(|| {
            MetalKernelError::InvalidShape("host attention KV extent overflow".into())
        })?;
        if kv_dim != expected_kv_dim {
            return Err(MetalKernelError::InvalidShape(format!(
                "host_f32_attention: kv_dim {kv_dim} != {}",
                expected_kv_dim
            )));
        }
        let capacity = u64::try_from(kv_dim)
            .ok()
            .and_then(|value| value.checked_mul(4))
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                MetalKernelError::InvalidShape("host attention cache stride overflow".into())
            })?;
        let max_slot = u64::from(*slots.iter().max().unwrap());
        let required_cache = max_slot
            .checked_add(1)
            .and_then(|value| value.checked_mul(capacity))
            .ok_or_else(|| {
                MetalKernelError::InvalidShape("host attention cache extent overflow".into())
            })?;
        require_range(k_cache, 0, required_cache, "host attention K cache")?;
        require_range(v_cache, 0, required_cache, "host attention V cache")?;

        let num_heads_u32 = u32::try_from(num_heads)
            .map_err(|_| MetalKernelError::InvalidShape("num_heads exceeds u32".into()))?;
        let num_kv_heads_u32 = u32::try_from(num_kv_heads)
            .map_err(|_| MetalKernelError::InvalidShape("num_kv_heads exceeds u32".into()))?;
        let head_dim_u32 = u32::try_from(head_dim)
            .map_err(|_| MetalKernelError::InvalidShape("head_dim exceeds u32".into()))?;
        let kv_dim_u32 = u32::try_from(kv_dim)
            .map_err(|_| MetalKernelError::InvalidShape("kv_dim exceeds u32".into()))?;
        let len_u32 = u32::try_from(slots.len())
            .map_err(|_| MetalKernelError::InvalidShape("slot count exceeds u32".into()))?;

        let q_buf = self.ensure_input(q.len())?;
        let out_buf = self.ensure_output(q.len())?;
        let slots_buf = self.ensure_scratch(slots.len())?;
        copy_f32_to_buffer(&q_buf, q)?;
        copy_u32_to_buffer(&slots_buf, slots)?;

        let params = HostF32AttentionParams {
            num_heads: num_heads_u32,
            num_kv_heads: num_kv_heads_u32,
            head_dim: head_dim_u32,
            kv_dim: kv_dim_u32,
            len: len_u32,
            scale: 1.0 / (head_dim as f32).sqrt(),
            _pad1: 0,
            _pad2: 0,
        };

        let cmd = self.device.queue().new_command_buffer().to_owned();
        let enc = cmd.new_compute_command_encoder();
        call_host_f32_attention(
            &self.kernels,
            enc,
            &q_buf,
            k_cache,
            v_cache,
            &slots_buf,
            &out_buf,
            &params,
        )?;
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        ensure_command_completed(&cmd)?;
        read_f32_from_buffer_into(&out_buf, q.len(), out)
    }

    fn ensure_input(&mut self, elems: usize) -> Result<Buffer, MetalKernelError> {
        let bytes = elems
            .checked_mul(mem::size_of::<f32>())
            .ok_or_else(|| MetalKernelError::InvalidShape("input byte size overflow".into()))?;
        if self.input_bytes < bytes {
            self.input = Some(
                self.device
                    .device()
                    .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared),
            );
            self.input_bytes = bytes;
        }
        self.input.as_ref().cloned().ok_or_else(|| {
            MetalKernelError::DispatchFailed("input buffer allocation failed".into())
        })
    }

    fn ensure_output(&mut self, elems: usize) -> Result<Buffer, MetalKernelError> {
        let bytes = elems
            .checked_mul(mem::size_of::<f32>())
            .ok_or_else(|| MetalKernelError::InvalidShape("output byte size overflow".into()))?;
        if self.output_bytes < bytes {
            self.output = Some(
                self.device
                    .device()
                    .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared),
            );
            self.output_bytes = bytes;
        }
        self.output.as_ref().cloned().ok_or_else(|| {
            MetalKernelError::DispatchFailed("output buffer allocation failed".into())
        })
    }

    fn ensure_scratch(&mut self, elems: usize) -> Result<Buffer, MetalKernelError> {
        let bytes = elems
            .checked_mul(mem::size_of::<f32>())
            .ok_or_else(|| MetalKernelError::InvalidShape("scratch byte size overflow".into()))?;
        if self.scratch_bytes < bytes {
            self.scratch = Some(
                self.device
                    .device()
                    .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared),
            );
            self.scratch_bytes = bytes;
        }
        self.scratch.as_ref().cloned().ok_or_else(|| {
            MetalKernelError::DispatchFailed("scratch buffer allocation failed".into())
        })
    }

    fn ensure_argmax_partials(&mut self, elems: usize) -> Result<Buffer, MetalKernelError> {
        let bytes = elems
            .checked_mul(mem::size_of::<ArgmaxResult>())
            .ok_or_else(|| MetalKernelError::InvalidShape("argmax partial byte overflow".into()))?;
        if self.argmax_partials_bytes < bytes {
            self.argmax_partials = Some(
                self.device
                    .device()
                    .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared),
            );
            self.argmax_partials_bytes = bytes;
        }
        self.argmax_partials.as_ref().cloned().ok_or_else(|| {
            MetalKernelError::DispatchFailed("argmax partial allocation failed".into())
        })
    }

    fn ensure_argmax_result(&mut self) -> Result<Buffer, MetalKernelError> {
        let bytes = mem::size_of::<ArgmaxResult>();
        if self.argmax_result_bytes < bytes {
            self.argmax_result = Some(
                self.device
                    .device()
                    .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared),
            );
            self.argmax_result_bytes = bytes;
        }
        self.argmax_result.as_ref().cloned().ok_or_else(|| {
            MetalKernelError::DispatchFailed("argmax result allocation failed".into())
        })
    }

    fn ensure_nll_result(&mut self) -> Result<Buffer, MetalKernelError> {
        let bytes = mem::size_of::<f32>();
        if self.nll_result_bytes < bytes {
            self.nll_result = Some(
                self.device
                    .device()
                    .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared),
            );
            self.nll_result_bytes = bytes;
        }
        self.nll_result
            .as_ref()
            .cloned()
            .ok_or_else(|| MetalKernelError::DispatchFailed("nll result allocation failed".into()))
    }
}

#[allow(clippy::too_many_arguments)]
fn try_call_fp8_many_row_scaled_gemv_f32(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    specs: &[Fp8GemvInput<'_>],
    input: &Buffer,
    output: &Buffer,
    output_offset: u64,
    total_rows: usize,
) -> Result<bool, MetalKernelError> {
    if specs.len() < 2 || specs.len() > MANY_GEMV_MAX_SPECS {
        return Ok(false);
    }
    let scale_dtype = specs[0].scale_dtype;
    if !matches!(scale_dtype, DType::Bf16 | DType::F32)
        || specs.iter().any(|s| s.scale_dtype != scale_dtype)
    {
        return Ok(false);
    }

    let params = make_many_gemv_params(specs, total_rows)?;
    for spec in specs {
        validate_fp8_gemv_buffers(
            spec.weight,
            spec.weight_offset,
            spec.scale,
            spec.scale_offset,
            spec.scale_dtype,
            spec.scale_layout,
            spec.scale_stride,
            input,
            output,
            output_offset,
            u32::try_from(spec.rows)
                .map_err(|_| MetalKernelError::InvalidShape("rows exceed u32".into()))?,
            u32::try_from(spec.cols)
                .map_err(|_| MetalKernelError::InvalidShape("cols exceed u32".into()))?,
        )?;
    }
    let name = match scale_dtype {
        DType::Bf16 => "rvllm_fp8_many_gemv_bf16scale_f32",
        DType::F32 => "rvllm_fp8_many_gemv_f32scale_f32",
        _ => unreachable!("filtered above"),
    };

    let pipeline = kernels.pipeline(name)?;
    require_threads(&pipeline, GEMV_THREADS, "fp8 many GEMV")?;
    encoder.set_compute_pipeline_state(&pipeline);
    for i in 0..MANY_GEMV_MAX_SPECS {
        let spec = specs.get(i).unwrap_or(&specs[0]);
        encoder.set_buffer((i * 2) as u64, Some(spec.weight), spec.weight_offset);
        encoder.set_buffer((i * 2 + 1) as u64, Some(spec.scale), spec.scale_offset);
    }
    encoder.set_buffer(6, Some(input), 0);
    encoder.set_buffer(7, Some(output), output_offset);
    encoder.set_bytes(
        8,
        mem::size_of::<ManyGemvParams>() as u64,
        (&params as *const ManyGemvParams).cast::<c_void>(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: total_rows as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: GEMV_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(true)
}

pub fn call_fp8_many_row_scaled_gemv_f32(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    specs: &[Fp8GemvInput<'_>],
    input: &Buffer,
    output: &Buffer,
    output_offset: u64,
) -> Result<(), MetalKernelError> {
    if specs.is_empty() {
        return Ok(());
    }

    let cols = specs[0].cols;
    let mut total_rows = 0usize;
    for spec in specs {
        if spec.cols != cols {
            return Err(MetalKernelError::InvalidShape(format!(
                "fp8_many cols mismatch: {} != {cols}",
                spec.cols
            )));
        }
        validate_dims(
            spec.rows,
            spec.cols,
            cols,
            "call_fp8_many_row_scaled_gemv_f32",
        )?;
        total_rows = total_rows
            .checked_add(spec.rows)
            .ok_or_else(|| MetalKernelError::InvalidShape("fp8_many output row overflow".into()))?;
    }

    let input_bytes = cols
        .checked_mul(mem::size_of::<f32>())
        .ok_or_else(|| MetalKernelError::InvalidShape("fp8_many input byte overflow".into()))?;
    if input.length() < input_bytes as u64 {
        return Err(MetalKernelError::InvalidShape(format!(
            "fp8_many input buffer too small: {} < {input_bytes}",
            input.length()
        )));
    }

    let output_bytes = total_rows
        .checked_mul(mem::size_of::<f32>())
        .ok_or_else(|| MetalKernelError::InvalidShape("fp8_many output byte overflow".into()))?;
    let output_end = output_offset
        .checked_add(output_bytes as u64)
        .ok_or_else(|| MetalKernelError::InvalidShape("fp8_many output offset overflow".into()))?;
    if output.length() < output_end {
        return Err(MetalKernelError::InvalidShape(format!(
            "fp8_many output buffer too small: {} < {output_end}",
            output.length()
        )));
    }

    if !try_call_fp8_many_row_scaled_gemv_f32(
        kernels,
        encoder,
        specs,
        input,
        output,
        output_offset,
        total_rows,
    )? {
        let mut row_offset = 0usize;
        for spec in specs {
            let offset = row_offset
                .checked_mul(mem::size_of::<f32>())
                .and_then(|n| output_offset.checked_add(n as u64))
                .ok_or_else(|| {
                    MetalKernelError::InvalidShape("fp8_many output offset overflow".into())
                })?;
            call_fp8_row_scaled_gemv_f32_at(
                kernels,
                encoder,
                spec.weight,
                spec.weight_offset,
                spec.scale,
                spec.scale_offset,
                spec.scale_dtype,
                spec.scale_layout,
                spec.scale_stride,
                input,
                output,
                offset,
                spec.rows as u32,
                spec.cols as u32,
            )?;
            row_offset += spec.rows;
        }
    }

    Ok(())
}

fn make_many_gemv_params(
    specs: &[Fp8GemvInput<'_>],
    total_rows: usize,
) -> Result<ManyGemvParams, MetalKernelError> {
    let total_rows_u32 = u32::try_from(total_rows)
        .map_err(|_| MetalKernelError::InvalidShape("fp8_many rows exceed u32".into()))?;
    let empty = ManyGemvSpecParams {
        rows: 0,
        cols: 0,
        row_offset: total_rows_u32,
        scale_layout: ScaleLayout::Single as u32,
        scale_stride: 1,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let mut param_specs = [empty; MANY_GEMV_MAX_SPECS];
    let mut row_offset = 0u32;
    for (idx, spec) in specs.iter().enumerate() {
        let rows = u32::try_from(spec.rows)
            .map_err(|_| MetalKernelError::InvalidShape("fp8_many rows exceed u32".into()))?;
        let cols = u32::try_from(spec.cols)
            .map_err(|_| MetalKernelError::InvalidShape("fp8_many cols exceed u32".into()))?;
        param_specs[idx] = ManyGemvSpecParams {
            rows,
            cols,
            row_offset,
            scale_layout: spec.scale_layout as u32,
            scale_stride: spec.scale_stride,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        row_offset = row_offset
            .checked_add(rows)
            .ok_or_else(|| MetalKernelError::InvalidShape("fp8_many row overflow".into()))?;
    }

    Ok(ManyGemvParams {
        total_rows: total_rows_u32,
        spec_count: specs.len() as u32,
        _pad0: 0,
        _pad1: 0,
        specs: param_specs,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn call_fp8_row_scaled_gemv_f32(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    weight: &Buffer,
    weight_offset: u64,
    scale: &Buffer,
    scale_offset: u64,
    scale_dtype: DType,
    scale_layout: ScaleLayout,
    scale_stride: u32,
    input: &Buffer,
    output: &Buffer,
    rows: u32,
    cols: u32,
) -> Result<(), MetalKernelError> {
    call_fp8_row_scaled_gemv_f32_at(
        kernels,
        encoder,
        weight,
        weight_offset,
        scale,
        scale_offset,
        scale_dtype,
        scale_layout,
        scale_stride,
        input,
        output,
        0,
        rows,
        cols,
    )
}

#[allow(clippy::too_many_arguments)]
fn call_fp8_row_scaled_gemv_f32_at(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    weight: &Buffer,
    weight_offset: u64,
    scale: &Buffer,
    scale_offset: u64,
    scale_dtype: DType,
    scale_layout: ScaleLayout,
    scale_stride: u32,
    input: &Buffer,
    output: &Buffer,
    output_offset: u64,
    rows: u32,
    cols: u32,
) -> Result<(), MetalKernelError> {
    if rows == 0 || cols == 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "fp8 gemv rows={rows} cols={cols}"
        )));
    }
    let name = match scale_dtype {
        DType::Bf16 => "rvllm_fp8_gemv_bf16scale_f32",
        DType::F32 => "rvllm_fp8_gemv_f32scale_f32",
        _ => {
            return Err(MetalKernelError::InvalidShape(format!(
                "fp8 gemv scale dtype {scale_dtype:?}"
            )))
        }
    };
    let params = GemvParams {
        rows,
        cols,
        scale_layout: scale_layout as u32,
        scale_stride,
    };
    validate_fp8_gemv_buffers(
        weight,
        weight_offset,
        scale,
        scale_offset,
        scale_dtype,
        scale_layout,
        scale_stride,
        input,
        output,
        output_offset,
        rows,
        cols,
    )?;

    let pipeline = kernels.pipeline(name)?;
    require_threads(&pipeline, GEMV_THREADS, "FP8 GEMV")?;
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(weight), weight_offset);
    encoder.set_buffer(1, Some(scale), scale_offset);
    encoder.set_buffer(2, Some(input), 0);
    encoder.set_buffer(3, Some(output), output_offset);
    encoder.set_bytes(
        4,
        mem::size_of::<GemvParams>() as u64,
        (&params as *const GemvParams).cast::<c_void>(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: GEMV_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn call_bf16_gemv_f32(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    weight: &Buffer,
    weight_offset: u64,
    input: &Buffer,
    output: &Buffer,
    rows: u32,
    cols: u32,
) -> Result<(), MetalKernelError> {
    if rows == 0 || cols == 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "bf16 gemv rows={rows} cols={cols}"
        )));
    }
    let params = GemvParams {
        rows,
        cols,
        scale_layout: ScaleLayout::Single as u32,
        scale_stride: 1,
    };
    let weight_bytes = u64::from(rows)
        .checked_mul(u64::from(cols))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| MetalKernelError::InvalidShape("BF16 weight extent overflow".into()))?;
    require_range(weight, weight_offset, weight_bytes, "BF16 GEMV weight")?;
    require_range(input, 0, u64::from(cols) * 4, "BF16 GEMV input")?;
    require_range(output, 0, u64::from(rows) * 4, "BF16 GEMV output")?;

    let pipeline = kernels.pipeline("rvllm_bf16_gemv_f32")?;
    require_threads(&pipeline, GEMV_THREADS, "BF16 GEMV")?;
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(weight), weight_offset);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_bytes(
        3,
        mem::size_of::<GemvParams>() as u64,
        (&params as *const GemvParams).cast::<c_void>(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: GEMV_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn call_gelu_tanh_mul_f32(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    gate_up: &Buffer,
    rows: u32,
) -> Result<(), MetalKernelError> {
    if rows == 0 {
        return Err(MetalKernelError::InvalidShape(
            "gelu_tanh_mul rows=0".into(),
        ));
    }
    let bytes = (rows as u64)
        .checked_mul(2)
        .and_then(|n| n.checked_mul(mem::size_of::<f32>() as u64))
        .ok_or_else(|| MetalKernelError::InvalidShape("gelu_tanh_mul byte overflow".into()))?;
    if gate_up.length() < bytes {
        return Err(MetalKernelError::InvalidShape(format!(
            "gelu_tanh_mul buffer too small: {} < {bytes}",
            gate_up.length()
        )));
    }
    let params = GeluMulParams {
        rows,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let pipeline = kernels.pipeline("rvllm_gelu_tanh_mul_f32")?;
    require_threads(&pipeline, GEMV_THREADS, "GELU multiply")?;
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(gate_up), 0);
    encoder.set_bytes(
        1,
        mem::size_of::<GeluMulParams>() as u64,
        (&params as *const GeluMulParams).cast::<c_void>(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: div_ceil_u32(rows, GEMV_THREADS as u32) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: GEMV_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn call_fp8_gelu_down_gemv_f32(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    weight: &Buffer,
    weight_offset: u64,
    scale: &Buffer,
    scale_offset: u64,
    scale_dtype: DType,
    scale_layout: ScaleLayout,
    scale_stride: u32,
    gate_up: &Buffer,
    output: &Buffer,
    rows: u32,
    cols: u32,
) -> Result<(), MetalKernelError> {
    if rows == 0 || cols == 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "fp8 gelu down rows={rows} cols={cols}"
        )));
    }
    let gate_up_bytes = (cols as u64)
        .checked_mul(2)
        .and_then(|n| n.checked_mul(mem::size_of::<f32>() as u64))
        .ok_or_else(|| MetalKernelError::InvalidShape("fp8 gelu down input overflow".into()))?;
    let output_bytes = (rows as u64)
        .checked_mul(mem::size_of::<f32>() as u64)
        .ok_or_else(|| MetalKernelError::InvalidShape("fp8 gelu down output overflow".into()))?;
    if gate_up.length() < gate_up_bytes {
        return Err(MetalKernelError::InvalidShape(format!(
            "fp8 gelu down input buffer too small: {} < {gate_up_bytes}",
            gate_up.length()
        )));
    }
    if output.length() < output_bytes {
        return Err(MetalKernelError::InvalidShape(format!(
            "fp8 gelu down output buffer too small: {} < {output_bytes}",
            output.length()
        )));
    }

    let name = match scale_dtype {
        DType::Bf16 => "rvllm_fp8_gelu_down_bf16scale_f32",
        DType::F32 => "rvllm_fp8_gelu_down_f32scale_f32",
        _ => {
            return Err(MetalKernelError::InvalidShape(format!(
                "fp8 gelu down scale dtype {scale_dtype:?}"
            )))
        }
    };
    let params = GemvParams {
        rows,
        cols,
        scale_layout: scale_layout as u32,
        scale_stride,
    };
    validate_fp8_gemv_buffers(
        weight,
        weight_offset,
        scale,
        scale_offset,
        scale_dtype,
        scale_layout,
        scale_stride,
        gate_up,
        output,
        0,
        rows,
        cols,
    )?;

    let pipeline = kernels.pipeline(name)?;
    require_threads(&pipeline, GEMV_THREADS, "FP8 GELU-down GEMV")?;
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(weight), weight_offset);
    encoder.set_buffer(1, Some(scale), scale_offset);
    encoder.set_buffer(2, Some(gate_up), 0);
    encoder.set_buffer(3, Some(output), 0);
    encoder.set_bytes(
        4,
        mem::size_of::<GemvParams>() as u64,
        (&params as *const GemvParams).cast::<c_void>(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: GEMV_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn call_host_f32_attention(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    q: &Buffer,
    k_cache: &Buffer,
    v_cache: &Buffer,
    slots: &Buffer,
    out: &Buffer,
    params: &HostF32AttentionParams,
) -> Result<(), MetalKernelError> {
    if params.len == 0
        || params.num_heads == 0
        || params.num_kv_heads == 0
        || params.head_dim == 0
        || params.kv_dim == 0
    {
        return Err(MetalKernelError::InvalidShape(format!(
            "host_f32_attention invalid dims heads={} kv_heads={} head_dim={} kv_dim={} len={}",
            params.num_heads, params.num_kv_heads, params.head_dim, params.kv_dim, params.len
        )));
    }
    let q_elems = (params.num_heads as u64) * (params.head_dim as u64);
    let q_bytes = q_elems
        .checked_mul(mem::size_of::<f32>() as u64)
        .ok_or_else(|| MetalKernelError::InvalidShape("host attention q overflow".into()))?;
    let slots_bytes = (params.len as u64)
        .checked_mul(mem::size_of::<u32>() as u64)
        .ok_or_else(|| MetalKernelError::InvalidShape("host attention slots overflow".into()))?;
    if q.length() < q_bytes {
        return Err(MetalKernelError::InvalidShape(format!(
            "host attention q buffer too small: {} < {q_bytes}",
            q.length()
        )));
    }
    if out.length() < q_bytes {
        return Err(MetalKernelError::InvalidShape(format!(
            "host attention out buffer too small: {} < {q_bytes}",
            out.length()
        )));
    }
    if slots.length() < slots_bytes {
        return Err(MetalKernelError::InvalidShape(format!(
            "host attention slots buffer too small: {} < {slots_bytes}",
            slots.length()
        )));
    }
    if k_cache.length() == 0 || v_cache.length() == 0 {
        return Err(MetalKernelError::InvalidShape(
            "host attention empty KV cache".into(),
        ));
    }

    let pipeline = kernels.pipeline("rvllm_host_f32_attention")?;
    require_threads(&pipeline, GEMV_THREADS, "host F32 attention")?;
    let dynamic_memory = u64::from(params.len)
        .checked_add(GEMV_THREADS)
        .and_then(|value| value.checked_mul(mem::size_of::<f32>() as u64))
        .ok_or_else(|| {
            MetalKernelError::InvalidShape("host attention workspace overflow".into())
        })?;
    let total_memory = dynamic_memory
        .checked_add(pipeline.static_threadgroup_memory_length() as u64)
        .ok_or_else(|| {
            MetalKernelError::InvalidShape("host attention workspace overflow".into())
        })?;
    let max_memory = kernels.library().device().max_threadgroup_memory_length() as u64;
    if total_memory > max_memory {
        return Err(MetalKernelError::InvalidShape(format!(
            "host attention needs {total_memory} bytes of threadgroup memory, device supports {max_memory}"
        )));
    }
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(q), 0);
    encoder.set_buffer(1, Some(k_cache), 0);
    encoder.set_buffer(2, Some(v_cache), 0);
    encoder.set_buffer(3, Some(slots), 0);
    encoder.set_buffer(4, Some(out), 0);
    encoder.set_bytes(
        5,
        mem::size_of::<HostF32AttentionParams>() as u64,
        (params as *const HostF32AttentionParams).cast::<c_void>(),
    );
    encoder.set_threadgroup_memory_length(0, dynamic_memory);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: params.num_heads as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: GEMV_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn call_lm_head_logsumexp_f32(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    logits: &Buffer,
    result: &Buffer,
    rows: u32,
    softcap: f32,
) -> Result<(), MetalKernelError> {
    if rows == 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "lm_head logsumexp rows={rows}"
        )));
    }
    if !softcap.is_finite() || softcap < 0.0 {
        return Err(MetalKernelError::InvalidShape(
            "lm_head softcap must be finite and >= 0".into(),
        ));
    }
    let logits_bytes = (rows as u64)
        .checked_mul(mem::size_of::<f32>() as u64)
        .ok_or_else(|| MetalKernelError::InvalidShape("lm_head nll logits overflow".into()))?;
    if logits.length() < logits_bytes {
        return Err(MetalKernelError::InvalidShape(format!(
            "lm_head nll logits buffer too small: {} < {logits_bytes}",
            logits.length()
        )));
    }
    if result.length() < mem::size_of::<f32>() as u64 {
        return Err(MetalKernelError::InvalidShape(
            "lm_head nll result buffer too small".into(),
        ));
    }
    let params = LmHeadNllParams {
        rows,
        softcap,
        _pad0: 0.0,
        _pad1: 0.0,
    };

    let pipeline = kernels.pipeline("rvllm_lm_head_logsumexp_f32")?;
    require_threads(&pipeline, GEMV_THREADS, "LM-head logsumexp")?;
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(logits), 0);
    encoder.set_buffer(1, Some(result), 0);
    encoder.set_bytes(
        2,
        mem::size_of::<LmHeadNllParams>() as u64,
        (&params as *const LmHeadNllParams).cast::<c_void>(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: GEMV_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_bf16_lm_head_argmax_gemv(
    kernels: &MetalKernels,
    encoder: &ComputeCommandEncoderRef,
    weight: &Buffer,
    weight_offset: u64,
    input: &Buffer,
    partials: &Buffer,
    result: &Buffer,
    rows: u32,
    cols: u32,
    rows_per_group: u32,
    partial_count: u32,
) -> Result<(), MetalKernelError> {
    if rows == 0 || cols == 0 || rows_per_group == 0 || partial_count == 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "bf16 lm_head argmax rows={rows} cols={cols} rows_per_group={rows_per_group} partial_count={partial_count}"
        )));
    }
    if partial_count != div_ceil_u32(rows, rows_per_group)
        || partial_count > LM_HEAD_ARGMAX_MAX_PARTIALS
    {
        return Err(MetalKernelError::InvalidShape(
            "bf16 lm_head argmax partial geometry is inconsistent".into(),
        ));
    }
    let weight_bytes = u64::from(rows)
        .checked_mul(u64::from(cols))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| MetalKernelError::InvalidShape("argmax weight extent overflow".into()))?;
    require_range(weight, weight_offset, weight_bytes, "argmax weight")?;
    require_range(input, 0, u64::from(cols) * 4, "argmax input")?;
    require_range(
        partials,
        0,
        u64::from(partial_count) * mem::size_of::<ArgmaxResult>() as u64,
        "argmax partials",
    )?;
    require_range(
        result,
        0,
        mem::size_of::<ArgmaxResult>() as u64,
        "argmax result",
    )?;
    let params = Bf16ArgmaxGemvParams {
        rows,
        cols,
        rows_per_group,
        partial_count,
    };

    let pipeline = kernels.pipeline("rvllm_bf16_lm_head_argmax_gemv")?;
    require_threads(&pipeline, GEMV_THREADS, "LM-head argmax")?;
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(weight), weight_offset);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(
        2,
        Some(if partial_count == 1 { result } else { partials }),
        0,
    );
    encoder.set_bytes(
        3,
        mem::size_of::<Bf16ArgmaxGemvParams>() as u64,
        (&params as *const Bf16ArgmaxGemvParams).cast::<c_void>(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: partial_count as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: GEMV_THREADS,
            height: 1,
            depth: 1,
        },
    );

    if partial_count == 1 {
        return Ok(());
    }

    let pipeline = kernels.pipeline("rvllm_lm_head_argmax_reduce")?;
    require_threads(&pipeline, GEMV_THREADS, "LM-head argmax reduction")?;
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(partials), 0);
    encoder.set_buffer(1, Some(result), 0);
    encoder.set_bytes(
        2,
        mem::size_of::<Bf16ArgmaxGemvParams>() as u64,
        (&params as *const Bf16ArgmaxGemvParams).cast::<c_void>(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: GEMV_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn validate_dims(
    rows: usize,
    cols: usize,
    input_len: usize,
    op: &'static str,
) -> Result<(), MetalKernelError> {
    if rows == 0 || cols == 0 {
        return Err(MetalKernelError::InvalidShape(format!(
            "{op}: rows={rows} cols={cols}"
        )));
    }
    if cols != input_len {
        return Err(MetalKernelError::InvalidShape(format!(
            "{op}: cols {cols} != input len {input_len}"
        )));
    }
    u32::try_from(rows)
        .map_err(|_| MetalKernelError::InvalidShape(format!("{op}: rows exceed u32")))?;
    u32::try_from(cols)
        .map_err(|_| MetalKernelError::InvalidShape(format!("{op}: cols exceed u32")))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_fp8_gemv_buffers(
    weight: &Buffer,
    weight_offset: u64,
    scale: &Buffer,
    scale_offset: u64,
    scale_dtype: DType,
    scale_layout: ScaleLayout,
    scale_stride: u32,
    input: &Buffer,
    output: &Buffer,
    output_offset: u64,
    rows: u32,
    cols: u32,
) -> Result<(), MetalKernelError> {
    let weight_bytes = u64::from(rows)
        .checked_mul(u64::from(cols))
        .ok_or_else(|| MetalKernelError::InvalidShape("FP8 weight extent overflow".into()))?;
    require_range(weight, weight_offset, weight_bytes, "FP8 GEMV weight")?;
    require_range(input, 0, u64::from(cols) * 4, "FP8 GEMV input")?;
    require_range(
        output,
        output_offset,
        u64::from(rows) * 4,
        "FP8 GEMV output",
    )?;
    let scale_bytes = match scale_dtype {
        DType::Bf16 => 2u64,
        DType::F32 => 4u64,
        _ => {
            return Err(MetalKernelError::InvalidShape(format!(
                "unsupported FP8 scale dtype {scale_dtype:?}"
            )))
        }
    };
    let scale_entries = match scale_layout {
        ScaleLayout::Single => 1u64,
        ScaleLayout::PerRow => {
            if scale_stride == 0 {
                return Err(MetalKernelError::InvalidShape(
                    "per-row scale_stride must be > 0".into(),
                ));
            }
            u64::from(rows - 1)
                .checked_mul(u64::from(scale_stride))
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| MetalKernelError::InvalidShape("scale extent overflow".into()))?
        }
        ScaleLayout::BlockRow128 => {
            if scale_stride == 0 {
                return Err(MetalKernelError::InvalidShape(
                    "block-row scale_stride must be > 0".into(),
                ));
            }
            u64::from((rows - 1) / 128)
                .checked_mul(u64::from(scale_stride))
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| MetalKernelError::InvalidShape("scale extent overflow".into()))?
        }
    };
    require_range(
        scale,
        scale_offset,
        scale_entries
            .checked_mul(scale_bytes)
            .ok_or_else(|| MetalKernelError::InvalidShape("scale bytes overflow".into()))?,
        "FP8 GEMV scales",
    )
}

fn require_range(
    buffer: &Buffer,
    offset: u64,
    bytes: u64,
    label: &str,
) -> Result<(), MetalKernelError> {
    let end = offset
        .checked_add(bytes)
        .ok_or_else(|| MetalKernelError::InvalidShape(format!("{label} offset overflow")))?;
    if end > buffer.length() {
        return Err(MetalKernelError::InvalidShape(format!(
            "{label} range ends at {end}, buffer has {} bytes",
            buffer.length()
        )));
    }
    Ok(())
}

fn require_threads(
    pipeline: &metal::ComputePipelineStateRef,
    requested: u64,
    label: &str,
) -> Result<(), MetalKernelError> {
    if requested > pipeline.max_total_threads_per_threadgroup() as u64 {
        return Err(MetalKernelError::DispatchFailed(format!(
            "{label} needs {requested} threads, pipeline supports {}",
            pipeline.max_total_threads_per_threadgroup()
        )));
    }
    Ok(())
}

fn ensure_command_completed(command: &CommandBufferRef) -> Result<(), MetalKernelError> {
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(MetalKernelError::DispatchFailed(format!(
            "Metal command buffer completed with status {:?}",
            command.status()
        )));
    }
    Ok(())
}

fn div_ceil_u32(n: u32, d: u32) -> u32 {
    ((n - 1) / d) + 1
}

fn copy_f32_to_buffer(buf: &Buffer, x: &[f32]) -> Result<(), MetalKernelError> {
    let bytes = x
        .len()
        .checked_mul(mem::size_of::<f32>())
        .ok_or_else(|| MetalKernelError::InvalidShape("F32 copy size overflow".into()))?;
    require_range(buf, 0, bytes as u64, "F32 upload")?;
    let destination = buf.contents().cast::<f32>();
    if destination.is_null() {
        return Err(MetalKernelError::DispatchFailed(
            "F32 upload buffer is not CPU-addressable".into(),
        ));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), destination, x.len());
    }
    Ok(())
}

fn copy_u32_to_buffer(buf: &Buffer, x: &[u32]) -> Result<(), MetalKernelError> {
    let bytes = x
        .len()
        .checked_mul(mem::size_of::<u32>())
        .ok_or_else(|| MetalKernelError::InvalidShape("U32 copy size overflow".into()))?;
    require_range(buf, 0, bytes as u64, "U32 upload")?;
    let destination = buf.contents().cast::<u32>();
    if destination.is_null() {
        return Err(MetalKernelError::DispatchFailed(
            "U32 upload buffer is not CPU-addressable".into(),
        ));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), destination, x.len());
    }
    Ok(())
}

fn read_f32_from_buffer_into(
    buf: &Buffer,
    len: usize,
    out: &mut Vec<f32>,
) -> Result<(), MetalKernelError> {
    if len == 0 {
        return Err(MetalKernelError::InvalidShape(
            "read_f32_from_buffer len=0".into(),
        ));
    }
    require_range(buf, 0, (len * mem::size_of::<f32>()) as u64, "F32 readback")?;
    let pointer = buf.contents().cast::<f32>();
    if pointer.is_null() {
        return Err(MetalKernelError::DispatchFailed(
            "F32 readback buffer is not CPU-addressable".into(),
        ));
    }
    let src = unsafe { std::slice::from_raw_parts(pointer, len) };
    out.clear();
    out.extend_from_slice(src);
    Ok(())
}

fn read_f32_chunks_from_buffer_into(
    buf: &Buffer,
    specs: &[Fp8GemvInput<'_>],
    total_len: usize,
    out: &mut Vec<Vec<f32>>,
) -> Result<(), MetalKernelError> {
    if total_len == 0 {
        return Err(MetalKernelError::InvalidShape(
            "read_f32_chunks_from_buffer len=0".into(),
        ));
    }
    require_range(buf, 0, (total_len * 4) as u64, "chunked F32 readback")?;
    let pointer = buf.contents().cast::<f32>();
    if pointer.is_null() {
        return Err(MetalKernelError::DispatchFailed(
            "chunked F32 readback buffer is not CPU-addressable".into(),
        ));
    }
    let combined = unsafe { std::slice::from_raw_parts(pointer, total_len) };
    out.truncate(specs.len());
    while out.len() < specs.len() {
        out.push(Vec::new());
    }
    let mut start = 0usize;
    for (dst, spec) in out.iter_mut().zip(specs.iter()) {
        let end = start + spec.rows;
        dst.clear();
        dst.extend_from_slice(&combined[start..end]);
        start = end;
    }
    Ok(())
}

fn read_f32_chunks_from_buffer_into_outputs(
    buf: &Buffer,
    specs: &[Fp8GemvInput<'_>],
    total_len: usize,
    out: &mut [&mut Vec<f32>],
) -> Result<(), MetalKernelError> {
    if total_len == 0 {
        return Err(MetalKernelError::InvalidShape(
            "read_f32_chunks_from_buffer len=0".into(),
        ));
    }
    if out.len() != specs.len() {
        return Err(MetalKernelError::InvalidShape(format!(
            "read_f32_chunks output count {} != spec count {}",
            out.len(),
            specs.len()
        )));
    }
    require_range(buf, 0, (total_len * 4) as u64, "chunked F32 readback")?;
    let pointer = buf.contents().cast::<f32>();
    if pointer.is_null() {
        return Err(MetalKernelError::DispatchFailed(
            "chunked F32 readback buffer is not CPU-addressable".into(),
        ));
    }
    let combined = unsafe { std::slice::from_raw_parts(pointer, total_len) };
    let mut start = 0usize;
    for (dst, spec) in out.iter_mut().zip(specs.iter()) {
        let end = start + spec.rows;
        dst.clear();
        dst.extend_from_slice(&combined[start..end]);
        start = end;
    }
    Ok(())
}

fn read_argmax_result(buf: &Buffer) -> Result<(u32, f32), MetalKernelError> {
    require_range(
        buf,
        0,
        mem::size_of::<ArgmaxResult>() as u64,
        "argmax readback",
    )?;
    let pointer = buf.contents().cast::<ArgmaxResult>();
    if pointer.is_null() {
        return Err(MetalKernelError::DispatchFailed(
            "argmax result is not CPU-addressable".into(),
        ));
    }
    let out = unsafe { *pointer };
    Ok((out.index, out.score))
}

fn read_f32_scalar(buf: &Buffer) -> Result<f32, MetalKernelError> {
    read_f32_at(buf, 0)
}

fn read_f32_at(buf: &Buffer, idx: usize) -> Result<f32, MetalKernelError> {
    let offset = idx
        .checked_mul(mem::size_of::<f32>())
        .ok_or_else(|| MetalKernelError::InvalidShape("F32 read offset overflow".into()))?;
    require_range(buf, offset as u64, 4, "F32 scalar readback")?;
    let pointer = buf.contents().cast::<f32>();
    if pointer.is_null() {
        return Err(MetalKernelError::DispatchFailed(
            "F32 scalar buffer is not CPU-addressable".into(),
        ));
    }
    Ok(unsafe { *pointer.add(idx) })
}

fn softcapped_logit_f64(logit: f32, softcap: f32) -> f64 {
    if softcap > 0.0 {
        let cap = softcap as f64;
        cap * ((logit as f64) / cap).tanh()
    } else {
        logit as f64
    }
}
