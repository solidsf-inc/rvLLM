//! Launcher descriptors for the fused kernel set.
//!
//! Each kernel is a (validated-params struct, launch fn) pair. `launch`
//! builds the kernel argv and calls `launch_raw` with the right
//! grid/block/smem. The argv holds local bindings so every arg address
//! survives until `cuLaunchKernel` returns.

use rvllm_core::{Result, RvllmError, SampleCtx, SamplingError};
use rvllm_kernels::KernelFn;

use crate::launch_raw::launch_raw;

/// Common alignment rule: FP8 and f16 kernels using uint4 loads require
/// the last dim to be a multiple of 8 halves (for f16) or 16 bytes (for
/// u8). Check at validate time — misalignment here → `Err`, not a silent
/// crash under graph replay.
pub fn require_multiple(got: usize, of: usize, what: &'static str) -> Result<()> {
    if of == 0 || got % of != 0 {
        return Err(invalid(what, "must be multiple"));
    }
    Ok(())
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::Sampling {
        err: SamplingError::InvalidParams {
            reason: format!("{field}: {reason}"),
        },
        ctx: SampleCtx {
            op: "validate",
            stream: 0,
        },
    }
}

fn require_ptr(ptr: u64, field: &'static str) -> Result<()> {
    if ptr == 0 {
        return Err(invalid(field, "device pointer must be non-null"));
    }
    Ok(())
}

fn rope_capacities(
    block_size: u32,
    max_blocks_per_seq: u32,
    num_blocks_total: u32,
) -> Result<(i32, i32)> {
    if block_size == 0 || max_blocks_per_seq == 0 || num_blocks_total == 0 {
        return Err(invalid("rope cache layout", "dimensions must be nonzero"));
    }
    let max_positions = max_blocks_per_seq
        .checked_mul(block_size)
        .filter(|&value| value <= i32::MAX as u32)
        .ok_or_else(|| invalid("max_positions", "extent exceeds kernel ABI"))?;
    let num_cache_slots = num_blocks_total
        .checked_mul(block_size)
        .filter(|&value| value <= i32::MAX as u32)
        .ok_or_else(|| invalid("num_cache_slots", "extent exceeds kernel ABI"))?;
    Ok((max_positions as i32, num_cache_slots as i32))
}

// ---------------------------------------------------------------------------
// embedding_gather
// ---------------------------------------------------------------------------

pub struct EmbeddingGatherLaunch {
    pub num_tokens: u32,
    pub hidden: u32,
    pub vocab: u32,
}

impl EmbeddingGatherLaunch {
    pub fn validate(&self) -> Result<()> {
        if self.num_tokens == 0 || self.hidden == 0 || self.vocab == 0 {
            return Err(invalid("embedding_gather", "zero dim"));
        }
        Ok(())
    }

    /// # Safety
    /// Caller owns the device pointers for the kernel's duration.
    pub unsafe fn launch(
        &self,
        kernel: &KernelFn,
        out_ptr: u64,
        weight_ptr: u64,
        token_ids_ptr: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut out_ptr = out_ptr;
        let mut weight_ptr = weight_ptr;
        let mut token_ids_ptr = token_ids_ptr;
        let mut hidden = self.hidden as i32;
        let mut vocab = self.vocab as i32;
        let args = [
            (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut weight_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut token_ids_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut hidden) as *mut i32 as *mut core::ffi::c_void,
            (&mut vocab) as *mut i32 as *mut core::ffi::c_void,
        ];
        let block = (self.hidden.min(1024), 1, 1);
        let grid = (self.num_tokens, 1, 1);
        launch_raw(kernel, grid, block, 0, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// fused_add_rmsnorm_fp8_quant
// ---------------------------------------------------------------------------

pub struct FusedAddRmsnormFp8QuantLaunch {
    pub num_tokens: u32,
    pub hidden: u32,
    pub eps: f32,
}

impl FusedAddRmsnormFp8QuantLaunch {
    pub fn validate(&self) -> Result<()> {
        require_multiple(self.hidden as usize, 8, "hidden")?;
        if self.num_tokens == 0 {
            return Err(invalid("num_tokens", "must be > 0"));
        }
        Ok(())
    }

    /// Kernel sig: `(out_fp8, scale, residual_out, in_hidden,
    /// residual_in, gamma, eps, hidden)`.
    ///
    /// # Safety
    /// Caller owns pointers for the call's duration.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        kernel: &KernelFn,
        out_fp8: u64,
        scale: u64,
        residual_out: u64,
        in_hidden: u64,
        residual_in: u64,
        gamma: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut out_fp8 = out_fp8;
        let mut scale = scale;
        let mut residual_out = residual_out;
        let mut in_hidden = in_hidden;
        let mut residual_in = residual_in;
        let mut gamma = gamma;
        let mut eps = self.eps;
        let mut hidden = self.hidden as i32;
        let args = [
            (&mut out_fp8) as *mut u64 as *mut core::ffi::c_void,
            (&mut scale) as *mut u64 as *mut core::ffi::c_void,
            (&mut residual_out) as *mut u64 as *mut core::ffi::c_void,
            (&mut in_hidden) as *mut u64 as *mut core::ffi::c_void,
            (&mut residual_in) as *mut u64 as *mut core::ffi::c_void,
            (&mut gamma) as *mut u64 as *mut core::ffi::c_void,
            (&mut eps) as *mut f32 as *mut core::ffi::c_void,
            (&mut hidden) as *mut i32 as *mut core::ffi::c_void,
        ];
        const SMEM: u32 = 32 * 4; // WARPS_MAX * sizeof(float)
        let block = (self.hidden.min(1024), 1, 1);
        let grid = (self.num_tokens, 1, 1);
        launch_raw(kernel, grid, block, SMEM, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// fused_rmsnorm_fp8_quant (no residual variant — first layer input path)
// ---------------------------------------------------------------------------

pub struct FusedRmsnormFp8QuantLaunch {
    pub num_tokens: u32,
    pub hidden: u32,
    pub eps: f32,
}

impl FusedRmsnormFp8QuantLaunch {
    pub fn validate(&self) -> Result<()> {
        require_multiple(self.hidden as usize, 8, "hidden")?;
        if self.num_tokens == 0 {
            return Err(invalid("num_tokens", "must be > 0"));
        }
        Ok(())
    }

    /// Kernel sig: `(out_fp8, scale, in_hidden, gamma, eps, hidden)`.
    ///
    /// # Safety
    /// Device pointers must outlive the call.
    pub unsafe fn launch(
        &self,
        kernel: &KernelFn,
        out_fp8: u64,
        scale: u64,
        in_hidden: u64,
        gamma: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut out_fp8 = out_fp8;
        let mut scale = scale;
        let mut in_hidden = in_hidden;
        let mut gamma = gamma;
        let mut eps = self.eps;
        let mut hidden = self.hidden as i32;
        let args = [
            (&mut out_fp8) as *mut u64 as *mut core::ffi::c_void,
            (&mut scale) as *mut u64 as *mut core::ffi::c_void,
            (&mut in_hidden) as *mut u64 as *mut core::ffi::c_void,
            (&mut gamma) as *mut u64 as *mut core::ffi::c_void,
            (&mut eps) as *mut f32 as *mut core::ffi::c_void,
            (&mut hidden) as *mut i32 as *mut core::ffi::c_void,
        ];
        const SMEM: u32 = 32 * 4;
        let block = (self.hidden.min(1024), 1, 1);
        let grid = (self.num_tokens, 1, 1);
        launch_raw(kernel, grid, block, SMEM, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// quantize_fp8_per_token
// ---------------------------------------------------------------------------

pub struct QuantizeFp8PerTokenLaunch {
    pub num_tokens: u32,
    pub dim: u32,
}

impl QuantizeFp8PerTokenLaunch {
    pub fn validate(&self) -> Result<()> {
        require_multiple(self.dim as usize, 8, "dim")?;
        if self.num_tokens == 0 {
            return Err(invalid("num_tokens", "must be > 0"));
        }
        if self.dim > 65536 {
            return Err(invalid("dim", "must be <= 65536"));
        }
        Ok(())
    }

    /// Kernel sig: `(out_fp8, scale, in_f16, dim)`.
    ///
    /// # Safety
    /// Caller owns pointers for the call's duration.
    pub unsafe fn launch(
        &self,
        kernel: &KernelFn,
        out_fp8: u64,
        scale: u64,
        in_f16: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut out_fp8 = out_fp8;
        let mut scale = scale;
        let mut in_f16 = in_f16;
        let mut dim = self.dim as i32;
        let args = [
            (&mut out_fp8) as *mut u64 as *mut core::ffi::c_void,
            (&mut scale) as *mut u64 as *mut core::ffi::c_void,
            (&mut in_f16) as *mut u64 as *mut core::ffi::c_void,
            (&mut dim) as *mut i32 as *mut core::ffi::c_void,
        ];
        const SMEM: u32 = 32 * 4;
        let block = (self.dim.min(1024).max(256), 1, 1);
        let grid = (self.num_tokens, 1, 1);
        launch_raw(kernel, grid, block, SMEM, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// fused_silu_mul_fp8_quant
// ---------------------------------------------------------------------------

pub struct FusedSiluMulFp8QuantLaunch {
    pub num_tokens: u32,
    pub intermediate: u32,
}

impl FusedSiluMulFp8QuantLaunch {
    pub fn validate(&self) -> Result<()> {
        require_multiple(self.intermediate as usize, 8, "intermediate")?;
        if self.num_tokens == 0 {
            return Err(invalid("num_tokens", "must be > 0"));
        }
        Ok(())
    }

    /// Kernel sig (v2 layout): `(out_fp8, scale, gate_up_f16, intermediate)`.
    /// Grid.x is num_tokens; kernel reads blockIdx.x as row.
    ///
    /// # Safety
    /// Caller owns pointers for the call's duration.
    pub unsafe fn launch(
        &self,
        kernel: &KernelFn,
        out_fp8: u64,
        scale: u64,
        gate_up: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut out_fp8 = out_fp8;
        let mut scale = scale;
        let mut gate_up = gate_up;
        let mut intermediate = self.intermediate as i32;
        let args = [
            (&mut out_fp8) as *mut u64 as *mut core::ffi::c_void,
            (&mut scale) as *mut u64 as *mut core::ffi::c_void,
            (&mut gate_up) as *mut u64 as *mut core::ffi::c_void,
            (&mut intermediate) as *mut i32 as *mut core::ffi::c_void,
        ];
        const SMEM: u32 = 32 * 4;
        let block = (self.intermediate.min(1024), 1, 1);
        let grid = (self.num_tokens, 1, 1);
        launch_raw(kernel, grid, block, SMEM, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// argmax
// ---------------------------------------------------------------------------

const ARGMAX_GRID_F32_KERNEL: &str = "argmax_grid_f32_kernel";
const ARGMAX_GRID_BLOCK: u32 = 256;
const ARGMAX_GRID_ELEMS_PER_CTA: u32 = 1024;

pub struct ArgmaxLaunch {
    pub num_tokens: u32,
    pub vocab: u32,
}

impl ArgmaxLaunch {
    pub fn validate(&self) -> Result<()> {
        if self.vocab == 0 {
            return Err(invalid("vocab", "must be > 0"));
        }
        if self.num_tokens == 0 {
            return Err(invalid("num_tokens", "must be > 0"));
        }
        Ok(())
    }

    /// Kernel sig: `(logits_f32, out_i32, vocab)`.
    ///
    /// # Safety
    /// Caller owns pointers for the call's duration.
    pub unsafe fn launch(
        &self,
        kernel: &KernelFn,
        logits_ptr: u64,
        out_ptr: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut logits_ptr = logits_ptr;
        let mut out_ptr = out_ptr;
        let mut vocab = self.vocab as i32;
        let args = [
            (&mut logits_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut vocab) as *mut i32 as *mut core::ffi::c_void,
        ];
        let (grid, block) = if kernel.name() == ARGMAX_GRID_F32_KERNEL {
            #[cfg(feature = "cuda")]
            {
                use cudarc::driver::sys::*;
                let r = cuMemsetD32Async(
                    out_ptr as CUdeviceptr,
                    0,
                    self.num_tokens as usize,
                    stream as CUstream,
                );
                if r != CUresult::CUDA_SUCCESS {
                    return Err(RvllmError::cuda(
                        "cuMemsetD32Async",
                        rvllm_core::CudaErrorKind::LaunchFailed,
                        rvllm_core::CudaCtx {
                            stream,
                            kernel: kernel.name(),
                            launch: None,
                            device: -1,
                        },
                    ));
                }
            }
            let parts = (self.vocab + ARGMAX_GRID_ELEMS_PER_CTA - 1) / ARGMAX_GRID_ELEMS_PER_CTA;
            ((parts, self.num_tokens, 1), (ARGMAX_GRID_BLOCK, 1, 1))
        } else {
            ((self.num_tokens, 1, 1), (self.vocab.min(1024), 1, 1))
        };
        launch_raw(kernel, grid, block, 0, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// map_token_id
// ---------------------------------------------------------------------------

pub struct MapTokenIdLaunch {
    pub keep_len: u32,
}

impl MapTokenIdLaunch {
    pub fn validate(&self) -> Result<()> {
        if self.keep_len == 0 {
            return Err(invalid("keep_len", "must be > 0"));
        }
        Ok(())
    }

    /// Kernel sig: `(row_id_i32, token_id_i32, keep_ids_i32, keep_len)`.
    ///
    /// # Safety
    /// Caller owns pointers for the call's duration.
    pub unsafe fn launch(
        &self,
        kernel: &KernelFn,
        row_id_ptr: u64,
        token_id_ptr: u64,
        keep_ids_ptr: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut row_id_ptr = row_id_ptr;
        let mut token_id_ptr = token_id_ptr;
        let mut keep_ids_ptr = keep_ids_ptr;
        let mut keep_len = self.keep_len as i32;
        let args = [
            (&mut row_id_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut token_id_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut keep_ids_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut keep_len) as *mut i32 as *mut core::ffi::c_void,
        ];
        launch_raw(kernel, (1, 1, 1), (1, 1, 1), 0, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// fused_rope_kv_write
// ---------------------------------------------------------------------------

pub struct FusedRopeKvWriteLaunch {
    pub num_tokens: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
}

impl FusedRopeKvWriteLaunch {
    pub fn validate(&self) -> Result<()> {
        if !matches!(self.head_dim, 128 | 256 | 512) {
            return Err(invalid(
                "head_dim",
                "v3 FA3 path requires head_dim in {128, 256, 512}",
            ));
        }
        if self.num_kv_heads == 0 || self.num_heads % self.num_kv_heads != 0 {
            return Err(invalid(
                "num_heads/num_kv_heads",
                "num_heads must be a multiple of num_kv_heads",
            ));
        }
        Ok(())
    }

    /// Kernel sig (v2 `fused_rope_cache_f16_kernel`):
    /// `(q, k, v, key_cache, value_cache, cos, sin, positions,
    ///   slot_mapping, num_tokens, num_heads, num_kv_heads, head_dim)`.
    /// q and k are modified in place (RoPE applied); v is read-only.
    ///
    /// # Safety
    /// Caller owns pointers for the call's duration.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        kernel: &KernelFn,
        q: u64,
        k: u64,
        v: u64,
        k_cache: u64,
        v_cache: u64,
        cos: u64,
        sin: u64,
        positions: u64,
        slot_mapping: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut q = q;
        let mut k = k;
        let mut v = v;
        let mut k_cache = k_cache;
        let mut v_cache = v_cache;
        let mut cos = cos;
        let mut sin = sin;
        let mut positions = positions;
        let mut slot_mapping = slot_mapping;
        let mut num_tokens = self.num_tokens as i32;
        let mut num_heads = self.num_heads as i32;
        let mut num_kv_heads = self.num_kv_heads as i32;
        let mut head_dim = self.head_dim as i32;
        let args = [
            (&mut q) as *mut u64 as *mut core::ffi::c_void,
            (&mut k) as *mut u64 as *mut core::ffi::c_void,
            (&mut v) as *mut u64 as *mut core::ffi::c_void,
            (&mut k_cache) as *mut u64 as *mut core::ffi::c_void,
            (&mut v_cache) as *mut u64 as *mut core::ffi::c_void,
            (&mut cos) as *mut u64 as *mut core::ffi::c_void,
            (&mut sin) as *mut u64 as *mut core::ffi::c_void,
            (&mut positions) as *mut u64 as *mut core::ffi::c_void,
            (&mut slot_mapping) as *mut u64 as *mut core::ffi::c_void,
            (&mut num_tokens) as *mut i32 as *mut core::ffi::c_void,
            (&mut num_heads) as *mut i32 as *mut core::ffi::c_void,
            (&mut num_kv_heads) as *mut i32 as *mut core::ffi::c_void,
            (&mut head_dim) as *mut i32 as *mut core::ffi::c_void,
        ];
        // Grid: (num_tokens, num_heads_max) — kernel uses (blockIdx.x, blockIdx.y).
        let max_heads = self.num_heads.max(self.num_kv_heads);
        let grid = (self.num_tokens, max_heads, 1);
        let block = ((self.head_dim / 2).max(32), 1, 1);
        launch_raw(kernel, grid, block, 0, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// residual_add_f16
// ---------------------------------------------------------------------------

pub struct AddBiasF16Launch {
    pub num_tokens: u32,
    pub dim: u32,
}

impl AddBiasF16Launch {
    pub fn validate(&self) -> Result<()> {
        if self.num_tokens == 0 || self.dim == 0 {
            return Err(invalid("add_bias_f16", "zero dim"));
        }
        Ok(())
    }

    /// Kernel sig: `(tensor_inout, bias, dim)`. Grid.x = num_tokens.
    ///
    /// # Safety
    /// Caller owns pointers.
    pub unsafe fn launch(
        &self,
        kernel: &KernelFn,
        tensor: u64,
        bias: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let mut tensor = tensor;
        let mut bias = bias;
        let mut dim = self.dim as i32;
        let args = [
            (&mut tensor) as *mut u64 as *mut core::ffi::c_void,
            (&mut bias) as *mut u64 as *mut core::ffi::c_void,
            (&mut dim) as *mut i32 as *mut core::ffi::c_void,
        ];
        let block = (self.dim.min(1024), 1, 1);
        let grid = (self.num_tokens, 1, 1);
        launch_raw(kernel, grid, block, 0, stream, &args)
    }
}

/// FP8 variant of the fused rope + KV write. Inputs are still f16 (the
/// output of the QKV GEMM); outputs Q / K / V cache are FP8 E4M3. Two
/// per-tensor f32 scalars (`q_scale`, `kv_scale`) carry the quantization.
pub struct FusedRopeCacheFp8KvLaunch {
    pub num_tokens: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
}

impl FusedRopeCacheFp8KvLaunch {
    pub fn validate(&self) -> Result<()> {
        if self.num_tokens == 0 || self.num_tokens > i32::MAX as u32 {
            return Err(invalid("num_tokens", "must fit positive i32"));
        }
        if self.num_heads == 0 || self.num_heads > i32::MAX as u32 {
            return Err(invalid("num_heads", "must fit positive i32"));
        }
        if self.num_kv_heads > i32::MAX as u32 {
            return Err(invalid("num_kv_heads", "must fit i32"));
        }
        if !matches!(self.head_dim, 128 | 256 | 512) {
            return Err(invalid(
                "head_dim",
                "v3 FA3 path requires head_dim in {128, 256, 512}",
            ));
        }
        if self.num_kv_heads > 0 && self.num_heads % self.num_kv_heads != 0 {
            return Err(invalid(
                "num_heads/num_kv_heads",
                "num_heads must be a multiple of num_kv_heads",
            ));
        }
        let _ = rope_capacities(
            self.block_size,
            self.max_blocks_per_seq,
            self.num_blocks_total,
        )?;
        Ok(())
    }

    /// Kernel sig:
    /// `(q, k, v, q_fp8, key_cache, value_cache, cos, sin, positions,
    ///   slot_mapping, q_scale_ptr, kv_scale_ptr, num_tokens, num_heads,
    ///   num_kv_heads, head_dim)`
    ///
    /// # Safety
    /// All device pointers valid for the call; scales point at single f32 scalars.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        kernel: &KernelFn,
        q_in: u64,
        k_in: u64,
        v_in: u64,
        q_fp8_out: u64,
        k_cache_fp8: u64,
        v_cache_fp8: u64,
        cos: u64,
        sin: u64,
        positions: u64,
        slot_mapping: u64,
        q_scale_ptr: u64,
        kv_scale_ptr: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        for (ptr, name) in [
            (q_in, "q_in"),
            (q_fp8_out, "q_fp8_out"),
            (cos, "cos"),
            (sin, "sin"),
            (positions, "positions"),
            (q_scale_ptr, "q_scale_ptr"),
        ] {
            require_ptr(ptr, name)?;
        }
        if self.num_kv_heads > 0 {
            for (ptr, name) in [
                (k_in, "k_in"),
                (v_in, "v_in"),
                (k_cache_fp8, "k_cache_fp8"),
                (v_cache_fp8, "v_cache_fp8"),
                (slot_mapping, "slot_mapping"),
                (kv_scale_ptr, "kv_scale_ptr"),
            ] {
                require_ptr(ptr, name)?;
            }
        }
        let (mut max_positions, mut num_cache_slots) = rope_capacities(
            self.block_size,
            self.max_blocks_per_seq,
            self.num_blocks_total,
        )?;
        let mut q_in = q_in;
        let mut k_in = k_in;
        let mut v_in = v_in;
        let mut q_fp8_out = q_fp8_out;
        let mut k_cache_fp8 = k_cache_fp8;
        let mut v_cache_fp8 = v_cache_fp8;
        let mut cos = cos;
        let mut sin = sin;
        let mut positions = positions;
        let mut slot_mapping = slot_mapping;
        let mut q_scale_ptr = q_scale_ptr;
        let mut kv_scale_ptr = kv_scale_ptr;
        let mut num_tokens = self.num_tokens as i32;
        let mut num_heads = self.num_heads as i32;
        let mut num_kv_heads = self.num_kv_heads as i32;
        let mut head_dim = self.head_dim as i32;
        let args = [
            (&mut q_in) as *mut u64 as *mut core::ffi::c_void,
            (&mut k_in) as *mut u64 as *mut core::ffi::c_void,
            (&mut v_in) as *mut u64 as *mut core::ffi::c_void,
            (&mut q_fp8_out) as *mut u64 as *mut core::ffi::c_void,
            (&mut k_cache_fp8) as *mut u64 as *mut core::ffi::c_void,
            (&mut v_cache_fp8) as *mut u64 as *mut core::ffi::c_void,
            (&mut cos) as *mut u64 as *mut core::ffi::c_void,
            (&mut sin) as *mut u64 as *mut core::ffi::c_void,
            (&mut positions) as *mut u64 as *mut core::ffi::c_void,
            (&mut slot_mapping) as *mut u64 as *mut core::ffi::c_void,
            (&mut q_scale_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut kv_scale_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut num_tokens) as *mut i32 as *mut core::ffi::c_void,
            (&mut num_heads) as *mut i32 as *mut core::ffi::c_void,
            (&mut num_kv_heads) as *mut i32 as *mut core::ffi::c_void,
            (&mut head_dim) as *mut i32 as *mut core::ffi::c_void,
            (&mut max_positions) as *mut i32 as *mut core::ffi::c_void,
            (&mut num_cache_slots) as *mut i32 as *mut core::ffi::c_void,
        ];
        let max_heads = self.num_heads.max(self.num_kv_heads);
        let grid = (self.num_tokens, max_heads, 1);
        let block = ((self.head_dim / 2).max(32), 1, 1);
        launch_raw(kernel, grid, block, 0, stream, &args)
    }
}

pub struct ResidualAddF16Launch {
    pub n: u32,
}

impl ResidualAddF16Launch {
    pub fn validate(&self) -> Result<()> {
        if self.n == 0 {
            return Err(invalid("n", "must be > 0"));
        }
        Ok(())
    }

    /// Kernel sig: `(x_inout, y, n)`.
    ///
    /// # Safety
    /// Caller owns pointers for the call's duration.
    pub unsafe fn launch(&self, kernel: &KernelFn, x: u64, y: u64, stream: u64) -> Result<()> {
        self.validate()?;
        let mut x = x;
        let mut y = y;
        let mut n = self.n as i32;
        let args = [
            (&mut x) as *mut u64 as *mut core::ffi::c_void,
            (&mut y) as *mut u64 as *mut core::ffi::c_void,
            (&mut n) as *mut i32 as *mut core::ffi::c_void,
        ];
        let block = (256, 1, 1);
        let grid = ((self.n + 255) / 256, 1, 1);
        launch_raw(kernel, grid, block, 0, stream, &args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_rejects_non_multiple_of_8() {
        let l = QuantizeFp8PerTokenLaunch {
            num_tokens: 1,
            dim: 13,
        };
        assert!(l.validate().is_err());
    }

    #[test]
    fn quant_accepts_power_of_two() {
        let l = QuantizeFp8PerTokenLaunch {
            num_tokens: 1,
            dim: 3584,
        };
        assert!(l.validate().is_ok());
    }

    #[test]
    fn rope_requires_head_dim_128() {
        let l = FusedRopeKvWriteLaunch {
            num_tokens: 1,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 64,
        };
        assert!(l.validate().is_err());
    }

    #[test]
    fn argmax_rejects_zero_vocab() {
        let l = ArgmaxLaunch {
            num_tokens: 32,
            vocab: 0,
        };
        assert!(l.validate().is_err());
    }

    #[test]
    fn embedding_rejects_zero_dims() {
        let l = EmbeddingGatherLaunch {
            num_tokens: 1,
            hidden: 0,
            vocab: 128,
        };
        assert!(l.validate().is_err());
    }

    #[test]
    fn residual_add_rejects_zero_n() {
        assert!(ResidualAddF16Launch { n: 0 }.validate().is_err());
    }
}
