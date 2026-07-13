//! Paged-prefill launcher. Same .so as decode; different entry point.
//!
//! CUDA FP8 prefill consumes variable-length query batches described by
//! `num_tokens` and `cu_seqlens_q`. The current Metal BF16 path uses the
//! decode-shaped paged-attention dispatcher and therefore accepts exactly
//! one query token per sequence.

use rvllm_core::{AttentionError, AttnCtx, Result, RvllmError};

const SUPPORTED_HEAD_DIMS: &[u32] = &[128, 256, 512];

#[derive(Copy, Clone, Debug)]
pub struct PagedPrefillParams {
    pub num_tokens: u32,
    pub num_seqs: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
    pub scale: f32,
    pub window_size_left: i32,
}

impl PagedPrefillParams {
    pub fn validate(&self) -> Result<()> {
        let ctx = || AttnCtx {
            op: "paged_prefill.validate",
            stream: 0,
            num_seqs: self.num_seqs,
            head_dim: self.head_dim,
        };
        if !SUPPORTED_HEAD_DIMS.contains(&self.head_dim) {
            return Err(RvllmError::Attention {
                err: AttentionError::UnsupportedHeadDim {
                    got: self.head_dim,
                    supported: SUPPORTED_HEAD_DIMS,
                },
                ctx: ctx(),
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if self.num_kv_heads == 0 || self.num_heads % self.num_kv_heads != 0 {
            return Err(RvllmError::Attention {
                err: AttentionError::GqaRatioInvalid {
                    num_heads: self.num_heads,
                    num_kv_heads: self.num_kv_heads,
                },
                ctx: ctx(),
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if self.num_tokens == 0
            || self.num_tokens > i32::MAX as u32
            || self.num_seqs == 0
            || self.num_seqs > i32::MAX as u32
            || self.num_heads == 0
            || self.num_heads > 65_535
            || self.num_kv_heads > i32::MAX as u32
            || self.block_size == 0
            || self.block_size > i32::MAX as u32
            || self.max_blocks_per_seq == 0
            || self.max_blocks_per_seq > i32::MAX as u32
            || self.num_blocks_total == 0
            || self.num_blocks_total > i32::MAX as u32
            || self.max_blocks_per_seq > self.num_blocks_total
        {
            return Err(invalid_params(
                self,
                "dimensions must be nonzero, fit the CUDA ABI, and fit the cache",
            ));
        }
        if self
            .max_blocks_per_seq
            .checked_mul(self.block_size)
            .filter(|&value| value <= i32::MAX as u32)
            .is_none()
            || self
                .num_blocks_total
                .checked_mul(self.block_size)
                .filter(|&value| value <= i32::MAX as u32)
                .is_none()
        {
            return Err(invalid_params(self, "KV capacity overflows the CUDA ABI"));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(invalid_params(self, "scale must be finite and positive"));
        }
        if self.window_size_left < -1 {
            return Err(invalid_params(
                self,
                "window_size_left must be -1 or nonnegative",
            ));
        }
        Ok(())
    }
}

fn invalid_params(params: &PagedPrefillParams, reason: impl Into<String>) -> RvllmError {
    RvllmError::Attention {
        err: AttentionError::InvalidParams {
            reason: reason.into(),
        },
        ctx: AttnCtx {
            op: "paged_prefill.validate",
            stream: 0,
            num_seqs: params.num_seqs,
            head_dim: params.head_dim,
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

fn validate_metal_prefill_geometry(params: &PagedPrefillParams) -> Result<()> {
    if params.num_tokens != params.num_seqs {
        return Err(invalid_params(
            params,
            "Metal paged prefill requires exactly one query token per sequence (num_tokens == num_seqs)",
        ));
    }
    Ok(())
}

fn require_device_ptrs(
    params: &PagedPrefillParams,
    stream: u64,
    op: &'static str,
    ptrs: &[(&'static str, u64)],
) -> Result<()> {
    if let Some((name, _)) = ptrs.iter().find(|(_, ptr)| *ptr == 0) {
        return Err(RvllmError::Attention {
            err: AttentionError::InvalidParams {
                reason: format!("{name} must be a non-null device pointer"),
            },
            ctx: AttnCtx {
                op,
                stream,
                num_seqs: params.num_seqs,
                head_dim: params.head_dim,
            },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    Ok(())
}

pub struct PagedPrefillLauncher<'a> {
    backend: &'a super::AttentionBackend,
}

impl<'a> PagedPrefillLauncher<'a> {
    pub fn new(backend: &'a super::AttentionBackend) -> Self {
        Self { backend }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        params: PagedPrefillParams,
        out_ptr: u64,
        q_ptr: u64,
        k_cache_ptr: u64,
        v_cache_ptr: u64,
        block_tables_ptr: u64,
        context_lens_ptr: u64,
        cu_seqlens_q_ptr: u64,
        cu_seqlens_k_ptr: u64,
        workspace_ptr: u64,
        stream: u64,
    ) -> Result<()> {
        params.validate()?;
        require_device_ptrs(
            &params,
            stream,
            "paged_prefill",
            &[
                ("out_ptr", out_ptr),
                ("q_ptr", q_ptr),
                ("k_cache_ptr", k_cache_ptr),
                ("v_cache_ptr", v_cache_ptr),
                ("block_tables_ptr", block_tables_ptr),
                ("context_lens_ptr", context_lens_ptr),
            ],
        )?;
        // Metal BF16 prefill routes through the decode-shaped paged-attention
        // dispatcher, which consumes one query token per sequence.
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        {
            if let super::AttentionBackend::Metal(m) = self.backend {
                validate_metal_prefill_geometry(&params)?;
                if params.window_size_left >= 0 {
                    return Err(RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            backend: "Metal",
                            op: "paged_prefill: sliding window is not wired in Metal paged attention",
                        },
                        ctx: AttnCtx {
                            op: "paged_prefill (Metal sliding window)",
                            stream,
                            num_seqs: params.num_seqs,
                            head_dim: params.head_dim,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    });
                }
                // This exact geometry does not need cu_seqlens. Variable-length
                // batches must use a future Metal prefill implementation that
                // consumes them; this dispatcher intentionally ignores them.
                let _ = (cu_seqlens_q_ptr, cu_seqlens_k_ptr, workspace_ptr);
                let resolve = |ptr: u64, name: &'static str| {
                    m.registry.lookup(ptr).ok_or_else(|| RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            backend: "Metal",
                            op: "paged_prefill: launcher u64 not registered with MetalBufferRegistry",
                        },
                        ctx: AttnCtx {
                            op: name,
                            stream,
                            num_seqs: params.num_seqs,
                            head_dim: params.head_dim,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    })
                };
                let (out_buf, _, _) = resolve(out_ptr, "paged_prefill: out_ptr lookup")?;
                let (q_buf, _, _) = resolve(q_ptr, "paged_prefill: q_ptr lookup")?;
                let (k_cache_buf, _, _) =
                    resolve(k_cache_ptr, "paged_prefill: k_cache_ptr lookup")?;
                let (v_cache_buf, _, _) =
                    resolve(v_cache_ptr, "paged_prefill: v_cache_ptr lookup")?;
                let (block_tables_buf, _, _) =
                    resolve(block_tables_ptr, "paged_prefill: block_tables_ptr lookup")?;
                let (context_lens_buf, _, _) =
                    resolve(context_lens_ptr, "paged_prefill: context_lens_ptr lookup")?;
                let cmd_buf = m.device.queue().new_command_buffer();
                let encoder = cmd_buf.new_compute_command_encoder();
                let dispatch_res = rvllm_metal::paged_attention::call_paged_attention_metal(
                    &m.kernels,
                    encoder,
                    m.dtype,
                    m.dtype,
                    params.head_dim,
                    params.block_size,
                    &q_buf,
                    &k_cache_buf,
                    &v_cache_buf,
                    &out_buf,
                    &block_tables_buf,
                    &context_lens_buf,
                    params.num_seqs,
                    params.num_kv_heads,
                    params.num_heads,
                    params.scale,
                    /* softcapping = */ None,
                    params.max_blocks_per_seq.saturating_mul(params.block_size),
                );
                encoder.end_encoding();
                if let Err(e) = dispatch_res {
                    return Err(RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            backend: "Metal",
                            op: "paged_prefill: rvllm_metal::call_paged_attention_metal failed",
                        },
                        ctx: AttnCtx {
                            op: "paged_prefill (Metal dispatch)",
                            stream,
                            num_seqs: params.num_seqs,
                            head_dim: params.head_dim,
                        },
                        bt: {
                            let _ = e;
                            std::backtrace::Backtrace::capture()
                        },
                    });
                }
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
                return Ok(());
            }
        }
        let _ = (
            self.backend,
            cu_seqlens_q_ptr,
            cu_seqlens_k_ptr,
            workspace_ptr,
        );
        Err(RvllmError::Attention {
            err: AttentionError::FeatureNotAvailable {
                backend: "non-Metal backend",
                op: "paged_prefill has no verified CUDA dispatch",
            },
            ctx: AttnCtx {
                op: "paged_prefill",
                stream,
                num_seqs: params.num_seqs,
                head_dim: params.head_dim,
            },
            bt: std::backtrace::Backtrace::capture(),
        })
    }
}

/// FP8 E4M3 paged-prefill launcher. Q / K / V are FP8 with per-tensor
/// descales. Multi-query self-attention with a per-seq causal mask.
/// `Fa2Ptx` callers use the decode-per-query fallback in the Gemma 4 runtime.
pub struct PagedPrefillFp8Launcher<'a> {
    backend: &'a super::AttentionBackend,
}

impl<'a> PagedPrefillFp8Launcher<'a> {
    pub fn new(backend: &'a super::AttentionBackend) -> Self {
        Self { backend }
    }

    /// # Safety
    /// Caller owns all device pointers. `cu_seqlens_q` is a
    /// [batch+1]-len i32 prefix-sum device buffer; `max_seqlen_q` is the
    /// longest per-seq Q length; `total_q` is the sum (= Q tensor's
    /// leading dim).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        params: PagedPrefillParams,
        o_f16: u64,
        q_fp8: u64,
        k_cache_fp8: u64,
        v_cache_fp8: u64,
        block_tables: u64,
        context_lens: u64,
        cu_seqlens_q: u64,
        workspace: u64,
        workspace_bytes: usize,
        k_scale_cache: u64,
        v_scale_cache: u64,
        q_scale_cache: u64,
        q_descale_ptr: u64,
        k_descale_ptr: u64,
        v_descale_ptr: u64,
        max_seqlen_q: u32,
        stream: u64,
    ) -> Result<()> {
        params.validate()?;
        if max_seqlen_q == 0 || max_seqlen_q > params.num_tokens || max_seqlen_q > i32::MAX as u32 {
            return Err(invalid_params(
                &params,
                "max_seqlen_q must be in 1..=num_tokens and fit the CUDA ABI",
            ));
        }
        let mut required = vec![
            ("o_f16", o_f16),
            ("q_fp8", q_fp8),
            ("k_cache_fp8", k_cache_fp8),
            ("v_cache_fp8", v_cache_fp8),
            ("block_tables", block_tables),
            ("context_lens", context_lens),
            ("cu_seqlens_q", cu_seqlens_q),
        ];
        if q_scale_cache == 0 {
            required.push(("q_descale_ptr", q_descale_ptr));
        }
        if k_scale_cache == 0 {
            required.push(("k_descale_ptr", k_descale_ptr));
        }
        if v_scale_cache == 0 {
            required.push(("v_descale_ptr", v_descale_ptr));
        }
        require_device_ptrs(&params, stream, "paged_prefill_fp8", &required)?;
        // Metal: FP8 paged attention is v2-only.
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        {
            if let super::AttentionBackend::Metal(_) = self.backend {
                let _ = (
                    o_f16,
                    q_fp8,
                    k_cache_fp8,
                    v_cache_fp8,
                    block_tables,
                    context_lens,
                    cu_seqlens_q,
                    workspace,
                    workspace_bytes,
                    k_scale_cache,
                    v_scale_cache,
                    q_scale_cache,
                    q_descale_ptr,
                    k_descale_ptr,
                    v_descale_ptr,
                    max_seqlen_q,
                );
                return Err(RvllmError::Attention {
                    err: AttentionError::FeatureNotAvailable {
                        backend: "Metal",
                        op: "FP8 paged attention on Metal — v2 only; use BF16 KV in v1",
                    },
                    ctx: AttnCtx {
                        op: "paged_prefill_fp8 (Metal)",
                        stream,
                        num_seqs: params.num_seqs,
                        head_dim: params.head_dim,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
        }
        #[cfg(feature = "cuda")]
        {
            let fa3 = match self.backend {
                super::AttentionBackend::Fa3(fa3) => fa3,
                #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
                super::AttentionBackend::Metal(_) => unreachable!("handled above"),
                super::AttentionBackend::Fa2Ptx(_) => {
                    // sm_121 no longer ships a dedicated FA2 prefill
                    // kernel — Gemma 4's unified attention in
                    // gemma4_layer_exec.rs replaces batch prefill with
                    // a loop of single-query decode launches, which is
                    // numerically identical to the per-token decode
                    // path rvllm-ppl validates. Callers on sm_121
                    // should route through `PagedDecodeFp8Launcher`
                    // directly; keeping this arm would just tempt them
                    // back into the less-accurate FA2 prefill.
                    let _ = (
                        o_f16,
                        q_fp8,
                        k_cache_fp8,
                        v_cache_fp8,
                        block_tables,
                        context_lens,
                        cu_seqlens_q,
                        workspace,
                        workspace_bytes,
                        q_descale_ptr,
                        k_descale_ptr,
                        v_descale_ptr,
                        max_seqlen_q,
                        stream,
                    );
                    return Err(RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            op: "paged_prefill_fp8 Fa2Ptx (use decode-per-qi loop)",
                            backend: "Fa2Ptx",
                        },
                        ctx: AttnCtx {
                            op: "paged_prefill_fp8 (Fa2Ptx)",
                            stream,
                            num_seqs: params.num_seqs,
                            head_dim: params.head_dim,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    });
                }
            };
            let Some(f) = fa3.fn_paged_prefill_fp8 else {
                return Err(RvllmError::Attention {
                    err: AttentionError::Fa3SoMissing {
                        path: fa3.so_path.clone(),
                    },
                    ctx: AttnCtx {
                        op: "paged_prefill_fp8 symbol missing from .so (rebuild fa3)",
                        stream,
                        num_seqs: params.num_seqs,
                        head_dim: params.head_dim,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            };
            require_device_ptrs(
                &params,
                stream,
                "paged_prefill_fp8 (Fa3)",
                &[("workspace", workspace)],
            )?;
            let required_workspace = fa3.prefill_workspace_size(&params, max_seqlen_q)?;
            super::require_workspace_capacity(
                workspace_bytes,
                required_workspace,
                "paged_prefill_fp8",
                params.num_seqs,
                params.head_dim,
                stream,
            )?;
            let rc = f(
                q_fp8 as *mut std::ffi::c_void,
                k_cache_fp8 as *mut std::ffi::c_void,
                v_cache_fp8 as *mut std::ffi::c_void,
                o_f16 as *mut std::ffi::c_void,
                block_tables as *mut std::ffi::c_void,
                context_lens as *mut std::ffi::c_void,
                cu_seqlens_q as *mut std::ffi::c_void,
                workspace as *mut std::ffi::c_void,
                workspace_bytes,
                k_scale_cache as *mut std::ffi::c_void,
                v_scale_cache as *mut std::ffi::c_void,
                q_scale_cache as *mut std::ffi::c_void,
                q_descale_ptr as *mut f32,
                k_descale_ptr as *mut f32,
                v_descale_ptr as *mut f32,
                params.scale,
                params.num_tokens as i32,
                max_seqlen_q as i32,
                params.num_seqs as i32,
                params.num_heads as i32,
                params.num_kv_heads as i32,
                params.head_dim as i32,
                params.block_size as i32,
                params.max_blocks_per_seq as i32,
                params.num_blocks_total as i32,
                params.window_size_left,
                stream as *mut std::ffi::c_void,
            );
            if rc != 0 {
                return Err(RvllmError::Attention {
                    err: AttentionError::KernelLaunchFailed {
                        cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                    },
                    ctx: AttnCtx {
                        op: "paged_prefill_fp8",
                        stream,
                        num_seqs: params.num_seqs,
                        head_dim: params.head_dim,
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (workspace, workspace_bytes);
            Err(RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    backend: "non-CUDA build",
                    op: "paged_prefill_fp8",
                },
                ctx: AttnCtx {
                    op: "paged_prefill_fp8",
                    stream,
                    num_seqs: params.num_seqs,
                    head_dim: params.head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            })
        }
        #[cfg(feature = "cuda")]
        {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_validates_head_dim() {
        let p = PagedPrefillParams {
            num_tokens: 256,
            num_seqs: 4,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 64, // bad
            block_size: 64,
            max_blocks_per_seq: 33,
            num_blocks_total: 1024,
            scale: 1.0,
            window_size_left: -1,
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn prefill_accepts_head_dim_256() {
        let p = PagedPrefillParams {
            num_tokens: 256,
            num_seqs: 4,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 256,
            block_size: 64,
            max_blocks_per_seq: 33,
            num_blocks_total: 1024,
            scale: 1.0 / (256f32).sqrt(),
            window_size_left: -1,
        };
        assert!(p.validate().is_ok());
    }

    fn valid_metal_params(num_tokens: u32, num_seqs: u32) -> PagedPrefillParams {
        PagedPrefillParams {
            num_tokens,
            num_seqs,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 256,
            block_size: 64,
            max_blocks_per_seq: 33,
            num_blocks_total: 1024,
            scale: 1.0 / (256f32).sqrt(),
            window_size_left: -1,
        }
    }

    #[test]
    fn metal_prefill_accepts_one_token_per_sequence() {
        assert!(validate_metal_prefill_geometry(&valid_metal_params(4, 4)).is_ok());
    }

    #[test]
    fn metal_prefill_rejects_multiple_tokens_per_sequence() {
        assert!(validate_metal_prefill_geometry(&valid_metal_params(256, 4)).is_err());
    }

    #[test]
    fn metal_prefill_rejects_fewer_tokens_than_sequences() {
        assert!(validate_metal_prefill_geometry(&valid_metal_params(3, 4)).is_err());
    }
}
