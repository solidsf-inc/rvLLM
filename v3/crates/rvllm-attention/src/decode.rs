//! Paged-decode launcher.
//!
//! One query per sequence. Kernel reads context_lens[seq] and walks
//! block_tables[seq, 0..ceil(context_lens/block_size)] to find KV
//! pages. `context_lens[i] == 0` is a valid padded slot; kernel must
//! predicate and never touch block_tables[i,*].

use rvllm_core::{AttentionError, AttnCtx, Result, RvllmError};

const SUPPORTED_HEAD_DIMS: &[u32] = &[128, 256, 512];

/// Parameters for one paged decode launch.
#[derive(Copy, Clone, Debug)]
pub struct PagedDecodeParams {
    pub num_seqs: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
    pub scale: f32,
    pub window_size_left: i32, // -1 = full, >= 0 = sliding window
}

impl PagedDecodeParams {
    pub fn validate(&self) -> Result<()> {
        let ctx = || AttnCtx {
            op: "paged_decode.validate",
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
        if self.num_seqs == 0 {
            return Err(RvllmError::Attention {
                err: AttentionError::ContextExceedsBucket { context: 0, max: 0 },
                ctx: ctx(),
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if self.num_seqs > i32::MAX as u32
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

fn invalid_params(params: &PagedDecodeParams, reason: impl Into<String>) -> RvllmError {
    RvllmError::Attention {
        err: AttentionError::InvalidParams {
            reason: reason.into(),
        },
        ctx: AttnCtx {
            op: "paged_decode.validate",
            stream: 0,
            num_seqs: params.num_seqs,
            head_dim: params.head_dim,
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

fn require_device_ptrs(
    params: &PagedDecodeParams,
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

/// Launcher. Constructed from `&AttentionBackend`. The `Fa3` variant
/// dispatches through an authenticated shared object; `Fa2Ptx` launches
/// the target-specific PTX implementation.
pub struct PagedDecodeLauncher<'a> {
    backend: &'a super::AttentionBackend,
}

impl<'a> PagedDecodeLauncher<'a> {
    pub fn new(backend: &'a super::AttentionBackend) -> Self {
        Self { backend }
    }

    /// Validate params + issue the launch.
    ///
    /// # Safety
    /// Under `feature = "cuda"` this dispatches the fa3_sm90 kernel via
    /// an opaque C fn ptr. All device pointers must be valid for the
    /// kernel's duration and `workspace_bytes` must describe the allocation
    /// beginning at `workspace_ptr`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        params: PagedDecodeParams,
        out_ptr: u64,
        q_ptr: u64,
        k_cache_ptr: u64,
        v_cache_ptr: u64,
        block_tables_ptr: u64,
        context_lens_ptr: u64,
        workspace_ptr: u64,
        workspace_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        params.validate()?;
        require_device_ptrs(
            &params,
            stream,
            "paged_decode",
            &[
                ("out_ptr", out_ptr),
                ("q_ptr", q_ptr),
                ("k_cache_ptr", k_cache_ptr),
                ("v_cache_ptr", v_cache_ptr),
                ("block_tables_ptr", block_tables_ptr),
                ("context_lens_ptr", context_lens_ptr),
            ],
        )?;
        // Metal dispatch is checked first because the enum variant is
        // `#[cfg]`-gated; an `AttentionBackend::Metal(_)` value can
        // only exist when `feature = "metal"` is on, so this block
        // compiles away on CUDA-only builds.
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        {
            if let super::AttentionBackend::Metal(m) = self.backend {
                if params.window_size_left >= 0 {
                    return Err(RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            backend: "Metal",
                            op:
                                "paged_decode: sliding window is not wired in Metal paged attention",
                        },
                        ctx: AttnCtx {
                            op: "paged_decode (Metal sliding window)",
                            stream,
                            num_seqs: params.num_seqs,
                            head_dim: params.head_dim,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    });
                }
                let _ = (workspace_ptr, workspace_bytes); // not used by paged_attention v1
                                                          // Resolve every launcher u64 to its registered MTLBuffer.
                                                          // The runtime is responsible for registering buffers
                                                          // before invoking the launcher. Any missing entry is a
                                                          // setup bug — fail loud.
                let resolve = |ptr: u64, name: &'static str| {
                    m.registry.lookup(ptr).ok_or_else(|| RvllmError::Attention {
                        err: AttentionError::FeatureNotAvailable {
                            backend: "Metal",
                            op: "paged_decode: launcher u64 not registered with MetalBufferRegistry",
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
                let (out_buf, _, _) = resolve(out_ptr, "paged_decode: out_ptr lookup")?;
                let (q_buf, _, _) = resolve(q_ptr, "paged_decode: q_ptr lookup")?;
                let (k_cache_buf, _, _) = resolve(k_cache_ptr, "paged_decode: k_cache_ptr lookup")?;
                let (v_cache_buf, _, _) = resolve(v_cache_ptr, "paged_decode: v_cache_ptr lookup")?;
                let (block_tables_buf, _, _) =
                    resolve(block_tables_ptr, "paged_decode: block_tables_ptr lookup")?;
                let (context_lens_buf, _, _) =
                    resolve(context_lens_ptr, "paged_decode: context_lens_ptr lookup")?;

                // Build a command buffer + encoder, dispatch, wait for
                // completion synchronously. Matches the CUDA backends'
                // blocking semantics so callers don't need to track
                // streams just to use Metal. Future perf work can move
                // to deferred-encode + explicit fences.
                let cmd_buf = m.device.queue().new_command_buffer();
                let encoder = cmd_buf.new_compute_command_encoder();
                let dispatch_res = rvllm_metal::paged_attention::call_paged_attention_metal(
                    &m.kernels,
                    encoder,
                    m.dtype, // output dtype
                    m.dtype, // cache dtype (BF16 KV in v1)
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
                            op: "paged_decode: rvllm_metal::call_paged_attention_metal failed",
                        },
                        ctx: AttnCtx {
                            op: "paged_decode (Metal dispatch)",
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
        #[cfg(feature = "cuda")]
        {
            let fa3 = match self.backend {
                super::AttentionBackend::Fa3(fa3) => fa3,
                #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
                super::AttentionBackend::Metal(_) => unreachable!("handled above"),
                super::AttentionBackend::Fa2Ptx(fa2) => {
                    // sm_121 F16 KV path: dispatch
                    // `flash_attention_2_decode_f16io_kernel` (f16 I/O
                    // against the paged f16 cache). head_dim=512 does
                    // NOT fit the BC=32 smem budget on sm_121's 99 KB
                    // opt-in cap — gate it off. Gemma 4 global layers
                    // must use FP8 KV; F16 KV on sm_121 today is a
                    // head_dim ≤ 256 path (Llama/Qwen or Gemma 4
                    // sliding-only).
                    if params.head_dim > 256 {
                        return Err(RvllmError::Attention {
                            err: AttentionError::FeatureNotAvailable {
                                backend: "Fa2Ptx",
                                op: "paged_decode (F16 KV, head_dim>256 — use FP8 KV)",
                            },
                            ctx: AttnCtx {
                                op: "paged_decode",
                                stream,
                                num_seqs: params.num_seqs,
                                head_dim: params.head_dim,
                            },
                            bt: std::backtrace::Backtrace::capture(),
                        });
                    }
                    const FA2_THREADS: u32 = 128;
                    let fa2_bc = fa2.f16_tile_cols;
                    let hd = params.head_dim;
                    let smem_bytes = 2 * fa2_bc * hd * 4 + fa2_bc * 4 + (FA2_THREADS / 32) * 4;

                    let scale = params.scale;
                    let num_heads = params.num_heads as i32;
                    let num_kv_heads = params.num_kv_heads as i32;
                    let head_dim = params.head_dim as i32;
                    let block_size = params.block_size as i32;
                    let max_blocks_per_seq = params.max_blocks_per_seq as i32;
                    let num_blocks_total = params.num_blocks_total as i32;

                    let mut arg_out = out_ptr;
                    let mut arg_q = q_ptr;
                    let mut arg_k = k_cache_ptr;
                    let mut arg_v = v_cache_ptr;
                    let mut arg_bt = block_tables_ptr;
                    let mut arg_cl = context_lens_ptr;
                    let _ = (workspace_ptr, workspace_bytes); // FA2 uses dynamic smem

                    let args: [*mut core::ffi::c_void; 13] = [
                        &mut arg_out as *mut _ as *mut _,
                        &mut arg_q as *mut _ as *mut _,
                        &mut arg_k as *mut _ as *mut _,
                        &mut arg_v as *mut _ as *mut _,
                        &mut arg_bt as *mut _ as *mut _,
                        &mut arg_cl as *mut _ as *mut _,
                        &scale as *const _ as *mut _,
                        &num_heads as *const _ as *mut _,
                        &num_kv_heads as *const _ as *mut _,
                        &head_dim as *const _ as *mut _,
                        &block_size as *const _ as *mut _,
                        &max_blocks_per_seq as *const _ as *mut _,
                        &num_blocks_total as *const _ as *mut _,
                    ];
                    fa2.fn_decode_f16io.launch_raw(
                        (params.num_seqs, params.num_heads, 1),
                        (FA2_THREADS, 1, 1),
                        smem_bytes,
                        stream,
                        &args,
                    )?;
                    return Ok(());
                }
            };
            require_device_ptrs(
                &params,
                stream,
                "paged_decode (Fa3)",
                &[("workspace_ptr", workspace_ptr)],
            )?;
            let required_workspace = fa3.decode_workspace_size(&params, false)?;
            super::require_workspace_capacity(
                workspace_bytes,
                required_workspace,
                "paged_decode",
                params.num_seqs,
                params.head_dim,
                stream,
            )?;
            let rc = (fa3.fn_paged_decode)(
                q_ptr as *mut std::ffi::c_void,
                k_cache_ptr as *mut std::ffi::c_void,
                v_cache_ptr as *mut std::ffi::c_void,
                out_ptr as *mut std::ffi::c_void,
                block_tables_ptr as *mut std::ffi::c_void,
                context_lens_ptr as *mut std::ffi::c_void,
                workspace_ptr as *mut std::ffi::c_void,
                workspace_bytes,
                params.scale,
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
                        op: "paged_decode",
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
            let _ = (workspace_ptr, workspace_bytes);
            Err(RvllmError::Attention {
                err: AttentionError::FeatureNotAvailable {
                    backend: "non-CUDA build",
                    op: "paged_decode",
                },
                ctx: AttnCtx {
                    op: "paged_decode",
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

/// FP8 E4M3 paged-decode launcher. Same param validation as the FP16
/// path; dispatches the FP8 entry point and threads per-tensor scales.
/// `Fa2Ptx` dispatches the FP8-KV decode kernel directly.
pub struct PagedDecodeFp8Launcher<'a> {
    backend: &'a super::AttentionBackend,
}

impl<'a> PagedDecodeFp8Launcher<'a> {
    pub fn new(backend: &'a super::AttentionBackend) -> Self {
        Self { backend }
    }

    /// # Safety
    /// Every pointer must be valid device memory; `q_descale_ptr`,
    /// `k_descale_ptr`, `v_descale_ptr` point at single f32 scalars.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        params: PagedDecodeParams,
        o_f16: u64,
        q_fp8: u64,
        k_cache_fp8: u64,
        v_cache_fp8: u64,
        // `k_scale_cache` / `v_scale_cache`: per-slot f32 arrays
        //   (Gemma 4). Pass `0` to fall back to the scalar
        //   `k_descale_fallback_ptr` / `v_descale_fallback_ptr`
        //   (Llama/Qwen path).
        k_scale_cache: u64,
        v_scale_cache: u64,
        // `q_scale_cache`: `[num_seqs * num_heads]` f32 array of
        //   per-(seq, head) Q scales. Pass `0` to fall back to the
        //   scalar `q_descale_ptr`.
        q_scale_cache: u64,
        k_descale_fallback_ptr: u64,
        v_descale_fallback_ptr: u64,
        block_tables: u64,
        context_lens: u64,
        workspace: u64,
        workspace_bytes: usize,
        q_descale_ptr: u64,
        stream: u64,
    ) -> Result<()> {
        params.validate()?;
        let mut required = vec![
            ("o_f16", o_f16),
            ("q_fp8", q_fp8),
            ("k_cache_fp8", k_cache_fp8),
            ("v_cache_fp8", v_cache_fp8),
            ("block_tables", block_tables),
            ("context_lens", context_lens),
        ];
        if q_scale_cache == 0 {
            required.push(("q_descale_ptr", q_descale_ptr));
        }
        if k_scale_cache == 0 {
            required.push(("k_descale_fallback_ptr", k_descale_fallback_ptr));
        }
        if v_scale_cache == 0 {
            required.push(("v_descale_fallback_ptr", v_descale_fallback_ptr));
        }
        require_device_ptrs(&params, stream, "paged_decode_fp8", &required)?;
        // Metal: FP8 paged attention is v2-only; v1 ships with BF16 KV
        // (see `KERNEL_NAMES` in rvllm-metal — the `_cache_uchar_`
        // entries are listed but not wired through this launcher in
        // v1). Fail loudly per the "no silent fallbacks" rule.
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        {
            if let super::AttentionBackend::Metal(_) = self.backend {
                let _ = (
                    o_f16,
                    q_fp8,
                    k_cache_fp8,
                    v_cache_fp8,
                    k_scale_cache,
                    v_scale_cache,
                    q_scale_cache,
                    k_descale_fallback_ptr,
                    v_descale_fallback_ptr,
                    block_tables,
                    context_lens,
                    workspace,
                    workspace_bytes,
                    q_descale_ptr,
                );
                return Err(RvllmError::Attention {
                    err: AttentionError::FeatureNotAvailable {
                        backend: "Metal",
                        op: "FP8 paged attention on Metal — v2 only; use BF16 KV in v1",
                    },
                    ctx: AttnCtx {
                        op: "paged_decode_fp8 (Metal)",
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
                super::AttentionBackend::Fa2Ptx(fa2) => {
                    // sm_121 path: dispatch the PTX-built
                    // `flash_attention_2_decode_fp8kv_kernel`. Internal
                    // math f32, on-load dequant from FP8 E4M3 with
                    // per-tensor descales, f16 output to match the
                    // FA3 ABI (`o_f16`).
                    //
                    // Launch config:
                    //   Grid  (num_seqs, num_heads, 1)
                    //   Block (FA2_THREADS=128, 1, 1)
                    //   Smem  = 2 * FA2_BC * head_dim * 4 + FA2_BC * 4
                    //           + (FA2_THREADS / 32) * 4
                    // FA2_BC is 32 for sm_100+ (arch-conditional in
                    // flash_attention.cu). head_dim=256 with BC=32
                    // blows past the 48 KB static-smem ceiling, so we
                    // opt in to dynamic smem via `cuFuncSetAttribute`
                    // once per process.
                    // sm_121 FP8-KV decode is BC=16 only. head_dim=512
                    // never fit BC=32 within the 99 KB opt-in smem cap,
                    // and head_dim=256 measurably favours BC=16 too
                    // (+2.5%/+5.5% at batch=128/256 — halving the tile
                    // lets 2+ blocks live per SM and hides per-tile
                    // __syncthreads latency). The BC=32 kernel was
                    // removed from flash_attention.cu in the cleanup
                    // that followed the ncu profile.
                    const FA2_THREADS: u32 = 128;
                    const FA2_BC: u32 = 16;
                    let hd = params.head_dim;
                    let kernel_fn = &fa2.fn_decode_fp8kv;
                    let smem_bytes = 2 * FA2_BC * hd * 4 + FA2_BC * 4 + (FA2_THREADS / 32) * 4;

                    // Scalar args must outlive cuLaunchKernel.
                    let scale = params.scale;
                    let num_heads = params.num_heads as i32;
                    let num_kv_heads = params.num_kv_heads as i32;
                    let head_dim = params.head_dim as i32;
                    let block_size = params.block_size as i32;
                    let max_blocks_per_seq = params.max_blocks_per_seq as i32;
                    let window_size_left = params.window_size_left;
                    let num_blocks_total = params.num_blocks_total as i32;

                    let mut arg_out = o_f16;
                    let mut arg_q = q_fp8;
                    let mut arg_k = k_cache_fp8;
                    let mut arg_v = v_cache_fp8;
                    let mut arg_ks = k_scale_cache;
                    let mut arg_vs = v_scale_cache;
                    let mut arg_qs = q_scale_cache;
                    let mut arg_kd = k_descale_fallback_ptr;
                    let mut arg_vd = v_descale_fallback_ptr;
                    let mut arg_bt = block_tables;
                    let mut arg_cl = context_lens;
                    let mut arg_qd = q_descale_ptr;
                    let _ = (workspace, workspace_bytes); // FA2 allocates in smem

                    let args: [*mut core::ffi::c_void; 20] = [
                        &mut arg_out as *mut _ as *mut _,
                        &mut arg_q as *mut _ as *mut _,
                        &mut arg_k as *mut _ as *mut _,
                        &mut arg_v as *mut _ as *mut _,
                        &mut arg_ks as *mut _ as *mut _,
                        &mut arg_vs as *mut _ as *mut _,
                        &mut arg_qs as *mut _ as *mut _,
                        &mut arg_kd as *mut _ as *mut _,
                        &mut arg_vd as *mut _ as *mut _,
                        &mut arg_bt as *mut _ as *mut _,
                        &mut arg_cl as *mut _ as *mut _,
                        &mut arg_qd as *mut _ as *mut _,
                        &scale as *const _ as *mut _,
                        &num_heads as *const _ as *mut _,
                        &num_kv_heads as *const _ as *mut _,
                        &head_dim as *const _ as *mut _,
                        &block_size as *const _ as *mut _,
                        &max_blocks_per_seq as *const _ as *mut _,
                        &window_size_left as *const _ as *mut _,
                        &num_blocks_total as *const _ as *mut _,
                    ];
                    kernel_fn.launch_raw(
                        (params.num_seqs, params.num_heads, 1),
                        (FA2_THREADS, 1, 1),
                        smem_bytes,
                        stream,
                        &args,
                    )?;
                    return Ok(());
                }
            };
            require_device_ptrs(
                &params,
                stream,
                "paged_decode_fp8 (Fa3)",
                &[("workspace", workspace)],
            )?;
            let required_workspace = fa3.decode_workspace_size(&params, true)?;
            super::require_workspace_capacity(
                workspace_bytes,
                required_workspace,
                "paged_decode_fp8",
                params.num_seqs,
                params.head_dim,
                stream,
            )?;
            let rc = (fa3.fn_paged_decode_fp8)(
                q_fp8 as *mut std::ffi::c_void,
                k_cache_fp8 as *mut std::ffi::c_void,
                v_cache_fp8 as *mut std::ffi::c_void,
                o_f16 as *mut std::ffi::c_void,
                block_tables as *mut std::ffi::c_void,
                context_lens as *mut std::ffi::c_void,
                workspace as *mut std::ffi::c_void,
                workspace_bytes,
                k_scale_cache as *mut std::ffi::c_void,
                v_scale_cache as *mut std::ffi::c_void,
                q_scale_cache as *mut std::ffi::c_void,
                q_descale_ptr as *mut f32,
                k_descale_fallback_ptr as *mut f32,
                v_descale_fallback_ptr as *mut f32,
                params.scale,
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
                        op: "paged_decode_fp8",
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
                    op: "paged_decode_fp8",
                },
                ctx: AttnCtx {
                    op: "paged_decode_fp8",
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

    fn good() -> PagedDecodeParams {
        PagedDecodeParams {
            num_seqs: 32,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            block_size: 64,
            max_blocks_per_seq: 33,
            num_blocks_total: 1024,
            scale: 1.0 / (128f32).sqrt(),
            window_size_left: -1,
        }
    }

    #[test]
    fn rejects_head_dim_64() {
        let mut p = good();
        p.head_dim = 64;
        assert!(p.validate().is_err());
    }

    #[test]
    fn rejects_gqa_ratio_not_divisible() {
        let mut p = good();
        p.num_heads = 7;
        p.num_kv_heads = 4;
        assert!(p.validate().is_err());
    }

    #[test]
    fn accepts_qwen_shape() {
        assert!(good().validate().is_ok());
    }

    #[test]
    fn accepts_head_dim_256() {
        let mut p = good();
        p.head_dim = 256;
        p.scale = 1.0 / (256f32).sqrt();
        assert!(p.validate().is_ok());
    }

    #[test]
    fn accepts_head_dim_512() {
        let mut p = good();
        p.head_dim = 512;
        p.scale = 1.0 / (512f32).sqrt();
        assert!(p.validate().is_ok());
    }
}
