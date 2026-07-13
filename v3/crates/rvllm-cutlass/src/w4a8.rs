//! W4A8 GEMM wrapper around `libw4a8_gemm.so` (CUTLASS SM90, int4×fp8).
//!
//! Mirrors the `CublasLt::fp8_gemm` API surface so the dispatcher in
//! `layer_exec` can swap between FP8 and W4A8 paths by a per-linear
//! flag, without changing call-site shape.
//!
//! Weight format on disk: INT4 two's-complement, AWQ-re-encoded,
//! memory-reordered offline into the `LayoutB_Reordered` atom layout.
//! Scales: per-group (g=128) FP8 E4M3 LUT blocks (8 packed scale factors
//! per group × N). See `rvllm-loader` for the encoder.

use std::path::PathBuf;

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

#[cfg(feature = "cuda")]
type W4a8GemmFn = unsafe extern "C" fn(
    a_fp8: *const std::ffi::c_void,
    b_int4_reordered: *const std::ffi::c_void,
    b_scales_packed: *const std::ffi::c_void,
    c_f16: *const std::ffi::c_void,
    d_f16: *mut std::ffi::c_void,
    m: i32,
    n: i32,
    k: i32,
    group_size: i32,
    alpha: f32,
    beta: f32,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

#[cfg(feature = "cuda")]
type W4a8WorkspaceFn = unsafe extern "C" fn(m: i32, n: i32, k: i32) -> usize;

// Weight encoder: FP16 [N,K] -> unified-encoded INT4 [N,K/2] bytes + LUT
// packed FP8 scales [N, K/g, 8] bytes.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
type W4a8EncodeFp16Fn = unsafe extern "C" fn(
    w_fp16: *const std::ffi::c_void,
    n: i32,
    k: i32,
    group_size: i32,
    w_int4_out: *mut std::ffi::c_void,
    scales_packed_out: *mut std::ffi::c_void,
    scales_f32_workspace: *mut std::ffi::c_void,
    shuffle: i32,
    stream: *mut std::ffi::c_void,
) -> i32;

pub struct W4a8Lib {
    pub so_path: PathBuf,
    #[cfg(feature = "cuda")]
    _lib: crate::lib_so::AuthenticatedLibrary,
    #[cfg(feature = "cuda")]
    gemm_run: W4a8GemmFn,
    #[cfg(feature = "cuda")]
    gemm_ws: W4a8WorkspaceFn,
    #[cfg(feature = "cuda")]
    fn_encode_fp16: W4a8EncodeFp16Fn,
}

impl W4a8Lib {
    pub fn load(path: PathBuf) -> Result<Self> {
        if !path.exists() {
            return Err(RvllmError::cuda(
                "W4a8Lib::load missing .so",
                CudaErrorKind::LaunchFailed,
                CudaCtx {
                    stream: 0,
                    kernel: "w4a8",
                    launch: None,
                    device: -1,
                },
            ));
        }
        #[cfg(feature = "cuda")]
        unsafe {
            let _lib = crate::lib_so::load_authenticated_library(&path)?;
            let run_sym: libloading::Symbol<W4a8GemmFn> =
                _lib.get(b"rvllm_w4a8_gemm_run\0").map_err(|_| {
                    RvllmError::cuda(
                        "dlsym rvllm_w4a8_gemm_run",
                        CudaErrorKind::LaunchFailed,
                        CudaCtx {
                            stream: 0,
                            kernel: "w4a8",
                            launch: None,
                            device: -1,
                        },
                    )
                })?;
            let ws_sym: libloading::Symbol<W4a8WorkspaceFn> =
                _lib.get(b"rvllm_w4a8_gemm_workspace_size\0").map_err(|_| {
                    RvllmError::cuda(
                        "dlsym rvllm_w4a8_gemm_workspace_size",
                        CudaErrorKind::LaunchFailed,
                        CudaCtx {
                            stream: 0,
                            kernel: "w4a8",
                            launch: None,
                            device: -1,
                        },
                    )
                })?;
            let enc_sym: libloading::Symbol<W4a8EncodeFp16Fn> =
                _lib.get(b"rvllm_w4a8_encode_weight_fp16\0").map_err(|_| {
                    RvllmError::cuda(
                        "dlsym rvllm_w4a8_encode_weight_fp16",
                        CudaErrorKind::LaunchFailed,
                        CudaCtx {
                            stream: 0,
                            kernel: "w4a8",
                            launch: None,
                            device: -1,
                        },
                    )
                })?;
            let gemm_run = *run_sym;
            let gemm_ws = *ws_sym;
            let fn_encode_fp16 = *enc_sym;
            Ok(Self {
                so_path: path,
                _lib,
                gemm_run,
                gemm_ws,
                fn_encode_fp16,
            })
        }
        #[cfg(not(feature = "cuda"))]
        Ok(Self { so_path: path })
    }

    /// Per-shape workspace size.
    pub fn workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        #[cfg(feature = "cuda")]
        unsafe {
            if m <= 0 || n <= 0 || k <= 0 {
                return 0;
            }
            (self.gemm_ws)(m, n, k)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (m, n, k);
            0
        }
    }

    /// D = alpha * A_fp8 * B_w4_unquant + beta * C.
    ///
    /// - `a_fp8`: `[m, k]` RowMajor E4M3 activations, device pointer.
    /// - `b_int4_reordered`: `[k, n]` INT4 ColMajor AWQ-shuffled weights,
    ///   device pointer. Already offline-reordered to the LayoutB_Reordered
    ///   atom layout expected by the kernel.
    /// - `b_scales_packed`: `[n, k/group_size]` packed FP8 LUT blocks
    ///   (each block is 8 packed E4M3 scales). Device pointer.
    /// - `c_f16`: optional `[m, n]` RowMajor residual. Pass 0 if `beta==0`.
    /// - `d_f16`: `[m, n]` RowMajor output. Device pointer.
    /// - `workspace`/`workspace_bytes`: scratch; size via `workspace_size`.
    ///
    /// # Safety
    /// All device pointers must be valid for the duration of the call on
    /// the given stream.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn w4a8_gemm(
        &self,
        a_fp8: u64,
        b_int4_reordered: u64,
        b_scales_packed: u64,
        c_f16: u64,
        d_f16: u64,
        m: i32,
        n: i32,
        k: i32,
        group_size: i32,
        alpha: f32,
        beta: f32,
        workspace: u64,
        workspace_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let (mu, nu, ku) =
            crate::lib_so::validate_gemm_dims(m, n, k, stream, "w4a8_gemm dimensions")?;
        if group_size != 128
            || k % group_size != 0
            || k % 2 != 0
            || !alpha.is_finite()
            || !beta.is_finite()
        {
            return Err(validation_error("w4a8_gemm parameters", stream));
        }
        let mn = mu
            .checked_mul(nu)
            .ok_or_else(|| validation_error("w4a8_gemm output span", stream))?;
        let mk = mu
            .checked_mul(ku)
            .ok_or_else(|| validation_error("w4a8_gemm activation span", stream))?;
        let nk = nu
            .checked_mul(ku)
            .ok_or_else(|| validation_error("w4a8_gemm weight span", stream))?;
        let scale_elements = nu
            .checked_mul(ku / group_size as usize)
            .and_then(|value| value.checked_mul(8))
            .ok_or_else(|| validation_error("w4a8_gemm scale span", stream))?;
        let required_workspace = (self.gemm_ws)(m, n, k);
        if required_workspace > workspace_bytes {
            return Err(validation_error("w4a8_gemm workspace", stream));
        }
        crate::lib_so::validate_device_span(
            a_fp8,
            crate::lib_so::checked_span(mk, 1)?,
            stream,
            "w4a8_gemm activation span",
        )?;
        crate::lib_so::validate_device_span(
            b_int4_reordered,
            crate::lib_so::checked_span(nk / 2, 1)?,
            stream,
            "w4a8_gemm weight span",
        )?;
        crate::lib_so::validate_device_span(
            b_scales_packed,
            crate::lib_so::checked_span(scale_elements, 1)?,
            stream,
            "w4a8_gemm scale span",
        )?;
        if beta != 0.0 {
            crate::lib_so::validate_device_span(
                c_f16,
                crate::lib_so::checked_span(mn, 2)?,
                stream,
                "w4a8_gemm residual span",
            )?;
        }
        crate::lib_so::validate_device_span(
            d_f16,
            crate::lib_so::checked_span(mn, 2)?,
            stream,
            "w4a8_gemm output span",
        )?;
        crate::lib_so::validate_device_span(
            workspace,
            required_workspace,
            stream,
            "w4a8_gemm workspace",
        )?;
        let rc = (self.gemm_run)(
            a_fp8 as *const _,
            b_int4_reordered as *const _,
            b_scales_packed as *const _,
            c_f16 as *const _,
            d_f16 as *mut _,
            m,
            n,
            k,
            group_size,
            alpha,
            beta,
            workspace as *mut _,
            workspace_bytes,
            stream as *mut _,
        );
        if rc != 0 {
            return Err(RvllmError::cuda(
                "w4a8_gemm_run",
                CudaErrorKind::LaunchFailed,
                CudaCtx {
                    stream,
                    kernel: "rvllm_w4a8_gemm_run",
                    launch: None,
                    device: -1,
                },
            ));
        }
        Ok(())
    }

    /// Encode FP16 weights [N, K] (row-major, device ptr) to unified-
    /// encoded INT4 [N, K/2] bytes + LUT packed FP8 scales [N, K/g, 8]
    /// bytes. Needs a scratch buffer `scales_f32_ws` of at least
    /// `N * K/g * 4` bytes.
    ///
    /// # Safety
    /// All device pointers must be valid for the duration of the call.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn encode_fp16(
        &self,
        w_fp16: u64,
        n: i32,
        k: i32,
        group_size: i32,
        w_int4_out: u64,
        scales_packed_out: u64,
        scales_f32_ws: u64,
        shuffle: bool,
        stream: u64,
    ) -> Result<()> {
        let (_, nu, ku) =
            crate::lib_so::validate_gemm_dims(1, n, k, stream, "w4a8_encode_fp16 dimensions")?;
        if group_size != 128 || k % group_size != 0 || k % 2 != 0 {
            return Err(validation_error("w4a8_encode_fp16 parameters", stream));
        }
        let nk = nu
            .checked_mul(ku)
            .ok_or_else(|| validation_error("w4a8_encode_fp16 weight span", stream))?;
        let groups = nu
            .checked_mul(ku / group_size as usize)
            .ok_or_else(|| validation_error("w4a8_encode_fp16 scale span", stream))?;
        crate::lib_so::validate_device_span(
            w_fp16,
            crate::lib_so::checked_span(nk, 2)?,
            stream,
            "w4a8_encode_fp16 input span",
        )?;
        crate::lib_so::validate_device_span(
            w_int4_out,
            crate::lib_so::checked_span(nk / 2, 1)?,
            stream,
            "w4a8_encode_fp16 output span",
        )?;
        crate::lib_so::validate_device_span(
            scales_packed_out,
            crate::lib_so::checked_span(
                groups.checked_mul(8).ok_or_else(|| {
                    validation_error("w4a8_encode_fp16 packed-scale span", stream)
                })?,
                1,
            )?,
            stream,
            "w4a8_encode_fp16 packed-scale span",
        )?;
        crate::lib_so::validate_device_span(
            scales_f32_ws,
            crate::lib_so::checked_span(groups, 4)?,
            stream,
            "w4a8_encode_fp16 workspace span",
        )?;
        let rc = (self.fn_encode_fp16)(
            w_fp16 as *const _,
            n,
            k,
            group_size,
            w_int4_out as *mut _,
            scales_packed_out as *mut _,
            scales_f32_ws as *mut _,
            if shuffle { 1 } else { 0 },
            stream as *mut _,
        );
        if rc != 0 {
            return Err(RvllmError::cuda(
                "w4a8_encode_fp16",
                CudaErrorKind::LaunchFailed,
                CudaCtx {
                    stream,
                    kernel: "rvllm_w4a8_encode_weight_fp16",
                    launch: None,
                    device: -1,
                },
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn validation_error(op: &'static str, stream: u64) -> RvllmError {
    RvllmError::cuda(
        op,
        CudaErrorKind::Other,
        CudaCtx {
            stream,
            kernel: "w4a8",
            launch: None,
            device: -1,
        },
    )
}
