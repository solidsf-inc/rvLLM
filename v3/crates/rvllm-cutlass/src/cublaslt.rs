//! cuBLASLt FP8 GEMM wrappers.
//!
//! Three entry points share one dispatcher (`fp8_gemm_inner`):
//!   - `fp8_gemm`          : D = A * B^T
//!   - `fp8_gemm_bias`     : D = A * B^T + bias   (CUBLASLT_EPILOGUE_BIAS)
//!   - `fp8_gemm_residual` : D = A * B^T + C      (alpha=1, beta=1; C is residual)
//!
//! FP8 E4M3 TN on Hopper: we pass row-major inputs and let cuBLASLt
//! (column-major) see their transposes, swapping A/B arguments and
//! setting transa=T, transb=N.

#[cfg(feature = "cuda")]
use cudarc::cublaslt::sys as lt;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

#[cfg(feature = "cuda")]
const MAX_ALGO_CACHE_ENTRIES: usize = 4096;

#[cfg(any(feature = "cuda", test))]
fn bind_workspace_stream(bound: &mut Option<u64>, stream: u64) -> bool {
    match *bound {
        Some(existing) => existing == stream,
        None => {
            *bound = Some(stream);
            true
        }
    }
}

/// Key for the per-shape algorithm cache. Distinguishes plain / bias /
/// residual dispatch because the matmul descriptor differs and cuBLASLt's
/// heuristic returns different algos.
#[cfg(feature = "cuda")]
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct AlgoKey {
    m: i32,
    n: i32,
    k: i32,
    kind: u8,
}

pub struct CublasLt {
    #[cfg(feature = "cuda")]
    handle: lt::cublasLtHandle_t,
    workspace: u64,
    workspace_bytes: usize,
    /// Per-(M,N,K,kind) cache of the heuristic-picked algorithm. cuBLASLt's
    /// algo struct is opaque but `Copy+Hash+Eq` — we reuse it on subsequent
    /// calls with the same shape instead of re-running the heuristic.
    #[cfg(feature = "cuda")]
    algo_cache: Mutex<HashMap<AlgoKey, lt::cublasLtMatmulAlgo_t>>,
    /// Serializes host dispatch and permanently binds the shared workspace
    /// to one stream. Same-stream launches are ordered by CUDA; allowing a
    /// second stream would reuse the workspace before prior work completes.
    #[cfg(feature = "cuda")]
    dispatch_lock: Mutex<Option<u64>>,
}

unsafe impl Send for CublasLt {}
unsafe impl Sync for CublasLt {}

#[cfg(feature = "cuda")]
struct LtResources {
    desc: lt::cublasLtMatmulDesc_t,
    layout_a: lt::cublasLtMatrixLayout_t,
    layout_b: lt::cublasLtMatrixLayout_t,
    layout_d: lt::cublasLtMatrixLayout_t,
    preference: lt::cublasLtMatmulPreference_t,
}

#[cfg(feature = "cuda")]
impl LtResources {
    fn new() -> Self {
        Self {
            desc: std::ptr::null_mut(),
            layout_a: std::ptr::null_mut(),
            layout_b: std::ptr::null_mut(),
            layout_d: std::ptr::null_mut(),
            preference: std::ptr::null_mut(),
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for LtResources {
    fn drop(&mut self) {
        unsafe {
            if !self.preference.is_null() {
                let _ = lt::cublasLtMatmulPreferenceDestroy(self.preference);
            }
            if !self.layout_d.is_null() {
                let _ = lt::cublasLtMatrixLayoutDestroy(self.layout_d);
            }
            if !self.layout_b.is_null() {
                let _ = lt::cublasLtMatrixLayoutDestroy(self.layout_b);
            }
            if !self.layout_a.is_null() {
                let _ = lt::cublasLtMatrixLayoutDestroy(self.layout_a);
            }
            if !self.desc.is_null() {
                let _ = lt::cublasLtMatmulDescDestroy(self.desc);
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn gemm_extents(m: i32, n: i32, k: i32) -> Result<(usize, usize, usize, usize, usize, usize)> {
    let (m, n, k) = crate::lib_so::validate_gemm_dims(m, n, k, 0, "cuBLASLt dimensions")?;
    let mk = m
        .checked_mul(k)
        .ok_or_else(|| cublaslt_err("cuBLASLt A extent overflow"))?;
    let nk = n
        .checked_mul(k)
        .ok_or_else(|| cublaslt_err("cuBLASLt B extent overflow"))?;
    let mn = m
        .checked_mul(n)
        .ok_or_else(|| cublaslt_err("cuBLASLt D extent overflow"))?;
    Ok((m, n, k, mk, nk, mn))
}

impl CublasLt {
    #[cfg(feature = "cuda")]
    pub fn new(workspace: u64, workspace_bytes: usize) -> Result<Self> {
        unsafe {
            crate::lib_so::validate_device_span(
                workspace,
                workspace_bytes,
                0,
                "CublasLt workspace",
            )?;
        }
        let mut handle: lt::cublasLtHandle_t = std::ptr::null_mut();
        let r = unsafe { lt::cublasLtCreate(&mut handle) };
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(RvllmError::cuda(
                "cublasLtCreate",
                CudaErrorKind::Other,
                CudaCtx::setup(),
            ));
        }
        Ok(Self {
            handle,
            workspace,
            workspace_bytes,
            algo_cache: Mutex::new(HashMap::new()),
            dispatch_lock: Mutex::new(None),
        })
    }

    #[cfg(feature = "cuda")]
    fn cached_algo(&self, key: AlgoKey) -> Result<Option<lt::cublasLtMatmulAlgo_t>> {
        self.algo_cache
            .lock()
            .map(|cache| cache.get(&key).copied())
            .map_err(|_| cublaslt_err("cuBLASLt algorithm cache poisoned"))
    }

    #[cfg(feature = "cuda")]
    fn cache_algo(&self, key: AlgoKey, algo: lt::cublasLtMatmulAlgo_t) -> Result<()> {
        let mut cache = self
            .algo_cache
            .lock()
            .map_err(|_| cublaslt_err("cuBLASLt algorithm cache poisoned"))?;
        if cache.len() < MAX_ALGO_CACHE_ENTRIES || cache.contains_key(&key) {
            cache.insert(key, algo);
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(workspace: u64, workspace_bytes: usize) -> Result<Self> {
        Ok(Self {
            workspace,
            workspace_bytes,
        })
    }

    /// Plain FP8 E4M3 matmul: D = A * B^T.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_gemm(
        &self,
        a_fp8: u64,
        b_fp8: u64,
        d_f16: u64,
        m: i32,
        n: i32,
        k: i32,
        a_scale: u64,
        b_scale: u64,
        stream: u64,
    ) -> Result<()> {
        self.fp8_gemm_inner(
            a_fp8, b_fp8, 0, 0, d_f16, m, n, k, a_scale, b_scale, stream, false, 0, None,
        )
    }

    /// Plain FP8 E4M3 matmul with bf16 output: D_bf16 = A * B^T.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_gemm_bf16(
        &self,
        a_fp8: u64,
        b_fp8: u64,
        d_bf16: u64,
        m: i32,
        n: i32,
        k: i32,
        a_scale: u64,
        b_scale: u64,
        stream: u64,
    ) -> Result<()> {
        self.fp8_gemm_inner(
            a_fp8, b_fp8, 0, 0, d_bf16, m, n, k, a_scale, b_scale, stream, false, 1, None,
        )
    }

    /// Plain FP8 E4M3 matmul with f32 output: D_f32 = A * B^T.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_gemm_f32(
        &self,
        a_fp8: u64,
        b_fp8: u64,
        d_f32: u64,
        m: i32,
        n: i32,
        k: i32,
        a_scale: u64,
        b_scale: u64,
        stream: u64,
    ) -> Result<()> {
        self.fp8_gemm_inner(
            a_fp8, b_fp8, 0, 0, d_f32, m, n, k, a_scale, b_scale, stream, false, 2, None,
        )
    }

    /// FP8 matmul with f32 output and per-channel weight scales (OUTER_VEC_32F).
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_gemm_f32_channelscale(
        &self,
        a_fp8: u64,
        b_fp8: u64,
        d_f32: u64,
        m: i32,
        n: i32,
        k: i32,
        a_scale: u64,
        b_channelscale: u64,
        stream: u64,
    ) -> Result<()> {
        self.fp8_gemm_inner(
            a_fp8,
            b_fp8,
            0,
            0,
            d_f32,
            m,
            n,
            k,
            a_scale,
            0,
            stream,
            false,
            2,
            Some(b_channelscale),
        )
    }

    /// F16 x F16 matmul with F32 output: D_f32 = A_f16 * B_f16^T.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn f16_gemm_f32(
        &self,
        a_f16: u64,
        b_f16: u64,
        d_f32: u64,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> Result<()> {
        let mut dispatch = self
            .dispatch_lock
            .lock()
            .map_err(|_| cublaslt_err("cuBLASLt dispatch lock poisoned"))?;
        let (mu, nu, _ku, mk, nk, mn) = gemm_extents(m, n, k)?;
        crate::lib_so::validate_device_span(
            a_f16,
            crate::lib_so::checked_span(mk, 2)?,
            stream,
            "cublasLt f16 A span",
        )?;
        crate::lib_so::validate_device_span(
            b_f16,
            crate::lib_so::checked_span(nk, 2)?,
            stream,
            "cublasLt f16 B span",
        )?;
        crate::lib_so::validate_device_span(
            d_f32,
            crate::lib_so::checked_span(mn, 4)?,
            stream,
            "cublasLt f32 D span",
        )?;
        crate::lib_so::validate_device_span(
            self.workspace,
            self.workspace_bytes,
            stream,
            "cublasLt workspace",
        )?;
        let mut resources = LtResources::new();
        let rc = lt::cublasLtMatmulDescCreate(
            &mut resources.desc,
            lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            lt::cudaDataType_t::CUDA_R_32F,
        );
        if rc != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("cublasLtMatmulDescCreate(f16)"));
        }
        let transa: i32 = 1;
        let transb: i32 = 0;
        set_attr(
            resources.desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &transa as *const _ as *const _,
            std::mem::size_of_val(&transa),
        )?;
        set_attr(
            resources.desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &transb as *const _ as *const _,
            std::mem::size_of_val(&transb),
        )?;

        let r = lt::cublasLtMatrixLayoutCreate(
            &mut resources.layout_a,
            lt::cudaDataType_t::CUDA_R_16F,
            k as u64,
            nu as u64,
            k as i64,
        );
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("layout A(f16)"));
        }
        let r = lt::cublasLtMatrixLayoutCreate(
            &mut resources.layout_b,
            lt::cudaDataType_t::CUDA_R_16F,
            k as u64,
            mu as u64,
            k as i64,
        );
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("layout B(f16)"));
        }
        let r = lt::cublasLtMatrixLayoutCreate(
            &mut resources.layout_d,
            lt::cudaDataType_t::CUDA_R_32F,
            nu as u64,
            mu as u64,
            n as i64,
        );
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("layout D(f16)"));
        }

        let key = AlgoKey { m, n, k, kind: 20 };
        let cached_algo = self.cached_algo(key)?;
        let algo = if let Some(a) = cached_algo {
            a
        } else {
            let status = lt::cublasLtMatmulPreferenceCreate(&mut resources.preference);
            if status != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(cublaslt_err("preference create(f16)"));
            }
            let ws = self.workspace_bytes;
            let status = lt::cublasLtMatmulPreferenceSetAttribute(
                resources.preference,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws as *const _ as *const _,
                std::mem::size_of::<usize>(),
            );
            if status != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(cublaslt_err("preference workspace(f16)"));
            }
            let mut heur: [lt::cublasLtMatmulHeuristicResult_t; 8] = std::mem::zeroed();
            let mut ret: i32 = 0;
            let r = lt::cublasLtMatmulAlgoGetHeuristic(
                self.handle,
                resources.desc,
                resources.layout_a,
                resources.layout_b,
                resources.layout_d,
                resources.layout_d,
                resources.preference,
                8,
                heur.as_mut_ptr(),
                &mut ret,
            );
            if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS || ret == 0 {
                return Err(cublaslt_err("heuristic(f16)"));
            }
            // Select a successful algorithm within the configured workspace.
            let mut best_opt = None;
            for i in 0..(ret as usize).min(heur.len()) {
                if heur[i].state == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                    && heur[i].workspaceSize <= self.workspace_bytes
                {
                    best_opt = Some(heur[i].algo);
                    break;
                }
            }
            let best = match best_opt {
                Some(a) => a,
                None => {
                    return Err(cublaslt_err("heuristic(f16): no SUCCESS-state algo"));
                }
            };
            self.cache_algo(key, best)?;
            best
        };

        let one: f32 = 1.0;
        let zero: f32 = 0.0;
        if !bind_workspace_stream(&mut dispatch, stream) {
            return Err(cublaslt_err(
                "CublasLt workspace cannot be shared across CUDA streams",
            ));
        }
        let r = lt::cublasLtMatmul(
            self.handle,
            resources.desc,
            &one as *const _ as *const _,
            b_f16 as *const _,
            resources.layout_a,
            a_f16 as *const _,
            resources.layout_b,
            &zero as *const _ as *const _,
            d_f32 as *const _,
            resources.layout_d,
            d_f32 as *mut _,
            resources.layout_d,
            &algo,
            self.workspace as *mut _,
            self.workspace_bytes,
            stream as _,
        );
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("cublasLtMatmul(f16)"));
        }
        Ok(())
    }

    /// FP8 matmul with row-broadcast f16 bias epilogue.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_gemm_bias(
        &self,
        a_fp8: u64,
        b_fp8: u64,
        bias_f16: u64,
        d_f16: u64,
        m: i32,
        n: i32,
        k: i32,
        a_scale: u64,
        b_scale: u64,
        stream: u64,
    ) -> Result<()> {
        self.fp8_gemm_inner(
            a_fp8, b_fp8, bias_f16, 0, d_f16, m, n, k, a_scale, b_scale, stream, false, 0, None,
        )
    }

    /// FP8 matmul with residual-add epilogue: D = A*B^T + residual (C).
    /// `residual_f16` and `d_f16` may alias to do the add in place.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fp8_gemm_residual(
        &self,
        a_fp8: u64,
        b_fp8: u64,
        residual_f16: u64,
        d_f16: u64,
        m: i32,
        n: i32,
        k: i32,
        a_scale: u64,
        b_scale: u64,
        stream: u64,
    ) -> Result<()> {
        self.fp8_gemm_inner(
            a_fp8,
            b_fp8,
            0,
            residual_f16,
            d_f16,
            m,
            n,
            k,
            a_scale,
            b_scale,
            stream,
            true,
            0,
            None,
        )
    }

    /// Shared body. `bias_f16=0` means no bias epilogue. `beta_one=true`
    /// requires `c_residual` and enables the residual path.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn fp8_gemm_inner(
        &self,
        a_fp8: u64,
        b_fp8: u64,
        bias_f16: u64,
        c_residual: u64,
        d_f16: u64,
        m: i32,
        n: i32,
        k: i32,
        a_scale: u64,
        b_scale: u64,
        stream: u64,
        beta_one: bool,
        d_out_type: u8,              // 0=f16, 1=bf16, 2=f32
        b_channelscale: Option<u64>, // per-channel weight scale (OUTER_VEC_32F)
    ) -> Result<()> {
        let mut dispatch = self
            .dispatch_lock
            .lock()
            .map_err(|_| cublaslt_err("cuBLASLt dispatch lock poisoned"))?;
        let (mu, nu, _ku, mk, nk, mn) = gemm_extents(m, n, k)?;
        if d_out_type > 2
            || (bias_f16 != 0 && beta_one)
            || (beta_one && d_out_type != 0)
            || b_channelscale == Some(0)
        {
            return Err(cublaslt_err("invalid cuBLASLt FP8 dispatch mode"));
        }
        crate::lib_so::validate_device_span(
            a_fp8,
            crate::lib_so::checked_span(mk, 1)?,
            stream,
            "cublasLt FP8 A span",
        )?;
        crate::lib_so::validate_device_span(
            b_fp8,
            crate::lib_so::checked_span(nk, 1)?,
            stream,
            "cublasLt FP8 B span",
        )?;
        crate::lib_so::validate_device_span(a_scale, 4, stream, "cublasLt A scale")?;
        if let Some(scale) = b_channelscale {
            crate::lib_so::validate_device_span(
                scale,
                crate::lib_so::checked_span(nu, 4)?,
                stream,
                "cublasLt B channel scale",
            )?;
        } else {
            crate::lib_so::validate_device_span(b_scale, 4, stream, "cublasLt B scale")?;
        }
        let output_bytes = match d_out_type {
            2 => 4,
            _ => 2,
        };
        crate::lib_so::validate_device_span(
            d_f16,
            crate::lib_so::checked_span(mn, output_bytes)?,
            stream,
            "cublasLt D span",
        )?;
        if bias_f16 != 0 {
            crate::lib_so::validate_device_span(
                bias_f16,
                crate::lib_so::checked_span(nu, 2)?,
                stream,
                "cublasLt bias span",
            )?;
        }
        if beta_one {
            crate::lib_so::validate_device_span(
                c_residual,
                crate::lib_so::checked_span(mn, 2)?,
                stream,
                "cublasLt residual span",
            )?;
        }
        crate::lib_so::validate_device_span(
            self.workspace,
            self.workspace_bytes,
            stream,
            "cublasLt workspace",
        )?;
        let mut resources = LtResources::new();
        let rc = lt::cublasLtMatmulDescCreate(
            &mut resources.desc,
            lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            lt::cudaDataType_t::CUDA_R_32F,
        );
        if rc != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("cublasLtMatmulDescCreate"));
        }

        let transa: i32 = 1; // T
        let transb: i32 = 0; // N
        set_attr(
            resources.desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &transa as *const _ as *const _,
            std::mem::size_of_val(&transa),
        )?;
        set_attr(
            resources.desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &transb as *const _ as *const _,
            std::mem::size_of_val(&transb),
        )?;

        if bias_f16 != 0 {
            let epi = lt::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS;
            set_attr(
                resources.desc,
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                &epi as *const _ as *const _,
                std::mem::size_of_val(&epi),
            )?;
            set_attr(
                resources.desc,
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias_f16 as *const _ as *const _,
                std::mem::size_of_val(&bias_f16),
            )?;
        }

        // TN swap: cuBLAS A = our weight (b_fp8), cuBLAS B = our activation (a_fp8).
        // Scale pointers must match: A_SCALE = weight scale, B_SCALE = activation scale.
        let cublas_a_scale = if let Some(cs) = b_channelscale {
            cs
        } else {
            b_scale
        };
        let cublas_b_scale = a_scale;
        set_attr(
            resources.desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &cublas_a_scale as *const _ as *const _,
            std::mem::size_of_val(&cublas_a_scale),
        )?;
        set_attr(
            resources.desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &cublas_b_scale as *const _ as *const _,
            std::mem::size_of_val(&cublas_b_scale),
        )?;

        if b_channelscale.is_some() {
            let scale_mode: u32 = 3; // OUTER_VEC_32F
            let attr_a_scale_mode: u32 = 31; // CUBLASLT_MATMUL_DESC_A_SCALE_MODE
            set_attr(
                resources.desc,
                unsafe {
                    std::mem::transmute::<u32, lt::cublasLtMatmulDescAttributes_t>(
                        attr_a_scale_mode,
                    )
                },
                &scale_mode as *const _ as *const _,
                std::mem::size_of_val(&scale_mode),
            )?;
        }

        // Layouts: col-major view of our row-major buffers, TN.
        let r = lt::cublasLtMatrixLayoutCreate(
            &mut resources.layout_a,
            lt::cudaDataType_t::CUDA_R_8F_E4M3,
            k as u64,
            nu as u64,
            k as i64,
        );
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("layout A"));
        }
        let r = lt::cublasLtMatrixLayoutCreate(
            &mut resources.layout_b,
            lt::cudaDataType_t::CUDA_R_8F_E4M3,
            k as u64,
            mu as u64,
            k as i64,
        );
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("layout B"));
        }
        let d_type = match d_out_type {
            1 => lt::cudaDataType_t::CUDA_R_16BF,
            2 => lt::cudaDataType_t::CUDA_R_32F,
            _ => lt::cudaDataType_t::CUDA_R_16F,
        };
        let r = lt::cublasLtMatrixLayoutCreate(
            &mut resources.layout_d,
            d_type,
            nu as u64,
            mu as u64,
            n as i64,
        );
        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("layout D"));
        }

        // Cache the selected algorithm by every descriptor-affecting mode.
        let key = AlgoKey {
            m,
            n,
            k,
            kind: match (
                bias_f16 != 0,
                beta_one,
                d_out_type,
                b_channelscale.is_some(),
            ) {
                (true, _, _, true) => 31u8,
                (true, _, _, false) => 1u8,
                (_, true, _, true) => 32 + d_out_type,
                (_, true, _, false) => 2 + d_out_type,
                (_, false, _, true) => 40 + d_out_type,
                (_, false, _, false) => 10 + d_out_type,
            },
        };
        let cached_algo = self.cached_algo(key)?;
        let algo = if let Some(a) = cached_algo {
            a
        } else {
            let r = lt::cublasLtMatmulPreferenceCreate(&mut resources.preference);
            if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(cublaslt_err("preference create"));
            }
            let ws_bytes = self.workspace_bytes;
            let r = lt::cublasLtMatmulPreferenceSetAttribute(
                resources.preference,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_bytes as *const _ as *const _,
                std::mem::size_of::<usize>(),
            );
            if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(cublaslt_err("pref set workspace"));
            }
            let mut heur: [lt::cublasLtMatmulHeuristicResult_t; 8] = std::mem::zeroed();
            let mut ret: i32 = 0;
            let r = lt::cublasLtMatmulAlgoGetHeuristic(
                self.handle,
                resources.desc,
                resources.layout_a,
                resources.layout_b,
                resources.layout_d,
                resources.layout_d,
                resources.preference,
                8,
                heur.as_mut_ptr(),
                &mut ret,
            );
            if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS || ret == 0 {
                return Err(cublaslt_err("heuristic"));
            }
            // Select the first successful result whose workspace fits.
            let mut best_opt = None;
            for i in 0..(ret as usize).min(heur.len()) {
                if heur[i].state == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                    && heur[i].workspaceSize <= self.workspace_bytes
                {
                    best_opt = Some(heur[i].algo);
                    break;
                }
            }
            let best_algo = match best_opt {
                Some(a) => a,
                None => {
                    return Err(cublaslt_err("heuristic: no SUCCESS-state algo"));
                }
            };
            self.cache_algo(key, best_algo)?;
            best_algo
        };

        let one_f32: f32 = 1.0;
        let zero_f32: f32 = 0.0;
        let c_ptr = if beta_one {
            c_residual as *const _
        } else {
            d_f16 as *const _
        };
        if !bind_workspace_stream(&mut dispatch, stream) {
            return Err(cublaslt_err(
                "CublasLt workspace cannot be shared across CUDA streams",
            ));
        }
        let r = lt::cublasLtMatmul(
            self.handle,
            resources.desc,
            &one_f32 as *const _ as *const _,
            b_fp8 as *const _, // cublas "A" := our weight (transa=T)
            resources.layout_a,
            a_fp8 as *const _, // cublas "B" := our activation (transb=N)
            resources.layout_b,
            if beta_one { &one_f32 } else { &zero_f32 } as *const _ as *const _,
            c_ptr,
            resources.layout_d,
            d_f16 as *mut _,
            resources.layout_d,
            &algo,
            self.workspace as *mut _,
            self.workspace_bytes,
            stream as _,
        );

        if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublaslt_err("cublasLtMatmul"));
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn cublaslt_err(op: &'static str) -> RvllmError {
    RvllmError::cuda(op, CudaErrorKind::LaunchFailed, CudaCtx::setup())
}

#[cfg(feature = "cuda")]
unsafe fn set_attr(
    desc: lt::cublasLtMatmulDesc_t,
    attr: lt::cublasLtMatmulDescAttributes_t,
    buf: *const core::ffi::c_void,
    size: usize,
) -> Result<()> {
    let r = lt::cublasLtMatmulDescSetAttribute(desc, attr, buf, size);
    if r != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(cublaslt_err("cublasLtMatmulDescSetAttribute"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::bind_workspace_stream;

    #[test]
    fn workspace_is_bound_to_one_stream() {
        let mut bound = None;
        assert!(bind_workspace_stream(&mut bound, 7));
        assert!(bind_workspace_stream(&mut bound, 7));
        assert!(!bind_workspace_stream(&mut bound, 8));
        assert_eq!(bound, Some(7));
    }
}

impl Drop for CublasLt {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            if !self.handle.is_null() {
                let _ = lt::cublasLtDestroy(self.handle);
            }
        }
    }
}
