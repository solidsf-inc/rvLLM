//! `libcutlass_kernels.so` dlopen + variant fn-pointer table.
//!
//! Opens the CUTLASS shared library once at engine init, resolves every
//! variant that appears in the autotune `Policy`, and caches the fn
//! pointers for zero-cost dispatch. A variant referenced by the policy
//! that's missing from the .so returns a typed error at load time — the
//! engine refuses to start rather than silently downgrade.

#[cfg(feature = "cuda")]
use std::ffi::c_void;
use std::path::PathBuf;

use rvllm_core::{CutlassCtx, CutlassError, Result, RvllmError};

use crate::variants::VariantId;

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub(crate) struct AuthenticatedLibrary {
    library: libloading::Library,
    #[cfg(target_os = "linux")]
    _backing: std::fs::File,
}

#[cfg(feature = "cuda")]
impl AuthenticatedLibrary {
    pub(crate) unsafe fn get<T>(
        &self,
        symbol: &[u8],
    ) -> std::result::Result<libloading::Symbol<'_, T>, libloading::Error> {
        self.library.get(symbol)
    }
}

/// Authenticate a shared object through the release manifest, then load the
/// verified bytes from an immutable anonymous file. The original path is not
/// reopened after verification.
#[cfg(feature = "cuda")]
pub(crate) fn load_authenticated_library(path: &std::path::Path) -> Result<AuthenticatedLibrary> {
    if !path.is_file() {
        return Err(cutlass_miss(path));
    }
    let canonical = path
        .canonicalize()
        .map_err(|error| so_integrity_error(path, format!("canonicalize: {error}")))?;
    let manifest_path = path
        .parent()
        .ok_or_else(|| so_integrity_error(path, "shared object has no parent directory".into()))?
        .join("manifest.json");
    let manifest = rvllm_kernels::manifest::KernelManifest::load_and_verify(&manifest_path)?;
    let mut matches = manifest
        .manifest()
        .entries
        .keys()
        .filter(|name| manifest.path_of(name).as_deref() == Some(canonical.as_path()));
    let name = matches.next().ok_or_else(|| {
        so_integrity_error(path, "shared object is not in the verified manifest".into())
    })?;
    if matches.next().is_some() {
        return Err(so_integrity_error(
            path,
            "shared object appears more than once in the verified manifest".into(),
        ));
    }
    let artifact = manifest
        .artifact(name)
        .ok_or_else(|| so_integrity_error(path, "verified artifact disappeared".into()))?;
    if artifact.kind() != rvllm_kernels::manifest::ArtifactKind::SharedObject
        || !artifact.abi().starts_with("rvllm-cuda-so-v")
    {
        return Err(so_integrity_error(
            path,
            format!("artifact has incompatible ABI {:?}", artifact.abi()),
        ));
    }
    load_verified_bytes(path, artifact.bytes())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn load_verified_bytes(path: &std::path::Path, bytes: &[u8]) -> Result<AuthenticatedLibrary> {
    use std::io::Write;
    use std::os::fd::{AsRawFd, FromRawFd};

    const MFD_CLOEXEC: u32 = 0x0001;
    const MFD_ALLOW_SEALING: u32 = 0x0002;
    const F_ADD_SEALS: i32 = 1033;
    const F_SEAL_SEAL: i32 = 0x0001;
    const F_SEAL_SHRINK: i32 = 0x0002;
    const F_SEAL_GROW: i32 = 0x0004;
    const F_SEAL_WRITE: i32 = 0x0008;

    unsafe extern "C" {
        fn memfd_create(name: *const std::ffi::c_char, flags: u32) -> i32;
        fn fcntl(fd: i32, command: i32, ...) -> i32;
    }

    let name = std::ffi::CString::new("rvllm-authenticated-kernel").expect("static string");
    let fd = unsafe { memfd_create(name.as_ptr(), MFD_CLOEXEC | MFD_ALLOW_SEALING) };
    if fd < 0 {
        return Err(so_integrity_error(
            path,
            format!("memfd_create: {}", std::io::Error::last_os_error()),
        ));
    }
    let mut backing = unsafe { std::fs::File::from_raw_fd(fd) };
    backing
        .write_all(bytes)
        .and_then(|_| backing.flush())
        .map_err(|error| {
            so_integrity_error(path, format!("materialize verified bytes: {error}"))
        })?;
    let seals = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE;
    if unsafe { fcntl(backing.as_raw_fd(), F_ADD_SEALS, seals) } != 0 {
        return Err(so_integrity_error(
            path,
            format!("seal verified bytes: {}", std::io::Error::last_os_error()),
        ));
    }
    let fd_path = PathBuf::from(format!("/proc/self/fd/{}", backing.as_raw_fd()));
    let library = unsafe { libloading::Library::new(&fd_path) }
        .map_err(|error| so_integrity_error(path, format!("dlopen verified bytes: {error}")))?;
    Ok(AuthenticatedLibrary {
        library,
        _backing: backing,
    })
}

#[cfg(all(feature = "cuda", not(target_os = "linux")))]
fn load_verified_bytes(path: &std::path::Path, _bytes: &[u8]) -> Result<AuthenticatedLibrary> {
    Err(so_integrity_error(
        path,
        "authenticated CUDA shared-object loading requires Linux memfd sealing".into(),
    ))
}

// Non-residual FP8 GEMM variant fn.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub type Fp8GemmFn = unsafe extern "C" fn(
    output: *mut c_void,
    a: *const c_void,
    b: *const c_void,
    a_scales: *const c_void,
    b_scale: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

// Residual-fused FP8 GEMM variant fn (epilogue adds a host-provided C).
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub type Fp8GemmResidualFn = unsafe extern "C" fn(
    output: *mut c_void,
    a: *const c_void,
    b: *const c_void,
    a_scales: *const c_void,
    b_scale: *const c_void,
    residual: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

#[cfg(feature = "cuda")]
pub type WorkspaceSizeFn = unsafe extern "C" fn(m: i32, n: i32, k: i32) -> usize;

#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub type Fp8GemmChannelscaleFn = unsafe extern "C" fn(
    output: *mut c_void,
    a: *const c_void,
    b: *const c_void,
    row_scale: *const c_void,
    col_scale: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

#[cfg(feature = "cuda")]
pub type ChannelscaleWorkspaceFn = unsafe extern "C" fn(m: i32, n: i32, k: i32) -> usize;

/// Signature of `cutlass_fp8_gemm_blockscale_sm120` in
/// `libcutlass_sm120.so`. The scale tensors carry 128×128 blockwise
/// semantics (Gemma 4 fp8-block format) — `a_scale` is SFA sized
/// `[ceil(M/128), K/128]`, `b_scale` is SFB sized `[N/128, K/128]`.
/// That's DIFFERENT from the per-vector SM90 channelscale ABI above;
/// we keep them as distinct fn-types so a miswiring fails at compile
/// time instead of silently passing the wrong pointer shape.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub type Fp8GemmBlockscaleSm120Fn = unsafe extern "C" fn(
    output: *mut c_void,
    a: *const c_void,
    b: *const c_void,
    a_scale_sfa: *const c_void,
    b_scale_sfb: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

#[cfg(feature = "cuda")]
pub type BlockscaleSm120WorkspaceFn = unsafe extern "C" fn(m: i32, n: i32, k: i32) -> usize;

/// Scratch-sizing entry points for SFA / SFB staging tensors. Same
/// signature shape as `BlockscaleSm120WorkspaceFn` but taking only two
/// problem-shape ints (SFA depends on M,K; SFB depends on N,K).
#[cfg(feature = "cuda")]
pub type BlockscaleSm120SfBytesFn = unsafe extern "C" fn(a: i32, b: i32) -> usize;

/// Prep-kernel signature for the SFA broadcast + SFB transpose passes
/// that convert Gemma 4 fp8-block's per-token a_scale / row-major
/// b_chscale into the CUTLASS-layout SFA/SFB scratch tensors.
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
pub type BlockscaleSm120PrepFn = unsafe extern "C" fn(
    src: *const c_void,
    dst: *mut c_void,
    dim_outer: i32,
    dim_inner: i32,
    stream: *mut c_void,
) -> i32;

#[cfg(feature = "cuda")]
pub(crate) fn validate_gemm_dims(
    m: i32,
    n: i32,
    k: i32,
    stream: u64,
    kernel: &'static str,
) -> Result<(usize, usize, usize)> {
    if m <= 0 || n <= 0 || k <= 0 {
        return Err(launch_validation_error(kernel, stream));
    }
    Ok((m as usize, n as usize, k as usize))
}

#[cfg(feature = "cuda")]
pub(crate) fn checked_span(elements: usize, bytes_per_element: usize) -> Result<usize> {
    elements
        .checked_mul(bytes_per_element)
        .ok_or_else(|| launch_validation_error("device byte-span overflow", 0))
}

#[cfg(feature = "cuda")]
pub(crate) unsafe fn validate_device_span(
    ptr: u64,
    bytes: usize,
    stream: u64,
    kernel: &'static str,
) -> Result<()> {
    use cudarc::driver::sys::{cuMemGetAddressRange_v2, CUresult};

    if bytes == 0 {
        return Ok(());
    }
    let last = ptr
        .checked_add((bytes - 1) as u64)
        .ok_or_else(|| launch_validation_error(kernel, stream))?;
    if ptr == 0 || last < ptr {
        return Err(launch_validation_error(kernel, stream));
    }
    let mut base = 0u64;
    let mut allocation_bytes = 0usize;
    let status = cuMemGetAddressRange_v2(&mut base, &mut allocation_bytes, ptr);
    if status != CUresult::CUDA_SUCCESS {
        return Err(launch_validation_error(kernel, stream));
    }
    let allocation_end = base
        .checked_add(allocation_bytes as u64)
        .ok_or_else(|| launch_validation_error(kernel, stream))?;
    let requested_end = last
        .checked_add(1)
        .ok_or_else(|| launch_validation_error(kernel, stream))?;
    if ptr < base || requested_end > allocation_end {
        return Err(launch_validation_error(kernel, stream));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn launch_validation_error(kernel: &'static str, stream: u64) -> RvllmError {
    RvllmError::cutlass(
        CutlassError::KernelLaunchFailed {
            variant: 0,
            cuda: rvllm_core::CudaErrorKind::Other,
        },
        CutlassCtx { kernel, stream },
    )
}

/// Resolved CUTLASS .so + variant fn-pointer table.
#[derive(Debug)]
pub struct CutlassLib {
    pub so_path: PathBuf,
    #[cfg(feature = "cuda")]
    _lib: AuthenticatedLibrary,
    /// Keyed by `VariantId`; loading fails if any requested export is absent.
    #[cfg(feature = "cuda")]
    pub fp8_gemm: std::collections::BTreeMap<VariantId, Fp8GemmFn>,
    #[cfg(feature = "cuda")]
    pub fp8_gemm_ws: std::collections::BTreeMap<VariantId, WorkspaceSizeFn>,
    #[cfg(feature = "cuda")]
    pub fp8_gemm_residual: std::collections::BTreeMap<VariantId, Fp8GemmResidualFn>,
    #[cfg(feature = "cuda")]
    pub fp8_gemm_residual_ws: std::collections::BTreeMap<VariantId, WorkspaceSizeFn>,
    #[cfg(feature = "cuda")]
    pub fp8_gemm_channelscale: Option<Fp8GemmChannelscaleFn>,
    #[cfg(feature = "cuda")]
    pub fp8_gemm_channelscale_ws: Option<ChannelscaleWorkspaceFn>,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CutlassLib {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CutlassLib {}

impl CutlassLib {
    #[cfg(feature = "cuda")]
    pub fn load(path: PathBuf, policy_variants: &[VariantId]) -> Result<Self> {
        if policy_variants.len() > 1024
            || policy_variants
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != policy_variants.len()
        {
            return Err(so_integrity_error(
                &path,
                "variant catalog is duplicated or exceeds 1024 entries".into(),
            ));
        }
        let lib = load_authenticated_library(&path)?;
        let mut fp8_gemm = std::collections::BTreeMap::new();
        let mut fp8_gemm_ws = std::collections::BTreeMap::new();
        let mut fp8_gemm_residual = std::collections::BTreeMap::new();
        let mut fp8_gemm_residual_ws = std::collections::BTreeMap::new();

        // VariantId -> shared-object symbol name.
        //   0..=99  -> cutlass_fp8_gemm[*]  (+ _workspace_size)
        //  100..    -> cutlass_fp8_gemm_residual[*]
        for &vid in policy_variants {
            let (fn_name_str, ws_name_str) = variant_symbol_names(vid);
            let is_residual = vid.0 >= 100;
            if is_residual {
                let fn_c = format!("{fn_name_str}\0");
                let ws_c = format!("{ws_name_str}\0");
                unsafe {
                    let f: libloading::Symbol<Fp8GemmResidualFn> = lib
                        .get(fn_c.as_bytes())
                        .map_err(|_| variant_missing(&path, vid, "fp8_gemm_residual"))?;
                    let w: libloading::Symbol<WorkspaceSizeFn> = lib
                        .get(ws_c.as_bytes())
                        .map_err(|_| variant_missing(&path, vid, "fp8_gemm_residual_ws"))?;
                    fp8_gemm_residual.insert(vid, *f);
                    fp8_gemm_residual_ws.insert(vid, *w);
                }
            } else {
                let fn_c = format!("{fn_name_str}\0");
                let ws_c = format!("{ws_name_str}\0");
                unsafe {
                    let f: libloading::Symbol<Fp8GemmFn> = lib
                        .get(fn_c.as_bytes())
                        .or_else(|_| {
                            if vid.0 == 0 {
                                lib.get(b"cutlass_fp8_gemm\0")
                            } else {
                                lib.get(fn_c.as_bytes())
                            }
                        })
                        .map_err(|_| variant_missing(&path, vid, "fp8_gemm"))?;
                    let w: libloading::Symbol<WorkspaceSizeFn> = lib
                        .get(ws_c.as_bytes())
                        .or_else(|_| {
                            if vid.0 == 0 {
                                lib.get(b"cutlass_fp8_gemm_workspace_size\0")
                            } else {
                                lib.get(ws_c.as_bytes())
                            }
                        })
                        .map_err(|_| variant_missing(&path, vid, "fp8_gemm_ws"))?;
                    fp8_gemm.insert(vid, *f);
                    fp8_gemm_ws.insert(vid, *w);
                }
            }
        }

        let fp8_gemm_channelscale: Option<Fp8GemmChannelscaleFn> =
            unsafe { lib.get(b"cutlass_fp8_gemm_channelscale\0").ok().map(|s| *s) };
        let fp8_gemm_channelscale_ws: Option<ChannelscaleWorkspaceFn> = unsafe {
            lib.get(b"cutlass_fp8_gemm_channelscale_workspace\0")
                .ok()
                .map(|s| *s)
        };
        if fp8_gemm_channelscale.is_some() != fp8_gemm_channelscale_ws.is_some() {
            return Err(so_integrity_error(
                &path,
                "channelscale ABI is partial: kernel and workspace symbols must both exist".into(),
            ));
        }

        Ok(Self {
            so_path: path,
            _lib: lib,
            fp8_gemm,
            fp8_gemm_ws,
            fp8_gemm_residual,
            fp8_gemm_residual_ws,
            fp8_gemm_channelscale,
            fp8_gemm_channelscale_ws,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(path: PathBuf, _policy_variants: &[VariantId]) -> Result<Self> {
        if !path.exists() {
            return Err(cutlass_miss(&path));
        }
        Ok(Self { so_path: path })
    }

    /// Dispatch a non-residual FP8 GEMM. `workspace` may be null if the
    /// plan's `workspace_bytes == 0`; otherwise it must point at >=
    /// `plan.workspace_bytes` of device memory.
    ///
    /// # Safety
    /// All pointers must be valid device memory for the kernel's duration.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm(
        &self,
        plan: &crate::plan::Fp8GemmPlan,
        output: u64,
        a: u64,
        b: u64,
        a_scales: u64,
        b_scale: u64,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        plan.check_workspace(workspace_size)?;
        let m = i32::try_from(plan.m)
            .map_err(|_| launch_validation_error("fp8_gemm dimensions", stream))?;
        let n = i32::try_from(plan.n)
            .map_err(|_| launch_validation_error("fp8_gemm dimensions", stream))?;
        let k = i32::try_from(plan.k)
            .map_err(|_| launch_validation_error("fp8_gemm dimensions", stream))?;
        let (mu, nu, ku) = validate_gemm_dims(m, n, k, stream, "fp8_gemm dimensions")?;
        if plan.dtype != rvllm_core::DType::Fp8E4M3 || plan.variant.0 >= 100 {
            return Err(launch_validation_error("fp8_gemm plan", stream));
        }
        let f = self.fp8_gemm.get(&plan.variant).ok_or_else(|| {
            variant_missing(&self.so_path, plan.variant, "fp8_gemm (runtime lookup)")
        })?;
        let ws = self.fp8_gemm_ws.get(&plan.variant).ok_or_else(|| {
            variant_missing(&self.so_path, plan.variant, "fp8_gemm_ws (runtime lookup)")
        })?;
        let required_workspace = ws(m, n, k);
        if required_workspace > workspace_size
            || required_workspace > usize::try_from(plan.workspace_bytes).unwrap_or(usize::MAX)
        {
            return Err(launch_validation_error("fp8_gemm workspace", stream));
        }
        let mk = mu
            .checked_mul(ku)
            .ok_or_else(|| launch_validation_error("fp8_gemm A span", stream))?;
        let nk = nu
            .checked_mul(ku)
            .ok_or_else(|| launch_validation_error("fp8_gemm B span", stream))?;
        let mn = mu
            .checked_mul(nu)
            .ok_or_else(|| launch_validation_error("fp8_gemm output span", stream))?;
        validate_device_span(a, checked_span(mk, 1)?, stream, "fp8_gemm A span")?;
        validate_device_span(b, checked_span(nk, 1)?, stream, "fp8_gemm B span")?;
        validate_device_span(output, checked_span(mn, 2)?, stream, "fp8_gemm output span")?;
        validate_device_span(
            a_scales,
            checked_span(mu, 4)?,
            stream,
            "fp8_gemm row scales",
        )?;
        validate_device_span(b_scale, 4, stream, "fp8_gemm weight scale")?;
        validate_device_span(workspace, required_workspace, stream, "fp8_gemm workspace")?;
        let rc = f(
            output as *mut c_void,
            a as *const c_void,
            b as *const c_void,
            a_scales as *const c_void,
            b_scale as *const c_void,
            m,
            n,
            k,
            workspace as *mut c_void,
            workspace_size,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: plan.variant.0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx {
                    kernel: "fp8_gemm",
                    stream,
                },
            ));
        }
        Ok(())
    }

    /// Same, residual-fused variant. `residual` is the C-tensor the
    /// epilogue adds into `output`.
    ///
    /// # Safety
    /// All pointers valid for the call.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm_residual(
        &self,
        plan: &crate::plan::Fp8GemmPlan,
        output: u64,
        a: u64,
        b: u64,
        a_scales: u64,
        b_scale: u64,
        residual: u64,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        plan.check_workspace(workspace_size)?;
        let m = i32::try_from(plan.m)
            .map_err(|_| launch_validation_error("fp8_gemm_residual dimensions", stream))?;
        let n = i32::try_from(plan.n)
            .map_err(|_| launch_validation_error("fp8_gemm_residual dimensions", stream))?;
        let k = i32::try_from(plan.k)
            .map_err(|_| launch_validation_error("fp8_gemm_residual dimensions", stream))?;
        let (mu, nu, ku) = validate_gemm_dims(m, n, k, stream, "fp8_gemm_residual dimensions")?;
        if plan.dtype != rvllm_core::DType::Fp8E4M3 || plan.variant.0 < 100 {
            return Err(launch_validation_error("fp8_gemm_residual plan", stream));
        }
        let f = self.fp8_gemm_residual.get(&plan.variant).ok_or_else(|| {
            variant_missing(
                &self.so_path,
                plan.variant,
                "fp8_gemm_residual (runtime lookup)",
            )
        })?;
        let ws = self
            .fp8_gemm_residual_ws
            .get(&plan.variant)
            .ok_or_else(|| {
                variant_missing(
                    &self.so_path,
                    plan.variant,
                    "fp8_gemm_residual_ws (runtime lookup)",
                )
            })?;
        let required_workspace = ws(m, n, k);
        if required_workspace > workspace_size
            || required_workspace > usize::try_from(plan.workspace_bytes).unwrap_or(usize::MAX)
        {
            return Err(launch_validation_error(
                "fp8_gemm_residual workspace",
                stream,
            ));
        }
        let mk = mu
            .checked_mul(ku)
            .ok_or_else(|| launch_validation_error("fp8_gemm_residual A span", stream))?;
        let nk = nu
            .checked_mul(ku)
            .ok_or_else(|| launch_validation_error("fp8_gemm_residual B span", stream))?;
        let mn = mu
            .checked_mul(nu)
            .ok_or_else(|| launch_validation_error("fp8_gemm_residual output span", stream))?;
        validate_device_span(a, checked_span(mk, 1)?, stream, "fp8_gemm_residual A span")?;
        validate_device_span(b, checked_span(nk, 1)?, stream, "fp8_gemm_residual B span")?;
        validate_device_span(
            output,
            checked_span(mn, 2)?,
            stream,
            "fp8_gemm_residual output span",
        )?;
        validate_device_span(
            residual,
            checked_span(mn, 2)?,
            stream,
            "fp8_gemm_residual input span",
        )?;
        validate_device_span(
            a_scales,
            checked_span(mu, 4)?,
            stream,
            "fp8_gemm_residual row scales",
        )?;
        validate_device_span(b_scale, 4, stream, "fp8_gemm_residual weight scale")?;
        validate_device_span(
            workspace,
            required_workspace,
            stream,
            "fp8_gemm_residual workspace",
        )?;
        let rc = f(
            output as *mut c_void,
            a as *const c_void,
            b as *const c_void,
            a_scales as *const c_void,
            b_scale as *const c_void,
            residual as *const c_void,
            m,
            n,
            k,
            workspace as *mut c_void,
            workspace_size,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: plan.variant.0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx {
                    kernel: "fp8_gemm_residual",
                    stream,
                },
            ));
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm_channelscale(
        &self,
        output: u64,
        a: u64,
        b: u64,
        row_scale: u64,
        col_scale: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        let (mu, nu, ku) = validate_gemm_dims(m, n, k, stream, "fp8_gemm_channelscale dimensions")?;
        let f = self.fp8_gemm_channelscale.ok_or_else(|| {
            RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::Other,
                },
                CutlassCtx {
                    kernel: "fp8_gemm_channelscale (missing from .so)",
                    stream,
                },
            )
        })?;
        let ws = self.fp8_gemm_channelscale_ws.ok_or_else(|| {
            launch_validation_error("fp8_gemm_channelscale workspace ABI", stream)
        })?;
        let required_workspace = ws(m, n, k);
        if required_workspace > workspace_size {
            return Err(launch_validation_error(
                "fp8_gemm_channelscale workspace",
                stream,
            ));
        }
        let mk = mu
            .checked_mul(ku)
            .ok_or_else(|| launch_validation_error("fp8_gemm_channelscale A span", stream))?;
        let nk = nu
            .checked_mul(ku)
            .ok_or_else(|| launch_validation_error("fp8_gemm_channelscale B span", stream))?;
        let mn = mu
            .checked_mul(nu)
            .ok_or_else(|| launch_validation_error("fp8_gemm_channelscale output span", stream))?;
        validate_device_span(
            a,
            checked_span(mk, 1)?,
            stream,
            "fp8_gemm_channelscale A span",
        )?;
        validate_device_span(
            b,
            checked_span(nk, 1)?,
            stream,
            "fp8_gemm_channelscale B span",
        )?;
        validate_device_span(
            output,
            checked_span(mn, 2)?,
            stream,
            "fp8_gemm_channelscale output span",
        )?;
        validate_device_span(
            row_scale,
            checked_span(mu, 4)?,
            stream,
            "fp8_gemm_channelscale row scale",
        )?;
        validate_device_span(
            col_scale,
            checked_span(nu, 4)?,
            stream,
            "fp8_gemm_channelscale col scale",
        )?;
        validate_device_span(
            workspace,
            required_workspace,
            stream,
            "fp8_gemm_channelscale workspace",
        )?;
        let rc = f(
            output as *mut c_void,
            a as *const c_void,
            b as *const c_void,
            row_scale as *const c_void,
            col_scale as *const c_void,
            m,
            n,
            k,
            workspace as *mut c_void,
            workspace_size,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx {
                    kernel: "fp8_gemm_channelscale",
                    stream,
                },
            ));
        }
        Ok(())
    }
}

// ============================================================================
// CutlassBackend — architecture-exact shared libraries plus explicit absence
// ============================================================================

/// Which CUTLASS backend the runtime is using on the live device.
///
///   * SM90 → `So(CutlassLib)` — dlopen `libcutlass_kernels.so`, fn-ptr
///     table keyed by `VariantId`.
///   * SM121 (Blackwell consumer) → `SoSm120(CutlassSm120Lib)` when a
///     `libcutlass_sm120.so` is found (built via
///     `kernels/build_cutlass_sm120_so.sh`), exposing the native
///     `cutlass_fp8_gemm_blockscale_sm120` kernel with correct
///     128×128 block-scale semantics for Gemma 4 fp8-block. Missing
///     optional coverage is represented by `Absent` for explicit routing.
///
/// `#[non_exhaustive]` leaves room for more backends to slide in.
#[derive(Debug)]
#[non_exhaustive]
pub enum CutlassBackend {
    /// Hopper SM90 .so.
    So(CutlassLib),
    /// Blackwell-Geforce (SM120/SM121) .so.
    SoSm120(CutlassSm120Lib),
    /// No CUTLASS coverage — callers must route through the PTX /
    /// cuBLASLt fallback.
    Absent,
}

/// Find the SM121-compatible blockscale `.so` via env override or next to the
/// selected architecture's other artifacts.
fn resolve_sm120_so_path(sm90_hint: &std::path::Path) -> Option<PathBuf> {
    if let Some(env) = std::env::var_os("RVLLM_CUTLASS_SM120_SO") {
        let p = PathBuf::from(env);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(parent) = sm90_hint.parent() {
        let candidate = parent.join("libcutlass_sm120.so");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Resolved `libcutlass_sm120.so` fixed-symbol dispatch table.
#[derive(Debug)]
pub struct CutlassSm120Lib {
    pub so_path: PathBuf,
    #[cfg(feature = "cuda")]
    _lib: AuthenticatedLibrary,
    #[cfg(feature = "cuda")]
    pub fp8_gemm_blockscale: Option<Fp8GemmBlockscaleSm120Fn>,
    #[cfg(feature = "cuda")]
    pub fp8_gemm_blockscale_ws: Option<BlockscaleSm120WorkspaceFn>,
    #[cfg(feature = "cuda")]
    pub sfa_bytes: Option<BlockscaleSm120SfBytesFn>,
    #[cfg(feature = "cuda")]
    pub sfb_bytes: Option<BlockscaleSm120SfBytesFn>,
    #[cfg(feature = "cuda")]
    pub prep_sfa: Option<BlockscaleSm120PrepFn>,
    #[cfg(feature = "cuda")]
    pub prep_sfb: Option<BlockscaleSm120PrepFn>,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CutlassSm120Lib {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CutlassSm120Lib {}

impl CutlassSm120Lib {
    /// Load a complete, manifest-authenticated SM120 ABI catalog.
    #[cfg(feature = "cuda")]
    pub fn load(path: PathBuf) -> Result<Self> {
        let lib = load_authenticated_library(&path)?;
        let fp8_gemm_blockscale: Option<Fp8GemmBlockscaleSm120Fn> = unsafe {
            lib.get(b"cutlass_fp8_gemm_blockscale_sm120\0")
                .ok()
                .map(|s| *s)
        };
        let fp8_gemm_blockscale_ws: Option<BlockscaleSm120WorkspaceFn> = unsafe {
            lib.get(b"cutlass_fp8_gemm_blockscale_sm120_workspace\0")
                .ok()
                .map(|s| *s)
        };
        let sfa_bytes: Option<BlockscaleSm120SfBytesFn> = unsafe {
            lib.get(b"cutlass_fp8_gemm_blockscale_sm120_sfa_bytes\0")
                .ok()
                .map(|s| *s)
        };
        let sfb_bytes: Option<BlockscaleSm120SfBytesFn> = unsafe {
            lib.get(b"cutlass_fp8_gemm_blockscale_sm120_sfb_bytes\0")
                .ok()
                .map(|s| *s)
        };
        let prep_sfa: Option<BlockscaleSm120PrepFn> = unsafe {
            lib.get(b"cutlass_fp8_gemm_blockscale_sm120_prep_sfa\0")
                .ok()
                .map(|s| *s)
        };
        let prep_sfb: Option<BlockscaleSm120PrepFn> = unsafe {
            lib.get(b"cutlass_fp8_gemm_blockscale_sm120_prep_sfb\0")
                .ok()
                .map(|s| *s)
        };
        if fp8_gemm_blockscale.is_none()
            || fp8_gemm_blockscale_ws.is_none()
            || sfa_bytes.is_none()
            || sfb_bytes.is_none()
            || prep_sfa.is_none()
            || prep_sfb.is_none()
        {
            return Err(so_integrity_error(
                &path,
                "SM120 ABI catalog is incomplete".into(),
            ));
        }
        Ok(Self {
            so_path: path,
            _lib: lib,
            fp8_gemm_blockscale,
            fp8_gemm_blockscale_ws,
            sfa_bytes,
            sfb_bytes,
            prep_sfa,
            prep_sfb,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load(path: PathBuf) -> Result<Self> {
        if !path.exists() {
            return Err(cutlass_miss(&path));
        }
        Ok(Self { so_path: path })
    }

    /// Workspace size CUTLASS asks for at this problem shape. `0`
    /// when the symbol is missing — the caller's workspace-size
    /// check will trip if a non-zero requirement was silently ignored.
    #[cfg(feature = "cuda")]
    pub fn workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        if m <= 0 || n <= 0 || k <= 0 {
            return 0;
        }
        self.fp8_gemm_blockscale_ws
            .map(|f| unsafe { f(m, n, k) })
            .unwrap_or(0)
    }

    /// SFA / SFB scratch sizes in bytes. `0` when the `.so` is missing
    /// the helper (legacy build) — caller must treat 0 as "CUTLASS
    /// path unavailable" and fall back.
    #[cfg(feature = "cuda")]
    pub fn sfa_bytes(&self, m: i32, k: i32) -> usize {
        if m <= 0 || k <= 0 {
            return 0;
        }
        self.sfa_bytes.map(|f| unsafe { f(m, k) }).unwrap_or(0)
    }
    #[cfg(feature = "cuda")]
    pub fn sfb_bytes(&self, n: i32, k: i32) -> usize {
        if n <= 0 || k <= 0 {
            return 0;
        }
        self.sfb_bytes.map(|f| unsafe { f(n, k) }).unwrap_or(0)
    }

    /// Populate SFA scratch from the per-token `a_scale[M]` vector.
    /// Each row scale is replicated across its K/128 entries in the
    /// CUTLASS SFA layout; no reduction is performed.
    ///
    /// # Safety
    /// `a_scale` and `sfa` must be valid device pointers for the
    /// kernel's duration; `sfa` must be at least `sfa_bytes(m, k)` wide.
    #[cfg(feature = "cuda")]
    pub unsafe fn launch_prep_sfa(
        &self,
        a_scale: u64,
        sfa: u64,
        m: i32,
        k: i32,
        stream: u64,
    ) -> Result<()> {
        let (mu, _, ku) = validate_gemm_dims(m, 1, k, stream, "prep_sfa dimensions")?;
        if k % 128 != 0 {
            return Err(launch_validation_error("prep_sfa K alignment", stream));
        }
        let f = self.prep_sfa.ok_or_else(|| {
            RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::Other,
                },
                CutlassCtx {
                    kernel: "prep_sfa_sm120 (missing from .so)",
                    stream,
                },
            )
        })?;
        let dst_bytes = self.sfa_bytes(m, k);
        if dst_bytes == 0 {
            return Err(launch_validation_error("prep_sfa output extent", stream));
        }
        let _ = ku;
        validate_device_span(a_scale, checked_span(mu, 4)?, stream, "prep_sfa input span")?;
        validate_device_span(sfa, dst_bytes, stream, "prep_sfa output span")?;
        let rc = f(
            a_scale as *const c_void,
            sfa as *mut c_void,
            m,
            k,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx {
                    kernel: "prep_sfa_sm120",
                    stream,
                },
            ));
        }
        Ok(())
    }

    /// Populate SFB scratch by transposing row-major `[N/128, K/128]`
    /// `b_chscale` into the CUTLASS SFB layout (N-tile fastest).
    ///
    /// # Safety
    /// `b_chscale` and `sfb` must be valid device pointers; `sfb`
    /// must be at least `sfb_bytes(n, k)` wide.
    #[cfg(feature = "cuda")]
    pub unsafe fn launch_prep_sfb(
        &self,
        b_chscale: u64,
        sfb: u64,
        n: i32,
        k: i32,
        stream: u64,
    ) -> Result<()> {
        let (_, nu, ku) = validate_gemm_dims(1, n, k, stream, "prep_sfb dimensions")?;
        if k % 128 != 0 {
            return Err(launch_validation_error("prep_sfb K alignment", stream));
        }
        let f = self.prep_sfb.ok_or_else(|| {
            RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::Other,
                },
                CutlassCtx {
                    kernel: "prep_sfb_sm120 (missing from .so)",
                    stream,
                },
            )
        })?;
        let dst_bytes = self.sfb_bytes(n, k);
        if dst_bytes == 0 {
            return Err(launch_validation_error("prep_sfb output extent", stream));
        }
        let src_elements = nu
            .div_ceil(128)
            .checked_mul(ku.div_ceil(128))
            .ok_or_else(|| launch_validation_error("prep_sfb input span", stream))?;
        validate_device_span(
            b_chscale,
            checked_span(src_elements, 4)?,
            stream,
            "prep_sfb input span",
        )?;
        validate_device_span(sfb, dst_bytes, stream, "prep_sfb output span")?;
        let rc = f(
            b_chscale as *const c_void,
            sfb as *mut c_void,
            n,
            k,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx {
                    kernel: "prep_sfb_sm120",
                    stream,
                },
            ));
        }
        Ok(())
    }

    /// # Safety
    /// All device pointers must be valid for the kernel's duration.
    /// `a_scale_sfa`, `b_scale_sfb` carry 128×128 block-scale
    /// semantics per `cutlass::detail::sm120_trivial_blockwise_scale_config`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm_blockscale(
        &self,
        output: u64,
        a: u64,
        b: u64,
        a_scale_sfa: u64,
        b_scale_sfb: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        let (mu, nu, ku) =
            validate_gemm_dims(m, n, k, stream, "fp8_gemm_blockscale_sm120 dimensions")?;
        if k % 128 != 0 {
            return Err(launch_validation_error(
                "fp8_gemm_blockscale_sm120 K alignment",
                stream,
            ));
        }
        let f = self.fp8_gemm_blockscale.ok_or_else(|| {
            RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::Other,
                },
                CutlassCtx {
                    kernel: "fp8_gemm_blockscale_sm120 (missing from .so)",
                    stream,
                },
            )
        })?;
        let required_workspace = self.workspace_size(m, n, k);
        let sfa_bytes = self.sfa_bytes(m, k);
        let sfb_bytes = self.sfb_bytes(n, k);
        if required_workspace > workspace_size || sfa_bytes == 0 || sfb_bytes == 0 {
            return Err(launch_validation_error(
                "fp8_gemm_blockscale_sm120 extents",
                stream,
            ));
        }
        let mk = mu
            .checked_mul(ku)
            .ok_or_else(|| launch_validation_error("fp8_gemm_blockscale_sm120 A span", stream))?;
        let nk = nu
            .checked_mul(ku)
            .ok_or_else(|| launch_validation_error("fp8_gemm_blockscale_sm120 B span", stream))?;
        let mn = mu.checked_mul(nu).ok_or_else(|| {
            launch_validation_error("fp8_gemm_blockscale_sm120 output span", stream)
        })?;
        validate_device_span(
            a,
            checked_span(mk, 1)?,
            stream,
            "fp8_gemm_blockscale_sm120 A span",
        )?;
        validate_device_span(
            b,
            checked_span(nk, 1)?,
            stream,
            "fp8_gemm_blockscale_sm120 B span",
        )?;
        validate_device_span(
            output,
            checked_span(mn, 2)?,
            stream,
            "fp8_gemm_blockscale_sm120 output span",
        )?;
        validate_device_span(
            a_scale_sfa,
            sfa_bytes,
            stream,
            "fp8_gemm_blockscale_sm120 SFA span",
        )?;
        validate_device_span(
            b_scale_sfb,
            sfb_bytes,
            stream,
            "fp8_gemm_blockscale_sm120 SFB span",
        )?;
        validate_device_span(
            workspace,
            required_workspace,
            stream,
            "fp8_gemm_blockscale_sm120 workspace",
        )?;
        let rc = f(
            output as *mut c_void,
            a as *const c_void,
            b as *const c_void,
            a_scale_sfa as *const c_void,
            b_scale_sfb as *const c_void,
            m,
            n,
            k,
            workspace as *mut c_void,
            workspace_size,
            stream as *mut c_void,
        );
        if rc != 0 {
            return Err(RvllmError::cutlass(
                CutlassError::KernelLaunchFailed {
                    variant: 0,
                    cuda: rvllm_core::CudaErrorKind::LaunchFailed,
                },
                CutlassCtx {
                    kernel: "fp8_gemm_blockscale_sm120",
                    stream,
                },
            ));
        }
        Ok(())
    }
}

impl CutlassBackend {
    /// Construct a backend for a device `CompileTarget`. On sm_121, resolve
    /// the architecture-specific block-scale shared object in this order:
    ///   1. env `RVLLM_CUTLASS_SM120_SO` (absolute path).
    ///   2. `<path.parent()>/libcutlass_sm120.so` — the per-architecture layout
    ///      produced by `kernels/build_cutlass_sm120_so.sh`, which
    ///      keeps the library beside the selected architecture's artifacts.
    /// If neither is present, return `Absent` for explicit caller routing.
    #[cfg(feature = "cuda")]
    pub fn load_for(
        target: Option<rvllm_core::CompileTarget>,
        path: PathBuf,
        policy_variants: &[VariantId],
    ) -> Result<Self> {
        if matches!(target, Some(rvllm_core::CompileTarget::Sm121)) {
            if let Some(sm120_path) = resolve_sm120_so_path(&path) {
                return Ok(CutlassBackend::SoSm120(CutlassSm120Lib::load(sm120_path)?));
            }
            return Ok(CutlassBackend::Absent);
        }
        // The generic library is built from Hopper-only SM90 schedules.
        // Ampere, Ada, and datacenter Blackwell must use their explicit
        // cuBLASLt/PTX fallbacks rather than dlopen incompatible code.
        if matches!(target, Some(rvllm_core::CompileTarget::Sm100)) {
            set_lt_fp8_default_off(true);
        }
        if !matches!(target, Some(rvllm_core::CompileTarget::Sm90)) {
            return Ok(CutlassBackend::Absent);
        }
        Ok(CutlassBackend::So(CutlassLib::load(path, policy_variants)?))
    }

    /// Without the `cuda` feature there is no runtime to dlopen
    /// against, so every backend collapses to `Absent`. Callers
    /// already have to route through the non-CUDA code paths; this
    /// keeps the signature shape identical to the cuda build.
    #[cfg(not(feature = "cuda"))]
    pub fn load_for(
        _target: Option<rvllm_core::CompileTarget>,
        _path: PathBuf,
        _policy_variants: &[VariantId],
    ) -> Result<Self> {
        Ok(CutlassBackend::Absent)
    }

    /// Path the `.so` was (or would be) loaded from — exposed for
    /// probe / diagnostic output. Empty `PathBuf` on the `Absent`
    /// variant.
    #[must_use]
    pub fn so_path(&self) -> std::path::PathBuf {
        match self {
            CutlassBackend::So(lib) => lib.so_path.clone(),
            CutlassBackend::SoSm120(lib) => lib.so_path.clone(),
            CutlassBackend::Absent => PathBuf::new(),
        }
    }

    /// Dispatch `launch_fp8_gemm` to the underlying backend.
    ///
    /// # Safety
    /// Same as `CutlassLib::launch_fp8_gemm`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm(
        &self,
        plan: &crate::plan::Fp8GemmPlan,
        output: u64,
        a: u64,
        b: u64,
        a_scales: u64,
        b_scale: u64,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        match self {
            CutlassBackend::So(lib) => lib.launch_fp8_gemm(
                plan,
                output,
                a,
                b,
                a_scales,
                b_scale,
                workspace,
                workspace_size,
                stream,
            ),
            CutlassBackend::Absent => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm (no compatible CUTLASS shared library for this target)",
                },
                CutlassCtx {
                    kernel: "fp8_gemm",
                    stream,
                },
            )),
            CutlassBackend::SoSm120(_) => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm (non-channelscale) — SoSm120 only ships the blockwise entry",
                },
                CutlassCtx {
                    kernel: "fp8_gemm",
                    stream,
                },
            )),
        }
    }

    /// Dispatch `launch_fp8_gemm_residual` to the underlying backend.
    ///
    /// # Safety
    /// Same as `CutlassLib::launch_fp8_gemm_residual`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm_residual(
        &self,
        plan: &crate::plan::Fp8GemmPlan,
        output: u64,
        a: u64,
        b: u64,
        a_scales: u64,
        b_scale: u64,
        residual: u64,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        match self {
            CutlassBackend::So(lib) => lib.launch_fp8_gemm_residual(
                plan,
                output,
                a,
                b,
                a_scales,
                b_scale,
                residual,
                workspace,
                workspace_size,
                stream,
            ),
            CutlassBackend::Absent => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm_residual (no compatible CUTLASS shared library for this target)",
                },
                CutlassCtx {
                    kernel: "fp8_gemm_residual",
                    stream,
                },
            )),
            CutlassBackend::SoSm120(_) => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm_residual — SoSm120 has no residual-fused entry yet",
                },
                CutlassCtx {
                    kernel: "fp8_gemm_residual",
                    stream,
                },
            )),
        }
    }
}

impl CutlassBackend {
    /// Dispatch `launch_fp8_gemm_channelscale` — upstream added this
    /// as a row×col-scale epilogue variant. `Absent` has no
    /// equivalent kernel; it returns `FeatureNotAvailable`.
    ///
    /// # Safety
    /// Same as `CutlassLib::launch_fp8_gemm_channelscale`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm_channelscale(
        &self,
        output: u64,
        a: u64,
        b: u64,
        row_scale: u64,
        col_scale: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        match self {
            CutlassBackend::So(lib) => lib.launch_fp8_gemm_channelscale(
                output,
                a,
                b,
                row_scale,
                col_scale,
                m,
                n,
                k,
                workspace,
                workspace_size,
                stream,
            ),
            CutlassBackend::Absent => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm_channelscale (no compatible CUTLASS shared library for this target)",
                },
                CutlassCtx {
                    kernel: "fp8_gemm_channelscale",
                    stream,
                },
            )),
            CutlassBackend::SoSm120(_) => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm_channelscale — SoSm120 uses the blockscale ABI; call launch_fp8_gemm_blockscale_sm120 instead",
                },
                CutlassCtx {
                    kernel: "fp8_gemm_channelscale",
                    stream,
                },
            )),
        }
    }

    /// Dispatch `launch_fp8_gemm_blockscale_sm120` — native Blackwell-
    /// Geforce 128×128 blockwise FP8 GEMM. Only the `SoSm120` variant
    /// has this kernel; `So` (SM90) and `Absent` return
    /// `FeatureNotAvailable` so the caller falls back to the PTX /
    /// channelscale path.
    ///
    /// # Safety
    /// Same as `CutlassSm120Lib::launch_fp8_gemm_blockscale`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_fp8_gemm_blockscale_sm120(
        &self,
        output: u64,
        a: u64,
        b: u64,
        a_scale_sfa: u64,
        b_scale_sfb: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<()> {
        match self {
            CutlassBackend::SoSm120(lib) => lib.launch_fp8_gemm_blockscale(
                output,
                a,
                b,
                a_scale_sfa,
                b_scale_sfb,
                m,
                n,
                k,
                workspace,
                workspace_size,
                stream,
            ),
            CutlassBackend::So(_) => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm_blockscale_sm120 — SM90 .so has no Blackwell blockwise kernel",
                },
                CutlassCtx {
                    kernel: "fp8_gemm_blockscale_sm120",
                    stream,
                },
            )),
            CutlassBackend::Absent => Err(RvllmError::cutlass(
                CutlassError::FeatureNotAvailable {
                    op: "fp8_gemm_blockscale_sm120 (CutlassBackend::Absent — libcutlass_sm120.so not built)",
                },
                CutlassCtx {
                    kernel: "fp8_gemm_blockscale_sm120",
                    stream,
                },
            )),
        }
    }

    /// Workspace size for the SM120 blockwise kernel. Returns `0` for
    /// non-`SoSm120` variants so the caller can uniformly allocate
    /// `max(workspace_size(...), other_requirements)`.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn fp8_gemm_blockscale_sm120_workspace(&self, m: i32, n: i32, k: i32) -> usize {
        match self {
            CutlassBackend::SoSm120(lib) => lib.workspace_size(m, n, k),
            _ => 0,
        }
    }
}

impl From<CutlassLib> for CutlassBackend {
    fn from(lib: CutlassLib) -> Self {
        CutlassBackend::So(lib)
    }
}

/// Map a policy `VariantId` to the C-linkage symbol names in
/// `libcutlass_kernels.so`. id <100 uses the `cutlass_fp8_gemm_v{id}`
/// autotune suite; id >=100 uses `cutlass_fp8_gemm_residual_v{id-100}`.
/// Returns a heap-allocated pair (empty when out-of-range).
fn variant_symbol_names(vid: VariantId) -> (String, String) {
    if vid.0 >= 100 {
        let i = vid.0 - 100;
        (
            format!("cutlass_fp8_gemm_residual_v{i}"),
            format!("cutlass_fp8_gemm_residual_v{i}_workspace_size"),
        )
    } else {
        let i = vid.0;
        (
            format!("cutlass_fp8_gemm_v{i}"),
            format!("cutlass_fp8_gemm_v{i}_workspace_size"),
        )
    }
}

fn cutlass_miss(path: &std::path::Path) -> RvllmError {
    RvllmError::cutlass(
        CutlassError::AutotuneCacheMiss {
            m: 0,
            n: 0,
            k: 0,
            dtype: rvllm_core::DType::Fp8E4M3,
        },
        CutlassCtx {
            kernel: "libcutlass_kernels.so",
            stream: 0,
        },
    )
    // note: the actual error classifies as SoMissing; we overload
    // AutotuneCacheMiss here until the core error enum adds CutlassSoMissing.
    .into_cutlass_so_missing(path.to_path_buf())
}

fn so_integrity_error(path: &std::path::Path, detail: String) -> RvllmError {
    RvllmError::Loader {
        err: rvllm_core::LoaderError::Corrupt { detail },
        ctx: rvllm_core::LoaderCtx {
            path: path.to_path_buf(),
            tensor: None,
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

fn variant_missing(path: &std::path::Path, vid: VariantId, kind: &'static str) -> RvllmError {
    RvllmError::cutlass(
        CutlassError::KernelLaunchFailed {
            variant: vid.0,
            cuda: rvllm_core::CudaErrorKind::ModuleLoadFailed,
        },
        CutlassCtx {
            kernel: kind,
            stream: 0,
        },
    )
    .into_cutlass_variant_missing(path.to_path_buf(), vid)
}

// Small extension to chain on an existing error. Avoids adding new
// variants to rvllm_core::RvllmError for this one case.
trait CutlassErrExt {
    fn into_cutlass_so_missing(self, path: PathBuf) -> RvllmError;
    fn into_cutlass_variant_missing(self, path: PathBuf, vid: VariantId) -> RvllmError;
}

impl CutlassErrExt for RvllmError {
    fn into_cutlass_so_missing(self, path: PathBuf) -> RvllmError {
        // Repackage with a loader-style path context.
        RvllmError::Loader {
            err: rvllm_core::LoaderError::Corrupt {
                detail: format!("libcutlass_kernels.so not at {}", path.display()),
            },
            ctx: rvllm_core::LoaderCtx { path, tensor: None },
            bt: std::backtrace::Backtrace::capture(),
        }
    }
    fn into_cutlass_variant_missing(self, path: PathBuf, vid: VariantId) -> RvllmError {
        RvllmError::Loader {
            err: rvllm_core::LoaderError::Corrupt {
                detail: format!(
                    "libcutlass_kernels.so at {} missing variant {}",
                    path.display(),
                    vid.0,
                ),
            },
            ctx: rvllm_core::LoaderCtx { path, tensor: None },
            bt: std::backtrace::Backtrace::capture(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_so_rejected() {
        let err = CutlassLib::load("/nonexistent/libcutlass.so".into(), &[]).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("libcutlass_kernels.so not at"));
    }
}

/// Process-wide default for the optional cuBLASLt FP8 small-M route.
/// Sm100 disables the route unless the operator explicitly enables it.
static LT_FP8_DEFAULT_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn set_lt_fp8_default_off(off: bool) {
    LT_FP8_DEFAULT_OFF.store(off, std::sync::atomic::Ordering::SeqCst);
}

#[must_use]
pub fn lt_fp8_default_off() -> bool {
    LT_FP8_DEFAULT_OFF.load(std::sync::atomic::Ordering::SeqCst)
}

#[cfg(test)]
mod lt_fp8_flag_tests {
    #[test]
    fn defaults_on() {
        // Default state: reroute allowed (H100 behaviour unchanged).
        assert!(!super::lt_fp8_default_off());
    }
}
