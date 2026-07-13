//! CUDA module and function ownership.

use std::sync::Arc;

use rvllm_core::{CudaCtx, CudaErrorKind, Launch, Result, RvllmError};
use rvllm_mem::{context::CudaLaunchLimits, CudaContextHandle};

struct ModuleInner {
    raw: u64,
    path: std::path::PathBuf,
    context: CudaContextHandle,
}

impl std::fmt::Debug for ModuleInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModuleInner").finish_non_exhaustive()
    }
}

/// Kernel-function handle that retains its owning CUDA module.
#[derive(Clone)]
pub struct KernelFn {
    raw: u64,
    name: &'static str,
    _owner: Arc<ModuleInner>,
}

impl std::fmt::Debug for KernelFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelFn")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl KernelFn {
    pub fn raw(&self) -> u64 {
        self.raw
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    #[cfg(feature = "cuda")]
    pub fn context(&self) -> &CudaContextHandle {
        &self._owner.context
    }

    /// Launch this function in the CUDA context that owns its module.
    ///
    /// # Safety
    /// Each `args` element must point at live host storage whose type matches
    /// the corresponding kernel parameter. Device-pointer values stored there
    /// must be valid for the launch and remain live until the stream completes.
    pub unsafe fn launch_raw(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        stream: u64,
        args: &[*mut core::ffi::c_void],
    ) -> Result<()> {
        let portable_limits = CudaLaunchLimits {
            max_threads_per_block: 1024,
            max_block_dim: (1024, 1024, 64),
            max_grid_dim: (i32::MAX as u32, 65_535, 65_535),
            max_shared_mem_per_block: u32::MAX,
            max_shared_mem_per_block_optin: u32::MAX,
        };
        validate_launch_geometry(grid, block, shared_mem_bytes, portable_limits).map_err(|op| {
            self.launch_error(
                op,
                CudaErrorKind::Other,
                grid,
                block,
                shared_mem_bytes,
                stream,
                -1,
            )
        })?;
        if args.is_empty() || args.iter().any(|arg| arg.is_null()) {
            return Err(self.launch_error(
                "invalid CUDA kernel argument storage",
                CudaErrorKind::Other,
                grid,
                block,
                shared_mem_bytes,
                stream,
                -1,
            ));
        }

        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys::*;
            if self.raw == 0 {
                return Err(self.launch_error(
                    "CUDA function handle is null",
                    CudaErrorKind::Other,
                    grid,
                    block,
                    shared_mem_bytes,
                    stream,
                    self.context().device(),
                ));
            }
            let context = self.context();
            let limits = context.launch_limits();
            validate_launch_geometry(grid, block, shared_mem_bytes, limits).map_err(|op| {
                self.launch_error(
                    op,
                    CudaErrorKind::Other,
                    grid,
                    block,
                    shared_mem_bytes,
                    stream,
                    context.device(),
                )
            })?;
            let _guard = context.make_current()?;
            if shared_mem_bytes > limits.max_shared_mem_per_block {
                let requested = i32::try_from(shared_mem_bytes).map_err(|_| {
                    self.launch_error(
                        "CUDA dynamic shared memory does not fit the driver ABI",
                        CudaErrorKind::Other,
                        grid,
                        block,
                        shared_mem_bytes,
                        stream,
                        context.device(),
                    )
                })?;
                let status = cuFuncSetAttribute(
                    self.raw as CUfunction,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    requested,
                );
                if status != CUresult::CUDA_SUCCESS {
                    return Err(self.launch_error(
                        "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
                        CudaErrorKind::DriverStatus(status as i32),
                        grid,
                        block,
                        shared_mem_bytes,
                        stream,
                        context.device(),
                    ));
                }
            }
            let status = cuLaunchKernel(
                self.raw as CUfunction,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem_bytes,
                stream as CUstream,
                args.as_ptr() as *mut *mut core::ffi::c_void,
                core::ptr::null_mut(),
            );
            if status != CUresult::CUDA_SUCCESS {
                return Err(self.launch_error(
                    "cuLaunchKernel",
                    CudaErrorKind::DriverStatus(status as i32),
                    grid,
                    block,
                    shared_mem_bytes,
                    stream,
                    context.device(),
                ));
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(self.launch_error(
                "CUDA feature is not available",
                CudaErrorKind::FeatureNotAvailable,
                grid,
                block,
                shared_mem_bytes,
                stream,
                -1,
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_error(
        &self,
        op: &'static str,
        kind: CudaErrorKind,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        stream: u64,
        device: i32,
    ) -> RvllmError {
        RvllmError::cuda(
            op,
            kind,
            CudaCtx {
                stream,
                kernel: self.name,
                launch: Some(Launch {
                    grid,
                    block,
                    smem: shared_mem_bytes,
                }),
                device,
            },
        )
    }
}

fn validate_launch_geometry(
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem_bytes: u32,
    limits: CudaLaunchLimits,
) -> core::result::Result<(), &'static str> {
    if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
        return Err("CUDA grid dimensions must be nonzero");
    }
    if block.0 == 0 || block.1 == 0 || block.2 == 0 {
        return Err("CUDA block dimensions must be nonzero");
    }
    if grid.0 > limits.max_grid_dim.0
        || grid.1 > limits.max_grid_dim.1
        || grid.2 > limits.max_grid_dim.2
    {
        return Err("CUDA grid exceeds device limits");
    }
    if block.0 > limits.max_block_dim.0
        || block.1 > limits.max_block_dim.1
        || block.2 > limits.max_block_dim.2
    {
        return Err("CUDA block exceeds device dimension limits");
    }
    let threads = u64::from(block.0)
        .checked_mul(u64::from(block.1))
        .and_then(|value| value.checked_mul(u64::from(block.2)))
        .ok_or("CUDA block thread count overflow")?;
    if threads > u64::from(limits.max_threads_per_block) {
        return Err("CUDA block exceeds device thread limit");
    }
    if shared_mem_bytes > limits.max_shared_mem_per_block_optin {
        return Err("CUDA dynamic shared memory exceeds device limit");
    }
    Ok(())
}

/// A reference-counted CUDA module. Functions cloned from it keep the module
/// and its retained primary context alive.
#[derive(Clone, Debug)]
pub struct LoadedModule {
    inner: Arc<ModuleInner>,
}

impl LoadedModule {
    /// Load the already-open, digest-verified PTX bytes. The path is retained
    /// only for diagnostics and is never reopened.
    #[cfg(feature = "cuda")]
    pub fn load_from_bytes(
        context: &CudaContextHandle,
        path: std::path::PathBuf,
        bytes: &[u8],
    ) -> Result<Self> {
        use cudarc::driver::sys::*;
        if bytes.is_empty() || bytes.len() > 512 * 1024 * 1024 {
            return Err(module_error(
                "cuModuleLoadData (invalid PTX extent)",
                CudaErrorKind::ModuleLoadFailed,
                context.device(),
            ));
        }
        let ptx = std::ffi::CString::new(bytes).map_err(|_| {
            module_error(
                "cuModuleLoadData (interior NUL)",
                CudaErrorKind::ModuleLoadFailed,
                context.device(),
            )
        })?;
        let _guard = context.make_current()?;
        let mut module: CUmodule = core::ptr::null_mut();
        let status = unsafe { cuModuleLoadData(&mut module, ptx.as_ptr().cast()) };
        if status != CUresult::CUDA_SUCCESS || module.is_null() {
            return Err(module_error(
                "cuModuleLoadData",
                CudaErrorKind::DriverStatus(status as i32),
                context.device(),
            ));
        }
        Ok(Self {
            inner: Arc::new(ModuleInner {
                raw: module as u64,
                path,
                context: context.clone(),
            }),
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load_from_bytes(
        context: &CudaContextHandle,
        _path: std::path::PathBuf,
        _bytes: &[u8],
    ) -> Result<Self> {
        Err(module_error(
            "LoadedModule::load_from_bytes (CUDA unavailable)",
            CudaErrorKind::ModuleLoadFailed,
            context.device(),
        ))
    }

    pub fn path(&self) -> &std::path::Path {
        &self.inner.path
    }

    pub fn raw(&self) -> u64 {
        self.inner.raw
    }

    /// Resolve a validated CUDA symbol and retain the module in the returned
    /// handle.
    pub fn get_function(&self, name: &'static str) -> Result<KernelFn> {
        validate_symbol_name(name)?;
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys::*;
            let cname = std::ffi::CString::new(name).map_err(|_| {
                module_error(
                    "cuModuleGetFunction (interior NUL)",
                    CudaErrorKind::ModuleLoadFailed,
                    self.inner.context.device(),
                )
            })?;
            let _guard = self.inner.context.make_current()?;
            let mut function: CUfunction = core::ptr::null_mut();
            let status = unsafe {
                cuModuleGetFunction(&mut function, self.inner.raw as CUmodule, cname.as_ptr())
            };
            if status != CUresult::CUDA_SUCCESS || function.is_null() {
                return Err(module_error(
                    "cuModuleGetFunction",
                    CudaErrorKind::DriverStatus(status as i32),
                    self.inner.context.device(),
                ));
            }
            Ok(KernelFn {
                raw: function as u64,
                name,
                _owner: Arc::clone(&self.inner),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(module_error(
                "LoadedModule::get_function (CUDA unavailable)",
                CudaErrorKind::ModuleLoadFailed,
                self.inner.context.device(),
            ))
        }
    }
}

fn validate_symbol_name(name: &'static str) -> Result<()> {
    if name.is_empty()
        || name.len() > 256
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'$'))
    {
        return Err(module_error(
            "invalid CUDA symbol name",
            CudaErrorKind::ModuleLoadFailed,
            -1,
        ));
    }
    Ok(())
}

fn module_error(op: &'static str, kind: CudaErrorKind, device: i32) -> RvllmError {
    RvllmError::cuda(
        op,
        kind,
        CudaCtx {
            stream: 0,
            kernel: op,
            launch: None,
            device,
        },
    )
}

#[cfg(feature = "cuda")]
impl Drop for ModuleInner {
    fn drop(&mut self) {
        use cudarc::driver::sys::*;
        if self.raw == 0 {
            return;
        }
        let Ok(_guard) = self.context.make_current() else {
            core::mem::forget(self.context.clone());
            return;
        };
        let sync = unsafe { cuCtxSynchronize() };
        if sync != CUresult::CUDA_SUCCESS {
            core::mem::forget(self.context.clone());
            return;
        }
        let unload = unsafe { cuModuleUnload(self.raw as CUmodule) };
        if unload != CUresult::CUDA_SUCCESS {
            core::mem::forget(self.context.clone());
            return;
        }
        self.raw = 0;
    }
}
