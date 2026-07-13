//! CUDA context initialization.
//!
//! Called once at engine init. Under `feature = "cuda"`, drives
//! `cuInit(0)` + `cuDeviceGet` + primary-context retain. Under no-cuda
//! it's a trivial host value so the types compile.
//!
//! Uses `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent` instead of
//! legacy `cuCtxCreate_v2`. Rationale: cudarc 0.19 only cfg-wraps
//! `cuCtxCreate_v2` for CUDA toolkits 11.07..12.09, so building with
//! `feature = "cuda"` on a CUDA 13 host fails to resolve that symbol.
//! Primary-context retain is the modern API (what the CUDA runtime
//! itself uses), has no cudarc cfg gate, and is ABI-stable across
//! CUDA 11 / 12 / 13. No behavioural change for the engine — rvllm
//! uses exactly one context for the lifetime of the process anyway.

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

use std::sync::Arc;

#[derive(Debug)]
struct ContextInner {
    pub(crate) device: i32,
    #[cfg(feature = "cuda")]
    pub(crate) cu_device: cudarc::driver::sys::CUdevice,
    #[cfg(feature = "cuda")]
    pub(crate) _ctx: cudarc::driver::sys::CUcontext,
    /// Compute capability `(major, minor)`. Queried once in `init` and
    /// cached — it can't change over the handle's lifetime, and callers
    /// (manifest resolver, kernel dispatcher, bench harness) ask
    /// repeatedly.
    #[cfg(feature = "cuda")]
    pub(crate) compute_cap: (i32, i32),
    #[cfg(feature = "cuda")]
    pub(crate) launch_limits: CudaLaunchLimits,
}

#[derive(Copy, Clone, Debug)]
pub struct CudaLaunchLimits {
    pub max_threads_per_block: u32,
    pub max_block_dim: (u32, u32, u32),
    pub max_grid_dim: (u32, u32, u32),
    pub max_shared_mem_per_block: u32,
    pub max_shared_mem_per_block_optin: u32,
}

#[derive(Debug)]
pub struct CudaContextHandle {
    inner: Arc<ContextInner>,
    // CUDA current-context state is thread-local. Handles stay on their
    // creating worker thread while child resources retain the lease.
    _not_send_sync: core::marker::PhantomData<*const ()>,
}

impl Clone for CudaContextHandle {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            _not_send_sync: core::marker::PhantomData,
        }
    }
}

/// Build a typed CUDA error for a failing driver call (context init,
/// device attribute read, primary-context release, …). All call sites
/// in this module share `stream: 0` + `launch: None` because none of
/// them are on the kernel-launch path. `driver_call` is the CUDA
/// function name that failed — it flows into both `op` (rvllm's
/// operation label) and `CudaCtx.kernel` because for a driver call
/// there is no separate "kernel" identity.
fn cuda_err(driver_call: &'static str, device: i32) -> RvllmError {
    RvllmError::cuda(
        driver_call,
        CudaErrorKind::Other,
        CudaCtx {
            stream: 0,
            kernel: driver_call,
            launch: None,
            device,
        },
    )
}

/// One-shot driver call: read a device attribute into an `i32`.
/// Factored out because we query two CC attributes + could grow more
/// later. Returns the typed `RvllmError` that `init` propagates.
#[cfg(feature = "cuda")]
fn device_attr(
    cu_device: cudarc::driver::sys::CUdevice,
    attr: cudarc::driver::sys::CUdevice_attribute,
    device_ordinal: i32,
    op: &'static str,
) -> Result<i32> {
    use cudarc::driver::sys::*;
    let mut value: i32 = 0;
    if unsafe { cuDeviceGetAttribute(&mut value, attr, cu_device) } != CUresult::CUDA_SUCCESS {
        return Err(cuda_err(op, device_ordinal));
    }
    Ok(value)
}

impl CudaContextHandle {
    #[cfg(feature = "cuda")]
    pub fn init(device: i32) -> Result<Self> {
        use cudarc::driver::sys::*;
        if unsafe { cuInit(0) } != CUresult::CUDA_SUCCESS {
            return Err(cuda_err("cuInit", device));
        }
        let mut count = 0;
        if unsafe { cuDeviceGetCount(&mut count) } != CUresult::CUDA_SUCCESS {
            return Err(cuda_err("cuDeviceGetCount", device));
        }
        if device < 0 || device >= count {
            return Err(cuda_err("device ordinal out of range", device));
        }
        let mut dev: CUdevice = 0;
        if unsafe { cuDeviceGet(&mut dev, device) } != CUresult::CUDA_SUCCESS {
            return Err(cuda_err("cuDeviceGet", device));
        }

        // Read compute capability now — it's immutable for the
        // device, and downstream code (arch resolver, kernel picker)
        // asks repeatedly. One FFI round trip at init beats one per
        // call site forever.
        let cc_major = device_attr(
            dev,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
            "cuDeviceGetAttribute(CC_MAJOR)",
        )?;
        let cc_minor = device_attr(
            dev,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
            "cuDeviceGetAttribute(CC_MINOR)",
        )?;
        let positive_attr = |attr, op| -> Result<u32> {
            let value = device_attr(dev, attr, device, op)?;
            u32::try_from(value)
                .ok()
                .filter(|&value| value != 0)
                .ok_or_else(|| cuda_err(op, device))
        };
        let launch_limits = CudaLaunchLimits {
            max_threads_per_block: positive_attr(
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                "cuDeviceGetAttribute(MAX_THREADS_PER_BLOCK)",
            )?,
            max_block_dim: (
                positive_attr(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                    "cuDeviceGetAttribute(MAX_BLOCK_DIM_X)",
                )?,
                positive_attr(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                    "cuDeviceGetAttribute(MAX_BLOCK_DIM_Y)",
                )?,
                positive_attr(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                    "cuDeviceGetAttribute(MAX_BLOCK_DIM_Z)",
                )?,
            ),
            max_grid_dim: (
                positive_attr(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                    "cuDeviceGetAttribute(MAX_GRID_DIM_X)",
                )?,
                positive_attr(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                    "cuDeviceGetAttribute(MAX_GRID_DIM_Y)",
                )?,
                positive_attr(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
                    "cuDeviceGetAttribute(MAX_GRID_DIM_Z)",
                )?,
            ),
            max_shared_mem_per_block: positive_attr(
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK)",
            )?,
            max_shared_mem_per_block_optin: positive_attr(
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)",
            )?,
        };

        // Retain the primary context (ref-counted; Release in Drop) and make
        // it current on this worker thread. Child resources still guard their
        // own driver calls so teardown and temporary cross-context use restore
        // the prior current context correctly.
        let mut ctx: CUcontext = std::ptr::null_mut();
        if unsafe { cuDevicePrimaryCtxRetain(&mut ctx, dev) } != CUresult::CUDA_SUCCESS {
            return Err(cuda_err("cuDevicePrimaryCtxRetain", device));
        }
        if unsafe { cuCtxSetCurrent(ctx) } != CUresult::CUDA_SUCCESS {
            unsafe {
                let _ = cuDevicePrimaryCtxRelease_v2(dev);
            }
            return Err(cuda_err("cuCtxSetCurrent", device));
        }
        Ok(Self {
            inner: Arc::new(ContextInner {
                device,
                cu_device: dev,
                _ctx: ctx,
                compute_cap: (cc_major, cc_minor),
                launch_limits,
            }),
            _not_send_sync: core::marker::PhantomData,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn init(device: i32) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(ContextInner { device }),
            _not_send_sync: core::marker::PhantomData,
        })
    }

    pub fn host_stub() -> Self {
        Self {
            inner: Arc::new(ContextInner {
                device: -1,
                #[cfg(feature = "cuda")]
                cu_device: 0,
                #[cfg(feature = "cuda")]
                _ctx: std::ptr::null_mut(),
                #[cfg(feature = "cuda")]
                compute_cap: (0, 0),
                #[cfg(feature = "cuda")]
                launch_limits: CudaLaunchLimits {
                    max_threads_per_block: 0,
                    max_block_dim: (0, 0, 0),
                    max_grid_dim: (0, 0, 0),
                    max_shared_mem_per_block: 0,
                    max_shared_mem_per_block_optin: 0,
                },
            }),
            _not_send_sync: core::marker::PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub fn device(&self) -> i32 {
        self.inner.device
    }

    /// Device compute capability `(major, minor)`, read once at init and
    /// cached. Cheap field access — no FFI on the call path. Callers
    /// pass this pair through `CompileTarget::from_compute_capability`
    /// to pick the matching `kernels/<sm_*>/` subdirectory; a device
    /// whose compute capability has no PTX build should be rejected at
    /// bring-up (no silent fallback).
    ///
    /// Only defined under `feature = "cuda"` — every call site is
    /// already cuda-gated (no-cuda builds don't have a real device to
    /// query). A `host_stub()` under cuda returns `(0, 0)`, which
    /// `CompileTarget::from_compute_capability` maps to `None` so the
    /// bring-up path fails closed.
    #[cfg(feature = "cuda")]
    #[inline]
    #[must_use]
    pub fn compute_capability(&self) -> (i32, i32) {
        self.inner.compute_cap
    }

    #[cfg(feature = "cuda")]
    #[inline]
    #[must_use]
    pub fn launch_limits(&self) -> CudaLaunchLimits {
        self.inner.launch_limits
    }

    #[cfg(feature = "cuda")]
    pub fn make_current(&self) -> Result<CurrentContextGuard<'_>> {
        use cudarc::driver::sys::*;
        let mut previous: CUcontext = std::ptr::null_mut();
        if unsafe { cuCtxGetCurrent(&mut previous) } != CUresult::CUDA_SUCCESS {
            return Err(cuda_err("cuCtxGetCurrent", self.device()));
        }
        let changed = previous != self.inner._ctx;
        if changed && unsafe { cuCtxSetCurrent(self.inner._ctx) } != CUresult::CUDA_SUCCESS {
            return Err(cuda_err("cuCtxSetCurrent", self.device()));
        }
        Ok(CurrentContextGuard {
            owner: self,
            previous,
            changed,
        })
    }
}

#[cfg(feature = "cuda")]
pub struct CurrentContextGuard<'a> {
    owner: &'a CudaContextHandle,
    previous: cudarc::driver::sys::CUcontext,
    changed: bool,
}

#[cfg(feature = "cuda")]
impl Drop for CurrentContextGuard<'_> {
    fn drop(&mut self) {
        if self.changed {
            let status = unsafe { cudarc::driver::sys::cuCtxSetCurrent(self.previous) };
            debug_assert_eq!(status, cudarc::driver::sys::CUresult::CUDA_SUCCESS);
        }
        let _ = self.owner;
    }
}

#[cfg(feature = "cuda")]
impl Drop for ContextInner {
    fn drop(&mut self) {
        if self._ctx.is_null() {
            return;
        }
        let status = unsafe { cudarc::driver::sys::cuDevicePrimaryCtxRelease_v2(self.cu_device) };
        debug_assert_eq!(status, cudarc::driver::sys::CUresult::CUDA_SUCCESS);
    }
}
