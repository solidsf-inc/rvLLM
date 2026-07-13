//! Generic `cuLaunchKernel` wrapper.
//!
//! Every fused-kernel launcher goes through `launch_raw` under feature
//! `cuda`. A no-CUDA build fails closed instead of pretending a launch ran.

use rvllm_core::{CudaCtx, CudaErrorKind, Launch, Result, RvllmError};
use rvllm_kernels::KernelFn;
use rvllm_mem::context::CudaLaunchLimits;

/// Low-level cuLaunchKernel wrapper. Args are opaque pointers to the
/// scalars/device-ptrs the kernel reads. Caller is responsible that
/// each entry in `args` points at memory that outlives this call.
///
/// # Safety
/// Caller must ensure `args` elements point at valid storage with
/// types matching the kernel's extern "C" signature.
pub unsafe fn launch_raw(
    kernel: &KernelFn,
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
    validate_geometry(grid, block, shared_mem_bytes, portable_limits).map_err(|op| {
        launch_error(
            op,
            CudaErrorKind::Other,
            kernel,
            grid,
            block,
            shared_mem_bytes,
            stream,
            -1,
        )
    })?;
    if args.iter().any(|arg| arg.is_null()) {
        return Err(launch_error(
            "kernel argument storage is null",
            CudaErrorKind::Other,
            kernel,
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
        if kernel.raw() == 0 {
            return Err(launch_error(
                "CUDA function handle is null",
                CudaErrorKind::Other,
                kernel,
                grid,
                block,
                shared_mem_bytes,
                stream,
                kernel.context().device(),
            ));
        }
        let context = kernel.context();
        validate_geometry(grid, block, shared_mem_bytes, context.launch_limits()).map_err(
            |op| {
                launch_error(
                    op,
                    CudaErrorKind::Other,
                    kernel,
                    grid,
                    block,
                    shared_mem_bytes,
                    stream,
                    context.device(),
                )
            },
        )?;
        let _guard = context.make_current()?;
        let limits = context.launch_limits();
        if shared_mem_bytes > limits.max_shared_mem_per_block {
            let requested = i32::try_from(shared_mem_bytes).map_err(|_| {
                launch_error(
                    "CUDA dynamic shared memory does not fit the driver ABI",
                    CudaErrorKind::Other,
                    kernel,
                    grid,
                    block,
                    shared_mem_bytes,
                    stream,
                    context.device(),
                )
            })?;
            let status = cuFuncSetAttribute(
                kernel.raw() as CUfunction,
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                requested,
            );
            if status != CUresult::CUDA_SUCCESS {
                return Err(launch_error(
                    "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
                    CudaErrorKind::DriverStatus(status as i32),
                    kernel,
                    grid,
                    block,
                    shared_mem_bytes,
                    stream,
                    context.device(),
                ));
            }
        }
        let status = cuLaunchKernel(
            kernel.raw() as CUfunction,
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
            return Err(launch_error(
                "cuLaunchKernel",
                CudaErrorKind::DriverStatus(status as i32),
                kernel,
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
        let _ = args;
        Err(launch_error(
            "CUDA feature is not available",
            CudaErrorKind::FeatureNotAvailable,
            kernel,
            grid,
            block,
            shared_mem_bytes,
            stream,
            -1,
        ))
    }
}

fn validate_geometry(
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

#[allow(clippy::too_many_arguments)]
fn launch_error(
    op: &'static str,
    kind: CudaErrorKind,
    kernel: &KernelFn,
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
            kernel: kernel.name(),
            launch: Some(Launch {
                grid,
                block,
                smem: shared_mem_bytes,
            }),
            device,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> CudaLaunchLimits {
        CudaLaunchLimits {
            max_threads_per_block: 1024,
            max_block_dim: (1024, 1024, 64),
            max_grid_dim: (i32::MAX as u32, 65_535, 65_535),
            max_shared_mem_per_block: 48 * 1024,
            max_shared_mem_per_block_optin: 99 * 1024,
        }
    }

    #[test]
    fn rejects_zero_and_oversized_geometry() {
        assert!(validate_geometry((0, 1, 1), (1, 1, 1), 0, limits()).is_err());
        assert!(validate_geometry((1, 1, 1), (1024, 2, 1), 0, limits()).is_err());
        assert!(validate_geometry((1, 1, 1), (256, 1, 1), 100 * 1024, limits()).is_err());
    }

    #[test]
    fn accepts_geometry_within_device_limits() {
        assert!(validate_geometry((128, 16, 1), (256, 1, 1), 1024, limits()).is_ok());
    }
}
