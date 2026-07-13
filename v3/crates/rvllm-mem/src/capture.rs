//! `CaptureScope`: the only handle that can bind tensors into a captured
//! CUDA graph.
//!
//! The bind API accepts only `GraphSafe` handles. The capture closure remains
//! unsafe because Rust closures can capture allocators and other side effects;
//! callers must not allocate, free, or mutate pointer ownership during capture.
//!
//! `record` runs the caller's closure under a real CUDA capture and destroys
//! the validation graph after capture ends. Replayable graphs are owned by
//! `rvllm-graph`; this helper exists to enforce scoped resource binding.

use core::marker::PhantomData;

use rvllm_core::Result;

use crate::graph_safe::GraphSafe;
use crate::stream::Stream;

/// Token tying the lifetime of bound handles to the scope.
pub struct BoundHandle<'g> {
    device_ptr: u64,
    _scope: PhantomData<&'g ()>,
}

impl<'g> BoundHandle<'g> {
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }
}

/// A graph capture scope. Created by `record(stream, |scope| ...)`.
pub struct CaptureScope<'g, 's> {
    stream: &'s Stream,
    _scope: PhantomData<&'g ()>,
}

impl<'g, 's> CaptureScope<'g, 's> {
    /// Bind a `GraphSafe` value by shared reference. Returns a handle
    /// whose lifetime is tied to the scope, so the value outlives the
    /// capture.
    ///
    /// Trait bound ensures callers cannot pass `&mut HbmArena` or any
    /// realloc-capable wrapper.
    pub fn bind<T>(&mut self, value: &'g T) -> BoundHandle<'g>
    where
        T: GraphSafe + HasDevicePtr,
    {
        BoundHandle {
            device_ptr: value.device_ptr(),
            _scope: PhantomData,
        }
    }

    pub fn stream(&self) -> &'s Stream {
        self.stream
    }
}

/// Types that expose a device pointer for graph binding. Implemented by
/// `Region`, `Tensor`, and any other `GraphSafe` value that backs a
/// kernel argument.
pub trait HasDevicePtr {
    fn device_ptr(&self) -> u64;
}

impl<'a> HasDevicePtr for crate::hbm::Region<'a> {
    fn device_ptr(&self) -> u64 {
        crate::hbm::Region::device_ptr(self)
    }
}
impl<'a, T: Copy + 'static> HasDevicePtr for crate::tensor::Tensor<'a, T> {
    fn device_ptr(&self) -> u64 {
        crate::tensor::Tensor::device_ptr(self)
    }
}

/// Run `body` under graph capture on `stream`.
///
/// The scope lifetime matches the stream borrow; bound values need only
/// outlive the stream, which by construction outlives the scope.
///
/// # Safety
/// `body` must issue only stream-capture-safe operations. It must not allocate,
/// free, restore an arena checkpoint, or otherwise invalidate any device
/// pointer used by the capture. Every captured pointer must remain valid for
/// the lifetime required by the resulting graph owner.
#[cfg(feature = "cuda")]
pub unsafe fn record<'s, F, R>(stream: &'s Stream, body: F) -> Result<R>
where
    F: FnOnce(&mut CaptureScope<'s, 's>) -> Result<R>,
{
    let _context_guard = {
        use rvllm_core::{CudaCtx, CudaErrorKind, RvllmError};
        let context = stream.context().ok_or_else(|| {
            RvllmError::cuda(
                "capture::record (missing context)",
                CudaErrorKind::GraphFailed,
                CudaCtx::setup(),
            )
        })?;
        context.make_current()?
    };

    use cudarc::driver::sys::*;
    if stream.raw() == 0 {
        return Err(rvllm_core::RvllmError::cuda(
            "capture::record (null stream)",
            rvllm_core::CudaErrorKind::GraphFailed,
            rvllm_core::CudaCtx::setup(),
        ));
    }
    let status = unsafe {
        cuStreamBeginCapture_v2(
            stream.raw() as CUstream,
            CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
    };
    if status != CUresult::CUDA_SUCCESS {
        return Err(capture_error(stream.raw(), "cuStreamBeginCapture_v2"));
    }

    let mut scope = CaptureScope {
        stream,
        _scope: PhantomData,
    };
    let body_result = body(&mut scope);

    let mut graph: CUgraph = core::ptr::null_mut();
    let end_status = unsafe { cuStreamEndCapture(stream.raw() as CUstream, &mut graph) };
    let destroy_status = if graph.is_null() {
        CUresult::CUDA_SUCCESS
    } else {
        unsafe { cuGraphDestroy(graph) }
    };
    if end_status != CUresult::CUDA_SUCCESS {
        return body_result.and(Err(capture_error(stream.raw(), "cuStreamEndCapture")));
    }
    if destroy_status != CUresult::CUDA_SUCCESS {
        return body_result.and(Err(capture_error(stream.raw(), "cuGraphDestroy")));
    }
    body_result
}

#[cfg(not(feature = "cuda"))]
/// Host-only stub for [`record`].
///
/// # Safety
/// The same capture-safety contract as the CUDA implementation applies; this
/// stub always returns an error before invoking `body`.
pub unsafe fn record<'s, F, R>(_stream: &'s Stream, _body: F) -> Result<R>
where
    F: FnOnce(&mut CaptureScope<'s, 's>) -> Result<R>,
{
    Err(rvllm_core::RvllmError::cuda(
        "capture::record (CUDA unavailable)",
        rvllm_core::CudaErrorKind::GraphFailed,
        rvllm_core::CudaCtx::setup(),
    ))
}

#[cfg(feature = "cuda")]
fn capture_error(stream: u64, op: &'static str) -> rvllm_core::RvllmError {
    rvllm_core::RvllmError::cuda(
        op,
        rvllm_core::CudaErrorKind::GraphFailed,
        rvllm_core::CudaCtx {
            stream,
            kernel: op,
            launch: None,
            device: -1,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn capture_fails_without_cuda() {
        let s = Stream::host_stub();
        assert!(unsafe { record(&s, |_scope| Ok(())) }.is_err());
    }
}
