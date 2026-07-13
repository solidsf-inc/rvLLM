//! Pinned (page-locked) host buffer + double-buffer DtoH pool.

use core::marker::PhantomData;

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

mod sealed {
    pub trait Sealed {}
}

/// Plain device-transfer data whose every bit pattern is a valid value.
///
/// This trait is sealed so safe code cannot add a type with invalid bit
/// patterns, padding, references, or drop glue and then expose arbitrary DMA
/// bytes through `PinnedBuf::as_slice`.
///
/// # Safety
///
/// Implementations must have no padding or invalid bit patterns and must be
/// safe to overwrite byte-for-byte from a device transfer.
pub unsafe trait DevicePod: Copy + Default + Send + 'static + sealed::Sealed {}

macro_rules! impl_device_pod {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl sealed::Sealed for $ty {}
            unsafe impl DevicePod for $ty {}
        )+
    };
}

impl_device_pod!(
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    f32,
    f64,
    half::f16,
    half::bf16,
);

pub struct PinnedBuf<T: DevicePod> {
    ptr: *mut T,
    len: usize,
    owns_cuda: bool,
    _not_send_sync: PhantomData<*const ()>,
}

unsafe impl<T: DevicePod> Send for PinnedBuf<T> {}

impl<T: DevicePod> PinnedBuf<T> {
    /// Allocate via `cuMemAllocHost_v2` when cuda feature is on;
    /// otherwise a heap Box<[T]>.
    pub fn new(len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: core::ptr::null_mut(),
                len: 0,
                owns_cuda: false,
                _not_send_sync: PhantomData,
            });
        }
        if core::mem::size_of::<T>() == 0 {
            return Err(RvllmError::cuda(
                "PinnedBuf::new (zero-sized element type)",
                CudaErrorKind::AllocFailed,
                CudaCtx::setup(),
            ));
        }

        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys::*;
            let bytes = len.checked_mul(core::mem::size_of::<T>()).ok_or_else(|| {
                RvllmError::cuda(
                    "PinnedBuf::new (byte size overflow)",
                    CudaErrorKind::AllocFailed,
                    CudaCtx::setup(),
                )
            })?;
            let mut p: *mut core::ffi::c_void = core::ptr::null_mut();
            let r = unsafe { cuMemAllocHost_v2(&mut p, bytes) };
            if r != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "cuMemAllocHost_v2",
                    CudaErrorKind::AllocFailed,
                    CudaCtx {
                        stream: 0,
                        kernel: "cuMemAllocHost_v2",
                        launch: None,
                        device: -1,
                    },
                ));
            }
            // `DevicePod` guarantees both that `Default` constructs a valid
            // value and that arbitrary device bytes remain valid to read.
            for index in 0..len {
                unsafe {
                    (p as *mut T).add(index).write(T::default());
                }
            }
            Ok(Self {
                ptr: p as *mut T,
                len,
                owns_cuda: true,
                _not_send_sync: PhantomData,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            let data: Box<[T]> = vec![T::default(); len].into_boxed_slice();
            let ptr = Box::into_raw(data) as *mut T;
            Ok(Self {
                ptr,
                len,
                owns_cuda: false,
                _not_send_sync: PhantomData,
            })
        }
    }
}

impl<T: DevicePod> PinnedBuf<T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_slice(&self) -> &[T] {
        if self.ptr.is_null() {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.ptr.is_null() {
            &mut []
        } else {
            unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Queue a device-to-host copy into this pinned allocation.
    ///
    /// # Safety
    ///
    /// `src` must reference at least `count * size_of::<T>()` readable device
    /// bytes in the CUDA context associated with `stream`. Until that stream
    /// or an event recorded after this copy has completed, the caller must not
    /// read, mutate, move, or drop this buffer.
    pub unsafe fn copy_from_device_async(
        &mut self,
        src: u64,
        count: usize,
        stream: u64,
    ) -> Result<()> {
        if count > self.len {
            return Err(RvllmError::cuda(
                "PinnedBuf::copy_from_device_async (capacity)",
                CudaErrorKind::MemcpyFailed,
                CudaCtx {
                    stream,
                    kernel: "cuMemcpyDtoHAsync_v2",
                    launch: None,
                    device: -1,
                },
            ));
        }
        if count == 0 {
            return Ok(());
        }
        if src == 0 || self.ptr.is_null() {
            return Err(RvllmError::cuda(
                "PinnedBuf::copy_from_device_async (null pointer)",
                CudaErrorKind::MemcpyFailed,
                CudaCtx {
                    stream,
                    kernel: "cuMemcpyDtoHAsync_v2",
                    launch: None,
                    device: -1,
                },
            ));
        }
        let bytes = count
            .checked_mul(core::mem::size_of::<T>())
            .ok_or_else(|| {
                RvllmError::cuda(
                    "PinnedBuf::copy_from_device_async (byte size overflow)",
                    CudaErrorKind::MemcpyFailed,
                    CudaCtx {
                        stream,
                        kernel: "cuMemcpyDtoHAsync_v2",
                        launch: None,
                        device: -1,
                    },
                )
            })?;

        #[cfg(feature = "cuda")]
        unsafe {
            use cudarc::driver::sys::*;
            let status = cuMemcpyDtoHAsync_v2(
                self.ptr as *mut core::ffi::c_void,
                src,
                bytes,
                stream as CUstream,
            );
            if status != CUresult::CUDA_SUCCESS {
                return Err(RvllmError::cuda(
                    "cuMemcpyDtoHAsync_v2",
                    CudaErrorKind::MemcpyFailed,
                    CudaCtx {
                        stream,
                        kernel: "cuMemcpyDtoHAsync_v2",
                        launch: None,
                        device: -1,
                    },
                ));
            }
            Ok(())
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (bytes, stream);
            Err(RvllmError::cuda(
                "PinnedBuf::copy_from_device_async (CUDA unavailable)",
                CudaErrorKind::MemcpyFailed,
                CudaCtx::setup(),
            ))
        }
    }
}

impl<T: DevicePod> Drop for PinnedBuf<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() || self.len == 0 {
            return;
        }
        #[cfg(feature = "cuda")]
        unsafe {
            if self.owns_cuda {
                let _ = cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut core::ffi::c_void);
                return;
            }
        }
        #[cfg(not(feature = "cuda"))]
        unsafe {
            let _ = Box::<[T]>::from_raw(core::ptr::slice_from_raw_parts_mut(self.ptr, self.len));
        }
    }
}

pub struct PinnedPool<T: DevicePod> {
    buffers: [PinnedBuf<T>; 2],
    write_idx: u8,
}

impl<T: DevicePod> PinnedPool<T> {
    pub fn new(len_per_buf: usize) -> Result<Self> {
        Ok(Self {
            buffers: [PinnedBuf::new(len_per_buf)?, PinnedBuf::new(len_per_buf)?],
            write_idx: 0,
        })
    }
}

impl<T: DevicePod> PinnedPool<T> {
    pub fn write_idx(&self) -> usize {
        self.write_idx as usize
    }
    pub fn read_idx(&self) -> usize {
        1 - self.write_idx as usize
    }
    pub fn write_buf_mut(&mut self) -> &mut PinnedBuf<T> {
        &mut self.buffers[self.write_idx as usize]
    }
    pub fn read_buf(&self) -> &PinnedBuf<T> {
        &self.buffers[1 - self.write_idx as usize]
    }
    pub fn flip(&mut self) {
        self.write_idx = 1 - self.write_idx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_flips_between_buffers() {
        let mut p: PinnedPool<i32> = PinnedPool::new(128).unwrap();
        assert_eq!(p.write_idx(), 0);
        assert_eq!(p.read_idx(), 1);
        p.flip();
        assert_eq!(p.write_idx(), 1);
        assert_eq!(p.read_idx(), 0);
    }

    #[test]
    fn buf_is_zero_initialized() {
        let b: PinnedBuf<i32> = PinnedBuf::new(16).unwrap();
        assert_eq!(b.len(), 16);
        assert!(b.as_slice().iter().all(|x| *x == 0));
    }
}
