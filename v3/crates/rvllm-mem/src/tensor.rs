//! `Tensor<'a, T>`: a typed, shaped view into a `Region`.
//!
//! `device_ptr()` is the only way to reach a raw pointer, and it borrows
//! `self`. Two tensors cannot hold `&mut` into the same region without
//! the borrow checker catching it.

use core::marker::PhantomData;

use rvllm_core::{ConfigError, DType, Result, RvllmError, Shape};

use crate::graph_safe::GraphSafe;
use crate::hbm::Region;

/// Typed view into a region. `T` is the element type; `dtype` is
/// carried separately so FP8 views can use a `u8`-sized element with a
/// distinct dtype tag.
pub struct Tensor<'a, T> {
    region: &'a Region<'a>,
    shape: Shape,
    dtype: DType,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T: Copy + 'static> Tensor<'a, T> {
    /// Build a checked tensor view over a region.
    pub fn new(region: &'a Region<'a>, shape: Shape, dtype: DType) -> Result<Self> {
        if core::mem::size_of::<T>() != dtype.bytes() {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "tensor dtype",
                    reason: format!(
                        "Rust element is {} bytes but {dtype:?} is {} bytes",
                        core::mem::size_of::<T>(),
                        dtype.bytes()
                    ),
                },
                "tensor dtype",
            ));
        }
        let needed = shape.numel()?.checked_mul(dtype.bytes()).ok_or_else(|| {
            RvllmError::config(
                ConfigError::InvalidField {
                    name: "tensor shape",
                    reason: "byte size overflow".into(),
                },
                "tensor shape",
            )
        })?;
        if needed > region.len() {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "tensor shape",
                    reason: format!(
                        "shape {shape:?} at {dtype:?} needs {needed} bytes but region '{}' has {}",
                        region.name(),
                        region.len()
                    ),
                },
                "tensor shape",
            ));
        }
        Ok(Self {
            region,
            shape,
            dtype,
            _phantom: PhantomData,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Device pointer of element `[0,0,...,0]`. Borrows `self` so the
    /// pointer cannot outlive the view.
    pub fn device_ptr(&self) -> u64 {
        self.region.device_ptr()
    }

    /// Element byte stride along each axis (row-major).
    pub fn byte_strides(&self) -> Result<[usize; rvllm_core::MAX_RANK]> {
        let mut s = self.shape.strides()?;
        let b = self.dtype.bytes();
        for x in s.iter_mut() {
            *x = x.checked_mul(b).ok_or_else(|| {
                RvllmError::config(
                    ConfigError::InvalidField {
                        name: "tensor strides",
                        reason: "byte stride overflow".into(),
                    },
                    "tensor strides",
                )
            })?;
        }
        Ok(s)
    }
}

// `Tensor<'a, T>` is GraphSafe iff its region is, which we've already
// guaranteed. Capture may bind `&Tensor`.
unsafe impl<'a, T: Copy + Send + Sync + 'static> GraphSafe for Tensor<'a, T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hbm::HbmArena;

    #[test]
    fn tensor_ptr_matches_region_ptr() {
        let arena = HbmArena::new_host_stub(1 << 20);
        let r = arena.region("t", 4096, 16).unwrap();
        let t: Tensor<half::f16> =
            Tensor::new(&r, Shape::new(&[8, 128]).unwrap(), DType::F16).unwrap();
        assert_eq!(t.device_ptr(), r.device_ptr());
        assert_eq!(t.byte_strides().unwrap()[0], 128 * 2);
        assert_eq!(t.byte_strides().unwrap()[1], 2);
    }

    #[test]
    fn oversized_shape_is_rejected() {
        let arena = HbmArena::new_host_stub(4096);
        let r = arena.region("small", 16, 1).unwrap();
        // 16 f16 = 32 B > region 16 B
        assert!(Tensor::<half::f16>::new(&r, Shape::new(&[16]).unwrap(), DType::F16).is_err());
    }
}
