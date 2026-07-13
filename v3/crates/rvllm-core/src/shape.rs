//! Tensor shape with explicit row-major strides.
//!
//! Fixed-capacity; the runtime only allocates up to 4-D tensors for the
//! decode path (plus the KV cache which we model as 5-D in `rvllm-mem`).

use core::fmt;

use crate::error::{Result, RvllmError, ShapeError};

/// Maximum rank supported by the core shape. KV-cache layout bumps this
/// to 5 via a dedicated type in `rvllm-mem`.
pub const MAX_RANK: usize = 4;

/// Shape + row-major strides. Rank is stored inline; no heap alloc.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Shape {
    rank: u8,
    dims: [usize; MAX_RANK],
}

impl Shape {
    /// Build from a slice, rejecting unsupported rank and overflowing shape
    /// products before the shape can reach an allocator.
    pub fn new(dims: &[usize]) -> Result<Self> {
        if dims.len() > MAX_RANK {
            return Err(RvllmError::Shape {
                err: ShapeError::RankTooLarge {
                    rank: dims.len(),
                    max: MAX_RANK,
                },
            });
        }
        dims.iter().try_fold(1usize, |n, &dim| {
            n.checked_mul(dim).ok_or(RvllmError::Shape {
                err: ShapeError::ElementCountOverflow,
            })
        })?;
        let mut out = [0usize; MAX_RANK];
        out[..dims.len()].copy_from_slice(dims);
        Ok(Self {
            rank: dims.len() as u8,
            dims: out,
        })
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank as usize
    }

    #[inline]
    pub fn dim(&self, i: usize) -> Result<usize> {
        if i >= self.rank() {
            return Err(RvllmError::Shape {
                err: ShapeError::IndexOutOfRange {
                    index: i,
                    rank: self.rank(),
                },
            });
        }
        Ok(self.dims[i])
    }

    /// Total number of elements.
    pub fn numel(&self) -> Result<usize> {
        self.dims[..self.rank()].iter().try_fold(1usize, |n, &dim| {
            n.checked_mul(dim).ok_or(RvllmError::Shape {
                err: ShapeError::ElementCountOverflow,
            })
        })
    }

    /// Row-major strides in elements.
    pub fn strides(&self) -> Result<[usize; MAX_RANK]> {
        let mut s = [0usize; MAX_RANK];
        let r = self.rank as usize;
        if r == 0 {
            return Ok(s);
        }
        s[r - 1] = 1;
        for i in (0..r - 1).rev() {
            s[i] = s[i + 1]
                .checked_mul(self.dims[i + 1])
                .ok_or(RvllmError::Shape {
                    err: ShapeError::StrideOverflow,
                })?;
        }
        Ok(s)
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut dbg = f.debug_list();
        for i in 0..self.rank as usize {
            dbg.entry(&self.dims[i]);
        }
        dbg.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_and_strides() {
        let s = Shape::new(&[2, 3, 4]).unwrap();
        assert_eq!(s.numel().unwrap(), 24);
        assert_eq!(&s.strides().unwrap()[..3], &[12, 4, 1]);
    }

    #[test]
    fn zero_rank() {
        let s = Shape::new(&[]).unwrap();
        assert_eq!(s.rank(), 0);
        // numel of empty product is 1 (scalar)
        assert_eq!(s.numel().unwrap(), 1);
    }

    #[test]
    fn rank_overflow_is_rejected() {
        assert!(Shape::new(&[1; MAX_RANK + 1]).is_err());
    }

    #[test]
    fn dimension_index_is_checked() {
        let s = Shape::new(&[2, 3]).unwrap();
        assert_eq!(s.dim(1).unwrap(), 3);
        assert!(s.dim(2).is_err());
    }
}
