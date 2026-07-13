//! GPU compile-target enumeration.
//!
//! Every kernel artifact is pinned to one compute capability.
//! The runtime queries the device's compute capability via
//! `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR})`,
//! maps it to a `CompileTarget`, and selects the matching kernel directory
//! under `RVLLM_KERNEL_DIR/<arch>/*.ptx`. A device whose compute capability
//! is not listed here is rejected at `bring_up` time (no silent fallback).
//!
//! New targets require a matching kernel tree and runtime validation.

use serde::{Deserialize, Serialize};

/// Supported GPU compile targets.
///
/// Add a new variant when we need to support a new architecture; update
/// `from_compute_capability`, `as_sm_str`, and the build system's kernel
/// output directories in the same PR.
///
/// `#[non_exhaustive]` so future arches (sm_100, sm_120, sm_122, …) can
/// join without breaking downstream `match` expressions. Internal
/// matches inside this crate stay exhaustive by design.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[non_exhaustive]
#[must_use]
pub enum CompileTarget {
    /// NVIDIA compute capability 8.0.
    Sm80,
    /// NVIDIA compute capability 8.9.
    Sm89,
    /// NVIDIA compute capability 9.0.
    Sm90,
    /// NVIDIA compute capability 10.0.
    Sm100,
    /// NVIDIA compute capability 12.1.
    Sm121,
}

impl CompileTarget {
    /// Map a compute-capability tuple to a compile target.
    ///
    /// Returns `None` for compute capabilities we do not yet build PTX for.
    /// The caller is responsible for turning that `None` into a hard error
    /// (the runtime refuses to boot on an unsupported device rather than
    /// falling back to a generic path).
    #[inline]
    #[must_use]
    pub const fn from_compute_capability(major: i32, minor: i32) -> Option<Self> {
        match (major, minor) {
            (8, 0) => Some(CompileTarget::Sm80),
            (8, 9) => Some(CompileTarget::Sm89),
            (9, 0) => Some(CompileTarget::Sm90),
            (10, 0) => Some(CompileTarget::Sm100),
            (12, 1) => Some(CompileTarget::Sm121),
            _ => None,
        }
    }

    /// The `sm_XYZ` string as accepted by `nvcc -arch=` and used as the
    /// kernel subdirectory name (e.g. `kernels/sm_121/fp8_gemv.ptx`).
    #[inline]
    #[must_use]
    pub const fn as_sm_str(self) -> &'static str {
        match self {
            CompileTarget::Sm80 => "sm_80",
            CompileTarget::Sm89 => "sm_89",
            CompileTarget::Sm90 => "sm_90",
            CompileTarget::Sm100 => "sm_100",
            CompileTarget::Sm121 => "sm_121",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sm_str_matches_nvcc_flag() {
        assert_eq!(CompileTarget::Sm80.as_sm_str(), "sm_80");
        assert_eq!(CompileTarget::Sm89.as_sm_str(), "sm_89");
        assert_eq!(CompileTarget::Sm90.as_sm_str(), "sm_90");
        assert_eq!(CompileTarget::Sm100.as_sm_str(), "sm_100");
        assert_eq!(CompileTarget::Sm121.as_sm_str(), "sm_121");
    }

    #[test]
    fn compute_cap_to_target() {
        assert_eq!(
            CompileTarget::from_compute_capability(9, 0),
            Some(CompileTarget::Sm90),
        );
        assert_eq!(
            CompileTarget::from_compute_capability(10, 0),
            Some(CompileTarget::Sm100),
        );
        assert_eq!(
            CompileTarget::from_compute_capability(12, 1),
            Some(CompileTarget::Sm121),
        );
        assert_eq!(
            CompileTarget::from_compute_capability(8, 0),
            Some(CompileTarget::Sm80),
        );
    }

    #[test]
    fn unknown_cc_returns_none() {
        // No kernel tree is shipped for these capabilities.
        assert_eq!(CompileTarget::from_compute_capability(12, 0), None);
        assert_eq!(CompileTarget::from_compute_capability(12, 2), None);
        assert_eq!(CompileTarget::from_compute_capability(9, 5), None);
    }

    #[test]
    fn sm121_is_distinct_from_sm120_and_sm122() {
        // Distinct compute capabilities must never share artifacts implicitly.
        assert_ne!(
            CompileTarget::from_compute_capability(12, 1),
            CompileTarget::from_compute_capability(12, 0),
        );
        assert_ne!(
            CompileTarget::from_compute_capability(10, 0),
            CompileTarget::from_compute_capability(12, 1),
        );
    }
}
