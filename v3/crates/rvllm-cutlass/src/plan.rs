//! `Fp8GemmPlan`: resolved variant + shape + workspace budget.
//!
//! Constructed from a `Policy` + a concrete (M, N, K, dtype). Holds the
//! workspace bytes so the allocator can size a single slab from the max
//! across every plan the runtime will execute.

use rvllm_core::{CutlassCtx, CutlassError, DType, Result, RvllmError};

use crate::policy::Policy;
use crate::variants::VariantId;

#[derive(Clone, Debug)]
pub struct Fp8GemmPlan {
    pub variant: VariantId,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub dtype: DType,
    pub workspace_bytes: u64,
}

impl Fp8GemmPlan {
    pub fn from_policy(policy: &Policy, m: u32, n: u32, k: u32, dtype: DType) -> Result<Self> {
        policy.validate()?;
        let entry = policy.lookup(m as usize, n as usize, k as usize, dtype)?;
        Self::validated(entry.variant, entry.workspace_bytes, m, n, k, dtype)
    }

    /// Plan for a residual-epilogue GEMM. Uses `Policy::lookup_residual`
    /// so (M, N, K)-collisions between a base and residual GEMM don't
    /// overwrite each other.
    pub fn from_policy_residual(
        policy: &Policy,
        m: u32,
        n: u32,
        k: u32,
        dtype: DType,
    ) -> Result<Self> {
        policy.validate()?;
        let entry = policy.lookup_residual(m as usize, n as usize, k as usize, dtype)?;
        Self::validated(entry.variant, entry.workspace_bytes, m, n, k, dtype)
    }

    fn validated(
        variant: VariantId,
        workspace_bytes: u64,
        m: u32,
        n: u32,
        k: u32,
        dtype: DType,
    ) -> Result<Self> {
        if m == 0
            || n == 0
            || k == 0
            || m > i32::MAX as u32
            || n > i32::MAX as u32
            || k > i32::MAX as u32
            || dtype != DType::Fp8E4M3
            || usize::try_from(workspace_bytes).is_err()
        {
            return Err(RvllmError::cutlass(
                CutlassError::AutotuneCacheMiss {
                    m: m as usize,
                    n: n as usize,
                    k: k as usize,
                    dtype,
                },
                CutlassCtx {
                    kernel: "Fp8GemmPlan::validated",
                    stream: 0,
                },
            ));
        }
        Ok(Self {
            variant,
            m,
            n,
            k,
            dtype,
            workspace_bytes,
        })
    }

    /// Raise an error if `available` workspace is insufficient for the
    /// plan. Called by `Cutlass::run` before the launch.
    pub fn check_workspace(&self, available: usize) -> Result<()> {
        let needed = usize::try_from(self.workspace_bytes).map_err(|_| {
            RvllmError::cutlass(
                CutlassError::WorkspaceTooSmall {
                    variant: self.variant.0,
                    m: self.m as usize,
                    n: self.n as usize,
                    k: self.k as usize,
                    needed: usize::MAX,
                    given: available,
                },
                CutlassCtx {
                    kernel: "Fp8GemmPlan::check_workspace",
                    stream: 0,
                },
            )
        })?;
        if available < needed {
            return Err(RvllmError::cutlass(
                CutlassError::WorkspaceTooSmall {
                    variant: self.variant.0,
                    m: self.m as usize,
                    n: self.n as usize,
                    k: self.k as usize,
                    needed,
                    given: available,
                },
                CutlassCtx {
                    kernel: "Fp8GemmPlan::check_workspace",
                    stream: 0,
                },
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::PolicyEntry;
    use crate::schedule::ScheduleTag;
    use crate::variants::{ClusterShape, TileShape, VariantDescriptor};
    use std::collections::BTreeMap;

    fn make_policy() -> Policy {
        let mut entries = BTreeMap::new();
        entries.insert(
            Policy::entry_key(32, 152064, 3584, DType::Fp8E4M3),
            PolicyEntry {
                variant: VariantId(0),
                workspace_bytes: 2048,
            },
        );
        Policy {
            revision: "0123456".into(),
            arch: "sm_90".into(),
            variants: vec![VariantDescriptor {
                id: VariantId(0),
                tile: TileShape::new(128, 128, 128),
                cluster: ClusterShape::one(),
                mainloop: ScheduleTag::Coop,
                epilogue: ScheduleTag::Coop,
            }],
            entries,
        }
    }

    #[test]
    fn plan_from_policy_carries_workspace() {
        let p = make_policy();
        let plan = Fp8GemmPlan::from_policy(&p, 32, 152064, 3584, DType::Fp8E4M3).unwrap();
        assert_eq!(plan.variant, VariantId(0));
        assert_eq!(plan.workspace_bytes, 2048);
        assert!(plan.check_workspace(2048).is_ok());
        let err = plan.check_workspace(2047).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("WorkspaceTooSmall"));
    }

    #[test]
    fn missing_policy_entry_does_not_default() {
        let p = make_policy();
        let err = Fp8GemmPlan::from_policy(&p, 31, 152064, 3584, DType::Fp8E4M3).unwrap_err();
        assert!(format!("{err}").contains("AutotuneCacheMiss"));
    }
}
