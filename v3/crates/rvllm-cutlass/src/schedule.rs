//! Mainloop/epilogue schedule pairing enforced at the type level.
//!
//! This is the Rust mirror of the CUDA `static_assert(is_same_v<Mainloop,
//! Epilogue>)` in `cutlass_fp8_gemm_residual.cu`. Adding a variant
//! whose mainloop schedule does not match its epilogue schedule is a
//! compile error here — not a crash-under-graph-replay footgun at run
//! time.
//!
//! The four supported SM90 schedules. `WS` = KernelTmaWarpSpecialized,
//! `Coop` = KernelTmaWarpSpecializedCooperative, `FP8WS` / `FP8Coop` =
//! the FP8FastAccum variants. `Pingpong` is documented but not yet
//! used for the residual-fused path because its arithmetic has shown
//! subtle mismatches with the residual EVT epilogue.

use serde::{Deserialize, Serialize};

/// Trait implemented by every CUTLASS schedule marker type.
pub trait Schedule: sealed::Sealed + 'static {
    const NAME: &'static str;
}

/// `(Mainloop, Epilogue)` pairs that CUTLASS accepts at SM90.
/// Any variant that fails this bound fails to compile.
pub trait MatchedPair: sealed::Sealed {}

/// Marker types for the schedules.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct WS;

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct Coop;

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct Fp8WS;

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct Fp8Coop;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::WS {}
    impl Sealed for super::Coop {}
    impl Sealed for super::Fp8WS {}
    impl Sealed for super::Fp8Coop {}
    // Impl for tuples: gated on which pairs we accept.
    impl Sealed for (super::WS, super::WS) {}
    impl Sealed for (super::Coop, super::Coop) {}
    impl Sealed for (super::Fp8WS, super::Fp8WS) {}
    impl Sealed for (super::Fp8Coop, super::Fp8Coop) {}
}

impl Schedule for WS {
    const NAME: &'static str = "KernelTmaWarpSpecialized";
}
impl Schedule for Coop {
    const NAME: &'static str = "KernelTmaWarpSpecializedCooperative";
}
impl Schedule for Fp8WS {
    const NAME: &'static str = "KernelTmaWarpSpecializedFP8FastAccum";
}
impl Schedule for Fp8Coop {
    const NAME: &'static str = "KernelTmaWarpSpecializedCooperativeFP8FastAccum";
}

// Only matched pairs implement MatchedPair. Unmatched pairs do NOT
// implement it, so a `Variant<WS, Coop>` would fail to construct.
impl MatchedPair for (WS, WS) {}
impl MatchedPair for (Coop, Coop) {}
impl MatchedPair for (Fp8WS, Fp8WS) {}
impl MatchedPair for (Fp8Coop, Fp8Coop) {}

/// Serializable tag used in policy.json entries.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub enum ScheduleTag {
    WS,
    Coop,
    Fp8WS,
    Fp8Coop,
}

impl ScheduleTag {
    pub const fn name(self) -> &'static str {
        match self {
            ScheduleTag::WS => WS::NAME,
            ScheduleTag::Coop => Coop::NAME,
            ScheduleTag::Fp8WS => Fp8WS::NAME,
            ScheduleTag::Fp8Coop => Fp8Coop::NAME,
        }
    }
    /// Runtime-check two tags pair. The type-level `MatchedPair`
    /// trait is the canonical gate; this is the serialized-form check
    /// used when loading policy.json.
    pub const fn matches(self, epilogue: ScheduleTag) -> bool {
        matches!(
            (self, epilogue),
            (ScheduleTag::WS, ScheduleTag::WS)
                | (ScheduleTag::Coop, ScheduleTag::Coop)
                | (ScheduleTag::Fp8WS, ScheduleTag::Fp8WS)
                | (ScheduleTag::Fp8Coop, ScheduleTag::Fp8Coop)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Accepts matched pairs only.
    fn accept_matched<M: Schedule, E: Schedule>()
    where
        (M, E): MatchedPair,
    {
    }

    #[test]
    fn matched_pairs_compile() {
        accept_matched::<WS, WS>();
        accept_matched::<Coop, Coop>();
        accept_matched::<Fp8WS, Fp8WS>();
        accept_matched::<Fp8Coop, Fp8Coop>();
    }

    // The negative case (accept_matched::<WS, Coop>()) is a trybuild
    // compile-fail test, added alongside rvllm-invariants trybuild
    // harness.

    #[test]
    fn schedule_tag_matching() {
        assert!(ScheduleTag::Coop.matches(ScheduleTag::Coop));
        assert!(!ScheduleTag::WS.matches(ScheduleTag::Coop));
        assert!(!ScheduleTag::Fp8WS.matches(ScheduleTag::Fp8Coop));
    }
}
