//! rvllm-cutlass: variant catalog + policy loader + plan.
//!
//! The invariants this crate carries:
//! - **Schedule pairing is a type-level invariant.** `Variant<M, E>`
//!   requires `(M, E): MatchedPair`. Only matched pairs compile.
//! - **Policy lookup is fail closed.** A missing policy entry for a shape
//!   returns `CutlassError::AutotuneCacheMiss`; fallback backends are
//!   selected explicitly by the runtime rather than inside policy lookup.
//! - **Workspace is plan-owned.** `Fp8GemmPlan::workspace_bytes` is the
//!   authoritative number the allocator sizes against; if the runtime
//!   hands the kernel less, `check_workspace` returns `WorkspaceTooSmall`.

pub mod cublaslt;
pub mod lib_so;
pub mod plan;
pub mod policy;
pub mod schedule;
pub mod variants;
pub mod w4a8;

pub use cublaslt::CublasLt;
pub use lib_so::{lt_fp8_default_off, set_lt_fp8_default_off, CutlassBackend, CutlassLib};
pub use plan::Fp8GemmPlan;
pub use policy::{Policy, PolicyEntry, ShapeKey};
pub use schedule::{Coop, Fp8Coop, Fp8WS, MatchedPair, Schedule, ScheduleTag, WS};
pub use variants::{
    canonical_variants, ClusterShape, TileShape, Variant, VariantDescriptor, VariantId,
    FP8_GEMM_COOP_128_128_128, FP8_GEMM_COOP_128_256_128, FP8_GEMM_FP8COOP_128_128_128,
    FP8_GEMM_FP8WS_64_128_128, FP8_GEMM_RESIDUAL_COOP, FP8_GEMM_WS_64_128_128,
};
pub use w4a8::W4a8Lib;
