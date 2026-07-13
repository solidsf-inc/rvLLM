//! rvllm-runtime: Engine + scheduler + layer_exec per specs 07, 09.
//!
//! The public API surface for v3 callers:
//! - `Engine::new()` → init
//! - `engine.step_launch()` → returns `PendingStep<'_>`
//! - `engine.step_collect(ticket)` → waits DtoH, returns per-request
//!   outputs
//!
//! One codepath. No sync vs pipelined duality. Graph replay is a
//! transparent implementation detail.

pub mod bring_up;
pub mod engine;
pub mod gemma4_bring_up;
pub mod gemma4_int4;
pub mod gemma4_layer_exec;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub mod gemma4_metal;
#[cfg(feature = "cuda")]
pub mod gemma4_serve;
pub mod layer_exec;
pub mod sched_state;
pub mod scheduler;

pub use bring_up::{Bringup, EnginePaths, FusedModules, PplResult};
pub use engine::{Engine, PendingStep, StepExecutor, StepOutput, StepTicket};
pub use layer_exec::{forward, LayerDims};
pub use sched_state::{ReqState, Request};
pub use scheduler::{bucket_for, BatchPlan, Scheduler, DECODE_BUCKETS};
