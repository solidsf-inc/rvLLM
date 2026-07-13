//! rvllm-graph: captured-graph pool per spec 14.
//!
//! Invariants:
//! - Graphs are captured at engine init for every declared bucket —
//!   NO lazy capture during warmup.
//! - Every replay is gated on a `MetadataLayout` hash check; a drifted
//!   layout (meaning the captured graph is not structurally valid for
//!   the current bucket) returns `GraphError::CaptureMetadataMismatch`
//!   instead of crashing under `cuGraphLaunch` with ILLEGAL_ADDRESS.
//! - Missing bucket is a typed error (engine-init-time); no fallback
//!   to non-graph execution.

pub mod pool;

pub use pool::{CapturedGraph, GraphFingerprint, GraphPool};
