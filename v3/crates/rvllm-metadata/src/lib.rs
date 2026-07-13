//! Packed metadata layout and upload API.
//!
//! The enforced invariants are:
//! - `MetadataLayout` is keyed on `(bucket, max_blocks_per_seq)`,
//!   and contains checked offsets for the packed buffer.
//! - `upload()` validates and writes that layout.
//! - `MetadataLayout::hash()` gives a sha256 that `rvllm-graph` stores
//!   at capture and compares before replay.

pub mod layout;
pub mod pack;
pub mod plan;

pub use layout::MetadataLayout;
pub use pack::upload;
pub use plan::BatchPlan;
