//! FP8 host references live in `rvllm-core` so fused kernels and loaders use
//! exactly the same encoder.

pub use rvllm_core::fp8::{f32_to_fp8_e4m3, FP8_E4M3_MAX};
