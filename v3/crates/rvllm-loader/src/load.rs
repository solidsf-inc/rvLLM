//! Canonical CUDA loader exports.
//!
//! The implementation lives in `load_multiformat` so BF16/F16 and
//! pre-quantized FP8 checkpoints share one checked parsing path.

pub use crate::load_multiformat::{load_model, LayerAttnType, MlpActivation, ModelArch};
