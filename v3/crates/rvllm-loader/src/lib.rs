//! Checked safetensors loading for CUDA and Metal backends.
//!
//! The invariants:
//! - Weights are stored in typed fields, not parallel Vecs indexed by
//!   integer (v2's frequent desync source).
//! - FP8 per-tensor quant runs the clamp-% gate; a tensor exceeding
//!   10 ppm clamp rate returns `LoaderError::Fp8MisScaled` — the model
//!   is mis-scaled, not a viable FP8 candidate, and the engine refuses
//!   to proceed.
//! - CUDA loaders keep the full weight set resident before first forward.
//! - The optional Metal loader uses a bounded layer cache.

pub mod fp8_quant;
pub mod gemma4_arch;
pub mod gemma4_load;
pub mod gemma4_weights;
pub mod load;
pub mod load_multiformat;
pub mod safetensors;
pub mod weights;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub mod metal_host;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub mod metal_loader;

pub use fp8_quant::{check_clamp_gate, quantize_per_tensor_ref, QuantResult, FP8_E4M3_MAX};
pub use gemma4_arch::{Gemma4Arch, Gemma4LayerType, RopeParams, RopeType};
pub use gemma4_load::{load_gemma4_e4b_model, load_gemma4_model};
pub use gemma4_weights::{
    Bf16Weight, E4bLayerWeights, E4bLoadedModel, Gemma4LayerWeights, Gemma4LoadedModel, PleTables,
    PrunedLmHead, WPacked,
};
pub use load::{load_model, LayerAttnType, MlpActivation, ModelArch};
pub use safetensors::{ShardHeader, ShardIndex, TensorEntry};
pub use weights::{F16Weight, Fp8Weight, LayerWeights, LoadedModel};
