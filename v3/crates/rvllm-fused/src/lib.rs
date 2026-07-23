//! Checked launch descriptors and CPU references for fused CUDA kernels.
//!
//! Typed launchers validate their documented shape and alignment constraints.
//! `launch_raw` remains available for kernels whose full contract is enforced
//! by the higher-level runtime wrapper.

pub mod fp8;
pub mod gemma4_launcher;
pub mod gemma4_reference;
pub mod launch_raw;
pub mod launcher;
pub mod reference;

pub use gemma4_launcher::{FusedRopePartialF16KvLaunch, PleGateLaunch, PleProjectionCombineLaunch};
pub use gemma4_reference::{
    ple_gate_ref, ple_gather_scale_ref, ple_model_projection_combine_ref, rmsnorm_gamma_ref,
};
pub use launch_raw::launch_raw;
pub use launcher::{
    require_multiple, AddBiasF16Launch, ArgmaxLaunch, EmbeddingGatherLaunch,
    FusedAddRmsnormFp8QuantLaunch, FusedRmsnormFp8QuantLaunch, FusedRopeCacheFp8KvLaunch,
    FusedRopeKvWriteLaunch, FusedSiluMulFp8QuantLaunch, MapTokenIdLaunch,
    QuantizeFp8PerTokenLaunch, ResidualAddF16Launch,
};
pub use reference::{
    argmax_ref, embedding_gather_ref, fused_add_rmsnorm_fp8_quant_ref,
    fused_gelu_mul_fp8_quant_ref, fused_silu_mul_fp8_quant_ref, quantize_fp8_per_token_ref,
    residual_add_ref, rmsnorm_ref, rope_ref, FP8_E4M3_MAX,
};
