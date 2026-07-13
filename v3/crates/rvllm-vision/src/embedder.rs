// Ported from huggingface/transformers revision
// 10555512868d663ee1ff627e4f5c5c260114235b:
// src/transformers/models/gemma4/modular_gemma4.py and
// src/transformers/models/gemma3n/modeling_gemma3n.py.
// Apache-2.0 License, Copyright (c) HuggingFace and Google.
// Source classes: Gemma4MultimodalEmbedder (gemma4) and its parent
// Gemma3nMultimodalEmbedder (gemma3n). Modifications: vision-only path,
// audio-related attributes dropped, candle 0.10 port.

//! Gemma 4 vision-to-text projector: scale-less f32 RMS normalization followed
//! by a bias-free linear projection. HF applies no text-embedding sqrt scale.

use candle_core::{DType, Tensor, D};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

use crate::config::Gemma4VisionConfig;

/// Vision-encoder-output → text-decoder-embedding-space projector.
pub struct Gemma4MultimodalEmbedder {
    /// Vision (or audio) tower hidden size. For Gemma 4 vision this is
    /// `vision_config.hidden_size` (1152 for the released checkpoints).
    multimodal_hidden_size: usize,
    /// Text decoder hidden size. Output rows have this width.
    text_hidden_size: usize,
    /// RMS-norm epsilon, taken from the vision config's `rms_norm_eps`.
    eps: f32,
    /// `Linear(multimodal_hidden_size -> text_hidden_size, bias=False)`.
    embedding_projection: Linear,
}

impl Gemma4MultimodalEmbedder {
    /// Construct the embedder. `vb` should be scoped at the embedder's
    /// own weight prefix (e.g. `vb.pp("model.embed_vision")`); the only
    /// learned weight loaded is `embedding_projection.weight` of shape
    /// `[text_hidden_size, multimodal_hidden_size]`.
    pub fn new(
        vision_cfg: &Gemma4VisionConfig,
        text_hidden_size: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let multimodal_hidden_size = vision_cfg.hidden_size;
        if multimodal_hidden_size == 0 || text_hidden_size == 0 || text_hidden_size > 1_048_576 {
            candle_core::bail!(
                "invalid embedder dimensions vision={multimodal_hidden_size} text={text_hidden_size}"
            );
        }
        if !vision_cfg.rms_norm_eps.is_finite() || vision_cfg.rms_norm_eps <= 0.0 {
            candle_core::bail!("embedder rms_norm_eps must be finite and > 0");
        }
        let embedding_projection = linear_no_bias(
            multimodal_hidden_size,
            text_hidden_size,
            vb.pp("embedding_projection"),
        )?;
        Ok(Self {
            multimodal_hidden_size,
            text_hidden_size,
            eps: vision_cfg.rms_norm_eps,
            embedding_projection,
        })
    }
    /// Apply the scale-less RMS norm followed by the no-bias linear
    /// projection.
    ///
    /// Input shape:  `[output_length, multimodal_hidden_size]`
    /// Output shape: `[output_length, text_hidden_size]`
    pub fn forward(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        if hidden.rank() != 2 {
            candle_core::bail!(
                "Gemma4MultimodalEmbedder::forward expects rank 2, got {:?}",
                hidden.dims()
            );
        }
        if !matches!(hidden.dtype(), DType::F16 | DType::BF16 | DType::F32) {
            candle_core::bail!(
                "Gemma4MultimodalEmbedder::forward unsupported dtype {:?}",
                hidden.dtype()
            );
        }
        let last = hidden.dim(D::Minus1)?;
        if last != self.multimodal_hidden_size {
            candle_core::bail!(
                "Gemma4MultimodalEmbedder::forward: last dim {last} != multimodal_hidden_size {}",
                self.multimodal_hidden_size
            );
        }
        let rows = hidden.dim(0)?;
        let normed = rms_norm_no_scale(hidden, self.eps)?;
        let output = self.embedding_projection.forward(&normed)?;
        if output.dims() != [rows, self.text_hidden_size] {
            candle_core::bail!(
                "embedder output shape {:?} != [{rows}, {}]",
                output.dims(),
                self.text_hidden_size
            );
        }
        Ok(output)
    }
    #[inline]
    pub fn text_hidden_size(&self) -> usize {
        self.text_hidden_size
    }
    #[inline]
    pub fn multimodal_hidden_size(&self) -> usize {
        self.multimodal_hidden_size
    }
}

/// `Gemma3nRMSNorm` with `with_scale=False`: cast to f32, normalize by
/// `x * (mean(x^2) + eps)^(-0.5)` along the last dim, cast back to the
/// original dtype. Matches the Python `_norm` exactly (uses pow(-0.5)
/// rather than rsqrt, as the HF source notes for JAX/torch parity).
fn rms_norm_no_scale(x: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let mean_sq = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let scale = (mean_sq + eps as f64)?.powf(-0.5)?;
    let normed = x_f32.broadcast_mul(&scale)?;
    normed.to_dtype(orig_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarBuilder;
    fn fake_vision_cfg() -> Gemma4VisionConfig {
        // Minimal subset that satisfies serde; values mirror the public
        // Gemma 4 vision config (hidden=1152, rms_norm_eps=1e-6).
        let raw = r#"{
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_hidden_layers": 27,
            "head_dim": 72,
            "patch_size": 16,
            "position_embedding_size": 10240,
            "pooling_kernel_size": 3,
            "max_position_embeddings": 131072,
            "standardize": true
        }"#;
        serde_json::from_str(raw).unwrap()
    }
    #[test]
    fn forward_shape_and_finite() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vision_cfg = fake_vision_cfg();
        let text_hidden_size = 3584usize; // a plausible Gemma 4 text hidden size
                                          // Zero-initialized weights. With `embedding_projection.weight = 0`
                                          // the linear output is the zero tensor, which is finite and has
                                          // the expected shape; shape correctness does not depend on values.
        let vb = VarBuilder::zeros(dtype, &device);
        let embedder = Gemma4MultimodalEmbedder::new(&vision_cfg, text_hidden_size, vb)?;
        let output_length = 280usize;
        let hidden = Tensor::randn(0f32, 1f32, (output_length, vision_cfg.hidden_size), &device)?;
        let out = embedder.forward(&hidden)?;
        assert_eq!(out.dims(), &[output_length, text_hidden_size]);
        // No NaN check.
        let flat = out.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "embedder produced non-finite values"
        );
        Ok(())
    }
    #[test]
    fn rms_norm_matches_reference_on_simple_input() -> candle_core::Result<()> {
        let device = Device::Cpu;
        // x = [1, 2, 3, 4]; mean(x^2)=7.5; (7.5+eps)^-0.5 ~= 0.36514837
        // normed ~= [0.36515, 0.73030, 1.09545, 1.46059]
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 4), &device)?;
        let normed = rms_norm_no_scale(&x, 1e-6)?;
        let got = normed.flatten_all()?.to_vec1::<f32>()?;
        let scale = (7.5f32 + 1e-6).powf(-0.5);
        let expected = [1.0 * scale, 2.0 * scale, 3.0 * scale, 4.0 * scale];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5, "rms_norm mismatch: got {g}, want {e}");
        }
        Ok(())
    }
    #[test]
    fn rejects_wrong_last_dim() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vision_cfg = fake_vision_cfg();
        let vb = VarBuilder::zeros(dtype, &device);
        let embedder = Gemma4MultimodalEmbedder::new(&vision_cfg, 3584, vb)?;
        let bad = Tensor::zeros((280, 999), DType::F32, &device)?;
        assert!(embedder.forward(&bad).is_err());
        Ok(())
    }
}
