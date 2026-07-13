// Ported from huggingface/transformers revision
// 10555512868d663ee1ff627e4f5c5c260114235b:
// src/transformers/models/gemma4/modular_gemma4.py
// Apache-2.0 License, Copyright (c) HuggingFace and Google.
// Source class: Gemma4VisionModel (public API surface). Modifications: rvLLM
// wrapper context; the encoder, embedder, and loader live in sibling modules.

//! `rvllm-vision`: Gemma 4 vision tower + multimodal embedder bridge.
//!
//! Provides [`VisionContext`] — load Gemma 4 vision weights into candle and
//! encode one image at a time into `[output_length x text_hidden_size]`
//! embeddings ready to inject at `<image_soft_token>` positions in the
//! text decoder's prefill stream.
//!
//! Provenance and local modifications are recorded in
//! `LICENSES/Apache-2.0-transformers` at the repository root.

pub mod chat;
pub mod config;
pub mod embedder;
pub mod loader;
pub mod model;
pub mod preprocess;

pub use config::{Gemma4Config, Gemma4VisionConfig, VisionRopeParameters};

use std::path::Path;

use anyhow::{anyhow, Context};
use candle_core::{DType, Device};

/// Top-level vision context. Owns the loaded encoder, the multimodal
/// embedder, and the parsed config.
pub struct VisionContext {
    pub device: Device,
    pub config: Gemma4Config,
    pub model: model::Gemma4VisionModel,
    pub embedder: embedder::Gemma4MultimodalEmbedder,
    pub text_hidden_size: usize,
}

impl VisionContext {
    /// Load the Gemma 4 vision tower + multimodal embedder from a
    /// safetensors weights directory.
    ///
    pub fn load(weights_dir: &Path, device: Device) -> anyhow::Result<Self> {
        let config = Gemma4Config::load_from_dir(weights_dir)?;
        config.validate()?;
        let text_hidden_size = config
            .text_config
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                anyhow!(
                    "Gemma4Config.text_config.hidden_size missing in {}/config.json",
                    weights_dir.display()
                )
            })? as usize;
        let dtype = candle_core::DType::F32;
        let (model, embedder) = loader::load_vision(weights_dir, &config, &device, dtype)?;
        Ok(Self {
            device,
            config,
            model,
            embedder,
            text_hidden_size,
        })
    }
    /// Encode a single image into Gemma 4 soft-token embeddings.
    ///
    /// Returns a row-major `(output_length x text_hidden_size)` flat buffer
    /// plus `output_length` (the number of soft tokens the encoder emitted
    /// for this image, which the chat-template wiring uses to splice the
    /// right number of `<image_soft_token>` ids into the prompt).
    pub fn encode_image(&self, image: &image::DynamicImage) -> anyhow::Result<(Vec<f32>, usize)> {
        let prepared = preprocess::prepare_image(
            image,
            self.config.vision_config.patch_size as u32,
            self.config.vision_config.pooling_kernel_size as u32,
            self.config.vision_soft_tokens_per_image,
        )
        .context("vision image preprocessing")?;
        let pixel_values = prepared
            .pixel_values
            .to_device(&self.device)
            .context("move pixel_values to vision device")?;
        let pixel_position_ids = prepared
            .pixel_position_ids
            .to_device(&self.device)
            .context("move pixel_position_ids to vision device")?;
        let features = self
            .model
            .forward(
                &pixel_values,
                &pixel_position_ids,
                self.config.vision_soft_tokens_per_image as usize,
            )
            .context("Gemma4VisionModel::forward")?;
        let soft = self
            .embedder
            .forward(&features)
            .context("Gemma4MultimodalEmbedder::forward")?;
        let dims = soft.dims();
        if dims.len() != 2 || dims[1] != self.text_hidden_size {
            anyhow::bail!(
                "vision soft-token shape {:?} does not match [N, text_hidden_size={}]",
                dims,
                self.text_hidden_size
            );
        }
        let output_length = dims[0];
        if output_length == 0 || output_length > self.config.vision_soft_tokens_per_image as usize {
            anyhow::bail!(
                "vision output length {output_length} is outside 1..={}",
                self.config.vision_soft_tokens_per_image
            );
        }
        let flat = soft
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
            .context("copy vision soft tokens to host")?;
        if flat.len() != output_length * self.text_hidden_size
            || flat.iter().any(|value| !value.is_finite())
        {
            anyhow::bail!("vision encoder produced invalid host output");
        }
        Ok((flat, output_length))
    }
    #[inline]
    pub fn image_token_id(&self) -> u32 {
        self.config.image_token_id
    }
    #[inline]
    pub fn boi_token_id(&self) -> u32 {
        self.config.boi_token_id
    }
    #[inline]
    pub fn eoi_token_id(&self) -> u32 {
        self.config.eoi_token_id
    }
    #[inline]
    pub fn vision_soft_tokens_per_image(&self) -> usize {
        self.config.vision_soft_tokens_per_image as usize
    }
}
