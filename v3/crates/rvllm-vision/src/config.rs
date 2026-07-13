// Ported from huggingface/transformers revision
// 10555512868d663ee1ff627e4f5c5c260114235b:
// src/transformers/models/gemma4/configuration_gemma4.py
// Apache-2.0 License, Copyright (c) HuggingFace and Google.
// Source class: Gemma4VisionConfig, Gemma4Config. Modifications: subset of
// fields required by the rvllm vision tower; text/audio configs left opaque
// because rvllm's hand-rolled FP8 path parses them separately.

use std::io::Read;

use anyhow::{anyhow, ensure};
use serde::{Deserialize, Serialize};

pub const MAX_CONFIG_BYTES: usize = 256 * 1024;
pub const MAX_VISION_SOFT_TOKENS: u32 = 1_120;
pub const MAX_VISION_PATCHES: usize = 10_080;
const MAX_VISION_HIDDEN_SIZE: usize = 8_192;
const MAX_VISION_INTERMEDIATE_SIZE: usize = 32_768;
const MAX_VISION_ATTENTION_HEADS: usize = 128;
const MAX_VISION_KV_HEADS: usize = 128;
const MAX_VISION_HIDDEN_LAYERS: usize = 128;
const MAX_VISION_HEAD_DIM: usize = 512;
const MAX_VISION_PATCH_SIZE: usize = 64;
const MAX_VISION_POSITION_EMBEDDING_SIZE: usize = 65_536;
const MAX_VISION_POOLING_KERNEL_SIZE: usize = 16;
const MAX_VISION_MAX_POSITION_EMBEDDINGS: usize = 1_048_576;
const MAX_TEXT_HIDDEN_SIZE: usize = 65_536;
const MAX_VISION_PATCH_FEATURES: usize = 3 * MAX_VISION_PATCH_SIZE * MAX_VISION_PATCH_SIZE;
const MAX_VISION_PROJECTION_WIDTH: usize = 16_384;
const MAX_VISION_WEIGHT_ELEMENTS: usize = 64 * 1024 * 1024;
const MAX_VISION_MODEL_MATRIX_ELEMENTS: usize = 1_000_000_000;
const MAX_VISION_PREPARED_ELEMENTS: usize = 64_000_000;
const MAX_VISION_OUTPUT_ELEMENTS: usize = 80_000_000;
const MAX_VISION_ATTENTION_SCORE_ELEMENTS: usize = 2_000_000_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionRopeParameters {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

fn default_rope_type() -> String {
    "default".to_string()
}

fn default_rope_theta() -> f32 {
    100.0
}

impl Default for VisionRopeParameters {
    fn default() -> Self {
        Self {
            rope_type: default_rope_type(),
            rope_theta: default_rope_theta(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gemma4VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub position_embedding_size: usize,
    pub pooling_kernel_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_hidden_activation")]
    pub hidden_activation: String,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default)]
    pub standardize: bool,
    #[serde(default)]
    pub use_clipped_linears: bool,
    #[serde(default)]
    pub rope_parameters: VisionRopeParameters,
}

fn default_hidden_activation() -> String {
    "gelu_pytorch_tanh".to_string()
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gemma4Config {
    #[serde(default)]
    pub text_config: serde_json::Value,
    pub vision_config: Gemma4VisionConfig,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
    #[serde(default = "default_boi_token_id")]
    pub boi_token_id: u32,
    #[serde(default = "default_eoi_token_id")]
    pub eoi_token_id: u32,
    #[serde(default = "default_vision_soft_tokens_per_image")]
    pub vision_soft_tokens_per_image: u32,
}

fn default_image_token_id() -> u32 {
    258_880
}

fn default_boi_token_id() -> u32 {
    255_999
}

fn default_eoi_token_id() -> u32 {
    258_882
}

fn default_vision_soft_tokens_per_image() -> u32 {
    280
}

impl Gemma4Config {
    pub fn load_from_dir(weights_dir: &std::path::Path) -> anyhow::Result<Self> {
        let root = std::fs::canonicalize(weights_dir)
            .map_err(|e| anyhow!("failed to resolve {}: {e}", weights_dir.display()))?;
        let requested = root.join("config.json");
        let cfg_path = std::fs::canonicalize(&requested)
            .map_err(|e| anyhow!("failed to resolve {}: {e}", requested.display()))?;
        ensure!(
            cfg_path.starts_with(&root),
            "{} escapes the model directory",
            cfg_path.display()
        );
        let file = std::fs::File::open(&cfg_path)
            .map_err(|e| anyhow!("failed to open {}: {e}", cfg_path.display()))?;
        ensure!(
            file.metadata()
                .map_err(|e| anyhow!("failed to inspect {}: {e}", cfg_path.display()))?
                .is_file(),
            "{} is not a regular file",
            cfg_path.display()
        );
        let mut raw = Vec::new();
        file.take((MAX_CONFIG_BYTES + 1) as u64)
            .read_to_end(&mut raw)
            .map_err(|e| anyhow!("failed to read {}: {e}", cfg_path.display()))?;
        ensure!(
            raw.len() <= MAX_CONFIG_BYTES,
            "{} exceeds the {MAX_CONFIG_BYTES}-byte config limit",
            cfg_path.display()
        );
        let cfg: Self = serde_json::from_slice(&raw).map_err(|e| {
            anyhow!(
                "failed to parse Gemma4Config from {}: {e}",
                cfg_path.display()
            )
        })?;
        cfg.validate()?;
        Ok(cfg)
    }
    pub fn validate(&self) -> anyhow::Result<()> {
        let v = &self.vision_config;
        macro_rules! bounds {
            ($($field:ident => $maximum:expr),*) => {
                $(bound(stringify!($field), v.$field, $maximum)?;)*
            };
        }
        bounds!(
            hidden_size => MAX_VISION_HIDDEN_SIZE,
            intermediate_size => MAX_VISION_INTERMEDIATE_SIZE,
            num_attention_heads => MAX_VISION_ATTENTION_HEADS,
            num_key_value_heads => MAX_VISION_KV_HEADS,
            num_hidden_layers => MAX_VISION_HIDDEN_LAYERS,
            head_dim => MAX_VISION_HEAD_DIM,
            patch_size => MAX_VISION_PATCH_SIZE,
            position_embedding_size => MAX_VISION_POSITION_EMBEDDING_SIZE,
            pooling_kernel_size => MAX_VISION_POOLING_KERNEL_SIZE,
            max_position_embeddings => MAX_VISION_MAX_POSITION_EMBEDDINGS
        );
        ensure!(
            v.num_attention_heads % v.num_key_value_heads == 0,
            "vision attention heads must be divisible by KV heads"
        );
        ensure!(
            v.head_dim % 4 == 0,
            "vision head_dim must be divisible by 4 for two-axis RoPE"
        );
        ensure!(
            v.hidden_activation == "gelu_pytorch_tanh",
            "unsupported vision hidden_activation {:?}",
            v.hidden_activation
        );
        ensure!(
            v.rms_norm_eps.is_finite() && v.rms_norm_eps > 0.0,
            "vision rms_norm_eps must be finite and > 0"
        );
        ensure!(
            v.attention_dropout.is_finite() && v.attention_dropout == 0.0,
            "rvLLM vision inference requires attention_dropout=0"
        );
        ensure!(
            v.rope_parameters.rope_type == "default",
            "unsupported vision rope_type {:?}",
            v.rope_parameters.rope_type
        );
        ensure!(
            v.rope_parameters.rope_theta.is_finite() && v.rope_parameters.rope_theta > 0.0,
            "vision rope_theta must be finite and > 0"
        );
        ensure!(
            self.vision_soft_tokens_per_image > 0
                && self.vision_soft_tokens_per_image <= MAX_VISION_SOFT_TOKENS,
            "vision_soft_tokens_per_image must be in 1..={MAX_VISION_SOFT_TOKENS}"
        );
        let patch_features = bounded_product(
            "vision patch feature count",
            &[3, v.patch_size, v.patch_size],
            MAX_VISION_PATCH_FEATURES,
        )?;
        let query_width = bounded_product(
            "vision query projection width",
            &[v.num_attention_heads, v.head_dim],
            MAX_VISION_PROJECTION_WIDTH,
        )?;
        let kv_width = bounded_product(
            "vision KV projection width",
            &[v.num_key_value_heads, v.head_dim],
            MAX_VISION_PROJECTION_WIDTH,
        )?;
        bounded_product(
            "vision pooling stride",
            &[v.pooling_kernel_size, v.patch_size],
            MAX_VISION_POOLING_KERNEL_SIZE * MAX_VISION_PATCH_SIZE,
        )?;
        let patches = bounded_product(
            "vision patch budget",
            &[
                usize::try_from(self.vision_soft_tokens_per_image)?,
                v.pooling_kernel_size,
                v.pooling_kernel_size,
            ],
            MAX_VISION_PATCHES,
        )?;
        bounded_product(
            "vision prepared image elements",
            &[patches, patch_features],
            MAX_VISION_PREPARED_ELEMENTS,
        )?;
        bounded_product(
            "vision attention score elements",
            &[v.num_attention_heads, patches, patches],
            MAX_VISION_ATTENTION_SCORE_ELEMENTS,
        )?;
        let text_hidden = self
            .text_config
            .get("hidden_size")
            .and_then(|value| value.as_u64())
            .ok_or_else(|| anyhow!("text_config.hidden_size is required"))?;
        let text_hidden = usize::try_from(text_hidden)
            .map_err(|_| anyhow!("text_config.hidden_size does not fit usize"))?;
        ensure!(
            text_hidden > 0 && text_hidden <= MAX_TEXT_HIDDEN_SIZE,
            "text_config.hidden_size must be in 1..={MAX_TEXT_HIDDEN_SIZE}, got {text_hidden}"
        );
        bounded_product(
            "vision soft-token output elements",
            &[
                usize::try_from(self.vision_soft_tokens_per_image)?,
                text_hidden,
            ],
            MAX_VISION_OUTPUT_ELEMENTS,
        )?;
        let patch_projection = bounded_product(
            "vision patch projection weight",
            &[v.hidden_size, patch_features],
            MAX_VISION_WEIGHT_ELEMENTS,
        )?;
        let position_table = bounded_product(
            "vision position embedding table",
            &[2, v.position_embedding_size, v.hidden_size],
            MAX_VISION_WEIGHT_ELEMENTS,
        )?;
        let query_projection = bounded_product(
            "vision query projection weight",
            &[v.hidden_size, query_width],
            MAX_VISION_WEIGHT_ELEMENTS,
        )?;
        let kv_projection = bounded_product(
            "vision KV projection weight",
            &[v.hidden_size, kv_width],
            MAX_VISION_WEIGHT_ELEMENTS,
        )?;
        let mlp_projection = bounded_product(
            "vision MLP projection weight",
            &[v.hidden_size, v.intermediate_size],
            MAX_VISION_WEIGHT_ELEMENTS,
        )?;
        let embed_projection = bounded_product(
            "vision multimodal projection weight",
            &[text_hidden, v.hidden_size],
            MAX_VISION_WEIGHT_ELEMENTS,
        )?;
        let per_layer = checked_sum(
            "vision per-layer matrix elements",
            &[
                query_projection,
                query_projection,
                kv_projection,
                kv_projection,
                mlp_projection,
                mlp_projection,
                mlp_projection,
            ],
            MAX_VISION_MODEL_MATRIX_ELEMENTS,
        )?;
        let layer_matrices = bounded_product(
            "vision layer matrix elements",
            &[v.num_hidden_layers, per_layer],
            MAX_VISION_MODEL_MATRIX_ELEMENTS,
        )?;
        checked_sum(
            "vision model matrix elements",
            &[
                layer_matrices,
                patch_projection,
                position_table,
                embed_projection,
            ],
            MAX_VISION_MODEL_MATRIX_ELEMENTS,
        )?;
        ensure!(
            self.image_token_id != self.boi_token_id
                && self.image_token_id != self.eoi_token_id
                && self.boi_token_id != self.eoi_token_id,
            "image/boi/eoi token ids must be distinct"
        );
        Ok(())
    }
}

fn bounded_product(name: &str, factors: &[usize], maximum: usize) -> anyhow::Result<usize> {
    let product = factors.iter().try_fold(1usize, |product, factor| {
        product
            .checked_mul(*factor)
            .ok_or_else(|| anyhow!("{name} overflow"))
    })?;
    ensure!(product <= maximum, "{name} {product} exceeds {maximum}");
    Ok(product)
}

fn bound(name: &str, value: usize, maximum: usize) -> anyhow::Result<()> {
    ensure!(
        value > 0 && value <= maximum,
        "vision_config.{name} must be in 1..={maximum}, got {value}"
    );
    Ok(())
}

fn checked_sum(name: &str, terms: &[usize], maximum: usize) -> anyhow::Result<usize> {
    let sum = terms.iter().try_fold(0usize, |sum, term| {
        sum.checked_add(*term)
            .ok_or_else(|| anyhow!("{name} overflow"))
    })?;
    ensure!(sum <= maximum, "{name} {sum} exceeds {maximum}");
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_vision_config_with_defaults() {
        let raw = r#"{"hidden_size":1152,"intermediate_size":4304,"num_attention_heads":16,"num_key_value_heads":16,"num_hidden_layers":27,"head_dim":72,"patch_size":16,"position_embedding_size":10240,"pooling_kernel_size":3,"max_position_embeddings":131072,"standardize":true}"#;
        let cfg: Gemma4VisionConfig = serde_json::from_str(raw).unwrap();
        assert_eq!(cfg.hidden_size, 1152);
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.pooling_kernel_size, 3);
        assert!(cfg.standardize);
        assert_eq!(cfg.hidden_activation, "gelu_pytorch_tanh");
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!((cfg.rope_parameters.rope_theta - 100.0).abs() < 1e-6);
    }
    #[test]
    fn parses_top_level_with_special_token_defaults() {
        let raw = r#"{"vision_config":{"hidden_size":1152,"intermediate_size":4304,"num_attention_heads":16,"num_key_value_heads":16,"num_hidden_layers":27,"head_dim":72,"patch_size":16,"position_embedding_size":10240,"pooling_kernel_size":3,"max_position_embeddings":131072}}"#;
        let cfg: Gemma4Config = serde_json::from_str(raw).unwrap();
        assert_eq!(cfg.image_token_id, 258_880);
        assert_eq!(cfg.boi_token_id, 255_999);
        assert_eq!(cfg.eoi_token_id, 258_882);
        assert_eq!(cfg.vision_soft_tokens_per_image, 280);
    }
    #[test]
    fn validation_rejects_bad_rope_and_head_geometry() {
        let raw = r#"{"text_config":{"hidden_size":128},"vision_config":{"hidden_size":32,"intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":3,"num_hidden_layers":1,"head_dim":7,"patch_size":4,"position_embedding_size":16,"pooling_kernel_size":1,"max_position_embeddings":16}}"#;
        let cfg: Gemma4Config = serde_json::from_str(raw).unwrap();
        assert!(cfg.validate().is_err());
    }
    #[test]
    fn released_gemma4_config_passes_resource_bounds() {
        let mut cfg = released_cfg();
        cfg.validate().unwrap();
        cfg.vision_soft_tokens_per_image = MAX_VISION_SOFT_TOKENS;
        cfg.validate().unwrap();
    }
    #[test]
    fn validation_rejects_dimension_above_cap() {
        let mut cfg = released_cfg();
        cfg.vision_config.hidden_size = MAX_VISION_HIDDEN_SIZE + 1;
        let error = cfg.validate().unwrap_err().to_string();
        assert!(error.contains("hidden_size"), "unexpected error: {error}");
    }
    #[test]
    fn validation_rejects_compounded_weight_budget() {
        let mut cfg = released_cfg();
        cfg.vision_config.position_embedding_size = MAX_VISION_POSITION_EMBEDDING_SIZE;
        let error = cfg.validate().unwrap_err().to_string();
        assert!(
            error.contains("position embedding table"),
            "unexpected error: {error}"
        );
    }
    #[test]
    fn product_and_sum_bounds_reject_overflow() {
        assert_eq!(
            bounded_product("boundary", &[1_024, 1_024], 1_048_576).unwrap(),
            1_048_576
        );
        assert!(bounded_product("boundary", &[1_024, 1_025], 1_048_576).is_err());
        assert!(bounded_product("overflow", &[usize::MAX, 2], usize::MAX).is_err());
        assert!(checked_sum("overflow", &[usize::MAX, 1], usize::MAX).is_err());
    }
    #[test]
    fn load_rejects_oversized_config_before_parsing() {
        let dir = tempdir();
        std::fs::File::create(dir.join("config.json"))
            .unwrap()
            .set_len((MAX_CONFIG_BYTES + 1) as u64)
            .unwrap();
        let error = Gemma4Config::load_from_dir(&dir).unwrap_err().to_string();
        assert!(
            error.contains("exceeds the 262144-byte config limit"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[cfg(unix)]
    #[test]
    fn load_rejects_config_symlink_escape() {
        use std::os::unix::fs::symlink;
        let dir = tempdir();
        let outside = dir.with_extension("outside.json");
        std::fs::write(&outside, b"{}").unwrap();
        symlink(&outside, dir.join("config.json")).unwrap();
        let error = Gemma4Config::load_from_dir(&dir).unwrap_err().to_string();
        assert!(error.contains("escapes"), "unexpected error: {error}");
        std::fs::remove_file(outside).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }
    fn released_cfg() -> Gemma4Config {
        serde_json::from_str(r#"{"text_config":{"hidden_size":5376},"vision_config":{"hidden_size":1152,"intermediate_size":4304,"num_attention_heads":16,"num_key_value_heads":16,"num_hidden_layers":27,"head_dim":72,"patch_size":16,"position_embedding_size":10240,"pooling_kernel_size":3,"max_position_embeddings":131072,"standardize":true}}"#).unwrap()
    }
    fn tempdir() -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("rvllm-vision-config-test-{nanos}"));
        std::fs::create_dir_all(&path).unwrap();
        path
    }
}
