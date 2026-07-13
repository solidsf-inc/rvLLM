//! Gemma 4 model architecture parser.
//!
//! Parses text, multimodal, per-layer-embedding, and KV-sharing variants
//! from their on-disk configuration and weight metadata.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use rvllm_core::{LoaderCtx, LoaderError, Result, RvllmError};

use crate::load_multiformat::read_model_config;

const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_LAYERS: usize = 1024;
const MAX_WIDTH: usize = 1 << 20;
const MAX_VOCAB: usize = 1 << 22;
const MAX_CONTEXT: usize = 1 << 20;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Gemma4LayerType {
    SlidingAttention,
    GlobalAttention,
}

/// RoPE type for a Gemma4 attention class. `rope_parameters` carries a
/// `rope_type` per class:
///   - `full_attention`:    `{rope_type:"proportional", partial_rotary_factor:0.25, theta:1e6}`
///   - `sliding_attention`: `{rope_type:"default", theta:1e4}` (full rotation)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RopeType {
    /// Standard NTK RoPE over the full head_dim.
    Default,
    /// HF "proportional" variant (partial rotation of head_dim).
    Proportional,
}

impl RopeType {
    fn parse(s: &str) -> Option<RopeType> {
        match s {
            "default" => Some(RopeType::Default),
            "proportional" => Some(RopeType::Proportional),
            _ => None,
        }
    }
}

/// Parsed `rope_parameters[class]` sub-object.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RopeParams {
    pub rope_type: RopeType,
    pub rope_theta: f32,
    /// Fraction of head_dim that is rotated. `1.0` for full rotation.
    pub partial_rotary_factor: f32,
}

#[derive(Clone, Debug)]
pub struct Gemma4Arch {
    pub num_hidden_layers: usize,
    pub num_kv_shared_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub head_dim_sliding: usize,
    pub head_dim_global: usize,
    pub num_kv_heads_sliding: usize,
    pub num_kv_heads_global: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub hidden_size_per_layer_input: usize,
    pub vocab_size_per_layer_input: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub sliding_window_size: usize,
    pub rope_theta_sliding: f32,
    pub rope_theta_global: f32,
    pub partial_rotary_factor_global: f32,
    pub logit_softcap: f32,
    pub layer_types: Vec<Gemma4LayerType>,
    pub weight_prefix: String,
    pub tie_word_embeddings: bool,
    /// Full-attention layers reuse the K projection as V when enabled.
    pub attention_k_eq_v: bool,
    /// True when `config.json#architectures` names a vision-capable
    /// conditional-generation variant (Gemma3/Gemma4
    /// ForConditionalGeneration, including Gemma4Unified) AND the on-disk weights contain
    /// `vision_tower.*` and `multi_modal_projector.*` tensors.
    pub is_multimodal: bool,
    /// Directory containing the vision tower + mmproj safetensors
    /// shards. Currently the same as the LLM weights dir (HF ships
    /// Gemma 3/4 multimodal as a single weight set). `Some` iff
    /// `is_multimodal`.
    pub vision_weights_dir: Option<PathBuf>,

    // Optional per-layer-embedding and KV-sharing fields.
    pub final_logit_softcapping: f32,
    /// Full dual-rope parameters for the global/full-attention class.
    pub rope_full: RopeParams,
    /// Full dual-rope parameters for the sliding-attention class.
    pub rope_sliding: RopeParams,
    /// Explicit programmatic override. Config parsing never reads hidden
    /// environment overrides.
    pub runtime_sliding_window: Option<usize>,
    /// Activation named by the model config.
    pub hidden_activation: String,
}

impl Gemma4Arch {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let (bytes, p) = read_model_config(dir)?;
        let v: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| RvllmError::Loader {
                err: LoaderError::Corrupt {
                    detail: format!("config.json: {e}"),
                },
                ctx: LoaderCtx {
                    path: p.clone(),
                    tensor: None,
                },
                bt: std::backtrace::Backtrace::capture(),
            })?;
        let bad = |detail: String| RvllmError::Loader {
            err: LoaderError::Corrupt { detail },
            ctx: LoaderCtx {
                path: p.clone(),
                tensor: None,
            },
            bt: std::backtrace::Backtrace::capture(),
        };

        let tc = if let Some(text) = v.get("text_config") {
            text
        } else {
            &v
        };
        if !tc.is_object() {
            return Err(bad("text_config must be an object".into()));
        }
        let required_usize = |key: &str| -> Result<usize> {
            tc.get(key)
                .and_then(|value| value.as_u64())
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| bad(format!("missing or invalid required field {key}")))
        };
        let required_f32 = |key: &str| -> Result<f32> {
            let value = tc
                .get(key)
                .and_then(|value| value.as_f64())
                .ok_or_else(|| bad(format!("missing or invalid required field {key}")))?;
            let value = value as f32;
            if !value.is_finite() {
                return Err(bad(format!("field {key} must be finite")));
            }
            Ok(value)
        };

        let num_hidden_layers = required_usize("num_hidden_layers")?;
        let num_kv_shared_layers = tc
            .get("num_kv_shared_layers")
            .and_then(|value| value.as_u64())
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0);
        let hidden_size = required_usize("hidden_size")?;
        let num_attention_heads = required_usize("num_attention_heads")?;
        let head_dim_sliding = required_usize("head_dim")?;
        let head_dim_global = tc["global_head_dim"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(head_dim_sliding);
        let intermediate_size = required_usize("intermediate_size")?;
        let vocab_size = tc["vocab_size"]
            .as_u64()
            .or_else(|| v["vocab_size"].as_u64())
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| bad("missing or invalid required field vocab_size".into()))?;
        let hidden_size_per_layer_input = tc["hidden_size_per_layer_input"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0);
        let vocab_size_per_layer_input = if hidden_size_per_layer_input > 0 {
            required_usize("vocab_size_per_layer_input")?
        } else {
            0
        };
        let rms_norm_eps = required_f32("rms_norm_eps")?;
        let max_position_embeddings = required_usize("max_position_embeddings")?;
        let sliding_window_size = tc["sliding_window"]
            .as_u64()
            .or_else(|| tc["sliding_window_size"].as_u64())
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| bad("missing or invalid sliding_window".into()))?;
        let num_kv_heads_sliding = required_usize("num_key_value_heads")?;
        if num_hidden_layers == 0 || num_hidden_layers > MAX_LAYERS {
            return Err(bad(format!(
                "num_hidden_layers must be in 1..={MAX_LAYERS}"
            )));
        }
        if hidden_size == 0
            || hidden_size > MAX_WIDTH
            || intermediate_size == 0
            || intermediate_size > MAX_WIDTH
            || head_dim_sliding == 0
            || head_dim_sliding > MAX_WIDTH
            || head_dim_global == 0
            || head_dim_global > MAX_WIDTH
        {
            return Err(bad(format!(
                "hidden, intermediate, and head dimensions must be in 1..={MAX_WIDTH}"
            )));
        }
        if vocab_size == 0
            || vocab_size > MAX_VOCAB
            || max_position_embeddings == 0
            || max_position_embeddings > MAX_CONTEXT
        {
            return Err(bad(
                "vocab or context dimension exceeds the loader resource policy".into(),
            ));
        }
        if num_attention_heads == 0
            || num_kv_heads_sliding == 0
            || num_kv_heads_sliding > num_attention_heads
            || num_attention_heads % num_kv_heads_sliding != 0
        {
            return Err(bad("invalid attention/KV head counts".into()));
        }
        if num_kv_shared_layers >= num_hidden_layers && num_kv_shared_layers != 0 {
            return Err(bad(
                "num_kv_shared_layers must be smaller than num_hidden_layers".into(),
            ));
        }
        if sliding_window_size == 0 || sliding_window_size > max_position_embeddings {
            return Err(bad(
                "sliding_window must be within max_position_embeddings".into()
            ));
        }
        if hidden_size_per_layer_input > MAX_WIDTH {
            return Err(bad(
                "per-layer embedding width exceeds the resource policy".into()
            ));
        }
        let layer_types = Self::parse_layer_types(tc, num_hidden_layers).map_err(&bad)?;
        let weight_prefix = Self::detect_weight_prefix(dir).ok_or_else(|| {
            bad("could not detect a supported language-model weight prefix".into())
        })?;

        let num_kv_heads_global = tc["num_global_key_value_heads"]
            .as_u64()
            .or_else(|| tc["num_key_value_heads_global"].as_u64())
            .or_else(|| tc["global_num_key_value_heads"].as_u64())
            .and_then(|value| usize::try_from(value).ok())
            .or_else(|| {
                Self::derive_global_kv_heads(dir, &weight_prefix, &layer_types, head_dim_global)
            })
            .unwrap_or(num_kv_heads_sliding);

        let rope = &tc["rope_parameters"];
        let rope_theta_sliding = rope["sliding_attention"]["rope_theta"]
            .as_f64()
            .or_else(|| tc["rope_theta"].as_f64())
            .map(|value| value as f32)
            .ok_or_else(|| bad("missing sliding-attention rope_theta".into()))?;
        let rope_theta_global = rope["full_attention"]["rope_theta"]
            .as_f64()
            .or_else(|| tc["rope_theta_global"].as_f64())
            .map(|value| value as f32)
            .unwrap_or(rope_theta_sliding);
        let partial_rotary_factor_global = rope["full_attention"]["partial_rotary_factor"]
            .as_f64()
            .or_else(|| tc["partial_rotary_factor"].as_f64())
            .unwrap_or(1.0) as f32;
        let partial_rotary_factor_sliding = rope["sliding_attention"]["partial_rotary_factor"]
            .as_f64()
            .unwrap_or(1.0) as f32;

        let logit_softcap = tc["final_logit_softcapping"]
            .as_f64()
            .or_else(|| tc["logit_softcapping"].as_f64())
            .map(|value| value as f32)
            .ok_or_else(|| bad("missing final_logit_softcapping".into()))?;

        let tie_word_embeddings = tc["tie_word_embeddings"]
            .as_bool()
            .or_else(|| v["tie_word_embeddings"].as_bool())
            .ok_or_else(|| bad("missing tie_word_embeddings".into()))?;
        let attention_k_eq_v = match tc.get("attention_k_eq_v") {
            Some(value) => value
                .as_bool()
                .ok_or_else(|| bad("attention_k_eq_v must be a boolean".into()))?,
            None => false,
        };

        let final_logit_softcapping = logit_softcap;
        let hidden_activation = tc["hidden_activation"]
            .as_str()
            .or_else(|| tc["hidden_act"].as_str())
            .ok_or_else(|| bad("missing hidden_activation".into()))?
            .to_string();

        let sliding_rope_type = match rope["sliding_attention"]["rope_type"].as_str() {
            Some(value) => RopeType::parse(value)
                .ok_or_else(|| bad("unsupported sliding-attention rope_type".into()))?,
            None => RopeType::Default,
        };
        let full_rope_type = match rope["full_attention"]["rope_type"].as_str() {
            Some(value) => RopeType::parse(value)
                .ok_or_else(|| bad("unsupported full-attention rope_type".into()))?,
            None if partial_rotary_factor_global < 1.0 => RopeType::Proportional,
            None => RopeType::Default,
        };
        let rope_sliding = RopeParams {
            rope_type: sliding_rope_type,
            rope_theta: rope_theta_sliding,
            partial_rotary_factor: partial_rotary_factor_sliding,
        };
        let rope_full = RopeParams {
            rope_type: full_rope_type,
            rope_theta: rope_theta_global,
            partial_rotary_factor: partial_rotary_factor_global,
        };
        let runtime_sliding_window = None;

        let arch_is_conditional_gen = Self::has_conditional_generation_arch(&v);
        let weights_have_vision = if arch_is_conditional_gen {
            Self::weights_contain_vision(dir)
        } else {
            false
        };
        let is_multimodal = arch_is_conditional_gen && weights_have_vision;
        let vision_weights_dir = if is_multimodal {
            Some(dir.to_path_buf())
        } else {
            None
        };

        if num_attention_heads == 0
            || num_kv_heads_sliding == 0
            || num_kv_heads_global == 0
            || num_kv_heads_sliding > num_attention_heads
            || num_kv_heads_global > num_attention_heads
            || num_attention_heads % num_kv_heads_sliding != 0
            || num_attention_heads % num_kv_heads_global != 0
        {
            return Err(bad("invalid attention/KV head counts".into()));
        }
        if head_dim_sliding == 0
            || head_dim_global == 0
            || head_dim_sliding % 2 != 0
            || head_dim_global % 2 != 0
        {
            return Err(bad("head dimensions must be positive and even".into()));
        }
        if !rms_norm_eps.is_finite()
            || rms_norm_eps <= 0.0
            || !rope_theta_sliding.is_finite()
            || rope_theta_sliding <= 0.0
            || !rope_theta_global.is_finite()
            || rope_theta_global <= 0.0
            || !partial_rotary_factor_global.is_finite()
            || !(0.0..=1.0).contains(&partial_rotary_factor_global)
            || partial_rotary_factor_global == 0.0
            || !partial_rotary_factor_sliding.is_finite()
            || !(0.0..=1.0).contains(&partial_rotary_factor_sliding)
            || partial_rotary_factor_sliding == 0.0
            || rotary_dimension(head_dim_global, partial_rotary_factor_global) == 0
            || rotary_dimension(head_dim_sliding, partial_rotary_factor_sliding) == 0
            || !logit_softcap.is_finite()
            || logit_softcap < 0.0
        {
            return Err(bad("invalid normalization, RoPE, or softcap value".into()));
        }
        let attention_dimensions = [
            num_attention_heads.checked_mul(head_dim_global),
            num_attention_heads.checked_mul(head_dim_sliding),
            num_kv_heads_global.checked_mul(head_dim_global),
            num_kv_heads_sliding.checked_mul(head_dim_sliding),
        ];
        if attention_dimensions
            .into_iter()
            .any(|dimension| dimension.is_none_or(|value| value > MAX_WIDTH))
        {
            return Err(bad(
                "attention dimensions exceed the loader resource policy".into(),
            ));
        }
        if hidden_size_per_layer_input > 0 {
            let total = num_hidden_layers
                .checked_mul(hidden_size_per_layer_input)
                .ok_or_else(|| bad("per-layer embedding dimension overflow".into()))?;
            if total > MAX_WIDTH {
                return Err(bad(
                    "per-layer embedding dimension exceeds the loader resource policy".into(),
                ));
            }
            if vocab_size_per_layer_input == 0 || vocab_size_per_layer_input > MAX_VOCAB {
                return Err(bad("invalid vocab_size_per_layer_input".into()));
            }
        }
        Ok(Self {
            num_hidden_layers,
            num_kv_shared_layers,
            hidden_size,
            num_attention_heads,
            head_dim_sliding,
            head_dim_global,
            num_kv_heads_sliding,
            num_kv_heads_global,
            intermediate_size,
            vocab_size,
            hidden_size_per_layer_input,
            vocab_size_per_layer_input,
            rms_norm_eps,
            max_position_embeddings,
            sliding_window_size,
            rope_theta_sliding,
            rope_theta_global,
            partial_rotary_factor_global,
            logit_softcap,
            layer_types,
            weight_prefix,
            tie_word_embeddings,
            attention_k_eq_v,
            is_multimodal,
            vision_weights_dir,
            final_logit_softcapping,
            rope_full,
            rope_sliding,
            runtime_sliding_window,
            hidden_activation,
        })
    }

    pub fn layer_uses_k_for_v(&self, layer_idx: usize) -> bool {
        self.attention_k_eq_v
            && self
                .layer_types
                .get(layer_idx)
                .is_some_and(|layer_type| matches!(layer_type, Gemma4LayerType::GlobalAttention))
    }

    /// Returns true if the `architectures` array in config.json contains
    /// a Gemma 3/4 conditional-generation variant. These are the
    /// multimodal class names produced by HF's transformers; the
    /// text-only variant is `Gemma4ForCausalLM`.
    fn has_conditional_generation_arch(cfg: &serde_json::Value) -> bool {
        let Some(arr) = cfg.get("architectures").and_then(|v| v.as_array()) else {
            return false;
        };
        arr.iter().any(|v| match v.as_str() {
            Some("Gemma3ForConditionalGeneration") => true,
            Some("Gemma4ForConditionalGeneration") => true,
            Some("Gemma4UnifiedForConditionalGeneration") => true,
            _ => false,
        })
    }

    /// Scan the weights directory for tensors whose names start with
    /// `vision_tower.` and `multi_modal_projector.`. Returns true only
    /// if BOTH families are present (a vision tower without an mmproj
    /// is not usable).
    ///
    /// We prefer the cheap path: parse `model.safetensors.index.json`
    /// and inspect its `weight_map` keys. If no index exists, we fall
    /// back to parsing the single-shard safetensors header (no tensor
    /// data is read).
    fn weights_contain_vision(dir: &Path) -> bool {
        let mut has_tower = false;
        let mut has_mmproj = false;

        if let Ok(bytes) = read_bounded(dir, "model.safetensors.index.json", MAX_INDEX_BYTES) {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                if let Some(map) = v.get("weight_map").and_then(|m| m.as_object()) {
                    for key in map.keys() {
                        if Self::is_vision_tower_key(key) {
                            has_tower = true;
                        }
                        if Self::is_mmproj_key(key) {
                            has_mmproj = true;
                        }
                        if has_tower && has_mmproj {
                            return true;
                        }
                    }
                    return has_tower && has_mmproj;
                }
            }
        }

        if let Some(names) = Self::safetensors_tensor_names_from_dir(dir) {
            for key in names {
                if Self::is_vision_tower_key(&key) {
                    has_tower = true;
                }
                if Self::is_mmproj_key(&key) {
                    has_mmproj = true;
                }
                if has_tower && has_mmproj {
                    return true;
                }
            }
        }
        has_tower && has_mmproj
    }

    fn is_vision_tower_key(key: &str) -> bool {
        key.starts_with("vision_tower.")
            || key.starts_with("model.vision_tower.")
            || key.contains(".vision_tower.")
    }

    fn is_mmproj_key(key: &str) -> bool {
        key.starts_with("multi_modal_projector.")
            || key.starts_with("model.multi_modal_projector.")
            || key.contains(".multi_modal_projector.")
    }

    /// Read just the top-level tensor names out of a single safetensors file.
    /// Returns `None` on any structural mismatch — this is a best-effort probe,
    /// the real loader will surface errors later.
    fn safetensors_tensor_names_from_dir(dir: &Path) -> Option<Vec<String>> {
        let (mut f, _) = open_fixed_file(dir, "model.safetensors").ok()?;
        if f.metadata().ok()?.len() > crate::safetensors::MAX_SAFETENSORS_SHARD_BYTES {
            return None;
        }
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf).ok()?;
        let header_bytes = usize::try_from(u64::from_le_bytes(len_buf)).ok()?;
        if header_bytes > crate::safetensors::MAX_SAFETENSORS_HEADER_BYTES {
            return None;
        }
        f.seek(SeekFrom::Start(8)).ok()?;
        let mut header = vec![0u8; header_bytes];
        f.read_exact(&mut header).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&header).ok()?;
        let obj = v.as_object()?;
        let mut out = Vec::with_capacity(obj.len());
        for k in obj.keys() {
            if k != "__metadata__" {
                out.push(k.clone());
            }
        }
        Some(out)
    }

    fn parse_layer_types(
        tc: &serde_json::Value,
        n: usize,
    ) -> std::result::Result<Vec<Gemma4LayerType>, String> {
        if let Some(arr) = tc["layer_types"].as_array() {
            if arr.len() != n {
                return Err(format!("layer_types has len {}, expected {n}", arr.len()));
            }
            return arr
                .iter()
                .enumerate()
                .map(|(index, t)| match t.as_str() {
                    Some("global_attention" | "full_attention") => {
                        Ok(Gemma4LayerType::GlobalAttention)
                    }
                    Some("sliding_attention") => Ok(Gemma4LayerType::SlidingAttention),
                    Some(value) => Err(format!("unsupported layer_types[{index}]={value:?}")),
                    None => Err(format!("layer_types[{index}] is not a string")),
                })
                .collect();
        }
        if let Some(pattern) = tc["sliding_window_pattern"].as_u64() {
            let pattern = usize::try_from(pattern)
                .map_err(|_| "sliding_window_pattern does not fit usize".to_string())?;
            if pattern == 0 {
                return Err("sliding_window_pattern must be positive".into());
            }
            return Ok((0..n)
                .map(|i| {
                    if (i + 1) % pattern == 0 {
                        Gemma4LayerType::GlobalAttention
                    } else {
                        Gemma4LayerType::SlidingAttention
                    }
                })
                .collect());
        }
        Err("config must provide layer_types or sliding_window_pattern".into())
    }

    fn derive_global_kv_heads(
        dir: &Path,
        weight_prefix: &str,
        layer_types: &[Gemma4LayerType],
        head_dim_global: usize,
    ) -> Option<usize> {
        if head_dim_global == 0 {
            return None;
        }
        let global_idx = layer_types
            .iter()
            .position(|t| *t == Gemma4LayerType::GlobalAttention)?;
        let name = format!("{weight_prefix}.layers.{global_idx}.self_attn.k_proj.weight");
        let shape = Self::safetensors_tensor_shape_from_dir(dir, &name)?;
        let rows = *shape.first()?;
        (rows % head_dim_global == 0).then_some(rows / head_dim_global)
    }

    fn safetensors_tensor_shape_from_dir(dir: &Path, tensor_name: &str) -> Option<Vec<usize>> {
        let (mut f, _) = open_fixed_file(dir, "model.safetensors").ok()?;
        if f.metadata().ok()?.len() > crate::safetensors::MAX_SAFETENSORS_SHARD_BYTES {
            return None;
        }
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf).ok()?;
        let header_bytes = usize::try_from(u64::from_le_bytes(len_buf)).ok()?;
        if header_bytes > crate::safetensors::MAX_SAFETENSORS_HEADER_BYTES {
            return None;
        }
        f.seek(SeekFrom::Start(8)).ok()?;
        let mut header = vec![0u8; header_bytes];
        f.read_exact(&mut header).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&header).ok()?;
        v.get(tensor_name)?
            .get("shape")?
            .as_array()?
            .iter()
            .map(|v| v.as_u64().map(|n| n as usize))
            .collect()
    }

    fn detect_weight_prefix(dir: &Path) -> Option<String> {
        if let Ok(bytes) = read_bounded(dir, "model.safetensors.index.json", MAX_INDEX_BYTES) {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                if let Some(map) = v["weight_map"].as_object() {
                    if map
                        .keys()
                        .any(|key| key.starts_with("model.language_model."))
                    {
                        return Some("model.language_model".to_string());
                    }
                    if map.keys().any(|key| key.starts_with("language_model.")) {
                        return Some("language_model".to_string());
                    }
                    if map.keys().any(|key| key.starts_with("model.")) {
                        return Some("model".to_string());
                    }
                }
            }
        }
        if let Some(names) = Self::safetensors_tensor_names_from_dir(dir) {
            if names
                .iter()
                .any(|key| key.starts_with("model.language_model."))
            {
                return Some("model.language_model".to_string());
            }
            if names.iter().any(|key| key.starts_with("language_model.")) {
                return Some("language_model".to_string());
            }
            if names.iter().any(|key| key.starts_with("model.")) {
                return Some("model".to_string());
            }
        }
        None
    }

    pub fn head_dim_for_layer(&self, layer_idx: usize) -> usize {
        match self.layer_types[layer_idx] {
            Gemma4LayerType::SlidingAttention => self.head_dim_sliding,
            Gemma4LayerType::GlobalAttention => self.head_dim_global,
        }
    }

    pub fn num_kv_heads_for_layer(&self, layer_idx: usize) -> usize {
        match self.layer_types[layer_idx] {
            Gemma4LayerType::SlidingAttention => self.num_kv_heads_sliding,
            Gemma4LayerType::GlobalAttention => self.num_kv_heads_global,
        }
    }

    pub fn rotary_dim_for_layer(&self, layer_idx: usize) -> usize {
        match self.layer_types[layer_idx] {
            Gemma4LayerType::SlidingAttention => rotary_dimension(
                self.head_dim_sliding,
                self.rope_sliding.partial_rotary_factor,
            ),
            Gemma4LayerType::GlobalAttention => {
                rotary_dimension(self.head_dim_global, self.rope_full.partial_rotary_factor)
            }
        }
    }

    pub fn rope_theta_for_layer(&self, layer_idx: usize) -> f32 {
        match self.layer_types[layer_idx] {
            Gemma4LayerType::SlidingAttention => self.rope_theta_sliding,
            Gemma4LayerType::GlobalAttention => self.rope_theta_global,
        }
    }

    pub fn q_dim_for_layer(&self, layer_idx: usize) -> usize {
        self.num_attention_heads * self.head_dim_for_layer(layer_idx)
    }

    pub fn kv_dim_for_layer(&self, layer_idx: usize) -> usize {
        self.num_kv_heads_for_layer(layer_idx) * self.head_dim_for_layer(layer_idx)
    }

    pub fn max_head_dim(&self) -> usize {
        self.head_dim_sliding.max(self.head_dim_global)
    }

    pub fn max_kv_heads(&self) -> usize {
        self.num_kv_heads_sliding.max(self.num_kv_heads_global)
    }

    pub fn max_q_dim(&self) -> usize {
        self.num_attention_heads * self.max_head_dim()
    }

    /// True iff this configuration carries per-layer embeddings.
    pub fn is_e4b(&self) -> bool {
        self.hidden_size_per_layer_input > 0
    }

    /// Total width of the per-layer embedding table's second dim:
    /// `num_hidden_layers * hidden_size_per_layer_input`
    pub fn per_layer_embed_total(&self) -> usize {
        self.num_hidden_layers
            .checked_mul(self.hidden_size_per_layer_input)
            .expect("validated by Gemma4Arch::from_dir")
    }

    /// PLE embed-scale = `sqrt(hidden_size_per_layer_input)`.
    pub fn ple_embed_scale(&self) -> f32 {
        (self.hidden_size_per_layer_input as f32).sqrt()
    }

    /// Effective sliding-window length.
    pub fn effective_sliding_window(&self) -> usize {
        self.runtime_sliding_window
            .unwrap_or(self.sliding_window_size)
    }

    /// Index of the first KV-shared layer. Layers `[kv_shared_start, num_layers)`
    /// do not own K/V projections and instead read a share-source layer's KV.
    pub fn kv_shared_start(&self) -> usize {
        self.num_hidden_layers
            .saturating_sub(self.num_kv_shared_layers)
    }

    /// Returns `true` if `layer_idx` owns its own K/V projection weights.
    /// The first `num_layers - num_kv_shared_layers` layers own KV; the
    /// tail `num_kv_shared_layers` read a share source.
    pub fn layer_owns_kv(&self, layer_idx: usize) -> bool {
        layer_idx < self.kv_shared_start()
    }

    /// Build the KV-share source map: `kv_share_src[layer] -> Option<layer>`.
    ///
    /// For layers that own their own KV the entry is `None`. For each shared
    /// tail layer the entry is the index of the owning layer it reads from.
    ///
    /// Shared layers resolve to the most recent owning layer of the same
    /// attention class so cache geometry remains compatible.
    pub fn build_kv_share_src(&self) -> Result<Vec<Option<usize>>> {
        let n = self.num_hidden_layers;
        let start = self.kv_shared_start();
        let mut map = vec![None; n];
        for layer in start..n {
            let want = self.layer_types[layer];
            // Last owning layer (< start) with the same attention class.
            let src = (0..start).rev().find(|&i| self.layer_types[i] == want);
            match src {
                Some(s) => map[layer] = Some(s),
                None => {
                    return Err(RvllmError::Loader {
                        err: LoaderError::Corrupt {
                            detail: format!(
                                "KV-share: shared layer {layer} ({:?}) has no \
                                 same-class owning source in [0,{start})",
                                want
                            ),
                        },
                        ctx: LoaderCtx {
                            path: PathBuf::new(),
                            tensor: None,
                        },
                        bt: std::backtrace::Backtrace::capture(),
                    });
                }
            }
        }
        Ok(map)
    }

    /// RoPE params for a layer by its attention class.
    pub fn rope_params_for_layer(&self, layer_idx: usize) -> RopeParams {
        match self.layer_types[layer_idx] {
            Gemma4LayerType::SlidingAttention => self.rope_sliding,
            Gemma4LayerType::GlobalAttention => self.rope_full,
        }
    }
}

fn rotary_dimension(head_dim: usize, factor: f32) -> usize {
    let dimension = (head_dim as f32 * factor) as usize;
    (dimension / 2) * 2
}

fn open_fixed_file(root: &Path, name: &str) -> Result<(std::fs::File, PathBuf)> {
    let io = |source: std::io::Error, path: &Path| RvllmError::Io {
        err: rvllm_core::IoError::from(&source),
        path: path.to_path_buf(),
        source,
    };
    let root = std::fs::canonicalize(root).map_err(|e| io(e, root))?;
    let requested = root.join(name);
    let path = std::fs::canonicalize(&requested).map_err(|e| io(e, &requested))?;
    let bad = |detail| RvllmError::Loader {
        err: LoaderError::Corrupt { detail },
        ctx: LoaderCtx {
            path: path.to_path_buf(),
            tensor: None,
        },
        bt: std::backtrace::Backtrace::capture(),
    };
    if !path.starts_with(&root) {
        return Err(bad(format!("{name} escapes the model directory")));
    }
    let file = std::fs::File::open(&path).map_err(|e| io(e, &path))?;
    if !file.metadata().map_err(|e| io(e, &path))?.is_file() {
        return Err(bad(format!("{name} is not a regular file")));
    }
    Ok((file, path))
}

fn read_bounded(root: &Path, name: &str, limit: u64) -> Result<Vec<u8>> {
    let (file, path) = open_fixed_file(root, name)?;
    let len = file
        .metadata()
        .map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.clone(),
            source,
        })?
        .len();
    if len > limit {
        return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!("{} is {len} bytes; limit is {limit}", path.display()),
            },
            ctx: LoaderCtx {
                path: path.to_path_buf(),
                tensor: None,
            },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    let mut bytes = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() as u64 > limit {
        return Err(RvllmError::Loader {
            err: LoaderError::Corrupt {
                detail: format!("{} grew beyond {limit} bytes while reading", path.display()),
            },
            ctx: LoaderCtx {
                path: path.to_path_buf(),
                tensor: None,
            },
            bt: std::backtrace::Backtrace::capture(),
        });
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Minimal config.json content that satisfies the required fields
    /// `from_dir` reads. Tests append/replace specific fields per case.
    fn base_text_config() -> serde_json::Value {
        serde_json::json!({
            "text_config": {
                "num_hidden_layers": 6,
                "hidden_size": 256,
                "num_attention_heads": 4,
                "head_dim": 64,
                "global_head_dim": 64,
                "num_key_value_heads": 2,
                "num_global_key_value_heads": 2,
                "intermediate_size": 512,
                "vocab_size": 1024,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 8192,
                "sliding_window": 1024,
                "tie_word_embeddings": true,
                "hidden_activation": "gelu_pytorch_tanh",
                "final_logit_softcapping": 30.0,
                "rope_parameters": {
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                    "full_attention": {
                        "rope_type": "proportional",
                        "rope_theta": 1000000.0,
                        "partial_rotary_factor": 0.25
                    }
                },
                "layer_types": [
                    "sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "sliding_attention", "full_attention"
                ]
            }
        })
    }

    /// Write a minimal valid safetensors file containing the given
    /// tensor names. All tensors are zero-byte F32 placeholders — the
    /// loader probe only reads the JSON header, never the payload.
    fn write_minimal_safetensors(dir: &Path, names: &[&str]) {
        let mut header = serde_json::Map::new();
        for n in names {
            header.insert(
                (*n).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                }),
            );
        }
        let hjson = serde_json::to_string(&header).unwrap();
        let hb = hjson.as_bytes();
        let path = dir.join("model.safetensors");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hb).unwrap();
        f.write_all(&[0u8; 4]).unwrap(); // payload for the one tensor
    }

    #[test]
    fn default_layer_pattern_every_6th_global() {
        let value = serde_json::json!({"sliding_window_pattern": 6});
        let types = Gemma4Arch::parse_layer_types(&value, 12).unwrap();
        // 0:s 1:s 2:s 3:s 4:s 5:g 6:s 7:s 8:s 9:s 10:s 11:g
        assert_eq!(types[0], Gemma4LayerType::SlidingAttention);
        assert_eq!(types[4], Gemma4LayerType::SlidingAttention);
        assert_eq!(types[5], Gemma4LayerType::GlobalAttention);
        assert_eq!(types[11], Gemma4LayerType::GlobalAttention);
    }

    #[test]
    fn parses_real_layer_types() {
        let v: serde_json::Value = serde_json::json!({
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ]
        });
        let types = Gemma4Arch::parse_layer_types(&v, 6).unwrap();
        assert_eq!(types[5], Gemma4LayerType::GlobalAttention);
        assert_eq!(types[0], Gemma4LayerType::SlidingAttention);
    }

    #[test]
    fn parses_sliding_window_pattern() {
        let v: serde_json::Value = serde_json::json!({
            "sliding_window_pattern": 4
        });
        let types = Gemma4Arch::parse_layer_types(&v, 8).unwrap();
        assert_eq!(types[0], Gemma4LayerType::SlidingAttention);
        assert_eq!(types[3], Gemma4LayerType::GlobalAttention);
        assert_eq!(types[7], Gemma4LayerType::GlobalAttention);
    }

    #[test]
    fn rotary_dim_sliding_is_full() {
        let arch = Gemma4Arch {
            num_hidden_layers: 6,
            num_kv_shared_layers: 0,
            hidden_size: 5376,
            num_attention_heads: 32,
            head_dim_sliding: 256,
            head_dim_global: 512,
            num_kv_heads_sliding: 16,
            num_kv_heads_global: 4,
            intermediate_size: 21504,
            vocab_size: 262144,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 262144,
            sliding_window_size: 1024,
            rope_theta_sliding: 10000.0,
            rope_theta_global: 1000000.0,
            partial_rotary_factor_global: 0.25,
            logit_softcap: 30.0,
            layer_types: vec![Gemma4LayerType::SlidingAttention; 6],
            weight_prefix: "model".into(),
            tie_word_embeddings: true,
            attention_k_eq_v: false,
            is_multimodal: false,
            vision_weights_dir: None,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: 0,
            final_logit_softcapping: 30.0,
            rope_full: RopeParams {
                rope_type: RopeType::Proportional,
                rope_theta: 1_000_000.0,
                partial_rotary_factor: 0.25,
            },
            rope_sliding: RopeParams {
                rope_type: RopeType::Default,
                rope_theta: 10000.0,
                partial_rotary_factor: 1.0,
            },
            runtime_sliding_window: None,
            hidden_activation: "gelu_pytorch_tanh".into(),
        };
        assert_eq!(arch.rotary_dim_for_layer(0), 256);
    }

    #[test]
    fn rotary_dim_global_is_partial() {
        let arch = Gemma4Arch {
            num_hidden_layers: 6,
            hidden_size: 5376,
            num_attention_heads: 32,
            head_dim_sliding: 256,
            head_dim_global: 512,
            num_kv_heads_sliding: 16,
            num_kv_heads_global: 4,
            intermediate_size: 21504,
            vocab_size: 262144,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 262144,
            sliding_window_size: 1024,
            rope_theta_sliding: 10000.0,
            rope_theta_global: 1000000.0,
            partial_rotary_factor_global: 0.25,
            logit_softcap: 30.0,
            layer_types: vec![Gemma4LayerType::GlobalAttention; 6],
            weight_prefix: "model".into(),
            tie_word_embeddings: true,
            attention_k_eq_v: false,
            is_multimodal: false,
            vision_weights_dir: None,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: 0,
            num_kv_shared_layers: 0,
            final_logit_softcapping: 30.0,
            rope_full: RopeParams {
                rope_type: RopeType::Proportional,
                rope_theta: 1_000_000.0,
                partial_rotary_factor: 0.25,
            },
            rope_sliding: RopeParams {
                rope_type: RopeType::Default,
                rope_theta: 10000.0,
                partial_rotary_factor: 1.0,
            },
            runtime_sliding_window: None,
            hidden_activation: "gelu_pytorch_tanh".into(),
        };
        // 512 * 0.25 = 128
        assert_eq!(arch.rotary_dim_for_layer(0), 128);
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "rvllm-loader-gemma4-arch-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        base
    }

    #[cfg(unix)]
    #[test]
    fn rejects_config_symlink_escape() {
        use std::os::unix::fs::symlink;

        let dir = tmpdir("config_symlink_root");
        let outside = tmpdir("config_symlink_outside").join("outside.json");
        std::fs::write(&outside, base_text_config().to_string()).unwrap();
        symlink(&outside, dir.join("config.json")).unwrap();

        let error = Gemma4Arch::from_dir(&dir).unwrap_err();
        assert!(matches!(
            error,
            RvllmError::Loader {
                err: LoaderError::Corrupt { ref detail },
                ..
            } if detail.contains("escapes")
        ));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_fixed_metadata_symlink_escapes() {
        use std::os::unix::fs::symlink;

        let dir = tmpdir("metadata_symlink_root");
        let outside = tmpdir("metadata_symlink_outside");
        std::fs::write(outside.join("model.safetensors.index.json"), b"{}").unwrap();
        write_minimal_safetensors(&outside, &["model.embed_tokens.weight"]);
        for name in ["model.safetensors.index.json", "model.safetensors"] {
            symlink(outside.join(name), dir.join(name)).unwrap();
            let error = open_fixed_file(&dir, name).unwrap_err();
            assert!(format!("{error}").contains("escapes"));
        }
    }

    #[test]
    fn text_only_arch_is_unimodal() {
        let dir = tmpdir("textonly");
        let mut cfg = base_text_config();
        cfg["architectures"] = serde_json::json!(["Gemma4ForCausalLM"]);
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        write_minimal_safetensors(&dir, &["model.embed_tokens.weight"]);

        let arch = Gemma4Arch::from_dir(&dir).unwrap();
        assert!(!arch.is_multimodal);
        assert!(arch.vision_weights_dir.is_none());
        assert!(!arch.attention_k_eq_v);
    }

    #[test]
    fn attention_k_eq_v_is_boolean_and_global_only() {
        let dir = tmpdir("attention_k-eq-v");
        let mut cfg = base_text_config();
        cfg["text_config"]["attention_k_eq_v"] = serde_json::json!(true);
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        write_minimal_safetensors(&dir, &["model.embed_tokens.weight"]);

        let arch = Gemma4Arch::from_dir(&dir).unwrap();
        assert!(arch.attention_k_eq_v);
        assert!(!arch.layer_uses_k_for_v(0));
        assert!(arch.layer_uses_k_for_v(5));
        assert!(!arch.layer_uses_k_for_v(6));

        cfg["text_config"]["attention_k_eq_v"] = serde_json::json!("true");
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        assert!(Gemma4Arch::from_dir(&dir).is_err());
    }

    #[test]
    fn conditional_gen_without_vision_weights_is_unimodal() {
        // Arch claims multimodal but no vision tensors on disk -- we
        // refuse to flip the flag. The server must be able to trust
        // is_multimodal == there are weights to load.
        let dir = tmpdir("nocond_weights");
        let mut cfg = base_text_config();
        cfg["architectures"] = serde_json::json!(["Gemma3ForConditionalGeneration"]);
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        write_minimal_safetensors(&dir, &["model.embed_tokens.weight"]);

        let arch = Gemma4Arch::from_dir(&dir).unwrap();
        assert!(!arch.is_multimodal);
        assert!(arch.vision_weights_dir.is_none());
    }

    #[test]
    fn rejects_layer_count_before_layer_vector_allocation() {
        let dir = tmpdir("oversized_layers");
        let mut cfg = base_text_config();
        cfg["text_config"]["num_hidden_layers"] = serde_json::json!(MAX_LAYERS + 1);
        cfg["text_config"]["layer_types"] = serde_json::Value::Null;
        cfg["text_config"]["sliding_window_pattern"] = serde_json::json!(1);
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();

        let error = Gemma4Arch::from_dir(&dir).unwrap_err();
        assert!(matches!(
            error,
            RvllmError::Loader {
                err: LoaderError::Corrupt { ref detail },
                ..
            } if detail.contains("num_hidden_layers")
        ));
    }

    #[test]
    fn sliding_partial_rotary_factor_is_honored_and_validated() {
        let dir = tmpdir("sliding_partial_rope");
        let mut cfg = base_text_config();
        cfg["text_config"]["rope_parameters"]["sliding_attention"]["partial_rotary_factor"] =
            serde_json::json!(0.5);
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        write_minimal_safetensors(&dir, &["model.embed_tokens.weight"]);

        let arch = Gemma4Arch::from_dir(&dir).unwrap();
        assert_eq!(arch.rotary_dim_for_layer(0), 32);

        cfg["text_config"]["rope_parameters"]["sliding_attention"]["partial_rotary_factor"] =
            serde_json::json!(0.001);
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        assert!(Gemma4Arch::from_dir(&dir).is_err());
    }

    #[test]
    fn gemma3_conditional_gen_with_vision_weights_is_multimodal() {
        let dir = tmpdir("gemma3_mm");
        let mut cfg = base_text_config();
        cfg["architectures"] = serde_json::json!(["Gemma3ForConditionalGeneration"]);
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        write_minimal_safetensors(
            &dir,
            &[
                "language_model.embed_tokens.weight",
                "vision_tower.vision_model.embeddings.patch_embedding.weight",
                "multi_modal_projector.mm_input_projection_weight",
            ],
        );

        let arch = Gemma4Arch::from_dir(&dir).unwrap();
        assert!(arch.is_multimodal);
        assert_eq!(arch.vision_weights_dir.as_deref(), Some(dir.as_path()));
    }

    #[test]
    fn gemma4_conditional_gen_with_vision_weights_is_multimodal() {
        let dir = tmpdir("gemma4_mm");
        let mut cfg = base_text_config();
        cfg["architectures"] = serde_json::json!(["Gemma4ForConditionalGeneration"]);
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        write_minimal_safetensors(
            &dir,
            &[
                "model.language_model.embed_tokens.weight",
                "model.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight",
                "model.multi_modal_projector.mm_soft_emb_norm.weight",
            ],
        );

        let arch = Gemma4Arch::from_dir(&dir).unwrap();
        assert!(arch.is_multimodal);
        assert_eq!(arch.vision_weights_dir.as_deref(), Some(dir.as_path()));
    }

    #[test]
    fn gemma4_unified_conditional_gen_without_old_vision_keys_is_text_only() {
        let dir = tmpdir("gemma4_unified_text");
        let mut cfg = base_text_config();
        cfg["architectures"] = serde_json::json!(["Gemma4UnifiedForConditionalGeneration"]);
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        write_minimal_safetensors(
            &dir,
            &[
                "model.language_model.embed_tokens.weight",
                "model.embed_vision.embedding_projection.weight",
                "model.embed_audio.embedding_projection.weight",
            ],
        );

        let arch = Gemma4Arch::from_dir(&dir).unwrap();
        assert_eq!(arch.weight_prefix, "model.language_model");
        assert!(!arch.is_multimodal);
        assert!(arch.vision_weights_dir.is_none());
    }

    #[test]
    fn multimodal_detection_via_index_json() {
        let dir = tmpdir("index_mm");
        let mut cfg = base_text_config();
        cfg["architectures"] = serde_json::json!(["Gemma3ForConditionalGeneration"]);
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();

        // Two shards (referenced names only; the loader probe just
        // reads keys, never opens the shard files).
        let index = serde_json::json!({
            "metadata": {"total_size": 1234},
            "weight_map": {
                "language_model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "vision_tower.vision_model.encoder.layers.0.layer_norm1.weight":
                    "model-00002-of-00002.safetensors",
                "multi_modal_projector.mm_soft_emb_norm.weight":
                    "model-00002-of-00002.safetensors"
            }
        });
        std::fs::write(dir.join("model.safetensors.index.json"), index.to_string()).unwrap();

        let arch = Gemma4Arch::from_dir(&dir).unwrap();
        assert!(arch.is_multimodal);
        assert_eq!(arch.vision_weights_dir.as_deref(), Some(dir.as_path()));
    }

    /// Programmatic sliding-window override is honored when supplied.
    #[test]
    fn runtime_sliding_window_override() {
        let arch = Gemma4Arch {
            num_hidden_layers: 6,
            hidden_size: 256,
            num_attention_heads: 4,
            head_dim_sliding: 64,
            head_dim_global: 64,
            num_kv_heads_sliding: 2,
            num_kv_heads_global: 2,
            intermediate_size: 512,
            vocab_size: 1024,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 8192,
            sliding_window_size: 512,
            rope_theta_sliding: 10000.0,
            rope_theta_global: 1_000_000.0,
            partial_rotary_factor_global: 0.25,
            logit_softcap: 30.0,
            layer_types: vec![Gemma4LayerType::SlidingAttention; 6],
            weight_prefix: "model".into(),
            tie_word_embeddings: false,
            attention_k_eq_v: false,
            is_multimodal: false,
            vision_weights_dir: None,
            hidden_size_per_layer_input: 256,
            vocab_size_per_layer_input: 1024,
            num_kv_shared_layers: 2,
            final_logit_softcapping: 30.0,
            rope_full: RopeParams {
                rope_type: RopeType::Proportional,
                rope_theta: 1_000_000.0,
                partial_rotary_factor: 0.25,
            },
            rope_sliding: RopeParams {
                rope_type: RopeType::Default,
                rope_theta: 10000.0,
                partial_rotary_factor: 1.0,
            },
            runtime_sliding_window: Some(256),
            hidden_activation: "gelu_pytorch_tanh".into(),
        };
        assert_eq!(arch.effective_sliding_window(), 256);
        assert!(arch.is_e4b());
        assert_eq!(arch.kv_shared_start(), 4);
        let src = arch.build_kv_share_src().unwrap();
        assert_eq!(src[4], Some(3));
        assert_eq!(src[5], Some(3));
    }

    #[test]
    fn multimodal_requires_both_tower_and_mmproj() {
        // vision_tower present but mmproj missing -- refuse multimodal.
        let dir = tmpdir("tower_only");
        let mut cfg = base_text_config();
        cfg["architectures"] = serde_json::json!(["Gemma3ForConditionalGeneration"]);
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        write_minimal_safetensors(
            &dir,
            &[
                "language_model.embed_tokens.weight",
                "vision_tower.vision_model.embeddings.patch_embedding.weight",
            ],
        );

        let arch = Gemma4Arch::from_dir(&dir).unwrap();
        assert!(!arch.is_multimodal);
        assert!(arch.vision_weights_dir.is_none());
    }
}
