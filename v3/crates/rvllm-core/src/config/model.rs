//! Model-architecture config, parsed from HF `config.json`.

use std::io::Read;
use std::path::{Path, PathBuf};

use crate::dtype::DType;
use crate::error::{ConfigError, Result, RvllmError};

use super::hf;

const MAX_CONFIG_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ModelArch {
    Qwen2,
    Llama,
    Mistral,
    Gemma2,
    Gemma4,
}

impl ModelArch {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "Qwen2ForCausalLM" => Some(ModelArch::Qwen2),
            "LlamaForCausalLM" => Some(ModelArch::Llama),
            "MistralForCausalLM" => Some(ModelArch::Mistral),
            "Gemma2ForCausalLM" => Some(ModelArch::Gemma2),
            "Gemma4ForCausalLM" => Some(ModelArch::Gemma4),
            "Gemma4ForConditionalGeneration" => Some(ModelArch::Gemma4),
            "Gemma4UnifiedForConditionalGeneration" => Some(ModelArch::Gemma4),
            "Gemma4UnifiedForCausalLM" => Some(ModelArch::Gemma4),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub architecture: ModelArch,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub torch_dtype: DType,
}

impl ModelConfig {
    /// Parse an HF `config.json`. Every referenced field is required.
    pub fn load_hf(dir: &Path) -> Result<Self> {
        let (body, file) = read_config(dir)?;
        let v: serde_json::Value = serde_json::from_slice(&body).map_err(|e| {
            RvllmError::config(
                ConfigError::Inconsistent {
                    reasons: vec![format!("config.json is not valid JSON: {e}")],
                },
                "config.json",
            )
        })?;
        Self::from_hf_value(&v, &file)
    }

    fn from_hf_value(v: &serde_json::Value, file: &Path) -> Result<Self> {
        let arch_name = hf::str_field(v, "architectures.0", file)?;
        let architecture = ModelArch::parse(&arch_name).ok_or_else(|| {
            RvllmError::config(
                ConfigError::InvalidField {
                    name: "architectures[0]",
                    reason: format!("unsupported architecture: {arch_name}"),
                },
                "architectures[0]",
            )
        })?;

        // Gemma 3/4: text model fields nested under text_config.
        let tc = if v["text_config"]["hidden_size"].is_u64() {
            &v["text_config"]
        } else {
            v
        };

        let hidden_size = hf::usize_field(tc, "hidden_size", file)?;
        let num_layers = hf::usize_field(tc, "num_hidden_layers", file)?;
        let num_attention_heads = hf::usize_field(tc, "num_attention_heads", file)?;
        let num_kv_heads = hf::usize_field(tc, "num_key_value_heads", file)?;
        let intermediate_size = hf::usize_field(tc, "intermediate_size", file)?;
        let vocab_size = hf::usize_field(tc, "vocab_size", file)?;
        let max_position_embeddings = hf::usize_field(tc, "max_position_embeddings", file)?;
        let rms_norm_eps = hf::f32_field(tc, "rms_norm_eps", file)?;
        let rope_theta = match tc
            .get("rope_parameters")
            .and_then(|x| x.get("sliding_attention"))
            .and_then(|x| x.get("rope_theta"))
        {
            Some(value) => {
                let x = value.as_f64().ok_or_else(|| {
                    RvllmError::config(
                        ConfigError::HfTypeMismatch {
                            name: "rope_parameters.sliding_attention.rope_theta",
                            expected: "finite number",
                        },
                        "rope_theta",
                    )
                })?;
                let x32 = x as f32;
                if !x.is_finite() || !x32.is_finite() {
                    return Err(RvllmError::config(
                        ConfigError::InvalidField {
                            name: "rope_theta",
                            reason: "must be finite and representable as f32".into(),
                        },
                        "rope_theta",
                    ));
                }
                x32
            }
            None => hf::f32_field(tc, "rope_theta", file)?,
        };
        let tie_word_embeddings = hf::bool_field_opt(tc, "tie_word_embeddings")
            .or_else(|| hf::bool_field_opt(v, "tie_word_embeddings"))
            .ok_or_else(|| {
                RvllmError::config(
                    ConfigError::MissingHfField {
                        name: "tie_word_embeddings",
                        file: file.to_path_buf(),
                    },
                    "tie_word_embeddings",
                )
            })?;
        let torch_dtype = match hf::str_field(v, "torch_dtype", file)
            .or_else(|_| hf::str_field(v, "dtype", file))
            .or_else(|_| hf::str_field(tc, "dtype", file))?
            .as_str()
        {
            "float16" => DType::F16,
            "bfloat16" => DType::Bf16,
            other => {
                return Err(RvllmError::config(
                    ConfigError::InvalidField {
                        name: "torch_dtype",
                        reason: format!("unsupported torch_dtype: {other}"),
                    },
                    "torch_dtype",
                ));
            }
        };

        if num_attention_heads == 0 {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "num_attention_heads",
                    reason: "must be > 0".into(),
                },
                "num_attention_heads",
            ));
        }
        // Gemma 4 has explicit head_dim (256) that doesn't equal hidden_size/num_heads.
        let head_dim = match tc.get("head_dim") {
            Some(value) => {
                let raw = value.as_u64().ok_or_else(|| {
                    RvllmError::config(
                        ConfigError::HfTypeMismatch {
                            name: "head_dim",
                            expected: "non-negative integer",
                        },
                        "head_dim",
                    )
                })?;
                usize::try_from(raw).map_err(|_| {
                    RvllmError::config(
                        ConfigError::InvalidField {
                            name: "head_dim",
                            reason: "value exceeds platform usize".into(),
                        },
                        "head_dim",
                    )
                })?
            }
            None => hidden_size / num_attention_heads,
        };
        if tc["head_dim"].as_u64().is_none() && head_dim * num_attention_heads != hidden_size {
            return Err(RvllmError::config(
                ConfigError::Inconsistent {
                    reasons: vec![format!(
                        "hidden_size {hidden_size} not divisible by num_attention_heads {num_attention_heads}"
                    )],
                },
                "hidden_size",
            ));
        }

        let mut reasons = Vec::new();
        for (name, value) in [
            ("hidden_size", hidden_size),
            ("num_hidden_layers", num_layers),
            ("num_key_value_heads", num_kv_heads),
            ("head_dim", head_dim),
            ("intermediate_size", intermediate_size),
            ("vocab_size", vocab_size),
            ("max_position_embeddings", max_position_embeddings),
        ] {
            if value == 0 {
                reasons.push(format!("{name} must be > 0"));
            }
        }
        if num_attention_heads % num_kv_heads.max(1) != 0 {
            reasons.push(format!(
                "num_attention_heads {num_attention_heads} must be divisible by num_key_value_heads {num_kv_heads}"
            ));
        }
        if !rms_norm_eps.is_finite() || rms_norm_eps <= 0.0 {
            reasons.push("rms_norm_eps must be finite and > 0".into());
        }
        if !rope_theta.is_finite() || rope_theta <= 0.0 {
            reasons.push("rope_theta must be finite and > 0".into());
        }
        if !reasons.is_empty() {
            return Err(RvllmError::config(
                ConfigError::Inconsistent { reasons },
                "config.json",
            ));
        }

        Ok(Self {
            architecture,
            hidden_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            tie_word_embeddings,
            torch_dtype,
        })
    }
}

fn read_config(root: &Path) -> Result<(Vec<u8>, PathBuf)> {
    let io = |source: std::io::Error, path: &Path| RvllmError::Io {
        err: crate::error::IoError::from(&source),
        path: path.to_path_buf(),
        source,
    };
    let invalid = |reason| {
        RvllmError::config(
            ConfigError::InvalidField {
                name: "config.json",
                reason,
            },
            "config.json",
        )
    };
    let root = std::fs::canonicalize(root).map_err(|e| io(e, root))?;
    let requested = root.join("config.json");
    let path = std::fs::canonicalize(&requested).map_err(|e| io(e, &requested))?;
    if !path.starts_with(&root) {
        return Err(invalid("path escapes the model directory".into()));
    }
    let file = std::fs::File::open(&path).map_err(|e| io(e, &path))?;
    if !file.metadata().map_err(|e| io(e, &path))?.is_file() {
        return Err(invalid("path is not a regular file".into()));
    }
    let mut bytes = Vec::new();
    file.take(MAX_CONFIG_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| io(e, &path))?;
    if bytes.len() as u64 > MAX_CONFIG_BYTES {
        return Err(invalid(format!("file exceeds {MAX_CONFIG_BYTES} bytes")));
    }
    Ok((bytes, path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir(tag: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "rvllm-core-model-config-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn parses_gemma4_unified_top_level_dtype() {
        let cfg = serde_json::json!({
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "dtype": "bfloat16",
            "tie_word_embeddings": true,
            "text_config": {
                "hidden_size": 3840,
                "num_hidden_layers": 48,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 256,
                "intermediate_size": 15360,
                "vocab_size": 262144,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "sliding_attention": {"rope_theta": 10000.0}
                }
            }
        });
        let path = Path::new("config.json");

        let parsed = ModelConfig::from_hf_value(&cfg, path).unwrap();

        assert_eq!(parsed.architecture, ModelArch::Gemma4);
        assert_eq!(parsed.hidden_size, 3840);
        assert_eq!(parsed.num_layers, 48);
        assert_eq!(parsed.head_dim, 256);
        assert_eq!(parsed.torch_dtype, DType::Bf16);
    }

    #[cfg(unix)]
    #[test]
    fn load_rejects_config_symlink_escape() {
        use std::os::unix::fs::symlink;

        let root = tempdir("symlink-root");
        let outside = tempdir("symlink-outside").join("outside.json");
        std::fs::write(&outside, b"{}").unwrap();
        symlink(&outside, root.join("config.json")).unwrap();

        let error = ModelConfig::load_hf(&root).unwrap_err();
        assert!(format!("{error}").contains("path escapes"));
    }

    #[test]
    fn load_rejects_oversized_sparse_config() {
        let root = tempdir("oversized");
        std::fs::File::create(root.join("config.json"))
            .unwrap()
            .set_len(MAX_CONFIG_BYTES + 1)
            .unwrap();

        let error = ModelConfig::load_hf(&root).unwrap_err();
        assert!(format!("{error}").contains("exceeds"));
    }
}
