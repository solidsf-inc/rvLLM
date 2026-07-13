// Ported from huggingface/transformers revision
// 10555512868d663ee1ff627e4f5c5c260114235b:
// src/transformers/models/gemma4/modular_gemma4.py
// Apache-2.0 License, Copyright (c) HuggingFace and Google.
// Source: weight-key convention for `vision_tower` + `embed_vision` as
// emitted by HF's `Gemma4ForConditionalGeneration.state_dict()`.
// Modifications: Rust safetensors -> candle VarBuilder bridge for rvllm.

//! Bounded Gemma 4 safetensors loader for the vision tower and multimodal
//! projector under `model.vision_tower.*` and `model.embed_vision.*`.

use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::config::Gemma4Config;
use crate::embedder::Gemma4MultimodalEmbedder;
use crate::model::Gemma4VisionModel;

/// HF prefix for the vision tower in a Gemma 4 checkpoint.
const VISION_TOWER_PREFIX: &str = "model.vision_tower";

/// HF prefix for the multimodal (vision -> text) embedder.
const EMBED_VISION_PREFIX: &str = "model.embed_vision";

/// Sentinel keys probed before building the candle modules. If any of
/// these are missing the checkpoint is not a Gemma 4 multimodal model
/// (or is partially downloaded) and we hard-fail rather than letting
/// `VarBuilder::get` produce a confusing error deep inside the
/// encoder construction.
const VISION_SENTINEL_KEYS: &[&str] = &[
    "model.vision_tower.patch_embedder.input_proj.weight",
    "model.vision_tower.patch_embedder.position_embedding_table",
    "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
];

const EMBEDDER_SENTINEL_KEYS: &[&str] = &["model.embed_vision.embedding_projection.weight"];
const MAX_SHARDS: usize = 1_024;
const MAX_SAFETENSORS_HEADER: u64 = 100 * 1024 * 1024;
const MAX_SAFETENSORS_HEADER_BYTES_TOTAL: u64 = 256 * 1024 * 1024;
const MAX_TENSOR_ENTRIES: usize = 65_536;
const MAX_TENSOR_NAME_BYTES_TOTAL: usize = 16 * 1024 * 1024;
const MAX_SELECTED_TENSORS: usize = 4_096;
const MAX_SELECTED_TENSOR_BYTES: u64 = 256 * 1024 * 1024;
const MAX_SELECTED_TENSOR_BYTES_TOTAL: u64 = 4 * 1024 * 1024 * 1024;

/// Load the Gemma 4 vision tower + multimodal embedder from a
/// directory of safetensors shards.
///
/// `weights_dir` must contain one or more `*.safetensors` files
/// (typical layout: `model-00001-of-00002.safetensors`,
/// `model-00002-of-00002.safetensors`). Headers and tensor extents are
/// validated before selected vision tensors are copied into owned candle
/// tensors and exposed through a single [`candle_nn::VarBuilder`].
///
/// Hard-fails (no silent fallback) if:
///   - `weights_dir` does not exist or is not a directory;
///   - it contains zero `*.safetensors` files;
///   - any sentinel key is missing (the checkpoint is not a Gemma 4
///     multimodal model, or a shard is missing);
///   - `text_config.hidden_size` is missing from `cfg` (needed to
///     size the multimodal projector);
///   - `Gemma4VisionModel::new` or `Gemma4MultimodalEmbedder::new`
///     returns a candle error (a
///     specific expected tensor is missing or has the wrong shape).
pub fn load_vision(
    weights_dir: &Path,
    cfg: &Gemma4Config,
    device: &Device,
    dtype: DType,
) -> anyhow::Result<(Gemma4VisionModel, Gemma4MultimodalEmbedder)> {
    if !weights_dir.is_dir() {
        bail!(
            "rvllm-vision loader: weights_dir does not exist or is not a directory: {}",
            weights_dir.display()
        );
    }
    cfg.validate()?;
    let weights_dir = weights_dir
        .canonicalize()
        .with_context(|| format!("canonicalize {}", weights_dir.display()))?;
    let shards = collect_safetensors_shards(&weights_dir)?;
    if shards.is_empty() {
        bail!(
            "rvllm-vision loader: no *.safetensors files in {}",
            weights_dir.display()
        );
    }
    let signatures = file_signatures(&shards)?;
    let (shapes, locations) = inspect_shard_headers(&shards, cfg)?;
    validate_sentinel_shapes(cfg, &shapes)?;
    let tensors = load_selected_tensors(&locations, device)?;
    let vb = VarBuilder::from_tensors(tensors, dtype, device);
    for key in VISION_SENTINEL_KEYS {
        if !vb.contains_tensor(key) {
            bail!(
                "rvllm-vision loader: required key `{key}` not present in \
                 safetensors under {}. This is not a Gemma 4 multimodal \
                 checkpoint, or the vision shard is missing.",
                weights_dir.display(),
            );
        }
    }
    for key in EMBEDDER_SENTINEL_KEYS {
        if !vb.contains_tensor(key) {
            bail!(
                "rvllm-vision loader: required key `{key}` not present in \
                 safetensors under {}. The multimodal embedder weights \
                 are missing.",
                weights_dir.display(),
            );
        }
    }
    let text_hidden_size = cfg
        .text_config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            anyhow!(
                "rvllm-vision loader: text_config.hidden_size missing from \
                 Gemma4Config — needed to size the multimodal embedder \
                 projection."
            )
        })?;
    let text_hidden_size = usize::try_from(text_hidden_size)
        .map_err(|_| anyhow!("text_config.hidden_size does not fit usize"))?;
    let vision_vb = vb.pp(VISION_TOWER_PREFIX);
    let model = Gemma4VisionModel::new(&cfg.vision_config, vision_vb).with_context(|| {
        format!(
            "Gemma4VisionModel::new failed at prefix `{VISION_TOWER_PREFIX}` \
                 with weights from {}",
            weights_dir.display()
        )
    })?;
    let embedder_vb = vb.pp(EMBED_VISION_PREFIX);
    let embedder = Gemma4MultimodalEmbedder::new(&cfg.vision_config, text_hidden_size, embedder_vb)
        .with_context(|| {
            format!(
                "Gemma4MultimodalEmbedder::new failed at prefix \
             `{EMBED_VISION_PREFIX}` with weights from {}",
                weights_dir.display()
            )
        })?;
    if file_signatures(&shards)? != signatures {
        bail!("rvllm-vision loader: a safetensors shard changed while it was being read");
    }
    Ok((model, embedder))
}

/// Collect every `*.safetensors` file in `dir` (non-recursive),
/// sorted lexically so that sharded checkpoints
/// (`model-00001-of-00002.safetensors`, ...) load in order.
fn collect_safetensors_shards(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut shards: Vec<PathBuf> = Vec::new();
    let read_dir = std::fs::read_dir(dir).with_context(|| {
        format!(
            "rvllm-vision loader: cannot read directory {}",
            dir.display()
        )
    })?;
    for entry in read_dir {
        let entry = entry.with_context(|| {
            format!(
                "rvllm-vision loader: error iterating directory {}",
                dir.display()
            )
        })?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let canonical = path
                .canonicalize()
                .with_context(|| format!("canonicalize shard {}", path.display()))?;
            if !canonical.starts_with(dir) {
                bail!(
                    "safetensors shard escapes weights directory: {}",
                    path.display()
                );
            }
            let metadata = canonical.metadata()?;
            if !metadata.is_file() {
                bail!(
                    "safetensors shard is not a regular file: {}",
                    path.display()
                );
            }
            shards.push(canonical);
            if shards.len() > MAX_SHARDS {
                bail!(
                    "more than {MAX_SHARDS} safetensors shards in {}",
                    dir.display()
                );
            }
        }
    }
    shards.sort();
    Ok(shards)
}

fn file_signatures(paths: &[PathBuf]) -> anyhow::Result<Vec<(u64, std::time::SystemTime)>> {
    paths
        .iter()
        .map(|path| {
            let metadata = path.metadata()?;
            Ok((metadata.len(), metadata.modified()?))
        })
        .collect()
}

#[derive(Debug)]
struct TensorLocation {
    name: String,
    path: PathBuf,
    offset: u64,
    len: usize,
    dtype: DType,
    shape: Vec<usize>,
}

#[derive(Debug, Default)]
struct SelectedTensorBudget {
    count: usize,
    bytes: u64,
}

#[derive(Debug, Default)]
struct HeaderBudget {
    bytes: u64,
    tensor_entries: usize,
    tensor_name_bytes: usize,
}

impl HeaderBudget {
    fn include_header(&mut self, bytes: u64) -> anyhow::Result<()> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("safetensors header byte total overflow"))?;
        if self.bytes > MAX_SAFETENSORS_HEADER_BYTES_TOTAL {
            bail!(
                "safetensors header bytes {} exceed {}",
                self.bytes,
                MAX_SAFETENSORS_HEADER_BYTES_TOTAL
            );
        }
        Ok(())
    }
    fn include_tensor(&mut self, name: &str) -> anyhow::Result<()> {
        self.tensor_entries = self
            .tensor_entries
            .checked_add(1)
            .ok_or_else(|| anyhow!("safetensors tensor entry count overflow"))?;
        if self.tensor_entries > MAX_TENSOR_ENTRIES {
            bail!(
                "safetensors tensor entries {} exceed {MAX_TENSOR_ENTRIES}",
                self.tensor_entries
            );
        }
        self.tensor_name_bytes = self
            .tensor_name_bytes
            .checked_add(name.len())
            .ok_or_else(|| anyhow!("safetensors tensor-name byte total overflow"))?;
        if self.tensor_name_bytes > MAX_TENSOR_NAME_BYTES_TOTAL {
            bail!(
                "safetensors tensor-name bytes {} exceed {MAX_TENSOR_NAME_BYTES_TOTAL}",
                self.tensor_name_bytes
            );
        }
        Ok(())
    }
}

impl SelectedTensorBudget {
    fn include(&mut self, name: &str, bytes: u64) -> anyhow::Result<()> {
        if bytes == 0 || bytes > MAX_SELECTED_TENSOR_BYTES {
            bail!(
                "selected tensor `{name}` has {bytes} bytes; limit is {MAX_SELECTED_TENSOR_BYTES}"
            );
        }
        let count = self
            .count
            .checked_add(1)
            .ok_or_else(|| anyhow!("selected tensor count overflow"))?;
        if count > MAX_SELECTED_TENSORS {
            bail!("selected tensor count {count} exceeds {MAX_SELECTED_TENSORS}");
        }
        let total = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("selected tensor byte total overflow"))?;
        if total > MAX_SELECTED_TENSOR_BYTES_TOTAL {
            bail!("selected tensor bytes {total} exceed {MAX_SELECTED_TENSOR_BYTES_TOTAL}");
        }
        self.count = count;
        self.bytes = total;
        Ok(())
    }
}

fn inspect_shard_headers(
    paths: &[PathBuf],
    cfg: &Gemma4Config,
) -> anyhow::Result<(HashMap<String, Vec<usize>>, Vec<TensorLocation>)> {
    let mut seen = HashSet::new();
    let mut shapes = HashMap::new();
    let mut locations = Vec::new();
    let mut selected_budget = SelectedTensorBudget::default();
    let mut header_budget = HeaderBudget::default();
    let final_layer_sentinel = format!(
        "model.vision_tower.encoder.layers.{}.self_attn.q_proj.linear.weight",
        cfg.vision_config.num_hidden_layers - 1
    );
    for path in paths {
        let mut file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)
            .with_context(|| format!("read safetensors prefix {}", path.display()))?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER {
            bail!(
                "invalid safetensors header length {header_len} in {}",
                path.display()
            );
        }
        header_budget.include_header(header_len)?;
        let data_start = 8u64
            .checked_add(header_len)
            .ok_or_else(|| anyhow!("safetensors header offset overflow"))?;
        if data_start > file_len {
            bail!(
                "safetensors header exceeds file length in {}",
                path.display()
            );
        }
        let mut header = vec![0u8; usize::try_from(header_len)?];
        file.read_exact(&mut header)
            .with_context(|| format!("read safetensors header {}", path.display()))?;
        let object: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
            .with_context(|| format!("parse safetensors header {}", path.display()))?;
        let data_len = file_len - data_start;
        let mut extents = Vec::new();
        for (name, tensor) in object {
            if name == "__metadata__" {
                continue;
            }
            header_budget.include_tensor(&name)?;
            if !seen.insert(name.clone()) {
                bail!("duplicate safetensors tensor key `{name}` across shards");
            }
            let tensor = tensor
                .as_object()
                .ok_or_else(|| anyhow!("tensor `{name}` metadata is not an object"))?;
            let shape: Vec<usize> = tensor
                .get("shape")
                .and_then(|value| value.as_array())
                .ok_or_else(|| anyhow!("tensor `{name}` has no shape"))?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .ok_or_else(|| anyhow!("tensor `{name}` has an invalid shape"))
                        .and_then(|dim| usize::try_from(dim).map_err(Into::into))
                })
                .collect::<anyhow::Result<_>>()?;
            let element_count = shape.iter().try_fold(1usize, |product, dim| {
                if *dim == 0 {
                    return Err(anyhow!("tensor `{name}` has a zero-sized dimension"));
                }
                product
                    .checked_mul(*dim)
                    .ok_or_else(|| anyhow!("tensor `{name}` shape product overflow"))
            })?;
            let safetensors_dtype: safetensors::Dtype = serde_json::from_value(
                tensor
                    .get("dtype")
                    .cloned()
                    .ok_or_else(|| anyhow!("tensor `{name}` has no dtype"))?,
            )
            .with_context(|| format!("tensor `{name}` has an unsupported dtype"))?;
            let offsets = tensor
                .get("data_offsets")
                .and_then(|value| value.as_array())
                .filter(|value| value.len() == 2)
                .ok_or_else(|| anyhow!("tensor `{name}` has invalid data_offsets"))?;
            let start = offsets[0]
                .as_u64()
                .ok_or_else(|| anyhow!("tensor `{name}` has invalid start offset"))?;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| anyhow!("tensor `{name}` has invalid end offset"))?;
            if start > end || end > data_len {
                bail!("tensor `{name}` extent [{start}, {end}) exceeds shard data");
            }
            let expected_len = element_count
                .checked_mul(safetensors_dtype.bitsize())
                .and_then(|bits| bits.checked_add(7))
                .map(|bits| bits / 8)
                .ok_or_else(|| anyhow!("tensor `{name}` byte length overflow"))?;
            if u64::try_from(expected_len)? != end - start {
                bail!(
                    "tensor `{name}` extent has {} bytes, expected {expected_len}",
                    end - start
                );
            }
            if name.starts_with(VISION_TOWER_PREFIX) || name.starts_with(EMBED_VISION_PREFIX) {
                let dtype = DType::try_from(safetensors_dtype).with_context(|| {
                    format!("tensor `{name}` uses a dtype unsupported by candle")
                })?;
                selected_budget.include(&name, u64::try_from(expected_len)?)?;
                locations.push(TensorLocation {
                    name: name.clone(),
                    path: path.clone(),
                    offset: data_start
                        .checked_add(start)
                        .ok_or_else(|| anyhow!("tensor `{name}` absolute offset overflow"))?,
                    len: expected_len,
                    dtype,
                    shape: shape.clone(),
                });
            }
            extents.push((start, end));
            if VISION_SENTINEL_KEYS.contains(&name.as_str())
                || EMBEDDER_SENTINEL_KEYS.contains(&name.as_str())
                || name == final_layer_sentinel
            {
                shapes.insert(name, shape);
            }
        }
        extents.sort_by_key(|extent| extent.0);
        for pair in extents.windows(2) {
            if pair[0].1 > pair[1].0 {
                bail!(
                    "overlapping safetensors extents [{}, {}) and [{}, {}) in {}",
                    pair[0].0,
                    pair[0].1,
                    pair[1].0,
                    pair[1].1,
                    path.display()
                );
            }
        }
    }
    Ok((shapes, locations))
}

fn load_selected_tensors(
    locations: &[TensorLocation],
    device: &Device,
) -> anyhow::Result<HashMap<String, candle_core::Tensor>> {
    validate_selected_tensor_budget(locations)?;
    let mut tensors = HashMap::with_capacity(locations.len());
    for location in locations {
        let mut file = std::fs::File::open(&location.path)
            .with_context(|| format!("open shard {}", location.path.display()))?;
        file.seek(SeekFrom::Start(location.offset))
            .with_context(|| format!("seek to tensor `{}`", location.name))?;
        let mut bytes = vec![0u8; location.len];
        file.read_exact(&mut bytes)
            .with_context(|| format!("read tensor `{}`", location.name))?;
        let tensor =
            candle_core::Tensor::from_raw_buffer(&bytes, location.dtype, &location.shape, device)
                .with_context(|| format!("decode tensor `{}`", location.name))?;
        if tensors.insert(location.name.clone(), tensor).is_some() {
            bail!("duplicate selected tensor `{}`", location.name);
        }
    }
    Ok(tensors)
}

fn validate_selected_tensor_budget(locations: &[TensorLocation]) -> anyhow::Result<()> {
    let mut budget = SelectedTensorBudget::default();
    for location in locations {
        budget.include(&location.name, u64::try_from(location.len)?)?;
    }
    Ok(())
}

fn validate_sentinel_shapes(
    cfg: &Gemma4Config,
    shapes: &HashMap<String, Vec<usize>>,
) -> anyhow::Result<()> {
    let vision = &cfg.vision_config;
    let text_hidden = cfg.text_config["hidden_size"]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| anyhow!("text_config.hidden_size is invalid"))?;
    let patch_features = 3usize
        .checked_mul(vision.patch_size)
        .and_then(|value| value.checked_mul(vision.patch_size))
        .ok_or_else(|| anyhow!("patch feature count overflow"))?;
    for (key, expected) in [
        (
            VISION_SENTINEL_KEYS[0],
            vec![vision.hidden_size, patch_features],
        ),
        (
            VISION_SENTINEL_KEYS[1],
            vec![2, vision.position_embedding_size, vision.hidden_size],
        ),
        (
            EMBEDDER_SENTINEL_KEYS[0],
            vec![text_hidden, vision.hidden_size],
        ),
    ] {
        let actual = shapes
            .get(key)
            .ok_or_else(|| anyhow!("required tensor `{key}` is absent"))?;
        if actual != &expected {
            bail!("tensor `{key}` shape {actual:?} != {expected:?}");
        }
    }
    let last_layer = format!(
        "model.vision_tower.encoder.layers.{}.self_attn.q_proj.linear.weight",
        vision.num_hidden_layers - 1
    );
    if !shapes.contains_key(&last_layer) {
        bail!("required final vision layer tensor `{last_layer}` is absent");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_missing_directory() {
        let cfg = sample_cfg();
        let device = Device::Cpu;
        let res = load_vision(
            Path::new("/definitely/does/not/exist/rvllm-vision-test"),
            &cfg,
            &device,
            DType::F32,
        );
        let err = match res {
            Ok(_) => panic!("expected error for missing directory"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("weights_dir does not exist"),
            "unexpected error: {msg}"
        );
    }
    #[test]
    fn rejects_empty_directory() {
        let tmp = tempdir();
        let cfg = sample_cfg();
        let device = Device::Cpu;
        let res = load_vision(&tmp, &cfg, &device, DType::F32);
        let err = match res {
            Ok(_) => panic!("expected error for empty directory"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no *.safetensors files"),
            "unexpected error: {msg}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }
    #[test]
    fn selected_tensor_budget_accepts_exact_byte_boundaries() {
        let mut single = SelectedTensorBudget::default();
        single.include("single", MAX_SELECTED_TENSOR_BYTES).unwrap();
        let mut aggregate = SelectedTensorBudget::default();
        let chunks = MAX_SELECTED_TENSOR_BYTES_TOTAL / MAX_SELECTED_TENSOR_BYTES;
        for index in 0..chunks {
            aggregate
                .include(&format!("tensor-{index}"), MAX_SELECTED_TENSOR_BYTES)
                .unwrap();
        }
        assert_eq!(aggregate.bytes, MAX_SELECTED_TENSOR_BYTES_TOTAL);
    }
    #[test]
    fn selected_tensor_budget_rejects_count_size_and_aggregate_excess() {
        let mut oversized = SelectedTensorBudget::default();
        assert!(oversized
            .include("oversized", MAX_SELECTED_TENSOR_BYTES + 1)
            .is_err());
        let mut count = SelectedTensorBudget::default();
        for index in 0..MAX_SELECTED_TENSORS {
            count.include(&format!("tensor-{index}"), 1).unwrap();
        }
        assert!(count.include("one-too-many", 1).is_err());
        let mut aggregate = SelectedTensorBudget::default();
        let chunks = MAX_SELECTED_TENSOR_BYTES_TOTAL / MAX_SELECTED_TENSOR_BYTES;
        for index in 0..chunks {
            aggregate
                .include(&format!("tensor-{index}"), MAX_SELECTED_TENSOR_BYTES)
                .unwrap();
        }
        assert!(aggregate.include("one-byte-too-many", 1).is_err());
    }
    #[test]
    fn selected_tensor_budget_rejects_counter_overflow() {
        let mut count = SelectedTensorBudget {
            count: usize::MAX,
            bytes: 0,
        };
        assert!(count.include("count-overflow", 1).is_err());
        let mut bytes = SelectedTensorBudget {
            count: 0,
            bytes: u64::MAX,
        };
        assert!(bytes.include("byte-overflow", 1).is_err());
    }
    #[test]
    fn header_budget_accepts_exact_boundaries() {
        let mut budget = HeaderBudget::default();
        budget
            .include_header(MAX_SAFETENSORS_HEADER_BYTES_TOTAL)
            .unwrap();
        budget.tensor_entries = MAX_TENSOR_ENTRIES - 1;
        budget
            .include_tensor(&"x".repeat(MAX_TENSOR_NAME_BYTES_TOTAL))
            .unwrap();
        assert_eq!(budget.bytes, MAX_SAFETENSORS_HEADER_BYTES_TOTAL);
        assert_eq!(budget.tensor_name_bytes, MAX_TENSOR_NAME_BYTES_TOTAL);
    }
    #[test]
    fn header_budget_rejects_aggregate_limits_and_overflow() {
        let mut headers = HeaderBudget::default();
        headers.bytes = MAX_SAFETENSORS_HEADER_BYTES_TOTAL;
        assert!(headers.include_header(1).is_err());
        headers.bytes = u64::MAX;
        assert!(headers.include_header(1).is_err());
        let mut entries = HeaderBudget::default();
        entries.tensor_entries = MAX_TENSOR_ENTRIES;
        assert!(entries.include_tensor("x").is_err());
        entries.tensor_entries = usize::MAX;
        assert!(entries.include_tensor("x").is_err());
        let mut names = HeaderBudget::default();
        names.tensor_name_bytes = MAX_TENSOR_NAME_BYTES_TOTAL;
        assert!(names.include_tensor("x").is_err());
        names.tensor_name_bytes = usize::MAX;
        assert!(names.include_tensor("x").is_err());
    }
    fn sample_cfg() -> Gemma4Config {
        let raw = r#"{"text_config":{"hidden_size":5376},"vision_config":{"hidden_size":1152,"intermediate_size":4304,"num_attention_heads":16,"num_key_value_heads":16,"num_hidden_layers":27,"head_dim":72,"patch_size":16,"position_embedding_size":10240,"pooling_kernel_size":3,"max_position_embeddings":131072,"standardize":true}}"#;
        serde_json::from_str(raw).unwrap()
    }
    fn tempdir() -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("rvllm-vision-loader-test-{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
