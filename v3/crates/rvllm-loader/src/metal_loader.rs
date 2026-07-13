// Copyright 2026 m0at <47344131+m0at@users.noreply.github.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Lazy per-layer Metal weight loader.
//!
//! This module keeps safetensors shards `mmap`'d until
//! Metal allocates) and materializes each transformer layer's weights
//! into `MTLBuffer`s only on demand, behind an LRU. The
//! `max_cached_layers` setting bounds resident decoder layers.
//!
//! Global weights (`embed_tokens`, `lm_head`, final norm, RoPE tables
//! upstream) are loaded once via `load_global_weights` and pinned for
//! the lifetime of the process.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rvllm_core::{DType, LoaderCtx, LoaderError, Result, RvllmError};

use lru::LruCache;
use parking_lot::Mutex;
use rvllm_metal::MetalDevice;

const DEFAULT_MAX_CACHED_LAYERS: usize = 4;
const ENV_MAX_CACHED_LAYERS: &str = "RVLLM_METAL_MAX_CACHED_LAYERS";
const ENV_PREFETCH_LAYERS: &str = "RVLLM_METAL_PREFETCH_LAYERS";
const MAX_TOTAL_SHARD_BYTES: u64 = 1 << 40;

/// Residency policy for the Apple Silicon tiered-weight path.
///
/// The model stays in HF safetensors on NVMe via `mmap`. Global tensors are
/// pinned for the process lifetime; decoder layers are copied into shared
/// Metal buffers on demand and kept behind an LRU bounded by this policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalWeightCachePolicy {
    pub max_cached_layers: usize,
    pub prefetch_layers: usize,
}

impl Default for MetalWeightCachePolicy {
    fn default() -> Self {
        Self {
            max_cached_layers: DEFAULT_MAX_CACHED_LAYERS,
            prefetch_layers: 0,
        }
    }
}

impl MetalWeightCachePolicy {
    pub fn from_env(num_layers: usize) -> Self {
        let mut p = Self::default();
        if let Some(v) = read_usize_env(ENV_MAX_CACHED_LAYERS) {
            p.max_cached_layers = v.max(1);
        }
        if let Some(v) = read_usize_env(ENV_PREFETCH_LAYERS) {
            p.prefetch_layers = v.min(num_layers).min(p.max_cached_layers);
        }
        p
    }
}

/// Where a single tensor lives across the mmap'd shard set.
#[derive(Clone, Debug)]
pub struct TensorLocation {
    pub shard_idx: usize,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

/// Borrowed CPU view of a tensor in the mmap'd safetensors backing store.
pub struct HostTensorView<'a> {
    pub bytes: &'a [u8],
    pub dtype: DType,
    pub shape: &'a [usize],
}

/// Metal view of a tensor inside a whole-shard mmap-backed buffer.
#[derive(Clone)]
pub struct MetalTensorView {
    pub buffer: Arc<metal::Buffer>,
    pub byte_offset: u64,
    pub dtype: DType,
    pub shape: Arc<[usize]>,
}

/// One transformer layer's worth of Metal buffers.
///
/// Names match the HF Gemma 4 safetensors keys (under
/// `model.language_model.layers.<i>.*`). `*_scale` buffers are populated
/// only for the FP8-Dynamic checkpoint which ships per-channel BF16
/// scales next to each `F8_E4M3` linear weight; on a pre-quantized
/// checkpoint without scales (RedHatAI default) they are `None` and the
/// dequant kernel falls back to a per-tensor scalar at dispatch.
pub struct CachedMetalLayerWeights {
    pub q_proj: metal::Buffer,
    pub k_proj: metal::Buffer,
    pub v_proj: metal::Buffer,
    pub o_proj: metal::Buffer,
    pub q_norm: metal::Buffer,
    pub k_norm: metal::Buffer,
    pub gate_proj: metal::Buffer,
    pub up_proj: metal::Buffer,
    pub down_proj: metal::Buffer,
    pub input_layernorm: metal::Buffer,
    pub post_attention_layernorm: metal::Buffer,
    pub pre_feedforward_layernorm: metal::Buffer,
    pub post_feedforward_layernorm: metal::Buffer,
    pub layer_scalar: metal::Buffer,
    pub q_proj_scale: Option<metal::Buffer>,
    pub k_proj_scale: Option<metal::Buffer>,
    pub v_proj_scale: Option<metal::Buffer>,
    pub o_proj_scale: Option<metal::Buffer>,
    pub gate_proj_scale: Option<metal::Buffer>,
    pub up_proj_scale: Option<metal::Buffer>,
    pub down_proj_scale: Option<metal::Buffer>,
}

/// Always-resident weights — small enough to keep around forever.
pub struct GlobalMetalWeights {
    pub embed_tokens: metal::Buffer,
    pub embed_tokens_shape: Vec<usize>,
    pub lm_head: metal::Buffer,
    pub lm_head_shape: Vec<usize>,
    pub final_norm: metal::Buffer,
    pub final_norm_shape: Vec<usize>,
}

/// Lazy weight cache backed by mmap'd safetensors shards and a
/// per-layer LRU of materialized Metal buffers.
pub struct MetalWeightCache {
    device: Arc<MetalDevice>,
    mmap_shards: Vec<memmap2::Mmap>,
    tensor_index: HashMap<String, TensorLocation>,
    layer_cache: Mutex<LruCache<usize, Arc<CachedMetalLayerWeights>>>,
    shard_buffers: Mutex<Vec<Option<Arc<metal::Buffer>>>>,
    weight_prefix: String,
    max_cached_layers: usize,
}

impl MetalWeightCache {
    /// Build a cache directly from a Hugging Face safetensors model directory.
    ///
    /// This is the rvLLM-native Apple Silicon path: no GGUF conversion and no
    /// Ollama adapter. `config.json` remains the source of architecture truth,
    /// and `model.safetensors.index.json` / shard headers remain the weight map.
    pub fn from_dir(
        model_dir: &Path,
        device: Arc<MetalDevice>,
        policy: MetalWeightCachePolicy,
    ) -> Result<Self> {
        let idx = crate::safetensors::ShardIndex::resolve(model_dir)?;
        let cache = Self::new(idx.shards, device, policy.max_cached_layers)?;
        cache.prefetch_layers(policy.prefetch_layers)?;
        Ok(cache)
    }

    /// Same as [`Self::from_dir`], with residency controlled by:
    ///
    /// - `RVLLM_METAL_MAX_CACHED_LAYERS` (default 4)
    /// - `RVLLM_METAL_PREFETCH_LAYERS` (default 0)
    pub fn from_dir_env(
        model_dir: &Path,
        device: Arc<MetalDevice>,
        num_layers: usize,
    ) -> Result<Self> {
        Self::from_dir(
            model_dir,
            device,
            MetalWeightCachePolicy::from_env(num_layers),
        )
    }

    /// Open every shard via `mmap` and build a string-keyed tensor
    /// index covering all shards. No GPU memory is touched here — that
    /// happens lazily in `get_layer_weights`.
    pub fn new(
        shards: Vec<PathBuf>,
        device: Arc<MetalDevice>,
        max_cached_layers: usize,
    ) -> Result<Self> {
        if shards.is_empty() {
            return Err(loader_corrupt(
                PathBuf::new(),
                "MetalWeightCache::new called with empty shard list".into(),
            ));
        }
        let cap = NonZeroUsize::new(max_cached_layers.max(1)).expect("constant lower bound");

        let first = shards[0].canonicalize().map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: shards[0].clone(),
            source,
        })?;
        let root = first
            .parent()
            .ok_or_else(|| {
                loader_corrupt(first.clone(), "first shard has no parent directory".into())
            })?
            .to_path_buf();
        let mut canonical_shards = Vec::with_capacity(shards.len());
        for path in shards {
            let canonical = path.canonicalize().map_err(|source| RvllmError::Io {
                err: rvllm_core::IoError::from(&source),
                path: path.clone(),
                source,
            })?;
            if !canonical.starts_with(&root) || !canonical.is_file() {
                return Err(loader_corrupt(
                    canonical,
                    "shard resolves outside the model directory".into(),
                ));
            }
            canonical_shards.push(canonical);
        }

        let mut mmap_shards: Vec<memmap2::Mmap> = Vec::with_capacity(canonical_shards.len());
        let mut tensor_index: HashMap<String, TensorLocation> = HashMap::new();
        let mut total_bytes = 0u64;

        for (shard_idx, path) in canonical_shards.iter().enumerate() {
            let file = std::fs::File::open(path).map_err(|source| RvllmError::Io {
                err: rvllm_core::IoError::from(&source),
                path: path.clone(),
                source,
            })?;
            let shard_bytes = file
                .metadata()
                .map_err(|source| RvllmError::Io {
                    err: rvllm_core::IoError::from(&source),
                    path: path.clone(),
                    source,
                })?
                .len();
            total_bytes = total_bytes
                .checked_add(shard_bytes)
                .ok_or_else(|| loader_corrupt(path.clone(), "total shard size overflow".into()))?;
            if total_bytes > MAX_TOTAL_SHARD_BYTES {
                return Err(loader_corrupt(
                    path.clone(),
                    format!("total shard bytes exceed {MAX_TOTAL_SHARD_BYTES}"),
                ));
            }
            // SAFETY: we hold the file mapping for the whole cache
            // lifetime and never mutate it; safetensors shards are
            // immutable in the deploy layout (SHA-pinned tarball).
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| RvllmError::Io {
                err: rvllm_core::IoError::from(&source),
                path: path.clone(),
                source,
            })?;

            index_shard(&mmap, shard_idx, path, &mut tensor_index)?;
            mmap_shards.push(mmap);
        }

        // Auto-detect HF weight prefix. Gemma 4 multimodal ships text
        // weights under `model.language_model.*`; text-only checkpoints
        // (rare for Gemma 4) use `model.*`. Probe in that order.
        let weight_prefix = if tensor_index.contains_key("model.language_model.embed_tokens.weight")
        {
            "model.language_model".to_string()
        } else if tensor_index.contains_key("model.embed_tokens.weight") {
            "model".to_string()
        } else {
            return Err(loader_corrupt(
                canonical_shards[0].clone(),
                "neither model.language_model.embed_tokens.weight nor \
                 model.embed_tokens.weight found in shard index"
                    .into(),
            ));
        };

        Ok(Self {
            device,
            shard_buffers: Mutex::new(vec![None; mmap_shards.len()]),
            mmap_shards,
            tensor_index,
            layer_cache: Mutex::new(LruCache::new(cap)),
            weight_prefix,
            max_cached_layers: max_cached_layers.max(1),
        })
    }

    /// Materialize the small set of weights that must stay resident.
    pub fn load_global_weights(&self) -> Result<GlobalMetalWeights> {
        let embed_key = format!("{}.embed_tokens.weight", self.weight_prefix);
        let norm_key = format!("{}.norm.weight", self.weight_prefix);
        let lm_head_key = "lm_head.weight";

        let (embed_buf, embed_shape) = self.upload_tensor(&embed_key)?;
        let (norm_buf, norm_shape) = self.upload_tensor(&norm_key)?;
        let (lm_head_buf, lm_head_shape) = self
            .upload_tensor_opt(lm_head_key)?
            .unwrap_or_else(|| (embed_buf.clone(), embed_shape.clone()));

        Ok(GlobalMetalWeights {
            embed_tokens: embed_buf,
            embed_tokens_shape: embed_shape,
            lm_head: lm_head_buf,
            lm_head_shape,
            final_norm: norm_buf,
            final_norm_shape: norm_shape,
        })
    }

    /// Hit the LRU; on miss, copy every tensor for layer `layer_idx`
    /// into newly allocated `MTLStorageModeShared` buffers and insert.
    pub fn get_layer_weights(&self, layer_idx: usize) -> Result<Arc<CachedMetalLayerWeights>> {
        {
            let mut cache = self.layer_cache.lock();
            if let Some(hit) = cache.get(&layer_idx) {
                return Ok(hit.clone());
            }
        }

        let layer = self.materialize_layer(layer_idx)?;
        let layer = Arc::new(layer);

        let mut cache = self.layer_cache.lock();
        cache.put(layer_idx, layer.clone());
        Ok(layer)
    }

    /// Manual eviction hook for upstream scheduling.
    pub fn evict_layer(&self, layer_idx: usize) {
        let mut cache = self.layer_cache.lock();
        cache.pop(&layer_idx);
    }

    /// Honor `RVLLM_METAL_PREFETCH_LAYERS=N` by materializing layers
    /// `0..min(N, num_layers)` upfront. The runtime calls this right
    /// after `load_global_weights`.
    pub fn prefetch_from_env(&self, num_layers: usize) -> Result<()> {
        let n = read_usize_env(ENV_PREFETCH_LAYERS)
            .unwrap_or(0)
            .min(num_layers)
            .min(self.max_cached_layers);
        self.prefetch_layers(n)
    }

    pub fn prefetch_layers(&self, n: usize) -> Result<()> {
        for i in 0..n {
            let _ = self.get_layer_weights(i)?;
        }
        Ok(())
    }

    pub fn resident_layer_count(&self) -> usize {
        self.layer_cache.lock().len()
    }

    pub fn max_cached_layers(&self) -> usize {
        self.max_cached_layers
    }

    pub fn weight_prefix(&self) -> &str {
        &self.weight_prefix
    }

    pub fn contains(&self, key: &str) -> bool {
        self.tensor_index.contains_key(key)
    }

    pub fn tensor_view(&self, key: &str) -> Result<Option<HostTensorView<'_>>> {
        let Some(loc) = self.tensor_index.get(key) else {
            return Ok(None);
        };
        let shard = &self.mmap_shards[loc.shard_idx];
        let start = loc.byte_offset as usize;
        let end = start.checked_add(loc.byte_length as usize).ok_or_else(|| {
            loader_corrupt(
                PathBuf::new(),
                format!("{key}: byte_offset + byte_length overflows usize"),
            )
        })?;
        if end > shard.len() {
            return Err(loader_corrupt(
                PathBuf::new(),
                format!(
                    "{key}: tensor range {start}..{end} exceeds shard size {}",
                    shard.len()
                ),
            ));
        }
        Ok(Some(HostTensorView {
            bytes: &shard[start..end],
            dtype: loc.dtype,
            shape: &loc.shape,
        }))
    }

    pub fn require_tensor_view(&self, key: &str) -> Result<HostTensorView<'_>> {
        self.tensor_view(key)?.ok_or_else(|| RvllmError::Loader {
            err: LoaderError::MissingTensor {
                name: key.to_string(),
            },
            ctx: LoaderCtx {
                path: PathBuf::new(),
                tensor: Some(key.to_string()),
            },
            bt: std::backtrace::Backtrace::capture(),
        })
    }

    pub fn metal_tensor_view(&self, key: &str) -> Result<Option<MetalTensorView>> {
        let Some(loc) = self.tensor_index.get(key) else {
            return Ok(None);
        };
        let buffer = self.shard_buffer(loc.shard_idx)?;
        Ok(Some(MetalTensorView {
            buffer,
            byte_offset: loc.byte_offset,
            dtype: loc.dtype,
            shape: Arc::from(loc.shape.as_slice()),
        }))
    }

    pub fn require_metal_tensor_view(&self, key: &str) -> Result<MetalTensorView> {
        self.metal_tensor_view(key)?
            .ok_or_else(|| RvllmError::Loader {
                err: LoaderError::MissingTensor {
                    name: key.to_string(),
                },
                ctx: LoaderCtx {
                    path: PathBuf::new(),
                    tensor: Some(key.to_string()),
                },
                bt: std::backtrace::Backtrace::capture(),
            })
    }

    // ------------------------------------------------------------------
    // internals
    // ------------------------------------------------------------------

    fn shard_buffer(&self, shard_idx: usize) -> Result<Arc<metal::Buffer>> {
        {
            let buffers = self.shard_buffers.lock();
            if let Some(Some(buf)) = buffers.get(shard_idx) {
                return Ok(buf.clone());
            }
        }

        let shard = self.mmap_shards.get(shard_idx).ok_or_else(|| {
            loader_corrupt(
                PathBuf::new(),
                format!("shard index {shard_idx} exceeds shard count"),
            )
        })?;
        let buf = self.device.device().new_buffer_with_data(
            shard.as_ptr() as *const std::ffi::c_void,
            shard.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        if buf.length() < shard.len() as u64 {
            return Err(loader_corrupt(
                PathBuf::new(),
                format!(
                    "Metal shard buffer length {} < shard length {}",
                    buf.length(),
                    shard.len()
                ),
            ));
        }

        let buf = Arc::new(buf);
        let mut buffers = self.shard_buffers.lock();
        if let Some(Some(existing)) = buffers.get(shard_idx) {
            return Ok(existing.clone());
        }
        let slot = buffers.get_mut(shard_idx).ok_or_else(|| {
            loader_corrupt(
                PathBuf::new(),
                format!("shard index {shard_idx} exceeds shard buffer count"),
            )
        })?;
        *slot = Some(buf.clone());
        Ok(buf)
    }

    fn materialize_layer(&self, layer_idx: usize) -> Result<CachedMetalLayerWeights> {
        let p = &self.weight_prefix;
        let ln = |suffix: &str| format!("{p}.layers.{layer_idx}.{suffix}");

        let q_proj = self.upload_tensor(&ln("self_attn.q_proj.weight"))?.0;
        let k_proj = self.upload_tensor(&ln("self_attn.k_proj.weight"))?.0;
        // Some global-attention layers reuse K as V.
        let v_proj = match self.upload_tensor_opt(&ln("self_attn.v_proj.weight"))? {
            Some((buf, _)) => buf,
            None => k_proj.clone(),
        };
        let o_proj = self.upload_tensor(&ln("self_attn.o_proj.weight"))?.0;
        let q_norm = self.upload_tensor(&ln("self_attn.q_norm.weight"))?.0;
        let k_norm = self.upload_tensor(&ln("self_attn.k_norm.weight"))?.0;
        let gate_proj = self.upload_tensor(&ln("mlp.gate_proj.weight"))?.0;
        let up_proj = self.upload_tensor(&ln("mlp.up_proj.weight"))?.0;
        let down_proj = self.upload_tensor(&ln("mlp.down_proj.weight"))?.0;
        let input_layernorm = self.upload_tensor(&ln("input_layernorm.weight"))?.0;
        let post_attention_layernorm = self
            .upload_tensor(&ln("post_attention_layernorm.weight"))?
            .0;
        let pre_feedforward_layernorm = self
            .upload_tensor(&ln("pre_feedforward_layernorm.weight"))?
            .0;
        let post_feedforward_layernorm = self
            .upload_tensor(&ln("post_feedforward_layernorm.weight"))?
            .0;
        let layer_scalar = self.upload_tensor(&ln("layer_scalar"))?.0;

        let q_proj_scale = self
            .upload_tensor_opt(&ln("self_attn.q_proj.weight_scale"))?
            .map(|(b, _)| b);
        let k_proj_scale = self
            .upload_tensor_opt(&ln("self_attn.k_proj.weight_scale"))?
            .map(|(b, _)| b);
        let v_proj_scale = self
            .upload_tensor_opt(&ln("self_attn.v_proj.weight_scale"))?
            .map(|(b, _)| b);
        let o_proj_scale = self
            .upload_tensor_opt(&ln("self_attn.o_proj.weight_scale"))?
            .map(|(b, _)| b);
        let gate_proj_scale = self
            .upload_tensor_opt(&ln("mlp.gate_proj.weight_scale"))?
            .map(|(b, _)| b);
        let up_proj_scale = self
            .upload_tensor_opt(&ln("mlp.up_proj.weight_scale"))?
            .map(|(b, _)| b);
        let down_proj_scale = self
            .upload_tensor_opt(&ln("mlp.down_proj.weight_scale"))?
            .map(|(b, _)| b);

        Ok(CachedMetalLayerWeights {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            gate_proj,
            up_proj,
            down_proj,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            layer_scalar,
            q_proj_scale,
            k_proj_scale,
            v_proj_scale,
            o_proj_scale,
            gate_proj_scale,
            up_proj_scale,
            down_proj_scale,
        })
    }

    fn upload_tensor(&self, key: &str) -> Result<(metal::Buffer, Vec<usize>)> {
        match self.upload_tensor_opt(key)? {
            Some(v) => Ok(v),
            None => Err(RvllmError::Loader {
                err: LoaderError::MissingTensor {
                    name: key.to_string(),
                },
                ctx: LoaderCtx {
                    path: PathBuf::new(),
                    tensor: Some(key.to_string()),
                },
                bt: std::backtrace::Backtrace::capture(),
            }),
        }
    }

    fn upload_tensor_opt(&self, key: &str) -> Result<Option<(metal::Buffer, Vec<usize>)>> {
        let Some(loc) = self.tensor_index.get(key) else {
            return Ok(None);
        };
        let shard = &self.mmap_shards[loc.shard_idx];
        let start = loc.byte_offset as usize;
        let end = start.checked_add(loc.byte_length as usize).ok_or_else(|| {
            loader_corrupt(
                PathBuf::new(),
                format!("{key}: byte_offset + byte_length overflows usize"),
            )
        })?;
        if end > shard.len() {
            return Err(loader_corrupt(
                PathBuf::new(),
                format!(
                    "{key}: tensor range {start}..{end} exceeds shard size {}",
                    shard.len()
                ),
            ));
        }
        let bytes = &shard[start..end];

        // Storage-shared buffers live in unified memory; the GPU sees
        // them coherently without an explicit blit. The bytes are
        // copied (not aliased) so the mmap can be unmapped/replaced
        // independently of GPU lifetime.
        let buf = self.device.device().new_buffer_with_data(
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Ok(Some((buf, loc.shape.clone())))
    }
}

fn read_usize_env(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
}

// ----------------------------------------------------------------------
// safetensors header parsing
// ----------------------------------------------------------------------

fn index_shard(
    mmap: &memmap2::Mmap,
    shard_idx: usize,
    path: &std::path::Path,
    out: &mut HashMap<String, TensorLocation>,
) -> Result<()> {
    let (header_len, meta) = ::safetensors::tensor::SafeTensors::read_metadata(mmap.as_ref())
        .map_err(|e| {
            loader_corrupt(
                path.to_path_buf(),
                format!("safetensors::read_metadata failed: {e}"),
            )
        })?;
    // Tensor data lives after the 8-byte length prefix + header JSON.
    let payload_offset = (header_len as u64) + 8;

    for (name, info) in meta.tensors() {
        if name == "__metadata__" {
            continue;
        }
        let (start, end) = (info.data_offsets.0 as u64, info.data_offsets.1 as u64);
        let dtype = map_st_dtype(info.dtype).ok_or_else(|| {
            loader_corrupt(
                path.to_path_buf(),
                format!("{name}: unsupported dtype {:?}", info.dtype),
            )
        })?;
        let byte_offset = payload_offset.checked_add(start).ok_or_else(|| {
            loader_corrupt(path.to_path_buf(), format!("{name}: offset overflow"))
        })?;
        let byte_length = end.checked_sub(start).ok_or_else(|| {
            loader_corrupt(
                path.to_path_buf(),
                format!("{name}: end offset precedes start"),
            )
        })?;
        let expected = info
            .shape
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .and_then(|elements| elements.checked_mul(dtype.bytes()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                loader_corrupt(
                    path.to_path_buf(),
                    format!("{name}: shape byte size overflow"),
                )
            })?;
        if expected != byte_length
            || byte_offset
                .checked_add(byte_length)
                .is_none_or(|end| end > mmap.len() as u64)
        {
            return Err(loader_corrupt(
                path.to_path_buf(),
                format!("{name}: invalid tensor byte range or dtype/shape size"),
            ));
        }
        if out
            .insert(
                name,
                TensorLocation {
                    shard_idx,
                    byte_offset,
                    byte_length,
                    dtype,
                    shape: info.shape.clone(),
                },
            )
            .is_some()
        {
            return Err(loader_corrupt(
                path.to_path_buf(),
                "duplicate tensor name across shards".into(),
            ));
        }
    }
    Ok(())
}

fn map_st_dtype(d: ::safetensors::Dtype) -> Option<DType> {
    use safetensors::Dtype as S;
    Some(match d {
        S::F32 => DType::F32,
        S::F16 => DType::F16,
        S::BF16 => DType::Bf16,
        S::F8_E4M3 => DType::Fp8E4M3,
        S::F8_E5M2 => DType::Fp8E5M2,
        S::U8 => DType::U8,
        S::I32 => DType::I32,
        S::I64 => DType::I64,
        S::U32 => DType::U32,
        _ => return None,
    })
}

fn loader_corrupt(path: PathBuf, detail: String) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::Corrupt { detail },
        ctx: LoaderCtx { path, tensor: None },
        bt: std::backtrace::Backtrace::capture(),
    }
}
