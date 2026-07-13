#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::collections::HashMap;
use std::sync::Arc;

use metal::{Buffer, MTLResourceOptions};
use rayon::prelude::*;
use rvllm_core::DType;
use rvllm_metal::gemv::{Fp8GemvInput, MetalGemv, ScaleLayout};

use crate::gemma4_arch::{Gemma4Arch, Gemma4LayerType};
use crate::metal_loader::{HostTensorView, MetalWeightCache};

type HResult<T> = std::result::Result<T, String>;
const MAX_MODEL_LEN: usize = 1 << 20;
const MAX_HOST_KV_BYTES: usize = 64 * 1024 * 1024 * 1024;

pub struct Gemma4HostDecoder {
    arch: Gemma4Arch,
    max_model_len: usize,
    keys: DecoderKeys,
    kv: Vec<LayerKvCache>,
    rope_inv_freq: Vec<Vec<f32>>,
    rope_cache: Vec<RopeCache>,
    has_v_proj: Vec<Option<bool>>,
    layer_scalars: Vec<Option<f32>>,
    scratch: HostForwardScratch,
    embed_scale: f32,
    fp8_lut: [f32; 256],
    metal: Option<MetalHostAccelerator>,
    batch_gemv: bool,
    fused_mlp: bool,
    lmhead_argmax: bool,
    lmhead_nll: bool,
    metal_attention: bool,
    metal_attention_min_len: usize,
    parallel_attention_min_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct PplScore {
    pub total_nll: f64,
    pub tokens: usize,
    pub perplexity: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GenerateTiming {
    pub ttft_ms: f64,
    pub decode_ms: f64,
    pub decode_tokens: usize,
}

struct MetalHostAccelerator {
    device: Arc<rvllm_metal::MetalDevice>,
    gemv: MetalGemv,
    views: HashMap<String, crate::metal_loader::MetalTensorView>,
    kv: Vec<Option<MetalLayerKvCache>>,
}

impl MetalHostAccelerator {
    fn tensor_view(
        &mut self,
        weights: &MetalWeightCache,
        key: &str,
    ) -> HResult<crate::metal_loader::MetalTensorView> {
        if let Some(view) = self.views.get(key) {
            return Ok(view.clone());
        }
        let view = weights
            .require_metal_tensor_view(key)
            .map_err(|e| format!("{key}: {e}"))?;
        self.views.insert(key.to_owned(), view.clone());
        Ok(view)
    }

    fn ensure_layer_kv(
        &mut self,
        layer_idx: usize,
        slots: usize,
        kv_dim: usize,
    ) -> HResult<&MetalLayerKvCache> {
        if layer_idx >= self.kv.len() || slots == 0 || kv_dim == 0 {
            return Err(format!("layer {layer_idx}: invalid Metal KV geometry"));
        }
        if self.kv[layer_idx].is_none() {
            let elems = slots
                .checked_mul(kv_dim)
                .ok_or_else(|| format!("layer {layer_idx}: Metal KV element size overflow"))?;
            let bytes = elems
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| format!("layer {layer_idx}: Metal KV byte size overflow"))?;
            let bytes = u64::try_from(bytes)
                .map_err(|_| format!("layer {layer_idx}: Metal KV byte size exceeds u64"))?;
            let k = self
                .device
                .device()
                .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
            let v = self
                .device
                .device()
                .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
            if k.length() < bytes || v.length() < bytes {
                return Err(format!(
                    "layer {layer_idx}: Metal KV allocation was truncated"
                ));
            }
            self.kv[layer_idx] = Some(MetalLayerKvCache { k, v });
        }
        self.kv[layer_idx]
            .as_ref()
            .ok_or_else(|| format!("layer {layer_idx}: Metal KV allocation failed"))
    }
}

struct LayerKvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    slots: usize,
    kv_dim: usize,
}

struct MetalLayerKvCache {
    k: Buffer,
    v: Buffer,
}

struct RopeCache {
    pos: usize,
    sin_cos: Vec<(f32, f32)>,
}

#[derive(Default)]
struct HostForwardScratch {
    attn: Vec<f32>,
    attn_scores: Vec<f32>,
    attn_slots: Vec<usize>,
    metal_attn_slots: Vec<u32>,
    norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    down: Vec<f32>,
}

struct DecoderKeys {
    embed_tokens: Arc<str>,
    final_norm: Arc<str>,
    layers: Vec<Arc<LayerKeys>>,
}

#[derive(Clone)]
struct LayerKeys {
    input_layernorm: Arc<str>,
    q_proj: Arc<str>,
    q_proj_scale: Arc<str>,
    k_proj: Arc<str>,
    k_proj_scale: Arc<str>,
    v_proj: Arc<str>,
    v_proj_scale: Arc<str>,
    q_norm: Arc<str>,
    k_norm: Arc<str>,
    o_proj: Arc<str>,
    o_proj_scale: Arc<str>,
    post_attention_layernorm: Arc<str>,
    pre_feedforward_layernorm: Arc<str>,
    gate_proj: Arc<str>,
    gate_proj_scale: Arc<str>,
    up_proj: Arc<str>,
    up_proj_scale: Arc<str>,
    down_proj: Arc<str>,
    down_proj_scale: Arc<str>,
    layer_scalar: Arc<str>,
    post_feedforward_layernorm: Arc<str>,
}

impl DecoderKeys {
    fn new(prefix: &str, layers: usize) -> Self {
        Self {
            embed_tokens: Arc::from(format!("{prefix}.embed_tokens.weight")),
            final_norm: Arc::from(format!("{prefix}.norm.weight")),
            layers: (0..layers)
                .map(|i| {
                    let p = format!("{prefix}.layers.{i}");
                    Arc::new(LayerKeys {
                        input_layernorm: Arc::from(format!("{p}.input_layernorm.weight")),
                        q_proj: Arc::from(format!("{p}.self_attn.q_proj.weight")),
                        q_proj_scale: Arc::from(format!("{p}.self_attn.q_proj.weight_scale")),
                        k_proj: Arc::from(format!("{p}.self_attn.k_proj.weight")),
                        k_proj_scale: Arc::from(format!("{p}.self_attn.k_proj.weight_scale")),
                        v_proj: Arc::from(format!("{p}.self_attn.v_proj.weight")),
                        v_proj_scale: Arc::from(format!("{p}.self_attn.v_proj.weight_scale")),
                        q_norm: Arc::from(format!("{p}.self_attn.q_norm.weight")),
                        k_norm: Arc::from(format!("{p}.self_attn.k_norm.weight")),
                        o_proj: Arc::from(format!("{p}.self_attn.o_proj.weight")),
                        o_proj_scale: Arc::from(format!("{p}.self_attn.o_proj.weight_scale")),
                        post_attention_layernorm: Arc::from(format!(
                            "{p}.post_attention_layernorm.weight"
                        )),
                        pre_feedforward_layernorm: Arc::from(format!(
                            "{p}.pre_feedforward_layernorm.weight"
                        )),
                        gate_proj: Arc::from(format!("{p}.mlp.gate_proj.weight")),
                        gate_proj_scale: Arc::from(format!("{p}.mlp.gate_proj.weight_scale")),
                        up_proj: Arc::from(format!("{p}.mlp.up_proj.weight")),
                        up_proj_scale: Arc::from(format!("{p}.mlp.up_proj.weight_scale")),
                        down_proj: Arc::from(format!("{p}.mlp.down_proj.weight")),
                        down_proj_scale: Arc::from(format!("{p}.mlp.down_proj.weight_scale")),
                        layer_scalar: Arc::from(format!("{p}.layer_scalar")),
                        post_feedforward_layernorm: Arc::from(format!(
                            "{p}.post_feedforward_layernorm.weight"
                        )),
                    })
                })
                .collect(),
        }
    }
}

impl Gemma4HostDecoder {
    pub fn new(arch: Gemma4Arch, max_model_len: usize) -> std::result::Result<Self, String> {
        if max_model_len == 0 || max_model_len > MAX_MODEL_LEN {
            return Err(format!("max_model_len must be in 1..={MAX_MODEL_LEN}"));
        }
        let num_layers = arch.num_hidden_layers;
        if num_layers == 0 || arch.layer_types.len() != num_layers {
            return Err("architecture has invalid layer geometry".into());
        }
        if arch.num_attention_heads == 0
            || arch.num_kv_heads_sliding == 0
            || arch.num_kv_heads_global == 0
            || arch.num_attention_heads % arch.num_kv_heads_sliding != 0
            || arch.num_attention_heads % arch.num_kv_heads_global != 0
            || arch.head_dim_sliding == 0
            || arch.head_dim_global == 0
            || arch.hidden_size == 0
        {
            return Err("architecture has invalid attention geometry".into());
        }
        let mut total_kv_bytes = 0usize;
        let kv = (0..num_layers)
            .map(|layer_idx| {
                let kv_dim = arch.kv_dim_for_layer(layer_idx);
                let slots = match arch.layer_types[layer_idx] {
                    Gemma4LayerType::SlidingAttention => {
                        max_model_len.min(arch.sliding_window_size)
                    }
                    Gemma4LayerType::GlobalAttention => max_model_len,
                }
                .max(1);
                let elems = slots
                    .checked_mul(kv_dim)
                    .ok_or_else(|| format!("layer {layer_idx}: KV element count overflow"))?;
                let layer_bytes = elems
                    .checked_mul(std::mem::size_of::<f32>())
                    .and_then(|bytes| bytes.checked_mul(2))
                    .ok_or_else(|| format!("layer {layer_idx}: KV byte count overflow"))?;
                total_kv_bytes = total_kv_bytes
                    .checked_add(layer_bytes)
                    .ok_or_else(|| "total KV byte count overflow".to_string())?;
                if total_kv_bytes > MAX_HOST_KV_BYTES {
                    return Err(format!("host KV cache exceeds {MAX_HOST_KV_BYTES} bytes"));
                }
                let mut k = Vec::new();
                k.try_reserve_exact(elems)
                    .map_err(|_| format!("layer {layer_idx}: K cache allocation failed"))?;
                k.resize(elems, 0.0);
                let mut v = Vec::new();
                v.try_reserve_exact(elems)
                    .map_err(|_| format!("layer {layer_idx}: V cache allocation failed"))?;
                v.resize(elems, 0.0);
                Ok(LayerKvCache {
                    k,
                    v,
                    slots,
                    kv_dim,
                })
            })
            .collect::<HResult<Vec<_>>>()?;
        let rope_inv_freq = (0..num_layers)
            .map(|layer_idx| {
                let head_dim = arch.head_dim_for_layer(layer_idx);
                let rotary_dim = arch.rotary_dim_for_layer(layer_idx);
                let theta = arch.rope_theta_for_layer(layer_idx);
                (0..rotary_dim / 2)
                    .map(|i| theta.powf(-(2.0 * i as f32) / head_dim as f32))
                    .collect()
            })
            .collect();
        let rope_cache = (0..num_layers)
            .map(|_| RopeCache {
                pos: usize::MAX,
                sin_cos: Vec::new(),
            })
            .collect();
        let fp8_lut = std::array::from_fn(|i| fp8_e4m3_to_f32(i as u8));
        let embed_scale = (arch.hidden_size as f32).sqrt();
        Ok(Self {
            keys: DecoderKeys::new(&arch.weight_prefix, num_layers),
            arch,
            max_model_len,
            kv,
            rope_inv_freq,
            rope_cache,
            has_v_proj: vec![None; num_layers],
            layer_scalars: vec![None; num_layers],
            scratch: HostForwardScratch::default(),
            embed_scale,
            fp8_lut,
            metal: None,
            batch_gemv: env_bool_default("RVLLM_METAL_BATCH_GEMV", true),
            fused_mlp: env_bool_default("RVLLM_METAL_FUSED_MLP", true),
            lmhead_argmax: env_bool_default(
                "RVLLM_METAL_LMHEAD_ARGMAX",
                !env_bool("RVLLM_METAL_LMHEAD_FULL_LOGITS"),
            ),
            lmhead_nll: env_bool_default("RVLLM_METAL_LMHEAD_NLL", true),
            metal_attention: env_bool_default("RVLLM_METAL_ATTENTION", false),
            metal_attention_min_len: env_usize("RVLLM_METAL_ATTENTION_MIN").unwrap_or(16),
            parallel_attention_min_len: env_usize("RVLLM_METAL_PAR_ATTEND_MIN").unwrap_or(16),
        })
    }

    pub fn new_with_metal(
        arch: Gemma4Arch,
        max_model_len: usize,
        device: Arc<rvllm_metal::MetalDevice>,
        kernels: Arc<rvllm_metal::MetalKernels>,
    ) -> std::result::Result<Self, String> {
        let mut dec = Self::new(arch, max_model_len)?;
        let layers = dec.arch.num_hidden_layers;
        dec.metal = Some(MetalHostAccelerator {
            device: device.clone(),
            gemv: MetalGemv::new(device, kernels),
            views: HashMap::new(),
            kv: (0..layers).map(|_| None).collect(),
        });
        Ok(dec)
    }

    pub fn generate(
        &mut self,
        weights: &MetalWeightCache,
        prompt_ids: &[u32],
        max_new: usize,
        stop_ids: &[u32],
    ) -> HResult<Vec<u32>> {
        self.generate_timed(weights, prompt_ids, max_new, stop_ids)
            .map(|(ids, _)| ids)
    }

    pub fn generate_timed(
        &mut self,
        weights: &MetalWeightCache,
        prompt_ids: &[u32],
        max_new: usize,
        stop_ids: &[u32],
    ) -> HResult<(Vec<u32>, GenerateTiming)> {
        if prompt_ids.is_empty() {
            return Err("Gemma4HostDecoder requires at least one prompt token".into());
        }
        let requested_len = prompt_ids
            .len()
            .checked_add(max_new)
            .ok_or_else(|| "prompt + max_new length overflow".to_string())?;
        if requested_len > self.max_model_len {
            return Err(format!(
                "prompt + max_new exceeds max_model_len: {} + {} > {}",
                prompt_ids.len(),
                max_new,
                self.max_model_len
            ));
        }

        let start = std::time::Instant::now();
        let mut hidden = Vec::new();
        for (pos, &tok) in prompt_ids.iter().enumerate() {
            self.forward_hidden_into(weights, tok, pos, &mut hidden)?;
        }
        let mut logits_in = Vec::new();
        let mut next = self.argmax_next_into(weights, &hidden, &mut logits_in)?;
        let ttft_ms = start.elapsed().as_secs_f64() * 1000.0;
        let decode_start = std::time::Instant::now();

        let mut out = Vec::with_capacity(max_new);
        for step in 0..max_new {
            let tok = next;
            out.push(tok);
            if stop_ids.contains(&tok) {
                break;
            }
            if step + 1 == max_new {
                break;
            }
            self.forward_hidden_into(weights, tok, prompt_ids.len() + step, &mut hidden)?;
            next = self.argmax_next_into(weights, &hidden, &mut logits_in)?;
        }
        let decode_tokens = out.len();
        Ok((
            out,
            GenerateTiming {
                ttft_ms,
                decode_ms: decode_start.elapsed().as_secs_f64() * 1000.0,
                decode_tokens,
            },
        ))
    }

    pub fn score_nll(
        &mut self,
        weights: &MetalWeightCache,
        token_ids: &[u32],
        score_from: usize,
    ) -> HResult<PplScore> {
        if token_ids.len() < 2 {
            return Err("score_nll requires at least two tokens".into());
        }
        if token_ids.len() > self.max_model_len {
            return Err(format!(
                "token count {} exceeds max_model_len {}",
                token_ids.len(),
                self.max_model_len
            ));
        }

        let mut total_nll = 0.0f64;
        let mut tokens = 0usize;
        let mut logits_in = Vec::new();
        let softcap = if env_bool("RVLLM_NO_SOFTCAP") {
            0.0
        } else {
            self.arch.logit_softcap
        };
        let mut hidden = Vec::new();
        for pos in 0..token_ids.len() - 1 {
            self.forward_hidden_into(weights, token_ids[pos], pos, &mut hidden)?;
            if pos < score_from {
                continue;
            }
            self.final_logits_input_into(weights, &hidden, &mut logits_in)?;
            if self.lmhead_nll {
                if let Some(nll) =
                    self.lm_head_target_nll(weights, &logits_in, token_ids[pos + 1], softcap)?
                {
                    total_nll += nll;
                    tokens += 1;
                    continue;
                }
            }
            let logits = self.lm_head_logits(weights, &logits_in)?;
            total_nll += target_nll(&logits, token_ids[pos + 1], softcap)?;
            tokens += 1;
        }
        if tokens == 0 {
            return Err(format!(
                "score_from {score_from} leaves no target tokens in {} token sequence",
                token_ids.len()
            ));
        }
        Ok(PplScore {
            total_nll,
            tokens,
            perplexity: (total_nll / tokens as f64).exp(),
        })
    }

    fn forward_hidden_into(
        &mut self,
        weights: &MetalWeightCache,
        token_id: u32,
        pos: usize,
        out: &mut Vec<f32>,
    ) -> HResult<()> {
        let mut scratch = std::mem::take(&mut self.scratch);
        let result = self.forward_hidden_with_scratch(weights, token_id, pos, out, &mut scratch);
        self.scratch = scratch;
        result
    }

    fn forward_hidden_with_scratch(
        &mut self,
        weights: &MetalWeightCache,
        token_id: u32,
        pos: usize,
        residual: &mut Vec<f32>,
        scratch: &mut HostForwardScratch,
    ) -> HResult<()> {
        let hidden = self.arch.hidden_size;
        self.embedding_into(weights, token_id, residual)?;

        for layer_idx in 0..self.arch.num_hidden_layers {
            let lk = self.keys.layers[layer_idx].clone();
            let layer_type = self.arch.layer_types[layer_idx];
            let head_dim = self.arch.head_dim_for_layer(layer_idx);
            let num_heads = self.arch.num_attention_heads;
            let num_kv_heads = self.arch.num_kv_heads_for_layer(layer_idx);
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            self.rmsnorm_into(weights, &lk.input_layernorm, residual, &mut scratch.norm)?;
            let has_v = self.has_v_proj(weights, layer_idx, &lk.v_proj);
            if self.batch_gemv {
                let qkv2 = [
                    (lk.q_proj.as_ref(), lk.q_proj_scale.as_ref()),
                    (lk.k_proj.as_ref(), lk.k_proj_scale.as_ref()),
                ];
                let qkv3 = [
                    (lk.q_proj.as_ref(), lk.q_proj_scale.as_ref()),
                    (lk.k_proj.as_ref(), lk.k_proj_scale.as_ref()),
                    (lk.v_proj.as_ref(), lk.v_proj_scale.as_ref()),
                ];
                if has_v {
                    let mut outs = [&mut scratch.q, &mut scratch.k, &mut scratch.v];
                    self.fp8_matvec_many_into(weights, &qkv3, &scratch.norm, &mut outs)?;
                } else {
                    let mut outs = [&mut scratch.q, &mut scratch.k];
                    self.fp8_matvec_many_into(weights, &qkv2, &scratch.norm, &mut outs)?;
                    scratch.v.clear();
                    scratch.v.extend_from_slice(&scratch.k);
                }
            } else {
                self.fp8_matvec_into(
                    weights,
                    &lk.q_proj,
                    &lk.q_proj_scale,
                    &scratch.norm,
                    &mut scratch.q,
                )?;
                self.fp8_matvec_into(
                    weights,
                    &lk.k_proj,
                    &lk.k_proj_scale,
                    &scratch.norm,
                    &mut scratch.k,
                )?;
                if has_v {
                    self.fp8_matvec_into(
                        weights,
                        &lk.v_proj,
                        &lk.v_proj_scale,
                        &scratch.norm,
                        &mut scratch.v,
                    )?;
                } else {
                    scratch.v.clear();
                    scratch.v.extend_from_slice(&scratch.k);
                }
            }
            if scratch.q.len() != q_dim || scratch.k.len() != kv_dim || scratch.v.len() != kv_dim {
                return Err(format!(
                    "layer {layer_idx}: q/k/v shape mismatch got {}/{}/{}, expected {}/{}/{}",
                    scratch.q.len(),
                    scratch.k.len(),
                    scratch.v.len(),
                    q_dim,
                    kv_dim,
                    kv_dim
                ));
            }

            self.rmsnorm_heads_in_place(weights, &lk.q_norm, &mut scratch.q, num_heads, head_dim)?;
            self.rmsnorm_heads_in_place(
                weights,
                &lk.k_norm,
                &mut scratch.k,
                num_kv_heads,
                head_dim,
            )?;
            let attention_scale = (head_dim as f32).sqrt().recip();
            for value in &mut scratch.q {
                *value *= attention_scale;
            }
            self.apply_rope_and_cache(
                layer_idx,
                layer_type,
                pos,
                &mut scratch.q,
                &mut scratch.k,
                &scratch.v,
            )?;
            self.attend_into(
                layer_idx,
                layer_type,
                pos,
                &scratch.q,
                &mut scratch.attn,
                &mut scratch.attn_scores,
                &mut scratch.attn_slots,
                &mut scratch.metal_attn_slots,
            )?;

            self.fp8_matvec_into(
                weights,
                &lk.o_proj,
                &lk.o_proj_scale,
                &scratch.attn,
                &mut scratch.o,
            )?;
            if scratch.o.len() != hidden {
                return Err(format!(
                    "layer {layer_idx}: o_proj produced {}, expected {hidden}",
                    scratch.o.len()
                ));
            }
            self.norm_add(
                weights,
                &lk.post_attention_layernorm,
                residual,
                &scratch.o,
                None,
            )?;

            self.rmsnorm_into(
                weights,
                &lk.pre_feedforward_layernorm,
                residual,
                &mut scratch.norm,
            )?;
            self.mlp_down_into(weights, &lk, layer_idx, &scratch.norm, &mut scratch.down)?;
            let layer_scalar = self.layer_scalar_cached(weights, layer_idx, &lk.layer_scalar)?;
            self.norm_add(
                weights,
                &lk.post_feedforward_layernorm,
                residual,
                &scratch.down,
                Some(layer_scalar),
            )?;
        }

        Ok(())
    }

    fn argmax_next_into(
        &mut self,
        weights: &MetalWeightCache,
        residual: &[f32],
        logits_in: &mut Vec<f32>,
    ) -> HResult<u32> {
        self.final_logits_input_into(weights, residual, logits_in)?;
        self.argmax_lm_head(weights, logits_in)
    }

    fn final_logits_input_into(
        &self,
        weights: &MetalWeightCache,
        residual: &[f32],
        out: &mut Vec<f32>,
    ) -> HResult<()> {
        self.rmsnorm_into(weights, &self.keys.final_norm, residual, out)
    }

    fn mlp_down_into(
        &mut self,
        weights: &MetalWeightCache,
        lk: &LayerKeys,
        layer_idx: usize,
        ff_norm: &[f32],
        out: &mut Vec<f32>,
    ) -> HResult<()> {
        if self.fused_mlp {
            if self.fp8_mlp_fused_down_into(weights, lk, layer_idx, ff_norm, out)? {
                return Ok(());
            }
        }
        let (mut gate, up) = if self.batch_gemv {
            let mut gate_up = self
                .fp8_matvec_many(
                    weights,
                    &[
                        (lk.gate_proj.as_ref(), lk.gate_proj_scale.as_ref()),
                        (lk.up_proj.as_ref(), lk.up_proj_scale.as_ref()),
                    ],
                    ff_norm,
                )?
                .into_iter();
            let gate = gate_up
                .next()
                .ok_or_else(|| format!("layer {layer_idx}: gate_proj missing from batch"))?;
            let up = gate_up
                .next()
                .ok_or_else(|| format!("layer {layer_idx}: up_proj missing from batch"))?;
            (gate, up)
        } else {
            let gate = self.fp8_matvec(weights, &lk.gate_proj, &lk.gate_proj_scale, ff_norm)?;
            let up = self.fp8_matvec(weights, &lk.up_proj, &lk.up_proj_scale, ff_norm)?;
            (gate, up)
        };
        if gate.len() != self.arch.intermediate_size || up.len() != self.arch.intermediate_size {
            return Err(format!(
                "layer {layer_idx}: gate/up shape mismatch got {}/{} expected {}",
                gate.len(),
                up.len(),
                self.arch.intermediate_size
            ));
        }
        for (g, u) in gate.iter_mut().zip(up) {
            *g = gelu_tanh(*g) * u;
        }
        self.fp8_matvec_into(weights, &lk.down_proj, &lk.down_proj_scale, &gate, out)
    }

    fn fp8_mlp_fused_down_into(
        &mut self,
        weights: &MetalWeightCache,
        lk: &LayerKeys,
        layer_idx: usize,
        ff_norm: &[f32],
        out: &mut Vec<f32>,
    ) -> HResult<bool> {
        let Some(metal) = self.metal.as_mut() else {
            return Ok(false);
        };

        let gate_weight = metal.tensor_view(weights, &lk.gate_proj)?;
        let gate_scale = metal.tensor_view(weights, &lk.gate_proj_scale)?;
        let up_weight = metal.tensor_view(weights, &lk.up_proj)?;
        let up_scale = metal.tensor_view(weights, &lk.up_proj_scale)?;
        let down_weight = metal.tensor_view(weights, &lk.down_proj)?;
        let down_scale = metal.tensor_view(weights, &lk.down_proj_scale)?;

        for (key, t) in [
            (lk.gate_proj.as_ref(), &gate_weight),
            (lk.up_proj.as_ref(), &up_weight),
            (lk.down_proj.as_ref(), &down_weight),
        ] {
            expect_metal_dtype(t, DType::Fp8E4M3, key)?;
            expect_metal_rank2(t, key)?;
        }
        for (key, t) in [
            (lk.gate_proj_scale.as_ref(), &gate_scale),
            (lk.up_proj_scale.as_ref(), &up_scale),
            (lk.down_proj_scale.as_ref(), &down_scale),
        ] {
            if !matches!(t.dtype, DType::Bf16 | DType::F32) {
                return Err(format!("{key}: dtype {:?}, expected BF16 or F32", t.dtype));
            }
        }

        let inter = self.arch.intermediate_size;
        let hidden = self.arch.hidden_size;
        if gate_weight.shape.as_ref() != [inter, ff_norm.len()] {
            return Err(format!(
                "layer {layer_idx}: gate_proj shape {:?}, expected [{inter}, {}]",
                gate_weight.shape,
                ff_norm.len()
            ));
        }
        if up_weight.shape.as_ref() != [inter, ff_norm.len()] {
            return Err(format!(
                "layer {layer_idx}: up_proj shape {:?}, expected [{inter}, {}]",
                up_weight.shape,
                ff_norm.len()
            ));
        }
        if down_weight.shape.as_ref() != [hidden, inter] {
            return Err(format!(
                "layer {layer_idx}: down_proj shape {:?}, expected [{hidden}, {inter}]",
                down_weight.shape
            ));
        }

        let Some((gate_scale_layout, gate_scale_stride)) =
            metal_scale_layout_shape(&gate_scale.shape, inter)
        else {
            return Ok(false);
        };
        let Some((up_scale_layout, up_scale_stride)) =
            metal_scale_layout_shape(&up_scale.shape, inter)
        else {
            return Ok(false);
        };
        let Some((down_scale_layout, down_scale_stride)) =
            metal_scale_layout_shape(&down_scale.shape, hidden)
        else {
            return Ok(false);
        };

        let gate = Fp8GemvInput {
            weight: gate_weight.buffer.as_ref(),
            weight_offset: gate_weight.byte_offset,
            scale: gate_scale.buffer.as_ref(),
            scale_offset: gate_scale.byte_offset,
            scale_dtype: gate_scale.dtype,
            scale_layout: gate_scale_layout,
            scale_stride: gate_scale_stride,
            rows: inter,
            cols: ff_norm.len(),
        };
        let up = Fp8GemvInput {
            weight: up_weight.buffer.as_ref(),
            weight_offset: up_weight.byte_offset,
            scale: up_scale.buffer.as_ref(),
            scale_offset: up_scale.byte_offset,
            scale_dtype: up_scale.dtype,
            scale_layout: up_scale_layout,
            scale_stride: up_scale_stride,
            rows: inter,
            cols: ff_norm.len(),
        };
        let down = Fp8GemvInput {
            weight: down_weight.buffer.as_ref(),
            weight_offset: down_weight.byte_offset,
            scale: down_scale.buffer.as_ref(),
            scale_offset: down_scale.byte_offset,
            scale_dtype: down_scale.dtype,
            scale_layout: down_scale_layout,
            scale_stride: down_scale_stride,
            rows: hidden,
            cols: inter,
        };

        metal
            .gemv
            .fp8_gelu_down_f32_into(&gate, &up, &down, ff_norm, out)
            .map(|_| true)
            .map_err(|e| format!("layer {layer_idx}: metal fused mlp: {e}"))
    }

    fn embedding_into(
        &self,
        weights: &MetalWeightCache,
        token_id: u32,
        out: &mut Vec<f32>,
    ) -> HResult<()> {
        let t = require(weights, &self.keys.embed_tokens)?;
        expect_dtype(&t, DType::Bf16, &self.keys.embed_tokens)?;
        expect_rank2(&t, &self.keys.embed_tokens)?;
        let vocab = t.shape[0];
        let hidden = t.shape[1];
        let row = token_id as usize;
        if row >= vocab {
            return Err(format!("token id {token_id} exceeds vocab size {vocab}"));
        }
        out.clear();
        out.reserve(hidden);
        out.extend((0..hidden).map(|i| bf16_at(t.bytes, row * hidden + i) * self.embed_scale));
        Ok(())
    }

    fn rmsnorm_into(
        &self,
        weights: &MetalWeightCache,
        gamma_key: &str,
        x: &[f32],
        out: &mut Vec<f32>,
    ) -> HResult<()> {
        let gamma = require(weights, gamma_key)?;
        expect_dtype(&gamma, DType::Bf16, gamma_key)?;
        if gamma.shape != [x.len()] {
            return Err(format!(
                "{gamma_key}: gamma shape {:?}, expected [{}]",
                gamma.shape,
                x.len()
            ));
        }
        let inv = rms_inv(x, self.arch.rms_norm_eps);
        out.clear();
        out.reserve(x.len());
        out.extend(
            x.iter()
                .enumerate()
                .map(|(i, &v)| v * inv * bf16_at(gamma.bytes, i)),
        );
        Ok(())
    }

    fn rmsnorm_heads_in_place(
        &self,
        weights: &MetalWeightCache,
        gamma_key: &str,
        x: &mut [f32],
        heads: usize,
        head_dim: usize,
    ) -> HResult<()> {
        let gamma = require(weights, gamma_key)?;
        expect_dtype(&gamma, DType::Bf16, gamma_key)?;
        if gamma.shape != [head_dim] {
            return Err(format!(
                "{gamma_key}: gamma shape {:?}, expected [{head_dim}]",
                gamma.shape
            ));
        }
        for h in 0..heads {
            let base = h * head_dim;
            let inv = rms_inv(&x[base..base + head_dim], self.arch.rms_norm_eps);
            for i in 0..head_dim {
                x[base + i] *= inv * bf16_at(gamma.bytes, i);
            }
        }
        Ok(())
    }

    fn norm_add(
        &self,
        weights: &MetalWeightCache,
        gamma_key: &str,
        residual: &mut [f32],
        x: &[f32],
        layer_scalar: Option<f32>,
    ) -> HResult<()> {
        if residual.len() != x.len() {
            return Err(format!(
                "{gamma_key}: residual len {} != input len {}",
                residual.len(),
                x.len()
            ));
        }
        let gamma = require(weights, gamma_key)?;
        expect_dtype(&gamma, DType::Bf16, gamma_key)?;
        if gamma.shape != [x.len()] {
            return Err(format!(
                "{gamma_key}: gamma shape {:?}, expected [{}]",
                gamma.shape,
                x.len()
            ));
        }
        let inv = rms_inv(x, self.arch.rms_norm_eps);
        let ls = layer_scalar.unwrap_or(1.0);
        for i in 0..x.len() {
            residual[i] += x[i] * inv * bf16_at(gamma.bytes, i) * ls;
        }
        Ok(())
    }

    fn fp8_matvec(
        &mut self,
        weights: &MetalWeightCache,
        weight_key: &str,
        scale_key: &str,
        x: &[f32],
    ) -> HResult<Vec<f32>> {
        let mut out = Vec::new();
        self.fp8_matvec_into(weights, weight_key, scale_key, x, &mut out)?;
        Ok(out)
    }

    fn fp8_matvec_into(
        &mut self,
        weights: &MetalWeightCache,
        weight_key: &str,
        scale_key: &str,
        x: &[f32],
        out: &mut Vec<f32>,
    ) -> HResult<()> {
        if let Some(metal) = self.metal.as_mut() {
            let w_buf = metal.tensor_view(weights, weight_key)?;
            expect_metal_dtype(&w_buf, DType::Fp8E4M3, weight_key)?;
            expect_metal_rank2(&w_buf, weight_key)?;
            let rows = w_buf.shape[0];
            let cols = w_buf.shape[1];
            if cols != x.len() {
                return Err(format!(
                    "{weight_key}: cols {cols} != input len {}",
                    x.len()
                ));
            }
            let s_buf = metal.tensor_view(weights, scale_key)?;
            if !matches!(s_buf.dtype, DType::Bf16 | DType::F32) {
                return Err(format!(
                    "{scale_key}: dtype {:?}, expected BF16 or F32",
                    s_buf.dtype
                ));
            }
            if let Some((layout, stride)) = metal_scale_layout_shape(&s_buf.shape, rows) {
                return metal
                    .gemv
                    .fp8_row_scaled_f32_into(
                        w_buf.buffer.as_ref(),
                        w_buf.byte_offset,
                        s_buf.buffer.as_ref(),
                        s_buf.byte_offset,
                        s_buf.dtype,
                        layout,
                        stride,
                        x,
                        rows,
                        cols,
                        out,
                    )
                    .map_err(|e| format!("{weight_key}: metal fp8 gemv: {e}"));
            }
        }
        let w = require(weights, weight_key)?;
        expect_dtype(&w, DType::Fp8E4M3, weight_key)?;
        expect_rank2(&w, weight_key)?;
        let rows = w.shape[0];
        let cols = w.shape[1];
        if cols != x.len() {
            return Err(format!(
                "{weight_key}: cols {cols} != input len {}",
                x.len()
            ));
        }
        let scale = require(weights, scale_key)?;
        if !matches!(scale.dtype, DType::Bf16 | DType::F32) {
            return Err(format!(
                "{scale_key}: dtype {:?}, expected BF16 or F32",
                scale.dtype
            ));
        }
        let row_scales = scale_values_for_rows(&scale, rows, scale_key)?;
        let lut = &self.fp8_lut;
        out.clear();
        out.resize(rows, 0.0);
        out.par_iter_mut().enumerate().for_each(|(r, y)| {
            let row = &w.bytes[r * cols..(r + 1) * cols];
            let mut acc = 0.0f32;
            let mut c = 0usize;
            while c + 3 < cols {
                acc += lut[row[c] as usize] * x[c]
                    + lut[row[c + 1] as usize] * x[c + 1]
                    + lut[row[c + 2] as usize] * x[c + 2]
                    + lut[row[c + 3] as usize] * x[c + 3];
                c += 4;
            }
            while c < cols {
                acc += lut[row[c] as usize] * x[c];
                c += 1;
            }
            *y = acc * row_scales[r];
        });
        Ok(())
    }

    fn fp8_matvec_many(
        &mut self,
        weights: &MetalWeightCache,
        projections: &[(&str, &str)],
        x: &[f32],
    ) -> HResult<Vec<Vec<f32>>> {
        let mut out: Vec<Vec<f32>> = Vec::new();
        let mut refs: Vec<_> = Vec::with_capacity(projections.len());
        out.resize_with(projections.len(), Vec::new);
        for dst in out.iter_mut() {
            refs.push(dst);
        }
        self.fp8_matvec_many_into(weights, projections, x, &mut refs)?;
        drop(refs);
        Ok(out)
    }

    fn fp8_matvec_many_into(
        &mut self,
        weights: &MetalWeightCache,
        projections: &[(&str, &str)],
        x: &[f32],
        out: &mut [&mut Vec<f32>],
    ) -> HResult<()> {
        if projections.is_empty() {
            for dst in out.iter_mut() {
                dst.clear();
            }
            return Ok(());
        }
        if out.len() != projections.len() {
            return Err(format!(
                "fp8_matvec_many: output count {} != projection count {}",
                out.len(),
                projections.len()
            ));
        }
        if self.metal.is_none() || !self.batch_gemv {
            for ((weight_key, scale_key), dst) in projections.iter().copied().zip(out.iter_mut()) {
                self.fp8_matvec_into(weights, weight_key, scale_key, x, dst)?;
            }
            return Ok(());
        }

        struct Prepared {
            weight: crate::metal_loader::MetalTensorView,
            scale: crate::metal_loader::MetalTensorView,
            scale_layout: ScaleLayout,
            scale_stride: u32,
            rows: usize,
            cols: usize,
        }

        let mut prepared = Vec::with_capacity(projections.len());
        let metal = self.metal.as_mut().expect("checked above");
        for &(weight_key, scale_key) in projections {
            let weight = metal.tensor_view(weights, weight_key)?;
            expect_metal_dtype(&weight, DType::Fp8E4M3, weight_key)?;
            expect_metal_rank2(&weight, weight_key)?;
            let rows = weight.shape[0];
            let cols = weight.shape[1];
            if cols != x.len() {
                return Err(format!(
                    "{weight_key}: cols {cols} != input len {}",
                    x.len()
                ));
            }
            let scale = metal.tensor_view(weights, scale_key)?;
            if !matches!(scale.dtype, DType::Bf16 | DType::F32) {
                return Err(format!(
                    "{scale_key}: dtype {:?}, expected BF16 or F32",
                    scale.dtype
                ));
            }
            let Some((scale_layout, scale_stride)) = metal_scale_layout_shape(&scale.shape, rows)
            else {
                for ((weight_key, scale_key), dst) in
                    projections.iter().copied().zip(out.iter_mut())
                {
                    self.fp8_matvec_into(weights, weight_key, scale_key, x, dst)?;
                }
                return Ok(());
            };
            prepared.push(Prepared {
                weight,
                scale,
                scale_layout,
                scale_stride,
                rows,
                cols,
            });
        }

        let specs: Vec<_> = prepared
            .iter()
            .map(|p| Fp8GemvInput {
                weight: p.weight.buffer.as_ref(),
                weight_offset: p.weight.byte_offset,
                scale: p.scale.buffer.as_ref(),
                scale_offset: p.scale.byte_offset,
                scale_dtype: p.scale.dtype,
                scale_layout: p.scale_layout,
                scale_stride: p.scale_stride,
                rows: p.rows,
                cols: p.cols,
            })
            .collect();

        metal
            .gemv
            .fp8_many_row_scaled_f32_into_outputs(&specs, x, out)
            .map_err(|e| format!("metal fp8 batched gemv: {e}"))
    }

    fn apply_rope_and_cache(
        &mut self,
        layer_idx: usize,
        layer_type: Gemma4LayerType,
        pos: usize,
        q: &mut [f32],
        k: &mut [f32],
        v: &[f32],
    ) -> HResult<()> {
        let head_dim = self.arch.head_dim_for_layer(layer_idx);
        let num_heads = self.arch.num_attention_heads;
        let num_kv_heads = self.arch.num_kv_heads_for_layer(layer_idx);
        if q.len() != num_heads.saturating_mul(head_dim)
            || k.len() != num_kv_heads.saturating_mul(head_dim)
            || v.len() != num_kv_heads.saturating_mul(head_dim)
        {
            return Err(format!("layer {layer_idx}: invalid q/k/v cache geometry"));
        }
        let cache = &mut self.rope_cache[layer_idx];
        if cache.pos != pos {
            cache.sin_cos.clear();
            cache
                .sin_cos
                .extend(self.rope_inv_freq[layer_idx].iter().map(|&inv_freq| {
                    let angle = pos as f32 * inv_freq;
                    angle.sin_cos()
                }));
            cache.pos = pos;
        }
        apply_rope_cached(q, num_heads, head_dim, &cache.sin_cos);
        apply_rope_cached(k, num_kv_heads, head_dim, &cache.sin_cos);

        let (slot, slots, kv_dim) = {
            let kv = &mut self.kv[layer_idx];
            let slot = match layer_type {
                Gemma4LayerType::SlidingAttention => pos % kv.slots,
                Gemma4LayerType::GlobalAttention => pos,
            };
            if slot >= kv.slots {
                return Err(format!(
                    "layer {layer_idx}: KV slot {slot} exceeds {} slots",
                    kv.slots
                ));
            }
            let start = slot
                .checked_mul(kv.kv_dim)
                .ok_or_else(|| format!("layer {layer_idx}: KV offset overflow"))?;
            kv.k[start..start + kv.kv_dim].copy_from_slice(k);
            kv.v[start..start + kv.kv_dim].copy_from_slice(v);
            (slot, kv.slots, kv.kv_dim)
        };
        if self.metal_attention {
            self.write_metal_kv(layer_idx, slot, slots, kv_dim, k, v)?;
        }
        Ok(())
    }

    fn write_metal_kv(
        &mut self,
        layer_idx: usize,
        slot: usize,
        slots: usize,
        kv_dim: usize,
        k: &[f32],
        v: &[f32],
    ) -> HResult<()> {
        if k.len() != kv_dim || v.len() != kv_dim {
            return Err(format!(
                "layer {layer_idx}: Metal KV write shape mismatch got {}/{} expected {kv_dim}",
                k.len(),
                v.len()
            ));
        }
        let Some(metal) = self.metal.as_mut() else {
            return Ok(());
        };
        let cache = metal.ensure_layer_kv(layer_idx, slots, kv_dim)?;
        let offset = slot
            .checked_mul(kv_dim)
            .ok_or_else(|| format!("layer {layer_idx}: Metal KV slot offset overflow"))?;
        write_f32_to_metal_buffer(&cache.k, offset, k)?;
        write_f32_to_metal_buffer(&cache.v, offset, v)?;
        Ok(())
    }

    fn attend_metal_into(
        &mut self,
        layer_idx: usize,
        q: &[f32],
        slots: &[u32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        out: &mut Vec<f32>,
    ) -> HResult<bool> {
        let Some(metal) = self.metal.as_mut() else {
            return Ok(false);
        };
        let kv = &self.kv[layer_idx];
        let (k_buf, v_buf) = {
            let cache = metal.ensure_layer_kv(layer_idx, kv.slots, kv.kv_dim)?;
            (cache.k.clone(), cache.v.clone())
        };
        metal
            .gemv
            .host_f32_attention_into(
                q,
                &k_buf,
                &v_buf,
                slots,
                num_heads,
                num_kv_heads,
                head_dim,
                kv.kv_dim,
                out,
            )
            .map(|_| true)
            .map_err(|e| format!("layer {layer_idx}: metal host attention: {e}"))
    }

    fn attend_into(
        &mut self,
        layer_idx: usize,
        layer_type: Gemma4LayerType,
        pos: usize,
        q: &[f32],
        out: &mut Vec<f32>,
        score_scratch: &mut Vec<f32>,
        slot_scratch: &mut Vec<usize>,
        metal_slot_scratch: &mut Vec<u32>,
    ) -> HResult<()> {
        let head_dim = self.arch.head_dim_for_layer(layer_idx);
        let num_heads = self.arch.num_attention_heads;
        let num_kv_heads = self.arch.num_kv_heads_for_layer(layer_idx);
        let kv_slots = self.kv[layer_idx].slots;
        let kv_dim = self.kv[layer_idx].kv_dim;
        if num_heads == 0
            || num_kv_heads == 0
            || num_heads % num_kv_heads != 0
            || head_dim == 0
            || q.len() != num_heads.saturating_mul(head_dim)
            || kv_slots == 0
            || kv_dim != num_kv_heads.saturating_mul(head_dim)
            || pos >= self.max_model_len
        {
            return Err(format!("layer {layer_idx}: invalid attention geometry"));
        }
        let group = num_heads / num_kv_heads;
        let start_pos = match layer_type {
            Gemma4LayerType::SlidingAttention => pos.saturating_add(1).saturating_sub(kv_slots),
            Gemma4LayerType::GlobalAttention => 0,
        };
        let len = pos + 1 - start_pos;
        out.clear();
        out.resize(num_heads * head_dim, 0.0);

        if len == 1 {
            let slot = match layer_type {
                Gemma4LayerType::SlidingAttention => pos % kv_slots,
                Gemma4LayerType::GlobalAttention => pos,
            };
            let kv = &self.kv[layer_idx];
            for h in 0..num_heads {
                let kv_h = h / group;
                let out_base = h * head_dim;
                let v_base = slot * kv_dim + kv_h * head_dim;
                out[out_base..out_base + head_dim]
                    .copy_from_slice(&kv.v[v_base..v_base + head_dim]);
            }
            return Ok(());
        }

        slot_scratch.clear();
        slot_scratch.reserve(len);
        match layer_type {
            Gemma4LayerType::SlidingAttention => {
                for p in start_pos..=pos {
                    slot_scratch.push(p % kv_slots);
                }
            }
            Gemma4LayerType::GlobalAttention => {
                for p in start_pos..=pos {
                    slot_scratch.push(p);
                }
            }
        }

        if self.metal_attention && len >= self.metal_attention_min_len {
            metal_slot_scratch.clear();
            metal_slot_scratch.reserve(slot_scratch.len());
            for &slot in slot_scratch.iter() {
                metal_slot_scratch.push(u32::try_from(slot).map_err(|_| {
                    format!("layer {layer_idx}: KV slot exceeds u32 for Metal attention")
                })?);
            }
            if self.attend_metal_into(
                layer_idx,
                q,
                metal_slot_scratch,
                num_heads,
                num_kv_heads,
                head_dim,
                out,
            )? {
                return Ok(());
            }
        }

        let kv = &self.kv[layer_idx];
        if len < self.parallel_attention_min_len {
            score_scratch.clear();
            score_scratch.resize(len, 0.0);
            let scores = score_scratch.as_mut_slice();
            for h in 0..num_heads {
                let kv_h = h / group;
                let q_base = h * head_dim;
                for (j, &slot) in slot_scratch.iter().enumerate() {
                    let k_base = slot * kv_dim + kv_h * head_dim;
                    scores[j] = dot(
                        &q[q_base..q_base + head_dim],
                        &kv.k[k_base..k_base + head_dim],
                    );
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    denom += *s;
                }
                let out_base = h * head_dim;
                for (j, &slot) in slot_scratch.iter().enumerate() {
                    let prob = scores[j] / denom;
                    let v_base = slot * kv_dim + kv_h * head_dim;
                    for i in 0..head_dim {
                        out[out_base + i] += prob * kv.v[v_base + i];
                    }
                }
            }
            return Ok(());
        }

        score_scratch.clear();
        score_scratch.resize(num_heads * len, 0.0);
        out.par_chunks_mut(head_dim)
            .zip(score_scratch.par_chunks_mut(len))
            .enumerate()
            .for_each(|(h, (out_head, scores))| {
                let kv_h = h / group;
                let q_base = h * head_dim;
                for (j, &slot) in slot_scratch.iter().enumerate() {
                    let k_base = slot * kv_dim + kv_h * head_dim;
                    scores[j] = dot(
                        &q[q_base..q_base + head_dim],
                        &kv.k[k_base..k_base + head_dim],
                    );
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    denom += *s;
                }
                for (j, &slot) in slot_scratch.iter().enumerate() {
                    let prob = scores[j] / denom;
                    let v_base = slot * kv_dim + kv_h * head_dim;
                    for i in 0..head_dim {
                        out_head[i] += prob * kv.v[v_base + i];
                    }
                }
            });
        Ok(())
    }

    fn scalar_bf16(&self, weights: &MetalWeightCache, key: &str) -> HResult<f32> {
        let t = require(weights, key)?;
        expect_dtype(&t, DType::Bf16, key)?;
        if t.shape != [1] {
            return Err(format!("{key}: shape {:?}, expected [1]", t.shape));
        }
        Ok(bf16_at(t.bytes, 0))
    }

    fn has_v_proj(&mut self, weights: &MetalWeightCache, layer_idx: usize, key: &str) -> bool {
        match self.has_v_proj[layer_idx] {
            Some(has_v) => has_v,
            None => {
                let has_v = weights.contains(key);
                self.has_v_proj[layer_idx] = Some(has_v);
                has_v
            }
        }
    }

    fn layer_scalar_cached(
        &mut self,
        weights: &MetalWeightCache,
        layer_idx: usize,
        key: &str,
    ) -> HResult<f32> {
        if let Some(scalar) = self.layer_scalars[layer_idx] {
            return Ok(scalar);
        }
        let scalar = self.scalar_bf16(weights, key)?;
        self.layer_scalars[layer_idx] = Some(scalar);
        Ok(scalar)
    }

    fn argmax_lm_head(&mut self, weights: &MetalWeightCache, x: &[f32]) -> HResult<u32> {
        let lm_head_key = self.lm_head_key(weights);
        if let Some(metal) = self.metal.as_mut() {
            let t_buf = metal.tensor_view(weights, &lm_head_key)?;
            expect_metal_dtype(&t_buf, DType::Bf16, &lm_head_key)?;
            expect_metal_rank2(&t_buf, &lm_head_key)?;
            let vocab = t_buf.shape[0];
            let hidden = t_buf.shape[1];
            if hidden != x.len() {
                return Err(format!(
                    "{lm_head_key}: hidden {hidden} != input len {}",
                    x.len()
                ));
            }
            if self.lmhead_argmax {
                let idx = metal
                    .gemv
                    .bf16_argmax_f32(t_buf.buffer.as_ref(), t_buf.byte_offset, x, vocab, hidden)
                    .map_err(|e| format!("{lm_head_key}: metal bf16 argmax gemv: {e}"))?;
                return Ok(idx);
            }
            let logits = metal
                .gemv
                .bf16_f32(t_buf.buffer.as_ref(), t_buf.byte_offset, x, vocab, hidden)
                .map_err(|e| format!("{lm_head_key}: metal bf16 gemv: {e}"))?;
            let (idx, _) = logits
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .ok_or_else(|| format!("{lm_head_key} is empty"))?;
            return Ok(idx as u32);
        }
        let t = require(weights, &lm_head_key)?;
        expect_dtype(&t, DType::Bf16, &lm_head_key)?;
        expect_rank2(&t, &lm_head_key)?;
        let vocab = t.shape[0];
        let hidden = t.shape[1];
        if hidden != x.len() {
            return Err(format!(
                "{lm_head_key}: hidden {hidden} != input len {}",
                x.len()
            ));
        }
        let (idx, _) = (0..vocab)
            .into_par_iter()
            .map(|r| {
                let base = r * hidden;
                let mut acc = 0.0f32;
                for i in 0..hidden {
                    acc += bf16_at(t.bytes, base + i) * x[i];
                }
                (r as u32, acc)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or_else(|| format!("{lm_head_key} is empty"))?;
        Ok(idx)
    }

    fn lm_head_logits(&mut self, weights: &MetalWeightCache, x: &[f32]) -> HResult<Vec<f32>> {
        let lm_head_key = self.lm_head_key(weights);
        if let Some(metal) = self.metal.as_mut() {
            let t_buf = metal.tensor_view(weights, &lm_head_key)?;
            expect_metal_dtype(&t_buf, DType::Bf16, &lm_head_key)?;
            expect_metal_rank2(&t_buf, &lm_head_key)?;
            let vocab = t_buf.shape[0];
            let hidden = t_buf.shape[1];
            if hidden != x.len() {
                return Err(format!(
                    "{lm_head_key}: hidden {hidden} != input len {}",
                    x.len()
                ));
            }
            return metal
                .gemv
                .bf16_f32(t_buf.buffer.as_ref(), t_buf.byte_offset, x, vocab, hidden)
                .map_err(|e| format!("{lm_head_key}: metal bf16 gemv: {e}"));
        }
        let t = require(weights, &lm_head_key)?;
        expect_dtype(&t, DType::Bf16, &lm_head_key)?;
        expect_rank2(&t, &lm_head_key)?;
        let vocab = t.shape[0];
        let hidden = t.shape[1];
        if hidden != x.len() {
            return Err(format!(
                "{lm_head_key}: hidden {hidden} != input len {}",
                x.len()
            ));
        }
        Ok((0..vocab)
            .into_par_iter()
            .map(|r| {
                let base = r * hidden;
                let mut acc = 0.0f32;
                for i in 0..hidden {
                    acc += bf16_at(t.bytes, base + i) * x[i];
                }
                acc
            })
            .collect())
    }

    fn lm_head_target_nll(
        &mut self,
        weights: &MetalWeightCache,
        x: &[f32],
        target: u32,
        softcap: f32,
    ) -> HResult<Option<f64>> {
        let lm_head_key = self.lm_head_key(weights);
        let Some(metal) = self.metal.as_mut() else {
            return Ok(None);
        };
        let t_buf = metal.tensor_view(weights, &lm_head_key)?;
        expect_metal_dtype(&t_buf, DType::Bf16, &lm_head_key)?;
        expect_metal_rank2(&t_buf, &lm_head_key)?;
        let vocab = t_buf.shape[0];
        let hidden = t_buf.shape[1];
        if hidden != x.len() {
            return Err(format!(
                "{lm_head_key}: hidden {hidden} != input len {}",
                x.len()
            ));
        }
        if target as usize >= vocab {
            return Err(format!("target token {target} exceeds vocab size {vocab}"));
        }
        metal
            .gemv
            .bf16_target_nll_f32(
                t_buf.buffer.as_ref(),
                t_buf.byte_offset,
                x,
                vocab,
                hidden,
                target,
                softcap,
            )
            .map(Some)
            .map_err(|e| format!("{lm_head_key}: metal bf16 target nll: {e}"))
    }

    fn lm_head_key(&self, weights: &MetalWeightCache) -> Arc<str> {
        if weights.contains("lm_head.weight") {
            Arc::from("lm_head.weight")
        } else {
            self.keys.embed_tokens.clone()
        }
    }
}

fn target_nll(logits: &[f32], target: u32, softcap: f32) -> HResult<f64> {
    let target = target as usize;
    if target >= logits.len() {
        return Err(format!(
            "target token {target} exceeds vocab size {}",
            logits.len()
        ));
    }
    let mut max = f64::NEG_INFINITY;
    for &logit in logits {
        max = max.max(softcapped_logit(logit, softcap));
    }
    let mut sum = 0.0f64;
    for &logit in logits {
        sum += (softcapped_logit(logit, softcap) - max).exp();
    }
    let target_logit = softcapped_logit(logits[target], softcap);
    Ok(max + sum.ln() - target_logit)
}

fn softcapped_logit(logit: f32, softcap: f32) -> f64 {
    if softcap > 0.0 {
        let cap = softcap as f64;
        cap * ((logit as f64) / cap).tanh()
    } else {
        logit as f64
    }
}

fn metal_scale_layout_shape(shape: &[usize], rows: usize) -> Option<(ScaleLayout, u32)> {
    let (layout, stride) = match shape {
        [1] => (ScaleLayout::Single, 1),
        [r] if *r == rows => (ScaleLayout::PerRow, 1),
        [r, c] if *r == rows && *c > 0 => (ScaleLayout::PerRow, *c),
        [rb, cb] if *rb * 128 >= rows && *cb > 0 => (ScaleLayout::BlockRow128, *cb),
        _ => return None,
    };
    u32::try_from(stride).ok().map(|s| (layout, s))
}

fn env_bool(name: &str) -> bool {
    env_bool_default(name, false)
}

fn env_bool_default(name: &str, default: bool) -> bool {
    match std::env::var(name).ok().as_deref() {
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES") => true,
        Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO") => false,
        _ => default,
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|s| s.parse().ok())
}

fn require<'a>(weights: &'a MetalWeightCache, key: &str) -> HResult<HostTensorView<'a>> {
    weights
        .require_tensor_view(key)
        .map_err(|e| format!("{key}: {e}"))
}

fn expect_dtype(t: &HostTensorView<'_>, expected: DType, key: &str) -> HResult<()> {
    if t.dtype != expected {
        return Err(format!(
            "{key}: dtype {:?}, expected {:?}",
            t.dtype, expected
        ));
    }
    Ok(())
}

fn expect_rank2(t: &HostTensorView<'_>, key: &str) -> HResult<()> {
    if t.shape.len() != 2 {
        return Err(format!("{key}: shape {:?}, expected rank 2", t.shape));
    }
    Ok(())
}

fn expect_metal_dtype(
    t: &crate::metal_loader::MetalTensorView,
    expected: DType,
    key: &str,
) -> HResult<()> {
    if t.dtype != expected {
        return Err(format!(
            "{key}: dtype {:?}, expected {:?}",
            t.dtype, expected
        ));
    }
    Ok(())
}

fn expect_metal_rank2(t: &crate::metal_loader::MetalTensorView, key: &str) -> HResult<()> {
    if t.shape.len() != 2 {
        return Err(format!("{key}: shape {:?}, expected rank 2", t.shape));
    }
    Ok(())
}

fn bf16_at(bytes: &[u8], idx: usize) -> f32 {
    let lo = bytes[2 * idx];
    let hi = bytes[2 * idx + 1];
    f32::from_bits(u32::from_le_bytes([0, 0, lo, hi]))
}

fn f32_at(bytes: &[u8], idx: usize) -> f32 {
    f32::from_le_bytes(bytes[4 * idx..4 * idx + 4].try_into().unwrap())
}

fn write_f32_to_metal_buffer(buf: &Buffer, elem_offset: usize, x: &[f32]) -> HResult<()> {
    let end = elem_offset
        .checked_add(x.len())
        .ok_or_else(|| "Metal buffer element range overflow".to_string())?;
    let byte_end = end
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Metal buffer byte range overflow".to_string())?;
    if byte_end > buf.length() as usize {
        return Err(format!(
            "Metal buffer write ends at byte {byte_end}, beyond {}",
            buf.length()
        ));
    }
    let contents = buf.contents();
    if contents.is_null() || (contents as usize) % std::mem::align_of::<f32>() != 0 {
        return Err("Metal buffer has no aligned CPU-visible storage".into());
    }
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), contents.cast::<f32>().add(elem_offset), x.len());
    }
    Ok(())
}

fn scale_values_for_rows(scale: &HostTensorView<'_>, rows: usize, key: &str) -> HResult<Vec<f32>> {
    let elements = scale.shape.iter().try_fold(1usize, |count, &dim| {
        count
            .checked_mul(dim)
            .ok_or_else(|| format!("{key}: element count overflow"))
    })?;
    let expected_bytes = elements
        .checked_mul(scale.dtype.bytes())
        .ok_or_else(|| format!("{key}: byte count overflow"))?;
    if elements == 0 || scale.bytes.len() != expected_bytes {
        return Err(format!("{key}: invalid scale storage length"));
    }
    let index_for_row = |row: usize| -> Option<usize> {
        match scale.shape {
            [1] => Some(0),
            [r] if *r == rows => Some(row),
            [r, c] if *r == rows && *c > 0 => row.checked_mul(*c),
            [rb, cb] if *cb > 0 && rb.checked_mul(128).is_some_and(|n| n >= rows) => {
                (row / 128).checked_mul(*cb)
            }
            _ => None,
        }
    };
    (0..rows)
        .map(|row| {
            let idx = index_for_row(row)
                .filter(|&idx| idx < elements)
                .ok_or_else(|| format!("{key}: unsupported scale shape {:?}", scale.shape))?;
            let value = match scale.dtype {
                DType::Bf16 => bf16_at(scale.bytes, idx),
                DType::F32 => f32_at(scale.bytes, idx),
                _ => return Err(format!("{key}: unsupported scale dtype {:?}", scale.dtype)),
            };
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "{key}: scale at row {row} must be positive and finite"
                ));
            }
            Ok(value)
        })
        .collect()
}

fn rms_inv(x: &[f32], eps: f32) -> f32 {
    let ss = x.iter().map(|v| v * v).sum::<f32>();
    (ss / x.len() as f32 + eps).sqrt().recip()
}

fn apply_rope_cached(x: &mut [f32], heads: usize, head_dim: usize, sin_cos: &[(f32, f32)]) {
    let half_head = head_dim / 2;
    for h in 0..heads {
        let base = h * head_dim;
        for (i, &(sin, cos)) in sin_cos.iter().enumerate() {
            let lo = x[base + i];
            let hi = x[base + i + half_head];
            x[base + i] = lo * cos - hi * sin;
            x[base + i + half_head] = lo * sin + hi * cos;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let s = (b >> 7) & 1;
    let e = (b >> 3) & 0xF;
    let m = b & 0x7;
    let val = if e == 0 {
        if m == 0 {
            0.0
        } else {
            (m as f32) * (1.0 / 512.0)
        }
    } else if e == 15 && m == 7 {
        return f32::NAN;
    } else {
        f32::from_bits(((e as u32 + 120) << 23) | ((m as u32) << 20))
    };
    if s != 0 {
        -val
    } else {
        val
    }
}
