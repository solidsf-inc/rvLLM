// Ported from huggingface/transformers revision
// 10555512868d663ee1ff627e4f5c5c260114235b:
// src/transformers/models/gemma4/modular_gemma4.py
// Apache-2.0 License, Copyright (c) HuggingFace and Google.
// Source classes: Gemma4 vision model, encoder, attention, RoPE, pooler,
// patch embedder, RMSNorm, and masking helpers. Modifications: candle tensors,
// inference-only forwards, gathered position embeddings, and additive masks.

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{linear, linear_no_bias, Embedding, Linear, VarBuilder};

use crate::config::{Gemma4VisionConfig, MAX_VISION_PATCHES, MAX_VISION_SOFT_TOKENS};

// RMS Norm (Gemma4 / Gemma3n flavour: standard RMSNorm, NOT (1+w)*).

#[derive(Debug, Clone)]
pub struct Gemma4RmsNorm {
    weight: Option<Tensor>, // None when with_scale=false (e.g. v_norm)
    eps: f64,
}

impl Gemma4RmsNorm {
    pub fn new(
        dim: usize,
        eps: f32,
        with_scale: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        if dim == 0 || !eps.is_finite() || eps <= 0.0 {
            candle_core::bail!("RMSNorm requires dim > 0 and finite eps > 0");
        }
        let weight = if with_scale {
            Some(vb.get((dim,), "weight")?)
        } else {
            None
        };
        Ok(Self {
            weight,
            eps: eps as f64,
        })
    }
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let in_dtype = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        let mean_sq = x32.sqr()?.mean_keepdim(D::Minus1)?;
        let scale = (mean_sq + self.eps)?.powf(-0.5)?;
        let normed = x32.broadcast_mul(&scale)?;
        let out = match &self.weight {
            Some(w) => normed.broadcast_mul(&w.to_dtype(DType::F32)?)?,
            None => normed,
        };
        out.to_dtype(in_dtype)
    }
}

// Patch embedder: Linear(3*patch^2 -> hidden) + 2D learned position embedding.

#[derive(Debug)]
struct PatchEmbedder {
    input_proj: Linear,
    /// HF's single 2-axis table split into equivalent gathered embeddings.
    pos_axis_0: Embedding,
    pos_axis_1: Embedding,
}

impl PatchEmbedder {
    fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let in_features = 3usize
            .checked_mul(cfg.patch_size)
            .and_then(|value| value.checked_mul(cfg.patch_size))
            .ok_or_else(|| candle_core::Error::Msg("patch feature count overflow".into()))?;
        let input_proj = linear_no_bias(in_features, cfg.hidden_size, vb.pp("input_proj"))?;
        // Split HF's `[2, positions, hidden]` table by axis.
        let table = vb.get(
            (2, cfg.position_embedding_size, cfg.hidden_size),
            "position_embedding_table",
        )?;
        let pos_axis_0 = Embedding::new(table.i(0)?.contiguous()?, cfg.hidden_size);
        let pos_axis_1 = Embedding::new(table.i(1)?.contiguous()?, cfg.hidden_size);
        Ok(Self {
            input_proj,
            pos_axis_0,
            pos_axis_1,
        })
    }
    /// Inputs: pixels `[B,N,3*patch^2]`, positions `[B,N,2]`, padding `[B,N]`.
    fn forward(
        &self,
        pixel_values: &Tensor,
        pixel_position_ids: &Tensor,
        padding_positions: &Tensor,
    ) -> candle_core::Result<Tensor> {
        // Inline pixel scaling: 2 * (x - 0.5)
        let scaled = ((pixel_values - 0.5)? * 2.0)?;
        let scaled = scaled.to_dtype(self.input_proj.weight().dtype())?;
        let hidden = self.input_proj.forward(&scaled)?;
        // Gather each axis; clamp padding's -1 before converting to U32.
        let pos_clamped = pixel_position_ids.clamp(0i64, i64::MAX)?;
        let pos_x = pos_clamped.i((.., .., 0))?.contiguous()?;
        let pos_y = pos_clamped.i((.., .., 1))?.contiguous()?;
        let pos_x = pos_x.to_dtype(DType::U32)?;
        let pos_y = pos_y.to_dtype(DType::U32)?;
        let pe_x = self.pos_axis_0.forward(&pos_x)?; // [B, N, hidden]
        let pe_y = self.pos_axis_1.forward(&pos_y)?; // [B, N, hidden]
        let pe = (pe_x + pe_y)?.to_dtype(hidden.dtype())?;
        // Zero padded position embeddings.
        let valid = (1.0 - padding_positions.to_dtype(DType::F32)?)?;
        let valid = valid
            .unsqueeze(D::Minus1)? // [B, N, 1]
            .to_dtype(hidden.dtype())?;
        let pe = pe.broadcast_mul(&valid)?;
        hidden + pe
    }
}

// Two-axis vision RoPE, concatenated to `[B,N,head_dim]`.

#[derive(Debug)]
struct VisionRotary {
    /// inv_freq tensor of shape [spatial_dim/2] on cpu/device. f32.
    inv_freq: Tensor,
}

impl VisionRotary {
    fn new(cfg: &Gemma4VisionConfig, device: &Device) -> candle_core::Result<Self> {
        if cfg.head_dim == 0 || cfg.head_dim % 4 != 0 {
            candle_core::bail!("vision head_dim must be nonzero and divisible by 4");
        }
        if !cfg.rope_parameters.rope_theta.is_finite() || cfg.rope_parameters.rope_theta <= 0.0 {
            candle_core::bail!("vision rope_theta must be finite and > 0");
        }
        let base = cfg.rope_parameters.rope_theta as f64;
        let spatial_dim = cfg.head_dim / 2;
        // arange(0, spatial_dim, 2) — count of inverse freqs.
        let n = spatial_dim / 2;
        let mut buf = Vec::with_capacity(n);
        for k in 0..n {
            let exp = (2 * k) as f64 / spatial_dim as f64;
            buf.push((1.0 / base.powf(exp)) as f32);
        }
        let inv_freq = Tensor::from_vec(buf, (n,), device)?;
        Ok(Self { inv_freq })
    }
    /// Produce per-axis cosine/sine tables `[B,N,head_dim]`.
    fn forward(
        &self,
        pixel_position_ids: &Tensor,
        target_dtype: DType,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let b = pixel_position_ids.dim(0)?;
        let _n = pixel_position_ids.dim(1)?;
        let dev = pixel_position_ids.device();
        // inv_freq: [spatial_dim/2] -> [1, spatial_dim/2, 1] -> [B, spatial_dim/2, 1]
        let inv = self
            .inv_freq
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(D::Minus1)?
            .expand((b, self.inv_freq.dim(0)?, 1))?
            .to_device(dev)?;
        let mut cos_parts = Vec::with_capacity(2);
        let mut sin_parts = Vec::with_capacity(2);
        for i in 0..2 {
            let pos = pixel_position_ids
                .i((.., .., i))?
                .to_dtype(DType::F32)?
                .unsqueeze(1)?; // [B, 1, N]
                                // freqs = (inv @ pos).transpose(1, 2)  -> [B, N, spatial_dim/2]
            let freqs = inv.matmul(&pos)?.transpose(1, 2)?.contiguous()?;
            let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [B, N, spatial_dim]
            cos_parts.push(emb.cos()?);
            sin_parts.push(emb.sin()?);
        }
        let cos_refs: Vec<&Tensor> = cos_parts.iter().collect();
        let sin_refs: Vec<&Tensor> = sin_parts.iter().collect();
        let cos = Tensor::cat(&cos_refs, D::Minus1)?.to_dtype(target_dtype)?; // [B, N, head_dim]
        let sin = Tensor::cat(&sin_refs, D::Minus1)?.to_dtype(target_dtype)?;
        Ok((cos, sin))
    }
}

// Apply standard rotary independently to two equal spatial chunks.

fn rotate_half(x: &Tensor) -> candle_core::Result<Tensor> {
    let last = x.dim(D::Minus1)?;
    let half = last / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, last - half)?;
    let neg_x2 = x2.neg()?;
    Tensor::cat(&[&neg_x2, &x1], D::Minus1)
}

/// Rotate `[B,N,H,D]` using `[B,N,D]` cosine/sine tables.
fn apply_multidim_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    let ndim = 2usize;
    let num_input_channels = x.dim(D::Minus1)?;
    // num_rotated_channels_per_dim = 2 * (D // (2 * ndim))
    let per = 2 * (num_input_channels / (2 * ndim));
    if per == 0 {
        candle_core::bail!(
            "apply_multidim_rope: per-dim channel count is 0 (D={num_input_channels}, ndim={ndim})"
        );
    }
    let mut out_parts: Vec<Tensor> = Vec::with_capacity(ndim);
    for k in 0..ndim {
        let start = k * per;
        let x_part = x.narrow(D::Minus1, start, per)?;
        let cos_part = cos.narrow(D::Minus1, start, per)?.unsqueeze(2)?; // [B,N,1,per]
        let sin_part = sin.narrow(D::Minus1, start, per)?.unsqueeze(2)?;
        let rotated =
            (x_part.broadcast_mul(&cos_part)? + rotate_half(&x_part)?.broadcast_mul(&sin_part)?)?;
        out_parts.push(rotated);
    }
    let refs: Vec<&Tensor> = out_parts.iter().collect();
    Tensor::cat(&refs, D::Minus1)
}

// Attention (bidirectional, MHA, RmsNorm-q/k/v, multidim RoPE).

#[derive(Debug)]
struct VisionAttention {
    q_proj: ClippableLinear,
    k_proj: ClippableLinear,
    v_proj: ClippableLinear,
    o_proj: ClippableLinear,
    q_norm: Gemma4RmsNorm,
    k_norm: Gemma4RmsNorm,
    v_norm: Gemma4RmsNorm, // with_scale=false
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let d = cfg.head_dim;
        // HF stores every projection under its `.linear` submodule.
        let uc = cfg.use_clipped_linears;
        let q_proj = ClippableLinear::new(h, nh * d, cfg.attention_bias, uc, vb.pp("q_proj"))?;
        let k_proj = ClippableLinear::new(h, nkv * d, cfg.attention_bias, uc, vb.pp("k_proj"))?;
        let v_proj = ClippableLinear::new(h, nkv * d, cfg.attention_bias, uc, vb.pp("v_proj"))?;
        let o_proj = ClippableLinear::new(nh * d, h, cfg.attention_bias, uc, vb.pp("o_proj"))?;
        let q_norm = Gemma4RmsNorm::new(d, cfg.rms_norm_eps, true, vb.pp("q_norm"))?;
        let k_norm = Gemma4RmsNorm::new(d, cfg.rms_norm_eps, true, vb.pp("k_norm"))?;
        let v_norm = Gemma4RmsNorm::new(d, cfg.rms_norm_eps, false, vb.pp("v_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            num_heads: nh,
            num_kv_heads: nkv,
            num_kv_groups: nh / nkv,
            head_dim: d,
        })
    }
    /// hidden: [B, N, H]
    /// cos, sin: [B, N, head_dim]
    /// attn_mask_additive: [B, 1, N, N] (0 = keep, -inf = block) or None.
    fn forward(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attn_mask_additive: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (b, n, _h) = hidden.dims3()?;
        // Q: [B, N, nh, d] -> q_norm -> rope -> transpose -> [B, nh, N, d]
        let q = self
            .q_proj
            .forward(hidden)?
            .reshape((b, n, self.num_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        let q = apply_multidim_rope(&q, cos, sin)?;
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = self
            .k_proj
            .forward(hidden)?
            .reshape((b, n, self.num_kv_heads, self.head_dim))?;
        let k = self.k_norm.forward(&k)?;
        let k = apply_multidim_rope(&k, cos, sin)?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = self
            .v_proj
            .forward(hidden)?
            .reshape((b, n, self.num_kv_heads, self.head_dim))?;
        let v = self.v_norm.forward(&v)?;
        let v = v.transpose(1, 2)?.contiguous()?;
        // Released Gemma 4 uses groups=1; retain general grouped KV.
        let k = repeat_kv(&k, self.num_kv_groups)?;
        let v = repeat_kv(&v, self.num_kv_groups)?;
        // HF vision attention uses an explicit scale of 1.0.
        let attn_in_dtype = q.dtype();
        let q32 = q.to_dtype(DType::F32)?;
        let k32 = k.to_dtype(DType::F32)?;
        let v32 = v.to_dtype(DType::F32)?;
        // attn = Q @ K^T  shape: [B, nh, N, N]
        let scores = q32.matmul(&k32.transpose(2, 3)?.contiguous()?)?;
        let scores = if let Some(mask) = attn_mask_additive {
            scores.broadcast_add(&mask.to_dtype(DType::F32)?)?
        } else {
            scores
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v32)?; // [B, nh, N, d]
        let ctx = ctx.transpose(1, 2)?.contiguous()?; // [B, N, nh, d]
        let ctx = ctx.reshape((b, n, self.num_heads * self.head_dim))?;
        let ctx = ctx.to_dtype(attn_in_dtype)?;
        self.o_proj.forward(&ctx)
    }
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> candle_core::Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv, n, d) = x.dims4()?;
    let x = x
        .unsqueeze(2)?
        .expand((b, n_kv, n_rep, n, d))?
        .reshape((b, n_kv * n_rep, n, d))?;
    Ok(x)
}

fn build_linear(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
) -> candle_core::Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}

/// HF clippable linear: clamp input, apply `.linear`, clamp output.
#[derive(Debug)]
struct ClippableLinear {
    linear: Linear,
    clip: Option<ClipBounds>,
}

#[derive(Debug, Clone, Copy)]
struct ClipBounds {
    input_min: f64,
    input_max: f64,
    output_min: f64,
    output_max: f64,
}

impl ClippableLinear {
    /// `vb` is scoped at the proj level (e.g. `vb.pp("q_proj")`); the weight
    /// lives at `vb.pp("linear")` and the four bounds are scalar siblings.
    fn new(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        use_clipped: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let linear = build_linear(in_dim, out_dim, bias, vb.pp("linear"))?;
        let clip = if use_clipped {
            let bounds = ClipBounds {
                input_min: load_clip_scalar(&vb, "input_min")?,
                input_max: load_clip_scalar(&vb, "input_max")?,
                output_min: load_clip_scalar(&vb, "output_min")?,
                output_max: load_clip_scalar(&vb, "output_max")?,
            };
            if !bounds.input_min.is_finite()
                || !bounds.input_max.is_finite()
                || !bounds.output_min.is_finite()
                || !bounds.output_max.is_finite()
                || bounds.input_min > bounds.input_max
                || bounds.output_min > bounds.output_max
            {
                candle_core::bail!("invalid non-finite or inverted linear clip bounds");
            }
            Some(bounds)
        } else {
            None
        };
        Ok(Self { linear, clip })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self.clip {
            None => self.linear.forward(x),
            Some(c) => {
                let x = x.clamp(c.input_min, c.input_max)?;
                let y = self.linear.forward(&x)?;
                y.clamp(c.output_min, c.output_max)
            }
        }
    }
}

/// Load a scalar clip bound. The HF buffers are 0-d tensors; accept rank-0 or
/// `[1]` and extract the f32 value.
fn load_clip_scalar(vb: &VarBuilder, name: &str) -> candle_core::Result<f64> {
    let t = vb.get((), name).or_else(|_| vb.get((1,), name))?;
    let v = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    Ok(v[0] as f64)
}

// MLP: gate/up/down with gelu_pytorch_tanh on gate.

#[derive(Debug)]
struct VisionMlp {
    gate_proj: ClippableLinear,
    up_proj: ClippableLinear,
    down_proj: ClippableLinear,
}

impl VisionMlp {
    fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let uc = cfg.use_clipped_linears;
        Ok(Self {
            // ClippableLinear: same `.linear` infix as attention, plus the
            // per-Linear input/output clamps when use_clipped_linears (E-series).
            gate_proj: ClippableLinear::new(h, i, false, uc, vb.pp("gate_proj"))?,
            up_proj: ClippableLinear::new(h, i, false, uc, vb.pp("up_proj"))?,
            down_proj: ClippableLinear::new(i, h, false, uc, vb.pp("down_proj"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // gelu_pytorch_tanh ≈ candle's .gelu() (tanh approximation).
        let g = self.gate_proj.forward(x)?.gelu()?;
        let u = self.up_proj.forward(x)?;
        self.down_proj.forward(&(g * u)?)
    }
}

// Encoder layer (4 norms, pre-/post- attn and ffn).

#[derive(Debug)]
struct EncoderLayer {
    self_attn: VisionAttention,
    mlp: VisionMlp,
    input_layernorm: Gemma4RmsNorm,
    post_attention_layernorm: Gemma4RmsNorm,
    pre_feedforward_layernorm: Gemma4RmsNorm,
    post_feedforward_layernorm: Gemma4RmsNorm,
}

impl EncoderLayer {
    fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            self_attn: VisionAttention::new(cfg, vb.pp("self_attn"))?,
            mlp: VisionMlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: Gemma4RmsNorm::new(
                h,
                cfg.rms_norm_eps,
                true,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: Gemma4RmsNorm::new(
                h,
                cfg.rms_norm_eps,
                true,
                vb.pp("post_attention_layernorm"),
            )?,
            pre_feedforward_layernorm: Gemma4RmsNorm::new(
                h,
                cfg.rms_norm_eps,
                true,
                vb.pp("pre_feedforward_layernorm"),
            )?,
            post_feedforward_layernorm: Gemma4RmsNorm::new(
                h,
                cfg.rms_norm_eps,
                true,
                vb.pp("post_feedforward_layernorm"),
            )?,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attn_mask_additive: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, cos, sin, attn_mask_additive)?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = (residual + h)?;
        let residual = &h;
        let h2 = self.pre_feedforward_layernorm.forward(residual)?;
        let h2 = self.mlp.forward(&h2)?;
        let h2 = self.post_feedforward_layernorm.forward(&h2)?;
        residual + h2
    }
}

// Encoder.

#[derive(Debug)]
struct Encoder {
    layers: Vec<EncoderLayer>,
    rotary: VisionRotary,
}

impl Encoder {
    fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let lv = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::new(cfg, lv.pp(i.to_string()))?);
        }
        let rotary = VisionRotary::new(cfg, vb.device())?;
        Ok(Self { layers, rotary })
    }
    fn forward(
        &self,
        inputs_embeds: &Tensor,
        valid_mask: &Tensor, // [B, N] f32 (1 = valid, 0 = padding)
        pixel_position_ids: &Tensor,
    ) -> candle_core::Result<Tensor> {
        // Build additive bidirectional mask: [B, 1, N, N], -inf at padding cols
        // (and padding rows for safety; padding rows are discarded later).
        let attn_mask = bidirectional_additive_mask(valid_mask, inputs_embeds.dtype())?;
        let (cos, sin) = self
            .rotary
            .forward(pixel_position_ids, inputs_embeds.dtype())?;
        let mut h = inputs_embeds.clone();
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, Some(&attn_mask))?;
        }
        Ok(h)
    }
}

/// Build an additive attention mask: `0` where both query and key are
/// valid, `-inf` where either is padding. Shape `[B, 1, N, N]`.
fn bidirectional_additive_mask(valid_mask: &Tensor, dtype: DType) -> candle_core::Result<Tensor> {
    let (b, n) = valid_mask.dims2()?;
    let v = valid_mask.to_dtype(DType::F32)?;
    // pair_valid = v_q * v_k  -> [B, N, N]
    let v_q = v.unsqueeze(2)?; // [B, N, 1]
    let v_k = v.unsqueeze(1)?; // [B, 1, N]
    let pair_valid = v_q.broadcast_mul(&v_k)?; // [B, N, N]
                                               // 0 where both valid, -inf where either is padding.
                                               // additive = (pair_valid - 1) * INF  -> 0 if valid, -inf otherwise.
                                               // Use a large negative to stay finite for non-fp32 dtypes.
    let neg_large = match dtype {
        DType::F16 => -1.0e4_f32,
        DType::BF16 => -3.0e4_f32,
        _ => -1.0e30_f32,
    };
    let additive = ((pair_valid - 1.0)?.affine(-neg_large as f64, 0.0))?
        .reshape((b, 1, n, n))?
        .to_dtype(dtype)?;
    Ok(additive)
}

// Pooler (spatial position-aware avg pool + sqrt(H) scaling).

#[derive(Debug, Clone)]
struct Pooler {
    root_hidden_size: f64,
}

impl Pooler {
    fn new(cfg: &Gemma4VisionConfig) -> Self {
        Self {
            root_hidden_size: (cfg.hidden_size as f64).sqrt(),
        }
    }
    /// hidden: [B, N, H] (the current API requires B=1)
    /// pixel_position_ids: [B, N, 2] i64
    /// padding_positions:  [B, N] u8 (1 = padding)
    /// output_length: total soft tokens after pooling
    ///
    /// Returns (pooled, pool_valid_mask):
    ///   pooled:    [B, output_length, H]
    ///   pool_valid: [B, output_length] u8 (1 = valid group)
    fn forward(
        &self,
        hidden: &Tensor,
        pixel_position_ids: &Tensor,
        padding_positions: &Tensor,
        output_length: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (b, n, h) = hidden.dims3()?;
        if output_length > n {
            candle_core::bail!(
                "pooler: requested {output_length} soft tokens but only {n} patches available"
            );
        }
        // Zero out hidden states for padding rows.
        let valid = (1.0 - padding_positions.to_dtype(DType::F32)?)?
            .unsqueeze(D::Minus1)?
            .to_dtype(hidden.dtype())?;
        let hidden_masked = hidden.broadcast_mul(&valid)?;
        if n == output_length {
            // No spatial pooling. Just scale + return the (cheap) valid mask
            // computed from the per-patch padding tensor.
            let scaled = (hidden_masked.to_dtype(DType::F32)? * self.root_hidden_size)?
                .to_dtype(hidden.dtype())?;
            let pool_valid =
                (1.0 - padding_positions.to_dtype(DType::F32)?)?.to_dtype(DType::U8)?;
            return Ok((scaled, pool_valid));
        }
        // k^2 * output_length == n
        let k = ((n / output_length) as f64).sqrt() as usize;
        if k * k * output_length != n {
            candle_core::bail!(
                "pooler: cannot pool {n} patches to {output_length} (k^2*length must equal n; got k={k})"
            );
        }
        // Compute kernel index for every patch:
        //   max_x = positions[..., 0].max(dim=-1, keepdim=True) + 1
        //   kernel = positions // k
        //   kernel_idx = kernel[..., 0] + (max_x // k) * kernel[..., 1]
        // We do all of this in i64 via CPU readback (small) for simplicity
        // and correctness — the pooler isn't on a hot path.
        let pos_clamped = pixel_position_ids.clamp(0i64, i64::MAX)?;
        let pos_x = pos_clamped.i((.., .., 0))?.to_vec2::<i64>()?;
        let pos_y = pos_clamped.i((.., .., 1))?.to_vec2::<i64>()?;
        let mut kernel_idx_host: Vec<Vec<i64>> = Vec::with_capacity(b);
        let mut weights_host: Vec<Vec<f32>> = Vec::with_capacity(b); // [B, output_length * N]
        let inv_k_sq = 1.0 / ((k * k) as f32);
        let pad_host = padding_positions.to_vec2::<u8>()?;
        for bi in 0..b {
            let max_x = *pos_x[bi].iter().max().unwrap_or(&0) + 1;
            let max_x_div_k = max_x / k as i64;
            let mut kidx = Vec::with_capacity(n);
            for pi in 0..n {
                let kx = pos_x[bi][pi] / k as i64;
                let ky = pos_y[bi][pi] / k as i64;
                kidx.push(kx + max_x_div_k * ky);
            }
            kernel_idx_host.push(kidx.clone());
            // weights[bi]: [output_length, N], row-major.
            let mut w = vec![0.0f32; output_length * n];
            for pi in 0..n {
                if pad_host[bi][pi] != 0 {
                    continue;
                }
                let g = kidx[pi];
                if g < 0 || (g as usize) >= output_length {
                    candle_core::bail!(
                        "pooler: kernel index {g} out of range [0, {output_length}) for patch {pi} (k={k}, max_x_div_k={max_x_div_k})"
                    );
                }
                w[(g as usize) * n + pi] = inv_k_sq;
            }
            weights_host.push(w);
        }
        // Build [B, output_length, N] weights tensor.
        let mut flat: Vec<f32> = Vec::with_capacity(b * output_length * n);
        for bi in 0..b {
            flat.extend_from_slice(&weights_host[bi]);
        }
        let weights = Tensor::from_vec(flat, (b, output_length, n), hidden.device())?
            .to_dtype(hidden.dtype())?;
        // pooled = weights @ hidden_masked  -> [B, output_length, H]
        let pooled = weights.matmul(&hidden_masked)?;
        // Build pool_valid mask: a group is valid iff at least one weight is
        // non-zero (which means at least one non-padded patch maps to it).
        // We just check whether any element of the corresponding weight row
        // is non-zero (in float; since we set them to inv_k_sq > 0 exactly).
        let mut pool_valid_host = vec![0u8; b * output_length];
        for bi in 0..b {
            for gi in 0..output_length {
                let mut any = false;
                for pi in 0..n {
                    if weights_host[bi][gi * n + pi] != 0.0 {
                        any = true;
                        break;
                    }
                }
                pool_valid_host[bi * output_length + gi] = if any { 1 } else { 0 };
            }
        }
        let pool_valid = Tensor::from_vec(pool_valid_host, (b, output_length), hidden.device())?;
        // Scale by sqrt(hidden_size).
        let pooled =
            (pooled.to_dtype(DType::F32)? * self.root_hidden_size)?.to_dtype(hidden.dtype())?;
        let _ = h;
        Ok((pooled, pool_valid))
    }
}

// Top-level vision model.

#[derive(Debug)]
pub struct Gemma4VisionModel {
    patch_embedder: PatchEmbedder,
    encoder: Encoder,
    pooler: Pooler,
    /// Present only when cfg.standardize == true.
    std_bias: Option<Tensor>,
    std_scale: Option<Tensor>,
    patch_features: usize,
    position_embedding_size: usize,
}

impl Gemma4VisionModel {
    pub fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        validate_model_config(cfg)?;
        let patch_features = 3usize
            .checked_mul(cfg.patch_size)
            .and_then(|value| value.checked_mul(cfg.patch_size))
            .ok_or_else(|| candle_core::Error::Msg("patch feature count overflow".into()))?;
        let patch_embedder = PatchEmbedder::new(cfg, vb.pp("patch_embedder"))?;
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let pooler = Pooler::new(cfg);
        let (std_bias, std_scale) = if cfg.standardize {
            (
                Some(vb.get((cfg.hidden_size,), "std_bias")?),
                Some(vb.get((cfg.hidden_size,), "std_scale")?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            patch_embedder,
            encoder,
            pooler,
            std_bias,
            std_scale,
            patch_features,
            position_embedding_size: cfg.position_embedding_size,
        })
    }
    /// pixel_values:        [1, num_patches, 3*patch_size^2] f32 in [0, 1]
    /// pixel_position_ids:  [1, num_patches, 2] i64 (padding = (-1, -1))
    /// output_length:       how many soft tokens to emit (post-pool)
    ///
    /// Returns [output_length, hidden_size] after pooler + (optional)
    /// standardize. Padding groups (if any) are stripped.
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        pixel_position_ids: &Tensor,
        output_length: usize,
    ) -> candle_core::Result<Tensor> {
        let (batch, patches, features) = pixel_values.dims3()?;
        let position_dims = pixel_position_ids.dims3()?;
        if batch != 1 || position_dims != (batch, patches, 2) || features != self.patch_features {
            candle_core::bail!(
                "vision input must be pixel_values [1,N,{}] and position ids [1,N,2]",
                self.patch_features
            );
        }
        if pixel_values.dtype() != DType::F32 || pixel_position_ids.dtype() != DType::I64 {
            candle_core::bail!("vision input requires F32 pixels and I64 position ids");
        }
        if patches == 0 || patches > MAX_VISION_PATCHES {
            candle_core::bail!(
                "vision patch count must be in 1..={MAX_VISION_PATCHES}, got {patches}"
            );
        }
        if output_length == 0
            || output_length > patches
            || output_length > MAX_VISION_SOFT_TOKENS as usize
        {
            candle_core::bail!(
                "vision output length must be in 1..=min({patches}, {MAX_VISION_SOFT_TOKENS})"
            );
        }
        validate_position_ids(pixel_position_ids, self.position_embedding_size)?;
        // padding_positions = (pixel_position_ids == -1).all(dim=-1)  -> [B, N] u8
        let padding_positions = padding_mask_from_position_ids(pixel_position_ids)?;
        let inputs_embeds =
            self.patch_embedder
                .forward(pixel_values, pixel_position_ids, &padding_positions)?;
        let valid_mask = (1.0 - padding_positions.to_dtype(DType::F32)?)?; // [B, N]
        let encoded = self
            .encoder
            .forward(&inputs_embeds, &valid_mask, pixel_position_ids)?;
        let (pooled, pool_valid) = self.pooler.forward(
            &encoded,
            pixel_position_ids,
            &padding_positions,
            output_length,
        )?;
        // Strip pooled rows that correspond to fully-padded groups.
        // pool_valid: [B, output_length] u8. The current API requires B=1.
        let pool_valid_host = pool_valid.to_vec2::<u8>()?;
        let mut keep_rows: Vec<Tensor> = Vec::new();
        for gi in 0..output_length {
            if pool_valid_host[0][gi] != 0 {
                keep_rows.push(pooled.i((0, gi))?);
            }
        }
        if keep_rows.is_empty() {
            candle_core::bail!("vision model: all pooled groups were padding-only");
        }
        let keep_refs: Vec<&Tensor> = keep_rows.iter().collect();
        let stacked = Tensor::stack(&keep_refs, 0)?; // [valid_count, H]
        let out = if let (Some(b), Some(s)) = (&self.std_bias, &self.std_scale) {
            let b = b.to_dtype(stacked.dtype())?;
            let s = s.to_dtype(stacked.dtype())?;
            stacked.broadcast_sub(&b)?.broadcast_mul(&s)?
        } else {
            stacked
        };
        Ok(out)
    }
}

fn validate_model_config(cfg: &Gemma4VisionConfig) -> candle_core::Result<()> {
    for (name, value, maximum) in [
        ("hidden_size", cfg.hidden_size, 65_536),
        ("intermediate_size", cfg.intermediate_size, 262_144),
        ("num_attention_heads", cfg.num_attention_heads, 1_024),
        ("num_key_value_heads", cfg.num_key_value_heads, 1_024),
        ("num_hidden_layers", cfg.num_hidden_layers, 1_024),
        ("head_dim", cfg.head_dim, 4_096),
        ("patch_size", cfg.patch_size, 256),
        (
            "position_embedding_size",
            cfg.position_embedding_size,
            1_048_576,
        ),
        ("pooling_kernel_size", cfg.pooling_kernel_size, 256),
    ] {
        if value == 0 || value > maximum {
            candle_core::bail!("vision {name} must be in 1..={maximum}, got {value}");
        }
    }
    if cfg.num_attention_heads % cfg.num_key_value_heads != 0 || cfg.head_dim % 4 != 0 {
        candle_core::bail!("invalid vision attention head geometry");
    }
    if cfg.hidden_activation != "gelu_pytorch_tanh"
        || !cfg.rms_norm_eps.is_finite()
        || cfg.rms_norm_eps <= 0.0
        || cfg.attention_dropout != 0.0
        || cfg.rope_parameters.rope_type != "default"
    {
        candle_core::bail!("unsupported Gemma 4 vision configuration");
    }
    Ok(())
}

fn validate_position_ids(
    pixel_position_ids: &Tensor,
    position_embedding_size: usize,
) -> candle_core::Result<()> {
    let host = pixel_position_ids.to_vec3::<i64>()?;
    let mut saw_padding = false;
    let mut saw_valid = false;
    for pair in &host[0] {
        if pair.len() != 2 {
            candle_core::bail!("vision position id must contain exactly two coordinates");
        }
        match (pair[0], pair[1]) {
            (-1, -1) => saw_padding = true,
            (x, y) if x >= 0 && y >= 0 => {
                if saw_padding {
                    candle_core::bail!("vision padding positions must form a trailing suffix");
                }
                if x as usize >= position_embedding_size || y as usize >= position_embedding_size {
                    candle_core::bail!(
                        "vision position ({x}, {y}) exceeds embedding size {position_embedding_size}"
                    );
                }
                saw_valid = true;
            }
            _ => candle_core::bail!(
                "vision position ids must be nonnegative pairs or the (-1,-1) padding sentinel"
            ),
        }
    }
    if !saw_valid {
        candle_core::bail!("vision input contains no valid patches");
    }
    Ok(())
}

/// `(pixel_position_ids == -1).all(dim=-1)` reduced to a `[B, N]` u8 tensor
/// where 1 = padding patch.
fn padding_mask_from_position_ids(pixel_position_ids: &Tensor) -> candle_core::Result<Tensor> {
    let host = pixel_position_ids.to_vec3::<i64>()?;
    let b = host.len();
    let n = host[0].len();
    let mut out = Vec::with_capacity(b * n);
    for bi in 0..b {
        for pi in 0..n {
            let pad = host[bi][pi].iter().all(|&v| v == -1);
            out.push(if pad { 1u8 } else { 0u8 });
        }
    }
    Tensor::from_vec(out, (b, n), pixel_position_ids.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Gemma4VisionConfig, VisionRopeParameters};
    fn tiny_cfg() -> Gemma4VisionConfig {
        Gemma4VisionConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            num_hidden_layers: 2,
            head_dim: 8,
            patch_size: 4,
            position_embedding_size: 16,
            pooling_kernel_size: 1, // disable spatial pooling for the tiny test
            max_position_embeddings: 1024,
            hidden_activation: "gelu_pytorch_tanh".into(),
            rms_norm_eps: 1e-6,
            attention_bias: false,
            attention_dropout: 0.0,
            standardize: false,
            use_clipped_linears: false,
            rope_parameters: VisionRopeParameters {
                rope_type: "default".into(),
                rope_theta: 100.0,
            },
        }
    }
    #[test]
    fn forward_shape_no_nan() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        // 9 patches in a 3x3 grid.
        let model = Gemma4VisionModel::new(&cfg, vb).expect("build model");
        let num_patches = 9usize;
        let patch_dim = 3 * cfg.patch_size * cfg.patch_size;
        let pixels = Tensor::from_vec(
            vec![0.5f32; num_patches * patch_dim],
            (1, num_patches, patch_dim),
            &dev,
        )
        .unwrap();
        let mut pos = Vec::with_capacity(num_patches * 2);
        for y in 0..3i64 {
            for x in 0..3i64 {
                pos.push(x);
                pos.push(y);
            }
        }
        let positions = Tensor::from_vec(pos, (1, num_patches, 2), &dev).unwrap();
        let out = model
            .forward(&pixels, &positions, num_patches)
            .expect("forward");
        let shape = out.dims().to_vec();
        assert_eq!(shape, vec![num_patches, cfg.hidden_size]);
        let flat = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in &flat {
            assert!(v.is_finite(), "non-finite value in output: {v}");
        }
    }
    #[test]
    fn forward_with_spatial_pool() {
        // 16 patches in 4x4 grid, pool k=2 -> 4 output groups.
        let dev = Device::Cpu;
        let mut cfg = tiny_cfg();
        cfg.pooling_kernel_size = 2;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = Gemma4VisionModel::new(&cfg, vb).expect("build model");
        let grid = 4usize;
        let num_patches = grid * grid;
        let patch_dim = 3 * cfg.patch_size * cfg.patch_size;
        let pixels = Tensor::from_vec(
            vec![0.25f32; num_patches * patch_dim],
            (1, num_patches, patch_dim),
            &dev,
        )
        .unwrap();
        let mut pos = Vec::with_capacity(num_patches * 2);
        for y in 0..grid as i64 {
            for x in 0..grid as i64 {
                pos.push(x);
                pos.push(y);
            }
        }
        let positions = Tensor::from_vec(pos, (1, num_patches, 2), &dev).unwrap();
        // (4*4) / (k*k) = 16/4 = 4 pooled tokens.
        let output_length = 4usize;
        let out = model
            .forward(&pixels, &positions, output_length)
            .expect("forward");
        assert_eq!(out.dims().to_vec(), vec![output_length, cfg.hidden_size]);
        let flat = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in &flat {
            assert!(v.is_finite(), "non-finite value in pooled output: {v}");
        }
    }
}
