//! Gemma 4 weight structures.
//!
//! Sliding and global layers may use different attention geometry. Shapes
//! are carried by the parsed model configuration and validated at load time.

use crate::weights::{F16Weight, Fp8Weight};

#[derive(Debug)]
pub struct Gemma4LayerWeights {
    pub qkv: Fp8Weight,
    pub o_proj: Fp8Weight,
    pub gate_up: Fp8Weight,
    pub down_proj: Fp8Weight,
    pub qkv_f16: Option<F16Weight>,
    pub o_proj_f16: Option<F16Weight>,
    pub gate_up_f16: Option<F16Weight>,
    pub down_proj_f16: Option<F16Weight>,
    pub input_layernorm: F16Weight,
    pub post_attention_layernorm: F16Weight,
    pub pre_feedforward_layernorm: F16Weight,
    pub post_feedforward_layernorm: F16Weight,
    pub post_per_layer_input_norm: Option<F16Weight>,
    pub q_norm: F16Weight,
    pub k_norm: F16Weight,
    pub layer_scalar: F16Weight,
    pub per_layer_input_gate_f16: Option<F16Weight>,
    pub per_layer_projection_f16: Option<F16Weight>,
}

#[derive(Debug)]
pub struct PrunedVocab {
    pub full_vocab: usize,
    pub head_vocab: usize,
    pub keep_ids: Vec<u32>,
    pub full_to_keep: Vec<i32>,
}

#[derive(Debug)]
pub struct Gemma4LoadedModel {
    pub embedding: F16Weight,
    pub lm_head_fp8: Fp8Weight,
    pub lm_head_f16: F16Weight,
    pub pruned_vocab: Option<PrunedVocab>,
    pub final_norm: F16Weight,
    pub embed_tokens_per_layer: Option<F16Weight>,
    pub per_layer_model_projection_f16: Option<F16Weight>,
    pub per_layer_projection_norm: Option<F16Weight>,
    /// Sliding-attention RoPE tables from the model configuration.
    pub rope_cos_sliding: F16Weight,
    pub rope_sin_sliding: F16Weight,
    /// Global-attention RoPE tables from the model configuration.
    pub rope_cos_global: F16Weight,
    pub rope_sin_global: F16Weight,
    pub layers: Vec<Gemma4LayerWeights>,
}

// ===================================================================
// Packed INT4 + per-layer-embedding weight handles.
//
// The supported layout is compressed-tensors `pack-quantized` INT4:
// every decoder Linear ships `{weight_packed (I32), weight_scale (F16),
// weight_shape (I64)}`. Group size comes from the model configuration; the
// runtime currently supports symmetric four-bit weights. The PLE
// (per-layer embedding) machinery adds a per-layer embed table plus a
// per-layer gate + projection Linear, and the lm_head is row-pruned to
// a keepset.
// ===================================================================

/// One pack-quantized INT4 Linear's three device handles.
///
/// `packed`  : I32 region, logical shape `[out, in/8]` (8 nibbles / lane).
/// `scale`   : F16 region, per-group scale `[out, in/group_size]`
///             (or `[out, 1]` for the channel-strategy lm_head).
/// `shape`   : the LOGICAL `[out, in]` dims decoded from `weight_shape`.
/// `group_size`: configured group width, or `in` for one group per row.
///
/// All offsets are device pointers into the loader's HBM arena. The
/// dequant/GEMM path reads `packed` as `(out, in/8)` I32 and the
/// scale as `(out, in/group_size)` F16.
#[derive(Debug, Clone)]
pub struct WPacked {
    /// Device pointer to the I32 packed-weight region.
    pub packed: u64,
    /// Device pointer to the F16 per-group scale region.
    pub scale: u64,
    /// Logical `[out, in]` (from `weight_shape`).
    pub shape: [usize; 2],
    /// Packed column count = `in / 8` (8 int4 per I32 lane).
    pub packed_cols: usize,
    /// Scale group count along `in` = `in / group_size`.
    pub scale_groups: usize,
    /// Quantization group size.
    pub group_size: usize,
    /// Packed element width. The current runtime supports four bits.
    pub num_bits: u32,
    /// Whether the quantizer is symmetric.
    pub symmetric: bool,
}

impl WPacked {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        packed: u64,
        scale: u64,
        shape: [usize; 2],
        packed_cols: usize,
        scale_groups: usize,
        group_size: usize,
        num_bits: u32,
        symmetric: bool,
    ) -> std::result::Result<Self, String> {
        let [out, input] = shape;
        if packed == 0 || scale == 0 {
            return Err("packed and scale device pointers must be nonzero".into());
        }
        if out == 0 || input == 0 || group_size == 0 || scale_groups == 0 {
            return Err("packed weight dimensions and groups must be nonzero".into());
        }
        if num_bits != 4 || !symmetric {
            return Err("only symmetric four-bit packed weights are supported".into());
        }
        if input % 8 != 0 || packed_cols != input / 8 {
            return Err("packed column count does not match logical input width".into());
        }
        if input % group_size != 0 || scale_groups != input / group_size {
            return Err("scale group count does not match logical input width".into());
        }
        Ok(Self {
            packed,
            scale,
            shape,
            packed_cols,
            scale_groups,
            group_size,
            num_bits,
            symmetric,
        })
    }

    pub fn out_features(&self) -> usize {
        self.shape[0]
    }
    pub fn in_features(&self) -> usize {
        self.shape[1]
    }
}

/// A bf16 (unquantized) Linear weight that stayed out of INT4 — used for
/// the modules on the `ignore` list and for the PLE embed table.
pub type Bf16Weight = F16Weight;

/// Per-layer packed weights. Attention + MLP linears are INT4-packed. The
/// tail `num_kv_shared_layers` layers have `k_proj`/`v_proj == None`
/// (they read a share-source layer's KV; see `Gemma4Arch::build_kv_share_src`).
///
/// PLE per-layer extras (`per_layer_input_gate`, `per_layer_projection`)
/// are present on every layer and are themselves INT4-packed.
#[derive(Debug)]
pub struct E4bLayerWeights {
    pub layer_idx: usize,
    /// `sliding_attention` ⇒ false, `full_attention` ⇒ true.
    pub is_full_attention: bool,
    /// True for the tail KV-shared layers (no own k/v).
    pub kv_shared: bool,
    /// Index of the layer this one reads KV from, when `kv_shared`.
    pub kv_share_src: Option<usize>,

    pub q_proj: WPacked,
    /// `None` for KV-shared tail layers.
    pub k_proj: Option<WPacked>,
    /// `None` for KV-shared tail layers.
    pub v_proj: Option<WPacked>,
    pub o_proj: WPacked,
    pub gate_proj: WPacked,
    pub up_proj: WPacked,
    pub down_proj: WPacked,

    // PLE per-layer gate + projection (INT4-packed).
    /// `[hidden_size_per_layer_input, hidden_size]`.
    pub per_layer_input_gate: WPacked,
    /// `[hidden_size, hidden_size_per_layer_input]`.
    pub per_layer_projection: WPacked,

    // bf16 norms / scalars.
    pub input_layernorm: Bf16Weight,
    pub post_attention_layernorm: Bf16Weight,
    pub pre_feedforward_layernorm: Bf16Weight,
    pub post_feedforward_layernorm: Bf16Weight,
    /// PLE-specific norm applied to the projected per-layer input.
    pub post_per_layer_input_norm: Bf16Weight,
    /// Q-norm gamma `[head_dim]`. Present on every layer (Q is always
    /// projected from the layer's own residual).
    pub q_norm: Bf16Weight,
    /// K-norm gamma `[head_dim]`. `None` for KV-shared tail layers — they
    /// do not project/normalize their own K (they read the share-source's
    /// already-normed K cache).
    pub k_norm: Option<Bf16Weight>,
    /// Per-layer residual multiplier `[1]`.
    pub layer_scalar: Bf16Weight,
}

/// The PLE (Per-Layer Embeddings) global tables, shared across all layers.
#[derive(Debug)]
pub struct PleTables {
    /// `embed_tokens_per_layer.weight`: bf16 `[vocab, num_layers * ple_dim]`.
    pub embed_tokens_per_layer: Bf16Weight,
    /// True iff the `sqrt(ple_dim)` scale has been multiplied into the table.
    pub embed_scale_folded: bool,
    /// The scale value (`sqrt(ple_dim)`); recorded even when folded so a
    /// reference path can divide it back out for validation.
    pub embed_scale: f32,
    /// `per_layer_model_projection`: INT4 `[num_layers*ple_dim, hidden]`
    /// projects the residual stream into the per-layer
    /// input space before the per-layer gate/projection.
    pub per_layer_model_projection: WPacked,
    /// `per_layer_projection_norm.weight`: bf16 `[ple_dim]`.
    pub per_layer_projection_norm: Bf16Weight,
}

/// A pruned language-model head and its vocabulary remap.
#[derive(Debug)]
pub struct PrunedLmHead {
    /// INT4-packed pruned head: logical `[K, hidden]` where K = keepset size
    /// with channel-strategy scale `[K, 1]`.
    pub head: WPacked,
    /// `keep_ids[local_row] -> global_token_id`. Length = K, sorted ascending,
    /// values in `[0, full_vocab)`. Used to remap the pruned argmax winner
    /// back to a global token id (greedy path) or scatter to full vocab.
    pub keep_ids: Vec<u32>,
    /// Pruned row count K (== keep_ids.len()).
    pub pruned_vocab_k: usize,
    /// Full (unpruned) vocabulary size.
    pub full_vocab: usize,
}

/// A packed model's loaded weights.
#[derive(Debug)]
pub struct E4bLoadedModel {
    /// `embed_tokens.weight` bf16 `[vocab, hidden]`, pre-scaled by
    /// `sqrt(hidden_size)` at load.
    pub embedding: Bf16Weight,
    /// `norm.weight` bf16 `[hidden]`.
    pub final_norm: Bf16Weight,
    pub lm_head: PrunedLmHead,
    pub ple: PleTables,
    /// Sliding-attention RoPE tables.
    pub rope_cos_sliding: F16Weight,
    pub rope_sin_sliding: F16Weight,
    /// Global-attention RoPE tables.
    pub rope_cos_global: F16Weight,
    pub rope_sin_global: F16Weight,
    pub layers: Vec<E4bLayerWeights>,
    /// KV-share source map `kv_share_src[layer] -> Option<src_layer>`.
    pub kv_share_src: Vec<Option<usize>>,
}
