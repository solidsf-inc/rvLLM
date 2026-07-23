//! Gemma 4 weight loader.
//!
//! Handles: different weight prefixes (model.language_model.layers.*),
//! tied embeddings (lm_head = embed_tokens), 4 norms per layer,
//! QK-norm weights, and per-layer KV head variation.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use half::f16;
use rvllm_core::{
    fp8::f32_to_fp8_e4m3 as fp8_e4m3_encode, DType, LoaderCtx, LoaderError, Result, RvllmError,
};
use rvllm_mem::HbmArena;

use crate::fp8_quant::{check_clamp_gate, quantize_per_tensor_ref, FP8_E4M3_MAX};
use crate::gemma4_arch::Gemma4Arch;
use crate::gemma4_weights::{E4bLoadedModel, Gemma4LayerWeights, Gemma4LoadedModel, PrunedVocab};
use crate::safetensors::{ShardHeader, ShardIndex, TensorEntry};
use crate::weights::{F16Weight, Fp8Weight};

struct ShardMap {
    _mmap: memmap2::Mmap,
    header: ShardHeader,
}

const MAX_AUX_JSON_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone)]
struct HostF16Tensor {
    bytes: Vec<u8>,
    shape: Vec<usize>,
}

impl ShardMap {
    fn open(path: &Path) -> Result<Self> {
        let f = std::fs::File::open(path).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&f) }.map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
        let header = ShardHeader::parse(path, &mmap)?;
        Ok(Self {
            _mmap: mmap,
            header,
        })
    }

    fn bytes(&self) -> &[u8] {
        &self._mmap
    }
}

pub fn load_gemma4_model(
    model_dir: &Path,
    arena: &HbmArena,
    arch: &Gemma4Arch,
) -> Result<Gemma4LoadedModel> {
    let idx = ShardIndex::resolve(model_dir)?;
    let mut shards = Vec::with_capacity(idx.shards.len());
    for p in &idx.shards {
        shards.push(ShardMap::open(p)?);
    }
    let mut tensors: BTreeMap<String, (usize, TensorEntry)> = BTreeMap::new();
    for (si, sm) in shards.iter().enumerate() {
        for (name, entry) in &sm.header.tensors {
            if tensors.insert(name.clone(), (si, entry.clone())).is_some() {
                return Err(loader_corrupt(
                    model_dir,
                    Some(name.clone()),
                    "duplicate tensor name across shards",
                ));
            }
        }
    }

    let bytes_of = |si: usize, e: &TensorEntry| -> &[u8] {
        let s = shards[si].bytes();
        let start = e.file_offset as usize;
        &s[start..start + e.nbytes as usize]
    };

    let prefix = &arch.weight_prefix;

    let get_tensor = |name: &str| -> Option<(usize, TensorEntry)> { tensors.get(name).cloned() };

    let must_get = |name: &str| -> Result<(usize, TensorEntry)> {
        get_tensor(name).ok_or_else(|| RvllmError::Loader {
            err: LoaderError::MissingTensor {
                name: name.to_string(),
            },
            ctx: LoaderCtx {
                path: model_dir.to_path_buf(),
                tensor: Some(name.to_string()),
            },
            bt: std::backtrace::Backtrace::capture(),
        })
    };

    let packed_names = |hf_name: &str| -> Option<(String, String, String)> {
        let base = hf_name.strip_suffix(".weight")?;
        Some((
            format!("{base}.weight_packed"),
            format!("{base}.weight_scale"),
            format!("{base}.weight_shape"),
        ))
    };

    let has_host_weight = |hf_name: &str| -> bool {
        if get_tensor(hf_name).is_some() {
            return true;
        }
        let Some((packed, scale, shape)) = packed_names(hf_name) else {
            return false;
        };
        get_tensor(&packed).is_some()
            && get_tensor(&scale).is_some()
            && get_tensor(&shape).is_some()
    };

    let load_host_f16 = |hf_name: &str| -> Result<HostF16Tensor> {
        if let Some((si, e)) = get_tensor(hf_name) {
            return Ok(HostF16Tensor {
                bytes: tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?,
                shape: e.shape.clone(),
            });
        }
        if let Some((packed_name, scale_name, shape_name)) = packed_names(hf_name) {
            match (
                get_tensor(&packed_name),
                get_tensor(&scale_name),
                get_tensor(&shape_name),
            ) {
                (Some(packed), Some(scale), Some(shape)) => {
                    let (bytes, shape) = packed_int4_tensor_to_f16_bytes(
                        model_dir, &packed, &scale, &shape, &shards,
                    )?;
                    return Ok(HostF16Tensor { bytes, shape });
                }
                (Some(_), _, _) => {
                    return Err(loader_corrupt(
                        model_dir,
                        Some(packed_name),
                        &format!(
                            "packed INT4 tensor for {hf_name} requires matching weight_scale and weight_shape"
                        ),
                    ));
                }
                _ => {}
            }
        }
        let _ = must_get(hf_name)?;
        unreachable!("must_get returns an error for missing tensor")
    };

    let concat_host_f16 = |names: &[String]| -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for name in names {
            let h = load_host_f16(name)?;
            out.extend_from_slice(&h.bytes);
        }
        Ok(out)
    };

    let kv_share_src = arch.build_kv_share_src()?;
    let resolve_shared_kv_name = |layer_idx: usize, suffix: &str| -> Result<String> {
        let source_idx = kv_share_src
            .get(layer_idx)
            .copied()
            .ok_or_else(|| {
                loader_corrupt(
                    model_dir,
                    None,
                    &format!("KV-share layer index {layer_idx} is out of bounds"),
                )
            })?
            .unwrap_or(layer_idx);
        if source_idx != layer_idx {
            eprintln!(
                "[loader] declared KV-share: layer {layer_idx} {suffix} -> layer {source_idx}"
            );
        }
        Ok(format!("{prefix}.layers.{source_idx}.{suffix}"))
    };

    let upload_f16 = |name: &'static str, hf_name: &str| -> Result<F16Weight> {
        let host = load_host_f16(hf_name)?;
        let region = arena.region(name, host.bytes.len(), 16)?;
        unsafe { region.copy_from_host(&host.bytes)? };
        Ok(F16Weight {
            offset_bytes: region.device_ptr(),
            shape: host.shape,
        })
    };

    let upload_scaled_f16 = |name: &'static str, hf_name: &str, scale: f32| -> Result<F16Weight> {
        let mut host = load_host_f16(hf_name)?;
        for chunk in host.bytes.chunks_exact_mut(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let v = f16::from_bits(bits).to_f32() * scale;
            let out = f16::from_f32(v).to_le_bytes();
            chunk[0] = out[0];
            chunk[1] = out[1];
        }
        let region = arena.region(name, host.bytes.len(), 16)?;
        unsafe { region.copy_from_host(&host.bytes)? };
        Ok(F16Weight {
            offset_bytes: region.device_ptr(),
            shape: host.shape,
        })
    };

    let embed_name = format!("{prefix}.embed_tokens.weight");
    // Gemma models scale embeddings by sqrt(hidden_size) after lookup.
    // Pre-scale at load time so the embedding_gather kernel doesn't need modification.
    let embedding = {
        let (si, e) = must_get(&embed_name)?;
        if e.shape != [arch.vocab_size, arch.hidden_size] {
            return Err(loader_shape_mismatch(
                &embed_name,
                vec![arch.vocab_size, arch.hidden_size],
                e.shape.clone(),
                model_dir,
            ));
        }
        let mut buf = tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?;
        let scale = (arch.hidden_size as f32).sqrt();
        eprintln!(
            "[loader] Gemma embedding scale: sqrt({}) = {:.2}",
            arch.hidden_size, scale
        );
        let n = buf.len() / 2;
        for i in 0..n {
            let bits = u16::from_le_bytes([buf[2 * i], buf[2 * i + 1]]);
            let v = f16::from_bits(bits);
            let scaled = f16::from_f32(v.to_f32() * scale);
            let out = scaled.to_le_bytes();
            buf[2 * i] = out[0];
            buf[2 * i + 1] = out[1];
        }
        let region = arena.region("embedding", buf.len(), 16)?;
        unsafe { region.copy_from_host(&buf)? };
        F16Weight {
            offset_bytes: region.device_ptr(),
            shape: e.shape.clone(),
        }
    };

    let norm_name = format!("{prefix}.norm.weight");
    let final_norm_host = load_host_f16(&norm_name)?;
    expect_host_shape(&norm_name, &final_norm_host, &[arch.hidden_size], model_dir)?;
    let final_norm = upload_f16("final_norm", &norm_name)?;

    let ple_enabled = arch.hidden_size_per_layer_input > 0;
    let (embed_tokens_per_layer, per_layer_model_projection_f16, per_layer_projection_norm) =
        if ple_enabled {
            let ple_dim = arch.hidden_size_per_layer_input;
            let total_ple_dim = arch.num_hidden_layers * ple_dim;
            eprintln!(
                "[loader] Gemma 4 PLE enabled: layers={} dim={} total_dim={}",
                arch.num_hidden_layers, ple_dim, total_ple_dim
            );
            let ple_embed = upload_scaled_f16(
                "ple_embedding",
                &format!("{prefix}.embed_tokens_per_layer.weight"),
                (ple_dim as f32).sqrt(),
            )?;
            if ple_embed.shape
                != [
                    arch.vocab_size_per_layer_input,
                    arch.per_layer_embed_total(),
                ]
            {
                return Err(loader_shape_mismatch(
                    "embed_tokens_per_layer.weight",
                    vec![
                        arch.vocab_size_per_layer_input,
                        arch.per_layer_embed_total(),
                    ],
                    ple_embed.shape.clone(),
                    model_dir,
                ));
            }
            let ple_proj = upload_f16(
                "ple_model_projection_f16",
                &format!("{prefix}.per_layer_model_projection.weight"),
            )?;
            if ple_proj.shape != [arch.per_layer_embed_total(), arch.hidden_size] {
                return Err(loader_shape_mismatch(
                    "per_layer_model_projection.weight",
                    vec![arch.per_layer_embed_total(), arch.hidden_size],
                    ple_proj.shape.clone(),
                    model_dir,
                ));
            }
            let ple_norm = upload_f16(
                "ple_projection_norm",
                &format!("{prefix}.per_layer_projection_norm.weight"),
            )?;
            if ple_norm.shape != [ple_dim] {
                return Err(loader_shape_mismatch(
                    "per_layer_projection_norm.weight",
                    vec![ple_dim],
                    ple_norm.shape.clone(),
                    model_dir,
                ));
            }
            (Some(ple_embed), Some(ple_proj), Some(ple_norm))
        } else {
            (None, None, None)
        };

    // Detect pre-quantized FP8 weights from their declared tensor dtype.
    let probe_name = format!("{prefix}.layers.0.self_attn.q_proj.weight");
    let fp8_prequant = get_tensor(&probe_name)
        .map(|(_, e)| e.dtype == DType::Fp8E4M3)
        .unwrap_or(false);
    if fp8_prequant {
        eprintln!("[loader] Gemma 4 FP8 pre-quantized mode: uploading weights directly with cuBLASLt per-channel scales");
    } else if has_host_weight(&probe_name) && get_tensor(&probe_name).is_none() {
        eprintln!("[loader] packed INT4 mode: CPU-dequantizing weights for FP8 upload");
    } else {
        eprintln!("[loader] Gemma 4 BF16 mode: CPU-quantizing to FP8 at load time");
    }

    let packed_lm_head = match (
        get_tensor("lm_head.weight_packed"),
        get_tensor("lm_head.weight_scale"),
        get_tensor("lm_head.weight_shape"),
    ) {
        (Some(packed), Some(scale), Some(shape)) if get_tensor("lm_head.weight").is_none() => Some(
            packed_int4_tensor_to_f16_bytes(model_dir, &packed, &scale, &shape, &shards)?,
        ),
        (Some(_), _, _) if get_tensor("lm_head.weight").is_none() => {
            return Err(loader_corrupt(
                model_dir,
                Some("lm_head.weight_packed".to_string()),
                "packed INT4 lm_head requires lm_head.weight_scale and lm_head.weight_shape",
            ));
        }
        _ => None,
    };

    let explicit_lm_head_shape = if let Some((_, e)) = get_tensor("lm_head.weight") {
        Some((e.name, e.shape))
    } else if let Some((_, shape)) = &packed_lm_head {
        Some(("lm_head.weight_packed(dequant)".to_string(), shape.clone()))
    } else {
        None
    };
    let lm_head_rows = explicit_lm_head_shape.as_ref().map_or_else(
        || embedding.shape.first().copied().unwrap_or(0),
        |(_, shape)| shape.first().copied().unwrap_or(0),
    );
    let pruned_vocab = load_pruned_vocab(
        model_dir,
        arch.vocab_size,
        embedding.shape.first().copied().unwrap_or(0),
        lm_head_rows,
    )?;
    let expected_lm_head_rows = pruned_vocab
        .as_ref()
        .map_or(arch.vocab_size, |vocab| vocab.head_vocab);
    if let Some((tensor, shape)) = &explicit_lm_head_shape {
        expect_lm_head_shape(
            tensor,
            shape,
            expected_lm_head_rows,
            arch.hidden_size,
            model_dir,
        )?;
    }

    let lm_head_fp8 = if let Some((si, e)) = get_tensor("lm_head.weight") {
        if e.dtype == DType::Fp8E4M3 {
            let scale_entry = get_tensor("lm_head.weight_scale");
            upload_fp8_direct_channelscale(
                arena,
                "lm_head",
                &(si, e),
                scale_entry.as_ref(),
                &shards,
                model_dir,
            )?
        } else {
            upload_fp8(
                arena,
                "lm_head",
                &tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?,
                &e.shape,
                "lm_head.weight",
                model_dir,
            )?
        }
    } else if let Some((ref f16_bytes, ref shape)) = packed_lm_head {
        eprintln!(
            "[loader] packed INT4 lm_head: rows={} cols={} ({:.1} MB f16)",
            shape[0],
            shape[1],
            f16_bytes.len() as f64 / 1e6
        );
        upload_fp8(
            arena,
            "lm_head",
            f16_bytes,
            shape,
            "lm_head.weight_packed(dequant)",
            model_dir,
        )?
    } else {
        let (si, e) = must_get(&embed_name)?;
        eprintln!("[loader] tied embeddings: CPU-quantizing BF16 embed_tokens ({} elements) to FP8 for lm_head",
            e.shape.iter().product::<usize>());
        let buf = tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?;
        upload_fp8(
            arena,
            "lm_head",
            &buf,
            &e.shape,
            "lm_head(tied_embed)",
            model_dir,
        )?
    };

    let lm_head_f16 = {
        let (buf, shape) = if let Some((si, e)) = get_tensor("lm_head.weight") {
            (
                tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?,
                e.shape.clone(),
            )
        } else if let Some((ref f16_bytes, ref shape)) = packed_lm_head {
            (f16_bytes.clone(), shape.clone())
        } else {
            let (si, e) = must_get(&embed_name)?;
            (
                tensor_to_f16_bytes(&e, bytes_of(si, &e), model_dir)?,
                e.shape.clone(),
            )
        };
        eprintln!(
            "[loader] lm_head_f16: {} elements ({:.1} MB)",
            shape.iter().product::<usize>(),
            buf.len() as f64 / 1e6
        );
        let region = arena.region("lm_head_f16", buf.len(), 16)?;
        unsafe { region.copy_from_host(&buf)? };
        F16Weight {
            offset_bytes: region.device_ptr(),
            shape,
        }
    };

    let sliding_rotary_dim = arch.head_dim_sliding;
    let (cos_s, sin_s) = rope_cos_sin_bytes(
        arch.head_dim_sliding,
        arch.max_position_embeddings,
        arch.rope_theta_sliding,
        sliding_rotary_dim,
    )?;
    let global_rotary_dim = arch.rotary_dim_for_layer(
        arch.layer_types
            .iter()
            .position(|t| *t == crate::gemma4_arch::Gemma4LayerType::GlobalAttention)
            .unwrap_or(0),
    );
    let (cos_g, sin_g) = rope_cos_sin_bytes(
        arch.head_dim_global,
        arch.max_position_embeddings,
        arch.rope_theta_global,
        global_rotary_dim,
    )?;

    let rope_cos_sliding = upload_rope(arena, "rope_cos_sliding", &cos_s)?;
    let rope_sin_sliding = upload_rope(arena, "rope_sin_sliding", &sin_s)?;
    let rope_cos_global = upload_rope(arena, "rope_cos_global", &cos_g)?;
    let rope_sin_global = upload_rope(arena, "rope_sin_global", &sin_g)?;

    let mut layers = Vec::with_capacity(arch.num_hidden_layers);
    for l in 0..arch.num_hidden_layers {
        let ln = |s: &str| format!("{prefix}.layers.{l}.{s}");

        let layer_hd = arch.head_dim_for_layer(l);
        let layer_nkvh = arch.num_kv_heads_for_layer(l);
        let layer_q_dim = arch.num_attention_heads * layer_hd;
        let layer_kv_dim = layer_nkvh * layer_hd;

        let q_name = ln("self_attn.q_proj.weight");
        let k_name = resolve_shared_kv_name(l, "self_attn.k_proj.weight")?;
        let v_name = if arch.layer_uses_k_for_v(l) {
            eprintln!("[loader] attention_k_eq_v: layer {l} V -> K");
            k_name.clone()
        } else {
            resolve_shared_kv_name(l, "self_attn.v_proj.weight")?
        };
        let q_host = load_host_f16(&q_name)?;
        let k_host = load_host_f16(&k_name)?;
        let v_host = load_host_f16(&v_name)?;
        expect_host_shape(
            &q_name,
            &q_host,
            &[layer_q_dim, arch.hidden_size],
            model_dir,
        )?;
        expect_host_shape(
            &k_name,
            &k_host,
            &[layer_kv_dim, arch.hidden_size],
            model_dir,
        )?;
        expect_host_shape(
            &v_name,
            &v_host,
            &[layer_kv_dim, arch.hidden_size],
            model_dir,
        )?;
        let qkv_rows = layer_q_dim + 2 * layer_kv_dim;

        let (qkv, o_proj, gate_up, down_proj) = if fp8_prequant {
            let q_tensor = must_get(&q_name)?;
            let k_tensor = must_get(&k_name)?;
            let v_tensor = must_get(&v_name)?;
            let q_scale = get_tensor(&ln("self_attn.q_proj.weight_scale"));
            let k_scale = get_tensor(&k_name.replace(".weight", ".weight_scale"));
            let v_scale = get_tensor(&v_name.replace(".weight", ".weight_scale"));
            let qkv = fuse_fp8_direct_channelscale(
                arena,
                "qkv",
                &[&q_tensor, &k_tensor, &v_tensor],
                &[q_scale.as_ref(), k_scale.as_ref(), v_scale.as_ref()],
                &shards,
                &[qkv_rows, arch.hidden_size],
                model_dir,
            )?;

            let o_entry = must_get(&ln("self_attn.o_proj.weight"))?;
            expect_tensor_shape(&o_entry.1, &[arch.hidden_size, layer_q_dim], model_dir)?;
            let o_scale = get_tensor(&ln("self_attn.o_proj.weight_scale"));
            let o_proj = upload_fp8_direct_channelscale(
                arena,
                "o_proj",
                &o_entry,
                o_scale.as_ref(),
                &shards,
                model_dir,
            )?;

            let gate_entry = must_get(&ln("mlp.gate_proj.weight"))?;
            let up_entry = must_get(&ln("mlp.up_proj.weight"))?;
            let expected_mlp_input_shape = [arch.intermediate_size, arch.hidden_size];
            expect_tensor_shape(&gate_entry.1, &expected_mlp_input_shape, model_dir)?;
            expect_tensor_shape(&up_entry.1, &expected_mlp_input_shape, model_dir)?;
            let gate_scale = get_tensor(&ln("mlp.gate_proj.weight_scale"));
            let up_scale = get_tensor(&ln("mlp.up_proj.weight_scale"));
            let gate_up = fuse_fp8_direct_channelscale(
                arena,
                "gate_up",
                &[&gate_entry, &up_entry],
                &[gate_scale.as_ref(), up_scale.as_ref()],
                &shards,
                &[2 * arch.intermediate_size, arch.hidden_size],
                model_dir,
            )?;

            let down_entry = must_get(&ln("mlp.down_proj.weight"))?;
            expect_tensor_shape(
                &down_entry.1,
                &[arch.hidden_size, arch.intermediate_size],
                model_dir,
            )?;
            let down_scale = get_tensor(&ln("mlp.down_proj.weight_scale"));
            let down_proj = upload_fp8_direct_channelscale(
                arena,
                "down_proj",
                &down_entry,
                down_scale.as_ref(),
                &shards,
                model_dir,
            )?;

            (qkv, o_proj, gate_up, down_proj)
        } else {
            {
                let split_fp8 = true;

                if split_fp8 {
                    // Split quantization: Q, K, V get separate per-tensor FP8 scales,
                    // then concatenate bytes + build a per-row channelscale vector.
                    let q_f16 = q_host.bytes.clone();
                    let k_f16 = k_host.bytes.clone();
                    let v_f16 = v_host.bytes.clone();

                    let q_f32 = f16_bytes_to_f32(&q_f16);
                    let k_f32 = f16_bytes_to_f32(&k_f16);
                    let v_f32 = f16_bytes_to_f32(&v_f16);

                    let q_q = quantize_per_tensor_ref(&q_f32);
                    let k_q = quantize_per_tensor_ref(&k_f32);
                    let v_q = quantize_per_tensor_ref(&v_f32);

                    if l == 0 {
                        eprintln!(
                            "[loader] split QKV scales: q={:.6e} k={:.6e} v={:.6e}",
                            q_q.scale, k_q.scale, v_q.scale
                        );
                    }

                    let q_rows = q_host.shape[0];
                    let k_rows = k_host.shape[0];
                    let v_rows = v_host.shape[0];

                    let q_fp8 = quantize_to_fp8_bytes(&q_f32, q_q.scale);
                    let k_fp8 = quantize_to_fp8_bytes(&k_f32, k_q.scale);
                    let v_fp8 = quantize_to_fp8_bytes(&v_f32, v_q.scale);

                    let mut fused_bytes =
                        Vec::with_capacity(q_fp8.len() + k_fp8.len() + v_fp8.len());
                    fused_bytes.extend_from_slice(&q_fp8);
                    fused_bytes.extend_from_slice(&k_fp8);
                    fused_bytes.extend_from_slice(&v_fp8);

                    let region = arena.region("qkv", fused_bytes.len(), 16)?;
                    unsafe { region.copy_from_host(&fused_bytes)? };

                    // Per-row channelscale: each row gets its sub-matrix's scale
                    let mut chscales: Vec<f32> = Vec::with_capacity(q_rows + k_rows + v_rows);
                    chscales.extend(std::iter::repeat(q_q.scale).take(q_rows));
                    chscales.extend(std::iter::repeat(k_q.scale).take(k_rows));
                    chscales.extend(std::iter::repeat(v_q.scale).take(v_rows));
                    let cs_bytes: Vec<u8> = chscales.iter().flat_map(|s| s.to_le_bytes()).collect();
                    let cs_r = arena.region("qkv_chscale", cs_bytes.len(), 16)?;
                    unsafe { cs_r.copy_from_host(&cs_bytes)? };

                    let one = 1.0f32;
                    let one_r = arena.region("qkv_scale_one", 4, 4)?;
                    unsafe { one_r.copy_from_host(&one.to_le_bytes())? };

                    let qkv = Fp8Weight {
                        offset_bytes: region.device_ptr(),
                        scale_ptr: one_r.device_ptr(),
                        shape: vec![qkv_rows, arch.hidden_size],
                        scale: 1.0,
                        clamp_ppm: 0.0,
                        dtype: DType::Fp8E4M3,
                        channelscale_ptr: Some(cs_r.device_ptr()),
                        blockscale_ptr: None,
                        blockscale_n_blocks: 0,
                        blockscale_k_blocks: 0,
                    };

                    // gate_up: same split treatment
                    let gate_name = ln("mlp.gate_proj.weight");
                    let up_name = ln("mlp.up_proj.weight");
                    let gate_host = load_host_f16(&gate_name)?;
                    let up_host = load_host_f16(&up_name)?;
                    expect_host_shape(
                        &gate_name,
                        &gate_host,
                        &[arch.intermediate_size, arch.hidden_size],
                        model_dir,
                    )?;
                    expect_host_shape(
                        &up_name,
                        &up_host,
                        &[arch.intermediate_size, arch.hidden_size],
                        model_dir,
                    )?;
                    let gate_f16 = gate_host.bytes;
                    let up_f16 = up_host.bytes;
                    let gate_f32 = f16_bytes_to_f32(&gate_f16);
                    let up_f32 = f16_bytes_to_f32(&up_f16);
                    let gate_qq = quantize_per_tensor_ref(&gate_f32);
                    let up_qq = quantize_per_tensor_ref(&up_f32);
                    let gate_rows = gate_host.shape[0];
                    let up_rows = up_host.shape[0];
                    let gate_fp8 = quantize_to_fp8_bytes(&gate_f32, gate_qq.scale);
                    let up_fp8_bytes = quantize_to_fp8_bytes(&up_f32, up_qq.scale);
                    let mut gu_bytes = Vec::with_capacity(gate_fp8.len() + up_fp8_bytes.len());
                    gu_bytes.extend_from_slice(&gate_fp8);
                    gu_bytes.extend_from_slice(&up_fp8_bytes);
                    let gu_r = arena.region("gate_up", gu_bytes.len(), 16)?;
                    unsafe { gu_r.copy_from_host(&gu_bytes)? };
                    let mut gu_scales: Vec<f32> = Vec::with_capacity(gate_rows + up_rows);
                    gu_scales.extend(std::iter::repeat(gate_qq.scale).take(gate_rows));
                    gu_scales.extend(std::iter::repeat(up_qq.scale).take(up_rows));
                    let gus_bytes: Vec<u8> =
                        gu_scales.iter().flat_map(|s| s.to_le_bytes()).collect();
                    let gus_r = arena.region("gu_chscale", gus_bytes.len(), 16)?;
                    unsafe { gus_r.copy_from_host(&gus_bytes)? };
                    let gu_one_r = arena.region("gu_scale_one", 4, 4)?;
                    unsafe { gu_one_r.copy_from_host(&one.to_le_bytes())? };
                    let gate_up = Fp8Weight {
                        offset_bytes: gu_r.device_ptr(),
                        scale_ptr: gu_one_r.device_ptr(),
                        shape: vec![2 * arch.intermediate_size, arch.hidden_size],
                        scale: 1.0,
                        clamp_ppm: 0.0,
                        dtype: DType::Fp8E4M3,
                        channelscale_ptr: Some(gus_r.device_ptr()),
                        blockscale_ptr: None,
                        blockscale_n_blocks: 0,
                        blockscale_k_blocks: 0,
                    };

                    // O-proj and down-proj: single matrix, per-tensor is fine
                    let o_name = ln("self_attn.o_proj.weight");
                    let o_host = load_host_f16(&o_name)?;
                    expect_host_shape(
                        &o_name,
                        &o_host,
                        &[arch.hidden_size, layer_q_dim],
                        model_dir,
                    )?;
                    let o_proj = upload_fp8(
                        arena,
                        "o_proj",
                        &o_host.bytes,
                        &o_host.shape,
                        &o_name,
                        model_dir,
                    )?;
                    let down_name = ln("mlp.down_proj.weight");
                    let down_host = load_host_f16(&down_name)?;
                    expect_host_shape(
                        &down_name,
                        &down_host,
                        &[arch.hidden_size, arch.intermediate_size],
                        model_dir,
                    )?;
                    let down_proj = upload_fp8(
                        arena,
                        "down_proj",
                        &down_host.bytes,
                        &down_host.shape,
                        &down_name,
                        model_dir,
                    )?;

                    (qkv, o_proj, gate_up, down_proj)
                } else {
                    // Original fused path
                    let qkv_f16_bytes =
                        concat_host_f16(&[q_name.clone(), k_name.clone(), v_name.clone()])?;
                    let qkv = upload_fp8(
                        arena,
                        "qkv",
                        &qkv_f16_bytes,
                        &[qkv_rows, arch.hidden_size],
                        &ln("self_attn.qkv.weight"),
                        model_dir,
                    )?;
                    let o_name = ln("self_attn.o_proj.weight");
                    let o_host = load_host_f16(&o_name)?;
                    let o_proj = upload_fp8(
                        arena,
                        "o_proj",
                        &o_host.bytes,
                        &o_host.shape,
                        &o_name,
                        model_dir,
                    )?;
                    let gate_up_f16_bytes =
                        concat_host_f16(&[ln("mlp.gate_proj.weight"), ln("mlp.up_proj.weight")])?;
                    let gate_up = upload_fp8(
                        arena,
                        "gate_up",
                        &gate_up_f16_bytes,
                        &[2 * arch.intermediate_size, arch.hidden_size],
                        &ln("mlp.gate_up.weight"),
                        model_dir,
                    )?;
                    let down_name = ln("mlp.down_proj.weight");
                    let down_host = load_host_f16(&down_name)?;
                    let down_proj = upload_fp8(
                        arena,
                        "down_proj",
                        &down_host.bytes,
                        &down_host.shape,
                        &down_name,
                        model_dir,
                    )?;
                    (qkv, o_proj, gate_up, down_proj)
                }
            }
        };

        let (qkv_f16_w, o_proj_f16_w, gate_up_f16_w, down_proj_f16_w) = (None, None, None, None);

        let input_layernorm = upload_f16("input_ln", &ln("input_layernorm.weight"))?;
        let post_attention_layernorm =
            upload_f16("post_attn_ln", &ln("post_attention_layernorm.weight"))?;
        let pre_feedforward_layernorm =
            upload_f16("pre_ff_ln", &ln("pre_feedforward_layernorm.weight"))?;
        let post_feedforward_layernorm =
            upload_f16("post_ff_ln", &ln("post_feedforward_layernorm.weight"))?;
        let post_per_layer_input_norm = if ple_enabled {
            Some(upload_f16(
                "post_ple_ln",
                &ln("post_per_layer_input_norm.weight"),
            )?)
        } else {
            None
        };

        let q_norm = upload_f16("q_norm", &ln("self_attn.q_norm.weight"))?;
        let k_norm_name = resolve_shared_kv_name(l, "self_attn.k_norm.weight")?;
        let k_norm = upload_f16("k_norm", &k_norm_name)?;

        let layer_scalar = upload_f16("layer_scalar", &ln("layer_scalar"))?;
        let per_layer_input_gate_f16 = if ple_enabled {
            Some(upload_f16(
                "ple_input_gate_f16",
                &ln("per_layer_input_gate.weight"),
            )?)
        } else {
            None
        };
        let per_layer_projection_f16 = if ple_enabled {
            Some(upload_f16(
                "ple_projection_f16",
                &ln("per_layer_projection.weight"),
            )?)
        } else {
            None
        };

        for (name, weight, expected) in [
            ("input_layernorm", &input_layernorm, arch.hidden_size),
            (
                "post_attention_layernorm",
                &post_attention_layernorm,
                arch.hidden_size,
            ),
            (
                "pre_feedforward_layernorm",
                &pre_feedforward_layernorm,
                arch.hidden_size,
            ),
            (
                "post_feedforward_layernorm",
                &post_feedforward_layernorm,
                arch.hidden_size,
            ),
            ("q_norm", &q_norm, layer_hd),
            ("k_norm", &k_norm, layer_hd),
            ("layer_scalar", &layer_scalar, 1),
        ] {
            if weight.shape != [expected] {
                return Err(loader_shape_mismatch(
                    &format!("layer {l} {name}"),
                    vec![expected],
                    weight.shape.clone(),
                    model_dir,
                ));
            }
        }
        if let Some(weight) = &post_per_layer_input_norm {
            if weight.shape != [arch.hidden_size] {
                return Err(loader_shape_mismatch(
                    &format!("layer {l} post_per_layer_input_norm"),
                    vec![arch.hidden_size],
                    weight.shape.clone(),
                    model_dir,
                ));
            }
        }
        if let Some(weight) = &per_layer_input_gate_f16 {
            if weight.shape != [arch.hidden_size_per_layer_input, arch.hidden_size] {
                return Err(loader_shape_mismatch(
                    &format!("layer {l} per_layer_input_gate"),
                    vec![arch.hidden_size_per_layer_input, arch.hidden_size],
                    weight.shape.clone(),
                    model_dir,
                ));
            }
        }
        if let Some(weight) = &per_layer_projection_f16 {
            if weight.shape != [arch.hidden_size, arch.hidden_size_per_layer_input] {
                return Err(loader_shape_mismatch(
                    &format!("layer {l} per_layer_projection"),
                    vec![arch.hidden_size, arch.hidden_size_per_layer_input],
                    weight.shape.clone(),
                    model_dir,
                ));
            }
        }

        if l < 2 {
            eprintln!(
                "[loader] layer {l} FP8: qkv_scale={:.6e} o={:.6e} gate_up={:.6e} down={:.6e}",
                qkv.scale, o_proj.scale, gate_up.scale, down_proj.scale,
            );
        }

        layers.push(Gemma4LayerWeights {
            qkv,
            o_proj,
            gate_up,
            down_proj,
            qkv_f16: qkv_f16_w,
            o_proj_f16: o_proj_f16_w,
            gate_up_f16: gate_up_f16_w,
            down_proj_f16: down_proj_f16_w,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_per_layer_input_norm,
            q_norm,
            k_norm,
            layer_scalar,
            per_layer_input_gate_f16,
            per_layer_projection_f16,
        });
    }

    Ok(Gemma4LoadedModel {
        embedding,
        lm_head_fp8,
        lm_head_f16,
        pruned_vocab,
        final_norm,
        embed_tokens_per_layer,
        per_layer_model_projection_f16,
        per_layer_projection_norm,
        rope_cos_sliding,
        rope_sin_sliding,
        rope_cos_global,
        rope_sin_global,
        layers,
    })
}

fn loader_corrupt(model_dir: &Path, tensor: Option<String>, detail: &str) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::Corrupt {
            detail: detail.to_string(),
        },
        ctx: LoaderCtx {
            path: model_dir.to_path_buf(),
            tensor,
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

fn read_model_aux(model_dir: &Path, path: &Path) -> Result<Vec<u8>> {
    let root = model_dir.canonicalize().map_err(|source| RvllmError::Io {
        err: rvllm_core::IoError::from(&source),
        path: model_dir.to_path_buf(),
        source,
    })?;
    let canonical = path.canonicalize().map_err(|source| RvllmError::Io {
        err: rvllm_core::IoError::from(&source),
        path: path.to_path_buf(),
        source,
    })?;
    if !canonical.starts_with(&root) {
        return Err(loader_corrupt(
            model_dir,
            None,
            "auxiliary model file escapes the model directory",
        ));
    }
    let file = std::fs::File::open(&canonical).map_err(|source| RvllmError::Io {
        err: rvllm_core::IoError::from(&source),
        path: canonical.clone(),
        source,
    })?;
    let metadata = file.metadata().map_err(|source| RvllmError::Io {
        err: rvllm_core::IoError::from(&source),
        path: canonical.clone(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(loader_corrupt(
            model_dir,
            None,
            "auxiliary model file is not a regular file",
        ));
    }
    let mut bytes = Vec::new();
    file.take(MAX_AUX_JSON_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: canonical.clone(),
            source,
        })?;
    if bytes.len() as u64 > MAX_AUX_JSON_BYTES {
        return Err(loader_corrupt(
            model_dir,
            None,
            "auxiliary model JSON exceeds the size limit",
        ));
    }
    Ok(bytes)
}

fn load_pruned_vocab(
    model_dir: &Path,
    arch_vocab: usize,
    embedding_rows: usize,
    lm_head_rows: usize,
) -> Result<Option<PrunedVocab>> {
    let path = {
        let p = model_dir.join("pruned_vocab.json");
        p.exists().then_some(p)
    }
    .or_else(|| {
        let p = model_dir.join("keepset.json");
        p.exists().then_some(p)
    });

    let Some(path) = path else {
        if lm_head_rows != arch_vocab {
            return Err(loader_corrupt(
                model_dir,
                Some("lm_head.weight".to_string()),
                &format!(
                    "lm_head rows {lm_head_rows} != config vocab {arch_vocab}, but no pruned vocabulary map was found"
                ),
            ));
        }
        return Ok(None);
    };

    let bytes = read_model_aux(model_dir, &path)?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        loader_corrupt(
            model_dir,
            None,
            &format!("{}: invalid keepset JSON: {e}", path.display()),
        )
    })?;
    let keep_arr = value
        .get("keep_ids")
        .and_then(|v| v.as_array())
        .ok_or_else(|| loader_corrupt(model_dir, None, "keepset JSON missing keep_ids array"))?;
    let keep_ids: Vec<u32> = keep_arr
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| loader_corrupt(model_dir, None, "keepset keep_ids contains non-u32"))
        })
        .collect::<Result<Vec<_>>>()?;
    let full_vocab = value
        .get("full_vocab")
        .or_else(|| value.get("vocab_size"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| {
            loader_corrupt(
                model_dir,
                None,
                "keepset JSON missing full_vocab/vocab_size",
            )
        })?;

    if full_vocab != arch_vocab {
        return Err(loader_corrupt(
            model_dir,
            None,
            &format!("keepset full_vocab {full_vocab} != config vocab {arch_vocab}"),
        ));
    }
    if embedding_rows != full_vocab {
        return Err(loader_corrupt(
            model_dir,
            Some("embed_tokens.weight".to_string()),
            &format!("embedding rows {embedding_rows} != keepset full_vocab {full_vocab}"),
        ));
    }
    if keep_ids.len() != lm_head_rows {
        return Err(loader_corrupt(
            model_dir,
            Some("lm_head.weight".to_string()),
            &format!(
                "keepset length {} != lm_head rows {lm_head_rows}",
                keep_ids.len()
            ),
        ));
    }

    let mut full_to_keep = vec![-1i32; full_vocab];
    for (row, &token_id) in keep_ids.iter().enumerate() {
        let token = token_id as usize;
        if token >= full_vocab {
            return Err(loader_corrupt(
                model_dir,
                None,
                &format!("keepset token id {token} >= full_vocab {full_vocab}"),
            ));
        }
        if full_to_keep[token] != -1 {
            return Err(loader_corrupt(
                model_dir,
                None,
                &format!("duplicate keepset token id {token}"),
            ));
        }
        full_to_keep[token] = row as i32;
    }

    eprintln!(
        "[loader] pruned vocabulary active: head_vocab={} full_vocab={} ({})",
        keep_ids.len(),
        full_vocab,
        path.display()
    );

    Ok(Some(PrunedVocab {
        full_vocab,
        head_vocab: keep_ids.len(),
        keep_ids,
        full_to_keep,
    }))
}

fn upload_rope(arena: &HbmArena, name: &'static str, data: &[u8]) -> Result<F16Weight> {
    let r = arena.region(name, data.len(), 16)?;
    unsafe { r.copy_from_host(data)? };
    Ok(F16Weight {
        offset_bytes: r.device_ptr(),
        shape: vec![data.len() / 2],
    })
}

fn rope_cos_sin_bytes(
    head_dim: usize,
    max_pos: usize,
    theta: f32,
    rotary_dim: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    if head_dim == 0
        || rotary_dim == 0
        || rotary_dim > head_dim
        || rotary_dim % 2 != 0
        || max_pos == 0
        || !theta.is_finite()
        || theta <= 0.0
    {
        return Err(loader_corrupt(
            Path::new("config.json"),
            None,
            "invalid RoPE geometry or theta",
        ));
    }
    let half = rotary_dim / 2;
    let bytes = max_pos
        .checked_mul(half)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| {
            loader_corrupt(Path::new("config.json"), None, "RoPE table size overflow")
        })?;
    let mut cos = Vec::new();
    cos.try_reserve_exact(bytes)
        .map_err(|_| loader_corrupt(Path::new("config.json"), None, "RoPE allocation failed"))?;
    let mut sin = Vec::new();
    sin.try_reserve_exact(bytes)
        .map_err(|_| loader_corrupt(Path::new("config.json"), None, "RoPE allocation failed"))?;
    // Proportional RoPE: frequencies use head_dim as divisor, not rotary_dim.
    // Only `half` frequencies are computed (partial rotation), but each
    // frequency value is spaced as if the full head_dim were rotated.
    let inv_theta: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    for pos in 0..max_pos {
        for &freq in &inv_theta {
            let angle = pos as f32 * freq;
            cos.extend_from_slice(&f16::from_f32(angle.cos()).to_le_bytes());
            sin.extend_from_slice(&f16::from_f32(angle.sin()).to_le_bytes());
        }
    }
    Ok((cos, sin))
}

fn tensor_to_f16_bytes(e: &TensorEntry, raw: &[u8], model_dir: &Path) -> Result<Vec<u8>> {
    match e.dtype {
        DType::F16 => Ok(raw.to_vec()),
        DType::Bf16 => Ok(bf16_to_f16(raw)),
        DType::F32 => Ok(f32_to_f16(raw)),
        DType::Fp8E4M3 => Ok(fp8e4m3_to_f16(raw)),
        _ => Err(RvllmError::Loader {
            err: LoaderError::DtypeMismatch {
                tensor: e.name.clone(),
                expected: DType::F16,
                got: e.dtype,
            },
            ctx: LoaderCtx {
                path: model_dir.to_path_buf(),
                tensor: Some(e.name.clone()),
            },
            bt: std::backtrace::Backtrace::capture(),
        }),
    }
}

fn packed_int4_tensor_to_f16_bytes(
    model_dir: &Path,
    packed_entry: &(usize, TensorEntry),
    scale_entry: &(usize, TensorEntry),
    shape_entry: &(usize, TensorEntry),
    shards: &[ShardMap],
) -> Result<(Vec<u8>, Vec<usize>)> {
    let (_, packed) = packed_entry;
    let (_, scale) = scale_entry;
    let (_, shape) = shape_entry;
    if packed.dtype != DType::I32 || packed.shape.len() != 2 {
        return Err(loader_corrupt(
            model_dir,
            Some(packed.name.clone()),
            &format!(
                "packed INT4 weight_packed must be I32 [rows, cols/8], got {:?} {:?}",
                packed.dtype, packed.shape
            ),
        ));
    }
    if shape.dtype != DType::I64 || shape.shape != [2] {
        return Err(loader_corrupt(
            model_dir,
            Some(shape.name.clone()),
            &format!(
                "packed INT4 weight_shape must be I64 [2], got {:?} {:?}",
                shape.dtype, shape.shape
            ),
        ));
    }

    let shape_raw = tensor_raw(shape_entry, shards);
    if shape_raw.len() != 16 {
        return Err(loader_corrupt(
            model_dir,
            Some(shape.name.clone()),
            "packed INT4 weight_shape must contain exactly two i64 values",
        ));
    }
    let rows = i64::from_le_bytes(shape_raw[0..8].try_into().unwrap());
    let cols = i64::from_le_bytes(shape_raw[8..16].try_into().unwrap());
    if rows <= 0 || cols <= 0 || cols % 8 != 0 {
        return Err(loader_corrupt(
            model_dir,
            Some(shape.name.clone()),
            &format!("invalid packed INT4 logical shape [{rows}, {cols}]"),
        ));
    }
    let rows = rows as usize;
    let cols = cols as usize;
    let packed_cols = cols / 8;
    if packed.shape != [rows, packed_cols] {
        return Err(loader_corrupt(
            model_dir,
            Some(packed.name.clone()),
            &format!(
                "packed INT4 shape {:?} does not match logical [{rows}, {cols}]",
                packed.shape
            ),
        ));
    }

    let scale_groups = match scale.shape.as_slice() {
        [r] if *r == rows => 1,
        [r, groups] if *r == rows && *groups > 0 => *groups,
        _ => {
            return Err(loader_corrupt(
                model_dir,
                Some(scale.name.clone()),
                &format!(
                    "packed INT4 weight_scale must be [rows] or [rows,groups], got {:?}",
                    scale.shape
                ),
            ))
        }
    };
    if cols % scale_groups != 0 || (cols / scale_groups) % 8 != 0 {
        return Err(loader_corrupt(
            model_dir,
            Some(scale.name.clone()),
            &format!(
                "packed INT4 scale groups {scale_groups} do not divide logical columns {cols} into 8-aligned groups"
            ),
        ));
    }

    let packed_raw = tensor_raw(packed_entry, shards);
    let scale_raw = tensor_raw(scale_entry, shards);
    let scales = scalar_tensor_to_f32(scale, scale_raw, model_dir)?;
    let expected_scales = rows.checked_mul(scale_groups).ok_or_else(|| {
        loader_corrupt(
            model_dir,
            Some(scale.name.clone()),
            "packed INT4 scale count overflow",
        )
    })?;
    if scales.len() != expected_scales {
        return Err(loader_corrupt(
            model_dir,
            Some(scale.name.clone()),
            &format!(
                "packed INT4 scale count {} != rows*groups {}",
                scales.len(),
                expected_scales
            ),
        ));
    }
    if scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(loader_corrupt(
            model_dir,
            Some(scale.name.clone()),
            "packed INT4 scales must be positive and finite",
        ));
    }

    Ok((
        decode_packed_int4_i32_to_f16(packed_raw, &scales, rows, cols, scale_groups),
        vec![rows, cols],
    ))
}

fn decode_packed_int4_i32_to_f16(
    packed_raw: &[u8],
    scales: &[f32],
    rows: usize,
    cols: usize,
    scale_groups: usize,
) -> Vec<u8> {
    use rayon::prelude::*;
    let packed_cols = cols / 8;
    let group_cols = cols / scale_groups;
    let mut out = vec![0u8; rows * cols * 2];
    out.par_chunks_mut(cols * 2)
        .enumerate()
        .for_each(|(row, out_row)| {
            for packed_col in 0..packed_cols {
                let base = (row * packed_cols + packed_col) * 4;
                let word = u32::from_le_bytes([
                    packed_raw[base],
                    packed_raw[base + 1],
                    packed_raw[base + 2],
                    packed_raw[base + 3],
                ]);
                for lane in 0..8 {
                    let nibble = ((word >> (lane * 4)) & 0x0f) as i8;
                    let q = nibble - 8;
                    let col = packed_col * 8 + lane;
                    let group = col / group_cols;
                    let scale = scales[row * scale_groups + group];
                    let value = f16::from_f32((q as f32) * scale).to_le_bytes();
                    let dst = col * 2;
                    out_row[dst] = value[0];
                    out_row[dst + 1] = value[1];
                }
            }
        });
    out
}

fn tensor_raw<'a>(entry: &(usize, TensorEntry), shards: &'a [ShardMap]) -> &'a [u8] {
    let (si, e) = entry;
    let s = shards[*si].bytes();
    let start = e.file_offset as usize;
    &s[start..start + e.nbytes as usize]
}

fn scalar_tensor_to_f32(e: &TensorEntry, raw: &[u8], model_dir: &Path) -> Result<Vec<f32>> {
    match e.dtype {
        DType::F16 => Ok(raw
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        DType::Bf16 => Ok(raw
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from_le_bytes([0, 0, c[0], c[1]])))
            .collect()),
        DType::F32 => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        _ => Err(RvllmError::Loader {
            err: LoaderError::DtypeMismatch {
                tensor: e.name.clone(),
                expected: DType::F16,
                got: e.dtype,
            },
            ctx: LoaderCtx {
                path: model_dir.to_path_buf(),
                tensor: Some(e.name.clone()),
            },
            bt: std::backtrace::Backtrace::capture(),
        }),
    }
}

fn fp8e4m3_to_f16(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() * 2);
    for &b in raw {
        out.extend_from_slice(&f16::from_f32(fp8_e4m3_to_f32(b)).to_le_bytes());
    }
    out
}

fn bf16_to_f16(raw: &[u8]) -> Vec<u8> {
    let n = raw.len() / 2;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let as_f32 = f32::from_bits(u32::from_le_bytes([0, 0, raw[2 * i], raw[2 * i + 1]]));
        out.extend_from_slice(&f16::from_f32(as_f32).to_le_bytes());
    }
    out
}

fn f32_to_f16(raw: &[u8]) -> Vec<u8> {
    let n = raw.len() / 4;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = f32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap());
        out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    out
}

fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    use rayon::prelude::*;
    bytes
        .par_chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

fn concat_tensors(
    entries: &[&(usize, TensorEntry)],
    shards: &[ShardMap],
    model_dir: &Path,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for &&(si, ref e) in entries {
        let raw = &shards[si].bytes()[e.file_offset as usize..(e.file_offset + e.nbytes) as usize];
        let buf = tensor_to_f16_bytes(e, raw, model_dir)?;
        out.extend_from_slice(&buf);
    }
    Ok(out)
}

fn upload_fp8(
    arena: &HbmArena,
    region_name: &'static str,
    f16_bytes: &[u8],
    shape: &[usize],
    tensor_name: &str,
    model_dir: &Path,
) -> Result<Fp8Weight> {
    let f32_vals = f16_bytes_to_f32(f16_bytes);
    let q = quantize_per_tensor_ref(&f32_vals);
    check_clamp_gate(tensor_name, q.clamp_ppm, model_dir)?;
    let fp8 = quantize_to_fp8_bytes(&f32_vals, q.scale);
    let region = arena.region(region_name, fp8.len(), 16)?;
    unsafe { region.copy_from_host(&fp8)? };
    let scale_region = arena.region("fp8_scale", 4, 4)?;
    unsafe { scale_region.copy_from_host(&q.scale.to_le_bytes())? };
    Ok(Fp8Weight {
        offset_bytes: region.device_ptr(),
        scale_ptr: scale_region.device_ptr(),
        shape: shape.to_vec(),
        scale: q.scale,
        clamp_ppm: q.clamp_ppm,
        dtype: DType::Fp8E4M3,
        channelscale_ptr: None,
        blockscale_ptr: None,
        blockscale_n_blocks: 0,
        blockscale_k_blocks: 0,
    })
}

fn upload_fp8_from(
    arena: &HbmArena,
    region_name: &'static str,
    (si, entry): &(usize, TensorEntry),
    shards: &[ShardMap],
    model_dir: &Path,
) -> Result<Fp8Weight> {
    let raw = {
        let s = shards[*si].bytes();
        let start = entry.file_offset as usize;
        &s[start..start + entry.nbytes as usize]
    };
    let f16_bytes = tensor_to_f16_bytes(entry, raw, model_dir)?;
    upload_fp8(
        arena,
        region_name,
        &f16_bytes,
        &entry.shape,
        &entry.name,
        model_dir,
    )
}

/// Decode an exact per-row BF16 scale vector.
fn read_channelscale_bf16(
    scale_entry: &(usize, TensorEntry),
    shards: &[ShardMap],
    rows: usize,
    model_dir: &Path,
) -> Result<Vec<f32>> {
    let (_, e) = scale_entry;
    if e.dtype != DType::Bf16 {
        return Err(loader_corrupt(
            model_dir,
            Some(e.name.clone()),
            &format!("FP8 scale dtype must be BF16, got {:?}", e.dtype),
        ));
    }
    if e.shape.as_slice() != [rows] && e.shape.as_slice() != [rows, 1] {
        return Err(loader_corrupt(
            model_dir,
            Some(e.name.clone()),
            &format!("per-row FP8 scale shape must be [{rows}] or [{rows}, 1]"),
        ));
    }
    let raw = tensor_raw(scale_entry, shards);
    if raw.len() != rows.saturating_mul(2) {
        return Err(loader_corrupt(
            model_dir,
            Some(e.name.clone()),
            "per-row FP8 scale byte length is invalid",
        ));
    }
    raw.chunks_exact(2)
        .enumerate()
        .map(|(i, chunk)| {
            let value = f32::from_bits(u32::from_le_bytes([0, 0, chunk[0], chunk[1]]));
            if !value.is_finite() || value <= 0.0 {
                return Err(loader_corrupt(
                    model_dir,
                    Some(e.name.clone()),
                    &format!("FP8 scale {i} must be positive and finite"),
                ));
            }
            Ok(value)
        })
        .collect()
}

/// Pure-function core of `read_blockscale_bf16`: bf16-LE bytes →
/// `Vec<f32>`. Lifted out so we can unit-test the decode path
/// without constructing a `ShardMap` (which needs a real mmap'd
/// file). The outer `read_blockscale_bf16` wrapper handles the
/// shape dispatch + ShardMap slicing.
fn decode_blockscale_bytes(raw: &[u8], rows_blocks: usize, cols_blocks: usize) -> Option<Vec<f32>> {
    let expected = rows_blocks.checked_mul(cols_blocks)?.checked_mul(2)?;
    if raw.len() != expected {
        return None;
    }
    let n = raw.len() / 2;
    let bf16_le_to_f32 = |lo: u8, hi: u8| f32::from_bits(u32::from_le_bytes([0, 0, lo, hi]));
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(bf16_le_to_f32(raw[2 * i], raw[2 * i + 1]));
    }
    Some(out)
}

/// Decode a 128x128 block-scale tensor without collapsing columns.
fn read_blockscale_bf16(
    scale_entry: &(usize, TensorEntry),
    shards: &[ShardMap],
    rows: usize,
    cols: usize,
    model_dir: &Path,
) -> Result<Option<(Vec<f32>, usize, usize)>> {
    let (_, e) = scale_entry;
    if e.shape.len() != 2 || e.shape[1] == 1 {
        return Ok(None);
    }
    if e.dtype != DType::Bf16 {
        return Err(loader_corrupt(
            model_dir,
            Some(e.name.clone()),
            &format!("FP8 block scale dtype must be BF16, got {:?}", e.dtype),
        ));
    }
    let rows_blocks = e.shape[0];
    let cols_blocks = e.shape[1];
    let expected_rows = rows.div_ceil(128);
    let expected_cols = cols.div_ceil(128);
    if [rows_blocks, cols_blocks] != [expected_rows, expected_cols] {
        return Err(loader_shape_mismatch(
            &e.name,
            vec![expected_rows, expected_cols],
            e.shape.clone(),
            model_dir,
        ));
    }
    let decoded =
        decode_blockscale_bytes(tensor_raw(scale_entry, shards), rows_blocks, cols_blocks)
            .ok_or_else(|| {
                loader_corrupt(
                    model_dir,
                    Some(e.name.clone()),
                    "FP8 block scale byte length is invalid",
                )
            })?;
    if decoded.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(loader_corrupt(
            model_dir,
            Some(e.name.clone()),
            "FP8 block scales must be positive and finite",
        ));
    }
    Ok(Some((decoded, rows_blocks, cols_blocks)))
}

#[cfg(test)]
mod blockscale_tests {
    use super::*;

    // bf16 = upper 16 bits of f32.
    fn f32_to_bf16_le(x: f32) -> [u8; 2] {
        let bits = x.to_bits();
        let hi = ((bits >> 16) & 0xFFFF) as u16;
        hi.to_le_bytes()
    }

    #[test]
    fn decode_blockscale_roundtrips_representative_values() {
        // Two row-blocks × three col-blocks = 6 bf16 scales. Values
        // chosen so they survive bf16 round-trip exactly (powers of 2
        // + simple fractions).
        let src = [0.25_f32, 0.5, 1.0, 2.0, 4.0, 0.125];
        let mut raw = Vec::with_capacity(src.len() * 2);
        for v in &src {
            raw.extend_from_slice(&f32_to_bf16_le(*v));
        }

        let out = decode_blockscale_bytes(&raw, 2, 3).expect("shape fits");
        assert_eq!(out, src);
    }

    #[test]
    fn decode_blockscale_rejects_shape_mismatch() {
        // 8 bytes = 4 bf16 values, but caller claims 2x3 = 6. Should
        // reject rather than silently read garbage.
        let raw = vec![0u8; 8];
        assert!(decode_blockscale_bytes(&raw, 2, 3).is_none());
    }

    #[test]
    fn decode_blockscale_preserves_layout_order() {
        // Regression guard: read order must be row-major, matching the
        // on-disk safetensors layout. This is the exact bug the PR
        // reviewer flagged — `channelscale_ptr` projected the 2-D
        // tensor to per-row via `rb * cols_blocks` indexing. Check
        // the full blockscale preserves column ordering too.
        let src: Vec<f32> = (0..6).map(|i| (i as f32) * 0.5).collect();
        let raw: Vec<u8> = src.iter().flat_map(|v| f32_to_bf16_le(*v)).collect();
        let out = decode_blockscale_bytes(&raw, 2, 3).unwrap();
        // Expect row 0: [0.0, 0.5, 1.0], row 1: [1.5, 2.0, 2.5].
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.5);
        assert_eq!(out[2], 1.0);
        assert_eq!(out[3], 1.5);
        assert_eq!(out[4], 2.0);
        assert_eq!(out[5], 2.5);
    }

    #[test]
    fn decode_packed_int4_nibbles_to_f16() {
        let mut word = 0u32;
        for lane in 0..8 {
            word |= (lane as u32) << (lane * 4);
        }
        let mut word2 = 0u32;
        for lane in 0..8 {
            word2 |= ((lane + 8) as u32) << (lane * 4);
        }
        let mut packed = Vec::new();
        packed.extend_from_slice(&word.to_le_bytes());
        packed.extend_from_slice(&word2.to_le_bytes());

        let out = decode_packed_int4_i32_to_f16(&packed, &[0.5], 1, 16, 1);
        let vals: Vec<f32> = out
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        assert_eq!(
            vals,
            vec![
                -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                3.5,
            ]
        );
    }

    #[test]
    fn decode_packed_int4_group_scales_to_f16() {
        let mut word = 0u32;
        for lane in 0..8 {
            word |= (1u32) << (lane * 4);
        }
        let mut packed = Vec::new();
        packed.extend_from_slice(&word.to_le_bytes());
        packed.extend_from_slice(&word.to_le_bytes());

        let out = decode_packed_int4_i32_to_f16(&packed, &[0.5, 2.0], 1, 16, 2);
        let vals: Vec<f32> = out
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        assert_eq!(
            vals,
            vec![
                -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -14.0, -14.0, -14.0, -14.0, -14.0,
                -14.0, -14.0, -14.0,
            ]
        );
    }
}

/// Upload pre-quantized FP8 weight with per-channel BF16 scales.
/// Raw FP8 bytes go straight to GPU. Per-channel scales uploaded as f32
/// vector. Weight scalar scale set to 1.0 -- channelscale applied post-GEMM.
fn upload_fp8_direct_channelscale(
    arena: &HbmArena,
    region_name: &'static str,
    (si, entry): &(usize, TensorEntry),
    scale_entry: Option<&(usize, TensorEntry)>,
    shards: &[ShardMap],
    model_dir: &Path,
) -> Result<Fp8Weight> {
    if entry.dtype != DType::Fp8E4M3 || entry.shape.len() != 2 {
        return Err(loader_corrupt(
            model_dir,
            Some(entry.name.clone()),
            "direct FP8 weight must be an FP8 E4M3 rank-2 tensor",
        ));
    }
    let raw = tensor_raw(&(*si, entry.clone()), shards);
    let rows = entry.shape[0];
    let cols = entry.shape[1];
    let scale_entry = scale_entry.ok_or_else(|| {
        loader_corrupt(
            model_dir,
            Some(entry.name.clone()),
            "direct FP8 weight requires an explicit scale tensor",
        )
    })?;
    let region = arena.region(region_name, raw.len(), 16)?;
    unsafe { region.copy_from_host(raw)? };
    let one = 1.0f32;
    let one_r = arena.region("fp8_scale", 4, 4)?;
    unsafe { one_r.copy_from_host(&one.to_le_bytes())? };

    if let Some((block_scales, n_blocks, k_blocks)) =
        read_blockscale_bf16(scale_entry, shards, rows, cols, model_dir)?
    {
        let bytes: Vec<u8> = block_scales
            .iter()
            .flat_map(|scale| scale.to_le_bytes())
            .collect();
        let block_region = arena.region("fp8_blockscale", bytes.len(), 16)?;
        unsafe { block_region.copy_from_host(&bytes)? };
        Ok(Fp8Weight {
            offset_bytes: region.device_ptr(),
            scale_ptr: one_r.device_ptr(),
            shape: entry.shape.clone(),
            scale: 1.0,
            clamp_ppm: 0.0,
            dtype: DType::Fp8E4M3,
            channelscale_ptr: None,
            blockscale_ptr: Some(block_region.device_ptr()),
            blockscale_n_blocks: u32::try_from(n_blocks).map_err(|_| {
                loader_corrupt(model_dir, Some(entry.name.clone()), "block rows exceed u32")
            })?,
            blockscale_k_blocks: u32::try_from(k_blocks).map_err(|_| {
                loader_corrupt(
                    model_dir,
                    Some(entry.name.clone()),
                    "block columns exceed u32",
                )
            })?,
        })
    } else {
        let ch_scales = read_channelscale_bf16(scale_entry, shards, rows, model_dir)?;
        let scale_bytes: Vec<u8> = ch_scales.iter().flat_map(|s| s.to_le_bytes()).collect();
        let cs_r = arena.region("fp8_chscale", scale_bytes.len(), 16)?;
        unsafe { cs_r.copy_from_host(&scale_bytes)? };
        Ok(Fp8Weight {
            offset_bytes: region.device_ptr(),
            scale_ptr: one_r.device_ptr(),
            shape: entry.shape.clone(),
            scale: 1.0,
            clamp_ppm: 0.0,
            dtype: DType::Fp8E4M3,
            channelscale_ptr: Some(cs_r.device_ptr()),
            blockscale_ptr: None,
            blockscale_n_blocks: 0,
            blockscale_k_blocks: 0,
        })
    }
}

/// Fuse multiple pre-quantized FP8 tensors (QKV, gate+up) with per-channel
/// scales. Raw FP8 bytes concatenated, per-channel scale vectors concatenated.
/// Weight scalar scale = 1.0, channelscale applied post-GEMM.
fn fuse_fp8_direct_channelscale(
    arena: &HbmArena,
    region_name: &'static str,
    parts: &[&(usize, TensorEntry)],
    scale_entries: &[Option<&(usize, TensorEntry)>],
    shards: &[ShardMap],
    fused_shape: &[usize],
    model_dir: &Path,
) -> Result<Fp8Weight> {
    if parts.is_empty() || parts.len() != scale_entries.len() || fused_shape.len() != 2 {
        return Err(loader_corrupt(
            model_dir,
            None,
            "invalid fused FP8 part or scale count",
        ));
    }
    let mut fused_bytes = Vec::new();
    let mut fused_scales: Vec<f32> = Vec::new();
    let mut fused_rows = 0usize;
    let fused_cols = fused_shape[1];

    for (i, &(si, ref entry)) in parts.iter().enumerate() {
        if entry.dtype != DType::Fp8E4M3 || entry.shape.len() != 2 || entry.shape[1] != fused_cols {
            return Err(loader_corrupt(
                model_dir,
                Some(entry.name.clone()),
                "fused FP8 parts must be FP8 E4M3 matrices with a common column count",
            ));
        }
        let raw = tensor_raw(&(*si, entry.clone()), shards);
        fused_bytes.extend_from_slice(raw);
        let rows = entry.shape[0];
        fused_rows = fused_rows
            .checked_add(rows)
            .ok_or_else(|| loader_corrupt(model_dir, None, "fused FP8 row count overflow"))?;
        let scale_entry = scale_entries[i].ok_or_else(|| {
            loader_corrupt(
                model_dir,
                Some(entry.name.clone()),
                "fused FP8 part requires an explicit per-row scale",
            )
        })?;
        let ch = read_channelscale_bf16(scale_entry, shards, rows, model_dir)?;
        fused_scales.extend_from_slice(&ch);
    }
    if fused_rows != fused_shape[0] {
        return Err(loader_corrupt(
            model_dir,
            None,
            "fused FP8 row count does not match the declared shape",
        ));
    }

    let region = arena.region(region_name, fused_bytes.len(), 16)?;
    unsafe { region.copy_from_host(&fused_bytes)? };

    let scale_bytes: Vec<u8> = fused_scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let cs_r = arena.region("fp8_chscale", scale_bytes.len(), 16)?;
    unsafe { cs_r.copy_from_host(&scale_bytes)? };
    let one = 1.0f32;
    let one_r = arena.region("fp8_scale", 4, 4)?;
    unsafe { one_r.copy_from_host(&one.to_le_bytes())? };

    Ok(Fp8Weight {
        offset_bytes: region.device_ptr(),
        scale_ptr: one_r.device_ptr(),
        shape: fused_shape.to_vec(),
        scale: 1.0,
        clamp_ppm: 0.0,
        dtype: DType::Fp8E4M3,
        channelscale_ptr: Some(cs_r.device_ptr()),
        // Fused qkv / gate_up synthesis collapses per-part scales into
        // one concatenated per-row vector; 2-D blockscale reconstruction
        // across parts isn't well-defined (different parts ship with
        // different block alignments), so these weights never take
        // the blockscale fast path. Synthesised → `blockscale_ptr =
        // None`, any GEMM path that reads `blockscale_ptr` must fall
        // back to the channelscale-preserving path.
        blockscale_ptr: None,
        blockscale_n_blocks: 0,
        blockscale_k_blocks: 0,
    })
}

fn quantize_to_fp8_bytes(f32_vals: &[f32], scale: f32) -> Vec<u8> {
    use rayon::prelude::*;
    let inv = 1.0 / scale;
    f32_vals
        .par_iter()
        .map(|v| fp8_e4m3_encode((*v * inv).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)))
        .collect()
}

fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let s = (b >> 7) & 1;
    let e = (b >> 3) & 0xF;
    let m = b & 0x7;
    let val = if e == 0 {
        if m == 0 {
            0.0f32
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

// Packed INT4 + per-layer-embedding loader. All dimensions and quantization
// group sizes come from the checkpoint configuration and tensor metadata.

use crate::gemma4_weights::{E4bLayerWeights, PleTables, PrunedLmHead, WPacked};

/// Parsed `quantization_config` group params we need at load.
#[derive(Clone, Copy, Debug)]
struct QuantParams {
    group_size: usize,
    num_bits: u32,
    symmetric: bool,
}

fn loader_shape_mismatch(
    tensor: &str,
    expected: Vec<usize>,
    got: Vec<usize>,
    model_dir: &Path,
) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::ShapeMismatch {
            tensor: tensor.to_string(),
            expected,
            got,
        },
        ctx: LoaderCtx {
            path: model_dir.to_path_buf(),
            tensor: Some(tensor.to_string()),
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}

fn expect_host_shape(
    tensor: &str,
    host: &HostF16Tensor,
    expected: &[usize],
    model_dir: &Path,
) -> Result<()> {
    expect_shape(tensor, &host.shape, expected, model_dir)
}

fn expect_shape(tensor: &str, got: &[usize], expected: &[usize], model_dir: &Path) -> Result<()> {
    if got != expected {
        return Err(loader_shape_mismatch(
            tensor,
            expected.to_vec(),
            got.to_vec(),
            model_dir,
        ));
    }
    Ok(())
}

fn expect_tensor_shape(entry: &TensorEntry, expected: &[usize], model_dir: &Path) -> Result<()> {
    expect_shape(&entry.name, &entry.shape, expected, model_dir)
}

fn expect_lm_head_shape(
    tensor: &str,
    got: &[usize],
    expected_rows: usize,
    hidden_size: usize,
    model_dir: &Path,
) -> Result<()> {
    expect_shape(tensor, got, &[expected_rows, hidden_size], model_dir)
}

#[cfg(test)]
mod weight_shape_tests {
    use super::*;

    #[test]
    fn model_aux_rejects_non_regular_file() {
        let root = std::env::temp_dir().join(format!(
            "rvllm-model-aux-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&root).unwrap();
        let err = read_model_aux(&root, &root).unwrap_err();
        std::fs::remove_dir(&root).unwrap();
        match err {
            RvllmError::Loader {
                err: LoaderError::Corrupt { detail },
                ..
            } => assert_eq!(detail, "auxiliary model file is not a regular file"),
            err => panic!("expected corrupt auxiliary file error, got {err:?}"),
        }
    }

    fn fp8_entry(name: &str, shape: Vec<usize>) -> TensorEntry {
        TensorEntry {
            name: name.to_string(),
            dtype: DType::Fp8E4M3,
            shape,
            file_offset: 0,
            nbytes: 0,
        }
    }

    fn assert_shape_mismatch(name: &str, got: Vec<usize>, expected: Vec<usize>) {
        let entry = fp8_entry(name, got.clone());
        match expect_tensor_shape(&entry, &expected, Path::new("/model")).unwrap_err() {
            RvllmError::Loader {
                err:
                    LoaderError::ShapeMismatch {
                        tensor,
                        expected: actual_expected,
                        got: actual_got,
                    },
                ..
            } => {
                assert_eq!(tensor, name);
                assert_eq!(actual_expected, expected);
                assert_eq!(actual_got, got);
            }
            err => panic!("expected loader shape mismatch, got {err:?}"),
        }
    }

    #[test]
    fn direct_fp8_o_proj_rejects_malformed_arch_shape() {
        assert_shape_mismatch("o_proj.weight", vec![3_072, 3_072], vec![3_072, 4_096]);
    }

    #[test]
    fn direct_fp8_down_proj_rejects_malformed_arch_shape() {
        assert_shape_mismatch("down_proj.weight", vec![3_072, 12_287], vec![3_072, 12_288]);
    }

    #[test]
    fn direct_fp8_lm_head_rejects_malformed_arch_shape() {
        assert_shape_mismatch("lm_head.weight", vec![262_143, 3_072], vec![262_144, 3_072]);
    }

    #[test]
    fn direct_fp8_gate_up_rejects_offsetting_row_mismatches() {
        let expected = vec![12_288, 3_072];
        let gate_shape = vec![12_287, 3_072];
        let up_shape = vec![12_289, 3_072];
        assert_eq!(gate_shape[0] + up_shape[0], 2 * expected[0]);
        assert_shape_mismatch("gate_proj.weight", gate_shape, expected.clone());
        assert_shape_mismatch("up_proj.weight", up_shape, expected);
    }

    #[test]
    fn explicit_non_fp8_lm_head_rejects_correct_rows_wrong_columns() {
        assert!(expect_lm_head_shape(
            "lm_head.weight",
            &[262_144, 3_071],
            262_144,
            3_072,
            Path::new("/model"),
        )
        .is_err());
    }

    #[test]
    fn packed_lm_head_rejects_correct_rows_wrong_columns() {
        assert!(expect_lm_head_shape(
            "lm_head.weight_packed(dequant)",
            &[262_144, 3_073],
            262_144,
            3_072,
            Path::new("/model"),
        )
        .is_err());
    }
}

/// Read and validate the packed INT4 quantization configuration.
fn parse_quant_params(model_dir: &Path) -> Result<QuantParams> {
    let p = model_dir.join("config.json");
    let bytes = read_model_aux(model_dir, &p)?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| loader_corrupt(model_dir, None, &format!("config.json: {e}")))?;
    let qc = &v["quantization_config"];
    if qc.is_null() {
        return Err(loader_corrupt(
            model_dir,
            None,
            "packed INT4 loader requires quantization_config",
        ));
    }
    let method = qc["quant_method"].as_str().unwrap_or("");
    let format = qc["format"].as_str().unwrap_or("");
    if method != "compressed-tensors" || format != "pack-quantized" {
        return Err(loader_corrupt(
            model_dir,
            None,
            &format!("expected compressed-tensors/pack-quantized, got {method}/{format}"),
        ));
    }
    let g0 = &qc["config_groups"]["group_0"]["weights"];
    let group_size = g0["group_size"].as_u64().unwrap_or(0) as usize;
    let num_bits = g0["num_bits"].as_u64().unwrap_or(0) as u32;
    let symmetric = g0["symmetric"].as_bool().unwrap_or(false);
    if group_size == 0 || num_bits != 4 || !symmetric {
        return Err(loader_corrupt(
            model_dir,
            None,
            &format!(
                "packed INT4 requires a nonzero group_size, num_bits=4, symmetric=true; \
                 got group_size={group_size}, num_bits={num_bits}, symmetric={symmetric}"
            ),
        ));
    }
    Ok(QuantParams {
        group_size,
        num_bits,
        symmetric,
    })
}

/// Decode an `I64 [2]` `weight_shape` tensor into `[out, in]`.
fn decode_weight_shape(raw: &[u8], name: &str, model_dir: &Path) -> Result<[usize; 2]> {
    if raw.len() != 16 {
        return Err(loader_corrupt(
            model_dir,
            Some(name.to_string()),
            &format!(
                "{name}: weight_shape must be I64[2] (16 bytes), got {} bytes",
                raw.len()
            ),
        ));
    }
    let out = i64::from_le_bytes(raw[0..8].try_into().unwrap());
    let inn = i64::from_le_bytes(raw[8..16].try_into().unwrap());
    if out <= 0 || inn <= 0 {
        return Err(loader_corrupt(
            model_dir,
            Some(name.to_string()),
            &format!("{name}: weight_shape has non-positive dim [{out}, {inn}]"),
        ));
    }
    Ok([out as usize, inn as usize])
}

/// Load a packed INT4 + per-layer-embedding model from `model_dir`.
///
/// Fails loud on any layout mismatch (missing tensor, wrong packed/scale
/// geometry, ignore-list collision, and KV-share inconsistency). The PLE
/// embedding scale is always folded into the uploaded table.
pub fn load_gemma4_e4b_model(
    model_dir: &Path,
    arena: &HbmArena,
    arch: &Gemma4Arch,
) -> Result<E4bLoadedModel> {
    if !arch.is_e4b() {
        return Err(loader_corrupt(
            model_dir,
            None,
            "packed PLE loader requires hidden_size_per_layer_input > 0",
        ));
    }
    let quant = parse_quant_params(model_dir)?;

    let idx = ShardIndex::resolve(model_dir)?;
    let mut shards = Vec::with_capacity(idx.shards.len());
    for p in &idx.shards {
        shards.push(ShardMap::open(p)?);
    }
    let mut tensors: BTreeMap<String, (usize, TensorEntry)> = BTreeMap::new();
    for (si, sm) in shards.iter().enumerate() {
        for (name, entry) in &sm.header.tensors {
            if tensors.insert(name.clone(), (si, entry.clone())).is_some() {
                return Err(loader_corrupt(
                    model_dir,
                    Some(name.clone()),
                    "duplicate tensor name across shards",
                ));
            }
        }
    }
    let bytes_of = |si: usize, e: &TensorEntry| -> &[u8] {
        let s = shards[si].bytes();
        let start = e.file_offset as usize;
        &s[start..start + e.nbytes as usize]
    };
    let get = |name: &str| -> Option<(usize, TensorEntry)> { tensors.get(name).cloned() };
    let must = |name: &str| -> Result<(usize, TensorEntry)> {
        get(name).ok_or_else(|| RvllmError::Loader {
            err: LoaderError::MissingTensor {
                name: name.to_string(),
            },
            ctx: LoaderCtx {
                path: model_dir.to_path_buf(),
                tensor: Some(name.to_string()),
            },
            bt: std::backtrace::Backtrace::capture(),
        })
    };

    let prefix = &arch.weight_prefix;

    // --- ignore-list sanity: no language-model decoder Linear may be on it ---
    {
        let cfg_bytes = read_model_aux(model_dir, &model_dir.join("config.json"))?;
        let cfg: serde_json::Value = serde_json::from_slice(&cfg_bytes)
            .map_err(|e| loader_corrupt(model_dir, None, &format!("config.json: {e}")))?;
        if let Some(ignore) = cfg["quantization_config"]["ignore"].as_array() {
            for m in ignore {
                let s = m.as_str().unwrap_or("");
                if s.contains("language_model") || s == "lm_head" {
                    return Err(loader_corrupt(
                        model_dir,
                        Some(s.to_string()),
                        &format!(
                            "quantization ignore-list contains language-model module {s:?}; \
                             — loader assumes all decoder Linears are INT4-quantized"
                        ),
                    ));
                }
            }
        }
    }

    // ----- raw bf16 upload (no f16 conversion): PLE table needs true bf16 -----
    let upload_bf16_raw = |name: &'static str, e: &TensorEntry, raw: &[u8]| -> Result<F16Weight> {
        if e.dtype != DType::Bf16 {
            return Err(RvllmError::Loader {
                err: LoaderError::DtypeMismatch {
                    tensor: e.name.clone(),
                    expected: DType::Bf16,
                    got: e.dtype,
                },
                ctx: LoaderCtx {
                    path: model_dir.to_path_buf(),
                    tensor: Some(e.name.clone()),
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let region = arena.region(name, raw.len(), 16)?;
        unsafe { region.copy_from_host(raw)? };
        Ok(F16Weight {
            offset_bytes: region.device_ptr(),
            shape: e.shape.clone(),
        })
    };

    // Convert a (bf16/f16/f32) tensor to F16 and upload. Use for norm gammas
    // and any weight a half-precision kernel consumes. Uploading BF16 bytes
    // into an F16 region changes their values. `upload_bf16_raw` stays for weights a
    // bf16-reading kernel consumes (PLE gate post_norm, layer_scalar).
    let upload_bf16_to_f16 =
        |name: &'static str, e: &TensorEntry, raw: &[u8]| -> Result<F16Weight> {
            let f16_bytes = tensor_to_f16_bytes(e, raw, model_dir)?;
            let region = arena.region(name, f16_bytes.len(), 16)?;
            unsafe { region.copy_from_host(&f16_bytes)? };
            Ok(F16Weight {
                offset_bytes: region.device_ptr(),
                shape: e.shape.clone(),
            })
        };

    // ----- one INT4 pack-quantized Linear -----
    // `expect_group`: group_size for this Linear (the configured value by default,
    // `in` for the channel-strategy lm_head where scale is [out,1]).
    let load_wpacked =
        |region: &'static str, base: &str, expect_group: Option<usize>| -> Result<WPacked> {
            let pname = format!("{base}.weight_packed");
            let sname = format!("{base}.weight_scale");
            let hname = format!("{base}.weight_shape");
            let (psi, pe) = must(&pname)?;
            let (ssi, se) = must(&sname)?;
            let (hsi, he) = must(&hname)?;

            if pe.dtype != DType::I32 {
                return Err(RvllmError::Loader {
                    err: LoaderError::DtypeMismatch {
                        tensor: pname.clone(),
                        expected: DType::I32,
                        got: pe.dtype,
                    },
                    ctx: LoaderCtx {
                        path: model_dir.to_path_buf(),
                        tensor: Some(pname.clone()),
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
            if se.dtype != DType::F16 {
                return Err(RvllmError::Loader {
                    err: LoaderError::DtypeMismatch {
                        tensor: sname.clone(),
                        expected: DType::F16,
                        got: se.dtype,
                    },
                    ctx: LoaderCtx {
                        path: model_dir.to_path_buf(),
                        tensor: Some(sname.clone()),
                    },
                    bt: std::backtrace::Backtrace::capture(),
                });
            }
            let shape = decode_weight_shape(bytes_of(hsi, &he), &hname, model_dir)?;
            let [out, inn] = shape;

            // packed: [out, in/8]
            let packed_cols = inn / 8;
            if inn % 8 != 0 {
                return Err(loader_corrupt(
                    model_dir,
                    Some(pname.to_string()),
                    &format!("{pname}: in_features {inn} not divisible by 8 (int4 pack)"),
                ));
            }
            if pe.shape != vec![out, packed_cols] {
                return Err(loader_shape_mismatch(
                    &pname,
                    vec![out, packed_cols],
                    pe.shape.clone(),
                    model_dir,
                ));
            }

            // scale: [out, in/group]
            let group_size = expect_group.unwrap_or(quant.group_size);
            let scale_groups =
                if group_size >= inn {
                    1 // channel strategy: one scale per row
                } else {
                    if inn % group_size != 0 {
                        return Err(loader_corrupt(
                    model_dir,
                    Some(sname.to_string()),
                    &format!("{sname}: in_features {inn} not divisible by group_size {group_size}"),
                ));
                    }
                    inn / group_size
                };
            if se.shape != vec![out, scale_groups] {
                return Err(loader_shape_mismatch(
                    &sname,
                    vec![out, scale_groups],
                    se.shape.clone(),
                    model_dir,
                ));
            }

            let praw = bytes_of(psi, &pe);
            let preg = arena.region(region, praw.len(), 16)?;
            unsafe { preg.copy_from_host(praw)? };
            let sraw = bytes_of(ssi, &se);
            let sreg = arena.region(region, sraw.len(), 16)?;
            unsafe { sreg.copy_from_host(sraw)? };

            WPacked::new(
                preg.device_ptr(),
                sreg.device_ptr(),
                shape,
                packed_cols,
                scale_groups,
                if group_size >= inn { inn } else { group_size },
                quant.num_bits,
                quant.symmetric,
            )
            .map_err(|detail| loader_corrupt(model_dir, Some(base.to_string()), &detail))
        };

    // ===== global tensors =====
    // Embed tokens are pre-scaled by sqrt(hidden) during upload.
    let embed_name = format!("{prefix}.embed_tokens.weight");
    let embedding = {
        let (si, e) = must(&embed_name)?;
        if e.dtype != DType::Bf16 {
            return Err(loader_corrupt(
                model_dir,
                Some(embed_name.to_string()),
                &format!(
                    "{embed_name}: expected BF16 embed_tokens, got {:?}",
                    e.dtype
                ),
            ));
        }
        let raw = bytes_of(si, &e);
        let scale = (arch.hidden_size as f32).sqrt();
        // Scale by sqrt(hidden) and store as F16 — the embedding_gather_f16
        // kernel reads this table as `__half`. Storing bf16 bytes here (the old
        // path) mis-read every embedding row as f16 (bf16 4.28 != f16 4.28),
        // scrambling the residual stream at model input.
        let n = raw.len() / 2;
        let mut buf = Vec::with_capacity(raw.len());
        for i in 0..n {
            let bf = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
            let f = f32::from_bits((bf as u32) << 16);
            let scaled = f * scale;
            buf.extend_from_slice(&f16::from_f32(scaled).to_le_bytes());
        }
        let region = arena.region("e4b_embedding", buf.len(), 16)?;
        unsafe { region.copy_from_host(&buf)? };
        F16Weight {
            offset_bytes: region.device_ptr(),
            shape: e.shape.clone(),
        }
    };

    let final_norm = {
        let nname = format!("{prefix}.norm.weight");
        let (si, e) = must(&nname)?;
        upload_bf16_to_f16("e4b_final_norm", &e, bytes_of(si, &e))?
    };

    // ===== PLE tables =====
    let ple = {
        let table_name = format!("{prefix}.embed_tokens_per_layer.weight");
        let (si, e) = must(&table_name)?;
        let expected = vec![
            arch.vocab_size_per_layer_input,
            arch.per_layer_embed_total(),
        ];
        if e.shape != expected {
            return Err(loader_shape_mismatch(
                &table_name,
                expected,
                e.shape.clone(),
                model_dir,
            ));
        }
        if e.dtype != DType::Bf16 {
            return Err(loader_corrupt(
                model_dir,
                Some(table_name.to_string()),
                &format!("{table_name}: expected BF16 PLE table, got {:?}", e.dtype),
            ));
        }
        let embed_scale = arch.ple_embed_scale();
        let fold = true;
        let raw = bytes_of(si, &e);
        let table = if fold {
            // Fold the configured PLE scale into the F16 upload.
            let n = raw.len() / 2;
            let mut buf = Vec::with_capacity(raw.len());
            for i in 0..n {
                let bf = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
                let f = f32::from_bits((bf as u32) << 16);
                buf.extend_from_slice(&f16::from_f32(f * embed_scale).to_le_bytes());
            }
            let region = arena.region("e4b_ple_table", buf.len(), 16)?;
            unsafe { region.copy_from_host(&buf)? };
            F16Weight {
                offset_bytes: region.device_ptr(),
                shape: e.shape.clone(),
            }
        } else {
            upload_bf16_raw("e4b_ple_table", &e, raw)?
        };

        let per_layer_model_projection = load_wpacked(
            "e4b_ple_model_proj",
            &format!("{prefix}.per_layer_model_projection"),
            None,
        )?;
        // [L*ple, hidden]
        let exp = [arch.per_layer_embed_total(), arch.hidden_size];
        if per_layer_model_projection.shape != exp {
            return Err(loader_shape_mismatch(
                "per_layer_model_projection",
                exp.to_vec(),
                per_layer_model_projection.shape.to_vec(),
                model_dir,
            ));
        }

        let per_layer_projection_norm = {
            let nname = format!("{prefix}.per_layer_projection_norm.weight");
            let (si, e) = must(&nname)?;
            if e.shape != vec![arch.hidden_size_per_layer_input] {
                return Err(loader_shape_mismatch(
                    &nname,
                    vec![arch.hidden_size_per_layer_input],
                    e.shape.clone(),
                    model_dir,
                ));
            }
            upload_bf16_to_f16("e4b_ple_proj_norm", &e, bytes_of(si, &e))?
        };

        PleTables {
            embed_tokens_per_layer: table,
            embed_scale_folded: fold,
            embed_scale,
            per_layer_model_projection,
            per_layer_projection_norm,
        }
    };

    // ===== pruned lm_head + keepset =====
    let lm_head = {
        let head = load_wpacked("e4b_lm_head", "lm_head", Some(usize::MAX))?;
        // lm_head logical [K, hidden]; channel scale [K,1].
        if head.in_features() != arch.hidden_size {
            return Err(loader_shape_mismatch(
                "lm_head.weight_shape",
                vec![head.out_features(), arch.hidden_size],
                head.shape.to_vec(),
                model_dir,
            ));
        }
        if head.scale_groups != 1 {
            return Err(loader_corrupt(
                model_dir,
                Some("lm_head.weight".to_string()),
                &format!(
                    "lm_head expected channel-strategy scale [K,1], got scale_groups={}",
                    head.scale_groups
                ),
            ));
        }
        let (keep_ids, full_vocab) = load_keepset(model_dir)?;
        if keep_ids.len() != head.out_features() {
            return Err(loader_corrupt(
                model_dir,
                Some("lm_head.weight".to_string()),
                &format!(
                    "keepset size {} != pruned lm_head rows {}",
                    keep_ids.len(),
                    head.out_features()
                ),
            ));
        }
        if full_vocab != arch.vocab_size {
            return Err(loader_corrupt(
                model_dir,
                None,
                &format!(
                    "keepset full_vocab {} != config vocab_size {}",
                    full_vocab, arch.vocab_size
                ),
            ));
        }
        PrunedLmHead {
            pruned_vocab_k: keep_ids.len(),
            full_vocab,
            keep_ids,
            head,
        }
    };

    // ===== RoPE tables (dual) =====
    let sliding_rotary_dim = arch.head_dim_sliding;
    let (cos_s, sin_s) = rope_cos_sin_bytes(
        arch.head_dim_sliding,
        arch.max_position_embeddings,
        arch.rope_theta_sliding,
        sliding_rotary_dim,
    )?;
    let global_rotary_dim = arch.rotary_dim_for_layer(
        arch.layer_types
            .iter()
            .position(|t| *t == crate::gemma4_arch::Gemma4LayerType::GlobalAttention)
            .unwrap_or(0),
    );
    let (cos_g, sin_g) = rope_cos_sin_bytes(
        arch.head_dim_global,
        arch.max_position_embeddings,
        arch.rope_theta_global,
        global_rotary_dim,
    )?;
    let rope_cos_sliding = upload_rope(arena, "e4b_rope_cos_sliding", &cos_s)?;
    let rope_sin_sliding = upload_rope(arena, "e4b_rope_sin_sliding", &sin_s)?;
    let rope_cos_global = upload_rope(arena, "e4b_rope_cos_global", &cos_g)?;
    let rope_sin_global = upload_rope(arena, "e4b_rope_sin_global", &sin_g)?;

    // ===== KV-share source map =====
    let kv_share_src = arch.build_kv_share_src()?;

    // ===== per-layer weights =====
    let mut layers = Vec::with_capacity(arch.num_hidden_layers);
    for l in 0..arch.num_hidden_layers {
        let ln = |s: &str| format!("{prefix}.layers.{l}.{s}");
        let is_full = matches!(
            arch.layer_types[l],
            crate::gemma4_arch::Gemma4LayerType::GlobalAttention
        );
        let owns_kv = arch.layer_owns_kv(l);

        let q_proj = load_wpacked("e4b_q", &ln("self_attn.q_proj"), None)?;
        let o_proj = load_wpacked("e4b_o", &ln("self_attn.o_proj"), None)?;
        let (k_proj, v_proj) = if owns_kv {
            let k_proj = load_wpacked("e4b_k", &ln("self_attn.k_proj"), None)?;
            let v_proj = if arch.layer_uses_k_for_v(l) {
                eprintln!("[loader] attention_k_eq_v: packed layer {l} V -> K");
                k_proj.clone()
            } else {
                load_wpacked("e4b_v", &ln("self_attn.v_proj"), None)?
            };
            (Some(k_proj), Some(v_proj))
        } else {
            // Fail loud if a shared layer unexpectedly ships its own KV.
            if get(&ln("self_attn.k_proj.weight_packed")).is_some() {
                return Err(loader_corrupt(
                    model_dir,
                    Some(ln("self_attn.k_proj.weight_packed")),
                    &format!("layer {l} is KV-shared but ships its own k_proj"),
                ));
            }
            (None, None)
        };

        let gate_proj = load_wpacked("e4b_gate", &ln("mlp.gate_proj"), None)?;
        let up_proj = load_wpacked("e4b_up", &ln("mlp.up_proj"), None)?;
        let down_proj = load_wpacked("e4b_down", &ln("mlp.down_proj"), None)?;

        let per_layer_input_gate = load_wpacked("e4b_ple_gate", &ln("per_layer_input_gate"), None)?;
        let per_layer_projection = load_wpacked("e4b_ple_proj", &ln("per_layer_projection"), None)?;
        // [ple, hidden] and [hidden, ple]
        let gate_exp = [arch.hidden_size_per_layer_input, arch.hidden_size];
        if per_layer_input_gate.shape != gate_exp {
            return Err(loader_shape_mismatch(
                &ln("per_layer_input_gate"),
                gate_exp.to_vec(),
                per_layer_input_gate.shape.to_vec(),
                model_dir,
            ));
        }
        let proj_exp = [arch.hidden_size, arch.hidden_size_per_layer_input];
        if per_layer_projection.shape != proj_exp {
            return Err(loader_shape_mismatch(
                &ln("per_layer_projection"),
                proj_exp.to_vec(),
                per_layer_projection.shape.to_vec(),
                model_dir,
            ));
        }

        let bf = |region: &'static str, suffix: &str, expect: usize| -> Result<F16Weight> {
            let nm = ln(suffix);
            let (si, e) = must(&nm)?;
            if e.shape != vec![expect] {
                return Err(loader_shape_mismatch(
                    &nm,
                    vec![expect],
                    e.shape.clone(),
                    model_dir,
                ));
            }
            // bf16 -> f16: these gammas feed `__half`-reading RMSNorm kernels.
            upload_bf16_to_f16(region, &e, bytes_of(si, &e))
        };
        // Raw bf16 (kept bf16 on device) for the weights whose kernels read
        // bf16: the PLE-gate `post_norm` and the per-layer `layer_scalar`.
        let bfraw = |region: &'static str, suffix: &str, expect: usize| -> Result<F16Weight> {
            let nm = ln(suffix);
            let (si, e) = must(&nm)?;
            if e.shape != vec![expect] {
                return Err(loader_shape_mismatch(
                    &nm,
                    vec![expect],
                    e.shape.clone(),
                    model_dir,
                ));
            }
            upload_bf16_raw(region, &e, bytes_of(si, &e))
        };

        let layer_hd = arch.head_dim_for_layer(l);
        let input_layernorm = bf("e4b_in_ln", "input_layernorm.weight", arch.hidden_size)?;
        let post_attention_layernorm = bf(
            "e4b_post_attn_ln",
            "post_attention_layernorm.weight",
            arch.hidden_size,
        )?;
        let pre_feedforward_layernorm = bf(
            "e4b_pre_ff_ln",
            "pre_feedforward_layernorm.weight",
            arch.hidden_size,
        )?;
        let post_feedforward_layernorm = bf(
            "e4b_post_ff_ln",
            "post_feedforward_layernorm.weight",
            arch.hidden_size,
        )?;
        let post_per_layer_input_norm = bfraw(
            "e4b_post_ple_norm",
            "post_per_layer_input_norm.weight",
            arch.hidden_size,
        )?;
        let q_norm = bf("e4b_q_norm", "self_attn.q_norm.weight", layer_hd)?;
        // KV-shared layers do not own K → no k_norm. Owning layers must have it.
        let k_norm = if owns_kv {
            Some(bf("e4b_k_norm", "self_attn.k_norm.weight", layer_hd)?)
        } else {
            if get(&ln("self_attn.k_norm.weight")).is_some() {
                return Err(loader_corrupt(
                    model_dir,
                    Some(ln("self_attn.k_norm.weight")),
                    &format!("layer {l} is KV-shared but ships its own k_norm"),
                ));
            }
            None
        };
        let layer_scalar = bfraw("e4b_layer_scalar", "layer_scalar", 1)?;

        layers.push(E4bLayerWeights {
            layer_idx: l,
            is_full_attention: is_full,
            kv_shared: !owns_kv,
            kv_share_src: kv_share_src[l],
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
            per_layer_input_gate,
            per_layer_projection,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_per_layer_input_norm,
            q_norm,
            k_norm,
            layer_scalar,
        });
    }

    eprintln!(
        "[loader] packed PLE model loaded: {} layers ({} own KV, {} shared), PLE table {:?} \
         (scale_folded={}), pruned lm_head K={} / full_vocab={}",
        layers.len(),
        arch.kv_shared_start(),
        arch.num_kv_shared_layers,
        ple.embed_tokens_per_layer.shape,
        ple.embed_scale_folded,
        lm_head.pruned_vocab_k,
        lm_head.full_vocab,
    );

    Ok(E4bLoadedModel {
        embedding,
        final_norm,
        lm_head,
        ple,
        rope_cos_sliding,
        rope_sin_sliding,
        rope_cos_global,
        rope_sin_global,
        layers,
        kv_share_src,
    })
}

/// A zeroed FP8 weight handle: a placeholder for the INT4 E4B skeleton's
/// per-layer GEMM slots. The INT4 forward (`gemma4_e4b_int4_layer_forward`)
/// never reads these — it routes the 7 decoder GEMMs through the w4a8 INT4
/// handles built from `E4bLoadedModel`. They exist only so the non-Option
/// `Gemma4LoadedModel`/`Gemma4LayerWeights` structs can be constructed without
/// loading the (absent) bf16/FP8 decoder weights.
fn placeholder_fp8(shape: Vec<usize>) -> Fp8Weight {
    Fp8Weight {
        offset_bytes: 0,
        scale_ptr: 0,
        shape,
        scale: 1.0,
        clamp_ppm: 0.0,
        dtype: DType::Fp8E4M3,
        channelscale_ptr: None,
        blockscale_ptr: None,
        blockscale_n_blocks: 0,
        blockscale_k_blocks: 0,
    }
}

/// Build a `Gemma4LoadedModel` SKELETON from an already-loaded
/// `E4bLoadedModel`, reusing its device pointers for the non-quantized
/// tensors (embedding, final_norm, dual RoPE, per-layer norms, q/k-norm,
/// layer_scalar). The FP8/f16 decoder-GEMM and lm_head slots are zeroed
/// placeholders (`placeholder_fp8`/offset 0): the INT4 E4B forward consumes
/// the w4a8 INT4 handles + the dequantized pruned lm-head instead, so these
/// are never read.
///
/// This is the fix for the "FP8 fused load blocks INT4 ckpt" bug: the INT4
/// pack-quantized checkpoint has no `self_attn.q_proj.weight` (only
/// `q_proj.weight_packed`), so the FP8 `load_gemma4_model` hard-fails with
/// `MissingTensor`. On the `RVLLM_E4B + RVLLM_INT4` path the bring-up calls
/// THIS instead, building `self.model` from the (already validated) E4B
/// handles so all the norm/embedding/RoPE pointers the layer forward reads
/// stay valid, while the GEMMs go through INT4.
///
/// `k_norm` for KV-shared tail layers is `None` in the E4B model (they read
/// the share-source's normed K cache and never project their own K). The
/// INT4 forward drives the K-norm launch with `kv_heads_proj == 0` for those
/// layers, so the gamma pointer is never dereferenced — we fill it with the
/// q_norm pointer as a harmless non-null placeholder.
pub fn build_gemma4_skeleton_from_e4b(
    e4b: &E4bLoadedModel,
    arch: &Gemma4Arch,
) -> Gemma4LoadedModel {
    let clone_f16 = |w: &F16Weight| F16Weight {
        offset_bytes: w.offset_bytes,
        shape: w.shape.clone(),
    };
    let hidden = arch.hidden_size;
    let layers: Vec<Gemma4LayerWeights> = e4b
        .layers
        .iter()
        .map(|l| {
            let hd = arch.head_dim_for_layer(l.layer_idx);
            let nkvh = arch.num_kv_heads_for_layer(l.layer_idx);
            let q_dim = arch.num_attention_heads * hd;
            let kv_dim = nkvh * hd;
            let qkv_rows = q_dim + 2 * kv_dim;
            let inter = arch.intermediate_size;
            Gemma4LayerWeights {
                qkv: placeholder_fp8(vec![qkv_rows, hidden]),
                o_proj: placeholder_fp8(vec![hidden, q_dim]),
                gate_up: placeholder_fp8(vec![2 * inter, hidden]),
                down_proj: placeholder_fp8(vec![hidden, inter]),
                qkv_f16: None,
                o_proj_f16: None,
                gate_up_f16: None,
                down_proj_f16: None,
                input_layernorm: clone_f16(&l.input_layernorm),
                post_attention_layernorm: clone_f16(&l.post_attention_layernorm),
                pre_feedforward_layernorm: clone_f16(&l.pre_feedforward_layernorm),
                post_feedforward_layernorm: clone_f16(&l.post_feedforward_layernorm),
                post_per_layer_input_norm: None,
                q_norm: clone_f16(&l.q_norm),
                // KV-shared layers have no k_norm; never dereferenced (kv_heads_proj==0).
                k_norm: l
                    .k_norm
                    .as_ref()
                    .map(clone_f16)
                    .unwrap_or_else(|| clone_f16(&l.q_norm)),
                layer_scalar: clone_f16(&l.layer_scalar),
                per_layer_input_gate_f16: None,
                per_layer_projection_f16: None,
            }
        })
        .collect();

    Gemma4LoadedModel {
        embedding: clone_f16(&e4b.embedding),
        // lm_head FP8/f16 are placeholders: the INT4 path uses the dequantized
        // pruned head (`gemma4_int4::LmHeadPruned`) + full-vocab scatter.
        lm_head_fp8: placeholder_fp8(vec![e4b.lm_head.pruned_vocab_k, hidden]),
        lm_head_f16: F16Weight {
            offset_bytes: 0,
            shape: vec![e4b.lm_head.pruned_vocab_k, hidden],
        },
        pruned_vocab: None,
        embed_tokens_per_layer: None,
        per_layer_model_projection_f16: None,
        per_layer_projection_norm: None,
        final_norm: clone_f16(&e4b.final_norm),
        rope_cos_sliding: clone_f16(&e4b.rope_cos_sliding),
        rope_sin_sliding: clone_f16(&e4b.rope_sin_sliding),
        rope_cos_global: clone_f16(&e4b.rope_cos_global),
        rope_sin_global: clone_f16(&e4b.rope_sin_global),
        layers,
    }
}

/// Round an f32 to bf16 bits with round-to-nearest-even.
fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits >> 16) & 0xff == 0xff && (bits & 0xffff) != 0 {
        // NaN: keep it quiet.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

/// Load `pruned_vocab.json` as `(keep_ids, full_vocab)`.
fn load_keepset(model_dir: &Path) -> Result<(Vec<u32>, usize)> {
    let p = model_dir.join("pruned_vocab.json");
    let bytes = read_model_aux(model_dir, &p)?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| loader_corrupt(model_dir, None, &format!("pruned_vocab.json: {e}")))?;
    let keep_arr = v["keep_ids"].as_array().ok_or_else(|| {
        loader_corrupt(
            model_dir,
            None,
            "pruned_vocab.json: missing 'keep_ids' array",
        )
    })?;
    let full_vocab = v["full_vocab"]
        .as_u64()
        .or_else(|| v["vocab_size"].as_u64())
        .ok_or_else(|| {
            loader_corrupt(
                model_dir,
                None,
                "pruned_vocab.json: missing 'full_vocab'/'vocab_size'",
            )
        })? as usize;
    let mut keep_ids = Vec::with_capacity(keep_arr.len());
    let mut prev: i64 = -1;
    for (i, e) in keep_arr.iter().enumerate() {
        let id = e.as_u64().ok_or_else(|| {
            loader_corrupt(
                model_dir,
                None,
                &format!("keep_ids[{i}] is not a non-negative integer"),
            )
        })? as usize;
        if id >= full_vocab {
            return Err(loader_corrupt(
                model_dir,
                None,
                &format!("keep_ids[{i}]={id} >= full_vocab {full_vocab}"),
            ));
        }
        // Enforce strictly-ascending so the local→global remap is a clean map.
        if (id as i64) <= prev {
            return Err(loader_corrupt(
                model_dir,
                None,
                &format!("keep_ids not strictly ascending at index {i} (got {id} after {prev})"),
            ));
        }
        prev = id as i64;
        keep_ids.push(id as u32);
    }
    Ok((keep_ids, full_vocab))
}

#[cfg(test)]
mod fp8_tests {
    use super::*;

    fn all_fp8_values() -> Vec<(u8, f32)> {
        (0..=255u8)
            .filter_map(|b| {
                let v = fp8_e4m3_to_f32(b);
                if v.is_nan() {
                    None
                } else {
                    Some((b, v))
                }
            })
            .collect()
    }

    #[test]
    fn roundtrip_all_finite_bytes_with_canonical_zero() {
        let mut fails = Vec::new();
        for b in 0..=255u8 {
            let v = fp8_e4m3_to_f32(b);
            if v.is_nan() {
                continue;
            }
            let re = fp8_e4m3_encode(v);
            let expected = if b == 0x80 { 0x00 } else { b };
            if re != expected {
                fails.push((b, v, re));
            }
        }
        if !fails.is_empty() {
            for (b, v, re) in &fails {
                eprintln!(
                    "ROUNDTRIP FAIL: byte 0x{b:02x}({b}) -> f32={v} -> encode=0x{re:02x}({re})"
                );
            }
            panic!("{} of 255 roundtrips failed", fails.len());
        }
    }

    #[test]
    fn midpoints_bankers_rounding() {
        let vals = all_fp8_values();
        let positives: Vec<(u8, f32)> = vals.iter().filter(|(_, v)| *v > 0.0).copied().collect();
        let mut fails = Vec::new();
        for w in positives.windows(2) {
            let (b_lo, v_lo) = w[0];
            let (b_hi, v_hi) = w[1];
            let mid = (v_lo as f64 + v_hi as f64) / 2.0;
            let mid_f32 = mid as f32;
            if mid_f32 as f64 != mid {
                continue;
            }
            let m_lo = b_lo & 0x07;
            let expected = if m_lo % 2 == 0 { b_lo } else { b_hi };
            let got = fp8_e4m3_encode(mid_f32);
            if got != expected {
                fails.push((mid_f32, b_lo, b_hi, expected, got));
            }
        }
        if !fails.is_empty() {
            for (mid, lo, hi, exp, got) in &fails {
                eprintln!("MIDPOINT FAIL: {mid} between 0x{lo:02x}({lo}) and 0x{hi:02x}({hi}): expected 0x{exp:02x} got 0x{got:02x}");
            }
            panic!("{} midpoint rounding failures", fails.len());
        }
    }

    #[test]
    fn boundary_and_signed_zero_cases() {
        assert_eq!(fp8_e4m3_encode(0.0), 0x00);
        // Canonical `rvllm_core::fp8` contract: signed zero encodes as +0.
        assert_eq!(fp8_e4m3_encode(-0.0), 0x00);
        assert_eq!(fp8_e4m3_encode(448.0), 0x7e);
        assert_eq!(fp8_e4m3_encode(-448.0), 0xfe);
        assert_eq!(fp8_e4m3_encode(f32::INFINITY), 0x7e);
        assert_eq!(fp8_e4m3_encode(f32::NEG_INFINITY), 0xfe);
        assert_eq!(fp8_e4m3_encode(f32::NAN), 0x7f);
    }
}

#[cfg(test)]
mod e4b_tests {
    use super::*;

    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    #[test]
    fn f32_to_bf16_round_trip_and_scale() {
        // Exact bf16-representable values round-trip.
        for v in [0.0f32, 1.0, 2.0, 16.0, -16.0, 0.5, 256.0] {
            let b = f32_to_bf16_bits(v);
            assert_eq!(bf16_to_f32(b), v, "{v} did not round-trip through bf16");
        }
        // Round-to-nearest-even: a value just above a bf16 step rounds up.
        // 1.0 + 2^-8 has the guard bit set; bf16 has 8 mantissa bits.
        let x = 1.0f32 + (1.0 / 256.0);
        let b = f32_to_bf16_bits(x);
        // bf16(1.00390625) rounds to nearest representable; assert monotonic.
        assert!(bf16_to_f32(b) >= 1.0);
    }

    #[test]
    fn decode_weight_shape_ok_and_fail() {
        let md = Path::new("/tmp");
        // A small valid logical shape encoded as little-endian i64.
        let mut raw = Vec::new();
        raw.extend_from_slice(&3i64.to_le_bytes());
        raw.extend_from_slice(&8i64.to_le_bytes());
        assert_eq!(decode_weight_shape(&raw, "w", md).unwrap(), [3, 8]);
        // Wrong length fails.
        assert!(decode_weight_shape(&raw[..8], "w", md).is_err());
        // Non-positive dim fails.
        let mut bad = Vec::new();
        bad.extend_from_slice(&0i64.to_le_bytes());
        bad.extend_from_slice(&8i64.to_le_bytes());
        assert!(decode_weight_shape(&bad, "w", md).is_err());
    }
}
