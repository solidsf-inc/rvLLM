//! Persistent serve-side decode session for Gemma 4.
//!
//! `run_generate` owns per-call scratch and graph state.
//!
//! `Gemma4ServeSession` is the cross-request version of the same machinery:
//!   * All device regions are carved out of the bringup arena ONCE at
//!     engine init and never restored, so their pointers are stable for
//!     the process lifetime — the precondition for keeping captured
//!     graphs alive across requests.
//!   * Captured graphs live in a `GraphPool` keyed on `(bucket,
//!     max_blocks)` with the `MetaLayoutHash` checked before every replay
//!     (bucket 1 = the n=1 decode step, bucket n = the spec verify chunk
//!     of size n). Capture happens once per key per process, on the key's
//!     first use; every later request replays.
//!   * Speculative decode (greedy n-gram prompt-lookup + batched verify,
//!     same loop as `run_generate`) runs behind the API when
//!     `RVLLM_SPEC_DECODE=1`. The serve worker is single-seat by
//!     construction (one engine thread, jobs run sequentially), which is
//!     exactly the regime the spec loop is sound for.
//!
//! The session always runs the FP8 KV cache, the only layout supported by
//! the speculative verification path.
//! `RVLLM_F16_KV=1` is rejected at construction rather than silently
//! ignored. The emitted token stream is bit-identical to
//! `run_generate` under `RVLLM_F16_KV=0`: the kernels and values are the
//! same; only the fixed metadata source regions differ.

use std::time::Instant;

use rvllm_core::{ConfigError, Result, RvllmError};
use rvllm_graph::{CapturedGraph, GraphPool};
use rvllm_kernels::KernelFn;
use rvllm_loader::gemma4_arch::Gemma4LayerType;
use rvllm_mem::PinnedBuf;
use rvllm_metadata::MetadataLayout;

use crate::gemma4_bring_up::{
    validate_sampled_token, Gemma4Bringup, DEFAULT_KV_SCALE, DEFAULT_Q_SCALE,
};
use crate::gemma4_layer_exec::{
    gemma4_forward_phase, Gemma4LayerDims, Gemma4LayerKernels, Gemma4LayerScratch,
    Gemma4MetadataPtrs, Gemma4Phase,
};

pub struct Gemma4ServeOutput {
    pub ids: Vec<u32>,
    /// Time to first token: prompt forward + first lm_head, fenced.
    pub prefill_ms: f64,
    /// Wall time of the decode loop (everything after the first token).
    pub decode_ms: f64,
}

/// Per-layer KV cache geometry, fixed at construction.
struct LayerKv {
    kv_base: u64,
    kv_elems: u64,
    scale_base: u64,
    scale_slots_half: u64,
    blocks: u32,
}

pub struct Gemma4ServeSession {
    // --- decode/spec configuration (env, read once per process) ---
    spec_decode: bool,
    spec_k: u32,
    spec_ngram_max: usize,
    spec_n_max: u32,
    use_decode_graph: bool,
    spec_graph_enabled: bool,
    num_blocks_total: u32,
    block_size: u32,
    max_pos: usize,
    n_layers_active: usize,
    layer_is_sliding: Vec<bool>,
    sliding_window: usize,
    layout: MetadataLayout,

    // --- persistent device regions (raw pointers; the arena bump is
    // never restored below them, so they are stable process-long) ---
    hidden_fp8: u64,
    hidden_scale: u64,
    q_base: u64,
    q_normed: u64,
    k_normed: u64,
    v_normed: u64,
    q_fp8: u64,
    attn_out: u64,
    attn_out_fp8: u64,
    attn_out_scale: u64,
    gate_up_out: u64,
    gate_up_fp8: u64,
    gate_up_scale: u64,
    mlp_out_fp8: u64,
    mlp_out_scale: u64,
    delta_f16: u64,
    gemm_f32_tmp: u64,
    kv_cache: u64,
    kv_total_bytes: usize,
    kv_scale_cache: u64,
    kv_scale_total_bytes: usize,
    q_scale_scratch: u64,
    q_scale_scratch_bytes: usize,
    q_scale_cache_ptr: u64,
    q_scale: u64,
    kv_scale: u64,
    fa3_ws: u64,
    fa3_ws_bytes: usize,
    cutlass_ws: u64,
    cutlass_ws_bytes: usize,
    positions: u64,
    context_lens: u64,
    cu_seqlens_q: u64,
    block_tables: u64,
    residual: u64,
    logits_f32: u64,
    token_ids: u64,
    sampled: u64,
    slot_array: u64,
    ctx_array: u64,
    spec_slot_arr: u64,
    spec_sliding_ctx: u64,
    layer_kv: Vec<LayerKv>,

    // --- cross-request graph state ---
    graphs: GraphPool,
    captures_total: u64,

    // --- host scratch (reused across steps/requests) ---
    slot_host: Vec<i32>,
    ctx_host: Vec<i32>,
    host_tokens: PinnedBuf<i32>,
    max_new_cap: usize,
}

impl Gemma4ServeSession {
    /// Carve all persistent regions out of the bringup arena and freeze
    /// the decode/spec configuration from env. Must run at engine init,
    /// before anything else checkpoint/restores the arena.
    pub fn new(bu: &Gemma4Bringup, max_new_cap: usize) -> Result<Self> {
        if std::env::var("RVLLM_F16_KV").ok().as_deref() == Some("1") {
            return Err(RvllmError::config(
                ConfigError::InvalidField {
                    name: "RVLLM_F16_KV",
                    reason: "serve session always runs the FP8 KV cache (production decode \
                             attention + spec verify layout); unset RVLLM_F16_KV or set it to 0"
                        .into(),
                },
                "RVLLM_F16_KV",
            ));
        }

        let arch = &bu.arch;
        let arena = &bu.arena;
        let hidden = arch.hidden_size as u32;
        let vocab = arch.vocab_size as u32;
        let block_size: u32 = 32;
        let num_blocks_total: u32 = std::env::var("RVLLM_NUM_BLOCKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);

        let spec_decode = std::env::var("RVLLM_SPEC_DECODE").ok().as_deref() == Some("1");
        let spec_k: u32 = std::env::var("RVLLM_SPEC_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4)
            .clamp(0, 15);
        let spec_ngram_max: usize = std::env::var("RVLLM_SPEC_NGRAM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3)
            .clamp(1, 8);
        let spec_n_max: u32 = if spec_decode { spec_k + 1 } else { 1 };
        let st = spec_n_max; // scratch rows: decode is M=1, verify is M<=K+1

        let use_decode_graph = std::env::var("RVLLM_DECODE_GRAPH").ok().as_deref() != Some("0");
        // Spec verify graphs only on the Fa3 backends — the fallback
        // attention arms stage host data mid-forward (not capture-safe).
        let spec_graph_enabled = use_decode_graph
            && std::env::var("RVLLM_SPEC_GRAPH").ok().as_deref() != Some("0")
            && matches!(
                bu.sliding_attention,
                rvllm_attention::AttentionBackend::Fa3(_)
            )
            && matches!(
                bu.global_attention,
                rvllm_attention::AttentionBackend::Fa3(_)
            );

        let max_hd = arch.max_head_dim() as u32;
        let max_nkvh = arch.max_kv_heads() as u32;
        let max_q_dim = (arch.num_attention_heads * arch.max_head_dim()) as u32;
        let max_kv_dim = max_nkvh * max_hd;
        let max_qkv_rows = max_q_dim + 2 * max_kv_dim;
        let inter = arch.intermediate_size as u32;

        let max_layers = std::env::var("RVLLM_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(arch.num_hidden_layers);
        let n_layers_active = bu.model.layers.len().min(max_layers);
        let layer_is_sliding: Vec<bool> = (0..n_layers_active)
            .map(|l| arch.layer_types[l] == Gemma4LayerType::SlidingAttention)
            .collect();
        let sliding_blocks =
            ((arch.sliding_window_size as u32).saturating_add(block_size - 1) / block_size).max(1);

        // FP8 KV: 1 byte/elem; per-(slot, kv_head) f32 K/V scale cache.
        let mut layer_kv: Vec<LayerKv> = Vec::with_capacity(n_layers_active);
        let mut kv_total_bytes: u64 = 0;
        let mut kv_scale_total_bytes: u64 = 0;
        for l in 0..n_layers_active {
            let is_global = arch.layer_types[l] == Gemma4LayerType::GlobalAttention;
            let blocks = if is_global {
                num_blocks_total
            } else {
                sliding_blocks
            };
            let nkvh = arch.num_kv_heads_for_layer(l) as u32;
            let hd = arch.head_dim_for_layer(l) as u32;
            let kv_elems = 2u64 * blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
            let scale_slots = 2u64 * blocks as u64 * block_size as u64 * nkvh as u64;
            layer_kv.push(LayerKv {
                kv_base: kv_total_bytes, // patched to absolute below
                kv_elems,
                scale_base: kv_scale_total_bytes,
                scale_slots_half: scale_slots / 2,
                blocks,
            });
            kv_total_bytes += kv_elems; // 1 byte per FP8 elem
            kv_scale_total_bytes += scale_slots * 4;
        }

        let r = |name: &'static str, bytes: usize, align: usize| -> Result<u64> {
            Ok(arena.region(name, bytes, align)?.device_ptr())
        };

        let hidden_fp8 = r("srv_hidden_fp8", (st * hidden) as usize, 16)?;
        let hidden_scale = r("srv_hidden_scale", (st * 4) as usize, 16)?;
        let q_base = r("srv_qkv", (st * max_qkv_rows * 2) as usize, 16)?;
        let q_normed = r("srv_q_normed", (st * max_q_dim * 2) as usize, 16)?;
        let k_normed = r("srv_k_normed", (st * max_kv_dim * 2) as usize, 16)?;
        let v_normed = r("srv_v_normed", (st * max_kv_dim * 2) as usize, 16)?;
        let q_fp8 = r("srv_q_fp8", (st * max_q_dim) as usize, 16)?;
        let attn_out = r("srv_attn_out", (st * max_q_dim * 2) as usize, 16)?;
        let attn_out_fp8 = r("srv_attn_out_fp8", (st * max_q_dim) as usize, 16)?;
        let attn_out_scale = r("srv_attn_out_scale", (st * 4) as usize, 16)?;
        let gate_up_out = r("srv_gate_up", (st * 2 * inter * 2) as usize, 16)?;
        let gate_up_fp8 = r("srv_gate_up_fp8", (st * 2 * inter) as usize, 16)?;
        let gate_up_scale = r("srv_gate_up_scale", (st * 4) as usize, 16)?;
        let mlp_out_fp8 = r("srv_mlp_fp8", (st * inter) as usize, 16)?;
        let mlp_out_scale = r("srv_mlp_scale", (st * 4) as usize, 16)?;
        let delta_f16 = r("srv_delta", (st * hidden * 2) as usize, 16)?;
        let gemm_f32_max_n = std::cmp::max(max_qkv_rows, 2 * inter);
        let gemm_f32_tmp = r("srv_gemm_f32", (st * gemm_f32_max_n * 4) as usize, 16)?;

        let kv_cache = r("srv_kv", kv_total_bytes as usize, 256)?;
        let kv_scale_cache = r("srv_kv_scale_cache", kv_scale_total_bytes as usize, 16)?;
        for lk in layer_kv.iter_mut() {
            lk.kv_base += kv_cache;
            lk.scale_base += kv_scale_cache;
        }

        let q_scale_scratch_bytes = (st as usize) * arch.num_attention_heads * 4;
        let q_scale_scratch = r("srv_q_scale_scratch", q_scale_scratch_bytes, 16)?;
        // See run_bench: RVLLM_PER_TOKEN_Q_SCALE=0 opts out.
        let q_scale_cache_ptr: u64 =
            if std::env::var("RVLLM_PER_TOKEN_Q_SCALE").ok().as_deref() == Some("0") {
                0
            } else {
                q_scale_scratch
            };

        let q_scale_region = arena.region("srv_q_scale", 4, 4)?;
        let kv_scale_region = arena.region("srv_kv_scale", 4, 4)?;
        {
            let q_s: f32 = std::env::var("RVLLM_Q_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_Q_SCALE);
            let kv_s: f32 = std::env::var("RVLLM_KV_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_KV_SCALE);
            unsafe {
                q_scale_region.copy_from_host(&q_s.to_le_bytes())?;
                kv_scale_region.copy_from_host(&kv_s.to_le_bytes())?;
            }
        }
        let q_scale = q_scale_region.device_ptr();
        let kv_scale = kv_scale_region.device_ptr();

        let fa3_ws_bytes = bu.attention_workspace_bytes(
            n_layers_active,
            st,
            1,
            st,
            block_size,
            num_blocks_total,
            sliding_blocks,
            num_blocks_total,
            arch.sliding_window_size as u32,
            false,
            spec_decode,
        )?;
        let fa3_ws = r("srv_fa3_ws", fa3_ws_bytes, 256)?;
        let cutlass_ws_bytes: usize = 16 * 1024 * 1024;
        let cutlass_ws = r("srv_cutlass_ws", cutlass_ws_bytes, 256)?;

        let positions = r("srv_pos", (st * 4) as usize, 16)?;
        let context_lens = r("srv_ctx", 4, 16)?;
        let cu_seqlens_q = r("srv_cu_seqlens", ((st + 1) * 4) as usize, 16)?;
        let block_tables_region = arena.region("srv_bt", (num_blocks_total * 4) as usize, 16)?;
        {
            let bt: Vec<i32> = (0..num_blocks_total as i32).collect();
            unsafe { block_tables_region.copy_from_host(cast_i32(&bt))? };
        }
        let block_tables = block_tables_region.device_ptr();

        let residual = r("srv_residual", (st * hidden * 2) as usize, 16)?;
        let logits_f32 = r("srv_logits_f32", (st as usize) * (vocab as usize) * 4, 16)?;
        let token_ids = r("srv_tok_ids", (st * 4) as usize, 16)?;
        let sampled = r("srv_sampled", (st as usize) * 4, 16)?;
        let slot_array = r("srv_slot_arr", n_layers_active.max(1) * 4, 16)?;
        let ctx_array = r("srv_ctx_arr", n_layers_active.max(1) * 4, 16)?;
        let spec_slot_arr = r(
            "srv_spec_slot",
            n_layers_active.max(1) * st as usize * 4,
            16,
        )?;
        let spec_sliding_ctx = r("srv_spec_sctx", st as usize * 4, 16)?;

        eprintln!(
            "[serve-session] persistent regions ready: spec_decode={} spec_k={} graphs={} \
             fp8_kv num_blocks={} arena_used_gb={:.1}",
            spec_decode,
            spec_k,
            use_decode_graph,
            num_blocks_total,
            arena.used() as f64 / 1_073_741_824.0,
        );

        Ok(Self {
            spec_decode,
            spec_k,
            spec_ngram_max,
            spec_n_max,
            use_decode_graph,
            spec_graph_enabled,
            num_blocks_total,
            block_size,
            max_pos: (num_blocks_total as usize) * (block_size as usize),
            n_layers_active,
            layer_is_sliding,
            sliding_window: arch.sliding_window_size,
            layout: MetadataLayout::compute(1, num_blocks_total)?,
            hidden_fp8,
            hidden_scale,
            q_base,
            q_normed,
            k_normed,
            v_normed,
            q_fp8,
            attn_out,
            attn_out_fp8,
            attn_out_scale,
            gate_up_out,
            gate_up_fp8,
            gate_up_scale,
            mlp_out_fp8,
            mlp_out_scale,
            delta_f16,
            gemm_f32_tmp,
            kv_cache,
            kv_total_bytes: kv_total_bytes as usize,
            kv_scale_cache,
            kv_scale_total_bytes: kv_scale_total_bytes as usize,
            q_scale_scratch,
            q_scale_scratch_bytes,
            q_scale_cache_ptr,
            q_scale,
            kv_scale,
            fa3_ws,
            fa3_ws_bytes,
            cutlass_ws,
            cutlass_ws_bytes,
            positions,
            context_lens,
            cu_seqlens_q,
            block_tables,
            residual,
            logits_f32,
            token_ids,
            sampled,
            slot_array,
            ctx_array,
            spec_slot_arr,
            spec_sliding_ctx,
            layer_kv,
            graphs: GraphPool::new(),
            captures_total: 0,
            slot_host: vec![0; n_layers_active.max(1)],
            ctx_host: vec![0; n_layers_active.max(1)],
            host_tokens: PinnedBuf::new(max_new_cap.max(1))?,
            max_new_cap: max_new_cap.max(1),
        })
    }

    /// Total graph captures performed by this session (process lifetime).
    /// The journal proof for "capture once per key" is this number going
    /// flat across requests.
    pub fn graph_captures_total(&self) -> u64 {
        self.captures_total
    }

    pub fn spec_enabled(&self) -> bool {
        self.spec_decode
    }

    /// Greedy generation with persistent graphs (+ optional spec decode).
    /// Same token stream as `run_generate` under `RVLLM_F16_KV=0`.
    ///
    /// # Safety
    /// Caller guarantees `bu` is the same bringup this session was
    /// constructed from (region pointers were carved from its arena) and
    /// that no other work runs on the engine stream concurrently.
    pub unsafe fn generate(
        &mut self,
        bu: &Gemma4Bringup,
        fn_embed: &KernelFn,
        fn_argmax: &KernelFn,
        prompt_ids: &[u32],
        max_new: usize,
        eos_ids: &[u32],
        image_embeds: &[(usize, Vec<f32>)],
    ) -> Result<Gemma4ServeOutput> {
        let invalid = |reason: String| {
            RvllmError::config(
                ConfigError::InvalidField {
                    name: "request",
                    reason,
                },
                "request",
            )
        };
        if prompt_ids.is_empty() {
            return Err(invalid("empty prompt".into()));
        }
        if max_new == 0 {
            return Err(invalid("max_new must be >= 1".into()));
        }
        if max_new > self.max_new_cap {
            return Err(invalid(format!(
                "max_new {} exceeds session cap {}",
                max_new, self.max_new_cap
            )));
        }
        if prompt_ids.len() + max_new > self.max_pos {
            return Err(invalid(format!(
                "prompt ({}) + max_new ({}) exceeds KV capacity {} (RVLLM_NUM_BLOCKS={} x {})",
                prompt_ids.len(),
                max_new,
                self.max_pos,
                self.num_blocks_total,
                self.block_size
            )));
        }

        let stream = bu.stream.raw();
        let kernels = bu.layer_kernels();
        let arch = &bu.arch;
        let hidden = arch.hidden_size as u32;
        let vocab = arch.vocab_size as u32;

        // Fresh KV state per request (same memsets run_generate does on
        // its per-call regions).
        crate::bring_up::memset_d8_checked(self.kv_cache, 0, self.kv_total_bytes, stream)?;
        crate::bring_up::memset_d8_checked(
            self.kv_scale_cache,
            0,
            self.kv_scale_total_bytes,
            stream,
        )?;
        crate::bring_up::memset_d8_checked(
            self.q_scale_scratch,
            0,
            self.q_scale_scratch_bytes,
            stream,
        )?;

        let t0 = Instant::now();

        // Phase 1: prompt, per-token (correct-by-design reference path).
        // Metadata comes from fixed device arrays (one ordered HtoD per
        // step) instead of run_one_token's per-layer sync scalar copies —
        // same i32 values, same kernels, bit-identical output.
        for (i, &tok) in prompt_ids.iter().enumerate() {
            crate::bring_up::htod_ordered(self.token_ids, cast_i32(&[tok as i32]), stream)?;
            self.prepare_decode_inputs(i, stream)?;
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1,
                hidden,
                vocab,
            }
            .launch(
                fn_embed,
                self.residual,
                bu.model.embedding.offset_bytes,
                self.token_ids,
                stream,
            )?;
            crate::gemma4_bring_up::inject_image_embeds_f16(
                self.residual,
                hidden,
                i,
                1,
                (hidden as u64) * 2,
                image_embeds,
                stream,
            )?;
            self.forward_layers(bu, &kernels, 1, Gemma4Phase::Decode, MetaSrc::DecodeArrays)?;
        }

        // LM head on the last prompt token -> first output token.
        self.lm_head_tail(bu, &kernels, fn_argmax, 1)?;
        bu.stream.fence()?;
        let mut host_tok = [0i32; 1];
        crate::bring_up::dtoh_sync_checked(self.sampled, host_tok.as_mut_ptr().cast(), 4, stream)?;
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[serve-session] prefill {} tokens in {:.1}ms",
            prompt_ids.len(),
            prefill_ms
        );

        let mut output_ids: Vec<u32> = Vec::with_capacity(max_new);
        output_ids.push(validate_sampled_token(
            i64::from(host_tok[0]),
            vocab,
            "gemma4_serve_prefill",
            stream,
        )?);

        let t_decode = Instant::now();
        if !eos_ids.contains(&output_ids[0]) && max_new > 1 {
            if self.spec_decode {
                self.spec_decode_loop(
                    bu,
                    &kernels,
                    fn_embed,
                    fn_argmax,
                    prompt_ids,
                    max_new,
                    eos_ids,
                    &mut output_ids,
                )?;
            } else {
                self.graph_decode_loop(
                    bu,
                    &kernels,
                    fn_embed,
                    fn_argmax,
                    prompt_ids.len(),
                    max_new,
                    eos_ids,
                    &mut output_ids,
                )?;
            }
        }
        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let decoded_tokens = output_ids.len().saturating_sub(1);
        eprintln!(
            "[serve-session] {} tokens decoded in {:.1}ms ({:.1} tok/s), graph_captures_total={}",
            decoded_tokens,
            decode_ms,
            decoded_tokens as f64 / (decode_ms / 1000.0).max(1e-9),
            self.captures_total
        );

        Ok(Gemma4ServeOutput {
            ids: output_ids,
            prefill_ms,
            decode_ms,
        })
    }

    /// Plain decode: replay the captured n=1 step graph; capture it once
    /// per process on first use (after one eager step that warms the
    /// cuBLASLt algo cache). Token harvest batches async DtoH into pinned
    /// slots and fences every HARVEST_EVERY steps — same scheme and same
    /// bit-exact ordering as `run_generate`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn graph_decode_loop(
        &mut self,
        bu: &Gemma4Bringup,
        kernels: &Gemma4LayerKernels<'_>,
        fn_embed: &KernelFn,
        fn_argmax: &KernelFn,
        prompt_len: usize,
        max_new: usize,
        eos_ids: &[u32],
        output_ids: &mut Vec<u32>,
    ) -> Result<()> {
        let stream = bu.stream.raw();
        let vocab = bu.arch.vocab_size as u32;
        // Seed the device token buffer with the first decoded token; the
        // graph's embed gather reads from here and its closing DtoD keeps
        // it fed thereafter.
        crate::bring_up::htod_ordered(self.token_ids, cast_i32(&[output_ids[0] as i32]), stream)?;

        let mut next_step = 0usize; // decode_step producing output index step+1
        let mut stopped = false;
        if !self.use_decode_graph {
            // Eager escape hatch (RVLLM_DECODE_GRAPH=0): same body, no
            // capture, fence + DtoH every step.
            let mut host_tok = [0i32; 1];
            for decode_step in 0..max_new - 1 {
                self.prepare_decode_inputs(prompt_len + decode_step, stream)?;
                self.decode_step_body(bu, kernels, fn_embed, fn_argmax)?;
                bu.stream.fence()?;
                crate::bring_up::dtoh_sync_checked(
                    self.sampled,
                    host_tok.as_mut_ptr().cast(),
                    4,
                    stream,
                )?;
                let next_id = validate_sampled_token(
                    i64::from(host_tok[0]),
                    vocab,
                    "gemma4_serve_eager_decode",
                    stream,
                )?;
                output_ids.push(next_id);
                if eos_ids.contains(&next_id) {
                    break;
                }
            }
            return Ok(());
        }

        if self.graphs.get(1, self.num_blocks_total).is_none() {
            // First request of the process: step 0 runs eagerly (warms
            // cuBLASLt for the M=1 shapes), then the chain is captured.
            self.prepare_decode_inputs(prompt_len, stream)?;
            self.decode_step_body(bu, kernels, fn_embed, fn_argmax)?;
            bu.stream.fence()?;
            let mut host_tok = [0i32; 1];
            crate::bring_up::dtoh_sync_checked(
                self.sampled,
                host_tok.as_mut_ptr().cast(),
                4,
                stream,
            )?;
            let next_id = validate_sampled_token(
                i64::from(host_tok[0]),
                vocab,
                "gemma4_serve_graph_warmup",
                stream,
            )?;
            output_ids.push(next_id);
            stopped = eos_ids.contains(&next_id);
            next_step = 1;
            if !stopped {
                self.capture_decode_graph(bu, kernels, fn_embed, fn_argmax)?;
            }
        }

        if stopped || output_ids.len() >= max_new {
            return Ok(());
        }
        // Raw pointer up front: the pinned slots are written by async DtoH
        // and read back only after a fence, while `graph` below holds an
        // immutable borrow of self for the whole loop.
        let host_ptr: *mut i32 = self.host_tokens.as_mut_ptr();
        // check_before_replay verifies the MetaLayoutHash captured into
        // the pool entry still matches this session's layout.
        let graph = self
            .graphs
            .check_before_replay(1, self.num_blocks_total, &self.layout)?;

        const HARVEST_EVERY: usize = 16;
        // host slot s holds the token produced by decode_step s (output
        // index s+1). drained tracks the last slot moved into output_ids.
        let mut drained: isize = next_step as isize - 1;
        for decode_step in next_step..max_new - 1 {
            if stopped {
                break;
            }
            self.prepare_decode_inputs_host(prompt_len + decode_step, stream)?;
            graph.replay(stream)?;
            crate::bring_up::dtoh_async_sync(self.sampled, host_ptr.add(decode_step), 4, stream)?;
            let last_step = decode_step == max_new - 2;
            if decode_step as isize - drained >= HARVEST_EVERY as isize || last_step {
                bu.stream.fence()?;
                for s in (drained + 1) as usize..=decode_step {
                    let next_id = validate_sampled_token(
                        i64::from(*host_ptr.add(s)),
                        vocab,
                        "gemma4_serve_graph_harvest",
                        stream,
                    )?;
                    output_ids.push(next_id);
                    if eos_ids.contains(&next_id) {
                        stopped = true;
                        break;
                    }
                }
                drained = decode_step as isize;
            }
        }
        Ok(())
    }

    /// Speculative decode loop — the run_generate spec loop with the
    /// graphs held in the session pool: bucket 1 is the n=1 decode step
    /// (shared with graph_decode_loop), bucket n the verify chunk of
    /// size n. Each key is captured once per process, on first use.
    #[allow(clippy::too_many_arguments)]
    unsafe fn spec_decode_loop(
        &mut self,
        bu: &Gemma4Bringup,
        kernels: &Gemma4LayerKernels<'_>,
        fn_embed: &KernelFn,
        fn_argmax: &KernelFn,
        prompt_ids: &[u32],
        max_new: usize,
        eos_ids: &[u32],
        output_ids: &mut Vec<u32>,
    ) -> Result<()> {
        let stream = bu.stream.raw();
        let vocab = bu.arch.vocab_size as u32;
        let mut all_tokens: Vec<u32> = Vec::with_capacity(prompt_ids.len() + max_new);
        all_tokens.extend_from_slice(prompt_ids);
        all_tokens.push(output_ids[0]);
        let mut cur = prompt_ids.len();
        let mut spec_steps = 0usize;
        let mut spec_drafted = 0usize;
        let mut spec_accept = 0usize;
        let mut host_out = vec![0i32; self.spec_n_max as usize];
        let mut verified_out = vec![0u32; self.spec_n_max as usize];
        let mut stopped = false;
        while !stopped && output_ids.len() < max_new {
            let remaining = max_new - output_ids.len();
            let budget = (self.spec_k as usize)
                .min(remaining.saturating_sub(1))
                .min(self.max_pos.saturating_sub(cur + 1));
            let draft = if budget > 0 {
                crate::gemma4_bring_up::ngram_draft(&all_tokens, budget, self.spec_ngram_max)
            } else {
                Vec::new()
            };
            let last_tok = *output_ids.last().unwrap() as i32;
            let n = 1 + draft.len();
            if n == 1 {
                crate::bring_up::htod_ordered(self.token_ids, cast_i32(&[last_tok]), stream)?;
                self.prepare_decode_inputs(cur, stream)?;
                if self.graphs.get(1, self.num_blocks_total).is_some() {
                    self.graphs
                        .check_before_replay(1, self.num_blocks_total, &self.layout)?
                        .replay(stream)?;
                } else {
                    self.decode_step_body(bu, kernels, fn_embed, fn_argmax)?;
                    if self.spec_graph_enabled {
                        self.capture_decode_graph(bu, kernels, fn_embed, fn_argmax)?;
                    }
                }
            } else {
                let mut chunk: Vec<i32> = Vec::with_capacity(n);
                chunk.push(last_tok);
                chunk.extend(draft.iter().map(|&t| t as i32));
                self.spec_stage_metadata(&chunk, cur, stream)?;
                let nb = n as u32;
                let spec_layout = MetadataLayout::compute(nb, self.num_blocks_total)?;
                if self.graphs.get(nb, self.num_blocks_total).is_some() {
                    self.graphs
                        .check_before_replay(nb, self.num_blocks_total, &spec_layout)?
                        .replay(stream)?;
                } else {
                    // First chunk of this size in the process: run eagerly
                    // (warms cuBLASLt for the M=n shapes), then record the
                    // graph for every later occurrence. chunk_start is only
                    // consumed by the non-Fa3 fallback arm, which the
                    // spec_graph_enabled guard excludes from capture.
                    self.spec_verify_forward(bu, kernels, fn_embed, fn_argmax, nb, cur as u32)?;
                    if self.spec_graph_enabled {
                        let g = CapturedGraph::capture(
                            &bu.ctx,
                            nb,
                            self.num_blocks_total,
                            spec_layout.hash(),
                            stream,
                            || {
                                self.spec_verify_forward(
                                    bu, kernels, fn_embed, fn_argmax, nb, cur as u32,
                                )
                            },
                        )?;
                        self.graphs.insert(g)?;
                        self.captures_total += 1;
                    }
                }
            }
            bu.stream.fence()?;
            crate::bring_up::dtoh_sync_checked(
                self.sampled,
                host_out.as_mut_ptr().cast(),
                n * core::mem::size_of::<i32>(),
                stream,
            )?;
            for (token, &raw_id) in verified_out[..n].iter_mut().zip(&host_out[..n]) {
                *token = validate_sampled_token(
                    i64::from(raw_id),
                    vocab,
                    "gemma4_serve_spec_decode",
                    stream,
                )?;
            }
            let mut a = 0usize;
            while a < draft.len() && draft[a] == verified_out[a] {
                a += 1;
            }
            for &t in &verified_out[..=a] {
                output_ids.push(t);
                all_tokens.push(t);
                if eos_ids.contains(&t) {
                    stopped = true;
                    break;
                }
                if output_ids.len() >= max_new {
                    break;
                }
            }
            cur += 1 + a;
            spec_steps += 1;
            spec_drafted += draft.len();
            spec_accept += a;
        }
        eprintln!(
            "[spec] steps={} drafted={} accepted={} accept_rate={:.3} tok/step={:.3} (k={} ngram<={})",
            spec_steps,
            spec_drafted,
            spec_accept,
            if spec_drafted > 0 {
                spec_accept as f64 / spec_drafted as f64
            } else {
                0.0
            },
            if spec_steps > 0 {
                (output_ids.len() - 1) as f64 / spec_steps as f64
            } else {
                0.0
            },
            self.spec_k,
            self.spec_ngram_max
        );
        Ok(())
    }

    unsafe fn capture_decode_graph(
        &mut self,
        bu: &Gemma4Bringup,
        kernels: &Gemma4LayerKernels<'_>,
        fn_embed: &KernelFn,
        fn_argmax: &KernelFn,
    ) -> Result<()> {
        let stream = bu.stream.raw();
        let g = CapturedGraph::capture(
            &bu.ctx,
            1,
            self.num_blocks_total,
            self.layout.hash(),
            stream,
            || self.decode_step_body(bu, kernels, fn_embed, fn_argmax),
        )?;
        self.graphs.insert(g)?;
        self.captures_total += 1;
        Ok(())
    }

    /// Eager ordered refresh of positions + per-layer slot/ctx device
    /// arrays for a decode step at absolute position `step`. Runs between
    /// graph replays, never inside a capture.
    fn prepare_decode_inputs(&mut self, step: usize, stream: u64) -> Result<()> {
        for l in 0..self.n_layers_active {
            if self.layer_is_sliding[l] {
                self.slot_host[l] = (step % self.sliding_window) as i32;
                self.ctx_host[l] = (step + 1).min(self.sliding_window) as i32;
            } else {
                self.slot_host[l] = step as i32;
                self.ctx_host[l] = step as i32 + 1;
            }
        }
        let pos = [step as i32];
        unsafe {
            crate::bring_up::htod_ordered(self.positions, cast_i32(&pos), stream)?;
            crate::bring_up::htod_ordered(self.slot_array, cast_i32(&self.slot_host), stream)?;
            crate::bring_up::htod_ordered(self.ctx_array, cast_i32(&self.ctx_host), stream)?;
        }
        Ok(())
    }

    /// Borrow-splitting twin of `prepare_decode_inputs` for the harvest
    /// loop, where `self.graphs` is immutably borrowed across iterations.
    fn prepare_decode_inputs_host(&self, step: usize, stream: u64) -> Result<()> {
        let mut slot = vec![0i32; self.n_layers_active];
        let mut ctx = vec![0i32; self.n_layers_active];
        for l in 0..self.n_layers_active {
            if self.layer_is_sliding[l] {
                slot[l] = (step % self.sliding_window) as i32;
                ctx[l] = (step + 1).min(self.sliding_window) as i32;
            } else {
                slot[l] = step as i32;
                ctx[l] = step as i32 + 1;
            }
        }
        let pos = [step as i32];
        unsafe {
            crate::bring_up::htod_ordered(self.positions, cast_i32(&pos), stream)?;
            crate::bring_up::htod_ordered(self.slot_array, cast_i32(&slot), stream)?;
            crate::bring_up::htod_ordered(self.ctx_array, cast_i32(&ctx), stream)?;
        }
        Ok(())
    }

    /// Stage the verify-chunk metadata (token ids, positions, ctx,
    /// cu_seqlens, per-layer slot rows, per-query sliding ctx) with ordered
    /// HtoD into fixed device arrays, so the verify
    /// forward stays graph-capturable.
    unsafe fn spec_stage_metadata(&self, chunk: &[i32], cur: usize, stream: u64) -> Result<()> {
        let n = chunk.len() as u32;
        debug_assert!(n >= 2 && n <= self.spec_n_max);
        crate::bring_up::htod_ordered(self.token_ids, cast_i32(chunk), stream)?;
        let pos: Vec<i32> = (cur as i32..cur as i32 + n as i32).collect();
        crate::bring_up::htod_ordered(self.positions, cast_i32(&pos), stream)?;
        let ctx = [(cur as u32 + n) as i32];
        crate::bring_up::htod_ordered(self.context_lens, cast_i32(&ctx), stream)?;
        let cu_seq = [0i32, n as i32];
        crate::bring_up::htod_ordered(self.cu_seqlens_q, cast_i32(&cu_seq), stream)?;
        let mut slot_rows: Vec<i32> = vec![0; self.n_layers_active * self.spec_n_max as usize];
        for (l, row) in slot_rows.chunks_mut(self.spec_n_max as usize).enumerate() {
            for (i, s) in row.iter_mut().take(n as usize).enumerate() {
                let p = cur + i;
                *s = if self.layer_is_sliding[l] {
                    (p % self.sliding_window) as i32
                } else {
                    p as i32
                };
            }
        }
        crate::bring_up::htod_ordered(self.spec_slot_arr, cast_i32(&slot_rows), stream)?;
        let sctx: Vec<i32> = (0..n as usize)
            .map(|i| (cur + i + 1).min(self.sliding_window) as i32)
            .collect();
        crate::bring_up::htod_ordered(self.spec_sliding_ctx, cast_i32(&sctx), stream)?;
        Ok(())
    }

    /// The capturable n=1 decode step: embed gather (from the device
    /// token buffer) -> all layers -> final norm -> lm_head -> argmax ->
    /// DtoD token feedback. Pure device chain, zero sync host copies.
    unsafe fn decode_step_body(
        &self,
        bu: &Gemma4Bringup,
        kernels: &Gemma4LayerKernels<'_>,
        fn_embed: &KernelFn,
        fn_argmax: &KernelFn,
    ) -> Result<()> {
        let stream = bu.stream.raw();
        let arch = &bu.arch;
        rvllm_fused::EmbeddingGatherLaunch {
            num_tokens: 1,
            hidden: arch.hidden_size as u32,
            vocab: arch.vocab_size as u32,
        }
        .launch(
            fn_embed,
            self.residual,
            bu.model.embedding.offset_bytes,
            self.token_ids,
            stream,
        )?;
        self.forward_layers(bu, kernels, 1, Gemma4Phase::Decode, MetaSrc::DecodeArrays)?;
        self.lm_head_tail(bu, kernels, fn_argmax, 1)?;
        // This step's argmax becomes the next replay's embed input.
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(self.token_ids, self.sampled, 4, stream as _);
        Ok(())
    }

    /// The capturable verify forward at M=n over staged device metadata.
    unsafe fn spec_verify_forward(
        &self,
        bu: &Gemma4Bringup,
        kernels: &Gemma4LayerKernels<'_>,
        fn_embed: &KernelFn,
        fn_argmax: &KernelFn,
        n: u32,
        chunk_start: u32,
    ) -> Result<()> {
        let stream = bu.stream.raw();
        let arch = &bu.arch;
        rvllm_fused::EmbeddingGatherLaunch {
            num_tokens: n,
            hidden: arch.hidden_size as u32,
            vocab: arch.vocab_size as u32,
        }
        .launch(
            fn_embed,
            self.residual,
            bu.model.embedding.offset_bytes,
            self.token_ids,
            stream,
        )?;
        let phase = Gemma4Phase::Prefill {
            cu_seqlens_q: self.cu_seqlens_q,
            max_seqlen_q: n,
            num_seqs: 1,
            chunk_start,
            sliding_ctx_per_qi: self.spec_sliding_ctx,
        };
        self.forward_layers(bu, kernels, n, phase, MetaSrc::SpecRows)?;
        self.lm_head_tail(bu, kernels, fn_argmax, n)?;
        Ok(())
    }

    /// Final norm + lm_head GEMM + argmax over the first `n` residual rows.
    unsafe fn lm_head_tail(
        &self,
        bu: &Gemma4Bringup,
        kernels: &Gemma4LayerKernels<'_>,
        fn_argmax: &KernelFn,
        n: u32,
    ) -> Result<()> {
        let stream = bu.stream.raw();
        let arch = &bu.arch;
        let hidden = arch.hidden_size as u32;
        let vocab = arch.vocab_size as u32;
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: n,
            hidden,
            eps: arch.rms_norm_eps,
        }
        .launch(
            kernels.fused_rmsnorm,
            self.residual,
            bu.model.final_norm.offset_bytes,
            stream,
        )?;
        bu.cublaslt.f16_gemm_f32(
            self.residual,
            bu.model.lm_head_f16.offset_bytes,
            self.logits_f32,
            n as i32,
            vocab as i32,
            hidden as i32,
            stream,
        )?;
        rvllm_fused::ArgmaxLaunch {
            num_tokens: n,
            vocab,
        }
        .launch(fn_argmax, self.logits_f32, self.sampled, stream)?;
        Ok(())
    }

    /// One pass over all active layers. `meta_src` selects where each
    /// layer reads its slot_mapping / context_lens: the per-layer decode
    /// arrays, or the spec verify rows + staged scalar ctx.
    unsafe fn forward_layers(
        &self,
        bu: &Gemma4Bringup,
        kernels: &Gemma4LayerKernels<'_>,
        num_tokens: u32,
        phase: Gemma4Phase,
        meta_src: MetaSrc,
    ) -> Result<()> {
        let stream = bu.stream.raw();
        let arch = &bu.arch;
        let hidden = arch.hidden_size as u32;
        let inter = arch.intermediate_size as u32;
        for (layer_idx, layer) in bu.model.layers.iter().enumerate() {
            if layer_idx >= self.n_layers_active {
                break;
            }
            let lt = arch.layer_types[layer_idx];
            let hd = arch.head_dim_for_layer(layer_idx) as u32;
            let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
            let q_dim = (arch.num_attention_heads as u32) * hd;
            let kv_dim = nkvh * hd;
            let lk = &self.layer_kv[layer_idx];

            let dims = Gemma4LayerDims {
                num_tokens,
                hidden,
                num_heads: arch.num_attention_heads as u32,
                num_kv_heads: nkvh,
                head_dim: hd,
                rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                rope_table_rows: arch.max_position_embeddings as u32,
                intermediate: inter,
                block_size: self.block_size,
                max_blocks_per_seq: lk.blocks,
                num_blocks_total: lk.blocks,
                attn_scale: 1.0,
                rms_eps: arch.rms_norm_eps,
                layer_type: lt,
                sliding_window: arch.sliding_window_size as u32,
                f16_kv: false,
                num_hidden_layers: arch.num_hidden_layers as u32,
                layer_idx: layer_idx as u32,
                ple_dim: arch.hidden_size_per_layer_input as u32,
                // E4B KV-share wiring lives in the bring-up decode/graph
                // path. This compatibility path keeps the 31B default.
                kv_shared: false,
            };
            let w = crate::gemma4_layer_exec::Gemma4LayerWeightPtrs {
                attn_norm_gamma: layer.input_layernorm.offset_bytes,
                post_attn_norm_gamma: layer.post_attention_layernorm.offset_bytes,
                pre_ff_norm_gamma: layer.pre_feedforward_layernorm.offset_bytes,
                post_ff_norm_gamma: layer.post_feedforward_layernorm.offset_bytes,
                q_norm_gamma: layer.q_norm.offset_bytes,
                k_norm_gamma: layer.k_norm.offset_bytes,
                qkv_fp8: layer.qkv.offset_bytes,
                qkv_scale: layer.qkv.scale_ptr,
                o_fp8: layer.o_proj.offset_bytes,
                o_scale: layer.o_proj.scale_ptr,
                gate_up_fp8: layer.gate_up.offset_bytes,
                gate_up_scale: layer.gate_up.scale_ptr,
                down_fp8: layer.down_proj.offset_bytes,
                down_scale: layer.down_proj.scale_ptr,
                layer_scalar_ptr: layer.layer_scalar.offset_bytes,
                qkv_f16: layer.qkv_f16.as_ref().map_or(0, |w| w.offset_bytes),
                o_f16: layer.o_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                gate_up_f16: layer.gate_up_f16.as_ref().map_or(0, |w| w.offset_bytes),
                down_f16: layer.down_proj_f16.as_ref().map_or(0, |w| w.offset_bytes),
                ple_input_gate_f16: layer
                    .per_layer_input_gate_f16
                    .as_ref()
                    .map_or(0, |w| w.offset_bytes),
                ple_projection_f16: layer
                    .per_layer_projection_f16
                    .as_ref()
                    .map_or(0, |w| w.offset_bytes),
                post_ple_norm_gamma: layer
                    .post_per_layer_input_norm
                    .as_ref()
                    .map_or(0, |w| w.offset_bytes),
                qkv_chscale: layer.qkv.channelscale_ptr.unwrap_or(0),
                o_chscale: layer.o_proj.channelscale_ptr.unwrap_or(0),
                gate_up_chscale: layer.gate_up.channelscale_ptr.unwrap_or(0),
                down_chscale: layer.down_proj.channelscale_ptr.unwrap_or(0),
                qkv_blockscale: layer.qkv.blockscale_ptr.unwrap_or(0),
                o_blockscale: layer.o_proj.blockscale_ptr.unwrap_or(0),
                gate_up_blockscale: layer.gate_up.blockscale_ptr.unwrap_or(0),
                down_blockscale: layer.down_proj.blockscale_ptr.unwrap_or(0),
            };
            let k_out = self.q_base + (q_dim as u64) * 2;
            let v_out = k_out + (kv_dim as u64) * 2;
            let (cos, sin) = match lt {
                Gemma4LayerType::SlidingAttention => (
                    bu.model.rope_cos_sliding.offset_bytes,
                    bu.model.rope_sin_sliding.offset_bytes,
                ),
                Gemma4LayerType::GlobalAttention => (
                    bu.model.rope_cos_global.offset_bytes,
                    bu.model.rope_sin_global.offset_bytes,
                ),
            };
            let scratch = Gemma4LayerScratch {
                hidden_fp8: self.hidden_fp8,
                hidden_scale: self.hidden_scale,
                q_out: self.q_base,
                k_out,
                v_out,
                q_normed: self.q_normed,
                k_normed: self.k_normed,
                v_normed: self.v_normed,
                q_fp8: self.q_fp8,
                k_cache: lk.kv_base,
                v_cache: lk.kv_base + lk.kv_elems / 2, // FP8: 1 byte/elem
                q_scale_ptr: self.q_scale,
                kv_scale_ptr: self.kv_scale,
                k_scale_cache: lk.scale_base,
                v_scale_cache: lk.scale_base + lk.scale_slots_half * 4,
                q_scale_cache: self.q_scale_cache_ptr,
                attn_out: self.attn_out,
                attn_out_fp8: self.attn_out_fp8,
                attn_out_scale: self.attn_out_scale,
                delta_f16: self.delta_f16,
                gate_up_out: self.gate_up_out,
                gate_up_fp8: self.gate_up_fp8,
                gate_up_scale: self.gate_up_scale,
                mlp_out_fp8: self.mlp_out_fp8,
                mlp_out_scale: self.mlp_out_scale,
                gemm_f32_tmp: self.gemm_f32_tmp,
                cutlass_workspace: self.cutlass_ws,
                cutlass_workspace_bytes: self.cutlass_ws_bytes,
                fa3_workspace: self.fa3_ws,
                fa3_workspace_bytes: self.fa3_ws_bytes,
                ple_inputs: 0,
                ple_gate: 0,
            };
            let (slot_mapping, context_lens) = match meta_src {
                MetaSrc::DecodeArrays => (
                    self.slot_array + (layer_idx as u64) * 4,
                    self.ctx_array + (layer_idx as u64) * 4,
                ),
                MetaSrc::SpecRows => (
                    self.spec_slot_arr + (layer_idx as u64) * (self.spec_n_max as u64) * 4,
                    self.context_lens,
                ),
            };
            let meta = Gemma4MetadataPtrs {
                positions: self.positions,
                slot_mapping,
                cos,
                sin,
                block_tables: self.block_tables,
                context_lens,
            };
            gemma4_forward_phase(
                dims,
                kernels,
                &w,
                &scratch,
                &meta,
                &bu.cublaslt,
                &bu.cutlass,
                &bu.sliding_attention,
                &bu.global_attention,
                self.residual,
                stream,
                phase,
            )?;
        }
        Ok(())
    }
}

#[derive(Copy, Clone)]
enum MetaSrc {
    /// Per-layer scalar slot/context values from the decode arrays.
    DecodeArrays,
    /// Per-layer slot rows (stride spec_n_max) + staged scalar ctx.
    SpecRows,
}

fn cast_i32(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
