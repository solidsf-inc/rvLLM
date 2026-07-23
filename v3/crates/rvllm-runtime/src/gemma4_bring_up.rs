//! Gemma 4 engine bring-up.
//!
//! Parallel to `bring_up.rs` for Llama/Qwen. Assembles every subsystem
//! needed for Gemma 4 inference: variable-head attention, dual RoPE
//! tables, per-layer KV head variation, extra kernel modules.
//!
//! Usage: when `config.json` declares `"Gemma3ForCausalLM"` or similar,
//! the top-level dispatcher constructs `Gemma4Bringup` instead of
//! the regular `Bringup`.

use std::path::PathBuf;
use std::sync::Arc;

use rvllm_attention::{AttentionBackend, Fa3Kernels};
use rvllm_core::Result;
use rvllm_cutlass::{CublasLt, CutlassBackend, Policy};
use rvllm_kernels::{KernelFn, KernelLoader, LoadedModule};
use rvllm_mem::{context::CudaContextHandle, stream::Stream, HbmArena, PinnedBuf};

use crate::gemma4_layer_exec::Gemma4LayerKernels;

pub use crate::bring_up::HbmArenaCheckpoint;
// Serve plumbs per-request sampling params through this re-export so it
// never needs a direct rvllm-sampling dependency.
pub use rvllm_sampling::{SamplingParams, MIN_TEMPERATURE};
pub const DEFAULT_TOP_K: u32 = SamplingParams::sampled(1.0, 1.0, 0).top_k;

/// Conservative scalar fallback for FP8 attention. Dynamic per-head scaling
/// is the default; operators using the scalar path can supply calibrated
/// `RVLLM_Q_SCALE` and `RVLLM_KV_SCALE` values.
pub(crate) const DEFAULT_Q_SCALE: f32 = 1.0;
pub(crate) const DEFAULT_KV_SCALE: f32 = 1.0;

pub struct Gemma4EnginePaths {
    pub model_dir: PathBuf,
    pub kernels_dir: PathBuf,
    pub cutlass_so: PathBuf,
    pub fa3_so: PathBuf,
    pub policy_json: PathBuf,
    /// w4a8 INT4 GEMM `.so` (`libw4a8_gemm.so`, SM90). `None` => the
    /// bring-up resolves it from `RVLLM_W4A8_SO`. Only consumed on the
    /// `RVLLM_E4B + RVLLM_INT4` path; the 31B FP8 path ignores it.
    pub w4a8_so: Option<PathBuf>,
}

pub struct Gemma4FusedModules {
    pub rmsnorm_mod: LoadedModule,
    pub rmsnorm_inplace_mod: LoadedModule,
    pub rope_mod: LoadedModule,
    pub gelu_mod: LoadedModule,
    pub argmax_mod: LoadedModule,
    pub qk_norm_mod: LoadedModule,
    pub softcap_mod: LoadedModule,
    pub residual_scale_mod: LoadedModule,
    pub residual_scale_bf16s_mod: LoadedModule,
    pub vnorm_mod: LoadedModule,
    pub vector_add_mod: LoadedModule,
    pub bf16_to_f16_sat_mod: LoadedModule,
    pub rmsnorm_inplace_bf16_mod: LoadedModule,
    pub vector_add_bf16_to_f16_mod: LoadedModule,
    pub f32_to_bf16_mod: LoadedModule,
    pub f32_to_f16_sat_mod: LoadedModule,
    pub scale_cols_f32_mod: LoadedModule,
    pub scale_rows_f32_ratio_mod: LoadedModule,
    pub scale_rows_f16_pertoken_mod: LoadedModule,
    pub compute_qkv_scales_mod: LoadedModule,
    pub fused_gelu_mul_f16_mod: LoadedModule,
    pub fused_rope_partial_f16kv_mod: LoadedModule,
    pub fused_norm_add_residual_mod: LoadedModule,
    pub fn_rmsnorm: KernelFn,
    pub fn_rmsnorm_fp8_quant: KernelFn,
    pub fn_quantize: KernelFn,
    pub fn_rope_partial_fp8kv: KernelFn,
    pub fn_gelu_mul: KernelFn,
    pub fn_argmax: KernelFn,
    pub fn_qk_rmsnorm: KernelFn,
    pub fn_softcap: KernelFn,
    pub fn_residual_scale: KernelFn,
    /// Per-layer E4B `layer_scalar` (bf16) applied to the whole residual after
    /// the PLE gate, matching the mlx-lm reference `h = h * layer_scalar`.
    pub fn_residual_scale_bf16s: KernelFn,
    pub fn_vnorm: KernelFn,
    pub fn_vector_add: KernelFn,
    pub fn_bf16_to_f16_sat: KernelFn,
    pub fn_rmsnorm_inplace_bf16: KernelFn,
    pub fn_vector_add_bf16_to_f16: KernelFn,
    pub fn_f32_to_bf16: KernelFn,
    pub fn_f32_to_f16_sat: KernelFn,
    pub fn_scale_cols_f32: KernelFn,
    pub fn_scale_rows_f32_ratio: KernelFn,
    /// Per-token activation dequant applied to each w4a8 INT4 GEMM output
    /// (`scale_rows_f16_pertoken_kernel`). The w4a8 kernel applies only the
    /// weight group scales; the per-token activation scale must be applied
    /// here (its scalar `alpha` cannot carry per-token scales).
    pub fn_scale_rows_f16_pertoken: KernelFn,
    pub fn_compute_qkv_scales: KernelFn,
    pub fn_fused_gelu_mul_f16: KernelFn,
    pub fn_fused_rope_partial_f16kv: KernelFn,
    pub fn_fused_norm_add_residual: KernelFn,
    pub fn_fused_norm_add_residual_f16: KernelFn,
    /// Variant that reads f16 input and skips channelscale; used by the
    /// Sm121 decode fast path after `fp8_gemv_wpr_native_f16in` has
    /// already applied the per-channel scale in the GEMV epilogue.
    pub fn_fused_norm_add_residual_f16in: KernelFn,
    pub fused_norm_add_residual_f16_mod: LoadedModule,
    pub fn_fused_qkv_rmsnorm: KernelFn,
    pub fused_qkv_rmsnorm_mod: LoadedModule,
    pub fn_scale_cols_f16: KernelFn,
    pub scale_cols_f16_mod: LoadedModule,
    pub map_token_id_mod: LoadedModule,
    pub fn_map_token_id: KernelFn,
    pub ple_project_combine_mod: LoadedModule,
    pub fn_ple_project_combine: KernelFn,
    pub ple_gelu_mul_f16_mod: LoadedModule,
    pub fn_ple_gelu_mul_f16: KernelFn,
    pub fp8_channelscale_gemv_ktiled_mod: LoadedModule,
    pub fn_fp8_channelscale_gemv_ktiled: KernelFn,
    pub fp8_channelscale_gemv_splitk_mod: LoadedModule,
    pub fn_fp8_channelscale_gemv_splitk: KernelFn,

    // `fp8_gemv.ptx` — GB10 warp-per-row FP8 GEMV kernels. Loaded at
    // bringup so the Sm121 decode fast path (`launch_fp8_gemv_f16in`
    // in `gemma4_layer_exec.rs`) can call it without a per-step
    // module load. Only the f16-input variant is resolved — the
    // other enum variants in `Fp8GemvVariant` document what ships in
    // the PTX but nothing in the runtime path calls them.
    pub fp8_gemv_mod: LoadedModule,
    /// `None` when the live device is not Blackwell (sm_100+) — the
    /// native-CVT entry is gated on `__CUDA_ARCH__ >= 1000` in
    /// `kernels/fp8_gemv.cu`, so the symbol is absent from
    /// pre-Blackwell PTX. `Fp8GemvVariant::available_for(target)` is
    /// the source of truth for this gate. Used by the Sm121 decode
    /// path to run projection GEMMs (QKV / O / gate_up / down)
    /// directly off f16 activations, skipping the FP8 activation-
    /// quant step that cuBLASLt requires.
    pub fn_fp8_gemv_wpr_native_f16in: Option<KernelFn>,
    /// E4B PLE per-layer-embedding gate kernel
    /// (`gemma4_ple_gate_kernel`, from `kernels/gemma4_ple_gate.cu`).
    /// `None` when the manifest entry is absent (31B path / older
    /// kernel bundles). The E4B forward only injects PLE when this is
    /// `Some`; the loader gates `RVLLM_E4B_REQUIRE` on its presence.
    pub ple_gate_mod: Option<LoadedModule>,
    pub fn_ple_gate: Option<KernelFn>,
    /// Companion `gemma4_ple_projection_combine_kernel` (model-projection
    /// + combine, run once per step at model input).
    pub fn_ple_projection_combine: Option<KernelFn>,
    /// `lmhead_prune_argmax.cu` module — INT4 pack dequant + pruned lm-head
    /// greedy argmax. `None` on 31B / older bundles. The E4B path
    /// resolves `dequant_pack_to_f16_kernel` (used to dequant the PLE model
    /// projection at `populate_e4b`) and the pruned-head GEMV/argmax tail.
    pub lmhead_prune_mod: Option<LoadedModule>,
    pub fn_dequant_pack_to_f16: Option<KernelFn>,
    pub fn_lmhead_int4_gemv: Option<KernelFn>,
    pub fn_lmhead_argmax_remap: Option<KernelFn>,
    /// `lmhead_scatter_full_vocab_kernel` — scatter the pruned kept-row scores
    /// into a full-vocab f32 logits buffer with `-inf` at non-kept columns.
    /// Logprobs/PPL only (greedy decode never needs it). `None` on bundles
    /// without the lmhead_prune module.
    pub fn_lmhead_scatter_fullvocab: Option<KernelFn>,
    /// `embedding_gather_f16_kernel` — used by the E4B PLE combine to gather
    /// the per-layer embed rows. Resolved here so `run_ple_combine` does not
    /// need the kernel threaded in from the serve layer. `None` only if the
    /// bundle lacks the module (E4B `run_ple_combine` then hard-fails).
    pub fn_embed_gather: Option<KernelFn>,
    /// Owning module for `fn_embed_gather`. MUST be held for the lifetime of
    /// the struct: `KernelFn` stores only the raw `CUfunction`, and
    /// `LoadedModule::Drop` calls `cuModuleUnload`. If this field is dropped,
    /// `fn_embed_gather` becomes a dangling handle and the first kernel that
    /// reuses the freed module slot (e.g. the w4a8 `.so` load in
    /// `populate_e4b_int4`) makes the next embed-gather launch fail with
    /// `CUDA_ERROR_INVALID_HANDLE`.
    pub embed_gather_mod: Option<LoadedModule>,
}

/// Per-layer E4B PLE weight pointers (device offsets, bf16/f16) and the
/// KV-share source layer index. Populated by the loader for each
/// decoder layer when the model is E4B. Held in a `Vec` parallel to the
/// `Gemma4LoadedModel.layers` vector.
///
/// `gate_w` / `proj_w` are the DENSE bf16 forms of `per_layer_input_gate`
/// (`[h_ple, hidden]`) and `per_layer_projection` (`[hidden, h_ple]`) —
/// the loader dequantizes the tiny INT4 packed tensors at load. `post_norm`
/// is `post_per_layer_input_norm.weight` (`[hidden]`, bf16/f16).
///
/// `kv_share_src` is `Some(src_layer)` for a KV-shared tail layer (this
/// layer reads `src_layer`'s KV cache and writes none), `None` otherwise.
#[derive(Copy, Clone, Debug)]
pub struct Gemma4E4bLayerPtrs {
    pub gate_w: u64,
    pub proj_w: u64,
    pub post_norm: u64,
    pub kv_share_src: Option<usize>,
}

/// Whole-model E4B PLE runtime state, built once per step at model input.
///
/// `per_layer_inputs` is the device buffer holding the COMBINED per-layer
/// inputs `[num_tokens, num_layers, h_ple]` (output of
/// `gemma4_ple_projection_combine_kernel`): for each layer the kernel
/// reads `base + (token*num_layers + layer)*h_ple`. The loader provides
/// the gathered + scale-folded embed table view and the
/// `per_layer_model_projection` GEMM output that feed the combine.
#[derive(Clone, Debug)]
pub struct Gemma4E4bRuntime {
    pub layers: Vec<Gemma4E4bLayerPtrs>,
    pub h_ple: u32,
    /// Device ptr to the combined `[num_tokens, num_layers, h_ple]` f16
    /// per-layer-input buffer for the current step.
    pub per_layer_inputs: u64,
    /// Runtime sliding-window value applied to sliding layers.
    pub sliding_window: u32,
}

/// KV-share source layer for a Gemma 4 E4B layer.
///
/// Mirrors mlx-lm `Gemma4TextModel.previous_kvs`: the last
/// `num_kv_shared_layers` layers do not own KV; each reads the KV of the
/// LAST non-shared layer that has the SAME `layer_type`. Returns
/// `Some(src)` for a shared layer, `None` for a layer that owns its KV.
///
pub fn gemma4_kv_share_src(
    layer_types: &[rvllm_loader::gemma4_arch::Gemma4LayerType],
    num_kv_shared_layers: usize,
    layer_idx: usize,
) -> Option<usize> {
    let n = layer_types.len();
    if num_kv_shared_layers == 0 || num_kv_shared_layers >= n {
        return None;
    }
    let first_shared = n - num_kv_shared_layers;
    if layer_idx < first_shared {
        return None;
    }
    // last non-shared layer of the same type
    let want = layer_types[layer_idx];
    (0..first_shared).rev().find(|&i| layer_types[i] == want)
}

#[cfg(any(feature = "cuda", test))]
fn fill_ppl_layer_metadata(
    layer_types: &[rvllm_loader::gemma4_arch::Gemma4LayerType],
    kv_share_targets: &[Option<usize>],
    step: usize,
    sliding_window: usize,
    slots: &mut [i32],
    contexts: &mut [i32],
) {
    debug_assert_eq!(slots.len(), contexts.len());
    debug_assert!(slots.len() <= layer_types.len());
    debug_assert!(slots.len() <= kv_share_targets.len());
    debug_assert!(sliding_window > 0);
    for layer_idx in 0..slots.len() {
        let sliding =
            layer_types[layer_idx] == rvllm_loader::gemma4_arch::Gemma4LayerType::SlidingAttention;
        slots[layer_idx] = if kv_share_targets[layer_idx].is_some() {
            -1
        } else if sliding {
            (step % sliding_window) as i32
        } else {
            step as i32
        };
        contexts[layer_idx] = if sliding {
            (step + 1).min(sliding_window) as i32
        } else {
            step as i32 + 1
        };
    }
}

#[cfg(any(feature = "cuda", test))]
fn validate_generation_capacity(
    prompt_len: usize,
    max_new: usize,
    num_blocks_total: usize,
    block_size: usize,
    max_position_embeddings: usize,
) -> std::result::Result<usize, String> {
    if prompt_len == 0 {
        return Err("prompt must contain at least one token".to_string());
    }
    let kv_slots = num_blocks_total
        .checked_mul(block_size)
        .ok_or_else(|| format!("global KV capacity overflow: {num_blocks_total} x {block_size}"))?;
    let requested = prompt_len
        .checked_add(max_new)
        .ok_or_else(|| "prompt_len + max_new overflow".to_string())?;
    let capacity = kv_slots.min(max_position_embeddings);
    if requested > capacity {
        return Err(format!(
            "prompt ({prompt_len}) + max_new ({max_new}) exceeds generation capacity {capacity} (global KV slots {kv_slots}, model context {max_position_embeddings})"
        ));
    }
    Ok(capacity)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg(any(feature = "cuda", test))]
struct GenerationPrefillPlan {
    sequential: bool,
    staged: bool,
    reset_before_staged: bool,
}

#[cfg(any(feature = "cuda", test))]
fn generation_prefill_plan(
    skip_decode: bool,
    use_fast_prefill: bool,
    diag_compare: bool,
    prompt_len: u32,
) -> GenerationPrefillPlan {
    let staged =
        (diag_compare && prompt_len > 1) || skip_decode || (use_fast_prefill && prompt_len > 1);
    let sequential = !skip_decode && (diag_compare || !use_fast_prefill || prompt_len <= 1);
    GenerationPrefillPlan {
        sequential,
        staged,
        reset_before_staged: sequential && staged,
    }
}

#[cfg(any(feature = "cuda", test))]
fn fill_prefill_chunk_metadata(
    layer_types: &[rvllm_loader::gemma4_arch::Gemma4LayerType],
    chunk_start: u32,
    chunk_len: u32,
    sliding_window: u32,
    row_stride: usize,
    rows: &mut [i32],
    sliding_contexts: &mut [i32],
) {
    debug_assert_eq!(rows.len(), layer_types.len() * row_stride);
    debug_assert!(chunk_len as usize <= row_stride);
    debug_assert!(chunk_len as usize <= sliding_contexts.len());
    debug_assert!(sliding_window > 0);
    for (offset, context) in sliding_contexts
        .iter_mut()
        .take(chunk_len as usize)
        .enumerate()
    {
        let position = chunk_start + offset as u32;
        *context = position.saturating_add(1).min(sliding_window) as i32;
    }
    for (layer_type, row) in layer_types.iter().zip(rows.chunks_mut(row_stride)) {
        for (offset, slot) in row.iter_mut().take(chunk_len as usize).enumerate() {
            let position = chunk_start + offset as u32;
            *slot = if *layer_type == rvllm_loader::gemma4_arch::Gemma4LayerType::SlidingAttention {
                (position % sliding_window) as i32
            } else {
                position as i32
            };
        }
    }
}

pub struct Gemma4Bringup {
    pub fused: Gemma4FusedModules,
    pub sliding_attention: AttentionBackend,
    pub global_attention: AttentionBackend,
    pub cutlass: CutlassBackend,
    pub cublaslt: CublasLt,
    pub cublaslt_ws: HbmArenaCheckpoint,
    pub policy: Policy,
    pub arch: rvllm_loader::gemma4_arch::Gemma4Arch,
    pub model: rvllm_loader::gemma4_weights::Gemma4LoadedModel,
    pub kernels: Arc<KernelLoader>,
    pub stream: Stream,
    pub arena: HbmArena,
    pub ctx: Arc<CudaContextHandle>,
    /// E4B PLE + KV-share runtime. `Some` when the loaded model is E4B
    /// (`hidden_size_per_layer_input > 0`) and the loader wired
    /// the per-layer PLE pointers + KV-share map. `None` for the 31B path,
    /// which runs the original `gemma4_forward` unchanged. The `Vec` is
    /// parallel to `model.layers`. Populated post-load by `populate_e4b`
    /// Left `None` for the 31B bench / PPL path so it is
    /// byte-identical.
    pub e4b: Option<Gemma4E4bRuntime>,
    /// The loader's E4B handle: PLE tables, per-layer INT4 gate/proj +
    /// post-norm, pruned lm-head, KV-share map. `Some` only when
    /// `RVLLM_E4B=1` AND `arch.is_e4b()` — loaded alongside the FP8 model
    /// in `load`, then consumed by `populate_e4b` / `run_ple_combine`.
    /// `None` for the 31B path. Held here so the per-step PLE combine and
    /// the pruned lm-head argmax can reach the INT4 device handles.
    pub e4b_model: Option<rvllm_loader::gemma4_weights::E4bLoadedModel>,
    /// Dequantized (f16) `per_layer_model_projection` weight, `[num_layers*
    /// h_ple, hidden]`, produced once at `populate_e4b` from the INT4 pack.
    /// Device ptr; `0` when E4B is off. The per-step projection GEMM reads
    /// it; kept f16 (not w4a8-encoded) because it is tiny and runs once per
    /// step — INT4-encoding a single 9472x2560 weight buys nothing.
    pub e4b_proj_w_f16: u64,
    /// Scratch for the per-step PLE combine: gathered embed `[T, L*h_ple]`,
    /// projection f32 output `[T, L*h_ple]`, projection f16 `[T, L*h_ple]`.
    /// Device ptrs; `0` when E4B is off. Sized for `e4b_max_tokens`.
    pub e4b_gather_buf: u64,
    pub e4b_proj_f32_buf: u64,
    pub e4b_proj_f16_buf: u64,
    /// Max num_tokens the per-step PLE buffers were sized for.
    pub e4b_max_tokens: u32,
    /// Loaded w4a8 INT4 GEMM library. `Some` only
    /// on the E4B+INT4 path (`RVLLM_E4B && RVLLM_INT4 && is_e4b()` and the
    /// `.so` loaded). The 31B FP8 path leaves this `None` and never touches
    /// the INT4 GEMM. Held here so `populate_e4b_int4` can encode the
    /// per-layer weights and `gemma4_e4b_layer_forward` can route the decoder
    /// GEMMs through it.
    pub w4a8: Option<rvllm_cutlass::W4a8Lib>,
    /// Per-layer encoded INT4 decoder GEMMs plus the
    /// pruned lm-head, built by `populate_e4b_int4`. `Some` only when the
    /// w4a8 lib loaded AND every layer encoded. When this is `None` on an
    /// E4B run and `RVLLM_INT4_REQUIRE=1`, the bring-up / forward hard-error
    /// rather than run the zeroed FP8 placeholders the skeleton installed.
    pub e4b_int4: Option<crate::gemma4_int4::Gemma4Int4Runtime>,
}

impl Gemma4Bringup {
    pub fn load(paths: Gemma4EnginePaths, arena_bytes: usize) -> Result<Self> {
        let ctx = Arc::new(CudaContextHandle::init(0)?);
        // Resolve the compile target once per bring-up and thread it
        // through — every call to `ctx.compute_capability()` + the
        // lookup costs nothing individually but spreading it across 5
        // sites means "which CC are we on?" reads inconsistent if a
        // future refactor accidentally shadows `ctx`.
        #[cfg(feature = "cuda")]
        let compile_target: Option<rvllm_core::CompileTarget> = {
            let (major, minor) = ctx.compute_capability();
            rvllm_core::CompileTarget::from_compute_capability(major, minor)
        };
        #[cfg(not(feature = "cuda"))]
        let compile_target: Option<rvllm_core::CompileTarget> = None;

        // Arena backing picked per compute capability — see `Bringup::load`
        // in bring_up.rs for the full rationale (GB10 has no dedicated HBM,
        // cuMemAllocManaged is the right allocator there).
        let arena = {
            #[cfg(feature = "gb10")]
            {
                if matches!(compile_target, Some(rvllm_core::CompileTarget::Sm121)) {
                    rvllm_mem::UnifiedArena::new(&ctx, arena_bytes)?.into_inner()
                } else {
                    HbmArena::new(&ctx, arena_bytes)?
                }
            }
            #[cfg(not(feature = "gb10"))]
            {
                HbmArena::new(&ctx, arena_bytes)?
            }
        };
        let stream = Stream::new(&ctx)?;

        let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&paths.model_dir)?;

        // When RVLLM_E4B=1 and the loaded config is
        // E4B (`hidden_size_per_layer_input > 0`), load the INT4 E4B handle
        // (PLE tables + per-layer INT4 gate/proj + pruned lm-head + KV-share
        // map). Gated so the proven 31B path never loads or touches any of
        // this.
        //   RVLLM_E4B_REQUIRE=1 hard-fails if the flag is set but the config
        //   is not E4B (fail-closed: no silent 31B fallback for an E4B run).
        let e4b_requested = std::env::var("RVLLM_E4B").ok().as_deref() == Some("1");
        let e4b_required = std::env::var("RVLLM_E4B_REQUIRE").ok().as_deref() == Some("1");
        let int4_requested = crate::gemma4_int4::int4_requested();
        let e4b_model = if e4b_requested && arch.is_e4b() {
            Some(rvllm_loader::gemma4_load::load_gemma4_e4b_model(
                &paths.model_dir,
                &arena,
                &arch,
            )?)
        } else {
            if e4b_required && !arch.is_e4b() {
                return Err(rvllm_core::RvllmError::config(
                    rvllm_core::ConfigError::Inconsistent {
                        reasons: vec!["RVLLM_E4B_REQUIRE=1 but config is not E4B \
                             (hidden_size_per_layer_input == 0)"
                            .into()],
                    },
                    "RVLLM_E4B",
                ));
            }
            None
        };

        // Pack-quantized E4B models do not carry dense decoder projection
        // tensors. Build their runtime skeleton from the validated INT4 model;
        // dense models continue through the standard loader.
        let use_int4_skeleton = e4b_requested && int4_requested && arch.is_e4b();
        let model = match (&e4b_model, use_int4_skeleton) {
            (Some(e4b), true) => {
                rvllm_loader::gemma4_load::build_gemma4_skeleton_from_e4b(e4b, &arch)
            }
            _ => rvllm_loader::gemma4_load::load_gemma4_model(&paths.model_dir, &arena, &arch)?,
        };

        // On sm_121 the arena is `cuMemAllocManaged` pages that fault
        // to the GPU on first touch. After the weight upload the
        // populated region (~30 GiB for Gemma 4 31B) hasn't faulted
        // yet — prefetching it here removes the page-fault storm
        // from the first decode iteration, so first-token latency
        // stops carrying 30 GiB of H→D page migration cost. CUDA 13
        // dropped the single-arg `cuMemPrefetchAsync` in favour of
        // `_v2` with a `CUmemLocation`; cudarc 0.19 only wraps the
        // v2 form for cuda-13. Best-effort: a non-zero RC is logged
        // but doesn't fail bring-up.
        #[cfg(all(feature = "gb10", feature = "cuda"))]
        unsafe {
            if matches!(compile_target, Some(rvllm_core::CompileTarget::Sm121)) {
                let prefetch_bytes = arena.used();
                if prefetch_bytes > 0 {
                    let loc = cudarc::driver::sys::CUmemLocation {
                        type_: cudarc::driver::sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                        id: 0,
                    };
                    let rc = cudarc::driver::sys::cuMemPrefetchAsync_v2(
                        arena.base_ptr(),
                        prefetch_bytes,
                        loc,
                        0,
                        stream.raw() as _,
                    );
                    if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        tracing::warn!(
                            "cuMemPrefetchAsync_v2({prefetch_bytes} bytes) rc={rc:?} — first-token latency may spike"
                        );
                    } else {
                        let _ = cudarc::driver::sys::cuStreamSynchronize(stream.raw() as _);
                    }
                }
            }
        }

        // Per-arch kernel subdirectory resolution — see `resolve_kernels_dir`.
        let kernels_dir = crate::bring_up::resolve_kernels_dir(&ctx, &paths.kernels_dir)?;
        let manifest_path = kernels_dir.join("manifest.json");
        let manifest = rvllm_kernels::manifest::KernelManifest::load_and_verify(&manifest_path)?;
        let kernels = Arc::new(KernelLoader::new(manifest, &ctx));

        // Attention backend selection is architecture-exact:
        //   * SM80/SM89 use the rvLLM paged-attention shared object built
        //     for that target (`fa_sm89_*` is the stable ABI name).
        //   * SM90 uses FA3 for sliding attention and the rvLLM fallback
        //     for global attention.
        //   * SM100/SM121 use the PTX FA2/decode-per-query path.
        // Never load Hopper-only `sm_90a` code on an Ampere, Ada, or
        // Blackwell device.
        let (sliding_attention, global_attention) = {
            let force_fa2_ptx = std::env::var_os("RVLLM_FORCE_FA2_PTX").is_some();
            let use_fa2_ptx = force_fa2_ptx
                || matches!(
                    compile_target,
                    Some(rvllm_core::CompileTarget::Sm100 | rvllm_core::CompileTarget::Sm121)
                );
            let fallback_so = std::env::var_os("RVLLM_FA_FALLBACK_SO")
                .map(PathBuf::from)
                .unwrap_or_else(|| paths.fa3_so.with_file_name("libfa_sm89_kernels.so"));
            let load_fallback = |head_dim: u32| -> Result<AttentionBackend> {
                let kernels = Fa3Kernels::load(fallback_so.clone(), head_dim)?;
                if !kernels.is_sm89_backend {
                    return Err(rvllm_core::RvllmError::config(
                        rvllm_core::ConfigError::Inconsistent {
                            reasons: vec![format!(
                                "RVLLM_FA_FALLBACK_SO at {} does not export the fa_sm89_* ABI",
                                kernels.so_path.display()
                            )],
                        },
                        "RVLLM_FA_FALLBACK_SO",
                    ));
                }
                Ok(AttentionBackend::Fa3(kernels))
            };

            if use_fa2_ptx {
                if force_fa2_ptx
                    && !matches!(
                        compile_target,
                        Some(rvllm_core::CompileTarget::Sm100 | rvllm_core::CompileTarget::Sm121)
                    )
                {
                    eprintln!("[loader] RVLLM_FORCE_FA2_PTX=1: using PTX attention backend");
                }
                let sliding = AttentionBackend::Fa2Ptx(rvllm_attention::Fa2PtxKernels::load(
                    &kernels,
                    arch.head_dim_sliding as u32,
                )?);
                let global = AttentionBackend::Fa2Ptx(rvllm_attention::Fa2PtxKernels::load(
                    &kernels,
                    arch.head_dim_global as u32,
                )?);
                (sliding, global)
            } else {
                match compile_target {
                    Some(rvllm_core::CompileTarget::Sm80 | rvllm_core::CompileTarget::Sm89) => {
                        let sliding = load_fallback(arch.head_dim_sliding as u32)?;
                        let global = load_fallback(arch.head_dim_global as u32)?;
                        (sliding, global)
                    }
                    Some(rvllm_core::CompileTarget::Sm90) => {
                        let sliding_kernels =
                            Fa3Kernels::load(paths.fa3_so.clone(), arch.head_dim_sliding as u32)?;
                        if sliding_kernels.is_sm89_backend {
                            return Err(rvllm_core::RvllmError::config(
                                rvllm_core::ConfigError::Inconsistent {
                                    reasons: vec![format!(
                                        "RVLLM_FA3_SO at {} exports fa_sm89_* symbols, not \
                                         fa3_sm90_*; rebuild with kernels/build_fa3.sh.",
                                        sliding_kernels.so_path.display()
                                    )],
                                },
                                "RVLLM_FA3_SO",
                            ));
                        }
                        let sliding = AttentionBackend::Fa3(sliding_kernels);
                        let global = load_fallback(arch.head_dim_global as u32)?;
                        (sliding, global)
                    }
                    _ => {
                        return Err(rvllm_core::RvllmError::config(
                            rvllm_core::ConfigError::InvalidField {
                                name: "compute_capability",
                                reason: "no verified attention backend for this target".into(),
                            },
                            "compute_capability",
                        ));
                    }
                }
            }
        };

        // Only the Hopper SM90 shared library consumes the SM90 variant
        // table. Other targets use cuBLASLt/PTX or the SM121 block-scale ABI.
        let skip_policy = !matches!(compile_target, Some(rvllm_core::CompileTarget::Sm90));
        let (policy, variants): (Policy, Vec<_>) = if skip_policy {
            let empty = Policy {
                revision: String::new(),
                arch: compile_target
                    .map(|t| t.as_sm_str())
                    .unwrap_or("sm_121")
                    .into(),
                variants: Vec::new(),
                entries: Default::default(),
            };
            (empty, (0..16u32).map(rvllm_cutlass::VariantId).collect())
        } else {
            let policy = Policy::load(&paths.policy_json)?;
            let variants: std::collections::BTreeSet<_> =
                policy.entries.values().map(|e| e.variant).collect();
            (policy, variants.into_iter().collect())
        };
        // CUTLASS backend selection — see `bring_up::Bringup::load`
        // for the full rationale (sm_121 has no compatible `.so`).
        let cutlass =
            CutlassBackend::load_for(compile_target, paths.cutlass_so.clone(), &variants)?;

        let cublaslt_ws_bytes: usize = 32 * 1024 * 1024;
        let cublaslt_ws_region = arena.region("cublaslt_ws", cublaslt_ws_bytes, 256)?;
        let cublaslt = CublasLt::new(cublaslt_ws_region.device_ptr(), cublaslt_ws_bytes)?;
        let cublaslt_ws = HbmArenaCheckpoint {
            offset_bytes: 0,
            bytes: cublaslt_ws_bytes,
        };

        // `compile_target` is also what the fused loader uses to gate
        // `Fp8GemvVariant::WprNative` (sm_100+ only). `Some(None)` vs
        // `None` is distinct: it means "probe succeeded but CC isn't
        // in our target matrix", which falls back to `WprLut`.
        let fused = load_gemma4_fused(&kernels, compile_target)?;

        // Load the w4a8 INT4 GEMM `.so` when on the
        // E4B+INT4 path. The path is `paths.w4a8_so` (set by serve) or
        // resolved from `RVLLM_W4A8_SO`. The actual per-layer weight ENCODE
        // (dequant pack -> fp16 -> reorder) and the `Gemma4Int4Runtime` build
        // happen later in `populate_e4b_int4` (it needs the per-step scratch
        // sizing from `populate_e4b`). Here we only hold the loaded lib.
        //
        // Fail-closed (spec §0.5): `RVLLM_INT4_REQUIRE=1` hard-errors if the
        // INT4 path was requested for an E4B run but the `.so` is missing —
        // never silently fall through to the zeroed FP8 placeholders that
        // `build_gemma4_skeleton_from_e4b` installed.
        let int4_required = crate::gemma4_int4::int4_required();
        let w4a8 = if use_int4_skeleton {
            let so = paths
                .w4a8_so
                .clone()
                .or_else(|| std::env::var_os("RVLLM_W4A8_SO").map(PathBuf::from));
            match so {
                Some(p) => Some(rvllm_cutlass::W4a8Lib::load(p)?),
                None => {
                    if int4_required {
                        return Err(rvllm_core::RvllmError::config(
                            rvllm_core::ConfigError::Inconsistent {
                                reasons: vec!["RVLLM_INT4_REQUIRE=1 on an E4B INT4 run but no \
                                     w4a8 .so (set RVLLM_W4A8_SO): the decoder GEMM \
                                     weights are zeroed FP8 placeholders — refusing to \
                                     run silent-zeros."
                                    .into()],
                            },
                            "RVLLM_W4A8_SO",
                        ));
                    }
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            ctx,
            arena,
            stream,
            arch,
            model,
            kernels,
            cutlass,
            cublaslt,
            cublaslt_ws,
            sliding_attention,
            global_attention,
            policy,
            fused,
            w4a8,
            e4b_int4: None,
            // E4B PLE/KV-share runtime: `e4b_model` carries the loader
            // handles (or `None` for the 31B path). `e4b` (the per-step
            // runtime) + the PLE-combine scratch are populated by
            // `populate_e4b`, called once after `load` when the engine takes
            // the E4B path; the 31B bench/PPL path leaves all of these unset.
            e4b: None,
            e4b_model,
            e4b_proj_w_f16: 0,
            e4b_gather_buf: 0,
            e4b_proj_f32_buf: 0,
            e4b_proj_f16_buf: 0,
            e4b_max_tokens: 0,
        })
    }

    /// Build the per-step E4B PLE/KV-share runtime from the loader's
    /// `e4b_model` handle. Allocates the combined `per_layer_inputs`
    /// `[max_tokens, num_layers, h_ple]` buffer + the per-step gather/project
    /// scratch, dequantizes the INT4 `per_layer_model_projection` to f16 once,
    /// and fills `self.e4b` with the per-layer gate/proj/post-norm pointers +
    /// KV-share map + the runtime sliding window (256 via RVLLM_SLIDING_WINDOW).
    ///
    /// No-op (returns `Ok` leaving `self.e4b == None`) when `e4b_model` is
    /// `None` (the 31B path). Idempotent for a given `max_tokens`.
    ///
    /// `RVLLM_E4B_REQUIRE=1` makes a missing PLE-gate kernel a hard error
    /// (the per-layer injection can only engage when `fused.fn_ple_gate` and
    /// `fused.fn_ple_projection_combine` are both present in the kernel
    /// bundle) — fail-closed, never a silent skip of the PLE contribution.
    pub fn populate_e4b(&mut self, max_tokens: u32) -> Result<()> {
        let Some(e4b) = self.e4b_model.as_ref() else {
            return Ok(());
        };
        let arch = &self.arch;
        let num_layers = arch.num_hidden_layers as u32;
        let h_ple = arch.hidden_size_per_layer_input as u32;
        if h_ple == 0 || num_layers == 0 {
            return Err(rvllm_core::RvllmError::config(
                rvllm_core::ConfigError::Inconsistent {
                    reasons: vec![format!(
                        "populate_e4b: degenerate dims num_layers={num_layers} h_ple={h_ple}"
                    )],
                },
                "RVLLM_E4B",
            ));
        }

        // Fail-closed: the per-layer PLE gate + the once-per-step
        // projection-combine kernels must both be present, else the E4B
        // forward would silently run without the PLE contribution.
        let require = std::env::var("RVLLM_E4B_REQUIRE").ok().as_deref() == Some("1");
        let ple_kernels_present =
            self.fused.fn_ple_gate.is_some() && self.fused.fn_ple_projection_combine.is_some();
        if require && !ple_kernels_present {
            return Err(rvllm_core::RvllmError::config(
                rvllm_core::ConfigError::Inconsistent {
                    reasons: vec!["RVLLM_E4B_REQUIRE=1 but the PLE gate / projection-combine \
                         kernels are absent from the kernel bundle — the per-layer \
                         embedding contribution cannot engage"
                        .into()],
                },
                "RVLLM_E4B_REQUIRE",
            ));
        }

        let configured_window = u32::try_from(arch.sliding_window_size).map_err(|_| {
            rvllm_core::RvllmError::config(
                rvllm_core::ConfigError::InvalidField {
                    name: "sliding_window_size",
                    reason: "value exceeds u32::MAX".into(),
                },
                "sliding_window_size",
            )
        })?;
        if configured_window == 0 {
            return Err(rvllm_core::RvllmError::config(
                rvllm_core::ConfigError::InvalidField {
                    name: "sliding_window_size",
                    reason: "must be positive".into(),
                },
                "sliding_window_size",
            ));
        }
        let sliding_window = match std::env::var("RVLLM_SLIDING_WINDOW") {
            Ok(raw) => raw
                .parse::<u32>()
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    rvllm_core::RvllmError::config(
                        rvllm_core::ConfigError::InvalidField {
                            name: "RVLLM_SLIDING_WINDOW",
                            reason: format!("expected a positive u32, got {raw:?}"),
                        },
                        "RVLLM_SLIDING_WINDOW",
                    )
                })?,
            Err(std::env::VarError::NotPresent) => configured_window,
            Err(error) => {
                return Err(rvllm_core::RvllmError::config(
                    rvllm_core::ConfigError::InvalidField {
                        name: "RVLLM_SLIDING_WINDOW",
                        reason: error.to_string(),
                    },
                    "RVLLM_SLIDING_WINDOW",
                ));
            }
        };

        if e4b.layers.is_empty() {
            return Err(rvllm_core::RvllmError::config(
                rvllm_core::ConfigError::InvalidField {
                    name: "num_hidden_layers",
                    reason: "E4B model contains no decoder layers".into(),
                },
                "num_hidden_layers",
            ));
        }

        // Per-layer PLE gate/proj weights. The loader keeps
        // `per_layer_input_gate` / `per_layer_projection` INT4 pack-quantized,
        // but the PLE-gate kernel consumes DENSE f16. Allocate two arena
        // regions (one per-layer slice each) and dequant every pack into its
        // slice in the CUDA block below; `gate_w`/`proj_w` then point at the
        // f16 slices.
        let gate0 = &e4b.layers[0].per_layer_input_gate;
        let proj0 = &e4b.layers[0].per_layer_projection;
        let gate_elems = gate0.out_features() * gate0.in_features();
        let proj_elems = proj0.out_features() * proj0.in_features();
        let gate_w_all = self.arena.region(
            "e4b_ple_gate_w_f16",
            num_layers as usize * gate_elems * 2,
            256,
        )?;
        let proj_w_all = self.arena.region(
            "e4b_ple_proj_w_f16",
            num_layers as usize * proj_elems * 2,
            256,
        )?;
        let gate_w_base = gate_w_all.device_ptr();
        let proj_w_base = proj_w_all.device_ptr();
        let gate_slice_bytes = (gate_elems * 2) as u64;
        let proj_slice_bytes = (proj_elems * 2) as u64;
        let layers: Vec<Gemma4E4bLayerPtrs> = e4b
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| Gemma4E4bLayerPtrs {
                gate_w: gate_w_base + i as u64 * gate_slice_bytes,
                proj_w: proj_w_base + i as u64 * proj_slice_bytes,
                post_norm: l.post_per_layer_input_norm.offset_bytes,
                kv_share_src: l.kv_share_src,
            })
            .collect();
        if layers.len() != num_layers as usize {
            return Err(rvllm_core::RvllmError::config(
                rvllm_core::ConfigError::Inconsistent {
                    reasons: vec![format!(
                        "populate_e4b: loader produced {} E4B layers but arch has {} \
                         hidden layers",
                        layers.len(),
                        num_layers
                    )],
                },
                "RVLLM_E4B",
            ));
        }

        // Buffers (f16, 2 bytes/elem). `per_layer_inputs` is the combined
        // [T, L, h_ple] buffer the per-layer gate reads each layer.
        let row_elems = (num_layers * h_ple) as usize;
        let pli_bytes = max_tokens as usize * row_elems * 2;
        let per_layer_inputs = self.arena.region("e4b_per_layer_inputs", pli_bytes, 256)?;
        let gather = self.arena.region("e4b_ple_gather", pli_bytes, 256)?;
        let proj_f32 =
            self.arena
                .region("e4b_ple_proj_f32", max_tokens as usize * row_elems * 4, 256)?;
        let proj_f16 = self.arena.region("e4b_ple_proj_f16", pli_bytes, 256)?;
        // Dequant the INT4 per_layer_model_projection -> f16 [L*h_ple, hidden]
        // once. Device-side; only on the CUDA build (the kernel launch needs a
        // live context). On non-cuda the offset stays 0 (forward is cuda-only).
        let proj_w = &e4b.ple.per_layer_model_projection;
        let proj_w_out = proj_w.out_features();
        let proj_w_in = proj_w.in_features();
        let proj_w_f16 = self
            .arena
            .region("e4b_proj_w_f16", proj_w_out * proj_w_in * 2, 256)?;
        #[cfg(feature = "cuda")]
        unsafe {
            use crate::gemma4_int4::WPackedExt;
            proj_w.validate()?;
            let fn_dequant = self.fused.fn_dequant_pack_to_f16.as_ref().ok_or_else(|| {
                rvllm_core::RvllmError::config(
                    rvllm_core::ConfigError::Inconsistent {
                        reasons: vec!["populate_e4b: dequant_pack_to_f16 kernel absent \
                             (lmhead_prune_argmax bundle missing) — cannot dequant \
                             the PLE model projection"
                            .into()],
                    },
                    "RVLLM_E4B",
                )
            })?;
            crate::gemma4_int4::dequant_pack_to_f16(
                fn_dequant,
                proj_w.packed,
                proj_w.scale,
                proj_w_f16.device_ptr(),
                proj_w.out_features_i32()?,
                proj_w.in_features_i32()?,
                proj_w.group_size_i32()?,
                self.stream.raw(),
            )?;
            // Dequant each per-layer PLE gate + projection INT4 pack -> its f16
            // slice. The PLE gate kernel reads these as dense f16 (gate_w
            // [h_ple, hidden], proj_w [hidden, h_ple]).
            for (i, l) in e4b.layers.iter().enumerate() {
                let g = &l.per_layer_input_gate;
                g.validate()?;
                crate::gemma4_int4::dequant_pack_to_f16(
                    fn_dequant,
                    g.packed,
                    g.scale,
                    gate_w_base + i as u64 * gate_slice_bytes,
                    g.out_features_i32()?,
                    g.in_features_i32()?,
                    g.group_size_i32()?,
                    self.stream.raw(),
                )?;
                let p = &l.per_layer_projection;
                p.validate()?;
                crate::gemma4_int4::dequant_pack_to_f16(
                    fn_dequant,
                    p.packed,
                    p.scale,
                    proj_w_base + i as u64 * proj_slice_bytes,
                    p.out_features_i32()?,
                    p.in_features_i32()?,
                    p.group_size_i32()?,
                    self.stream.raw(),
                )?;
            }
            self.stream.fence()?;
        }

        self.e4b_proj_w_f16 = proj_w_f16.device_ptr();
        self.e4b_gather_buf = gather.device_ptr();
        self.e4b_proj_f32_buf = proj_f32.device_ptr();
        self.e4b_proj_f16_buf = proj_f16.device_ptr();
        self.e4b_max_tokens = max_tokens;
        self.e4b = Some(Gemma4E4bRuntime {
            layers,
            h_ple,
            per_layer_inputs: per_layer_inputs.device_ptr(),
            sliding_window,
        });
        Ok(())
    }

    /// Encode every E4B decoder layer's four logical GEMMs
    /// (QKV, O, gate_up, down) into the w4a8 kernel layout and build the
    /// pruned-lm-head handle, storing the result in `self.e4b_int4`. Must run
    /// AFTER `load` (needs `self.w4a8` + `self.e4b_model`) — call it from the
    /// serve worker / bench right after `populate_e4b`.
    ///
    /// The compressed-tensors pack stores q/k/v (and gate/up) as SEPARATE
    /// Linears, but the FP8 skeleton + the layer body operate on the FUSED
    /// matrices (`qkv` = q||k||v stacked along N, `gate_up` = gate||up). So we
    /// dequant each part into the right N-offset of an f16 scratch, then hand
    /// the fused `[N,K]` matrix to `W4a8Lib::encode_fp16`. A KV-shared tail
    /// layer owns no k/v Linear, so its "qkv" is Q-only (`N == q_dim`).
    ///
    /// No-op (leaves `self.e4b_int4 == None`) when `self.w4a8` is `None` (not
    /// the INT4 path). Fail-closed: when `RVLLM_INT4_REQUIRE=1` and the lib is
    /// absent OR the dequant kernel is missing, hard-error rather than leaving
    /// the zeroed FP8 placeholders in `self.model` to silently produce garbage.
    ///
    /// CUDA-only body (every encode launches a kernel + the w4a8 encoder); on
    /// non-cuda builds this validates the layouts and returns without building
    /// device handles (the forward is cuda-only).
    pub fn populate_e4b_int4(&mut self) -> Result<()> {
        let int4_required = crate::gemma4_int4::int4_required();
        let Some(e4b) = self.e4b_model.as_ref() else {
            // Not an E4B model. INT4 cannot apply; fail-closed if demanded.
            crate::gemma4_int4::enforce_required(false, int4_required, "INT4 (no E4B model)")?;
            return Ok(());
        };
        let Some(_w4a8) = self.w4a8.as_ref() else {
            // E4B model but no w4a8 lib. Fail-closed under REQUIRE — never run
            // the zeroed FP8 decoder placeholders.
            crate::gemma4_int4::enforce_required(false, int4_required, "INT4 w4a8 .so")?;
            return Ok(());
        };

        use crate::gemma4_int4::{Int4DecoderLayer, WPackedExt};
        let arch = &self.arch;
        let hidden = arch.hidden_size;

        // Validate every layer's pack layout up front (fail loud, mirror the
        // Python stack's fail-closed asserts). The lm-head too.
        for l in &e4b.layers {
            l.q_proj.validate()?;
            if let Some(k) = &l.k_proj {
                k.validate()?;
            }
            if let Some(v) = &l.v_proj {
                v.validate()?;
            }
            l.o_proj.validate()?;
            l.gate_proj.validate()?;
            l.up_proj.validate()?;
            l.down_proj.validate()?;
        }
        e4b.lm_head.head.validate()?;

        // Per-layer fused N dims. (k/v absent for KV-shared tail layers.)
        let fused_qkv_n = |l: &rvllm_loader::gemma4_weights::E4bLayerWeights| -> usize {
            let mut n = l.q_proj.out_features();
            if let Some(k) = &l.k_proj {
                n += k.out_features();
            }
            if let Some(v) = &l.v_proj {
                n += v.out_features();
            }
            n
        };
        let fused_gate_up_n = |l: &rvllm_loader::gemma4_weights::E4bLayerWeights| -> usize {
            l.gate_proj.out_features() + l.up_proj.out_features()
        };

        // Worst-case fused weight rows across all layers => scratch + dst sizing.
        let max_qkv_n = e4b.layers.iter().map(fused_qkv_n).max().unwrap_or(0);
        let max_gate_up_n = e4b.layers.iter().map(fused_gate_up_n).max().unwrap_or(0);
        let max_o_n = e4b
            .layers
            .iter()
            .map(|l| l.o_proj.out_features())
            .max()
            .unwrap_or(0);
        let max_down_n = e4b
            .layers
            .iter()
            .map(|l| l.down_proj.out_features())
            .max()
            .unwrap_or(0);
        // O reads q_dim cols; down reads intermediate cols. K dims:
        let max_o_k = e4b
            .layers
            .iter()
            .map(|l| l.o_proj.in_features())
            .max()
            .unwrap_or(0);
        let max_down_k = e4b
            .layers
            .iter()
            .map(|l| l.down_proj.in_features())
            .max()
            .unwrap_or(0);
        let max_n = max_qkv_n.max(max_gate_up_n).max(max_o_n).max(max_down_n);
        let max_k = hidden.max(max_o_k).max(max_down_k);
        let group_size = 128usize; // decoder Linears (lm-head is channel-strategy)

        // Scratch + destination regions, sized for the worst case and reused
        // across layers (encode is sequential at load, fenced per layer).
        //   f16 dequant scratch:  N*K*2 bytes
        //   dst INT4:             N*(K/2) bytes
        //   dst packed FP8 LUT:   N*(K/group)*8 bytes
        //   scales f32 workspace: N*(K/group)*4 bytes
        // We allocate FOUR sets per layer (qkv/o/gate_up/down keep distinct
        // device homes since they coexist in the layer forward), but reuse one
        // f16 scratch + one scales-f32 workspace across all encodes.
        let f16_scratch = self
            .arena
            .region("e4b_int4_f16_scratch", max_n * max_k * 2, 256)?;
        let scales_f32_ws = self.arena.region(
            "e4b_int4_scales_f32_ws",
            max_n * (max_k / group_size).max(1) * 4,
            256,
        )?;

        // GEMM workspace, sized for the largest decoder GEMM shape M=max-batch.
        // The exact size is a kernel query (`workspace_size`) on CUDA; on
        // non-cuda it returns 0. Use a generous static cap (4 MiB) so the
        // region is present for the runtime GEMM regardless.
        let workspace_bytes: usize = 4 * 1024 * 1024;
        let workspace = self
            .arena
            .region("e4b_int4_gemm_ws", workspace_bytes, 256)?;

        // Encode each layer. Each part gets its own INT4 + packed-scale region.
        let mut layers: Vec<Int4DecoderLayer> = Vec::with_capacity(e4b.layers.len());
        #[cfg(feature = "cuda")]
        let fn_dequant = self.fused.fn_dequant_pack_to_f16.as_ref().ok_or_else(|| {
            rvllm_core::RvllmError::config(
                rvllm_core::ConfigError::Inconsistent {
                    reasons: vec!["populate_e4b_int4: dequant_pack_to_f16 kernel absent — \
                         cannot encode the INT4 decoder weights"
                        .into()],
                },
                "RVLLM_INT4",
            )
        })?;

        for l in &e4b.layers {
            let qkv_n = fused_qkv_n(l);
            let gate_up_n = fused_gate_up_n(l);
            let o_n = l.o_proj.out_features();
            let o_k = l.o_proj.in_features();
            let down_n = l.down_proj.out_features();
            let down_k = l.down_proj.in_features();

            // Per-part destinations. The arena bump-allocates distinct offsets
            // per call; the static name is a debug label only (not a key), so
            // reusing one label per category across layers is fine.
            let int4_bytes = |n: usize, k: usize| n * (k / 2).max(1);
            let scale_bytes = |n: usize, k: usize| n * (k / group_size).max(1) * 8;

            let qkv_w = self
                .arena
                .region("e4b_int4_qkv_w", int4_bytes(qkv_n, hidden), 256)?;
            let qkv_s = self
                .arena
                .region("e4b_int4_qkv_s", scale_bytes(qkv_n, hidden), 256)?;
            let o_w = self
                .arena
                .region("e4b_int4_o_w", int4_bytes(o_n, o_k), 256)?;
            let o_s = self
                .arena
                .region("e4b_int4_o_s", scale_bytes(o_n, o_k), 256)?;
            let gu_w = self
                .arena
                .region("e4b_int4_gu_w", int4_bytes(gate_up_n, hidden), 256)?;
            let gu_s = self
                .arena
                .region("e4b_int4_gu_s", scale_bytes(gate_up_n, hidden), 256)?;
            let dn_w = self
                .arena
                .region("e4b_int4_dn_w", int4_bytes(down_n, down_k), 256)?;
            let dn_s = self
                .arena
                .region("e4b_int4_dn_s", scale_bytes(down_n, down_k), 256)?;

            // The CUDA path: dequant each part into the f16 scratch (fused QKV
            // and gate_up stack along N), then encode the fused matrix.
            #[cfg(feature = "cuda")]
            let (qkv, o, gate_up, down) = unsafe {
                // --- QKV: dequant q,(k),(v) stacked, then encode fused. ---
                let mut n_off = 0usize;
                let mut dequant_into =
                    |wp: &rvllm_loader::gemma4_weights::WPacked, n_offset: usize| -> Result<()> {
                        crate::gemma4_int4::dequant_pack_to_f16(
                            fn_dequant,
                            wp.packed,
                            wp.scale,
                            f16_scratch.device_ptr() + (n_offset * hidden * 2) as u64,
                            wp.out_features_i32()?,
                            wp.in_features_i32()?,
                            wp.group_size_i32()?,
                            self.stream.raw(),
                        )
                    };
                dequant_into(&l.q_proj, n_off)?;
                n_off += l.q_proj.out_features();
                if let Some(k) = &l.k_proj {
                    dequant_into(k, n_off)?;
                    n_off += k.out_features();
                }
                if let Some(v) = &l.v_proj {
                    dequant_into(v, n_off)?;
                }
                self.stream.fence()?;
                let qkv = _w4a8.encode_fp16(
                    f16_scratch.device_ptr(),
                    qkv_n as i32,
                    hidden as i32,
                    group_size as i32,
                    qkv_w.device_ptr(),
                    qkv_s.device_ptr(),
                    scales_f32_ws.device_ptr(),
                    true,
                    self.stream.raw(),
                )?;
                self.stream.fence()?;
                let _ = qkv;
                let qkv = crate::gemma4_int4::Int4LinearW4a8 {
                    b_int4_reordered: qkv_w.device_ptr(),
                    b_scales_packed: qkv_s.device_ptr(),
                    n: qkv_n as i32,
                    k: hidden as i32,
                    group_size: group_size as i32,
                };

                // --- O proj (single Linear). ---
                let o = crate::gemma4_int4::encode_from_pack(
                    _w4a8,
                    fn_dequant,
                    &l.o_proj,
                    f16_scratch.device_ptr(),
                    o_w.device_ptr(),
                    o_s.device_ptr(),
                    scales_f32_ws.device_ptr(),
                    self.stream.raw(),
                )?;
                self.stream.fence()?;

                // --- gate_up: dequant gate,up stacked, encode fused. ---
                crate::gemma4_int4::dequant_pack_to_f16(
                    fn_dequant,
                    l.gate_proj.packed,
                    l.gate_proj.scale,
                    f16_scratch.device_ptr(),
                    l.gate_proj.out_features_i32()?,
                    l.gate_proj.in_features_i32()?,
                    l.gate_proj.group_size_i32()?,
                    self.stream.raw(),
                )?;
                crate::gemma4_int4::dequant_pack_to_f16(
                    fn_dequant,
                    l.up_proj.packed,
                    l.up_proj.scale,
                    f16_scratch.device_ptr() + (l.gate_proj.out_features() * hidden * 2) as u64,
                    l.up_proj.out_features_i32()?,
                    l.up_proj.in_features_i32()?,
                    l.up_proj.group_size_i32()?,
                    self.stream.raw(),
                )?;
                self.stream.fence()?;
                _w4a8.encode_fp16(
                    f16_scratch.device_ptr(),
                    gate_up_n as i32,
                    hidden as i32,
                    group_size as i32,
                    gu_w.device_ptr(),
                    gu_s.device_ptr(),
                    scales_f32_ws.device_ptr(),
                    true,
                    self.stream.raw(),
                )?;
                self.stream.fence()?;
                let gate_up = crate::gemma4_int4::Int4LinearW4a8 {
                    b_int4_reordered: gu_w.device_ptr(),
                    b_scales_packed: gu_s.device_ptr(),
                    n: gate_up_n as i32,
                    k: hidden as i32,
                    group_size: group_size as i32,
                };

                // --- down proj (single Linear). ---
                let down = crate::gemma4_int4::encode_from_pack(
                    _w4a8,
                    fn_dequant,
                    &l.down_proj,
                    f16_scratch.device_ptr(),
                    dn_w.device_ptr(),
                    dn_s.device_ptr(),
                    scales_f32_ws.device_ptr(),
                    self.stream.raw(),
                )?;
                self.stream.fence()?;
                (qkv, o, gate_up, down)
            };
            // Non-cuda: build the handles with the dst device ptrs so the
            // structure is sound; the actual encode is a GPU op skipped here.
            #[cfg(not(feature = "cuda"))]
            let (qkv, o, gate_up, down) = {
                let mk = |w: &rvllm_mem::hbm::Region,
                          s: &rvllm_mem::hbm::Region,
                          n: usize,
                          k: usize| crate::gemma4_int4::Int4LinearW4a8 {
                    b_int4_reordered: w.device_ptr(),
                    b_scales_packed: s.device_ptr(),
                    n: n as i32,
                    k: k as i32,
                    group_size: group_size as i32,
                };
                (
                    mk(&qkv_w, &qkv_s, qkv_n, hidden),
                    mk(&o_w, &o_s, o_n, o_k),
                    mk(&gu_w, &gu_s, gate_up_n, hidden),
                    mk(&dn_w, &dn_s, down_n, down_k),
                )
            };

            layers.push(Int4DecoderLayer {
                qkv,
                o,
                gate_up,
                down,
            });
        }

        // Pruned lm-head handle. The head pack stays as loaded (the greedy
        // argmax tail reads it directly — no w4a8 reorder). `keep_ids` is a
        // host `Vec<u32>` on the loader handle; the argmax-remap kernel needs
        // it on device, so upload it once here into an arena region.
        let keep_ids_host: Vec<i32> = e4b.lm_head.keep_ids.iter().map(|&x| x as i32).collect();
        let keep_ids_region =
            self.arena
                .region("e4b_int4_lmhead_keep_ids", keep_ids_host.len() * 4, 256)?;
        #[cfg(feature = "cuda")]
        unsafe {
            let bytes = std::slice::from_raw_parts(
                keep_ids_host.as_ptr() as *const u8,
                keep_ids_host.len() * 4,
            );
            keep_ids_region.copy_from_host(bytes)?;
        }
        let lm_head = crate::gemma4_int4::LmHeadPruned {
            w: e4b.lm_head.head.clone(),
            keep_ids: keep_ids_region.device_ptr(),
            k_rows: e4b.lm_head.pruned_vocab_k as i32,
            full_vocab: e4b.lm_head.full_vocab as i32,
        };

        self.e4b_int4 = Some(crate::gemma4_int4::Gemma4Int4Runtime {
            layers,
            lm_head,
            workspace: workspace.device_ptr(),
            workspace_bytes,
        });
        Ok(())
    }

    /// Run the once-per-step PLE combine pipeline before the decoder
    /// layer loop. Gathers the per-layer embedding rows for `token_ids`,
    /// projects the model residual through `per_layer_model_projection`, and
    /// combines them into `e4b.per_layer_inputs` (the buffer each layer's PLE
    /// gate reads). `embed_residual` is the main residual at model input
    /// (`embed_tokens(id) * sqrt(hidden)`), already gathered by the caller.
    ///
    /// No-op when `self.e4b` is `None`. CUDA-only (launches kernels).
    ///
    /// # Safety
    /// `token_ids` is an i32 device ptr `[num_tokens]`; `embed_residual` is an
    /// f16 device ptr `[num_tokens, hidden]`. `num_tokens <= e4b_max_tokens`.
    #[cfg(feature = "cuda")]
    pub unsafe fn run_ple_combine(
        &self,
        token_ids: u64,
        embed_residual: u64,
        num_tokens: u32,
        stream: u64,
    ) -> Result<()> {
        let Some(rt) = self.e4b.as_ref() else {
            return Ok(());
        };
        let e4b = self
            .e4b_model
            .as_ref()
            .expect("e4b runtime set but e4b_model missing");
        if num_tokens > self.e4b_max_tokens {
            return Err(rvllm_core::RvllmError::config(
                rvllm_core::ConfigError::Inconsistent {
                    reasons: vec![format!(
                        "run_ple_combine: num_tokens {num_tokens} exceeds buffer cap {}",
                        self.e4b_max_tokens
                    )],
                },
                "RVLLM_E4B",
            ));
        }
        let arch = &self.arch;
        let num_layers = arch.num_hidden_layers as u32;
        let h_ple = rt.h_ple;
        let hidden = arch.hidden_size as u32;
        let row = num_layers * h_ple; // L*h_ple

        // 1) Gather the per-layer embed rows. `embed_tokens_per_layer` is
        //    [vocab, L*h_ple] bf16 (scale folded at load), so a plain
        //    embedding gather with hidden=L*h_ple yields the per-layer inputs.
        let fn_embed = self.fused.fn_embed_gather.as_ref().ok_or_else(|| {
            rvllm_core::RvllmError::config(
                rvllm_core::ConfigError::Inconsistent {
                    reasons: vec!["run_ple_combine: embedding_gather_f16 kernel absent".into()],
                },
                "RVLLM_E4B",
            )
        })?;
        rvllm_fused::EmbeddingGatherLaunch {
            num_tokens,
            hidden: row,
            vocab: arch.vocab_size as u32,
        }
        .launch(
            fn_embed,
            self.e4b_gather_buf,
            e4b.ple.embed_tokens_per_layer.offset_bytes,
            token_ids,
            stream,
        )?;

        // 2) proj_in = embed_residual @ per_layer_model_projection^T
        //    [T, hidden] x [L*h_ple, hidden]^T -> [T, L*h_ple] (f32 then f16).
        self.cublaslt.f16_gemm_f32(
            embed_residual,
            self.e4b_proj_w_f16,
            self.e4b_proj_f32_buf,
            num_tokens as i32,
            row as i32,
            hidden as i32,
            stream,
        )?;
        // f32 -> f16 (saturating) for the combine kernel. `Bf16ToF16SatLaunch`
        // drives the `f32_to_f16_sat_kernel` (dst f16, src f32).
        rvllm_fused::gemma4_launcher::Bf16ToF16SatLaunch {
            n: num_tokens * row,
        }
        .launch(
            &self.fused.fn_f32_to_f16_sat,
            self.e4b_proj_f16_buf,
            self.e4b_proj_f32_buf,
            stream,
        )?;

        // 3) combine: out = (rmsnorm(proj*hidden^-0.5) + gathered) * 2^-0.5.
        let combine = self
            .fused
            .fn_ple_projection_combine
            .as_ref()
            .ok_or_else(|| {
                rvllm_core::RvllmError::config(
                    rvllm_core::ConfigError::Inconsistent {
                        reasons: vec![
                            "run_ple_combine: PLE projection-combine kernel absent".into()
                        ],
                    },
                    "RVLLM_E4B",
                )
            })?;
        rvllm_fused::PleProjectionCombineLaunch {
            num_tokens,
            num_layers,
            h_ple,
            hidden,
            eps: arch.rms_norm_eps,
        }
        .launch(
            combine,
            self.e4b_proj_f16_buf,
            self.e4b_gather_buf,
            e4b.ple.per_layer_projection_norm.offset_bytes,
            rt.per_layer_inputs,
            stream,
        )?;
        #[cfg(feature = "cuda")]
        if std::env::var("RVLLM_DBG_PLE").is_ok() {
            cudarc::driver::sys::cuStreamSynchronize(stream as _);
            let mut er = [0u16; 8];
            let mut pw = [0u16; 8];
            let mut g = [0u16; 8];
            let mut pf = [0u16; 8];
            let mut o = [0u16; 8];
            cudarc::driver::sys::cuMemcpyDtoH_v2(er.as_mut_ptr() as *mut _, embed_residual, 16);
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                pw.as_mut_ptr() as *mut _,
                self.e4b_proj_w_f16,
                16,
            );
            cudarc::driver::sys::cuMemcpyDtoH_v2(g.as_mut_ptr() as *mut _, self.e4b_gather_buf, 16);
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                pf.as_mut_ptr() as *mut _,
                self.e4b_proj_f16_buf,
                16,
            );
            cudarc::driver::sys::cuMemcpyDtoH_v2(o.as_mut_ptr() as *mut _, rt.per_layer_inputs, 16);
            let f = |b: &[u16; 8]| -> Vec<f32> {
                b.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect()
            };
            eprintln!("[PLE-combine] embed_residual={:.3?}", f(&er));
            eprintln!("[PLE-combine] proj_w_f16={:.3?}", f(&pw));
            eprintln!("[PLE-combine] gather(bf16-as-f16)={:.3?}", f(&g));
            eprintln!("[PLE-combine] proj_f16={:.3?}", f(&pf));
            eprintln!("[PLE-combine] per_layer_inputs={:.3?}", f(&o));
            // Per-layer gate kernel weights (bf16) for layer 0 — the suspected
            // NaN source for the PLE gate injection.
            let l0 = &rt.layers[0];
            let mut gw = [0u16; 8];
            let mut pjw = [0u16; 8];
            let mut pn = [0u16; 8];
            cudarc::driver::sys::cuMemcpyDtoH_v2(gw.as_mut_ptr() as *mut _, l0.gate_w, 16);
            cudarc::driver::sys::cuMemcpyDtoH_v2(pjw.as_mut_ptr() as *mut _, l0.proj_w, 16);
            cudarc::driver::sys::cuMemcpyDtoH_v2(pn.as_mut_ptr() as *mut _, l0.post_norm, 16);
            let bf = |b: &[u16; 8]| -> Vec<f32> {
                b.iter()
                    .map(|&x| f32::from_bits((x as u32) << 16))
                    .collect()
            };
            eprintln!("[PLE-gate-w] gate_w(bf16)={:.4?}", bf(&gw));
            eprintln!("[PLE-gate-w] proj_w(bf16)={:.4?}", bf(&pjw));
            eprintln!("[PLE-gate-w] post_norm(bf16)={:.4?}", bf(&pn));
        }
        Ok(())
    }

    /// True when this engine is on the E4B path: `RVLLM_E4B=1` was
    /// set AND the loaded config is E4B, so `load` built the INT4 E4B handle.
    /// The serve worker / bench use this to decide whether the E4B forward +
    /// PLE combine + pruned-lm-head argmax path applies vs the 31B FP8 path.
    pub fn is_e4b_engine(&self) -> bool {
        self.e4b_model.is_some()
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn attention_workspace_bytes(
        &self,
        active_layers: usize,
        num_tokens: u32,
        num_seqs: u32,
        max_seqlen_q: u32,
        block_size: u32,
        num_blocks_total: u32,
        sliding_blocks: u32,
        max_blocks_per_seq: u32,
        sliding_window_size: u32,
        f16_kv: bool,
        include_prefill: bool,
    ) -> Result<usize> {
        use rvllm_loader::gemma4_arch::Gemma4LayerType;

        let mut required = 0usize;
        for layer_idx in 0..active_layers.min(self.arch.num_hidden_layers) {
            let layer_type = self.arch.layer_types[layer_idx];
            let head_dim = self.arch.head_dim_for_layer(layer_idx) as u32;
            let num_kv_heads = self.arch.num_kv_heads_for_layer(layer_idx) as u32;
            let layer_blocks_total = if layer_type == Gemma4LayerType::GlobalAttention {
                num_blocks_total
            } else {
                sliding_blocks
            };
            let layer_max_blocks_per_seq = max_blocks_per_seq.min(layer_blocks_total);
            let window_size_left = if layer_type == Gemma4LayerType::SlidingAttention {
                sliding_window_size.saturating_sub(1) as i32
            } else {
                -1
            };
            let layer_attention = match layer_type {
                Gemma4LayerType::SlidingAttention => &self.sliding_attention,
                Gemma4LayerType::GlobalAttention => &self.global_attention,
            };
            let decode_attention = if f16_kv {
                layer_attention
            } else {
                &self.global_attention
            };
            let decode = rvllm_attention::PagedDecodeParams {
                num_seqs,
                num_heads: self.arch.num_attention_heads as u32,
                num_kv_heads,
                head_dim,
                block_size,
                max_blocks_per_seq: layer_max_blocks_per_seq,
                num_blocks_total: layer_blocks_total,
                scale: 1.0,
                window_size_left,
            };
            required = required.max(decode_attention.decode_workspace_size(&decode, !f16_kv)?);

            if include_prefill {
                let prefill = rvllm_attention::PagedPrefillParams {
                    num_tokens,
                    num_seqs,
                    num_heads: self.arch.num_attention_heads as u32,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    max_blocks_per_seq: layer_max_blocks_per_seq,
                    num_blocks_total: layer_blocks_total,
                    scale: 1.0,
                    window_size_left,
                };
                required = required.max(
                    self.global_attention
                        .prefill_workspace_size(&prefill, max_seqlen_q)?,
                );

                let single_query = rvllm_attention::PagedDecodeParams {
                    num_seqs: 1,
                    ..decode
                };
                required = required.max(
                    self.global_attention
                        .decode_workspace_size(&single_query, true)?,
                );
                required =
                    required.max(layer_attention.decode_workspace_size(&single_query, true)?);
            }
        }
        Ok(required.max(256))
    }

    #[cfg(feature = "cuda")]
    pub unsafe fn run_bench(
        &self,
        num_seqs: u32,
        iters: u32,
        warmup: u32,
    ) -> Result<crate::bring_up::BenchResult> {
        use crate::gemma4_layer_exec::*;
        use rvllm_loader::gemma4_arch::Gemma4LayerType;

        let f16_only = false; // bench path always FP8
        let arch = &self.arch;
        let hidden = arch.hidden_size as u32;
        let max_hd = arch.max_head_dim() as u32;
        let max_nkvh = arch.max_kv_heads() as u32;
        let max_q_dim = (arch.num_attention_heads * arch.max_head_dim()) as u32;
        let max_kv_dim = (max_nkvh * max_hd) as u32;
        let max_qkv_rows = max_q_dim + 2 * max_kv_dim;
        let inter = arch.intermediate_size as u32;
        let vocab = arch.vocab_size as u32;
        let stream = self.stream.raw();

        let block_size: u32 = std::env::var("RVLLM_BLOCK_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);
        let num_blocks_total: u32 = std::env::var("RVLLM_NUM_BLOCKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        if num_seqs == 0 || block_size == 0 || num_blocks_total < num_seqs {
            return Err(rvllm_core::RvllmError::config(
                rvllm_core::ConfigError::Inconsistent {
                    reasons: vec![format!(
                        "benchmark requires nonzero block size and at least one KV block per sequence: block_size={block_size} num_blocks={num_blocks_total} sequences={num_seqs}"
                    )],
                },
                "RVLLM_NUM_BLOCKS",
            ));
        }
        let max_blocks_per_seq = num_blocks_total / num_seqs;
        let benchmark_sliding_window =
            (arch.sliding_window_size as u32).min(max_blocks_per_seq.saturating_mul(block_size));
        let sequence_slot_stride = u64::from(max_blocks_per_seq) * u64::from(block_size);
        let last_sequence_slot = u64::from(num_seqs - 1) * sequence_slot_stride;
        if last_sequence_slot > i32::MAX as u64 {
            return Err(rvllm_core::RvllmError::config(
                rvllm_core::ConfigError::Inconsistent {
                    reasons: vec!["benchmark KV slot mapping exceeds the i32 kernel ABI".into()],
                },
                "RVLLM_NUM_BLOCKS",
            ));
        }

        let arena = &self.arena;
        let hidden_fp8 = arena.region("hidden_fp8", (num_seqs * hidden) as usize, 16)?;
        let hidden_scale = arena.region("hidden_scale", (num_seqs * 4) as usize, 16)?;
        let qkv_out = arena.region("qkv_out", (num_seqs * max_qkv_rows * 2) as usize, 16)?;
        let q_base = qkv_out.device_ptr();
        let q_normed = arena.region("q_normed", (num_seqs * max_q_dim * 2) as usize, 16)?;
        let k_normed = arena.region("k_normed", (num_seqs * max_kv_dim * 2) as usize, 16)?;
        let v_normed = arena.region("v_normed", (num_seqs * max_kv_dim * 2) as usize, 16)?;
        let q_fp8 = arena.region("q_fp8", (num_seqs * max_q_dim) as usize, 16)?;
        let attn_out = arena.region("attn_out", (num_seqs * max_q_dim * 2) as usize, 16)?;
        let attn_out_fp8 = arena.region("attn_out_fp8", (num_seqs * max_q_dim) as usize, 16)?;
        let attn_out_scale = arena.region("attn_out_scale", (num_seqs * 4) as usize, 16)?;
        let gate_up_out = arena.region("gate_up_out", (num_seqs * 2 * inter * 2) as usize, 16)?;
        let gate_up_fp8 = arena.region("gate_up_fp8", (num_seqs * 2 * inter) as usize, 16)?;
        let gate_up_scale = arena.region("gate_up_scale", (num_seqs * 4) as usize, 16)?;
        let mlp_out_fp8 = arena.region("mlp_out_fp8", (num_seqs * inter) as usize, 16)?;
        let mlp_out_scale = arena.region("mlp_out_scale", (num_seqs * 4) as usize, 16)?;
        let delta_f16 = arena.region("delta_f16", (num_seqs * hidden * 2) as usize, 16)?;
        let ple_gate = arena.region(
            "ple_gate_bench",
            (num_seqs * arch.hidden_size_per_layer_input as u32 * 2) as usize,
            16,
        )?;
        let gemm_f32_max_n = std::cmp::max(max_qkv_rows, 2 * inter);
        let gemm_f32_tmp =
            arena.region("gemm_f32_tmp", (num_seqs * gemm_f32_max_n * 4) as usize, 16)?;

        let use_f16_kv = f16_only || std::env::var("RVLLM_F16_KV").map_or(false, |v| v != "0");
        let kv_bytes_per_elem: u32 = if use_f16_kv { 2 } else { 1 };
        // Sliding-layer slots must remain inside the ring allocation.
        // at every t the rope writes; the old cap sliding_blocks = sliding_window/block_size = 32
        // (= 1024 slots for Gemma 4) broke at prompt_len > sliding_window because slot_mapping
        // is linear 0..prompt_len-1 and index 1024+ ran off the end of the sliding KV region.
        // Proper fix is a per-sliding-layer ring buffer (slot = t mod sliding_window) but that
        // needs rope + attention kernel cooperation. For now give sliding layers the full pool —
        // ~10 GiB extra at num_blocks_total=1024, fits in the 50 GiB arena with Gemma 4 31B fp8.
        let sliding_blocks = num_blocks_total;

        let mut kv_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_total_bytes: u64 = 0;
        let mut kv_scale_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_scale_total_bytes: u64 = 0;
        for l in 0..arch.num_hidden_layers {
            kv_layer_offsets.push(kv_total_bytes);
            kv_scale_layer_offsets.push(kv_scale_total_bytes);
            let layer_blocks = if arch.layer_types[l]
                == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention
            {
                num_blocks_total
            } else {
                sliding_blocks
            };
            let nkvh_l = arch.num_kv_heads_for_layer(l) as u32;
            let hd_l = arch.head_dim_for_layer(l) as u32;
            let layer_elems =
                2u64 * layer_blocks as u64 * block_size as u64 * nkvh_l as u64 * hd_l as u64;
            kv_total_bytes += layer_elems * kv_bytes_per_elem as u64;
            let layer_scale_slots = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh_l as u64;
            kv_scale_total_bytes += layer_scale_slots * 4;
        }
        let kv_cache = arena.region("kv_cache", kv_total_bytes as usize, 256)?;
        let kv_scale_cache = arena.region("kv_scale_cache", kv_scale_total_bytes as usize, 16)?;
        // Per-(seq, head) Q scale scratch, written fresh by rope each
        // forward and consumed by the same step's attention.
        let q_scale_scratch_bytes = (num_seqs as u64) * (arch.num_attention_heads as u64) * 4;
        let q_scale_scratch =
            arena.region("q_scale_scratch", q_scale_scratch_bytes as usize, 16)?;
        // Opt-out for A/B testing: RVLLM_PER_TOKEN_Q_SCALE=0 falls back to
        // the scalar q_scale_ptr for explicit calibration comparisons.
        let q_scale_cache_ptr: u64 =
            if std::env::var("RVLLM_PER_TOKEN_Q_SCALE").ok().as_deref() == Some("0") {
                0
            } else {
                q_scale_scratch.device_ptr()
            };
        #[cfg(feature = "cuda")]
        {
            cudarc::driver::sys::cuMemsetD8_v2(kv_cache.device_ptr(), 0, kv_total_bytes as usize);
            cudarc::driver::sys::cuMemsetD8_v2(
                kv_scale_cache.device_ptr(),
                0,
                kv_scale_total_bytes as usize,
            );
            cudarc::driver::sys::cuMemsetD8_v2(
                q_scale_scratch.device_ptr(),
                0,
                q_scale_scratch_bytes as usize,
            );
        }

        let q_scale_region = arena.region("q_scale", 4, 4)?;
        let kv_scale_region = arena.region("kv_scale", 4, 4)?;
        {
            let q_s: f32 = std::env::var("RVLLM_Q_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_Q_SCALE);
            let kv_s: f32 = std::env::var("RVLLM_KV_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_KV_SCALE);
            q_scale_region.copy_from_host(&q_s.to_le_bytes())?;
            kv_scale_region.copy_from_host(&kv_s.to_le_bytes())?;
        }

        let fa3_ws_bytes = self.attention_workspace_bytes(
            self.model.layers.len(),
            num_seqs,
            num_seqs,
            1,
            block_size,
            num_blocks_total,
            sliding_blocks,
            max_blocks_per_seq,
            benchmark_sliding_window,
            use_f16_kv,
            false,
        )?;
        let fa3_ws = arena.region("fa3_ws", fa3_ws_bytes, 256)?;
        let residual = arena.region("residual", (num_seqs * hidden * 2) as usize, 16)?;
        cudarc::driver::sys::cuMemsetD8_v2(
            residual.device_ptr(),
            0,
            (num_seqs * hidden * 2) as usize,
        );

        let positions = arena.region("positions", (num_seqs * 4) as usize, 16)?;
        let slot_mapping = arena.region("slot_mapping", (num_seqs * 4) as usize, 16)?;
        let context_lens = arena.region("context_lens", (num_seqs * 4) as usize, 16)?;
        let block_tables = arena.region(
            "block_tables",
            (num_seqs * max_blocks_per_seq * 4) as usize,
            16,
        )?;
        {
            let n = num_seqs as usize;
            let pos: Vec<i32> = vec![0; n];
            let slot: Vec<i32> = (0..n)
                .map(|i| (i as u64 * sequence_slot_stride) as i32)
                .collect();
            let ctx: Vec<i32> = vec![1; n];
            let mut bt: Vec<i32> = Vec::with_capacity(n * max_blocks_per_seq as usize);
            for i in 0..n {
                for b in 0..max_blocks_per_seq as usize {
                    bt.push((i * max_blocks_per_seq as usize + b) as i32);
                }
            }
            positions.copy_from_host(bytemuck_cast_i32(&pos))?;
            slot_mapping.copy_from_host(bytemuck_cast_i32(&slot))?;
            context_lens.copy_from_host(bytemuck_cast_i32(&ctx))?;
            block_tables.copy_from_host(bytemuck_cast_i32(&bt))?;
        }

        let logits = arena.region("logits", (num_seqs * vocab * 2) as usize, 16)?;
        let sampled_tokens = arena.region("sampled_tokens", (num_seqs * 4) as usize, 16)?;
        let cutlass_ws_bytes: usize = 16 * 1024 * 1024;
        let cutlass_ws = arena.region("cutlass_ws_gemma4", cutlass_ws_bytes, 256)?;
        let residual_ptr = residual.device_ptr();
        let kernels = self.layer_kernels();

        let e4b_rt = self.e4b.as_ref();
        // INT4 decoder runtime + w4a8 lib for routing the per-layer GEMMs.
        // Both `Some` only on the E4B+INT4 path; `None` otherwise (FP8).
        let int4_rt = self.e4b_int4.as_ref();
        let w4a8_lib = self.w4a8.as_ref();
        let one_step = || -> rvllm_core::Result<()> {
            for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                let lt = arch.layer_types[layer_idx];
                let hd = arch.head_dim_for_layer(layer_idx) as u32;
                let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
                let q_dim = (arch.num_attention_heads as u32) * hd;
                let kv_dim = nkvh * hd;
                let _qkv_rows = q_dim + 2 * kv_dim;
                let layer_blocks = if lt == Gemma4LayerType::GlobalAttention {
                    num_blocks_total
                } else {
                    sliding_blocks
                };

                // E4B per-layer setup: KV-share source + the runtime
                // sliding-window override (256). `None` for the 31B path.
                let e4b_layer = e4b_rt.map(|rt| &rt.layers[layer_idx]);
                let kv_share_src = e4b_layer.and_then(|l| l.kv_share_src);
                let layer_sliding_window = match e4b_rt {
                    Some(rt) => rt.sliding_window,
                    None => benchmark_sliding_window,
                };

                let dims = Gemma4LayerDims {
                    num_tokens: num_seqs,
                    hidden,
                    num_heads: arch.num_attention_heads as u32,
                    num_kv_heads: nkvh,
                    head_dim: hd,
                    rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                    rope_table_rows: arch.max_position_embeddings as u32,
                    intermediate: inter,
                    block_size,
                    max_blocks_per_seq,
                    num_blocks_total: layer_blocks,
                    attn_scale: 1.0,
                    rms_eps: arch.rms_norm_eps,
                    layer_type: lt,
                    sliding_window: layer_sliding_window,
                    f16_kv: f16_only || std::env::var("RVLLM_F16_KV").map_or(false, |v| v != "0"),
                    num_hidden_layers: arch.num_hidden_layers as u32,
                    layer_idx: layer_idx as u32,
                    ple_dim: arch.hidden_size_per_layer_input as u32,
                    kv_shared: kv_share_src.is_some(),
                };

                // Row-major [num_tokens, q_dim+2*kv_dim]: k_out / v_out
                // point at row 0's K / V sub-slice. The rmsnorm kernel
                // applies `src_row_stride` to reach later tokens.
                let k_out = q_base + (q_dim as u64) * 2;
                let v_out = k_out + (kv_dim as u64) * 2;
                let is_global = lt == Gemma4LayerType::GlobalAttention;
                let layer_blocks = if is_global {
                    num_blocks_total
                } else {
                    sliding_blocks
                };
                let layer_kv_elems =
                    2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                let kv_layer_bytes = layer_kv_elems * kv_bytes_per_elem as u64;
                // KV-share: a shared tail layer reads the SOURCE layer's KV
                // cache (already written this step) and writes none. Source
                // and shared layer share the same attention type, so block
                // budget / kv-head count / head_dim all match — only the
                // base offset changes.
                let kv_src_layer = kv_share_src.unwrap_or(layer_idx);
                let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[kv_src_layer];
                let layer_kv_scale_base =
                    kv_scale_cache.device_ptr() + kv_scale_layer_offsets[kv_src_layer];
                let layer_kv_scale_slots_half =
                    (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);

                let (cos, sin) = match lt {
                    Gemma4LayerType::SlidingAttention => (
                        self.model.rope_cos_sliding.offset_bytes,
                        self.model.rope_sin_sliding.offset_bytes,
                    ),
                    Gemma4LayerType::GlobalAttention => (
                        self.model.rope_cos_global.offset_bytes,
                        self.model.rope_sin_global.offset_bytes,
                    ),
                };

                let w = Gemma4LayerWeightPtrs {
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

                let scratch = Gemma4LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(),
                    hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base,
                    k_out,
                    v_out,
                    q_normed: q_normed.device_ptr(),
                    k_normed: k_normed.device_ptr(),
                    v_normed: v_normed.device_ptr(),
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + kv_layer_bytes / 2,
                    k_scale_cache: layer_kv_scale_base,
                    v_scale_cache: layer_kv_scale_base + layer_kv_scale_slots_half * 4,
                    q_scale_cache: q_scale_cache_ptr,
                    q_scale_ptr: q_scale_region.device_ptr(),
                    kv_scale_ptr: kv_scale_region.device_ptr(),
                    attn_out: attn_out.device_ptr(),
                    attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(),
                    delta_f16: delta_f16.device_ptr(),
                    gate_up_out: gate_up_out.device_ptr(),
                    gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(),
                    mlp_out_scale: mlp_out_scale.device_ptr(),
                    gemm_f32_tmp: gemm_f32_tmp.device_ptr(),
                    cutlass_workspace: cutlass_ws.device_ptr(),
                    cutlass_workspace_bytes: cutlass_ws_bytes,
                    fa3_workspace: fa3_ws.device_ptr(),
                    fa3_workspace_bytes: fa3_ws_bytes,
                    ple_inputs: 0,
                    ple_gate: ple_gate.device_ptr(),
                };

                let meta = Gemma4MetadataPtrs {
                    positions: positions.device_ptr(),
                    slot_mapping: slot_mapping.device_ptr(),
                    cos,
                    sin,
                    block_tables: block_tables.device_ptr(),
                    context_lens: context_lens.device_ptr(),
                };

                match (e4b_rt, e4b_layer) {
                    (Some(rt), Some(pl)) => {
                        // E4B path: layer body (with KV-share + sliding-256
                        // baked into `dims`) + PLE gate injection. The PLE
                        // per-layer-input pointer indexes the combined
                        // `[num_tokens, num_layers, h_ple]` buffer at
                        // `layer_idx * h_ple` (num_tokens == 1 decode). For
                        // num_tokens > 1 the kernel reads
                        // `token*num_layers + layer`; the runtime here is
                        // decode (num_seqs rows) so the per-(token,layer)
                        // stride is supplied via the buffer layout — the
                        // launcher passes the layer-base and the kernel
                        // strides by `token * h_ple` within the launch's
                        // num_tokens. See Gemma4PleLayer docs.
                        let h_ple = rt.h_ple;
                        let pli_base =
                            rt.per_layer_inputs + (layer_idx as u64) * (h_ple as u64) * 2; // f16
                        let ple = crate::gemma4_layer_exec::Gemma4PleLayer {
                            gate_w: pl.gate_w,
                            proj_w: pl.proj_w,
                            per_layer_input: pli_base,
                            post_norm_gamma: pl.post_norm,
                            h_ple,
                            // [T, L, h_ple] -> per-token stride = num_layers * h_ple.
                            pli_stride: rt.layers.len() as u32 * h_ple,
                        };
                        let int4 = match (int4_rt, w4a8_lib) {
                            (Some(rt), Some(lib)) => Some(crate::gemma4_int4::Int4LayerExec {
                                w4a8: lib,
                                layer: &rt.layers[layer_idx],
                                workspace: rt.workspace,
                                workspace_bytes: rt.workspace_bytes,
                            }),
                            _ => None,
                        };
                        crate::gemma4_layer_exec::gemma4_e4b_layer_forward(
                            dims,
                            &kernels,
                            &w,
                            &scratch,
                            &meta,
                            &self.cublaslt,
                            &self.cutlass,
                            &self.sliding_attention,
                            &self.global_attention,
                            residual_ptr,
                            stream,
                            crate::gemma4_layer_exec::Gemma4Phase::Decode,
                            Some(ple),
                            int4,
                        )?;
                    }
                    _ => {
                        gemma4_forward(
                            dims,
                            &kernels,
                            &w,
                            &scratch,
                            &meta,
                            &self.cublaslt,
                            &self.cutlass,
                            &self.sliding_attention,
                            &self.global_attention,
                            residual_ptr,
                            stream,
                        )?;
                    }
                }
            }

            // LM head.
            if let Some(int4) = int4_rt {
                // INT4 pruned greedy tail: f16 final norm (in place) then the
                // pruned-head GEMV -> bf16 score -> argmax-over-kept-rows ->
                // remap to global token id (writes `sampled_tokens` directly).
                // No FP8 lm-head GEMM, no full-vocab softcap/argmax. Greedy is
                // identity-preserving under softcap (monotone tanh), so the
                // softcap is skipped on the argmax path (spec §0.4/§1.6).
                rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                    num_tokens: num_seqs,
                    hidden,
                    eps: arch.rms_norm_eps,
                }
                .launch(
                    kernels.fused_rmsnorm,
                    residual_ptr,
                    self.model.final_norm.offset_bytes,
                    stream,
                )?;
                let (fn_gemv, fn_argmax_remap) = match (
                    self.fused.fn_lmhead_int4_gemv.as_ref(),
                    self.fused.fn_lmhead_argmax_remap.as_ref(),
                ) {
                    (Some(g), Some(a)) => (g, a),
                    _ => {
                        return Err(rvllm_core::RvllmError::config(
                            rvllm_core::ConfigError::Inconsistent {
                                reasons: vec!["RVLLM_INT4 on but the lmhead_prune_argmax kernels \
                                     are absent from the bundle — cannot run the pruned \
                                     greedy tail (fail-closed)"
                                    .into()],
                            },
                            "RVLLM_INT4",
                        ));
                    }
                };
                // `logits` region (num_seqs*vocab*2 bytes) reused as the f32
                // scores scratch (needs num_seqs*k_rows*4 <= num_seqs*vocab*2
                // only when k_rows <= vocab/2; the pruned k_rows (16k) is far
                // below vocab (262144), so it fits with room to spare).
                crate::gemma4_int4::lmhead_prune_argmax(
                    fn_gemv,
                    fn_argmax_remap,
                    &int4.lm_head,
                    residual_ptr,
                    logits.device_ptr(),
                    sampled_tokens.device_ptr(),
                    num_seqs,
                    hidden,
                    stream,
                )?;
                return Ok(());
            }
            // FP8 path: final norm + FP8 quant + GEMM + softcap + argmax
            rvllm_fused::FusedRmsnormFp8QuantLaunch {
                num_tokens: num_seqs,
                hidden,
                eps: arch.rms_norm_eps,
            }
            .launch(
                kernels.fused_rmsnorm_fp8_quant,
                hidden_fp8.device_ptr(),
                hidden_scale.device_ptr(),
                residual_ptr,
                self.model.final_norm.offset_bytes,
                stream,
            )?;
            self.cublaslt.fp8_gemm(
                hidden_fp8.device_ptr(),
                self.model.lm_head_fp8.offset_bytes,
                logits.device_ptr(),
                num_seqs as i32,
                vocab as i32,
                hidden as i32,
                hidden_scale.device_ptr(),
                self.model.lm_head_fp8.scale_ptr,
                stream,
            )?;
            logit_softcap(
                &self.fused.fn_softcap,
                logits.device_ptr(),
                num_seqs,
                vocab,
                arch.logit_softcap,
                stream,
            )?;
            rvllm_fused::ArgmaxLaunch {
                num_tokens: num_seqs,
                vocab,
            }
            .launch(
                &self.fused.fn_argmax,
                logits.device_ptr(),
                sampled_tokens.device_ptr(),
                stream,
            )?;
            Ok(())
        };

        // Warmup
        for _ in 0..warmup {
            one_step()?;
        }
        self.stream.fence()?;

        // Timed
        let no_graph = std::env::var("RVLLM_NO_GRAPH").ok().as_deref() == Some("1");
        let elapsed = if no_graph {
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                one_step()?;
            }
            self.stream.fence()?;
            t0.elapsed()
        } else {
            let graph = rvllm_graph::CapturedGraph::capture(
                &self.ctx,
                num_seqs,
                max_blocks_per_seq,
                rvllm_metadata::MetadataLayout::compute(num_seqs, max_blocks_per_seq)?.hash(),
                stream,
                || one_step(),
            )?;
            self.stream.fence()?;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                graph.replay(stream)?;
            }
            self.stream.fence()?;
            t0.elapsed()
        };

        Ok(crate::bring_up::BenchResult {
            ns_per_step: elapsed.as_nanos() / iters.max(1) as u128,
            total_ns: elapsed.as_nanos(),
            iters,
            num_seqs,
            ttft_ns: None,
            ttft_hot_ns: None,
        })
    }

    #[cfg(feature = "cuda")]
    unsafe fn build_ple_inputs(
        &self,
        fn_embed: &rvllm_kernels::KernelFn,
        token_ids_region: u64,
        residual: u64,
        ple_inputs: u64,
        ple_projection_f16: u64,
        ple_projection_f32: u64,
        num_tokens: u32,
        stream: u64,
    ) -> Result<()> {
        let Some(ple_embedding) = self.model.embed_tokens_per_layer.as_ref() else {
            return Ok(());
        };
        let Some(ple_model_projection) = self.model.per_layer_model_projection_f16.as_ref() else {
            return Ok(());
        };
        let Some(ple_projection_norm) = self.model.per_layer_projection_norm.as_ref() else {
            return Ok(());
        };
        let ple_dim = self.arch.hidden_size_per_layer_input as u32;
        if ple_dim == 0 {
            return Ok(());
        }
        let total_ple_dim = (self.arch.num_hidden_layers as u32) * ple_dim;

        rvllm_fused::EmbeddingGatherLaunch {
            num_tokens,
            hidden: total_ple_dim,
            vocab: self.arch.vocab_size_per_layer_input as u32,
        }
        .launch(
            fn_embed,
            ple_inputs,
            ple_embedding.offset_bytes,
            token_ids_region,
            stream,
        )?;
        self.cublaslt.f16_gemm_f32(
            residual,
            ple_model_projection.offset_bytes,
            ple_projection_f32,
            num_tokens as i32,
            total_ple_dim as i32,
            self.arch.hidden_size as i32,
            stream,
        )?;
        rvllm_fused::gemma4_launcher::Bf16ToF16SatLaunch {
            n: num_tokens * total_ple_dim,
        }
        .launch(
            &self.fused.fn_f32_to_f16_sat,
            ple_projection_f16,
            ple_projection_f32,
            stream,
        )?;

        let mut out = ple_inputs;
        let mut projection = ple_projection_f16;
        let mut embeds = ple_inputs;
        let mut norm_gamma = ple_projection_norm.offset_bytes;
        let mut num_layers = self.arch.num_hidden_layers as i32;
        let mut ple_dim_i = ple_dim as i32;
        let mut projection_scale = (self.arch.hidden_size as f32).powf(-0.5);
        let mut combine_scale = 0.7071067811865476f32;
        let mut eps = self.arch.rms_norm_eps;
        let args = [
            (&mut out) as *mut u64 as *mut core::ffi::c_void,
            (&mut projection) as *mut u64 as *mut core::ffi::c_void,
            (&mut embeds) as *mut u64 as *mut core::ffi::c_void,
            (&mut norm_gamma) as *mut u64 as *mut core::ffi::c_void,
            (&mut num_layers) as *mut i32 as *mut core::ffi::c_void,
            (&mut ple_dim_i) as *mut i32 as *mut core::ffi::c_void,
            (&mut projection_scale) as *mut f32 as *mut core::ffi::c_void,
            (&mut combine_scale) as *mut f32 as *mut core::ffi::c_void,
            (&mut eps) as *mut f32 as *mut core::ffi::c_void,
        ];
        let block = (ple_dim.min(256), 1, 1);
        let grid = (num_tokens, self.arch.num_hidden_layers as u32, 1);
        rvllm_fused::launch_raw(
            &self.fused.fn_ple_project_combine,
            grid,
            block,
            32 * 4,
            stream,
            &args,
        )
    }

    #[cfg(feature = "cuda")]
    pub unsafe fn run_ppl(
        &self,
        fn_embed: &rvllm_kernels::KernelFn,
        token_ids: &[u32],
        score_from: usize,
    ) -> Result<crate::bring_up::PplResult> {
        use crate::bring_up::{dtoh_sync_checked, f16_to_f32};
        use crate::gemma4_layer_exec::*;
        use rvllm_loader::gemma4_arch::Gemma4LayerType;

        let arch = &self.arch;
        let hidden = arch.hidden_size as u32;
        let max_hd = arch.max_head_dim() as u32;
        let max_nkvh = arch.max_kv_heads() as u32;
        let max_q_dim = (arch.num_attention_heads * arch.max_head_dim()) as u32;
        let max_kv_dim = (max_nkvh * max_hd) as u32;
        let max_qkv_rows = max_q_dim + 2 * max_kv_dim;
        let inter = arch.intermediate_size as u32;
        let embed_vocab = arch.vocab_size as u32;
        let vocab = self
            .model
            .lm_head_f16
            .shape
            .first()
            .copied()
            .unwrap_or(arch.vocab_size) as u32;
        let stream = self.stream.raw();
        let num_seqs: u32 = 1;
        let pruned_vocab = self.model.pruned_vocab.as_ref();
        // Reclaim every region this call allocates (incl. the multi-GB KV
        // cache) on return, so sliding-window callers don't leak the arena
        // across windows (run_ppl used to OOM by the 3rd window).
        let ppl_arena_ck = self.arena.checkpoint();

        let max_layers: usize = std::env::var("RVLLM_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(arch.num_hidden_layers);
        let n_layers_active = self.model.layers.len().min(max_layers);
        let skip_softcap = std::env::var("RVLLM_NO_SOFTCAP").map_or(false, |v| v == "1");
        // RVLLM_PPL_FULLHEAD=1: score PPL over the FULL 262144 vocab using the
        // tied embed_tokens table as the lm-head, instead of the pruned INT4
        // head (K=16384) + full-vocab scatter. The pruned head cannot score
        // out-of-keepset targets (logit 0 -> NLL ~23) and the scatter is itself
        // buggy, so the pruned-head PPL number is an artifact. The embed table
        // is pre-scaled by sqrt(hidden) at load, so the logits are divided by
        // sqrt(hidden) before the softcap below.
        let fullhead = std::env::var("RVLLM_PPL_FULLHEAD").map_or(false, |v| v == "1");
        if pruned_vocab.is_some() && !fullhead {
            return Err(rvllm_core::RvllmError::config(
                rvllm_core::ConfigError::Inconsistent {
                    reasons: vec![
                        "pruned-vocabulary PPL is disabled because full-vocabulary scoring is required"
                            .into(),
                    ],
                },
                "PPL scoring",
            ));
        }
        if fullhead {
            eprintln!("[ppl] RVLLM_PPL_FULLHEAD=1: full-vocab tied-head PPL (embed_tokens, /sqrt(hidden))");
        }
        if max_layers < arch.num_hidden_layers {
            eprintln!(
                "[ppl] RVLLM_MAX_LAYERS={max_layers} (of {})",
                arch.num_hidden_layers
            );
        }
        if skip_softcap {
            eprintln!("[ppl] RVLLM_NO_SOFTCAP=1: softcap disabled");
        }
        eprintln!("[ppl] attn_scale=1.0 (Gemma4 QK-norm, no query_pre_attn_scalar)");

        let block_size: u32 = std::env::var("RVLLM_BLOCK_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);
        let num_blocks_total: u32 = std::env::var("RVLLM_NUM_BLOCKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let max_blocks_per_seq = (num_blocks_total / num_seqs).max(1);

        let arena = &self.arena;
        let hidden_fp8 = arena.region("hidden_fp8", (num_seqs * hidden) as usize, 16)?;
        let hidden_scale = arena.region("hidden_scale", (num_seqs * 4) as usize, 16)?;
        let qkv_out = arena.region("qkv_out", (num_seqs * max_qkv_rows * 2) as usize, 16)?;
        let q_base = qkv_out.device_ptr();
        let q_normed = arena.region("q_normed", (num_seqs * max_q_dim * 2) as usize, 16)?;
        let k_normed = arena.region("k_normed", (num_seqs * max_kv_dim * 2) as usize, 16)?;
        let v_normed = arena.region("v_normed", (num_seqs * max_kv_dim * 2) as usize, 16)?;
        let q_fp8 = arena.region("q_fp8", (num_seqs * max_q_dim) as usize, 16)?;
        let attn_out = arena.region("attn_out", (num_seqs * max_q_dim * 2) as usize, 16)?;
        let attn_out_fp8 = arena.region("attn_out_fp8", (num_seqs * max_q_dim) as usize, 16)?;
        let attn_out_scale = arena.region("attn_out_scale", (num_seqs * 4) as usize, 16)?;
        let gate_up_out = arena.region("gate_up_out", (num_seqs * 2 * inter * 2) as usize, 16)?;
        let gate_up_fp8 = arena.region("gate_up_fp8", (num_seqs * 2 * inter) as usize, 16)?;
        let gate_up_scale = arena.region("gate_up_scale", (num_seqs * 4) as usize, 16)?;
        let mlp_out_fp8 = arena.region("mlp_out_fp8", (num_seqs * inter) as usize, 16)?;
        let mlp_out_scale = arena.region("mlp_out_scale", (num_seqs * 4) as usize, 16)?;
        let delta_f16 = arena.region("delta_f16_ppl", (num_seqs * hidden * 2) as usize, 16)?;
        let ple_dim = arch.hidden_size_per_layer_input as u32;
        let ple_total_dim = (arch.num_hidden_layers as u32) * ple_dim;
        let gemm_f32_max_n = std::cmp::max(std::cmp::max(max_qkv_rows, 2 * inter), ple_total_dim);
        let gemm_f32_tmp = arena.region(
            "gemm_f32_tmp_ppl",
            (num_seqs * gemm_f32_max_n * 4) as usize,
            16,
        )?;
        let ple_elems = (num_seqs * ple_total_dim).max(1);
        let ple_inputs = arena.region("ple_inputs_ppl", (ple_elems * 2) as usize, 16)?;
        let ple_projection_f16 =
            arena.region("ple_projection_ppl", (ple_elems * 2) as usize, 16)?;
        let ple_gate_elems = (num_seqs * ple_dim).max(1);
        let ple_gate = arena.region("ple_gate_ppl", (ple_gate_elems * 2) as usize, 16)?;

        let f16_only = std::env::var("RVLLM_F16_ONLY").map_or(false, |v| v == "1");
        let use_f16_kv = f16_only || std::env::var("RVLLM_F16_KV").map_or(false, |v| v != "0");
        let sync_layers = std::env::var("RVLLM_SYNC_LAYERS").ok().as_deref() == Some("1");
        let kv_bytes_per_elem: u32 = if use_f16_kv { 2 } else { 1 };

        // Per-layer KV budget: sliding layers cap at sliding_window/block_size blocks,
        // global layers use full num_blocks_total. Saves ~5x KV memory for long context.
        // Sliding-layer slots must remain inside the ring allocation.
        // at every t the rope writes; the old cap sliding_blocks = sliding_window/block_size = 32
        // (= 1024 slots for Gemma 4) broke at prompt_len > sliding_window because slot_mapping
        // is linear 0..prompt_len-1 and index 1024+ ran off the end of the sliding KV region.
        // Proper fix is a per-sliding-layer ring buffer (slot = t mod sliding_window) but that
        // needs rope + attention kernel cooperation. For now give sliding layers the full pool —
        // ~10 GiB extra at num_blocks_total=1024, fits in the 50 GiB arena with Gemma 4 31B fp8.
        let sliding_blocks = num_blocks_total;

        let mut kv_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_total_bytes: u64 = 0;
        let mut kv_scale_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_scale_total_bytes: u64 = 0;
        for l in 0..arch.num_hidden_layers {
            kv_layer_offsets.push(kv_total_bytes);
            kv_scale_layer_offsets.push(kv_scale_total_bytes);
            let is_global =
                arch.layer_types[l] == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
            let layer_blocks = if is_global {
                num_blocks_total
            } else {
                sliding_blocks
            };
            let nkvh = arch.num_kv_heads_for_layer(l) as u32;
            let hd = arch.head_dim_for_layer(l) as u32;
            let layer_elems =
                2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
            kv_total_bytes += layer_elems * kv_bytes_per_elem as u64;
            let layer_scale_slots = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64;
            kv_scale_total_bytes += layer_scale_slots * 4;
        }
        let first_kv_shared = arch
            .num_hidden_layers
            .saturating_sub(arch.num_kv_shared_layers);
        let kv_share_targets: Vec<Option<usize>> = (0..arch.num_hidden_layers)
            .map(|layer_idx| {
                if arch.num_kv_shared_layers == 0 || layer_idx < first_kv_shared {
                    return None;
                }
                let lt = arch.layer_types[layer_idx];
                arch.layer_types[..first_kv_shared]
                    .iter()
                    .rposition(|&prev| prev == lt)
            })
            .collect();
        let kv_share_sources: Vec<bool> = (0..arch.num_hidden_layers)
            .map(|layer_idx| {
                kv_share_targets
                    .iter()
                    .any(|&target| target == Some(layer_idx))
            })
            .collect();
        eprintln!(
            "[ppl] KV cache: {:.1} MB (sliding={} blocks, global={} blocks, {} bytes/elem)",
            kv_total_bytes as f64 / 1e6,
            sliding_blocks,
            num_blocks_total,
            kv_bytes_per_elem
        );

        let kv_cache = arena.region("kv_cache", kv_total_bytes as usize, 256)?;
        cudarc::driver::sys::cuMemsetD8_v2(kv_cache.device_ptr(), 0, kv_total_bytes as usize);
        let kv_scale_cache = arena.region("kv_scale_cache", kv_scale_total_bytes as usize, 16)?;
        cudarc::driver::sys::cuMemsetD8_v2(
            kv_scale_cache.device_ptr(),
            0,
            kv_scale_total_bytes as usize,
        );
        let q_scale_scratch_bytes = (num_seqs as u64) * (arch.num_attention_heads as u64) * 4;
        let q_scale_scratch =
            arena.region("q_scale_scratch", q_scale_scratch_bytes as usize, 16)?;
        cudarc::driver::sys::cuMemsetD8_v2(
            q_scale_scratch.device_ptr(),
            0,
            q_scale_scratch_bytes as usize,
        );
        // See run_bench: RVLLM_PER_TOKEN_Q_SCALE=0 opts out.
        let q_scale_cache_ptr: u64 =
            if std::env::var("RVLLM_PER_TOKEN_Q_SCALE").ok().as_deref() == Some("0") {
                0
            } else {
                q_scale_scratch.device_ptr()
            };

        let q_scale_region = arena.region("q_scale", 4, 4)?;
        let kv_scale_region = arena.region("kv_scale", 4, 4)?;
        {
            let q_s: f32 = std::env::var("RVLLM_Q_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_Q_SCALE);
            let kv_s: f32 = std::env::var("RVLLM_KV_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_KV_SCALE);
            q_scale_region.copy_from_host(&q_s.to_le_bytes())?;
            kv_scale_region.copy_from_host(&kv_s.to_le_bytes())?;
        }

        let fa3_ws_bytes = self.attention_workspace_bytes(
            max_layers,
            num_seqs,
            num_seqs,
            1,
            block_size,
            num_blocks_total,
            sliding_blocks,
            max_blocks_per_seq,
            arch.sliding_window_size as u32,
            use_f16_kv,
            false,
        )?;
        let fa3_ws = arena.region("fa3_ws", fa3_ws_bytes, 256)?;
        let cutlass_ws_bytes: usize = 16 * 1024 * 1024;
        let cutlass_ws = arena.region("cutlass_ws_ppl", cutlass_ws_bytes, 256)?;

        let positions = arena.region("positions", (num_seqs * 4) as usize, 16)?;
        let slot_mapping = arena.region("slot_mapping", n_layers_active.max(1) * 4, 16)?;
        let context_lens = arena.region("context_lens", n_layers_active.max(1) * 4, 16)?;
        let block_tables = arena.region(
            "block_tables",
            (num_seqs * max_blocks_per_seq * 4) as usize,
            16,
        )?;
        {
            let mut bt: Vec<i32> = Vec::with_capacity(max_blocks_per_seq as usize);
            for b in 0..max_blocks_per_seq as i32 {
                bt.push(b);
            }
            block_tables.copy_from_host(bytemuck_cast_i32(&bt))?;
        }

        let residual = arena.region("residual", (num_seqs * hidden * 2) as usize, 16)?;
        let logits = arena.region("logits_ppl", (num_seqs * vocab * 2) as usize, 16)?;
        let logits_f32 = arena.region("logits_f32_ppl", (num_seqs * vocab * 4) as usize, 16)?;
        let token_ids_region = arena.region("token_ids_ppl", (num_seqs * 4) as usize, 16)?;
        let residual_ptr = residual.device_ptr();
        let kernels = self.layer_kernels();

        let e4b_rt = self.e4b.as_ref();
        // INT4 decoder runtime + w4a8 lib (Some only on the E4B+INT4 path).
        let int4_rt = self.e4b_int4.as_ref();
        let w4a8_lib = self.w4a8.as_ref();
        let step_counter = std::cell::Cell::new(0u32);
        let one_step = || -> Result<()> {
            for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                if layer_idx >= max_layers {
                    break;
                }
                let lt = arch.layer_types[layer_idx];
                let hd = arch.head_dim_for_layer(layer_idx) as u32;
                let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
                let q_dim = (arch.num_attention_heads as u32) * hd;
                let kv_dim = nkvh * hd;
                let layer_blocks = if lt == Gemma4LayerType::GlobalAttention {
                    num_blocks_total
                } else {
                    sliding_blocks
                };

                // E4B KV-share source + runtime sliding-window override
                // (see run_bench). `None` for the 31B path.
                let e4b_layer = e4b_rt.map(|rt| &rt.layers[layer_idx]);
                let kv_share_src = e4b_layer.and_then(|l| l.kv_share_src);
                let layer_sliding_window = match e4b_rt {
                    Some(rt) => rt.sliding_window,
                    None => arch.sliding_window_size as u32,
                };

                let dims = Gemma4LayerDims {
                    num_tokens: num_seqs,
                    hidden,
                    num_heads: arch.num_attention_heads as u32,
                    num_kv_heads: nkvh,
                    head_dim: hd,
                    rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                    rope_table_rows: arch.max_position_embeddings as u32,
                    intermediate: inter,
                    block_size,
                    max_blocks_per_seq: layer_blocks,
                    num_blocks_total: layer_blocks,
                    attn_scale: 1.0,
                    rms_eps: arch.rms_norm_eps,
                    layer_type: lt,
                    sliding_window: layer_sliding_window,
                    f16_kv: use_f16_kv,
                    num_hidden_layers: arch.num_hidden_layers as u32,
                    layer_idx: layer_idx as u32,
                    ple_dim,
                    kv_shared: kv_share_src.is_some(),
                };

                // Row-major [num_tokens, q_dim+2*kv_dim]: k_out / v_out
                // point at row 0's K / V sub-slice. The rmsnorm kernel
                // applies `src_row_stride` to reach later tokens.
                let k_out = q_base + (q_dim as u64) * 2;
                let v_out = k_out + (kv_dim as u64) * 2;
                let is_global = lt == Gemma4LayerType::GlobalAttention;
                let layer_blocks = if is_global {
                    num_blocks_total
                } else {
                    sliding_blocks
                };
                let layer_kv_elems =
                    2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                let kv_layer_bytes = layer_kv_elems * kv_bytes_per_elem as u64;
                // KV-share: shared tail layer reads the source layer's KV.
                // Main's typed `kv_share_src` supersedes the branch's
                // `cache_layer_idx`; both resolve to the share-source layer.
                let kv_src_layer = kv_share_src.unwrap_or(layer_idx);
                let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[kv_src_layer];
                let layer_kv_scale_base =
                    kv_scale_cache.device_ptr() + kv_scale_layer_offsets[kv_src_layer];
                let layer_kv_scale_slots_half =
                    (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);
                let (cos, sin) = match lt {
                    Gemma4LayerType::SlidingAttention => (
                        self.model.rope_cos_sliding.offset_bytes,
                        self.model.rope_sin_sliding.offset_bytes,
                    ),
                    Gemma4LayerType::GlobalAttention => (
                        self.model.rope_cos_global.offset_bytes,
                        self.model.rope_sin_global.offset_bytes,
                    ),
                };

                let w = Gemma4LayerWeightPtrs {
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

                let scratch = Gemma4LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(),
                    hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base,
                    k_out,
                    v_out,
                    q_normed: q_normed.device_ptr(),
                    k_normed: k_normed.device_ptr(),
                    v_normed: v_normed.device_ptr(),
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + kv_layer_bytes / 2,
                    k_scale_cache: layer_kv_scale_base,
                    v_scale_cache: layer_kv_scale_base + layer_kv_scale_slots_half * 4,
                    q_scale_cache: q_scale_cache_ptr,
                    q_scale_ptr: q_scale_region.device_ptr(),
                    kv_scale_ptr: kv_scale_region.device_ptr(),
                    attn_out: attn_out.device_ptr(),
                    attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(),
                    delta_f16: delta_f16.device_ptr(),
                    gate_up_out: gate_up_out.device_ptr(),
                    gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(),
                    mlp_out_scale: mlp_out_scale.device_ptr(),
                    gemm_f32_tmp: gemm_f32_tmp.device_ptr(),
                    cutlass_workspace: cutlass_ws.device_ptr(),
                    cutlass_workspace_bytes: cutlass_ws_bytes,
                    fa3_workspace: fa3_ws.device_ptr(),
                    fa3_workspace_bytes: fa3_ws_bytes,
                    ple_inputs: ple_inputs.device_ptr(),
                    ple_gate: ple_gate.device_ptr(),
                };

                let meta = Gemma4MetadataPtrs {
                    positions: positions.device_ptr(),
                    slot_mapping: slot_mapping.device_ptr() + (layer_idx as u64) * 4,
                    cos,
                    sin,
                    block_tables: block_tables.device_ptr(),
                    context_lens: context_lens.device_ptr() + (layer_idx as u64) * 4,
                };

                // E4B-aware forward dispatch (main): PLE-fused layer forward
                // when an E4B runtime + per-layer PLE descriptor are present,
                // else the plain forward. No-op PLE on the 31B path.
                match (e4b_rt, e4b_layer) {
                    (Some(rt), Some(pl)) => {
                        let h_ple = rt.h_ple;
                        let pli_base =
                            rt.per_layer_inputs + (layer_idx as u64) * (h_ple as u64) * 2; // f16
                        let ple = crate::gemma4_layer_exec::Gemma4PleLayer {
                            gate_w: pl.gate_w,
                            proj_w: pl.proj_w,
                            per_layer_input: pli_base,
                            post_norm_gamma: pl.post_norm,
                            h_ple,
                            // [T, L, h_ple] -> per-token stride = num_layers * h_ple.
                            pli_stride: rt.layers.len() as u32 * h_ple,
                        };
                        let int4 = match (int4_rt, w4a8_lib) {
                            (Some(rt), Some(lib)) => Some(crate::gemma4_int4::Int4LayerExec {
                                w4a8: lib,
                                layer: &rt.layers[layer_idx],
                                workspace: rt.workspace,
                                workspace_bytes: rt.workspace_bytes,
                            }),
                            _ => None,
                        };
                        crate::gemma4_layer_exec::gemma4_e4b_layer_forward(
                            dims,
                            &kernels,
                            &w,
                            &scratch,
                            &meta,
                            &self.cublaslt,
                            &self.cutlass,
                            &self.sliding_attention,
                            &self.global_attention,
                            residual_ptr,
                            stream,
                            crate::gemma4_layer_exec::Gemma4Phase::Decode,
                            Some(ple),
                            int4,
                        )?;
                    }
                    _ => {
                        gemma4_forward(
                            dims,
                            &kernels,
                            &w,
                            &scratch,
                            &meta,
                            &self.cublaslt,
                            &self.cutlass,
                            &self.sliding_attention,
                            &self.global_attention,
                            residual_ptr,
                            stream,
                        )?;
                    }
                }
                // PPL per-layer sync (branch): a KV-share SOURCE layer must
                // finish before its sharing tail layer reads the cache, so
                // fence after source layers (and when RVLLM_SYNC_LAYERS=1).
                if sync_layers || kv_share_sources[layer_idx] {
                    self.stream.fence().map_err(|e| {
                        eprintln!("[ppl sync] CUDA error after layer {layer_idx}");
                        e
                    })?;
                }

                if step_counter.get() == 0 && layer_idx == 0 {
                    cudarc::driver::sys::cuStreamSynchronize(stream as _);
                    let mut s = [0u16; 4];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, residual_ptr, 8);
                    let v: Vec<f32> = s.iter().map(|&x| f16_to_f32(x)).collect();
                    let mut amax = 0f32;
                    let n = hidden as usize;
                    let mut all = vec![0u16; n];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        all.as_mut_ptr() as *mut _,
                        residual_ptr,
                        (n * 2) as _,
                    );
                    for &b in &all {
                        let f = f16_to_f32(b).abs();
                        if f > amax && !f.is_nan() {
                            amax = f;
                        }
                    }
                    eprintln!("  [ppl L0] residual first4={:.6?} amax={:.6}", v, amax);
                    // Check layer_scalar value
                    let mut sc = [0u16; 1];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        sc.as_mut_ptr() as *mut _,
                        layer.layer_scalar.offset_bytes,
                        2,
                    );
                    eprintln!("  [ppl L0] layer_scalar={:.6}", f16_to_f32(sc[0]));
                    // Check norm gamma amax
                    let mut ng = vec![0u16; n];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        ng.as_mut_ptr() as *mut _,
                        layer.input_layernorm.offset_bytes,
                        (n * 2) as _,
                    );
                    let gamma_amax = ng.iter().map(|&b| f16_to_f32(b).abs()).fold(0f32, f32::max);
                    eprintln!("  [ppl L0] input_norm_gamma amax={:.6}", gamma_amax);
                }
                if step_counter.get() == 0 && std::env::var("RVLLM_DBG_RES").is_ok() {
                    cudarc::driver::sys::cuStreamSynchronize(stream as _);
                    let n = hidden as usize;
                    let mut all = vec![0u16; n];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        all.as_mut_ptr() as *mut _,
                        residual_ptr,
                        (n * 2) as _,
                    );
                    let mut amax = 0f32;
                    let mut nan = false;
                    for &b in &all {
                        let f = f16_to_f32(b);
                        if f.is_nan() {
                            nan = true;
                        } else if f.abs() > amax {
                            amax = f.abs();
                        }
                    }
                    eprintln!("  [ppl res L{:02}] amax={:.3} nan={}", layer_idx, amax, nan);
                }
            }

            // LM head: final norm (f16 in-place) + f16 GEMM -> f32 logits
            let dbg_lmhead = step_counter.get() == 0 && std::env::var("RVLLM_DBG_LAYER").is_ok();

            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: num_seqs,
                hidden,
                eps: arch.rms_norm_eps,
            }
            .launch(
                kernels.fused_rmsnorm,
                residual_ptr,
                self.model.final_norm.offset_bytes,
                stream,
            )?;
            if dbg_lmhead {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let mut s = [0u16; 4];
                cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, residual_ptr, 8);
                let v: Vec<f32> = s.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
                eprintln!("  [lm_head] after rmsnorm_f16: first4={:.4?}", v);
            }
            if !fullhead {
                if let Some(int4) = int4_rt {
                    // INT4 PPL tail: pruned-head GEMV -> kept-row bf16 scores ->
                    // scatter into the full-vocab f32 logits (`-inf` at non-kept
                    // columns). `residual_ptr` is already f16 final-normed. The
                    // host softcap + CE below then run on `logits_f32` unchanged.
                    // `logits` (f16, num_seqs*vocab*2 bytes) is reused as the f32
                    // scores scratch (num_seqs*k_rows*4 << that since k_rows<<vocab).
                    let (fn_gemv, fn_scatter) = match (
                        self.fused.fn_lmhead_int4_gemv.as_ref(),
                        self.fused.fn_lmhead_scatter_fullvocab.as_ref(),
                    ) {
                        (Some(g), Some(s)) => (g, s),
                        _ => {
                            return Err(rvllm_core::RvllmError::config(
                                rvllm_core::ConfigError::Inconsistent {
                                    reasons: vec!["RVLLM_INT4 PPL run but the lmhead int4 GEMV / \
                                     full-vocab scatter kernels are absent — cannot \
                                     produce full-vocab logits for PPL (fail-closed)"
                                        .into()],
                                },
                                "RVLLM_INT4",
                            ));
                        }
                    };
                    crate::gemma4_int4::lmhead_int4_scores(
                        fn_gemv,
                        &int4.lm_head,
                        residual_ptr,
                        logits.device_ptr(),
                        num_seqs,
                        hidden,
                        stream,
                    )?;
                    crate::gemma4_int4::lmhead_scatter_full_vocab(
                        fn_scatter,
                        &int4.lm_head,
                        logits.device_ptr(),
                        logits_f32.device_ptr(),
                        num_seqs,
                        stream,
                    )?;
                    step_counter.set(step_counter.get() + 1);
                    return Ok(());
                }
            }
            // FULLHEAD: use the full-vocab tied embed_tokens table as the head
            // (the only dense full-vocab head present; lm_head is INT4-pruned).
            let head_ptr = if fullhead {
                self.model.embedding.offset_bytes
            } else {
                self.model.lm_head_f16.offset_bytes
            };
            self.cublaslt.f16_gemm_f32(
                residual_ptr,
                head_ptr,
                logits_f32.device_ptr(),
                num_seqs as i32,
                vocab as i32,
                hidden as i32,
                stream,
            )?;
            if dbg_lmhead {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let total = (vocab as usize) * (num_seqs as usize);
                let mut buf = vec![0.0f32; total];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut _,
                    logits_f32.device_ptr(),
                    (total * 4) as _,
                );
                let amax = buf.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                eprintln!(
                    "  [lm_head] raw_f32_logits first8={:.4?} amax={:.6e} (n={})",
                    &buf[..8.min(total)],
                    amax,
                    total
                );
            }
            rvllm_fused::gemma4_launcher::Bf16ToF16SatLaunch {
                n: num_seqs * vocab,
            }
            .launch(
                kernels.f32_to_f16_sat,
                logits.device_ptr(),
                logits_f32.device_ptr(),
                stream,
            )?;
            if dbg_lmhead {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let mut s = [0u16; 4];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    s.as_mut_ptr() as *mut _,
                    logits.device_ptr(),
                    8,
                );
                let v: Vec<f32> = s.iter().map(|&x| f16_to_f32(x)).collect();
                eprintln!(
                    "  [lm_head] after f32_to_f16_sat: logits_f16 first4={:.4?}",
                    v
                );
            }
            if !skip_softcap {
                logit_softcap(
                    &self.fused.fn_softcap,
                    logits.device_ptr(),
                    num_seqs,
                    vocab,
                    arch.logit_softcap,
                    stream,
                )?;
            }
            if sync_layers {
                self.stream.fence().map_err(|e| {
                    eprintln!("[ppl sync] CUDA error after lm_head");
                    e
                })?;
            }
            if dbg_lmhead {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let mut s = [0u16; 4];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    s.as_mut_ptr() as *mut _,
                    logits.device_ptr(),
                    8,
                );
                let v: Vec<f32> = s.iter().map(|&x| f16_to_f32(x)).collect();
                eprintln!("  [lm_head] after softcap: logits_f16 first4={:.4?}", v);
            }
            step_counter.set(step_counter.get() + 1);
            Ok(())
        };

        let mut ppl_slot_host = vec![0i32; n_layers_active.max(1)];
        let mut ppl_ctx_host = vec![0i32; n_layers_active.max(1)];
        let stage_ppl_step =
            |tok_id: u32, step: usize, slots: &mut [i32], contexts: &mut [i32]| -> Result<()> {
                fill_ppl_layer_metadata(
                    &arch.layer_types,
                    &kv_share_targets,
                    step,
                    arch.sliding_window_size,
                    &mut slots[..n_layers_active],
                    &mut contexts[..n_layers_active],
                );
                let token = [tok_id as i32];
                let position = [step as i32];
                self.stream.fence()?;
                token_ids_region.copy_from_host(bytemuck_cast_i32(&token))?;
                positions.copy_from_host(bytemuck_cast_i32(&position))?;
                slot_mapping.copy_from_host(bytemuck_cast_i32(slots))?;
                context_lens.copy_from_host(bytemuck_cast_i32(contexts))?;
                Ok(())
            };

        let logits_row_elems = vocab as usize;
        let logits_row_bytes_f32 = logits_row_elems * 4;
        let mut logits_host_f32: Vec<f32> = vec![0.0f32; logits_row_elems];
        let mut total_nll: f64 = 0.0;
        let mut n_evaluated: usize = 0;
        let mut token_logprobs: Vec<Option<f64>> = vec![None; token_ids.len()];

        // Build a graph-capturable forward: embed + all layers + lm_head.
        // No debug probes (they break capture).
        let ppl_forward = || -> Result<()> {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1,
                hidden,
                vocab: embed_vocab,
            }
            .launch(
                fn_embed,
                residual_ptr,
                self.model.embedding.offset_bytes,
                token_ids_region.device_ptr(),
                stream,
            )?;
            // E4B per-step PLE combine, once before the layer loop.
            // No-op on the 31B path (`self.e4b == None`). The combine writes
            // `e4b.per_layer_inputs`, which each layer's PLE gate reads in
            // `gemma4_e4b_layer_forward`. Runs inside the captured region so
            // the per-layer inputs are refreshed every step. Main's
            // `run_ple_combine` encapsulates the PLE buffers that the branch
            // used to thread through `build_ple_inputs`.
            self.run_ple_combine(token_ids_region.device_ptr(), residual_ptr, 1, stream)?;
            one_step()
        };

        let use_graph = std::env::var("RVLLM_NO_GRAPH").ok().as_deref() != Some("1")
            && !sync_layers
            && !kv_share_sources.iter().any(|&source| source);
        let ppl_graph = if use_graph {
            // Dry run to populate KV cache slot 0
            stage_ppl_step(token_ids[0], 0, &mut ppl_slot_host, &mut ppl_ctx_host)?;
            ppl_forward()?;
            self.stream.fence()?;

            let g = rvllm_graph::CapturedGraph::capture(
                &self.ctx,
                num_seqs,
                max_blocks_per_seq,
                rvllm_metadata::MetadataLayout::compute(num_seqs, max_blocks_per_seq)?.hash(),
                stream,
                || ppl_forward(),
            )?;
            self.stream.fence()?;
            Some(g)
        } else {
            None
        };

        for (t, &tok_id) in token_ids.iter().enumerate() {
            stage_ppl_step(tok_id, t, &mut ppl_slot_host, &mut ppl_ctx_host)?;

            if let Some(ref graph) = ppl_graph {
                graph.replay(stream)?;
            } else {
                ppl_forward()?;
            }

            // Tokens before `score_from` are forwarded to build KV context
            // but not scored (sliding-window left context). The final token
            // has no target. Both fall through to the bare fence below.
            if t + 1 < token_ids.len() && t >= score_from {
                self.stream.fence()?;
                dtoh_sync_checked(
                    logits_f32.device_ptr(),
                    logits_host_f32.as_mut_ptr().cast(),
                    logits_row_bytes_f32,
                    stream,
                )?;

                // FULLHEAD: the embed_tokens table is stored pre-multiplied by
                // sqrt(hidden) (for the input-embedding lookup), so the tied-head
                // logits are sqrt(hidden)x too large; undo it before the softcap.
                if fullhead {
                    let inv = 1.0f32 / (hidden as f32).sqrt();
                    for x in logits_host_f32.iter_mut() {
                        *x *= inv;
                    }
                }

                let cap = arch.logit_softcap;
                if !skip_softcap && cap > 0.0 {
                    for x in logits_host_f32.iter_mut() {
                        *x = cap * (*x / cap).tanh();
                    }
                }

                let target_full = token_ids[t + 1] as usize;
                let target = if let Some(pv) = pruned_vocab {
                    pv.full_to_keep
                        .get(target_full)
                        .copied()
                        .filter(|&row| row >= 0)
                        .map(|row| row as usize)
                } else {
                    Some(target_full)
                };
                if t == 0 {
                    let first5: Vec<f32> = logits_host_f32[..5].to_vec();
                    let max_val = logits_host_f32
                        .iter()
                        .copied()
                        .filter(|v| !v.is_nan())
                        .fold(f32::MIN, f32::max);
                    let min_val = logits_host_f32
                        .iter()
                        .copied()
                        .filter(|v| !v.is_nan())
                        .fold(f32::MAX, f32::min);
                    eprintln!(
                        "  [ppl] logits(f32+softcap): first5={:?} min={:.1} max={:.1}",
                        first5, min_val, max_val
                    );
                }
                // FULLHEAD scores the true full-vocab token over the dense head;
                // the pruned path keeps the keepset mapping (None = out-of-keepset).
                let nll = if fullhead {
                    crate::bring_up::compute_nll_f32(&logits_host_f32, target_full)
                } else if let Some(target) = target {
                    crate::bring_up::compute_nll_f32(&logits_host_f32, target)
                } else {
                    eprintln!(
                        "  [ppl] target token {} missing from pruned vocabulary",
                        target_full
                    );
                    f64::INFINITY
                };
                if std::env::var("RVLLM_DBG_PRED").is_ok() && n_evaluated < 16 {
                    let (mut amax_i, mut amax_v) = (0usize, f32::MIN);
                    let mut rank = 0u32;
                    let tgt_logit = logits_host_f32
                        .get(target_full as usize)
                        .copied()
                        .unwrap_or(f32::NAN);
                    for (i, &v) in logits_host_f32.iter().enumerate() {
                        if v.is_nan() {
                            continue;
                        }
                        if v > amax_v {
                            amax_v = v;
                            amax_i = i;
                        }
                        if v > tgt_logit {
                            rank += 1;
                        }
                    }
                    eprintln!(
                        "  [pred] target={target_full} tgt_logit={tgt_logit:.2} nll={nll:.3} argmax={amax_i} argmax_logit={amax_v:.2} target_rank={rank}"
                    );
                }
                total_nll += nll;
                n_evaluated += 1;
                token_logprobs[t + 1] = Some(-nll);

                if (t + 1) % 32 == 0 || t + 1 == token_ids.len() - 1 {
                    let running_ppl = (total_nll / n_evaluated as f64).exp();
                    eprintln!(
                        "  step {}/{}: running_ppl={:.4}",
                        t + 1,
                        token_ids.len(),
                        running_ppl
                    );
                }
            } else {
                self.stream.fence()?;
            }
        }

        let ppl = if n_evaluated > 0 {
            (total_nll / n_evaluated as f64).exp()
        } else {
            0.0
        };
        // Drop the captured graph (it holds arena regions) before rewinding
        // the bump pointer, per HbmArena::restore's safety contract. Nothing
        // below touches the reclaimed regions.
        drop(ppl_graph);
        unsafe { self.arena.restore(ppl_arena_ck)? };
        Ok(crate::bring_up::PplResult {
            ppl,
            total_nll,
            n_evaluated,
            token_logprobs,
        })
    }

    #[cfg(feature = "cuda")]
    pub unsafe fn run_generate(
        &self,
        fn_embed: &rvllm_kernels::KernelFn,
        fn_argmax: &rvllm_kernels::KernelFn,
        prompt_ids: &[u32],
        max_new: usize,
        eos_ids: &[u32],
        image_embeds: &[(usize, Vec<f32>)],
    ) -> Result<Vec<u32>> {
        unsafe {
            self.run_generate_sampled(
                fn_embed,
                fn_argmax,
                prompt_ids,
                max_new,
                eos_ids,
                image_embeds,
                SamplingParams::greedy(),
            )
        }
    }

    /// `run_generate` with per-request sampling params. Greedy params
    /// (`temperature == 0`) take exactly the `run_generate` path — same
    /// kernels, same graph, bit-identical token stream. Non-greedy params
    /// switch the decode tail to the sampled path (top-K' selection kernel
    /// + seeded host draw); see the sampled decode arm below.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_generate_sampled(
        &self,
        fn_embed: &rvllm_kernels::KernelFn,
        fn_argmax: &rvllm_kernels::KernelFn,
        prompt_ids: &[u32],
        max_new: usize,
        eos_ids: &[u32],
        image_embeds: &[(usize, Vec<f32>)],
        sampling: SamplingParams,
    ) -> Result<Vec<u32>> {
        let arena_ck = self.arena.checkpoint();
        let result = unsafe {
            self.run_generate_inner(
                fn_embed,
                fn_argmax,
                prompt_ids,
                max_new,
                eos_ids,
                image_embeds,
                sampling,
            )
        };
        let fence_result = self.stream.fence();
        unsafe { self.arena.restore(arena_ck)? };
        fence_result?;
        result
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn run_generate_inner(
        &self,
        fn_embed: &rvllm_kernels::KernelFn,
        fn_argmax: &rvllm_kernels::KernelFn,
        prompt_ids: &[u32],
        max_new: usize,
        eos_ids: &[u32],
        image_embeds: &[(usize, Vec<f32>)],
        sampling: SamplingParams,
    ) -> Result<Vec<u32>> {
        let arch = &self.arch;
        let hidden = arch.hidden_size as u32;
        let embed_vocab = arch.vocab_size as u32;
        let vocab = self
            .model
            .lm_head_f16
            .shape
            .first()
            .copied()
            .unwrap_or(arch.vocab_size) as u32;
        let stream = self.stream.raw();
        let pruned_vocab = self.model.pruned_vocab.as_ref();
        let arena = &self.arena;
        let block_size: u32 = 32;
        let num_blocks_total: u32 = std::env::var("RVLLM_NUM_BLOCKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let generation_capacity = validate_generation_capacity(
            prompt_ids.len(),
            max_new,
            num_blocks_total as usize,
            block_size as usize,
            arch.max_position_embeddings,
        )
        .map_err(|reason| rvllm_core::RvllmError::Sampling {
            err: rvllm_core::SamplingError::InvalidParams { reason },
            ctx: rvllm_core::SampleCtx {
                op: "gemma4_generation_capacity",
                stream,
            },
        })?;
        let prompt_len =
            u32::try_from(prompt_ids.len()).map_err(|_| rvllm_core::RvllmError::Sampling {
                err: rvllm_core::SamplingError::InvalidParams {
                    reason: "prompt length does not fit the CUDA ABI".into(),
                },
                ctx: rvllm_core::SampleCtx {
                    op: "gemma4_generation_capacity",
                    stream,
                },
            })?;
        let sampled_vocab = if pruned_vocab.is_some() {
            vocab
        } else {
            embed_vocab
        };
        let map_sampled_token = |raw_id: i64, op: &'static str| -> Result<u32> {
            let row_id = validate_sampled_token(raw_id, sampled_vocab, op, stream)?;
            let token_id = match pruned_vocab {
                Some(pv) => pv.keep_ids.get(row_id as usize).copied().ok_or_else(|| {
                    rvllm_core::RvllmError::Sampling {
                        err: rvllm_core::SamplingError::InvalidParams {
                            reason: format!(
                                "sampled local row {row_id} has no pruned-vocabulary mapping"
                            ),
                        },
                        ctx: rvllm_core::SampleCtx { op, stream },
                    }
                })?,
                None => row_id,
            };
            validate_sampled_token(i64::from(token_id), embed_vocab, op, stream)
        };
        let keep_ids_region = if let Some(pv) = pruned_vocab {
            let keep_ids_i32: Vec<i32> = pv.keep_ids.iter().map(|&id| id as i32).collect();
            let region = arena.region("gen_pruned_vocab_ids", keep_ids_i32.len() * 4, 16)?;
            region.copy_from_host(bytemuck_cast_i32(&keep_ids_i32))?;
            Some(region)
        } else {
            None
        };
        let keep_ids_ptr = keep_ids_region
            .as_ref()
            .map_or(0, |region| region.device_ptr());
        let keep_ids_len = pruned_vocab.map_or(0, |pv| pv.keep_ids.len() as u32);

        let max_hd = arch.max_head_dim() as u32;
        let max_nkvh = arch.max_kv_heads() as u32;
        let max_q_dim = (arch.num_attention_heads * arch.max_head_dim()) as u32;
        let max_kv_dim = (max_nkvh * max_hd) as u32;
        let max_qkv_rows = max_q_dim + 2 * max_kv_dim;
        let inter = arch.intermediate_size as u32;
        let max_blocks_per_seq = num_blocks_total;

        let max_tokens = prompt_len.max(1);
        let sliding_window_tokens = arch.sliding_window_size as u32;
        let one_shot_prefill_safe = prompt_len <= sliding_window_tokens;
        let skip_decode_requested = std::env::var_os("RVLLM_DIAG_SKIP_DECODE").is_some();
        let batch_prefill_requested = std::env::var_os("RVLLM_BATCH_PREFILL").is_some();
        let skip_decode = skip_decode_requested && one_shot_prefill_safe;
        let use_batch_prefill = batch_prefill_requested && one_shot_prefill_safe;
        let prefill_chunk_tokens: u32 = std::env::var("RVLLM_PREFILL_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(sliding_window_tokens)
            .clamp(1, sliding_window_tokens.max(1));
        let use_chunked_prefill = batch_prefill_requested
            && !one_shot_prefill_safe
            && std::env::var_os("RVLLM_CHUNKED_PREFILL").is_some();
        if skip_decode_requested && !one_shot_prefill_safe {
            eprintln!(
                "[prefill] diagnostic one-shot prefill disabled: prompt_len={} sliding_window={}",
                prompt_len, arch.sliding_window_size
            );
        } else if batch_prefill_requested && !one_shot_prefill_safe && !use_chunked_prefill {
            eprintln!(
                "[prefill] one-shot batch prefill disabled: prompt_len={} sliding_window={}",
                prompt_len, arch.sliding_window_size
            );
        } else if use_chunked_prefill {
            eprintln!(
                "[prefill] chunked batch prefill: prompt_len={} chunk={}",
                prompt_len, prefill_chunk_tokens
            );
        }
        let use_fast_prefill = use_batch_prefill || use_chunked_prefill;
        let diag_compare_enabled = std::env::var_os("RVLLM_DIAG_COMPARE").is_some() && !skip_decode;

        // === Speculative decode (RVLLM_SPEC_DECODE=1) ===================
        // Greedy n-gram prompt-lookup drafting + batched verify. Each
        // decode step runs ONE forward over [last_token, draft_0..] at
        // M = K+1 (weights read once for up to K+1 tokens — the lever at
        // batch=1 where decode is weight-bandwidth-bound), argmaxes every
        // position, and accepts the longest draft prefix matching the
        // model's own argmaxes plus one bonus token. Greedy acceptance is
        // lossless: the emitted stream is exactly what step-by-step
        // greedy decode under the same forward would produce.
        //
        // The verify chunk is a continuation prefill chunk, so the whole
        // request runs FP8 KV (no F16 prefill kernel) — apples-to-apples
        // baseline is plain decode with RVLLM_F16_KV=0.
        //   RVLLM_SPEC_K     draft budget per step (default 4, max 15)
        //   RVLLM_SPEC_NGRAM max n-gram key length for the lookup (def 3)
        //
        // === Sampling tail (temperature / top-k / top-p / seed) =========
        // Greedy (`sampling.is_greedy()`, i.e. temperature == 0) keeps the
        // argmax tail bit-identical to before sampling existed. Non-greedy
        // takes the sampled decode arm below and forces spec decode off —
        // greedy acceptance is only lossless against an argmax stream.
        // RVLLM_SAMPLE_T / RVLLM_SAMPLE_TOPK / RVLLM_SAMPLE_TOPP /
        // RVLLM_SAMPLE_SEED override the caller's params (bench/debug
        // escape hatch; serve passes per-request params instead).
        let sampling = resolve_sampling_env(sampling);
        let sampled_mode = !sampling.is_greedy();
        let spec_decode = std::env::var("RVLLM_SPEC_DECODE").ok().as_deref() == Some("1")
            && !skip_decode
            && !use_fast_prefill
            && !diag_compare_enabled
            && !sampled_mode;
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
        // Max verify-chunk tokens; 1 when spec decode is off so every
        // buffer below keeps its existing size.
        let spec_n_max: u32 = if spec_decode { spec_k + 1 } else { 1 };

        let batch_shape_scratch = skip_decode || use_fast_prefill || diag_compare_enabled;
        let scratch_tokens = if use_chunked_prefill {
            prefill_chunk_tokens
        } else if batch_shape_scratch {
            max_tokens
        } else {
            1
        }
        .max(spec_n_max);

        let hidden_fp8 = arena.region("gen_hidden_fp8", (scratch_tokens * hidden) as usize, 16)?;
        let hidden_scale = arena.region("gen_hidden_scale", (scratch_tokens * 4) as usize, 16)?;
        let qkv_out = arena.region("gen_qkv", (scratch_tokens * max_qkv_rows * 2) as usize, 16)?;
        let q_base = qkv_out.device_ptr();
        let q_normed = arena.region(
            "gen_q_normed",
            (scratch_tokens * max_q_dim * 2) as usize,
            16,
        )?;
        let k_normed = arena.region(
            "gen_k_normed",
            (scratch_tokens * max_kv_dim * 2) as usize,
            16,
        )?;
        let v_normed = arena.region(
            "gen_v_normed",
            (scratch_tokens * max_kv_dim * 2) as usize,
            16,
        )?;
        let q_fp8 = arena.region("gen_q_fp8", (scratch_tokens * max_q_dim) as usize, 16)?;
        let attn_out = arena.region(
            "gen_attn_out",
            (scratch_tokens * max_q_dim * 2) as usize,
            16,
        )?;
        let attn_out_fp8 = arena.region(
            "gen_attn_out_fp8",
            (scratch_tokens * max_q_dim) as usize,
            16,
        )?;
        let attn_out_scale =
            arena.region("gen_attn_out_scale", (scratch_tokens * 4) as usize, 16)?;
        let gate_up_out =
            arena.region("gen_gate_up", (scratch_tokens * 2 * inter * 2) as usize, 16)?;
        let gate_up_fp8 =
            arena.region("gen_gate_up_fp8", (scratch_tokens * 2 * inter) as usize, 16)?;
        let gate_up_scale = arena.region("gen_gate_up_scale", (scratch_tokens * 4) as usize, 16)?;
        let mlp_out_fp8 = arena.region("gen_mlp_fp8", (scratch_tokens * inter) as usize, 16)?;
        let mlp_out_scale = arena.region("gen_mlp_scale", (scratch_tokens * 4) as usize, 16)?;
        let delta_f16 = arena.region("gen_delta", (scratch_tokens * hidden * 2) as usize, 16)?;
        let ple_dim = arch.hidden_size_per_layer_input as u32;
        let ple_total_dim = (arch.num_hidden_layers as u32) * ple_dim;
        let gemm_f32_max_n = std::cmp::max(std::cmp::max(max_qkv_rows, 2 * inter), ple_total_dim);
        let gemm_f32_tmp = arena.region(
            "gen_gemm_f32",
            (scratch_tokens * gemm_f32_max_n * 4) as usize,
            16,
        )?;
        let ple_elems = (scratch_tokens * ple_total_dim).max(1);
        let ple_inputs = arena.region("gen_ple_inputs", (ple_elems * 2) as usize, 16)?;
        let ple_projection_f16 =
            arena.region("gen_ple_projection", (ple_elems * 2) as usize, 16)?;
        let ple_gate_elems = (scratch_tokens * ple_dim).max(1);
        let ple_gate = arena.region("gen_ple_gate", (ple_gate_elems * 2) as usize, 16)?;

        let sliding_blocks =
            ((arch.sliding_window_size as u32).saturating_add(block_size - 1) / block_size).max(1);
        let use_f16_kv = std::env::var("RVLLM_F16_KV").map_or(false, |v| v != "0")
            && !use_fast_prefill
            && !spec_decode;
        let kv_bytes_per_elem: u32 = if use_f16_kv { 2 } else { 1 };
        let mut kv_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_total_bytes: u64 = 0;
        // Per-slot-per-head K/V scale cache offsets (f32 per entry).
        // One entry per (slot, kv_head) — factor `head_dim` smaller
        // than the FP8 KV cache region.
        let mut kv_scale_layer_offsets: Vec<u64> = Vec::with_capacity(arch.num_hidden_layers);
        let mut kv_scale_total_bytes: u64 = 0;
        for l in 0..arch.num_hidden_layers {
            kv_layer_offsets.push(kv_total_bytes);
            kv_scale_layer_offsets.push(kv_scale_total_bytes);
            let is_global =
                arch.layer_types[l] == rvllm_loader::gemma4_arch::Gemma4LayerType::GlobalAttention;
            let layer_blocks = if is_global {
                num_blocks_total
            } else {
                sliding_blocks
            };
            let nkvh = arch.num_kv_heads_for_layer(l) as u32;
            let hd = arch.head_dim_for_layer(l) as u32;
            let layer_elems =
                2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
            kv_total_bytes += layer_elems * kv_bytes_per_elem as u64;
            // Scale storage: [2 (K+V)] × num_slots × num_kv_heads × f32.
            let layer_scale_slots = 2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64;
            kv_scale_total_bytes += layer_scale_slots * 4;
        }
        let kv_cache = arena.region("gen_kv", kv_total_bytes as usize, 256)?;
        crate::bring_up::memset_d8_checked(
            kv_cache.device_ptr(),
            0,
            kv_total_bytes as usize,
            stream,
        )?;
        let kv_scale_cache =
            arena.region("gen_kv_scale_cache", kv_scale_total_bytes as usize, 16)?;
        crate::bring_up::memset_d8_checked(
            kv_scale_cache.device_ptr(),
            0,
            kv_scale_total_bytes as usize,
            stream,
        )?;
        let q_scale_scratch_bytes = (scratch_tokens as u64) * (arch.num_attention_heads as u64) * 4;
        let q_scale_scratch =
            arena.region("gen_q_scale_scratch", q_scale_scratch_bytes as usize, 16)?;
        crate::bring_up::memset_d8_checked(
            q_scale_scratch.device_ptr(),
            0,
            q_scale_scratch_bytes as usize,
            stream,
        )?;
        // See run_bench: RVLLM_PER_TOKEN_Q_SCALE=0 opts out.
        let q_scale_cache_ptr: u64 =
            if std::env::var("RVLLM_PER_TOKEN_Q_SCALE").ok().as_deref() == Some("0") {
                0
            } else {
                q_scale_scratch.device_ptr()
            };

        let q_scale_region = arena.region("gen_q_scale", 4, 4)?;
        let kv_scale_region = arena.region("gen_kv_scale", 4, 4)?;
        {
            let q_s: f32 = std::env::var("RVLLM_Q_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_Q_SCALE);
            let kv_s: f32 = std::env::var("RVLLM_KV_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_KV_SCALE);
            q_scale_region.copy_from_host(&q_s.to_le_bytes())?;
            kv_scale_region.copy_from_host(&kv_s.to_le_bytes())?;
        }

        let max_layers = std::env::var("RVLLM_MAX_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(arch.num_hidden_layers);
        let n_layers_active = self.model.layers.len().min(max_layers);
        let include_prefill =
            skip_decode || use_fast_prefill || diag_compare_enabled || spec_decode;
        let fa3_ws_bytes = self.attention_workspace_bytes(
            n_layers_active,
            scratch_tokens,
            1,
            scratch_tokens,
            block_size,
            num_blocks_total,
            sliding_blocks,
            max_blocks_per_seq,
            arch.sliding_window_size as u32,
            use_f16_kv,
            include_prefill,
        )?;
        let fa3_ws = arena.region("gen_fa3_ws", fa3_ws_bytes, 256)?;
        let cutlass_ws_bytes: usize = 16 * 1024 * 1024;
        let cutlass_ws = arena.region("gen_cutlass_ws", cutlass_ws_bytes, 256)?;

        let positions = arena.region("gen_pos", (scratch_tokens * 4) as usize, 16)?;
        let slot_mapping = arena.region(
            "gen_slot",
            n_layers_active.max(1) * (scratch_tokens as usize) * 4,
            16,
        )?;
        let context_lens = arena.region("gen_ctx", 4, 16)?;
        let prefill_sliding_ctx =
            arena.region("gen_prefill_sctx", (scratch_tokens as usize) * 4, 16)?;
        // Sized for max_tokens i32 entries (not just the 2-entry prefix
        // sum): the unified decode-per-qi attention loop reuses this
        // region to stage a per-qi context-lens array `[1, 2, ..., N]`
        // and indexes it by `qi * 4`. With only 8 bytes (old FA2
        // prefill layout) writing beyond entry 1 corrupted adjacent
        // arena regions and degenerated generation quality.
        let cu_seqlens_q =
            arena.region("gen_cu_seqlens", ((scratch_tokens + 1) * 4) as usize, 16)?;
        let block_tables = arena.region("gen_bt", (max_blocks_per_seq * 4) as usize, 16)?;
        {
            let bt: Vec<i32> = (0..max_blocks_per_seq as i32).collect();
            block_tables.copy_from_host(bytemuck_cast_i32(&bt))?;
        }

        let residual = arena.region("gen_residual", (scratch_tokens * hidden * 2) as usize, 16)?;
        // Sized for spec_n_max rows so the batched verify lm_head/argmax
        // can score every chunk position (spec_n_max == 1 off-spec).
        let logits_f32 = arena.region(
            "gen_logits_f32",
            (spec_n_max as usize) * (vocab as usize) * 4,
            16,
        )?;
        let token_ids_region = arena.region("gen_tok_ids", (scratch_tokens * 4) as usize, 16)?;
        let sampled = arena.region("gen_sampled", (spec_n_max as usize) * 4, 16)?;
        let residual_ptr = residual.device_ptr();
        let kernels = self.layer_kernels();

        // === Sampling-tail state (sampled sessions ONLY) ================
        // Greedy sessions allocate nothing here and never load the PTX, so
        // a kernels dir without `sample_topk_f32` keeps serving greedy.
        // Device: K'<=1024 candidate (val, idx) buffers + count (8 KB + 4 B).
        // Host: pinned DtoH targets + the per-request seeded draw stream.
        let mut sample_tail = if sampled_mode {
            let cap = rvllm_sampling::KERNEL_K_CAP as usize;
            let vals = arena.region("gen_sample_vals", cap * 4, 16)?;
            let idx = arena.region("gen_sample_idx", cap * 4, 16)?;
            let cnt = arena.region("gen_sample_cnt", 4, 16)?;
            let module = self.kernels.load_ptx("sample_topk_f32")?;
            let fn_sample = module.get_function("sample_topk_f32_kernel")?;
            Some(SampleTailState {
                vals_ptr: vals.device_ptr(),
                idx_ptr: idx.device_ptr(),
                cnt_ptr: cnt.device_ptr(),
                candidate_capacity: rvllm_sampling::KERNEL_K_CAP,
                fn_sample,
                _module: module,
                pin_vals: PinnedBuf::new(cap)?,
                pin_idx: PinnedBuf::new(cap)?,
                pin_cnt: PinnedBuf::new(1)?,
                cands: Vec::with_capacity(cap),
                rng: rvllm_sampling::SampleRng::new(sampling.seed),
            })
        } else {
            None
        };

        use rvllm_loader::gemma4_arch::Gemma4LayerType;

        // Per-step decode metadata. Both eager and graph paths stage all
        // layers before launching compute. Region::copy_from_host uses the
        // default stream and cannot safely overwrite buffers read by this
        // nonblocking stream, so no per-layer scalar metadata writes are
        // allowed while a token is in flight.
        let slot_array = arena.region("gen_slot_arr", n_layers_active.max(1) * 4, 16)?;
        let ctx_array = arena.region("gen_ctx_arr", n_layers_active.max(1) * 4, 16)?;
        let slot_array_ptr = slot_array.device_ptr();
        let ctx_array_ptr = ctx_array.device_ptr();
        // Reusable host scratch for the per-step per-layer slot/ctx values.
        let mut slot_host: Vec<i32> = vec![0; n_layers_active.max(1)];
        let mut ctx_host: Vec<i32> = vec![0; n_layers_active.max(1)];

        // === Spec-decode verify metadata (device, refreshed per step) ===
        // Per-layer slot rows for the verify chunk, row stride spec_n_max:
        // layer l, chunk token i lives at [l * spec_n_max + i] (global:
        // absolute position; sliding: position % window). One ordered HtoD
        // per step covers all layers, matching the decode metadata arrays
        // above, so the verify step stays graph-capturable.
        let spec_slot_arr = arena.region(
            "gen_spec_slot",
            n_layers_active.max(1) * spec_n_max as usize * 4,
            16,
        )?;
        // Per-query sliding-clamped context [min(cur+qi+1, window)] for
        // the interleaved sliding attention inside verify chunks.
        let spec_sliding_ctx = arena.region("gen_spec_sctx", spec_n_max as usize * 4, 16)?;
        let layer_is_sliding: Vec<bool> = (0..n_layers_active)
            .map(|l| arch.layer_types[l] == Gemma4LayerType::SlidingAttention)
            .collect();
        let sliding_window = arch.sliding_window_size;
        // Compute the per-layer (slot, ctx) for a decode step into the host
        // scratch buffers. Identical formula to the original inline code
        // (was lines ~1775-1784).
        let fill_layer_meta = |step: usize, slot_h: &mut [i32], ctx_h: &mut [i32]| {
            for l in 0..n_layers_active {
                if layer_is_sliding[l] {
                    slot_h[l] = (step % sliding_window) as i32;
                    ctx_h[l] = (step + 1).min(sliding_window) as i32;
                } else {
                    slot_h[l] = step as i32;
                    ctx_h[l] = step as i32 + 1;
                }
            }
        };
        // Ordered HtoD refresh of positions + per-layer slot/ctx arrays.
        // Runs between eager tokens or graph replays, never during compute.
        let prepare_decode_inputs =
            |step: usize, slot_h: &mut [i32], ctx_h: &mut [i32]| -> Result<()> {
                fill_layer_meta(step, slot_h, ctx_h);
                let pos = [step as i32];
                self.stream.fence()?;
                unsafe {
                    positions.copy_from_host(bytemuck_cast_i32(&pos))?;
                    slot_array.copy_from_host(bytemuck_cast_i32(slot_h))?;
                    ctx_array.copy_from_host(bytemuck_cast_i32(ctx_h))?;
                }
                Ok(())
            };

        // Helper: run one token through all layers (decode path)
        type HostMeta<'a> = (&'a mut [i32], &'a mut [i32]);
        let run_one_token = |tok_id: u32, step: usize, host: HostMeta<'_>| -> Result<()> {
            prepare_decode_inputs(step, host.0, host.1)?;
            let tok_i32 = [tok_id as i32];
            token_ids_region.copy_from_host(bytemuck_cast_i32(&tok_i32))?;
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1,
                hidden,
                vocab: embed_vocab,
            }
            .launch(
                fn_embed,
                residual_ptr,
                self.model.embedding.offset_bytes,
                token_ids_region.device_ptr(),
                stream,
            )?;

            // Vision: overwrite the just-gathered text embedding with a
            // pre-computed image embedding at any matching image-soft-token
            // position. No-op when `image_embeds` is empty.
            unsafe {
                inject_image_embeds_f16(
                    residual_ptr,
                    hidden,
                    step,
                    1,
                    (hidden as u64) * 2,
                    image_embeds,
                    stream,
                )?;
            }
            self.build_ple_inputs(
                fn_embed,
                token_ids_region.device_ptr(),
                residual_ptr,
                ple_inputs.device_ptr(),
                ple_projection_f16.device_ptr(),
                gemm_f32_tmp.device_ptr(),
                1,
                stream,
            )?;

            for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                if layer_idx >= max_layers {
                    break;
                }
                let lt = arch.layer_types[layer_idx];
                let hd = arch.head_dim_for_layer(layer_idx) as u32;
                let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
                let q_dim = (arch.num_attention_heads as u32) * hd;
                let kv_dim = nkvh * hd;
                let layer_blocks = if lt == Gemma4LayerType::GlobalAttention {
                    num_blocks_total
                } else {
                    sliding_blocks
                };
                let layer_kv_elems =
                    2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[layer_idx];
                let layer_kv_scale_base =
                    kv_scale_cache.device_ptr() + kv_scale_layer_offsets[layer_idx];
                let layer_kv_scale_slots_half =
                    (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);

                let dims = crate::gemma4_layer_exec::Gemma4LayerDims {
                    num_tokens: 1,
                    hidden,
                    num_heads: arch.num_attention_heads as u32,
                    num_kv_heads: nkvh,
                    head_dim: hd,
                    rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                    rope_table_rows: arch.max_position_embeddings as u32,
                    intermediate: inter,
                    block_size,
                    max_blocks_per_seq: layer_blocks,
                    num_blocks_total: layer_blocks,
                    attn_scale: 1.0,
                    rms_eps: arch.rms_norm_eps,
                    layer_type: lt,
                    sliding_window: arch.sliding_window_size as u32,
                    f16_kv: use_f16_kv,
                    num_hidden_layers: arch.num_hidden_layers as u32,
                    layer_idx: layer_idx as u32,
                    ple_dim,
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
                let k_out = q_base + (q_dim as u64) * 2;
                let v_out = k_out + (kv_dim as u64) * 2;
                let (cos, sin) = match lt {
                    Gemma4LayerType::SlidingAttention => (
                        self.model.rope_cos_sliding.offset_bytes,
                        self.model.rope_sin_sliding.offset_bytes,
                    ),
                    Gemma4LayerType::GlobalAttention => (
                        self.model.rope_cos_global.offset_bytes,
                        self.model.rope_sin_global.offset_bytes,
                    ),
                };
                let scratch = crate::gemma4_layer_exec::Gemma4LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(),
                    hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base,
                    k_out,
                    v_out,
                    q_normed: q_normed.device_ptr(),
                    k_normed: k_normed.device_ptr(),
                    v_normed: v_normed.device_ptr(),
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + (layer_kv_elems / 2) * kv_bytes_per_elem as u64,
                    q_scale_ptr: q_scale_region.device_ptr(),
                    kv_scale_ptr: kv_scale_region.device_ptr(),
                    k_scale_cache: layer_kv_scale_base,
                    v_scale_cache: layer_kv_scale_base + layer_kv_scale_slots_half * 4,
                    q_scale_cache: q_scale_cache_ptr,
                    attn_out: attn_out.device_ptr(),
                    attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(),
                    delta_f16: delta_f16.device_ptr(),
                    gate_up_out: gate_up_out.device_ptr(),
                    gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(),
                    mlp_out_scale: mlp_out_scale.device_ptr(),
                    gemm_f32_tmp: gemm_f32_tmp.device_ptr(),
                    cutlass_workspace: cutlass_ws.device_ptr(),
                    cutlass_workspace_bytes: cutlass_ws_bytes,
                    fa3_workspace: fa3_ws.device_ptr(),
                    fa3_workspace_bytes: fa3_ws_bytes,
                    ple_inputs: ple_inputs.device_ptr(),
                    ple_gate: ple_gate.device_ptr(),
                };
                let meta = crate::gemma4_layer_exec::Gemma4MetadataPtrs {
                    positions: positions.device_ptr(),
                    slot_mapping: slot_array_ptr + (layer_idx as u64) * 4,
                    cos,
                    sin,
                    block_tables: block_tables.device_ptr(),
                    context_lens: ctx_array_ptr + (layer_idx as u64) * 4,
                };
                crate::gemma4_layer_exec::gemma4_forward(
                    dims,
                    &kernels,
                    &w,
                    &scratch,
                    &meta,
                    &self.cublaslt,
                    &self.cutlass,
                    &self.sliding_attention,
                    &self.global_attention,
                    residual_ptr,
                    stream,
                )?;
            }
            Ok(())
        };

        // Capturable pure-device CUDA-graph decode chain.
        // Identical math to `run_one_token`'s decode body, but with ZERO
        // synchronous host->device copies so it is safe to record into a
        // CUDA graph and `replay()` each step. Differences vs `run_one_token`:
        //   * No token-id / positions / image-inject / per-layer slot/ctx
        //     HtoD. The embed gather reads the fixed `token_ids_region`
        //     (seeded once before capture; thereafter refreshed device->device
        //     from the previous step's argmax — see the DtoD at the tail).
        //     Per-step positions + per-layer slot/ctx come from device arrays
        //     refreshed eagerly by `prepare_decode_inputs` between replays.
        //   * Final norm + lm_head GEMM + argmax are inside the captured
        //     region; argmax writes `sampled` (i32). The closing DtoD copies
        //     `sampled` -> `token_ids_region` so the NEXT replay's embed
        //     gather consumes this step's token with no host round-trip.
        // Image embeds never apply during decode (positions are past the
        // prompt), so the inject step is intentionally omitted here.
        //
        // Split in two: `decode_forward_logits` stops after the lm_head GEMM
        // (the sampled decode arm captures/replays exactly this and runs the
        // sampling tail eagerly per step); `decode_forward` appends the
        // argmax + device token feedback — byte-for-byte the old greedy
        // chain, so greedy graphs are unchanged.
        let decode_forward_logits = || -> Result<()> {
            rvllm_fused::EmbeddingGatherLaunch {
                num_tokens: 1,
                hidden,
                vocab: embed_vocab,
            }
            .launch(
                fn_embed,
                residual_ptr,
                self.model.embedding.offset_bytes,
                token_ids_region.device_ptr(),
                stream,
            )?;
            self.build_ple_inputs(
                fn_embed,
                token_ids_region.device_ptr(),
                residual_ptr,
                ple_inputs.device_ptr(),
                ple_projection_f16.device_ptr(),
                gemm_f32_tmp.device_ptr(),
                1,
                stream,
            )?;

            for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                if layer_idx >= max_layers {
                    break;
                }
                let lt = arch.layer_types[layer_idx];
                let hd = arch.head_dim_for_layer(layer_idx) as u32;
                let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
                let q_dim = (arch.num_attention_heads as u32) * hd;
                let kv_dim = nkvh * hd;
                let layer_blocks = if lt == Gemma4LayerType::GlobalAttention {
                    num_blocks_total
                } else {
                    sliding_blocks
                };
                let layer_kv_elems =
                    2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[layer_idx];
                let layer_kv_scale_base =
                    kv_scale_cache.device_ptr() + kv_scale_layer_offsets[layer_idx];
                let layer_kv_scale_slots_half =
                    (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);

                let dims = crate::gemma4_layer_exec::Gemma4LayerDims {
                    num_tokens: 1,
                    hidden,
                    num_heads: arch.num_attention_heads as u32,
                    num_kv_heads: nkvh,
                    head_dim: hd,
                    rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                    rope_table_rows: arch.max_position_embeddings as u32,
                    intermediate: inter,
                    block_size,
                    max_blocks_per_seq: layer_blocks,
                    num_blocks_total: layer_blocks,
                    attn_scale: 1.0,
                    rms_eps: arch.rms_norm_eps,
                    layer_type: lt,
                    sliding_window: arch.sliding_window_size as u32,
                    f16_kv: use_f16_kv,
                    num_hidden_layers: arch.num_hidden_layers as u32,
                    layer_idx: layer_idx as u32,
                    ple_dim,
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
                let k_out = q_base + (q_dim as u64) * 2;
                let v_out = k_out + (kv_dim as u64) * 2;
                let (cos, sin) = match lt {
                    Gemma4LayerType::SlidingAttention => (
                        self.model.rope_cos_sliding.offset_bytes,
                        self.model.rope_sin_sliding.offset_bytes,
                    ),
                    Gemma4LayerType::GlobalAttention => (
                        self.model.rope_cos_global.offset_bytes,
                        self.model.rope_sin_global.offset_bytes,
                    ),
                };
                let scratch = crate::gemma4_layer_exec::Gemma4LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(),
                    hidden_scale: hidden_scale.device_ptr(),
                    q_out: q_base,
                    k_out,
                    v_out,
                    q_normed: q_normed.device_ptr(),
                    k_normed: k_normed.device_ptr(),
                    v_normed: v_normed.device_ptr(),
                    q_fp8: q_fp8.device_ptr(),
                    k_cache: layer_kv_base,
                    v_cache: layer_kv_base + (layer_kv_elems / 2) * kv_bytes_per_elem as u64,
                    q_scale_ptr: q_scale_region.device_ptr(),
                    kv_scale_ptr: kv_scale_region.device_ptr(),
                    k_scale_cache: layer_kv_scale_base,
                    v_scale_cache: layer_kv_scale_base + layer_kv_scale_slots_half * 4,
                    q_scale_cache: q_scale_cache_ptr,
                    attn_out: attn_out.device_ptr(),
                    attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(),
                    delta_f16: delta_f16.device_ptr(),
                    gate_up_out: gate_up_out.device_ptr(),
                    gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(),
                    mlp_out_scale: mlp_out_scale.device_ptr(),
                    gemm_f32_tmp: gemm_f32_tmp.device_ptr(),
                    cutlass_workspace: cutlass_ws.device_ptr(),
                    cutlass_workspace_bytes: cutlass_ws_bytes,
                    fa3_workspace: fa3_ws.device_ptr(),
                    fa3_workspace_bytes: fa3_ws_bytes,
                    ple_inputs: ple_inputs.device_ptr(),
                    ple_gate: ple_gate.device_ptr(),
                };
                // Per-layer (slot, ctx) sourced from the device arrays
                // refreshed by `prepare_decode_inputs` (vs `run_one_token`'s
                // inline per-layer HtoD). Same i32 values — capture-safe.
                let meta = crate::gemma4_layer_exec::Gemma4MetadataPtrs {
                    positions: positions.device_ptr(),
                    slot_mapping: slot_array_ptr + (layer_idx as u64) * 4,
                    cos,
                    sin,
                    block_tables: block_tables.device_ptr(),
                    context_lens: ctx_array_ptr + (layer_idx as u64) * 4,
                };
                crate::gemma4_layer_exec::gemma4_forward(
                    dims,
                    &kernels,
                    &w,
                    &scratch,
                    &meta,
                    &self.cublaslt,
                    &self.cutlass,
                    &self.sliding_attention,
                    &self.global_attention,
                    residual_ptr,
                    stream,
                )?;
            }

            // Final norm + lm_head GEMM (same chain the eager decode loop
            // runs after `run_one_token`), captured here so the whole step
            // replays as one graph.
            rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                num_tokens: 1,
                hidden,
                eps: arch.rms_norm_eps,
            }
            .launch(
                kernels.fused_rmsnorm,
                residual_ptr,
                self.model.final_norm.offset_bytes,
                stream,
            )?;
            self.cublaslt.f16_gemm_f32(
                residual_ptr,
                self.model.lm_head_f16.offset_bytes,
                logits_f32.device_ptr(),
                1,
                vocab as i32,
                hidden as i32,
                stream,
            )?;
            Ok(())
        };
        let decode_forward = || -> Result<()> {
            decode_forward_logits()?;
            rvllm_fused::ArgmaxLaunch {
                num_tokens: 1,
                vocab,
            }
            .launch(
                fn_argmax,
                logits_f32.device_ptr(),
                sampled.device_ptr(),
                stream,
            )?;
            // Device-side token feedback: this step's argmax becomes the
            // next replay's embed input with no host round-trip. A pruned
            // lm-head returns a local row id, so map through
            // keep_ids before the next embedding gather.
            if keep_ids_ptr != 0 {
                rvllm_fused::MapTokenIdLaunch {
                    keep_len: keep_ids_len,
                }
                .launch(
                    &self.fused.fn_map_token_id,
                    sampled.device_ptr(),
                    token_ids_region.device_ptr(),
                    keep_ids_ptr,
                    stream,
                )?;
            } else {
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    token_ids_region.device_ptr(),
                    sampled.device_ptr(),
                    4,
                    stream as _,
                );
            }
            Ok(())
        };

        // === Spec-decode verify forward =================================
        // One continuation-prefill chunk over `chunk` = [last_committed,
        // draft_0, ..] at absolute positions [cur, cur+n): embed -> all
        // layers (batched GEMMs/norms at M=n; global attention batched
        // via the SM89 prefill kernel; sliding attention interleaved
        // per-token inside gemma4_forward_phase — see
        // Gemma4Phase::Prefill::sliding_ctx_per_qi) -> final norm ->
        // lm_head GEMM (M=n) -> argmax per row into `sampled[0..n)`.
        //
        // KV invariant that makes rejected drafts harmless: every chunk
        // rewrites KV for exactly the positions it covers BEFORE
        // attending to them, and the causal mask stops reads past the
        // chunk end. Stale KV from a rejected draft only ever lives at
        // positions >= the next chunk's start, which the next chunk
        // overwrites (slot mapping is deterministic by position) or
        // never attends.
        //
        // All metadata is staged with ordered HtoD into device arrays
        // using fixed device arrays, so this step is graph-capturable later.
        // S3 split: metadata staging (eager every cycle — host-built values
        // copied into FIXED device arrays) vs the pure-device verify forward
        // (graph-capturable per chunk size n). On the sm_90/Fa3 route every
        // launch parameter in the forward is either fixed for a given n or
        // read from the staged device arrays; `chunk_start` is consumed only
        // by the non-Fa3 per-qi fallback arm (which also stages host data
        // mid-forward), hence the Fa3-backend guard on spec graphs below.
        let spec_stage_metadata = |chunk: &[i32], cur: usize| -> Result<()> {
            let n = chunk.len() as u32;
            debug_assert!(n >= 2 && n <= spec_n_max);
            unsafe {
                crate::bring_up::htod_ordered(
                    token_ids_region.device_ptr(),
                    bytemuck_cast_i32(chunk),
                    stream,
                )?;
                let pos: Vec<i32> = (cur as i32..cur as i32 + n as i32).collect();
                crate::bring_up::htod_ordered(
                    positions.device_ptr(),
                    bytemuck_cast_i32(&pos),
                    stream,
                )?;
                let ctx = [(cur as u32 + n) as i32];
                crate::bring_up::htod_ordered(
                    context_lens.device_ptr(),
                    bytemuck_cast_i32(&ctx),
                    stream,
                )?;
                let cu_seq = [0i32, n as i32];
                crate::bring_up::htod_ordered(
                    cu_seqlens_q.device_ptr(),
                    bytemuck_cast_i32(&cu_seq),
                    stream,
                )?;
                // Per-layer slot rows (row stride spec_n_max) + per-query
                // sliding-clamped ctx, one ordered HtoD each for all layers.
                let mut slot_rows: Vec<i32> = vec![0; n_layers_active * spec_n_max as usize];
                for (l, row) in slot_rows.chunks_mut(spec_n_max as usize).enumerate() {
                    for (i, s) in row.iter_mut().take(n as usize).enumerate() {
                        let p = cur + i;
                        *s = if layer_is_sliding[l] {
                            (p % sliding_window) as i32
                        } else {
                            p as i32
                        };
                    }
                }
                crate::bring_up::htod_ordered(
                    spec_slot_arr.device_ptr(),
                    bytemuck_cast_i32(&slot_rows),
                    stream,
                )?;
                let sctx: Vec<i32> = (0..n as usize)
                    .map(|i| (cur + i + 1).min(sliding_window) as i32)
                    .collect();
                crate::bring_up::htod_ordered(
                    spec_sliding_ctx.device_ptr(),
                    bytemuck_cast_i32(&sctx),
                    stream,
                )?;
            }
            Ok(())
        };
        // Pure-device verify forward at M=n. Reads only staged device
        // arrays + fixed weights/scratch; capturable (see staging comment).
        let spec_verify_forward = |n: u32, chunk_start: u32| -> Result<()> {
            unsafe {
                rvllm_fused::EmbeddingGatherLaunch {
                    num_tokens: n,
                    hidden,
                    vocab,
                }
                .launch(
                    fn_embed,
                    residual_ptr,
                    self.model.embedding.offset_bytes,
                    token_ids_region.device_ptr(),
                    stream,
                )?;
                // No image-embed inject: decode positions are past the
                // prompt by construction.

                let phase = crate::gemma4_layer_exec::Gemma4Phase::Prefill {
                    cu_seqlens_q: cu_seqlens_q.device_ptr(),
                    max_seqlen_q: n,
                    num_seqs: 1,
                    chunk_start,
                    sliding_ctx_per_qi: spec_sliding_ctx.device_ptr(),
                };

                for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                    if layer_idx >= max_layers {
                        break;
                    }
                    let lt = arch.layer_types[layer_idx];
                    let hd = arch.head_dim_for_layer(layer_idx) as u32;
                    let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
                    let q_dim = (arch.num_attention_heads as u32) * hd;
                    let kv_dim = nkvh * hd;
                    let layer_blocks = if lt == Gemma4LayerType::GlobalAttention {
                        num_blocks_total
                    } else {
                        sliding_blocks
                    };
                    let layer_kv_elems =
                        2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                    let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[layer_idx];
                    let layer_kv_scale_base =
                        kv_scale_cache.device_ptr() + kv_scale_layer_offsets[layer_idx];
                    let layer_kv_scale_slots_half =
                        (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);

                    let dims = crate::gemma4_layer_exec::Gemma4LayerDims {
                        num_tokens: n,
                        hidden,
                        num_heads: arch.num_attention_heads as u32,
                        num_kv_heads: nkvh,
                        head_dim: hd,
                        rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                        rope_table_rows: arch.max_position_embeddings as u32,
                        intermediate: inter,
                        block_size,
                        max_blocks_per_seq: layer_blocks,
                        num_blocks_total: layer_blocks,
                        attn_scale: 1.0,
                        rms_eps: arch.rms_norm_eps,
                        layer_type: lt,
                        sliding_window: arch.sliding_window_size as u32,
                        f16_kv: false, // verify chunk uses FP8 KV (prefill path)
                        num_hidden_layers: arch.num_hidden_layers as u32,
                        layer_idx: layer_idx as u32,
                        ple_dim,
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
                    let k_out = q_base + (q_dim as u64) * 2;
                    let v_out = k_out + (kv_dim as u64) * 2;
                    let (cos, sin) = match lt {
                        Gemma4LayerType::SlidingAttention => (
                            self.model.rope_cos_sliding.offset_bytes,
                            self.model.rope_sin_sliding.offset_bytes,
                        ),
                        Gemma4LayerType::GlobalAttention => (
                            self.model.rope_cos_global.offset_bytes,
                            self.model.rope_sin_global.offset_bytes,
                        ),
                    };
                    let scratch = crate::gemma4_layer_exec::Gemma4LayerScratch {
                        hidden_fp8: hidden_fp8.device_ptr(),
                        hidden_scale: hidden_scale.device_ptr(),
                        q_out: q_base,
                        k_out,
                        v_out,
                        q_normed: q_normed.device_ptr(),
                        k_normed: k_normed.device_ptr(),
                        v_normed: v_normed.device_ptr(),
                        q_fp8: q_fp8.device_ptr(),
                        k_cache: layer_kv_base,
                        v_cache: layer_kv_base + (layer_kv_elems / 2) * kv_bytes_per_elem as u64,
                        q_scale_ptr: q_scale_region.device_ptr(),
                        kv_scale_ptr: kv_scale_region.device_ptr(),
                        k_scale_cache: layer_kv_scale_base,
                        v_scale_cache: layer_kv_scale_base + layer_kv_scale_slots_half * 4,
                        q_scale_cache: q_scale_cache_ptr,
                        attn_out: attn_out.device_ptr(),
                        attn_out_fp8: attn_out_fp8.device_ptr(),
                        attn_out_scale: attn_out_scale.device_ptr(),
                        delta_f16: delta_f16.device_ptr(),
                        gate_up_out: gate_up_out.device_ptr(),
                        gate_up_fp8: gate_up_fp8.device_ptr(),
                        gate_up_scale: gate_up_scale.device_ptr(),
                        mlp_out_fp8: mlp_out_fp8.device_ptr(),
                        mlp_out_scale: mlp_out_scale.device_ptr(),
                        gemm_f32_tmp: gemm_f32_tmp.device_ptr(),
                        cutlass_workspace: cutlass_ws.device_ptr(),
                        cutlass_workspace_bytes: cutlass_ws_bytes,
                        fa3_workspace: fa3_ws.device_ptr(),
                        fa3_workspace_bytes: fa3_ws_bytes,
                        ple_inputs: ple_inputs.device_ptr(),
                        ple_gate: ple_gate.device_ptr(),
                    };
                    let meta = crate::gemma4_layer_exec::Gemma4MetadataPtrs {
                        positions: positions.device_ptr(),
                        slot_mapping: spec_slot_arr.device_ptr()
                            + (layer_idx as u64) * (spec_n_max as u64) * 4,
                        cos,
                        sin,
                        block_tables: block_tables.device_ptr(),
                        context_lens: context_lens.device_ptr(),
                    };
                    crate::gemma4_layer_exec::gemma4_forward_phase(
                        dims,
                        &kernels,
                        &w,
                        &scratch,
                        &meta,
                        &self.cublaslt,
                        &self.cutlass,
                        &self.sliding_attention,
                        &self.global_attention,
                        residual_ptr,
                        stream,
                        phase,
                    )?;
                }

                // Batched tail: final norm on every chunk row, lm_head
                // GEMM at M=n, per-row argmax into sampled[0..n).
                rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                    num_tokens: n,
                    hidden,
                    eps: arch.rms_norm_eps,
                }
                .launch(
                    kernels.fused_rmsnorm,
                    residual_ptr,
                    self.model.final_norm.offset_bytes,
                    stream,
                )?;
                self.cublaslt.f16_gemm_f32(
                    residual_ptr,
                    self.model.lm_head_f16.offset_bytes,
                    logits_f32.device_ptr(),
                    n as i32,
                    vocab as i32,
                    hidden as i32,
                    stream,
                )?;
                rvllm_fused::ArgmaxLaunch {
                    num_tokens: n,
                    vocab,
                }
                .launch(
                    fn_argmax,
                    logits_f32.device_ptr(),
                    sampled.device_ptr(),
                    stream,
                )?;
            }
            Ok(())
        };

        let t0 = std::time::Instant::now();
        let diag_compare = diag_compare_enabled;
        let prefill_plan =
            generation_prefill_plan(skip_decode, use_fast_prefill, diag_compare, prompt_len);

        // Phase 1: prompt through per-token decode (default, correct-by-design).
        //
        // On sm_121 with FP8 block-scale weights (Gemma 4 fp8-block), the
        // per-token path uses `fp8_gemv_blockwise_wpr_native_f16in_kernel`
        // which preserves the per-channel weight block-scale. The batch
        // (num_tokens>1) GEMM path goes through
        // `fp8_gemm_channelscale_or_fallback`, which on Blackwell consumer
        // collapses to a scalar weight scale because cuBLASLt's FP8
        // channelscale heuristic `LaunchFailed`s at this arch. That is a
        // genuine numerical difference, not a hidden bug — the two paths
        // are not bit-identical by design at num_tokens<CUTLASS_M_MIN(=128).
        //
        // Path forward for genuine batch-prefill speedup:
        //   * num_tokens >=128 : CUTLASS SM120 blockwise FP8 GEMM (landed;
        //     opt-in via RVLLM_FP8_GEMM_CUTLASS_SM120 + M>=128 gate in
        //     gemma4_layer_exec).  This preserves the per-channel scale
        //     via SFA/SFB prep.
        //   * num_tokens < 128 : per-token loop is optimal (fp8_gemv is
        //     M=1-only; running it T times reads weights T times but each
        //     call is already bandwidth-bound; cost parity with any batched
        //     solution at small M).
        //
        // So we keep the per-token loop as the default for ALL prompt
        // lengths today. RVLLM_BATCH_PREFILL=1 flips to the unified
        // batch path (diagnostic: verifies CUTLASS >=128 correctness,
        // or measures the collapsed-scalar quality floor at <128).
        if prefill_plan.sequential {
            for (i, &tok) in prompt_ids.iter().enumerate() {
                run_one_token(tok, i, (&mut slot_host, &mut ctx_host))?;
            }
        }

        // Optional prefill-vs-decode residual compare (RVLLM_DIAG_COMPARE=1).
        // Captures the last-token residual produced by per-token decode
        // (correct reference), resets KV, re-runs the prompt via batch
        // prefill, captures the same row, and prints the diff. Combine
        // with `RVLLM_MAX_LAYERS=N` to bisect where the two paths
        // diverge. Only fires when prompt_len > 1 (decode==prefill
        // trivially at prompt_len=1).
        let mut decode_ref_last: Vec<u16> = Vec::new();
        let mut decode_ref_first: Vec<u16> = Vec::new();
        if prefill_plan.reset_before_staged {
            // Already captured: residual_ptr holds LAST token's residual
            // after all prompt tokens were processed sequentially.
            self.stream.fence()?;
            decode_ref_last = vec![0u16; hidden as usize];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                decode_ref_last.as_mut_ptr() as *mut _,
                residual_ptr,
                (hidden * 2) as _,
            );

            // For FIRST token reference, re-run just token 0 through
            // a fresh KV cache — the residual after that step is what
            // prefill's row 0 should match (no prior context at
            // position 0 in either path).
            cudarc::driver::sys::cuMemsetD8_v2(kv_cache.device_ptr(), 0, kv_total_bytes as usize);
            self.stream.fence()?;
            run_one_token(prompt_ids[0], 0, (&mut slot_host, &mut ctx_host))?;
            self.stream.fence()?;
            decode_ref_first = vec![0u16; hidden as usize];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                decode_ref_first.as_mut_ptr() as *mut _,
                residual_ptr,
                (hidden * 2) as _,
            );

            // Reset KV again before prefill re-runs the whole prompt.
            cudarc::driver::sys::cuMemsetD8_v2(kv_cache.device_ptr(), 0, kv_total_bytes as usize);
            self.stream.fence()?;
        }

        // Staged batch/chunk prefill. Normal fast-prefill modes select this
        // instead of sequential prefill; diagnostic comparison is the only
        // two-pass mode and resets KV above before this rerun.
        if prefill_plan.staged {
            let diag_verbose =
                std::env::var_os("RVLLM_DIAG_VERBOSE").is_some() && !use_chunked_prefill;
            let prefill_chunk = if use_chunked_prefill {
                prefill_chunk_tokens
            } else {
                prompt_len
            };
            let mut last_prefill_len = 0u32;
            let mut chunk_start = 0u32;
            let prefill_slot_stride = scratch_tokens as usize;
            let mut prefill_slot_host = vec![0i32; n_layers_active.max(1) * prefill_slot_stride];
            let mut prefill_sliding_ctx_host = vec![0i32; prefill_slot_stride];
            while chunk_start < prompt_len {
                let chunk_len = (prompt_len - chunk_start).min(prefill_chunk);
                last_prefill_len = chunk_len;
                let tok_ids: Vec<i32> = prompt_ids
                    [chunk_start as usize..(chunk_start + chunk_len) as usize]
                    .iter()
                    .map(|&t| t as i32)
                    .collect();
                let pos: Vec<i32> =
                    (chunk_start as i32..(chunk_start + chunk_len) as i32).collect();
                let ctx = [(chunk_start + chunk_len) as i32];
                let cu_seq = [0i32, chunk_len as i32];
                fill_prefill_chunk_metadata(
                    &arch.layer_types[..n_layers_active],
                    chunk_start,
                    chunk_len,
                    sliding_window_tokens,
                    prefill_slot_stride,
                    &mut prefill_slot_host[..n_layers_active * prefill_slot_stride],
                    &mut prefill_sliding_ctx_host,
                );

                // Every chunk reuses token, metadata, residual, and scratch
                // regions. Finish the prior chunk, then stage all host inputs
                // before launching any work for this one.
                self.stream.fence()?;
                token_ids_region.copy_from_host(bytemuck_cast_i32(&tok_ids))?;
                positions.copy_from_host(bytemuck_cast_i32(&pos))?;
                context_lens.copy_from_host(bytemuck_cast_i32(&ctx))?;
                cu_seqlens_q.copy_from_host(bytemuck_cast_i32(&cu_seq))?;
                slot_mapping.copy_from_host(bytemuck_cast_i32(&prefill_slot_host))?;
                if use_chunked_prefill {
                    prefill_sliding_ctx.copy_from_host(bytemuck_cast_i32(
                        &prefill_sliding_ctx_host[..chunk_len as usize],
                    ))?;
                }
                if diag_verbose {
                    self.stream.fence()?;
                    let mut readback = vec![0i32; chunk_len as usize];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        readback.as_mut_ptr() as *mut _,
                        token_ids_region.device_ptr(),
                        (chunk_len * 4) as _,
                    );
                    eprintln!(
                        "[DIAG] token_ids_region len={} first={:?} last={:?}",
                        readback.len(),
                        readback.iter().take(8).copied().collect::<Vec<_>>(),
                        readback
                            .iter()
                            .rev()
                            .take(8)
                            .copied()
                            .collect::<Vec<_>>()
                            .into_iter()
                            .rev()
                            .collect::<Vec<_>>(),
                    );
                }
                rvllm_fused::EmbeddingGatherLaunch {
                    num_tokens: chunk_len,
                    hidden,
                    vocab: embed_vocab,
                }
                .launch(
                    fn_embed,
                    residual_ptr,
                    self.model.embedding.offset_bytes,
                    token_ids_region.device_ptr(),
                    stream,
                )?;
                // Vision: overwrite gathered rows at image-token positions.
                unsafe {
                    inject_image_embeds_f16(
                        residual_ptr,
                        hidden,
                        chunk_start as usize,
                        chunk_len as usize,
                        (hidden as u64) * 2,
                        image_embeds,
                        stream,
                    )?;
                }
                self.build_ple_inputs(
                    fn_embed,
                    token_ids_region.device_ptr(),
                    residual_ptr,
                    ple_inputs.device_ptr(),
                    ple_projection_f16.device_ptr(),
                    gemm_f32_tmp.device_ptr(),
                    chunk_len,
                    stream,
                )?;
                if diag_verbose {
                    self.stream.fence()?;
                    let mut r0 = vec![0u16; 4];
                    let mut r_n_minus_1 = vec![0u16; 4];
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        r0.as_mut_ptr() as *mut _,
                        residual_ptr,
                        8,
                    );
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        r_n_minus_1.as_mut_ptr() as *mut _,
                        residual_ptr + ((chunk_len - 1) as u64 * hidden as u64 * 2),
                        8,
                    );
                    eprintln!(
                        "[DIAG] post-gather row0[..4]={:?} rowN-1[..4]={:?}",
                        r0.iter()
                            .map(|&x| crate::bring_up::f16_to_f32(x))
                            .collect::<Vec<_>>(),
                        r_n_minus_1
                            .iter()
                            .map(|&x| crate::bring_up::f16_to_f32(x))
                            .collect::<Vec<_>>(),
                    );
                }

                let phase = crate::gemma4_layer_exec::Gemma4Phase::Prefill {
                    cu_seqlens_q: cu_seqlens_q.device_ptr(),
                    max_seqlen_q: chunk_len,
                    num_seqs: 1,
                    chunk_start,
                    sliding_ctx_per_qi: if use_chunked_prefill {
                        prefill_sliding_ctx.device_ptr()
                    } else {
                        0
                    },
                };

                for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                    if layer_idx >= max_layers {
                        break;
                    }
                    let lt = arch.layer_types[layer_idx];
                    let hd = arch.head_dim_for_layer(layer_idx) as u32;
                    let nkvh = arch.num_kv_heads_for_layer(layer_idx) as u32;
                    let q_dim = (arch.num_attention_heads as u32) * hd;
                    let kv_dim = nkvh * hd;
                    let layer_blocks = if lt == Gemma4LayerType::GlobalAttention {
                        num_blocks_total
                    } else {
                        sliding_blocks
                    };
                    let slot_mapping_ptr = slot_mapping.device_ptr()
                        + (layer_idx * prefill_slot_stride * core::mem::size_of::<i32>()) as u64;
                    let layer_kv_elems =
                        2u64 * layer_blocks as u64 * block_size as u64 * nkvh as u64 * hd as u64;
                    let layer_kv_base = kv_cache.device_ptr() + kv_layer_offsets[layer_idx];
                    let layer_kv_scale_base =
                        kv_scale_cache.device_ptr() + kv_scale_layer_offsets[layer_idx];
                    let layer_kv_scale_slots_half =
                        (layer_blocks as u64) * (block_size as u64) * (nkvh as u64);

                    let dims = crate::gemma4_layer_exec::Gemma4LayerDims {
                        num_tokens: chunk_len,
                        hidden,
                        num_heads: arch.num_attention_heads as u32,
                        num_kv_heads: nkvh,
                        head_dim: hd,
                        rotary_dim: arch.rotary_dim_for_layer(layer_idx) as u32,
                        rope_table_rows: arch.max_position_embeddings as u32,
                        intermediate: inter,
                        block_size,
                        max_blocks_per_seq: layer_blocks,
                        num_blocks_total: layer_blocks,
                        attn_scale: 1.0,
                        rms_eps: arch.rms_norm_eps,
                        layer_type: lt,
                        sliding_window: arch.sliding_window_size as u32,
                        f16_kv: false, // prefill uses FP8 KV (no F16 prefill kernel)
                        num_hidden_layers: arch.num_hidden_layers as u32,
                        layer_idx: layer_idx as u32,
                        ple_dim,
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
                    // Row-major [num_tokens, q_dim+2*kv_dim]: k_out / v_out
                    // point at row 0's K / V sub-slice. The rmsnorm kernel
                    // applies `src_row_stride` to reach later tokens — the
                    // old `num_tokens * q_dim * 2` formula assumed a
                    // columnar "all Q then all K then all V" layout that
                    // the cuBLASLt QKV GEMM does NOT produce.
                    let k_out = q_base + (q_dim as u64) * 2;
                    let v_out = k_out + (kv_dim as u64) * 2;
                    let (cos, sin) = match lt {
                        Gemma4LayerType::SlidingAttention => (
                            self.model.rope_cos_sliding.offset_bytes,
                            self.model.rope_sin_sliding.offset_bytes,
                        ),
                        Gemma4LayerType::GlobalAttention => (
                            self.model.rope_cos_global.offset_bytes,
                            self.model.rope_sin_global.offset_bytes,
                        ),
                    };
                    let scratch = crate::gemma4_layer_exec::Gemma4LayerScratch {
                        hidden_fp8: hidden_fp8.device_ptr(),
                        hidden_scale: hidden_scale.device_ptr(),
                        q_out: q_base,
                        k_out,
                        v_out,
                        q_normed: q_normed.device_ptr(),
                        k_normed: k_normed.device_ptr(),
                        v_normed: v_normed.device_ptr(),
                        q_fp8: q_fp8.device_ptr(),
                        k_cache: layer_kv_base,
                        v_cache: layer_kv_base + (layer_kv_elems / 2) * kv_bytes_per_elem as u64,
                        q_scale_ptr: q_scale_region.device_ptr(),
                        kv_scale_ptr: kv_scale_region.device_ptr(),
                        k_scale_cache: layer_kv_scale_base,
                        v_scale_cache: layer_kv_scale_base + layer_kv_scale_slots_half * 4,
                        q_scale_cache: q_scale_cache_ptr,
                        attn_out: attn_out.device_ptr(),
                        attn_out_fp8: attn_out_fp8.device_ptr(),
                        attn_out_scale: attn_out_scale.device_ptr(),
                        delta_f16: delta_f16.device_ptr(),
                        gate_up_out: gate_up_out.device_ptr(),
                        gate_up_fp8: gate_up_fp8.device_ptr(),
                        gate_up_scale: gate_up_scale.device_ptr(),
                        mlp_out_fp8: mlp_out_fp8.device_ptr(),
                        mlp_out_scale: mlp_out_scale.device_ptr(),
                        gemm_f32_tmp: gemm_f32_tmp.device_ptr(),
                        cutlass_workspace: cutlass_ws.device_ptr(),
                        cutlass_workspace_bytes: cutlass_ws_bytes,
                        fa3_workspace: fa3_ws.device_ptr(),
                        fa3_workspace_bytes: fa3_ws_bytes,
                        ple_inputs: ple_inputs.device_ptr(),
                        ple_gate: ple_gate.device_ptr(),
                    };
                    let meta = crate::gemma4_layer_exec::Gemma4MetadataPtrs {
                        positions: positions.device_ptr(),
                        slot_mapping: slot_mapping_ptr,
                        cos,
                        sin,
                        block_tables: block_tables.device_ptr(),
                        context_lens: context_lens.device_ptr(),
                    };
                    crate::gemma4_layer_exec::gemma4_forward_phase(
                        dims,
                        &kernels,
                        &w,
                        &scratch,
                        &meta,
                        &self.cublaslt,
                        &self.cutlass,
                        &self.sliding_attention,
                        &self.global_attention,
                        residual_ptr,
                        stream,
                        phase,
                    )?;
                }
                chunk_start += chunk_len;
            }

            // Diag capture BEFORE the extract-last memcpy: row 0 is
            // the first-token output (no prior context); row
            // prompt_len-1 is the last-token output the LM head
            // consumes. Without this ordering, the memcpy below
            // would clobber row 0 with row N-1 and the row-0
            // comparison would falsely report a bug.
            self.stream.fence()?;
            let mut prefill_first = vec![0u16; hidden as usize];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                prefill_first.as_mut_ptr() as *mut _,
                residual_ptr,
                (hidden * 2) as _,
            );
            let mut prefill_last = vec![0u16; hidden as usize];
            let last_off_diag = (last_prefill_len - 1) as u64 * hidden as u64 * 2;
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                prefill_last.as_mut_ptr() as *mut _,
                residual_ptr + last_off_diag,
                (hidden * 2) as _,
            );

            // Extract last token's residual for decode
            if last_prefill_len > 1 {
                let last_offset = (last_prefill_len - 1) as u64 * hidden as u64 * 2;
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    residual_ptr,
                    residual_ptr + last_offset,
                    (hidden * 2) as usize,
                    stream as _,
                );
            }

            if !diag_compare {
                if skip_decode {
                    return Ok(Vec::new());
                }
                // use_batch_prefill: fall through to LM head.
            } else {
                let stats = |label: &str, reference: &[u16], probe: &[u16]| {
                    let mut max_abs = 0f32;
                    let mut sum_sq_diff = 0f64;
                    let mut sum_sq_ref = 0f64;
                    let mut first_diffs: Vec<(f32, f32)> = Vec::new();
                    for i in 0..hidden as usize {
                        let d = crate::bring_up::f16_to_f32(reference[i]);
                        let p = crate::bring_up::f16_to_f32(probe[i]);
                        let diff = (d - p).abs();
                        if diff > max_abs {
                            max_abs = diff;
                        }
                        sum_sq_diff += (diff as f64) * (diff as f64);
                        sum_sq_ref += (d as f64) * (d as f64);
                        if first_diffs.len() < 4 {
                            first_diffs.push((d, p));
                        }
                    }
                    let rel_err = (sum_sq_diff / sum_sq_ref.max(1e-18)).sqrt();
                    eprintln!(
                        "[DIAG {label}] max_abs={max_abs:.4} rel_err={rel_err:.4e} \
                         first4_ref_probe={first_diffs:?}",
                    );
                };
                eprintln!(
                    "[DIAG] max_layers={} prompt_len={} hidden={}",
                    max_layers, prompt_len, hidden,
                );
                stats("row=0 (first token)", &decode_ref_first, &prefill_first);
                stats("row=N-1 (last token)", &decode_ref_last, &prefill_last);
            }
        }

        // LM head on last prompt token
        rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: 1,
            hidden,
            eps: arch.rms_norm_eps,
        }
        .launch(
            kernels.fused_rmsnorm,
            residual_ptr,
            self.model.final_norm.offset_bytes,
            stream,
        )?;
        self.cublaslt.f16_gemm_f32(
            residual_ptr,
            self.model.lm_head_f16.offset_bytes,
            logits_f32.device_ptr(),
            1,
            vocab as i32,
            hidden as i32,
            stream,
        )?;
        let mut host_tok = [0i32; 1];
        let first_raw = if let Some(tail) = sample_tail.as_mut() {
            // Sampled session: optional debug dumps for the statistical
            // gate, then sample the first token from the prefill logits.
            sample_debug_dumps(
                tail,
                logits_f32.device_ptr(),
                vocab,
                sampling,
                stream,
                &self.stream,
            )?;
            i64::from(sample_tail_step(
                tail,
                logits_f32.device_ptr(),
                vocab,
                sampling,
                stream,
                &self.stream,
            )?)
        } else {
            rvllm_fused::ArgmaxLaunch {
                num_tokens: 1,
                vocab,
            }
            .launch(
                fn_argmax,
                logits_f32.device_ptr(),
                sampled.device_ptr(),
                stream,
            )?;

            self.stream.fence()?;
            crate::bring_up::dtoh_sync_checked(
                sampled.device_ptr(),
                host_tok.as_mut_ptr().cast(),
                4,
                stream,
            )?;
            i64::from(host_tok[0])
        };
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[prefill] {} tokens in {:.1}ms (TTFT={:.1}ms)",
            prompt_ids.len(),
            prefill_ms,
            prefill_ms
        );

        let mut output_ids: Vec<u32> = Vec::with_capacity(max_new);
        let first_id = map_sampled_token(first_raw, "gemma4_prefill_sample")?;
        output_ids.push(first_id);
        if eos_ids.contains(&first_id) {
            return Ok(output_ids);
        }

        // Phase 2: Decode new tokens.
        //
        // Two paths, selected by `RVLLM_DECODE_GRAPH` (DEFAULT = graph; set
        // RVLLM_DECODE_GRAPH=0 to force the legacy eager path):
        //   * graph (default): capture the pure-device chain `decode_forward`
        //     ONCE and `replay()` it each step. Between replays only the small
        //     eager metadata prep runs (positions + per-layer slot/ctx via
        //     ordered HtoD). Token feedback is device->device inside the graph;
        //     tokens are harvested DtoH every K steps.
        //   * eager (RVLLM_DECODE_GRAPH=0): re-issue the per-step kernel chain
        //     host-side every step; ~122 sync HtoD/step, host-serialized.
        let use_decode_graph = std::env::var("RVLLM_DECODE_GRAPH").ok().as_deref() != Some("0");

        if sampled_mode && max_new > 1 {
            // --- Sampled decode loop ------------------------------------
            // The captured graph is `decode_forward_logits` — the greedy
            // chain WITHOUT the argmax + DtoD token-feedback tail. Each
            // step: replay to logits, then eagerly run the sampling tail
            // (selection kernel + ~8 KB DtoH + host draw) and feed the
            // drawn token back with one small sync HtoD. The per-step
            // fence the tail needs anyway also makes eos checks immediate.
            let tail = sample_tail
                .as_mut()
                .expect("sampled_mode without sample_tail state");
            let prompt_len = prompt_ids.len();
            let mut stopped = false;
            if use_decode_graph {
                // Seed the device token buffer with the first sampled
                // token; step 0 runs eagerly (warms cuBLASLt's per-shape
                // algo cache exactly like the greedy arm), then capture.
                {
                    let seed = [output_ids[0] as i32];
                    token_ids_region.copy_from_host(bytemuck_cast_i32(&seed))?;
                }
                prepare_decode_inputs(prompt_len, &mut slot_host, &mut ctx_host)?;
                decode_forward_logits()?;
                let sampled_row = sample_tail_step(
                    tail,
                    logits_f32.device_ptr(),
                    vocab,
                    sampling,
                    stream,
                    &self.stream,
                )?;
                let tok = map_sampled_token(i64::from(sampled_row), "gemma4_sampled_decode")?;
                output_ids.push(tok);
                stopped = eos_ids.contains(&tok);

                let graph = if !stopped {
                    Some(rvllm_graph::CapturedGraph::capture(
                        &self.ctx,
                        1,
                        max_blocks_per_seq,
                        rvllm_metadata::MetadataLayout::compute(1, max_blocks_per_seq)?.hash(),
                        stream,
                        || decode_forward_logits(),
                    )?)
                } else {
                    None
                };
                if let Some(ref graph) = graph {
                    for decode_step in 1..max_new - 1 {
                        if stopped {
                            break;
                        }
                        // Feed the previous step's drawn token. Sync HtoD:
                        // the tail fenced last step, so nothing is in
                        // flight and 4 B is host-trivial.
                        let tok_i32 = [*output_ids.last().unwrap() as i32];
                        token_ids_region.copy_from_host(bytemuck_cast_i32(&tok_i32))?;
                        prepare_decode_inputs(
                            prompt_len + decode_step,
                            &mut slot_host,
                            &mut ctx_host,
                        )?;
                        graph.replay(stream)?;
                        let sampled_row = sample_tail_step(
                            tail,
                            logits_f32.device_ptr(),
                            vocab,
                            sampling,
                            stream,
                            &self.stream,
                        )?;
                        let tok =
                            map_sampled_token(i64::from(sampled_row), "gemma4_sampled_decode")?;
                        output_ids.push(tok);
                        stopped = eos_ids.contains(&tok);
                    }
                }
                // Drop the captured graph BEFORE the caller's
                // `arena.restore` — its recorded kernels reference arena
                // device pointers that restore would invalidate.
                drop(graph);
            } else {
                // RVLLM_DECODE_GRAPH=0 escape hatch: eager per-step chain,
                // mirroring the greedy eager arm with the sampled tail.
                for decode_step in 0..max_new - 1 {
                    if stopped {
                        break;
                    }
                    let tok_id = *output_ids.last().unwrap();
                    run_one_token(
                        tok_id,
                        prompt_len + decode_step,
                        (&mut slot_host, &mut ctx_host),
                    )?;
                    rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                        num_tokens: 1,
                        hidden,
                        eps: arch.rms_norm_eps,
                    }
                    .launch(
                        kernels.fused_rmsnorm,
                        residual_ptr,
                        self.model.final_norm.offset_bytes,
                        stream,
                    )?;
                    self.cublaslt.f16_gemm_f32(
                        residual_ptr,
                        self.model.lm_head_f16.offset_bytes,
                        logits_f32.device_ptr(),
                        1,
                        vocab as i32,
                        hidden as i32,
                        stream,
                    )?;
                    let sampled_row = sample_tail_step(
                        tail,
                        logits_f32.device_ptr(),
                        vocab,
                        sampling,
                        stream,
                        &self.stream,
                    )?;
                    let tok = map_sampled_token(i64::from(sampled_row), "gemma4_sampled_decode")?;
                    output_ids.push(tok);
                    stopped = eos_ids.contains(&tok);
                }
            }
        } else if spec_decode && max_new > 1 {
            // --- Speculative decode loop (greedy, lossless) -------------
            // Each iteration: draft up to spec_k tokens by n-gram prompt
            // lookup over the committed stream, verify [last, drafts...]
            // in ONE forward, accept the longest draft prefix matching
            // the model's own argmaxes, emit accepted + 1 bonus token.
            // Every emitted token is a model argmax — the stream is
            // exactly greedy decode, drafts only change how many forwards
            // it takes to produce it.
            let mut all_tokens: Vec<u32> = Vec::with_capacity(prompt_ids.len() + max_new);
            all_tokens.extend_from_slice(prompt_ids);
            all_tokens.push(output_ids[0]);
            // Tokens fed into the model so far == next input position.
            let mut cur = prompt_ids.len();
            let max_pos = generation_capacity;
            let mut spec_steps = 0usize;
            let mut spec_drafted = 0usize;
            let mut spec_accept = 0usize;
            let mut host_out = vec![0i32; spec_n_max as usize];
            let mut stopped = false;
            // S3: graph-capture the fixed-shape device chains. One graph for
            // the n=1 decode step plus one per
            // verify chunk size n — captured lazily on each size's first
            // occurrence (which runs eagerly and warms cuBLASLt's algo cache
            // for the M=n shapes), replayed on every later occurrence.
            // Guarded to Fa3 backends: the non-Fa3 fallback arms stage host
            // data inside the forward, which is not capture-safe.
            // RVLLM_SPEC_GRAPH=0 forces the eager path.
            let spec_graph_enabled = use_decode_graph
                && std::env::var("RVLLM_SPEC_GRAPH").ok().as_deref() != Some("0")
                && matches!(
                    self.sliding_attention,
                    rvllm_attention::AttentionBackend::Fa3(_)
                )
                && matches!(
                    self.global_attention,
                    rvllm_attention::AttentionBackend::Fa3(_)
                );
            let mut spec_graphs: std::collections::HashMap<u32, rvllm_graph::CapturedGraph> =
                std::collections::HashMap::new();
            let mut spec_decode_graph: Option<rvllm_graph::CapturedGraph> = None;
            while !stopped && output_ids.len() < max_new {
                let remaining = max_new - output_ids.len();
                // Budget: <= remaining-1 so a fully-accepted step (a+1
                // emits) can never overshoot max_new; <= KV headroom.
                let budget = (spec_k as usize)
                    .min(remaining.saturating_sub(1))
                    .min(max_pos.saturating_sub(cur + 1));
                let draft = if budget == 0 {
                    Vec::new()
                } else {
                    ngram_draft(&all_tokens, budget, spec_ngram_max)
                };
                let last_tok = *output_ids.last().unwrap() as i32;
                let n = 1 + draft.len();
                if n == 1 {
                    // No draft found: a plain decode step through the
                    // same chain the graph path captures.
                    crate::bring_up::htod_ordered(
                        token_ids_region.device_ptr(),
                        bytemuck_cast_i32(&[last_tok]),
                        stream,
                    )?;
                    prepare_decode_inputs(cur, &mut slot_host, &mut ctx_host)?;
                    if let Some(g) = spec_decode_graph.as_ref() {
                        g.replay(stream)?;
                    } else {
                        decode_forward()?;
                        if spec_graph_enabled {
                            spec_decode_graph = Some(rvllm_graph::CapturedGraph::capture(
                                &self.ctx,
                                1,
                                max_blocks_per_seq,
                                rvllm_metadata::MetadataLayout::compute(1, max_blocks_per_seq)?
                                    .hash(),
                                stream,
                                || decode_forward(),
                            )?);
                        }
                    }
                } else {
                    let mut chunk: Vec<i32> = Vec::with_capacity(n);
                    chunk.push(last_tok);
                    chunk.extend(draft.iter().map(|&t| t as i32));
                    spec_stage_metadata(&chunk, cur)?;
                    let nb = n as u32;
                    if let Some(g) = spec_graphs.get(&nb) {
                        g.replay(stream)?;
                    } else {
                        // First chunk of this size: run eagerly (warms the
                        // cuBLASLt algo cache for the M=n shapes — a
                        // heuristic sync inside capture would abort it),
                        // then record the graph for every later occurrence.
                        // Capture records WITHOUT executing, so device
                        // state stays exactly post-eager-run.
                        spec_verify_forward(nb, cur as u32)?;
                        if spec_graph_enabled {
                            let g = rvllm_graph::CapturedGraph::capture(
                                &self.ctx,
                                nb,
                                max_blocks_per_seq,
                                rvllm_metadata::MetadataLayout::compute(nb, max_blocks_per_seq)?
                                    .hash(),
                                stream,
                                || spec_verify_forward(nb, cur as u32),
                            )?;
                            spec_graphs.insert(nb, g);
                        }
                    }
                }
                self.stream.fence()?;
                crate::bring_up::dtoh_sync_checked(
                    sampled.device_ptr(),
                    host_out.as_mut_ptr().cast(),
                    n * core::mem::size_of::<i32>(),
                    stream,
                )?;
                // Greedy accept/reject. `host_out[0..n)` are the M=n
                // per-position target argmaxes (`sampled[0..n)` from the verify
                // forward, rows 0..K then the bonus row K). The acceptance is
                // the longest matching prefix of `draft` vs the target argmax,
                // plus a bonus token, fed through the shared
                // `rvllm_sampling::greedy_accept`.
                // `n == 1 + draft.len()`, so target_argmax = host_out[0..K],
                // bonus = host_out[K] (== host_out[draft.len()]).
                let verified: Vec<u32> = host_out[..n]
                    .iter()
                    .map(|&t| map_sampled_token(i64::from(t), "gemma4_spec_decode"))
                    .collect::<Result<_>>()?;
                let k = draft.len();
                let target_argmax = &verified[..k];
                let bonus = verified[k];
                let accepted = rvllm_sampling::greedy_accept(target_argmax, &draft, bonus)?;
                let a = (accepted.valid_count as usize).saturating_sub(1);
                // Emit the accepted tokens (all TARGET argmaxes + bonus).
                for &t in accepted.tokens.iter() {
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
                spec_k,
                spec_ngram_max
            );
        } else if !use_decode_graph {
            for decode_step in 0..max_new - 1 {
                let tok_id = *output_ids.last().unwrap();
                run_one_token(
                    tok_id,
                    prompt_ids.len() + decode_step,
                    (&mut slot_host, &mut ctx_host),
                )?;

                rvllm_fused::gemma4_launcher::RmsnormInplaceLaunch {
                    num_tokens: 1,
                    hidden,
                    eps: arch.rms_norm_eps,
                }
                .launch(
                    kernels.fused_rmsnorm,
                    residual_ptr,
                    self.model.final_norm.offset_bytes,
                    stream,
                )?;
                self.cublaslt.f16_gemm_f32(
                    residual_ptr,
                    self.model.lm_head_f16.offset_bytes,
                    logits_f32.device_ptr(),
                    1,
                    vocab as i32,
                    hidden as i32,
                    stream,
                )?;
                rvllm_fused::ArgmaxLaunch {
                    num_tokens: 1,
                    vocab,
                }
                .launch(
                    fn_argmax,
                    logits_f32.device_ptr(),
                    sampled.device_ptr(),
                    stream,
                )?;

                self.stream.fence()?;
                crate::bring_up::dtoh_sync_checked(
                    sampled.device_ptr(),
                    host_tok.as_mut_ptr().cast(),
                    4,
                    stream,
                )?;
                let next_id = map_sampled_token(i64::from(host_tok[0]), "gemma4_eager_decode")?;
                output_ids.push(next_id);
                if eos_ids.contains(&next_id) {
                    break;
                }
            }
        } else if max_new > 1 {
            // CUDA-graph decode fast path.
            let prompt_len = prompt_ids.len();
            // Seed the device token buffer with the first decoded token
            // (already in `output_ids[0]`); the graph's embed gather reads
            // from here, and its closing DtoD keeps it fed thereafter.
            {
                let seed = [output_ids[0] as i32];
                token_ids_region.copy_from_host(bytemuck_cast_i32(&seed))?;
            }
            // Step 0 runs eagerly: it both warms cuBLASLt's per-shape algo
            // cache (so the captured replay never re-runs a heuristic /
            // syncs) and is a real decode step. Its closing DtoD advances
            // `token_ids_region` to output_ids[1].
            prepare_decode_inputs(prompt_len, &mut slot_host, &mut ctx_host)?;
            decode_forward()?;
            self.stream.fence()?;
            crate::bring_up::dtoh_sync_checked(
                sampled.device_ptr(),
                host_tok.as_mut_ptr().cast(),
                4,
                stream,
            )?;
            let step0_id = map_sampled_token(i64::from(host_tok[0]), "gemma4_graph_warmup")?;
            output_ids.push(step0_id);
            let mut stopped = eos_ids.contains(&step0_id);

            // Capture the device chain once (cuBLASLt now warm). Bucket/
            // max_blocks/layout-hash mirror run_bench/run_ppl (batch=1).
            let graph = if !stopped {
                Some(rvllm_graph::CapturedGraph::capture(
                    &self.ctx,
                    1,
                    max_blocks_per_seq,
                    rvllm_metadata::MetadataLayout::compute(1, max_blocks_per_seq)?.hash(),
                    stream,
                    || decode_forward(),
                )?)
            } else {
                None
            };
            // The capture closure issued no real work (capture records, does
            // not execute), so the device token buffer still holds
            // output_ids[1] from the eager step 0. Good to start replaying.

            // Token harvest WITHOUT a per-step host stall. After each replay
            // we enqueue a small async DtoH of that step's argmax into its
            // OWN host slot (`host_tokens[s]`), all on the engine stream.
            // Same-stream ordering guarantees slot `s` captures step `s`'s
            // token before the next replay's DtoD overwrites `sampled`. We
            // only `fence()` every HARVEST_EVERY steps to (a) make the slots
            // readable for the eos check and (b) bound in-flight async
            // copies. This is bit-exact (every token recorded in order) yet
            // avoids the eager path's fence+blocking DtoH every step.
            const HARVEST_EVERY: usize = 16;
            // Pinned storage keeps the asynchronous DtoH copies nonblocking and
            // valid until each harvest fence completes.
            let mut host_tokens: PinnedBuf<i32> = PinnedBuf::new(max_new)?;

            if let Some(ref graph) = graph {
                // decode_step s in [1, max_new-1): replay produces the token
                // stored at output index s+1. Track the inclusive range of
                // steps whose async DtoH has been enqueued but not yet
                // fenced/drained into `output_ids`.
                let mut drained_upto: usize = 0; // last decode_step drained
                for decode_step in 1..max_new - 1 {
                    if stopped {
                        break;
                    }
                    prepare_decode_inputs(prompt_len + decode_step, &mut slot_host, &mut ctx_host)?;
                    graph.replay(stream)?;
                    // Enqueue async DtoH of this step's token into its slot.
                    crate::bring_up::dtoh_async_sync(
                        sampled.device_ptr(),
                        unsafe { host_tokens.as_mut_ptr().add(decode_step) },
                        4,
                        stream,
                    )?;

                    let last_step = decode_step == max_new - 2;
                    if decode_step - drained_upto >= HARVEST_EVERY || last_step {
                        // Make all enqueued slots [drained_upto+1, decode_step]
                        // host-visible, then drain them into output_ids in
                        // order, checking eos.
                        self.stream.fence()?;
                        for s in (drained_upto + 1)..=decode_step {
                            let next_id = map_sampled_token(
                                i64::from(host_tokens.as_slice()[s]),
                                "gemma4_graph_harvest",
                            )?;
                            output_ids.push(next_id);
                            if eos_ids.contains(&next_id) {
                                stopped = true;
                                break;
                            }
                        }
                        drained_upto = decode_step;
                    }
                }
            }
            // Drop the captured graph BEFORE the caller's `arena.restore`
            // (which run_generate runs after this returns): the graph's
            // recorded kernels reference arena device pointers that restore
            // would invalidate.
            drop(graph);
        }

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let decode_ms = total_ms - prefill_ms;
        let decoded_tokens = output_ids.len().saturating_sub(1);
        eprintln!(
            "[generate] {} tokens decoded in {:.1}ms ({:.1} tok/s)",
            decoded_tokens,
            decode_ms,
            decoded_tokens as f64 / (decode_ms / 1000.0).max(1e-9)
        );
        Ok(output_ids)
    }

    pub fn layer_kernels(&self) -> Gemma4LayerKernels<'_> {
        Gemma4LayerKernels {
            fused_rmsnorm: &self.fused.fn_rmsnorm,
            fused_rmsnorm_fp8_quant: &self.fused.fn_rmsnorm_fp8_quant,
            fused_qk_rmsnorm: &self.fused.fn_qk_rmsnorm,
            fused_rope_partial_fp8kv: &self.fused.fn_rope_partial_fp8kv,
            fused_gelu_mul: &self.fused.fn_gelu_mul,
            quantize_fp8_per_token: &self.fused.fn_quantize,
            residual_scale_f16: &self.fused.fn_residual_scale,
            residual_scale_bf16s: &self.fused.fn_residual_scale_bf16s,
            vnorm_f16: &self.fused.fn_vnorm,
            vector_add_f16: &self.fused.fn_vector_add,
            bf16_to_f16_sat: &self.fused.fn_bf16_to_f16_sat,
            rmsnorm_inplace_bf16: &self.fused.fn_rmsnorm_inplace_bf16,
            vector_add_bf16_to_f16: &self.fused.fn_vector_add_bf16_to_f16,
            f32_to_bf16: &self.fused.fn_f32_to_bf16,
            f32_to_f16_sat: &self.fused.fn_f32_to_f16_sat,
            scale_cols_f32: &self.fused.fn_scale_cols_f32,
            scale_rows_f32_ratio: &self.fused.fn_scale_rows_f32_ratio,
            scale_rows_f16_pertoken: &self.fused.fn_scale_rows_f16_pertoken,
            compute_qkv_scales: &self.fused.fn_compute_qkv_scales,
            fused_gelu_mul_f16: &self.fused.fn_fused_gelu_mul_f16,
            fused_rope_partial_f16kv: &self.fused.fn_fused_rope_partial_f16kv,
            fused_norm_add_residual: &self.fused.fn_fused_norm_add_residual,
            fused_norm_add_residual_f16: &self.fused.fn_fused_norm_add_residual_f16,
            fused_norm_add_residual_f16in: &self.fused.fn_fused_norm_add_residual_f16in,
            fused_qkv_rmsnorm: &self.fused.fn_fused_qkv_rmsnorm,
            scale_cols_f16: &self.fused.fn_scale_cols_f16,
            ple_gelu_mul_f16: &self.fused.fn_ple_gelu_mul_f16,
            fp8_channelscale_gemv_ktiled: &self.fused.fn_fp8_channelscale_gemv_ktiled,
            fp8_channelscale_gemv_splitk: &self.fused.fn_fp8_channelscale_gemv_splitk,
            fp8_gemv_wpr_native_f16in: self.fused.fn_fp8_gemv_wpr_native_f16in.as_ref(),
            ple_gate: self.fused.fn_ple_gate.as_ref(),
        }
    }
}

fn bytemuck_cast_i32(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// RVLLM_SAMPLE_T / RVLLM_SAMPLE_TOPK / RVLLM_SAMPLE_TOPP /
/// RVLLM_SAMPLE_SEED: debug/bench override of the caller's sampling params
/// (lets `rvllm-bench --generate` drive the sampled tail with zero bench
/// plumbing). Unset vars keep the caller's values; serve passes per-request
/// params and never sets these.
#[cfg(feature = "cuda")]
fn resolve_sampling_env(mut s: SamplingParams) -> SamplingParams {
    fn env<T: std::str::FromStr>(name: &str) -> Option<T> {
        std::env::var(name).ok().and_then(|v| v.parse().ok())
    }
    if let Some(t) = env::<f32>("RVLLM_SAMPLE_T") {
        s.temperature = t;
    }
    if let Some(k) = env::<u32>("RVLLM_SAMPLE_TOPK") {
        s.top_k = k;
    }
    if let Some(p) = env::<f32>("RVLLM_SAMPLE_TOPP") {
        s.top_p = p;
    }
    if let Some(seed) = env::<u64>("RVLLM_SAMPLE_SEED") {
        s.seed = seed;
    }
    s
}

/// Per-request sampled-tail state: device candidate buffers (arena-backed,
/// raw pointers — `Region` drop is a no-op, the memory lives until
/// `run_generate_sampled`'s `arena.restore`), the selection kernel, pinned
/// DtoH targets, the candidate scratch, and the seeded draw stream.
#[cfg(feature = "cuda")]
struct SampleTailState {
    vals_ptr: u64,
    idx_ptr: u64,
    cnt_ptr: u64,
    candidate_capacity: u32,
    fn_sample: KernelFn,
    _module: LoadedModule,
    pin_vals: PinnedBuf<i32>,
    pin_idx: PinnedBuf<i32>,
    pin_cnt: PinnedBuf<i32>,
    cands: Vec<(f32, u32)>,
    rng: rvllm_sampling::SampleRng,
}

/// One sampled-tail step: selection kernel on the logits row, fence, DtoH
/// the K' compact candidates, host draw. Called once per decode step (and
/// N times by the RVLLM_SAMPLE_STAT harness). The count check doubles as a
/// per-step kernel-determinism assert.
#[cfg(feature = "cuda")]
unsafe fn sample_tail_step(
    t: &mut SampleTailState,
    logits_ptr: u64,
    vocab: u32,
    sampling: SamplingParams,
    stream: u64,
    stream_h: &Stream,
) -> Result<u32> {
    let k = sampling.kernel_k(vocab)?;
    rvllm_sampling::SampleTopKLaunch {
        vocab,
        k_select: k,
        out_capacity: t.candidate_capacity,
        inv_temp: 1.0 / sampling.temperature,
    }
    .launch(
        &t.fn_sample,
        logits_ptr,
        t.vals_ptr,
        t.idx_ptr,
        t.cnt_ptr,
        stream,
    )?;
    crate::bring_up::dtoh_async_sync(t.vals_ptr, t.pin_vals.as_mut_ptr(), k as usize * 4, stream)?;
    crate::bring_up::dtoh_async_sync(t.idx_ptr, t.pin_idx.as_mut_ptr(), k as usize * 4, stream)?;
    crate::bring_up::dtoh_async_sync(t.cnt_ptr, t.pin_cnt.as_mut_ptr(), 4, stream)?;
    stream_h.fence()?;
    let n = t.pin_cnt.as_slice()[0];
    if n != k as i32 {
        return Err(rvllm_core::RvllmError::Sampling {
            err: rvllm_core::SamplingError::InvalidParams {
                reason: format!("sample_topk_f32 selected {n} candidates, expected {k}"),
            },
            ctx: rvllm_core::SampleCtx {
                op: "sample_tail_step",
                stream,
            },
        });
    }
    let vals = t.pin_vals.as_slice();
    let idx = t.pin_idx.as_slice();
    decode_device_candidates(vals, idx, k as usize, vocab, stream, &mut t.cands)?;
    rvllm_sampling::sampler::sample_from_candidates_in_vocab(
        &mut t.cands,
        vocab,
        sampling.top_k,
        sampling.top_p,
        &mut t.rng,
    )
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn validate_sampled_token(
    raw_id: i64,
    vocab: u32,
    op: &'static str,
    stream: u64,
) -> Result<u32> {
    if raw_id < 0 || raw_id >= i64::from(vocab) {
        return Err(rvllm_core::RvllmError::Sampling {
            err: rvllm_core::SamplingError::InvalidParams {
                reason: format!("sampled token id {raw_id} is outside [0, {vocab})"),
            },
            ctx: rvllm_core::SampleCtx { op, stream },
        });
    }
    Ok(raw_id as u32)
}

#[cfg(any(feature = "cuda", test))]
fn decode_device_candidates(
    values: &[i32],
    indices: &[i32],
    count: usize,
    vocab: u32,
    stream: u64,
    output: &mut Vec<(f32, u32)>,
) -> Result<()> {
    if vocab == 0 || count == 0 || count > values.len() || count > indices.len() {
        return Err(rvllm_core::RvllmError::Sampling {
            err: rvllm_core::SamplingError::InvalidParams {
                reason: "device candidate buffers have invalid extents".into(),
            },
            ctx: rvllm_core::SampleCtx {
                op: "decode_device_candidates",
                stream,
            },
        });
    }
    for &index in &indices[..count] {
        if index < 0 || index as u32 >= vocab {
            return Err(rvllm_core::RvllmError::Sampling {
                err: rvllm_core::SamplingError::InvalidParams {
                    reason: format!("device candidate token id {index} is outside [0, {vocab})"),
                },
                ctx: rvllm_core::SampleCtx {
                    op: "decode_device_candidates",
                    stream,
                },
            });
        }
    }
    output.clear();
    output.extend(
        values[..count]
            .iter()
            .zip(&indices[..count])
            .map(|(&value, &index)| (f32::from_bits(value as u32), index as u32)),
    );
    Ok(())
}

/// Debug harnesses for the statistical sampling gate; both OFF unless their
/// env var is set, and only reachable in sampled mode.
///   RVLLM_SAMPLE_DUMP_LOGITS=path  raw f32 LE dump of the first sampled
///                                  step's logits row (reference softmax).
///   RVLLM_SAMPLE_STAT=N            draw N times from that fixed row
///                                  through the full tail (kernel + DtoH +
///                                  host draw each time) and print the
///                                  empirical histogram head.
#[cfg(feature = "cuda")]
unsafe fn sample_debug_dumps(
    t: &mut SampleTailState,
    logits_ptr: u64,
    vocab: u32,
    sampling: SamplingParams,
    stream: u64,
    stream_h: &Stream,
) -> Result<()> {
    if let Ok(path) = std::env::var("RVLLM_SAMPLE_DUMP_LOGITS") {
        let mut host: Vec<i32> = vec![0; vocab as usize];
        stream_h.fence()?;
        crate::bring_up::dtoh_sync_checked(
            logits_ptr,
            host.as_mut_ptr().cast(),
            vocab as usize * 4,
            stream,
        )?;
        let mut bytes = Vec::with_capacity(vocab as usize * 4);
        for v in &host {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, bytes).map_err(|source| rvllm_core::RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: std::path::PathBuf::from(&path),
            source,
        })?;
        eprintln!("[sample] dumped {vocab} f32 logits to {path}");
    }
    if let Some(n) = std::env::var("RVLLM_SAMPLE_STAT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for _ in 0..n {
            let tok = sample_tail_step(t, logits_ptr, vocab, sampling, stream, stream_h)?;
            *counts.entry(tok).or_insert(0) += 1;
        }
        // `cands` is canonically sorted after the last draw: [0] is argmax.
        let argmax_tok = t.cands[0].1;
        let argmax_n = counts.get(&argmax_tok).copied().unwrap_or(0);
        let mut top: Vec<(u32, usize)> = counts.into_iter().collect();
        top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        top.truncate(8);
        eprintln!(
            "[sample-stat] n={} argmax_token={} argmax_count={} argmax_freq={:.6} top8={:?}",
            n,
            argmax_tok,
            argmax_n,
            argmax_n as f64 / n as f64,
            top
        );
    }
    Ok(())
}

/// Prompt-lookup n-gram drafter for speculative decode: propose the
/// continuation of the most recent prior occurrence of the stream's
/// suffix, preferring longer suffix keys. Zero model, zero training —
/// host-side and trivially cheap next to a 31B forward. Returns up to
/// `max_draft` tokens (empty when no suffix recurs, which falls back to
/// a plain decode step).
#[cfg(feature = "cuda")]
pub(crate) fn ngram_draft(tokens: &[u32], max_draft: usize, max_ngram: usize) -> Vec<u32> {
    let len = tokens.len();
    if max_draft == 0 || len < 2 {
        return Vec::new();
    }
    for n in (1..=max_ngram.min(len - 1)).rev() {
        let key = &tokens[len - n..];
        // Rightmost (most recent) prior occurrence; `start < len - n`
        // keeps the key from matching itself and guarantees at least
        // one continuation token exists.
        for start in (0..len - n).rev() {
            if &tokens[start..start + n] == key {
                let cont_start = start + n;
                let cont_end = (cont_start + max_draft).min(len);
                return tokens[cont_start..cont_end].to_vec();
            }
        }
    }
    Vec::new()
}

/// Host-side IEEE-754 binary32 → binary16 (round-to-nearest, ties-to-even).
/// NaN preserved as quiet NaN. Saturates on overflow to +/-inf.
#[cfg(feature = "cuda")]
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;

    // NaN / Inf
    if exp == 0xff {
        if mant != 0 {
            // qNaN
            return (sign << 15) | 0x7e00;
        }
        return (sign << 15) | 0x7c00;
    }

    // Unbiased exponent for f16: subtract 127, add 15.
    let new_exp = exp - 127 + 15;

    if new_exp >= 0x1f {
        // Overflow → inf
        return (sign << 15) | 0x7c00;
    }
    if new_exp <= 0 {
        // Subnormal or zero
        if new_exp < -10 {
            return sign << 15;
        }
        // Round mantissa with implicit leading 1, then shift right.
        let m = mant | 0x0080_0000;
        let shift = (1 - new_exp) + 13;
        let rounded = (m + (1u32 << (shift - 1))) >> shift;
        return (sign << 15) | (rounded as u16);
    }

    // Normal: shift 23-bit mantissa down to 10 bits with round-to-nearest-even.
    let half_round = (mant + 0x0000_1000) >> 13;
    // Mantissa overflow into exponent
    if half_round & 0x0400 != 0 {
        let exp_bump = new_exp + 1;
        if exp_bump >= 0x1f {
            return (sign << 15) | 0x7c00;
        }
        return (sign << 15) | ((exp_bump as u16) << 10);
    }
    (sign << 15) | ((new_exp as u16) << 10) | (half_round as u16)
}

/// Inject pre-computed F16 image embeddings into the per-token embedding
/// buffer at the positions named in `image_embeds`. The buffer is the
/// `[seq_len, hidden]` F16 region produced by the embedding-gather kernel
/// (`residual_ptr` in `run_generate_inner`). Each `(pos, vec)` pair
/// overwrites the row at `pos` with the f32→f16 cast of `vec`. Out-of-range
/// `pos` or wrong-length `vec` is an error (no silent truncation).
///
/// Each row is cast on the host and copied with the checked ordered HtoD
/// helper. The helper consumes the pageable row before returning.
#[cfg(feature = "cuda")]
pub(crate) unsafe fn inject_image_embeds_f16(
    residual_ptr: u64,
    hidden: u32,
    seq_offset: usize,
    seq_len: usize,
    hidden_bytes_f16: u64,
    image_embeds: &[(usize, Vec<f32>)],
    stream: u64,
) -> Result<()> {
    let hidden_usize = hidden as usize;
    if image_embeds.is_empty() {
        return Ok(());
    }
    // Filter and validate first so conversion errors cannot leave a partial
    // image injection in device memory.
    let mut work: Vec<(usize, Vec<u16>)> = Vec::with_capacity(image_embeds.len());
    for (pos, embed) in image_embeds {
        if *pos < seq_offset || *pos >= seq_offset + seq_len {
            continue;
        }
        if embed.len() != hidden_usize {
            return Err(rvllm_core::RvllmError::Sampling {
                err: rvllm_core::SamplingError::InvalidParams {
                    reason: format!(
                        "image_embed at pos {} has {} elems, expected hidden_size {}",
                        pos,
                        embed.len(),
                        hidden_usize
                    ),
                },
                ctx: rvllm_core::SampleCtx {
                    op: "inject_image_embed",
                    stream: stream as u64,
                },
            });
        }
        let mut row_f16 = vec![0u16; hidden_usize];
        for (i, &v) in embed.iter().enumerate() {
            row_f16[i] = f32_to_f16_bits(v);
        }
        work.push((*pos, row_f16));
    }
    for (pos, row_f16) in &work {
        let row = (*pos - seq_offset) as u64;
        let dst = residual_ptr + row * hidden_bytes_f16;
        let row_bytes = core::slice::from_raw_parts(
            row_f16.as_ptr().cast::<u8>(),
            hidden_usize * core::mem::size_of::<u16>(),
        );
        crate::bring_up::htod_ordered(dst, row_bytes, stream)?;
    }
    Ok(())
}

fn load_gemma4_fused(
    loader: &KernelLoader,
    target: Option<rvllm_core::CompileTarget>,
) -> Result<Gemma4FusedModules> {
    let rmsnorm_mod = loader.load_ptx("fused_rmsnorm_fp8_quant")?;
    let rope_mod = loader.load_ptx("fused_rope_partial_fp8kv")?;
    let gelu_mod = loader.load_ptx("fused_gelu_mul_fp8_quant")?;
    let argmax_mod = loader.load_ptx("argmax")?;
    let qk_norm_mod = loader.load_ptx("fused_qk_rmsnorm")?;
    let softcap_mod = loader.load_ptx("logit_softcap")?;
    let residual_scale_bf16s_mod = loader.load_ptx("residual_scale_bf16s_f16")?;
    let fn_residual_scale_bf16s =
        residual_scale_bf16s_mod.get_function("residual_scale_bf16s_f16_kernel")?;
    let residual_scale_mod = loader.load_ptx("residual_scale_f16")?;
    let vnorm_mod = loader.load_ptx("vnorm_f16")?;
    let vector_add_mod = loader.load_ptx("vector_add_f16")?;
    let bf16_to_f16_sat_mod = loader.load_ptx("bf16_to_f16_sat")?;
    let rmsnorm_inplace_bf16_mod = loader.load_ptx("rmsnorm_inplace_bf16")?;
    let vector_add_bf16_to_f16_mod = loader.load_ptx("vector_add_bf16_to_f16")?;
    let f32_to_bf16_mod = loader.load_ptx("f32_to_bf16")?;
    let f32_to_f16_sat_mod = loader.load_ptx("f32_to_f16_sat")?;

    let rmsnorm_inplace_mod = loader.load_ptx("rmsnorm_inplace_f16")?;
    let fn_rmsnorm = rmsnorm_inplace_mod.get_function("rmsnorm_inplace_f16_kernel")?;
    let fn_rmsnorm_fp8_quant = rmsnorm_mod.get_function("fused_rmsnorm_fp8_quant_kernel")?;
    let fn_quantize = rmsnorm_mod.get_function("quantize_fp8_per_token_kernel")?;
    let fn_rope_partial_fp8kv = rope_mod.get_function("fused_rope_partial_fp8kv_kernel")?;
    let fn_gelu_mul = gelu_mod.get_function("fused_gelu_mul_fp8_quant_kernel")?;
    let fn_argmax = load_argmax_kernel(&argmax_mod, target)?;
    let fn_qk_rmsnorm = qk_norm_mod.get_function("fused_qk_rmsnorm_kernel")?;
    let fn_softcap = softcap_mod.get_function("logit_softcap_kernel")?;
    let fn_residual_scale = residual_scale_mod.get_function("residual_scale_f16_kernel")?;
    let fn_vnorm = vnorm_mod.get_function("vnorm_f16_kernel")?;
    let fn_vector_add = vector_add_mod.get_function("vector_add_f16_kernel")?;
    let fn_bf16_to_f16_sat = bf16_to_f16_sat_mod.get_function("bf16_to_f16_sat_kernel")?;
    let fn_rmsnorm_inplace_bf16 =
        rmsnorm_inplace_bf16_mod.get_function("rmsnorm_inplace_bf16_kernel")?;
    let fn_vector_add_bf16_to_f16 =
        vector_add_bf16_to_f16_mod.get_function("vector_add_bf16_to_f16_kernel")?;
    let fn_f32_to_bf16 = f32_to_bf16_mod.get_function("f32_to_bf16_kernel")?;
    let fn_f32_to_f16_sat = f32_to_f16_sat_mod.get_function("f32_to_f16_sat_kernel")?;

    let scale_cols_f32_mod = loader.load_ptx("scale_cols_f32")?;
    let fn_scale_cols_f32 = scale_cols_f32_mod.get_function("scale_cols_f32_kernel")?;
    let scale_rows_f32_ratio_mod = loader.load_ptx("scale_rows_f32_ratio")?;
    let fn_scale_rows_f32_ratio =
        scale_rows_f32_ratio_mod.get_function("scale_rows_f32_ratio_kernel")?;
    let scale_rows_f16_pertoken_mod = loader.load_ptx("scale_rows_f16_pertoken")?;
    let fn_scale_rows_f16_pertoken =
        scale_rows_f16_pertoken_mod.get_function("scale_rows_f16_pertoken_kernel")?;

    let compute_qkv_scales_mod = loader.load_ptx("compute_qkv_scales")?;
    let fn_compute_qkv_scales = compute_qkv_scales_mod.get_function("compute_qkv_scales_kernel")?;

    let fused_gelu_mul_f16_mod = loader.load_ptx("fused_gelu_mul_f16")?;
    let fn_fused_gelu_mul_f16 = fused_gelu_mul_f16_mod.get_function("fused_gelu_mul_f16_kernel")?;

    let fused_rope_partial_f16kv_mod = loader.load_ptx("fused_rope_partial_f16kv")?;
    let fn_fused_rope_partial_f16kv =
        fused_rope_partial_f16kv_mod.get_function("fused_rope_partial_f16kv_kernel")?;

    // `fp8_gemv.ptx` — see struct docs. The f16-input native-CVT
    // entry is gated on `__CUDA_ARCH__ >= 1000` in
    // `kernels/fp8_gemv.cu`, so we only resolve it when
    // `Fp8GemvVariant::available_for(target)` says yes.
    let fp8_gemv_mod = loader.load_ptx(rvllm_kernels::FP8_GEMV_PTX_STEM)?;
    let fn_fp8_gemv_wpr_native_f16in = match target {
        Some(t) if rvllm_kernels::Fp8GemvVariant::WprNativeF16In.available_for(t) => {
            Some(rvllm_kernels::Fp8GemvVariant::WprNativeF16In.load_verified(loader, t)?)
        }
        _ => None,
    };

    let fused_norm_add_residual_mod = loader.load_ptx("fused_norm_add_residual")?;
    let fn_fused_norm_add_residual =
        fused_norm_add_residual_mod.get_function("fused_norm_add_residual_kernel")?;

    let fused_norm_add_residual_f16_mod = loader.load_ptx("fused_norm_add_residual_f16")?;
    let fn_fused_norm_add_residual_f16 =
        fused_norm_add_residual_f16_mod.get_function("fused_norm_add_residual_f16_kernel")?;
    let fn_fused_norm_add_residual_f16in =
        fused_norm_add_residual_f16_mod.get_function("fused_norm_add_residual_f16in_kernel")?;

    let fused_qkv_rmsnorm_mod = loader.load_ptx("fused_qkv_rmsnorm")?;
    let fn_fused_qkv_rmsnorm = fused_qkv_rmsnorm_mod.get_function("fused_qkv_rmsnorm_kernel")?;

    let scale_cols_f16_mod = loader.load_ptx("scale_cols_f16")?;
    let fn_scale_cols_f16 = scale_cols_f16_mod.get_function("scale_cols_f16_kernel")?;
    let map_token_id_mod = loader.load_ptx("map_token_id")?;
    let fn_map_token_id = map_token_id_mod.get_function("map_i32_token_id_kernel")?;
    let ple_project_combine_mod = loader.load_ptx("ple_project_combine")?;
    let fn_ple_project_combine =
        ple_project_combine_mod.get_function("ple_project_combine_f16_kernel")?;
    let ple_gelu_mul_f16_mod = loader.load_ptx("ple_gelu_mul_f16")?;
    let fn_ple_gelu_mul_f16 = ple_gelu_mul_f16_mod.get_function("ple_gelu_mul_f16_kernel")?;
    let fp8_channelscale_gemv_ktiled_mod = loader.load_ptx("fp8_channelscale_gemv_ktiled")?;
    let fn_fp8_channelscale_gemv_ktiled =
        fp8_channelscale_gemv_ktiled_mod.get_function("fp8_channelscale_gemv_ktiled_kernel")?;
    let fp8_channelscale_gemv_splitk_mod = loader.load_ptx("fp8_channelscale_gemv_splitk")?;
    let fn_fp8_channelscale_gemv_splitk =
        fp8_channelscale_gemv_splitk_mod.get_function("fp8_channelscale_gemv_splitk_kernel")?;

    // E4B PLE gate kernel — OPTIONAL. Absent from 31B / pre-E4B kernel
    // bundles, so a load failure is not fatal here; the loader's
    // `RVLLM_E4B_REQUIRE` twin hard-fails separately if the E4B path is
    // demanded but the kernel is missing. When present we resolve both
    // entry points (`gemma4_ple_gate_kernel` for the per-layer hot path
    // and `gemma4_ple_projection_combine_kernel` for the once-per-step
    // model-projection combine).
    let (ple_gate_mod, fn_ple_gate, fn_ple_projection_combine) =
        match loader.load_ptx("gemma4_ple_gate") {
            Ok(m) => {
                let g = m.get_function("gemma4_ple_gate_kernel").ok();
                let c = m.get_function("gemma4_ple_projection_combine_kernel").ok();
                (Some(m), g, c)
            }
            Err(_) => (None, None, None),
        };

    // E4B INT4 pruned lm-head + pack dequant kernels. Optional,
    // same gating discipline as the PLE gate kernel: absent on 31B bundles,
    // and `RVLLM_E4B_REQUIRE` / `RVLLM_LMHEAD_PRUNE_REQUIRE` hard-fail
    // separately if demanded but missing.
    let (
        lmhead_prune_mod,
        fn_dequant_pack_to_f16,
        fn_lmhead_int4_gemv,
        fn_lmhead_argmax_remap,
        fn_lmhead_scatter_fullvocab,
    ) = match loader.load_ptx("lmhead_prune_argmax") {
        Ok(m) => {
            let dq = m
                .get_function(crate::gemma4_int4::FN_DEQUANT_PACK_TO_F16)
                .ok();
            let gv = m.get_function(crate::gemma4_int4::FN_LMHEAD_INT4_GEMV).ok();
            let ar = m
                .get_function(crate::gemma4_int4::FN_LMHEAD_ARGMAX_REMAP)
                .ok();
            let sc = m
                .get_function(crate::gemma4_int4::FN_LMHEAD_SCATTER_FULLVOCAB)
                .ok();
            (Some(m), dq, gv, ar, sc)
        }
        Err(_) => (None, None, None, None, None),
    };

    // Embedding gather kernel — the E4B PLE combine reuses it to gather the
    // per-layer embed rows. Same module the serve worker resolves `fn_embed`
    // from. OPTIONAL here (loaded only when present) so the 31B bench/PPL
    // bring-up is unaffected if a bundle omits it; the E4B `run_ple_combine`
    // hard-fails when it is needed but absent.
    // Keep the module alive: `get_function` returns only a raw `CUfunction`,
    // and dropping the `LoadedModule` calls `cuModuleUnload`, which would
    // dangle that handle (use-after-free surfacing as INVALID_HANDLE once the
    // freed module slot is reused — e.g. by the w4a8 `.so` load in
    // `populate_e4b_int4`). Store the module in the struct, like every other.
    let embed_gather_mod = loader.load_ptx("embedding_gather_f16").ok();
    let fn_embed_gather = embed_gather_mod
        .as_ref()
        .and_then(|m| m.get_function("embedding_gather_f16_kernel").ok());

    Ok(Gemma4FusedModules {
        rmsnorm_mod,
        rmsnorm_inplace_mod,
        rope_mod,
        gelu_mod,
        argmax_mod,
        qk_norm_mod,
        softcap_mod,
        residual_scale_mod,
        residual_scale_bf16s_mod,
        vnorm_mod,
        vector_add_mod,
        bf16_to_f16_sat_mod,
        rmsnorm_inplace_bf16_mod,
        vector_add_bf16_to_f16_mod,
        f32_to_bf16_mod,
        f32_to_f16_sat_mod,
        scale_cols_f32_mod,
        scale_rows_f32_ratio_mod,
        scale_rows_f16_pertoken_mod,
        compute_qkv_scales_mod,
        fused_gelu_mul_f16_mod,
        fused_rope_partial_f16kv_mod,
        fused_norm_add_residual_mod,
        fn_rmsnorm,
        fn_rmsnorm_fp8_quant,
        fn_quantize,
        fn_rope_partial_fp8kv,
        fn_gelu_mul,
        fn_argmax,
        fn_qk_rmsnorm,
        fn_softcap,
        fn_residual_scale,
        fn_residual_scale_bf16s,
        fn_vnorm,
        fn_vector_add,
        fn_bf16_to_f16_sat,
        fn_rmsnorm_inplace_bf16,
        fn_vector_add_bf16_to_f16,
        fn_f32_to_bf16,
        fn_f32_to_f16_sat,
        fn_scale_cols_f32,
        fn_scale_rows_f32_ratio,
        fn_scale_rows_f16_pertoken,
        fn_compute_qkv_scales,
        fn_fused_gelu_mul_f16,
        fn_fused_rope_partial_f16kv,
        fn_fused_norm_add_residual,
        fn_fused_norm_add_residual_f16,
        fn_fused_norm_add_residual_f16in,
        fused_norm_add_residual_f16_mod,
        fn_fused_qkv_rmsnorm,
        fused_qkv_rmsnorm_mod,
        fn_scale_cols_f16,
        scale_cols_f16_mod,
        map_token_id_mod,
        fn_map_token_id,
        ple_project_combine_mod,
        fn_ple_project_combine,
        ple_gelu_mul_f16_mod,
        fn_ple_gelu_mul_f16,
        fp8_channelscale_gemv_ktiled_mod,
        fn_fp8_channelscale_gemv_ktiled,
        fp8_channelscale_gemv_splitk_mod,
        fn_fp8_channelscale_gemv_splitk,
        fp8_gemv_mod,
        fn_fp8_gemv_wpr_native_f16in,
        ple_gate_mod,
        fn_ple_gate,
        fn_ple_projection_combine,
        lmhead_prune_mod,
        fn_dequant_pack_to_f16,
        fn_lmhead_int4_gemv,
        fn_lmhead_argmax_remap,
        fn_lmhead_scatter_fullvocab,
        fn_embed_gather,
        embed_gather_mod,
    })
}

fn load_argmax_kernel(
    argmax_mod: &LoadedModule,
    target: Option<rvllm_core::CompileTarget>,
) -> Result<KernelFn> {
    let default = argmax_mod.get_function("argmax_kernel")?;
    if std::env::var("RVLLM_ARGMAX_GRID").ok().as_deref() != Some("1")
        || !matches!(target, Some(rvllm_core::CompileTarget::Sm90))
    {
        return Ok(default);
    }
    match argmax_mod.get_function("argmax_grid_f32_kernel") {
        Ok(grid) => Ok(grid),
        Err(_) => Ok(default),
    }
}

#[cfg(test)]
mod device_candidate_tests {
    use super::{decode_device_candidates, validate_sampled_token};

    #[test]
    fn validates_sampled_token_before_narrowing() {
        assert!(validate_sampled_token(-1, 8, "test", 7).is_err());
        assert!(validate_sampled_token(8, 8, "test", 7).is_err());
        assert_eq!(validate_sampled_token(7, 8, "test", 7).unwrap(), 7);
    }

    #[test]
    fn validates_signed_device_token_ids_before_cast() {
        let values = [1.0f32.to_bits() as i32, 2.0f32.to_bits() as i32];
        let mut output = Vec::new();

        assert!(decode_device_candidates(&values, &[-1, 2], 2, 8, 7, &mut output).is_err());
        assert!(output.is_empty());
        assert!(decode_device_candidates(&values, &[1, 8], 2, 8, 7, &mut output).is_err());
        assert!(output.is_empty());

        decode_device_candidates(&values, &[1, 7], 2, 8, 7, &mut output).unwrap();
        assert_eq!(output, vec![(1.0, 1), (2.0, 7)]);
    }

    #[test]
    fn rejects_invalid_candidate_extents() {
        let values = [1.0f32.to_bits() as i32];
        let mut output = Vec::new();
        assert!(decode_device_candidates(&values, &[0], 0, 1, 0, &mut output).is_err());
        assert!(decode_device_candidates(&values, &[0], 2, 1, 0, &mut output).is_err());
        assert!(decode_device_candidates(&values, &[0], 1, 0, 0, &mut output).is_err());
    }
}

#[cfg(all(test, feature = "cuda"))]
mod spec_decode_tests {
    use super::ngram_draft;

    #[test]
    fn no_draft_when_no_suffix_recurs() {
        assert!(ngram_draft(&[1, 2, 3, 4, 5], 4, 3).is_empty());
        assert!(ngram_draft(&[7], 4, 3).is_empty());
        assert!(ngram_draft(&[], 4, 3).is_empty());
    }

    #[test]
    fn zero_budget_drafts_nothing() {
        assert!(ngram_draft(&[1, 2, 1, 2], 0, 3).is_empty());
    }

    #[test]
    fn proposes_continuation_of_prior_occurrence() {
        // suffix [8, 9] previously occurred followed by 10, 11, 12.
        let t = [8, 9, 10, 11, 12, 5, 8, 9];
        assert_eq!(ngram_draft(&t, 3, 3), vec![10, 11, 12]);
        assert_eq!(ngram_draft(&t, 2, 3), vec![10, 11]);
    }

    #[test]
    fn prefers_longest_suffix_key() {
        // 1-gram [4] recurs with continuation 99, but the longer
        // 2-gram [3, 4] also recurs — its continuation 7 must win.
        let t = [3, 4, 7, 4, 99, 3, 4];
        assert_eq!(ngram_draft(&t, 1, 3), vec![7]);
    }

    #[test]
    fn prefers_most_recent_occurrence() {
        // key [5] occurs twice; the later occurrence continues with 2.
        let t = [5, 1, 5, 2, 5];
        assert_eq!(ngram_draft(&t, 1, 1), vec![2]);
    }

    #[test]
    fn continuation_may_include_current_suffix() {
        // Repetition loops: [1,2,1,2,...] — key [1,2] at start continues
        // [1, 2], reaching into the present suffix (classic prompt-lookup).
        let t = [1, 2, 1, 2];
        assert_eq!(ngram_draft(&t, 4, 3), vec![1, 2]);
    }
}

#[cfg(test)]
mod e4b_kv_share_tests {
    use super::gemma4_kv_share_src;
    use rvllm_loader::gemma4_arch::Gemma4LayerType::{GlobalAttention as G, SlidingAttention as S};

    fn mixed_layer_types() -> Vec<rvllm_loader::gemma4_arch::Gemma4LayerType> {
        vec![S, G, S, S, G, S, G, S]
    }

    #[test]
    fn shared_layers_use_latest_owned_source_of_same_type() {
        let lt = mixed_layer_types();
        // Eight layers with three shared leaves layers 0..4 owning KV.
        for i in 0..5 {
            assert_eq!(
                gemma4_kv_share_src(&lt, 3, i),
                None,
                "layer {i} should own KV"
            );
        }
        assert_eq!(gemma4_kv_share_src(&lt, 3, 5), Some(3));
        assert_eq!(gemma4_kv_share_src(&lt, 3, 6), Some(4));
        assert_eq!(gemma4_kv_share_src(&lt, 3, 7), Some(3));
    }

    #[test]
    fn no_share_when_zero_shared_layers() {
        let lt = mixed_layer_types();
        for i in 0..lt.len() {
            assert_eq!(gemma4_kv_share_src(&lt, 0, i), None);
        }
    }
}

#[cfg(test)]
mod metadata_staging_tests {
    use super::{
        fill_ppl_layer_metadata, fill_prefill_chunk_metadata, generation_prefill_plan,
        validate_generation_capacity,
    };
    use rvllm_loader::gemma4_arch::Gemma4LayerType::{GlobalAttention as G, SlidingAttention as S};

    #[test]
    fn ppl_metadata_preserves_wrap_and_kv_share() {
        let layer_types = [S, G, S];
        let kv_share_targets = [None, None, Some(0)];
        let mut slots = [0; 3];
        let mut contexts = [0; 3];

        fill_ppl_layer_metadata(
            &layer_types,
            &kv_share_targets,
            1024,
            1024,
            &mut slots,
            &mut contexts,
        );

        assert_eq!(slots, [0, 1024, -1]);
        assert_eq!(contexts, [1024, 1025, 1024]);
    }

    #[test]
    fn prefill_chunk_metadata_is_correct_across_wrap_and_chunk_two() {
        let layer_types = [S, G];
        let mut rows = [-1; 8];
        let mut contexts = [-1; 4];

        fill_prefill_chunk_metadata(&layer_types, 1023, 2, 1024, 4, &mut rows, &mut contexts);
        assert_eq!(&rows[..4], &[1023, 0, -1, -1]);
        assert_eq!(&rows[4..], &[1023, 1024, -1, -1]);
        assert_eq!(&contexts[..2], &[1024, 1024]);

        rows.fill(-1);
        contexts.fill(-1);
        fill_prefill_chunk_metadata(&layer_types, 1024, 2, 1024, 4, &mut rows, &mut contexts);
        assert_eq!(&rows[..4], &[0, 1, -1, -1]);
        assert_eq!(&rows[4..], &[1024, 1025, -1, -1]);
        assert_eq!(&contexts[..2], &[1024, 1024]);

        rows.fill(-1);
        contexts.fill(-1);
        fill_prefill_chunk_metadata(&layer_types, 1025, 2, 1024, 4, &mut rows, &mut contexts);
        assert_eq!(&rows[..4], &[1, 2, -1, -1]);
        assert_eq!(&rows[4..], &[1025, 1026, -1, -1]);
        assert_eq!(&contexts[..2], &[1024, 1024]);
    }

    #[test]
    fn generation_capacity_checks_both_limits_and_overflow() {
        assert_eq!(validate_generation_capacity(31, 1, 1, 32, 64), Ok(32));
        assert_eq!(validate_generation_capacity(15, 1, 1024, 32, 16), Ok(16));
        assert!(validate_generation_capacity(32, 1, 1, 32, 64).is_err());
        assert!(validate_generation_capacity(16, 1, 1024, 32, 16).is_err());
        assert!(validate_generation_capacity(0, 1, 1, 32, 64).is_err());
        assert!(validate_generation_capacity(usize::MAX, 1, 1, 32, usize::MAX).is_err());
        assert!(validate_generation_capacity(0, 0, usize::MAX, 2, usize::MAX).is_err());
    }

    #[test]
    fn non_diagnostic_prefill_selects_exactly_one_fresh_kv_path() {
        let normal = generation_prefill_plan(false, false, false, 1026);
        assert_eq!((normal.sequential, normal.staged), (true, false));

        let one_shot_batch = generation_prefill_plan(false, true, false, 1024);
        assert_eq!(
            (one_shot_batch.sequential, one_shot_batch.staged),
            (false, true)
        );

        let chunk_two = generation_prefill_plan(false, true, false, 1026);
        assert_eq!((chunk_two.sequential, chunk_two.staged), (false, true));
        assert!(!chunk_two.reset_before_staged);

        let singleton_fast = generation_prefill_plan(false, true, false, 1);
        assert_eq!(
            (singleton_fast.sequential, singleton_fast.staged),
            (true, false)
        );

        for plan in [normal, one_shot_batch, chunk_two, singleton_fast] {
            assert_ne!(plan.sequential, plan.staged);
            assert!(!plan.reset_before_staged);
        }

        let diagnostic = generation_prefill_plan(false, true, true, 1026);
        assert!(diagnostic.sequential && diagnostic.staged);
        assert!(diagnostic.reset_before_staged);
    }
}
