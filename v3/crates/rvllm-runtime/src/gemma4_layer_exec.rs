//! Gemma 4 layer forward -- 14 kernel launches per layer.
//!
//! Differs from the Llama/Qwen path (layer_exec.rs) in:
//!   - 4 norms per layer (input, post_attn, pre_ff, post_ff)
//!   - QK-norm (RMSNorm on Q and K heads before RoPE)
//!   - v_norm (parameter-free RMS norm on V after projection)
//!   - GELU(tanh) activation instead of SiLU
//!   - Partial RoPE (only rotate first rotary_dim dims per head)
//!   - Per-layer KV head count (sliding vs global)
//!   - head_dim = 256 (requires FA3 .so compiled for 256)
//!   - Per-layer learnable scalar (applied ONCE after both sub-blocks)
//!
//! Launch sequence:
//!   1.  fused_rmsnorm_fp8_quant          input_layernorm
//!   2.  fp8_gemm (cuBLASLt)             Q||K||V projection
//!  2b.  vnorm_f16                       parameter-free RMS norm on V
//!   3.  fused_qk_rmsnorm                QK-norm on Q and K heads
//!   4.  fused_rope_partial_fp8kv        partial RoPE + FP8 Q + paged KV
//!   5.  paged_decode / paged_prefill    FA3 attention (head_dim=256)
//!   6.  quantize_fp8_per_token          attn_out -> fp8
//!   7.  fp8_gemm_residual (cuBLASLt)    O proj += residual
//!   8.  fused_rmsnorm                   post_attention_layernorm (norm only)
//!   9.  fused_rmsnorm_fp8_quant         pre_feedforward_layernorm
//!  10.  fp8_gemm (cuBLASLt)             gate||up projection
//!  11.  fused_gelu_mul_fp8_quant        GELU(tanh)(gate) * up -> FP8
//!  12.  fp8_gemm_residual (cuBLASLt)    down proj += residual
//!  13.  fused_rmsnorm                   post_feedforward_layernorm (norm only)
//!  14.  residual_scale_f16              residual *= layer_scalar (once)

use rvllm_core::Result;
use rvllm_cutlass::{CublasLt, CutlassBackend, Fp8GemmPlan};
use rvllm_fused::gemma4_launcher;
use rvllm_fused::FusedRmsnormFp8QuantLaunch;
use rvllm_kernels::KernelFn;

use rvllm_attention::{
    AttentionBackend, PagedDecodeFp8Launcher, PagedDecodeParams, PagedPrefillFp8Launcher,
    PagedPrefillParams,
};

use rvllm_loader::gemma4_arch::Gemma4LayerType;

#[derive(Copy, Clone, Debug)]
pub struct Gemma4LayerDims {
    pub num_tokens: u32,
    pub hidden: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    /// Number of rows actually allocated in the selected RoPE table.
    /// This is independent from paged-KV cache capacity.
    pub rope_table_rows: u32,
    pub intermediate: u32,
    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
    pub attn_scale: f32,
    pub rms_eps: f32,
    pub layer_type: Gemma4LayerType,
    pub sliding_window: u32,
    pub f16_kv: bool,
    pub num_hidden_layers: u32,
    pub layer_idx: u32,
    pub ple_dim: u32,
    /// E4B KV-shared tail layer: this layer owns NO K/V projection and
    /// writes NO KV cache. Its `scratch.{k,v}_cache` / `{k,v}_scale_cache`
    /// pointers are redirected (by the caller) to the share-SOURCE layer's
    /// cache bases, which were already populated this step by the source
    /// layer's forward. When true the per-layer forward runs Q-proj +
    /// Q-norm + Q-RoPE + attention-read only, skipping the K/V-norm and
    /// the RoPE KV-write. The QKV GEMM weight for a shared layer is a
    /// Q-only matrix (`num_kv_heads == 0` here), so the projection emits
    /// just Q. Default `false` (31B path and E4B non-shared layers).
    pub kv_shared: bool,
}

#[derive(Copy, Clone, Debug)]
pub struct Gemma4LayerWeightPtrs {
    pub attn_norm_gamma: u64,
    pub post_attn_norm_gamma: u64,
    pub pre_ff_norm_gamma: u64,
    pub post_ff_norm_gamma: u64,
    pub q_norm_gamma: u64,
    pub k_norm_gamma: u64,
    pub qkv_fp8: u64,
    pub qkv_scale: u64,
    pub o_fp8: u64,
    pub o_scale: u64,
    pub gate_up_fp8: u64,
    pub gate_up_scale: u64,
    pub down_fp8: u64,
    pub down_scale: u64,
    pub layer_scalar_ptr: u64, // [1] f16, per-layer residual multiplier
    pub qkv_f16: u64,          // 0 = use FP8, nonzero = use F16 GEMM
    pub o_f16: u64,
    pub gate_up_f16: u64,
    pub down_f16: u64,
    pub ple_input_gate_f16: u64,
    pub ple_projection_f16: u64,
    pub post_ple_norm_gamma: u64,
    pub qkv_chscale: u64, // 0 = scalar scale, nonzero = per-channel f32 vec
    pub o_chscale: u64,
    pub gate_up_chscale: u64,
    pub down_chscale: u64,
    /// 2-D blockscale tensor `[N_blocks, K_blocks]` f32 on device.
    /// `0` when the weight's source scale was per-row (or for
    /// synthesized fused qkv/gate_up — their per-part block
    /// alignments don't compose cleanly into a single 2-D tensor).
    /// Only consumed by kernels whose ABI expects the full 2-D
    /// shape (`Fp8GemvF16InLaunch`, CUTLASS SFB).  When `0`, any
    /// such caller MUST fall back to the channelscale-preserving
    /// path — reading `*_chscale` as 2-D produces garbage (walks
    /// off the end of the per-row vec).
    pub qkv_blockscale: u64,
    pub o_blockscale: u64,
    pub gate_up_blockscale: u64,
    pub down_blockscale: u64,
}

#[derive(Copy, Clone, Debug)]
pub struct Gemma4LayerScratch {
    pub hidden_fp8: u64,
    pub hidden_scale: u64,
    pub q_out: u64,
    pub k_out: u64,
    pub v_out: u64,
    pub q_normed: u64,
    pub k_normed: u64,
    /// V after RmsNorm, compact `[num_tokens, num_kv_heads, head_dim]`.
    /// Used by rope to read V before paged-cache write. Was previously
    /// in-place on `v_out` (inside the interleaved QKV buffer), which
    /// silently broke for `num_tokens > 1` because downstream kernels
    /// index as compact.
    pub v_normed: u64,
    pub q_fp8: u64,
    pub k_cache: u64,
    pub v_cache: u64,
    pub q_scale_ptr: u64,
    pub kv_scale_ptr: u64,
    /// Per-slot-per-head f32 K scale cache, shape
    /// `[num_blocks * block_size * num_kv_heads]`. Written by the
    /// rope kernel (amax/448 per slot) and read by the attention
    /// kernel during FP8→f32 dequant. Eliminates the per-tensor
    /// calibration guess.
    pub k_scale_cache: u64,
    /// Companion to `k_scale_cache` for V.
    pub v_scale_cache: u64,
    /// Per-(token, head) f32 Q scale scratch for this layer, shape
    /// `[num_tokens * num_heads]`. Written by the rope kernel when
    /// non-null; read by decode attention on load. Unlike the K/V
    /// caches this is transient — Q is consumed by THIS step's
    /// attention only, so the same region can be reused across
    /// layers and across per-token-decode iterations.
    pub q_scale_cache: u64,
    pub attn_out: u64,
    pub attn_out_fp8: u64,
    pub attn_out_scale: u64,
    pub delta_f16: u64,
    pub gate_up_out: u64,
    pub gate_up_fp8: u64,
    pub gate_up_scale: u64,
    pub mlp_out_fp8: u64,
    pub mlp_out_scale: u64,
    pub gemm_f32_tmp: u64,
    pub cutlass_workspace: u64,
    pub cutlass_workspace_bytes: usize,
    pub fa3_workspace: u64,
    pub fa3_workspace_bytes: usize,
    pub ple_inputs: u64,
    pub ple_gate: u64,
}

#[derive(Clone, Debug)]
pub struct Gemma4GemmPlans {
    pub qkv: Fp8GemmPlan,
    pub o: Fp8GemmPlan,
    pub gate_up: Fp8GemmPlan,
    pub down: Fp8GemmPlan,
}

#[derive(Copy, Clone, Debug)]
pub struct Gemma4MetadataPtrs {
    pub positions: u64,
    pub slot_mapping: u64,
    pub cos: u64,
    pub sin: u64,
    pub block_tables: u64,
    pub context_lens: u64,
}

#[derive(Copy, Clone, Debug)]
pub struct Gemma4LayerKernels<'a> {
    pub fused_rmsnorm: &'a KernelFn,
    pub fused_rmsnorm_fp8_quant: &'a KernelFn,
    pub fused_qk_rmsnorm: &'a KernelFn,
    pub fused_rope_partial_fp8kv: &'a KernelFn,
    pub fused_gelu_mul: &'a KernelFn,
    pub quantize_fp8_per_token: &'a KernelFn,
    pub residual_scale_f16: &'a KernelFn,
    /// E4B per-layer `layer_scalar` (bf16) applied whole-residual after the PLE
    /// gate, per the mlx-lm reference `h = h * layer_scalar`.
    pub residual_scale_bf16s: &'a KernelFn,
    pub vnorm_f16: &'a KernelFn,
    pub vector_add_f16: &'a KernelFn,
    pub bf16_to_f16_sat: &'a KernelFn,
    pub rmsnorm_inplace_bf16: &'a KernelFn,
    pub vector_add_bf16_to_f16: &'a KernelFn,
    pub f32_to_bf16: &'a KernelFn,
    pub f32_to_f16_sat: &'a KernelFn,
    pub scale_cols_f32: &'a KernelFn,
    /// Post-GEMM per-row scale RATIO correction for FP8 GEMM at M>1.
    /// cuBLASLt on sm_121 only supports SCALAR B_SCALE mode (OUTER_VEC
    /// loses the heuristic), so a per-token-scaled activation comes
    /// out of the GEMM with `scale[0]` applied uniformly. This
    /// kernel multiplies row m by `scale[m] / scale[0]` to recover
    /// the per-token scaling.
    pub scale_rows_f32_ratio: &'a KernelFn,
    /// Per-token activation dequant for the w4a8 INT4 decoder GEMMs. The
    /// kernel applies only the weight group scales; this multiplies each
    /// output row m by the per-token activation `scale[m]` the GEMM omits.
    pub scale_rows_f16_pertoken: &'a KernelFn,
    pub compute_qkv_scales: &'a KernelFn,
    pub fused_gelu_mul_f16: &'a KernelFn,
    pub fused_rope_partial_f16kv: &'a KernelFn,
    pub fused_norm_add_residual: &'a KernelFn,
    pub fused_norm_add_residual_f16: &'a KernelFn,
    /// F16-input variant of `fused_norm_add_residual_f16`. Reads f16
    /// gemm output directly (no channelscale broadcast), applies
    /// rmsnorm + residual add + optional layer_scalar. Used by the
    /// Sm121 O-proj and down-proj fast paths after the f16-input
    /// fp8_gemv has already baked the per-channel weight scale into
    /// its output.
    pub fused_norm_add_residual_f16in: &'a KernelFn,
    pub fused_qkv_rmsnorm: &'a KernelFn,
    pub scale_cols_f16: &'a KernelFn,
    pub ple_gelu_mul_f16: &'a KernelFn,
    pub fp8_channelscale_gemv_ktiled: &'a KernelFn,
    pub fp8_channelscale_gemv_splitk: &'a KernelFn,
    /// F16-input fp8_gemv kernel (`fp8_gemv_blockwise_wpr_native_f16in_kernel`).
    /// `None` on non-Blackwell targets — the kernel is gated on
    /// `__CUDA_ARCH__ >= 1000` in `kernels/fp8_gemv.cu`. When `Some` and
    /// the decode batch size is 1, the QKV projection skips the
    /// activation FP8-quant step and runs this kernel directly on the
    /// f16 rmsnorm output.
    pub fp8_gemv_wpr_native_f16in: Option<&'a KernelFn>,
    /// E4B per-layer-embedding gate kernel (`gemma4_ple_gate_kernel`).
    /// `None` for the 31B path (no PLE). When `Some` and the layer has a
    /// PLE injection (`Gemma4PleLayer`), it is launched after the layer
    /// body to add the per-layer-embedding contribution into the residual.
    pub ple_gate: Option<&'a KernelFn>,
}

/// E4B per-layer PLE injection inputs for one decoder layer.
///
/// All pointers are static engine buffers (CUDA-graph capture safe).
/// `gate_w` / `proj_w` are the dense bf16 forms of the layer's
/// `per_layer_input_gate` / `per_layer_projection` weights, dequantized
/// from the compact INT4 tensors at load.
/// `per_layer_input` points at this layer's slice of the combined
/// per-layer-input tensor `[num_tokens, num_layers, h_ple]` for the
/// current step, i.e. base + layer_idx * h_ple (in elements) per token —
/// the caller supplies the already-offset per-layer pointer when
/// `num_tokens == 1` (decode), or a `[num_tokens, h_ple]` strided view.
#[derive(Copy, Clone, Debug)]
pub struct Gemma4PleLayer {
    pub gate_w: u64,
    pub proj_w: u64,
    pub per_layer_input: u64,
    pub post_norm_gamma: u64,
    pub h_ple: u32,
    /// Per-TOKEN stride (in f16 elements) of the `[T, L, h_ple]` combined
    /// per-layer-input buffer = `num_layers * h_ple`. `per_layer_input`
    /// already points at this layer's slice (base + layer*h_ple); the gate
    /// kernel adds `token * pli_stride` to reach `(token*num_layers+layer)`.
    /// (Was `token * h_ple` — correct only for token 0, garbage for token>0.)
    pub pli_stride: u32,
}

#[derive(Copy, Clone, Debug)]
pub enum Gemma4Phase {
    Decode,
    Prefill {
        cu_seqlens_q: u64,
        max_seqlen_q: u32,
        num_seqs: u32,
        /// Absolute position of the chunk's first token: 0 for one-shot
        /// prompt prefill, the committed-token count for continuation
        /// chunks (chunked prefill, speculative-decode verify). Per-query
        /// context is `chunk_start + qi + 1`, NOT `qi + 1`.
        chunk_start: u32,
        /// Device pointer to `num_tokens` i32 per-query context lengths
        /// pre-clamped to the sliding window
        /// (`min(chunk_start + qi + 1, sliding_window)`), or 0.
        ///
        /// When nonzero, SLIDING layers take the interleaved per-token
        /// rope -> decode-attention path instead of the batched prefill
        /// kernel. Required for correctness once chunk positions pass
        /// the sliding window: `rope_fp8kv` writes ALL chunk tokens' KV
        /// into the ring (slot = pos % window) before any attention
        /// runs, so with a full ring a later chunk token overwrites the
        /// slot holding the oldest in-window position that an EARLIER
        /// chunk query still needs. Interleaving the KV write and the
        /// attention read per token reproduces exact decode semantics
        /// at any position. Global layers are unaffected (slot =
        /// position, never reused) and keep the batched kernel.
        sliding_ctx_per_qi: u64,
    },
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemma4_forward(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels<'_>,
    weights: &Gemma4LayerWeightPtrs,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    cublaslt: &CublasLt,
    cutlass: &CutlassBackend,
    sliding_attention: &AttentionBackend,
    global_attention: &AttentionBackend,
    residual: u64,
    stream: u64,
) -> Result<()> {
    gemma4_forward_phase(
        dims,
        kernels,
        weights,
        scratch,
        meta,
        cublaslt,
        cutlass,
        sliding_attention,
        global_attention,
        residual,
        stream,
        Gemma4Phase::Decode,
    )
}

/// Public entry — the 31B FP8 path. Forwards to `gemma4_forward_phase_impl`
/// with `int4 == None`, so the proven 31B forward is byte-identical. The E4B
/// INT4 path calls the impl directly with a `Some(Int4LayerExec)` from
/// `gemma4_e4b_layer_forward`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemma4_forward_phase(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels<'_>,
    weights: &Gemma4LayerWeightPtrs,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    cublaslt: &CublasLt,
    cutlass: &CutlassBackend,
    sliding_attention: &AttentionBackend,
    global_attention: &AttentionBackend,
    residual: u64,
    stream: u64,
    phase: Gemma4Phase,
) -> Result<()> {
    gemma4_forward_phase_impl(
        dims,
        kernels,
        weights,
        scratch,
        meta,
        cublaslt,
        cutlass,
        sliding_attention,
        global_attention,
        residual,
        stream,
        phase,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemma4_forward_phase_impl(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels<'_>,
    weights: &Gemma4LayerWeightPtrs,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    cublaslt: &CublasLt,
    cutlass: &CutlassBackend,
    sliding_attention: &AttentionBackend,
    global_attention: &AttentionBackend,
    residual: u64,
    stream: u64,
    phase: Gemma4Phase,
    // E4B INT4 dispatch. `None` => the FP8 path (31B + E4B-without-INT4). When
    // `Some`, the 4 decoder GEMM sites route through w4a8 instead of cuBLASLt
    // FP8: the QKV / O / gate_up / down weights come from the encoded INT4
    // handles, the A side stays the same FP8-quantized activations the FP8
    // path produces. O and down add the residual via the GEMM epilogue
    // (`beta=1`), matching the FP8 path's fused norm-add-residual ordering.
    int4: Option<crate::gemma4_int4::Int4LayerExec>,
) -> Result<()> {
    // Route small batches through the f16-input GEMV path. Larger batches use
    // the matrix path because each GEMV block reloads its weight tile.
    const FAST_PATH_M_MAX: u32 = 16;
    let q_dim = dims.num_heads * dims.head_dim;
    let _kv_dim = dims.num_kv_heads * dims.head_dim;
    let qkv_rows = (dims.num_heads + 2 * dims.num_kv_heads) * dims.head_dim;

    #[cfg(feature = "cuda")]
    let dbg_layer: i32 = {
        let wanted = std::env::var("RVLLM_DBG_LAYER")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());
        if wanted == Some(dims.layer_idx) {
            dims.layer_idx as i32
        } else {
            -1
        }
    };
    #[cfg(feature = "cuda")]
    macro_rules! probe {
        ($label:expr, $ptr:expr, $n:expr) => {
            if dbg_layer >= 0 {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let sample_n = ($n as usize).min(4096).max(4);
                let mut s = vec![0u16; sample_n];
                cudarc::driver::sys::cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut _, $ptr, sample_n * 2);
                let v: Vec<f32> = s.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
                let amax = v.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
                // amax silently skips NaN — count them or a poisoned
                // buffer reads as healthy (the sm_100 bring-up trap).
                let nans = v.iter().filter(|x| x.is_nan()).count();
                eprintln!(
                    "    [L{} {}] first4={:.4?} amax={:.6e} nans={}/{}",
                    dbg_layer,
                    $label,
                    &v[..4],
                    amax,
                    nans,
                    v.len()
                );
            }
        };
    }
    #[cfg(feature = "cuda")]
    macro_rules! probe_f32 {
        ($label:expr, $ptr:expr) => {
            if dbg_layer >= 0 {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let mut v = [0.0f32; 1];
                cudarc::driver::sys::cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut _, $ptr, 4);
                eprintln!("    [L{} {}] = {:.6e}", dbg_layer, $label, v[0]);
            }
        };
    }
    #[cfg(feature = "cuda")]
    macro_rules! probe_f32_slice {
        ($label:expr, $ptr:expr, $n:expr) => {
            if dbg_layer >= 0 {
                cudarc::driver::sys::cuStreamSynchronize(stream as _);
                let sample_n = ($n as usize).min(4096).max(4);
                let mut v = vec![0.0f32; sample_n];
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    v.as_mut_ptr() as *mut _,
                    $ptr,
                    sample_n * core::mem::size_of::<f32>(),
                );
                let amax = v.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
                let nans = v.iter().filter(|x| x.is_nan()).count();
                eprintln!(
                    "    [L{} {}] first4={:.4?} amax={:.6e} nans={}/{}",
                    dbg_layer,
                    $label,
                    &v[..4],
                    amax,
                    nans,
                    v.len()
                );
            }
        };
    }
    // 1. input_layernorm -> FP8 quant
    // Sm121 fast path for QKV writes f16 into delta_f16 via its
    // own rmsnorm — `scratch.hidden_fp8`/`hidden_scale` go unused. Skip
    // the quant-rmsnorm in that case to avoid the duplicate work.
    #[cfg(feature = "cuda")]
    // Must match the fast-path gate below: we can only skip the FP8
    // quant when `Fp8GemvF16InLaunch` will actually take over and
    // consume `delta_f16`. The fast path additionally requires
    // `qkv_blockscale != 0` — for fused QKV weights (blockscale == 0)
    // the kernel falls back to `fp8_gemm_channelscale_or_fallback`,
    // which reads `scratch.hidden_fp8` and needs the quant to have
    // produced it. Dropping `blockscale != 0` here silently zeroed
    // `hidden_fp8` and propagated zero logits through the LM head.
    let skip_attn_quant = dims.num_tokens == 1
        && weights.qkv_chscale != 0
        && weights.qkv_blockscale != 0
        && weights.qkv_f16 == 0
        && kernels.fp8_gemv_wpr_native_f16in.is_some()
        // INT4 (w4a8) consumes `scratch.hidden_fp8` as its A operand, so the
        // FP8 activation quant MUST run — never skip it on the INT4 path.
        && int4.is_none();
    #[cfg(not(feature = "cuda"))]
    let skip_attn_quant = false;
    if !skip_attn_quant {
        FusedRmsnormFp8QuantLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_rmsnorm_fp8_quant,
            scratch.hidden_fp8,
            scratch.hidden_scale,
            residual,
            weights.attn_norm_gamma,
            stream,
        )?;
    }

    #[cfg(feature = "cuda")]
    probe!("after_step1_residual", residual, dims.hidden);
    #[cfg(feature = "cuda")]
    probe_f32!("step1_hidden_scale", scratch.hidden_scale);
    #[cfg(feature = "cuda")]
    probe_f32!("step1_qkv_wscale", weights.qkv_scale);
    #[cfg(feature = "cuda")]
    {
        if dbg_layer >= 0 {
            cudarc::driver::sys::cuStreamSynchronize(stream as _);
            let mut hs = [0.0f32; 1];
            let mut ws = [0.0f32; 1];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                hs.as_mut_ptr() as *mut _,
                scratch.hidden_scale,
                4,
            );
            cudarc::driver::sys::cuMemcpyDtoH_v2(ws.as_mut_ptr() as *mut _, weights.qkv_scale, 4);
            eprintln!(
                "    [L{} step1_scale_product] hidden*qkv = {:.6e} * {:.6e} = {:.6e}",
                dbg_layer,
                hs[0],
                ws[0],
                hs[0] * ws[0]
            );
        }
    }

    #[cfg(feature = "cuda")]
    {
        if dbg_layer >= 0 {
            cudarc::driver::sys::cuStreamSynchronize(stream as _);
            let mut wb = [0u8; 8];
            cudarc::driver::sys::cuMemcpyDtoH_v2(wb.as_mut_ptr() as *mut _, weights.qkv_fp8, 8);
            eprintln!("    [L{} step2_qkv_fp8_bytes] first8={:?}", dbg_layer, wb);
            let mut hb = [0u8; 8];
            cudarc::driver::sys::cuMemcpyDtoH_v2(hb.as_mut_ptr() as *mut _, scratch.hidden_fp8, 8);
            eprintln!(
                "    [L{} step2_hidden_fp8_bytes] first8={:?}",
                dbg_layer, hb
            );
        }
    }

    // 2. Q||K||V projection
    #[cfg(feature = "cuda")]
    if let Some(ix) = int4 {
        // INT4 path: w4a8 GEMM `q_out = act_fp8 * W_int4^T`. The fused QKV
        // matrix's N == ix.layer.qkv.n (== qkv_rows for an own-KV layer, or
        // q_dim for a KV-shared layer). No residual (beta=0). Output f16 into
        // `scratch.q_out`, identical to the FP8 path's contract.
        crate::gemma4_int4::route_decoder_gemm_w4a8(
            ix.w4a8,
            &ix.layer.qkv,
            scratch.hidden_fp8,
            0,
            scratch.q_out,
            dims.num_tokens as i32,
            1.0,
            0.0,
            ix.workspace,
            ix.workspace_bytes,
            stream,
        )?;
        // w4a8 applies only the weight group scales; multiply each output row
        // by the per-token activation scale the GEMM omits (Q/K are RMSNorm'd
        // next so scale-invariant, but V is not — without this V and the whole
        // INT4 path are ~1/hidden_scale too large -> f16 overflow -> NaN).
        gemma4_launcher::ScaleRowsF16PertokenLaunch {
            num_rows: dims.num_tokens,
            n: ix.layer.qkv.n as u32,
        }
        .launch(
            kernels.scale_rows_f16_pertoken,
            scratch.q_out,
            scratch.hidden_scale,
            stream,
        )?;
    } else if weights.qkv_f16 != 0 {
        // F16 path: copy residual to delta_f16 scratch, apply rmsnorm in-place, use as GEMM input
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
            scratch.delta_f16,
            residual,
            (dims.num_tokens * dims.hidden * 2) as _,
            stream as _,
        );
        gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_rmsnorm,
            scratch.delta_f16,
            weights.attn_norm_gamma,
            stream,
        )?;
        cublaslt.f16_gemm_f32(
            scratch.delta_f16,
            weights.qkv_f16,
            scratch.gemm_f32_tmp,
            dims.num_tokens as i32,
            qkv_rows as i32,
            dims.hidden as i32,
            stream,
        )?;
        gemma4_launcher::Bf16ToF16SatLaunch {
            n: dims.num_tokens * qkv_rows,
        }
        .launch(
            kernels.f32_to_f16_sat,
            scratch.q_out,
            scratch.gemm_f32_tmp,
            stream,
        )?;
    } else if let (true, Some(fn_gemv)) = (
        // Blockscale gate: `Fp8GemvF16InLaunch` reads a 2-D
        // `[N/128, K/128]` tensor. Only enable it when the loader has
        // actually uploaded one (`*_blockscale != 0`). Weights whose
        // scale was per-row or synthesized have `blockscale == 0` and
        // stay on the channelscale-preserving fallback below.
        weights.qkv_blockscale != 0 && dims.num_tokens == 1,
        kernels.fp8_gemv_wpr_native_f16in,
    ) {
        // sm_121 fast path: skip the activation FP8-quant entirely
        // and run `fp8_gemv_blockwise_wpr_native_f16in_kernel` directly
        // against the f16 rmsnorm output. Wins over the
        // `fp8_gemm_channelscale_or_fallback` path on two axes:
        //
        //   * Quality: preserves the per-channel weight block-scale that
        //     the cuBLASLt fallback drops (the cuBLASLt FP8 channelscale
        //     heuristic `LaunchFailed`s on Blackwell consumer, so the
        //     fallback currently collapses to a scalar weight scale).
        //   * Speed: one kernel (f16 GEMV) instead of two (FP8 quant +
        //     cuBLASLt FP8 GEMM), no scratch round-trip through f32.
        //
        // The extra memcpy + rmsnorm-inplace here duplicates the work
        // already done by `fused_rmsnorm_fp8_quant` in step 1 — at M=1
        // that's ~5 KiB of rmsnorm work against a >30 MiB weight GEMV,
        // well below the noise floor.
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
            scratch.delta_f16,
            residual,
            (dims.num_tokens * dims.hidden * 2) as _,
            stream as _,
        );
        gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_rmsnorm,
            scratch.delta_f16,
            weights.attn_norm_gamma,
            stream,
        )?;
        gemma4_launcher::Fp8GemvF16InLaunch {
            m: dims.num_tokens,
            n: qkv_rows,
            k: dims.hidden,
        }
        .launch(
            fn_gemv,
            scratch.q_out,
            weights.qkv_fp8,
            weights.qkv_blockscale,
            scratch.delta_f16,
            stream,
        )?;
    } else if weights.qkv_chscale != 0 {
        fp8_gemm_channelscale_or_fallback(
            cutlass,
            cublaslt,
            kernels.f32_to_f16_sat,
            kernels.scale_cols_f32,
            kernels.scale_cols_f16,
            kernels.scale_rows_f32_ratio,
            kernels.fp8_channelscale_gemv_ktiled,
            kernels.fp8_channelscale_gemv_splitk,
            scratch.q_out,
            scratch.hidden_fp8,
            weights.qkv_fp8,
            scratch.hidden_scale,
            weights.qkv_chscale,
            weights.qkv_blockscale,
            weights.qkv_scale,
            dims.num_tokens as i32,
            qkv_rows as i32,
            dims.hidden as i32,
            scratch.gemm_f32_tmp,
            scratch.cutlass_workspace,
            scratch.cutlass_workspace_bytes,
            stream,
        )?;
    } else {
        cublaslt.fp8_gemm(
            scratch.hidden_fp8,
            weights.qkv_fp8,
            scratch.q_out,
            dims.num_tokens as i32,
            qkv_rows as i32,
            dims.hidden as i32,
            scratch.hidden_scale,
            weights.qkv_scale,
            stream,
        )?;
    }

    #[cfg(feature = "cuda")]
    probe!("step2_q_proj", scratch.q_out, dims.hidden);
    #[cfg(feature = "cuda")]
    {
        if dbg_layer >= 0 {
            let k_offset = q_dim as u64 * 2;
            let v_offset = (q_dim + dims.num_kv_heads * dims.head_dim) as u64 * 2;
            cudarc::driver::sys::cuStreamSynchronize(stream as _);
            let mut sk = [0u16; 4];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                sk.as_mut_ptr() as *mut _,
                scratch.q_out + k_offset,
                8,
            );
            let kv: Vec<f32> = sk.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
            eprintln!("    [L{} step2_k_proj] first4={:.4?}", dbg_layer, kv);
            let mut sv = [0u16; 4];
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                sv.as_mut_ptr() as *mut _,
                scratch.q_out + v_offset,
                8,
            );
            let vv: Vec<f32> = sv.iter().map(|&x| crate::bring_up::f16_to_f32(x)).collect();
            eprintln!("    [L{} step2_v_proj] first4={:.4?}", dbg_layer, vv);
        }
    }

    // 2b+3. Fused QKV-norm: Q/K with learned gamma, V parameter-free.
    // Src pointers (q_out/k_out/v_out) index into the shared interleaved
    // QKV GEMM output with row stride `qkv_rows`. Dsts are compact
    // scratch buffers so the downstream rope kernel sees a uniform
    // `[num_tokens, n_heads, head_dim]` layout across all three.
    //
    // E4B KV-share: a shared tail layer owns no K/V projection. Its QKV
    // weight is Q-only, so the GEMM emitted only `q_dim` columns and the
    // norm/rope must touch K/V for ZERO heads. Driving the norm + rope
    // launches with `kv_heads_proj == 0` collapses their K/V grid rows to
    // nothing (qkv-norm grid is `num_heads + 2*kv`, rope KV write is
    // guarded by `head_idx < num_kv_heads`) without any kernel change.
    // Attention below keeps the real `dims.num_kv_heads` (the source
    // layer's) so it reads the shared cache correctly.
    let kv_heads_proj = if dims.kv_shared { 0 } else { dims.num_kv_heads };
    let qkv_rows = q_dim + 2 * kv_heads_proj * dims.head_dim;
    gemma4_launcher::FusedQkvRmsnormLaunch {
        num_tokens: dims.num_tokens,
        num_heads: dims.num_heads,
        num_kv_heads: kv_heads_proj,
        head_dim: dims.head_dim,
        eps: dims.rms_eps,
        src_row_stride: qkv_rows,
    }
    .launch(
        kernels.fused_qkv_rmsnorm,
        scratch.q_out,
        scratch.k_out,
        scratch.v_out,
        scratch.q_normed,
        scratch.k_normed,
        scratch.v_normed,
        weights.q_norm_gamma,
        weights.k_norm_gamma,
        stream,
    )?;

    #[cfg(feature = "cuda")]
    probe!("step3_q_norm", scratch.q_normed, dims.hidden);
    #[cfg(feature = "cuda")]
    probe!("step3_k_norm", scratch.k_normed, dims.hidden);

    // 4-5. RoPE + attention (F16 or FP8 KV cache, decode or prefill)
    let attention = match dims.layer_type {
        Gemma4LayerType::SlidingAttention => sliding_attention,
        Gemma4LayerType::GlobalAttention => global_attention,
    };
    let window_size_left: i32 = match dims.layer_type {
        Gemma4LayerType::SlidingAttention => (dims.sliding_window as i32) - 1,
        Gemma4LayerType::GlobalAttention => -1,
    };

    // RoPE/KV-write dims: a KV-shared layer ropes Q only and writes no KV
    // cache, so the rope launches see `num_kv_heads == 0` (the KV-write
    // branch `head_idx < num_kv_heads` collapses to nothing). Attention
    // params below keep `dims.num_kv_heads` to read the shared source
    // cache. For non-shared layers `dims_proj == dims`.
    let dims_proj = Gemma4LayerDims {
        num_kv_heads: kv_heads_proj,
        ..dims
    };

    #[cfg(feature = "cuda")]
    match phase {
        Gemma4Phase::Decode => {
            let decode_params = PagedDecodeParams {
                num_seqs: dims.num_tokens,
                num_heads: dims.num_heads,
                num_kv_heads: dims.num_kv_heads,
                head_dim: dims.head_dim,
                block_size: dims.block_size,
                max_blocks_per_seq: dims.max_blocks_per_seq,
                num_blocks_total: dims.num_blocks_total,
                scale: dims.attn_scale,
                window_size_left,
            };
            if dims.f16_kv {
                // F16 KV cache path: RoPE outputs F16 Q and F16 KV cache
                rope_f16kv(dims_proj, kernels, scratch, meta, stream)?;
                if dbg_layer >= 0 {
                    let rc = cudarc::driver::sys::cuStreamSynchronize(stream as _);
                    eprintln!("    [L{} after_rope_f16kv_sync] rc={:?}", dbg_layer, rc);
                }
                let decode = rvllm_attention::PagedDecodeLauncher::new(attention);
                decode.launch(
                    decode_params,
                    scratch.attn_out,
                    scratch.q_normed,
                    scratch.k_cache,
                    scratch.v_cache,
                    meta.block_tables,
                    meta.context_lens,
                    scratch.fa3_workspace,
                    scratch.fa3_workspace_bytes,
                    stream,
                )?;
                if dbg_layer >= 0 {
                    let rc = cudarc::driver::sys::cuStreamSynchronize(stream as _);
                    eprintln!("    [L{} after_decode_f16kv_sync] rc={:?}", dbg_layer, rc);
                }
            } else {
                rope_fp8kv(dims_proj, kernels, scratch, meta, stream)?;
                // When batch prefill is active, KV is FP8 for the
                // whole request. Use the SM89 fallback backend for FP8
                // decode too: it supports Gemma 4 hdim 256/512 and
                // consumes the per-slot scale caches. The SM90 FA3 ABI
                // still only accepts scalar descales.
                let decode = PagedDecodeFp8Launcher::new(global_attention);
                decode.launch(
                    decode_params,
                    scratch.attn_out,
                    scratch.q_fp8,
                    scratch.k_cache,
                    scratch.v_cache,
                    scratch.k_scale_cache,
                    scratch.v_scale_cache,
                    scratch.q_scale_cache,
                    0, // k_descale_fallback (unused when per-slot populated)
                    0, // v_descale_fallback
                    meta.block_tables,
                    meta.context_lens,
                    scratch.fa3_workspace,
                    scratch.fa3_workspace_bytes,
                    scratch.q_scale_ptr,
                    stream,
                )?;
            }
        }
        Gemma4Phase::Prefill {
            cu_seqlens_q,
            max_seqlen_q,
            num_seqs,
            chunk_start,
            sliding_ctx_per_qi,
        } => {
            // Interleaved per-token mode for sliding layers (see the
            // `sliding_ctx_per_qi` field doc): rope + attention run
            // per token below, so the all-tokens rope must not run.
            let sliding_interleaved =
                sliding_ctx_per_qi != 0 && dims.layer_type == Gemma4LayerType::SlidingAttention;
            // Prefill always uses FP8 KV path (no F16 prefill kernel).
            if !sliding_interleaved {
                rope_fp8kv(dims_proj, kernels, scratch, meta, stream)?;
            }

            if sliding_interleaved {
                // --- Interleaved per-token rope -> decode attention ----
                // Correct at ANY chunk position, including past the
                // sliding-window ring wrap (the batched path reads KV
                // slots a later chunk token already overwrote). Token
                // qi's KV is written, then qi attends, before qi+1's KV
                // is written — exact decode semantics per token.
                //
                // Attention backend + call shape mirror the Decode-arm
                // FP8 path exactly (SM89 fallback backend, per-slot
                // scale caches), so a 1-token chunk is bit-identical to
                // a plain decode step.
                let decode_params = PagedDecodeParams {
                    num_seqs: 1,
                    num_heads: dims.num_heads,
                    num_kv_heads: dims.num_kv_heads,
                    head_dim: dims.head_dim,
                    block_size: dims.block_size,
                    max_blocks_per_seq: dims.max_blocks_per_seq,
                    num_blocks_total: dims.num_blocks_total,
                    scale: dims.attn_scale,
                    window_size_left,
                };
                let decode = PagedDecodeFp8Launcher::new(global_attention);
                let q_dim_b = (dims.num_heads as u64) * (dims.head_dim as u64);
                let kv_dim_b = (dims.num_kv_heads as u64) * (dims.head_dim as u64);
                // Inherit `dims_proj` so a KV-shared sliding layer ropes Q
                // only (num_kv_heads == 0) and writes no KV in the
                // per-token interleave; non-shared layers are unchanged.
                let dims_qi = Gemma4LayerDims {
                    num_tokens: 1,
                    ..dims_proj
                };
                for qi in 0..dims.num_tokens as u64 {
                    // Row-offset views of the chunk-shaped scratch /
                    // metadata. Same per-token strides the decode-per-qi
                    // fallback below applies (q_scale_cache stride
                    // rationale documented there).
                    let scr_qi = Gemma4LayerScratch {
                        q_normed: scratch.q_normed + qi * q_dim_b * 2,
                        k_normed: scratch.k_normed + qi * kv_dim_b * 2,
                        v_normed: scratch.v_normed + qi * kv_dim_b * 2,
                        q_fp8: scratch.q_fp8 + qi * q_dim_b,
                        q_scale_cache: if scratch.q_scale_cache != 0 {
                            scratch.q_scale_cache + qi * (dims.num_heads as u64) * 4
                        } else {
                            0
                        },
                        ..*scratch
                    };
                    let meta_qi = Gemma4MetadataPtrs {
                        positions: meta.positions + qi * 4,
                        slot_mapping: meta.slot_mapping + qi * 4,
                        ..*meta
                    };
                    rope_fp8kv(dims_qi, kernels, &scr_qi, &meta_qi, stream)?;
                    decode.launch(
                        decode_params,
                        scratch.attn_out + qi * q_dim_b * 2,
                        scr_qi.q_fp8,
                        scratch.k_cache,
                        scratch.v_cache,
                        scratch.k_scale_cache,
                        scratch.v_scale_cache,
                        scr_qi.q_scale_cache,
                        0, // k_descale_fallback (unused when per-slot populated)
                        0, // v_descale_fallback
                        meta.block_tables,
                        sliding_ctx_per_qi + qi * 4,
                        scratch.fa3_workspace,
                        scratch.fa3_workspace_bytes,
                        scratch.q_scale_ptr,
                        stream,
                    )?;
                }
            } else if matches!(global_attention, rvllm_attention::AttentionBackend::Fa3(_))
                && dims.num_tokens > 1
            {
                let prefill_params = PagedPrefillParams {
                    num_tokens: dims.num_tokens,
                    num_seqs,
                    num_heads: dims.num_heads,
                    num_kv_heads: dims.num_kv_heads,
                    head_dim: dims.head_dim,
                    block_size: dims.block_size,
                    max_blocks_per_seq: dims.max_blocks_per_seq,
                    num_blocks_total: dims.num_blocks_total,
                    scale: dims.attn_scale,
                    window_size_left,
                };
                // H100 route recovered from the known-good branch:
                // use SM90 for decode, but the SM89 paged-prefill .so
                // for prompt prefill. The SM90 varlen kernel is tight
                // on shared memory at Gemma 4 hdim>=256; SM89 supports
                // 128/256/512 and is stable for the one-shot prefill
                // path used by RVLLM_BATCH_PREFILL.
                let prefill = PagedPrefillFp8Launcher::new(global_attention);
                prefill.launch(
                    prefill_params,
                    scratch.attn_out,
                    scratch.q_fp8,
                    scratch.k_cache,
                    scratch.v_cache,
                    meta.block_tables,
                    meta.context_lens,
                    cu_seqlens_q,
                    scratch.fa3_workspace,
                    scratch.fa3_workspace_bytes,
                    scratch.k_scale_cache,
                    scratch.v_scale_cache,
                    scratch.q_scale_cache,
                    scratch.q_scale_ptr,
                    scratch.kv_scale_ptr,
                    scratch.kv_scale_ptr,
                    max_seqlen_q,
                    stream,
                )?;
            } else {
                // --- Decode-per-qi fallback -------------------------------
                // Replaces batch prefill with a loop of single-query decode
                // kernel calls — one per prompt position qi. Per-qi
                // context_lens value is NOT rewritten via HtoD each
                // iteration (that races against the non-default stream);
                // instead we reuse the `cu_seqlens_q` scratch region as a
                // pre-populated device array `[1, 2, ..., num_tokens]`
                // and let decode read ctx = (qi+1) by pointing into it at
                // offset qi. By construction this is bit-identical to the
                // per-token decode path rvllm-ppl validates.
                //
                // Cost: one attention launch per prompt token per layer.
                let decode_params = PagedDecodeParams {
                    num_seqs: 1,
                    num_heads: dims.num_heads,
                    num_kv_heads: dims.num_kv_heads,
                    head_dim: dims.head_dim,
                    block_size: dims.block_size,
                    max_blocks_per_seq: dims.max_blocks_per_seq,
                    num_blocks_total: dims.num_blocks_total,
                    scale: dims.attn_scale,
                    window_size_left,
                };
                let decode = PagedDecodeFp8Launcher::new(attention);
                let o_stride_bytes = (dims.num_heads as u64) * (dims.head_dim as u64) * 2; // f16
                let q_fp8_stride_bytes = (dims.num_heads as u64) * (dims.head_dim as u64); // fp8
                                                                                           // Per-(token, head) Q scale cache stride. Rope writes at
                                                                                           // `q_scale_cache[token_idx * num_heads + head_idx]` (see
                                                                                           // fused_rope_partial_fp8kv.cu). Per-qi decode reads at
                                                                                           // `q_scale_cache[seq_idx * num_heads + head_idx]` with
                                                                                           // seq_idx=0 (num_seqs=1 per launch), so we must advance
                                                                                           // the pointer by `qi * num_heads * sizeof::<f32>()` just
                                                                                           // like the Q FP8 pointer — otherwise token qi gets token 0's
                                                                                           // scale and prefill logits diverge from the per-token
                                                                                           // decode reference.
                let q_scale_stride_bytes = (dims.num_heads as u64) * 4;

                // Pre-populate cu_seqlens_q with per-query context lengths.
                // This source is an ephemeral Vec, so use the checked ordered
                // copy helper rather than retaining pageable host memory in an
                // asynchronous driver transfer. We reuse cu_seqlens_q because it's
                // already sized `(num_tokens + 1) * 4 bytes`. Query qi sits at absolute position
                // `chunk_start + qi`, so its context is
                // `chunk_start + qi + 1` (clamped to the window for
                // sliding layers) — the old `[1..=num_tokens]` was only
                // correct for chunk_start == 0.
                let ctx_host: Vec<i32> = (0..dims.num_tokens as i32)
                    .map(|qi| {
                        let abs = chunk_start as i32 + qi + 1;
                        match dims.layer_type {
                            Gemma4LayerType::SlidingAttention => {
                                abs.min(dims.sliding_window as i32)
                            }
                            Gemma4LayerType::GlobalAttention => abs,
                        }
                    })
                    .collect();
                let ctx_bytes = core::slice::from_raw_parts(
                    ctx_host.as_ptr().cast::<u8>(),
                    ctx_host.len() * core::mem::size_of::<i32>(),
                );
                crate::bring_up::htod_ordered(cu_seqlens_q, ctx_bytes, stream)?;

                for qi in 0..dims.num_tokens {
                    let q_scale_cache_qi = if scratch.q_scale_cache != 0 {
                        scratch.q_scale_cache + (qi as u64) * q_scale_stride_bytes
                    } else {
                        0
                    };
                    decode.launch(
                        decode_params,
                        scratch.attn_out + (qi as u64) * o_stride_bytes,
                        scratch.q_fp8 + (qi as u64) * q_fp8_stride_bytes,
                        scratch.k_cache,
                        scratch.v_cache,
                        scratch.k_scale_cache,
                        scratch.v_scale_cache,
                        q_scale_cache_qi,
                        0, // k_descale_fallback (unused when per-slot populated)
                        0, // v_descale_fallback
                        meta.block_tables,
                        cu_seqlens_q + (qi as u64) * 4,
                        scratch.fa3_workspace,
                        scratch.fa3_workspace_bytes,
                        scratch.q_scale_ptr,
                        stream,
                    )?;
                }
            } // end of decode-per-qi fallback
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = phase;

    #[cfg(feature = "cuda")]
    probe!("step5_attn_out", scratch.attn_out, q_dim);

    // 6. quantize attn_out -> fp8 per-token (skip when F16 KV + F16 O-proj,
    // or when the Sm121 fast path will read `scratch.attn_out`
    // as f16 directly in step 7).
    #[cfg(feature = "cuda")]
    let skip_o_quant = dims.num_tokens <= FAST_PATH_M_MAX
        && weights.o_f16 == 0
        && weights.o_chscale != 0
        && weights.o_blockscale != 0
        && kernels.fp8_gemv_wpr_native_f16in.is_some();
    #[cfg(not(feature = "cuda"))]
    let skip_o_quant = false;
    if (!dims.f16_kv || weights.o_f16 == 0) && !skip_o_quant {
        rvllm_fused::QuantizeFp8PerTokenLaunch {
            num_tokens: dims.num_tokens,
            dim: q_dim,
        }
        .launch(
            kernels.quantize_fp8_per_token,
            scratch.attn_out_fp8,
            scratch.attn_out_scale,
            scratch.attn_out,
            stream,
        )?;
    }

    // 7-8. O proj + channelscale + post_attn norm + residual add
    #[cfg(feature = "cuda")]
    if let Some(ix) = int4 {
        // INT4 O-proj: A = `scratch.attn_out_fp8` (the per-token-quantized
        // attention output), B = encoded INT4 O weight. beta=0 — the residual
        // add is applied by `fused_norm_add_residual_f16in` (post-attn-norm +
        // add residual), matching the FP8 fast-path arm. Output f16 into
        // `gemm_f32_tmp` (reused as f16 staging, >= num_tokens*hidden*2 bytes).
        crate::gemma4_int4::route_decoder_gemm_w4a8(
            ix.w4a8,
            &ix.layer.o,
            scratch.attn_out_fp8,
            0,
            scratch.gemm_f32_tmp,
            dims.num_tokens as i32,
            1.0,
            0.0,
            ix.workspace,
            ix.workspace_bytes,
            stream,
        )?;
        // No per-token activation rescale here: the O output feeds the
        // post-attn RMSNorm below, which is scale-invariant, so the missing
        // activation scale is normalized away. (Applying it would be a no-op
        // and would turn an inf per-token scale into a NaN.)
        gemma4_launcher::FusedNormAddResidualF16InLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_norm_add_residual_f16in,
            scratch.gemm_f32_tmp,
            weights.post_attn_norm_gamma,
            residual,
            0,
            stream,
        )?;
    } else if weights.o_f16 != 0 {
        cublaslt.f16_gemm_f32(
            scratch.attn_out,
            weights.o_f16,
            scratch.gemm_f32_tmp,
            dims.num_tokens as i32,
            dims.hidden as i32,
            q_dim as i32,
            stream,
        )?;
        gemma4_launcher::FusedNormAddResidualLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_norm_add_residual,
            scratch.gemm_f32_tmp,
            weights.post_attn_norm_gamma,
            residual,
            0,
            stream,
        )?;
    } else if let (true, Some(fn_gemv)) = (
        weights.o_blockscale != 0 && dims.num_tokens <= FAST_PATH_M_MAX,
        kernels.fp8_gemv_wpr_native_f16in,
    ) {
        // sm_121 fast path for O projection.
        // `scratch.attn_out` is already f16 (attention output), no
        // pre-rmsnorm needed — post-attn-norm runs in the epilogue via
        // `fused_norm_add_residual_f16in`. We write the GEMV result
        // into `gemm_f32_tmp` (reused as f16 scratch: we only need
        // num_tokens*hidden*2 bytes, well under gemm_f32_tmp's capacity).
        gemma4_launcher::Fp8GemvF16InLaunch {
            m: dims.num_tokens,
            n: dims.hidden,
            k: q_dim,
        }
        .launch(
            fn_gemv,
            scratch.gemm_f32_tmp,
            weights.o_fp8,
            weights.o_blockscale,
            scratch.attn_out,
            stream,
        )?;
        gemma4_launcher::FusedNormAddResidualF16InLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_norm_add_residual_f16in,
            scratch.gemm_f32_tmp,
            weights.post_attn_norm_gamma,
            residual,
            0,
            stream,
        )?;
    } else if weights.o_chscale != 0 {
        // O-proj batch path (num_tokens > FAST_PATH_M_MAX).
        //
        // Preserve the available two-dimensional block scale on the CUTLASS
        // path; otherwise use the channel-scale fallback rather than treating
        // a per-channel vector as a scalar.
        let o_out_f16 = scratch.gemm_f32_tmp; // reused as f16 staging, plenty big
        fp8_gemm_channelscale_or_fallback(
            cutlass,
            cublaslt,
            kernels.f32_to_f16_sat,
            kernels.scale_cols_f32,
            kernels.scale_cols_f16,
            kernels.scale_rows_f32_ratio,
            kernels.fp8_channelscale_gemv_ktiled,
            kernels.fp8_channelscale_gemv_splitk,
            o_out_f16,
            scratch.attn_out_fp8,
            weights.o_fp8,
            scratch.attn_out_scale,
            weights.o_chscale,
            weights.o_blockscale,
            weights.o_scale,
            dims.num_tokens as i32,
            dims.hidden as i32,
            q_dim as i32,
            scratch.gemm_f32_tmp + (dims.num_tokens * dims.hidden * 2) as u64,
            scratch.cutlass_workspace,
            scratch.cutlass_workspace_bytes,
            stream,
        )?;
        gemma4_launcher::FusedNormAddResidualF16InLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_norm_add_residual_f16in,
            o_out_f16,
            weights.post_attn_norm_gamma,
            residual,
            0,
            stream,
        )?;
    } else {
        cublaslt.fp8_gemm_f32(
            scratch.attn_out_fp8,
            weights.o_fp8,
            scratch.gemm_f32_tmp,
            dims.num_tokens as i32,
            dims.hidden as i32,
            q_dim as i32,
            scratch.attn_out_scale,
            weights.o_scale,
            stream,
        )?;
        gemma4_launcher::FusedNormAddResidualLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_norm_add_residual,
            scratch.gemm_f32_tmp,
            weights.post_attn_norm_gamma,
            residual,
            0,
            stream,
        )?;
    }

    #[cfg(feature = "cuda")]
    probe!("after_step8_residual", residual, dims.hidden);

    // 9. pre_feedforward_layernorm -> FP8 quant
    // Same fast-path skip as step 1: gate_up fast path does its own
    // f16 rmsnorm into delta_f16, leaving hidden_fp8/hidden_scale
    // unused.
    #[cfg(feature = "cuda")]
    let skip_ff_quant = dims.num_tokens <= FAST_PATH_M_MAX
        && weights.gate_up_chscale != 0
        && weights.gate_up_blockscale != 0
        && weights.gate_up_f16 == 0
        && kernels.fp8_gemv_wpr_native_f16in.is_some()
        // INT4 gate_up consumes `scratch.hidden_fp8`; the FF quant must run.
        && int4.is_none();
    #[cfg(not(feature = "cuda"))]
    let skip_ff_quant = false;
    if !skip_ff_quant {
        FusedRmsnormFp8QuantLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_rmsnorm_fp8_quant,
            scratch.hidden_fp8,
            scratch.hidden_scale,
            residual,
            weights.pre_ff_norm_gamma,
            stream,
        )?;
    }

    #[cfg(feature = "cuda")]
    probe_f32!("step9_hidden_scale", scratch.hidden_scale);
    #[cfg(feature = "cuda")]
    probe_f32!("step9_gate_up_wscale", weights.gate_up_scale);

    // 10. gate||up projection
    #[cfg(feature = "cuda")]
    if let Some(ix) = int4 {
        // INT4 gate||up: A = `scratch.hidden_fp8` (pre-FF-norm FP8-quantized
        // residual), B = encoded INT4 gate_up weight (N == 2*intermediate).
        // beta=0; output f16 into `scratch.gate_up_out`, which `fused_gelu_mul`
        // reads next — same contract as the FP8 fast-path arm.
        crate::gemma4_int4::route_decoder_gemm_w4a8(
            ix.w4a8,
            &ix.layer.gate_up,
            scratch.hidden_fp8,
            0,
            scratch.gate_up_out,
            dims.num_tokens as i32,
            1.0,
            0.0,
            ix.workspace,
            ix.workspace_bytes,
            stream,
        )?;
        // Per-token activation dequant — REQUIRED here: `fused_gelu_mul` reads
        // gate_up_out next, and GELU is nonlinear so a ~100x-too-large input
        // (the missing 1/hidden_scale factor) gives wrong activations and
        // overflows f16 (gelu(gate)*up) -> NaN.
        gemma4_launcher::ScaleRowsF16PertokenLaunch {
            num_rows: dims.num_tokens,
            n: ix.layer.gate_up.n as u32,
        }
        .launch(
            kernels.scale_rows_f16_pertoken,
            scratch.gate_up_out,
            scratch.hidden_scale,
            stream,
        )?;
    } else if weights.gate_up_f16 != 0 {
        // F16 path: norm residual into gate_up_out scratch, then F16 GEMM
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
            scratch.gate_up_out,
            residual,
            (dims.num_tokens * dims.hidden * 2) as _,
            stream as _,
        );
        gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_rmsnorm,
            scratch.gate_up_out,
            weights.pre_ff_norm_gamma,
            stream,
        )?;
        cublaslt.f16_gemm_f32(
            scratch.gate_up_out,
            weights.gate_up_f16,
            scratch.gemm_f32_tmp,
            dims.num_tokens as i32,
            (2 * dims.intermediate) as i32,
            dims.hidden as i32,
            stream,
        )?;
        gemma4_launcher::Bf16ToF16SatLaunch {
            n: dims.num_tokens * 2 * dims.intermediate,
        }
        .launch(
            kernels.f32_to_f16_sat,
            scratch.gate_up_out,
            scratch.gemm_f32_tmp,
            stream,
        )?;
    } else if let (true, Some(fn_gemv)) = (
        weights.gate_up_blockscale != 0 && dims.num_tokens <= FAST_PATH_M_MAX,
        kernels.fp8_gemv_wpr_native_f16in,
    ) {
        // sm_121 fast path for gate||up projection. Mirrors
        // the QKV fast path: f16 rmsnorm into delta_f16 (pre-FF norm
        // gamma this time), then f16-input fp8_gemv direct to
        // gate_up_out. Downstream `fused_gelu_mul` reads gate_up_out
        // as f16 so no epilogue change is needed.
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
            scratch.delta_f16,
            residual,
            (dims.num_tokens * dims.hidden * 2) as _,
            stream as _,
        );
        gemma4_launcher::RmsnormInplaceLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_rmsnorm,
            scratch.delta_f16,
            weights.pre_ff_norm_gamma,
            stream,
        )?;
        gemma4_launcher::Fp8GemvF16InLaunch {
            m: dims.num_tokens,
            n: 2 * dims.intermediate,
            k: dims.hidden,
        }
        .launch(
            fn_gemv,
            scratch.gate_up_out,
            weights.gate_up_fp8,
            weights.gate_up_blockscale,
            scratch.delta_f16,
            stream,
        )?;
    } else if weights.gate_up_chscale != 0 {
        fp8_gemm_channelscale_or_fallback(
            cutlass,
            cublaslt,
            kernels.f32_to_f16_sat,
            kernels.scale_cols_f32,
            kernels.scale_cols_f16,
            kernels.scale_rows_f32_ratio,
            kernels.fp8_channelscale_gemv_ktiled,
            kernels.fp8_channelscale_gemv_splitk,
            scratch.gate_up_out,
            scratch.hidden_fp8,
            weights.gate_up_fp8,
            scratch.hidden_scale,
            weights.gate_up_chscale,
            weights.gate_up_blockscale,
            weights.gate_up_scale,
            dims.num_tokens as i32,
            (2 * dims.intermediate) as i32,
            dims.hidden as i32,
            scratch.gemm_f32_tmp,
            scratch.cutlass_workspace,
            scratch.cutlass_workspace_bytes,
            stream,
        )?;
    } else {
        cublaslt.fp8_gemm(
            scratch.hidden_fp8,
            weights.gate_up_fp8,
            scratch.gate_up_out,
            dims.num_tokens as i32,
            (2 * dims.intermediate) as i32,
            dims.hidden as i32,
            scratch.hidden_scale,
            weights.gate_up_scale,
            stream,
        )?;
    }

    #[cfg(feature = "cuda")]
    probe!("step10_gate_up_out", scratch.gate_up_out, dims.intermediate);

    // 11-12. GELU*up + down_proj
    #[cfg(feature = "cuda")]
    if let Some(ix) = int4 {
        // INT4 down-proj: 11) FP8 GELU(gate)*up quant -> `scratch.mlp_out_fp8`
        // (same kernel the FP8 path uses), so the A operand is the per-token
        // FP8 mlp activation. 12) w4a8 GEMM (beta=0) -> f16 `gemm_f32_tmp`,
        // then `fused_norm_add_residual_f16in` applies post-FF norm + residual
        // add + the per-layer scalar — identical epilogue to the FP8 fast path.
        gemma4_launcher::FusedGeluMulFp8QuantLaunch {
            num_tokens: dims.num_tokens,
            intermediate: dims.intermediate,
        }
        .launch(
            kernels.fused_gelu_mul,
            scratch.mlp_out_fp8,
            scratch.mlp_out_scale,
            scratch.gate_up_out,
            stream,
        )?;
        probe!(
            "step10b_gate_up_UP",
            scratch.gate_up_out + (dims.intermediate as u64) * 2,
            dims.intermediate
        );
        probe_f32!("step11_mlp_scale", scratch.mlp_out_scale);
        crate::gemma4_int4::route_decoder_gemm_w4a8(
            ix.w4a8,
            &ix.layer.down,
            scratch.mlp_out_fp8,
            0,
            scratch.gemm_f32_tmp,
            dims.num_tokens as i32,
            1.0,
            0.0,
            ix.workspace,
            ix.workspace_bytes,
            stream,
        )?;
        probe!("step12_down_out", scratch.gemm_f32_tmp, dims.hidden);
        // No per-token activation rescale here: the down output feeds the
        // post-FF RMSNorm below (scale-invariant), so the missing activation
        // scale is normalized away.
        // layer_scalar is NOT applied here for the E4B path: the reference
        // applies it ONCE per layer, to the whole hidden, AFTER the PLE gate
        // (see end of `gemma4_e4b_layer_forward`). Pass null so the FF block
        // just adds its contribution (`residual = residual + post_ff_norm(mlp)`).
        gemma4_launcher::FusedNormAddResidualF16InLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_norm_add_residual_f16in,
            scratch.gemm_f32_tmp,
            weights.post_ff_norm_gamma,
            residual,
            0,
            stream,
        )?;
    } else if weights.down_f16 != 0 {
        // F16 path: GELU output to separate buffer (can't alias gate_up_out)
        {
            let mut out = scratch.gate_up_fp8; // use gate_up_fp8 scratch as f16 gelu output
            let mut inp = scratch.gate_up_out;
            let mut inter = dims.intermediate as i32;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                (&mut inter) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block = (dims.intermediate.min(1024), 1, 1);
            let grid = (dims.num_tokens, 1, 1);
            rvllm_fused::launch_raw(kernels.fused_gelu_mul_f16, grid, block, 0, stream, &args)?;
        }
        // F16 GEMM for down_proj (reads from gate_up_fp8 where GELU output was stored)
        cublaslt.f16_gemm_f32(
            scratch.gate_up_fp8,
            weights.down_f16,
            scratch.gemm_f32_tmp,
            dims.num_tokens as i32,
            dims.hidden as i32,
            dims.intermediate as i32,
            stream,
        )?;
        gemma4_launcher::FusedNormAddResidualLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_norm_add_residual,
            scratch.gemm_f32_tmp,
            weights.post_ff_norm_gamma,
            residual,
            0,
            stream,
        )?;
    } else if let (true, Some(fn_gemv)) = (
        weights.down_blockscale != 0 && dims.num_tokens <= FAST_PATH_M_MAX,
        kernels.fp8_gemv_wpr_native_f16in,
    ) {
        // sm_121 fast path for down projection.
        // Skip FP8 GELU-quant — run f16 GELU into `gate_up_fp8`
        // scratch (same aliasing trick as the f16-weights branch),
        // then f16-input fp8_gemv writes f16 directly to
        // `gemm_f32_tmp` (reused as f16 scratch), and
        // `fused_norm_add_residual_f16in` rolls the residual add +
        // post-FF norm in one pass.
        {
            let mut out = scratch.gate_up_fp8;
            let mut inp = scratch.gate_up_out;
            let mut inter = dims.intermediate as i32;
            let args = [
                (&mut out) as *mut u64 as *mut core::ffi::c_void,
                (&mut inp) as *mut u64 as *mut core::ffi::c_void,
                (&mut inter) as *mut i32 as *mut core::ffi::c_void,
            ];
            let block = (dims.intermediate.min(1024), 1, 1);
            let grid = (dims.num_tokens, 1, 1);
            rvllm_fused::launch_raw(kernels.fused_gelu_mul_f16, grid, block, 0, stream, &args)?;
        }
        gemma4_launcher::Fp8GemvF16InLaunch {
            m: dims.num_tokens,
            n: dims.hidden,
            k: dims.intermediate,
        }
        .launch(
            fn_gemv,
            scratch.gemm_f32_tmp,
            weights.down_fp8,
            weights.down_blockscale,
            scratch.gate_up_fp8,
            stream,
        )?;
        gemma4_launcher::FusedNormAddResidualF16InLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            eps: dims.rms_eps,
        }
        .launch(
            kernels.fused_norm_add_residual_f16in,
            scratch.gemm_f32_tmp,
            weights.post_ff_norm_gamma,
            residual,
            0,
            stream,
        )?;
    } else {
        // FP8 path
        gemma4_launcher::FusedGeluMulFp8QuantLaunch {
            num_tokens: dims.num_tokens,
            intermediate: dims.intermediate,
        }
        .launch(
            kernels.fused_gelu_mul,
            scratch.mlp_out_fp8,
            scratch.mlp_out_scale,
            scratch.gate_up_out,
            stream,
        )?;
        if weights.down_chscale != 0 {
            // Same reroute as O-proj: use fp8_gemm_channelscale_or_fallback
            // so the M>=128 batch path picks CUTLASS + full 2-D blockscale
            // (when available) instead of the cuBLASLt + channelscale
            // approximation that collapses K-block variation.
            let down_out_f16 = scratch.gemm_f32_tmp;
            fp8_gemm_channelscale_or_fallback(
                cutlass,
                cublaslt,
                kernels.f32_to_f16_sat,
                kernels.scale_cols_f32,
                kernels.scale_cols_f16,
                kernels.scale_rows_f32_ratio,
                kernels.fp8_channelscale_gemv_ktiled,
                kernels.fp8_channelscale_gemv_splitk,
                down_out_f16,
                scratch.mlp_out_fp8,
                weights.down_fp8,
                scratch.mlp_out_scale,
                weights.down_chscale,
                weights.down_blockscale,
                weights.down_scale,
                dims.num_tokens as i32,
                dims.hidden as i32,
                dims.intermediate as i32,
                scratch.gemm_f32_tmp + (dims.num_tokens * dims.hidden * 2) as u64,
                scratch.cutlass_workspace,
                scratch.cutlass_workspace_bytes,
                stream,
            )?;
            gemma4_launcher::FusedNormAddResidualF16InLaunch {
                num_tokens: dims.num_tokens,
                hidden: dims.hidden,
                eps: dims.rms_eps,
            }
            .launch(
                kernels.fused_norm_add_residual_f16in,
                down_out_f16,
                weights.post_ff_norm_gamma,
                residual,
                0,
                stream,
            )?;
        } else {
            cublaslt.fp8_gemm_f32(
                scratch.mlp_out_fp8,
                weights.down_fp8,
                scratch.gemm_f32_tmp,
                dims.num_tokens as i32,
                dims.hidden as i32,
                dims.intermediate as i32,
                scratch.mlp_out_scale,
                weights.down_scale,
                stream,
            )?;
            gemma4_launcher::FusedNormAddResidualLaunch {
                num_tokens: dims.num_tokens,
                hidden: dims.hidden,
                eps: dims.rms_eps,
            }
            .launch(
                kernels.fused_norm_add_residual,
                scratch.gemm_f32_tmp,
                weights.post_ff_norm_gamma,
                residual,
                0,
                stream,
            )?;
        }
    }

    #[cfg(feature = "cuda")]
    {
        if weights.ple_input_gate_f16 != 0
            && weights.ple_projection_f16 != 0
            && weights.post_ple_norm_gamma != 0
            && scratch.ple_inputs != 0
            && scratch.ple_gate != 0
            && dims.ple_dim != 0
        {
            let ple_layer = scratch.ple_inputs + (dims.layer_idx * dims.ple_dim * 2) as u64;
            probe!("ple_input", ple_layer, dims.ple_dim);
            cublaslt.f16_gemm_f32(
                residual,
                weights.ple_input_gate_f16,
                scratch.gemm_f32_tmp,
                dims.num_tokens as i32,
                dims.ple_dim as i32,
                dims.hidden as i32,
                stream,
            )?;
            probe_f32_slice!("ple_gate_f32", scratch.gemm_f32_tmp, dims.ple_dim);
            gemma4_launcher::Bf16ToF16SatLaunch {
                n: dims.num_tokens * dims.ple_dim,
            }
            .launch(
                kernels.f32_to_f16_sat,
                scratch.ple_gate,
                scratch.gemm_f32_tmp,
                stream,
            )?;
            probe!("ple_gate_f16", scratch.ple_gate, dims.ple_dim);
            {
                let mut gate = scratch.ple_gate;
                let mut per_layer_inputs = scratch.ple_inputs;
                let mut layer_idx = dims.layer_idx as i32;
                let mut num_layers = dims.num_hidden_layers as i32;
                let mut ple_dim = dims.ple_dim as i32;
                let args = [
                    (&mut gate) as *mut u64 as *mut core::ffi::c_void,
                    (&mut per_layer_inputs) as *mut u64 as *mut core::ffi::c_void,
                    (&mut layer_idx) as *mut i32 as *mut core::ffi::c_void,
                    (&mut num_layers) as *mut i32 as *mut core::ffi::c_void,
                    (&mut ple_dim) as *mut i32 as *mut core::ffi::c_void,
                ];
                let block = (dims.ple_dim.min(256), 1, 1);
                let grid = (dims.num_tokens, 1, 1);
                rvllm_fused::launch_raw(kernels.ple_gelu_mul_f16, grid, block, 0, stream, &args)?;
            }
            probe!("ple_gate_mul", scratch.ple_gate, dims.ple_dim);
            cublaslt.f16_gemm_f32(
                scratch.ple_gate,
                weights.ple_projection_f16,
                scratch.gemm_f32_tmp,
                dims.num_tokens as i32,
                dims.hidden as i32,
                dims.ple_dim as i32,
                stream,
            )?;
            probe_f32_slice!("ple_proj_f32", scratch.gemm_f32_tmp, dims.hidden);
            gemma4_launcher::FusedNormAddResidualLaunch {
                num_tokens: dims.num_tokens,
                hidden: dims.hidden,
                eps: dims.rms_eps,
            }
            .launch(
                kernels.fused_norm_add_residual,
                scratch.gemm_f32_tmp,
                weights.post_ple_norm_gamma,
                residual,
                0,
                stream,
            )?;
        }

        gemma4_launcher::ResidualScaleF16Launch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
        }
        .launch(
            kernels.residual_scale_f16,
            residual,
            weights.layer_scalar_ptr,
            stream,
        )?;
    }

    #[cfg(feature = "cuda")]
    probe!("after_step14_residual", residual, dims.hidden);

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (cublaslt, qkv_rows, _kv_dim, int4);
    }
    Ok(())
}

/// E4B per-layer forward: the standard Gemma 4 layer body (with KV-share
/// + the runtime sliding-window already baked into `dims`) followed by the
/// PLE gate injection into the residual stream.
///
/// This E4B branch leaves the 31B path unchanged:
/// `gemma4_forward_phase` runs first, then (when `ple` is `Some`
/// and the `ple_gate` kernel is loaded) `gemma4_ple_gate_kernel` adds the
/// per-layer-embedding contribution. The whole sequence is launched on a
/// single stream with static buffers, so it is CUDA-graph-capture safe
/// (no host syncs in the captured region).
///
/// PLE ordering (mlx-lm `DecoderLayer.__call__`): the injection happens
/// AFTER both residual sub-blocks (attn, MLP) and BEFORE the per-layer
/// scalar. E4B's `layer_scalar == 1.0`, so applying it inside
/// `fused_norm_add_residual` (as the 31B path does) and then adding the
/// PLE contribution is numerically identical to the reference order.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemma4_e4b_layer_forward(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels<'_>,
    weights: &Gemma4LayerWeightPtrs,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    cublaslt: &CublasLt,
    cutlass: &CutlassBackend,
    sliding_attention: &AttentionBackend,
    global_attention: &AttentionBackend,
    residual: u64,
    stream: u64,
    phase: Gemma4Phase,
    ple: Option<Gemma4PleLayer>,
    // E4B INT4 dispatch. When `Some`, the 4 decoder GEMMs route through w4a8.
    // `None` => the layer body runs the FP8 path (E4B-without-INT4). The PLE
    // gate injection runs the same way regardless.
    int4: Option<crate::gemma4_int4::Int4LayerExec>,
) -> Result<()> {
    gemma4_forward_phase_impl(
        dims,
        kernels,
        weights,
        scratch,
        meta,
        cublaslt,
        cutlass,
        sliding_attention,
        global_attention,
        residual,
        stream,
        phase,
        int4,
    )?;

    let skip_ple_gate = std::env::var("RVLLM_NO_PLE_GATE").is_ok();
    if let (Some(ple), Some(ple_kernel)) = (ple, kernels.ple_gate) {
        if skip_ple_gate {
            return Ok(());
        }
        // residual (== h after both sub-blocks) is read as `hidden_states`
        // AND written in place — the kernel stages h into smem first, so
        // the in/out aliasing is safe.
        rvllm_fused::PleGateLaunch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
            h_ple: ple.h_ple,
            pli_stride: ple.pli_stride,
            eps: dims.rms_eps,
        }
        .launch(
            ple_kernel,
            residual,
            ple.gate_w,
            ple.proj_w,
            ple.per_layer_input,
            ple.post_norm_gamma,
            stream,
        )?;
    }

    // Reference (mlx-lm Gemma4 DecoderLayer.__call__): `h = h * layer_scalar`,
    // applied to the WHOLE hidden AFTER attn + FF + PLE-gate. `layer_scalar` is
    // a per-layer BF16 weight. Read it with the bf16 kernel; interpreting its
    // bytes as f16 changes the value and corrupts the residual stream.
    if weights.layer_scalar_ptr != 0 {
        rvllm_fused::gemma4_launcher::ResidualScaleF16Launch {
            num_tokens: dims.num_tokens,
            hidden: dims.hidden,
        }
        .launch(
            kernels.residual_scale_bf16s,
            residual,
            weights.layer_scalar_ptr,
            stream,
        )?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
/// In-place per-column scale on an f16 buffer: data[m,n] *= scale[n].
/// Kernel grid is 2-D (ceil(n/256), m) per scale_cols_f16.cu.
#[cfg(feature = "cuda")]
unsafe fn launch_scale_cols_f16(
    kernel: &KernelFn,
    data: u64,
    scale: u64,
    m: u32,
    n: u32,
    stream: u64,
) -> Result<()> {
    let mut data = data;
    let mut scale = scale;
    let mut m_i = m as i32;
    let mut n_i = n as i32;
    let args = [
        (&mut data) as *mut u64 as *mut core::ffi::c_void,
        (&mut scale) as *mut u64 as *mut core::ffi::c_void,
        (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
        (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
    ];
    let block = (256u32, 1, 1);
    let grid = ((n + 255) / 256, m, 1);
    rvllm_fused::launch_raw(kernel, grid, block, 0, stream, &args)
}

unsafe fn launch_scale_cols_f32(
    kernel: &KernelFn,
    data: u64,
    scale: u64,
    m: u32,
    n: u32,
    stream: u64,
) -> Result<()> {
    let total = m * n;
    let mut data = data;
    let mut scale = scale;
    let mut m_i = m as i32;
    let mut n_i = n as i32;
    let args = [
        (&mut data) as *mut u64 as *mut core::ffi::c_void,
        (&mut scale) as *mut u64 as *mut core::ffi::c_void,
        (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
        (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
    ];
    let block = (256u32, 1, 1);
    let grid = ((total + 255) / 256, 1, 1);
    rvllm_fused::launch_raw(kernel, grid, block, 0, stream, &args)
}

/// Post-GEMM per-row scale RATIO correction: `data[m, n] *=
/// scale[m] / scale[0]` for an MxN f32 row-major buffer. Used to
/// fix up cuBLASLt FP8 GEMM output when the B_SCALE was a per-token
/// array but cuBLASLt ran in SCALAR mode (sm_121's only option).
#[cfg(feature = "cuda")]
unsafe fn launch_scale_rows_f32_ratio(
    kernel: &KernelFn,
    data: u64,
    scale: u64,
    m: u32,
    n: u32,
    stream: u64,
) -> Result<()> {
    let total = m * n;
    let mut data = data;
    let mut scale = scale;
    let mut m_i = m as i32;
    let mut n_i = n as i32;
    let args = [
        (&mut data) as *mut u64 as *mut core::ffi::c_void,
        (&mut scale) as *mut u64 as *mut core::ffi::c_void,
        (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
        (&mut n_i) as *mut i32 as *mut core::ffi::c_void,
    ];
    let block = (256u32, 1, 1);
    let grid = ((total + 255) / 256, 1, 1);
    rvllm_fused::launch_raw(kernel, grid, block, 0, stream, &args)
}

/// CUTLASS FP8 GEMM with row×col scale epilogue + sm_121 fallback.
///
/// On SM90 this delegates straight to the CUTLASS `.so`
/// wrapper (`cutlass_fp8_gemm_channelscale`), which applies the full
/// per-token row scale × per-channel column scale epilogue.
///
/// When a native CUTLASS route is unavailable, the fallback preserves the
/// per-channel scale through an f32 matmul, column scaling, and f16 cast.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
unsafe fn fp8_gemm_channelscale_or_fallback(
    cutlass: &CutlassBackend,
    cublaslt: &CublasLt,
    fn_f32_to_f16: &KernelFn,
    fn_scale_cols_f32: &KernelFn,
    fn_scale_cols_f16: &KernelFn,
    fn_scale_rows_f32_ratio: &KernelFn,
    fn_gemv_ktiled: &KernelFn,
    fn_gemv_splitk: &KernelFn,
    output_f16: u64,
    a_fp8: u64,
    b_fp8: u64,
    a_scale: u64,
    b_chscale: u64,
    // Optional per-(N/128, K/128) blockscale, row-major, passed DIRECTLY
    // to CUTLASS prep_sfb. `0` when weights were fused (e.g. QKV /
    // gate_up where multi-part blockwise reconstruction is undefined);
    // in that case the CUTLASS path is skipped and we fall through to
    // the channelscale-preserving fallback. Not interchangeable with
    // `b_chscale` — the latter is [rows] per-row scale, the former is
    // [n_blocks, k_blocks] 2-D block scale.
    b_blockscale: u64,
    b_scale_scalar: u64,
    m: i32,
    n: i32,
    k: i32,
    scratch_f32: u64,
    cutlass_workspace: u64,
    cutlass_workspace_bytes: usize,
    stream: u64,
) -> Result<()> {
    // Safety rail from PR review: `b_chscale != 0` means the real
    // per-channel scale lives in the vector; `b_scale_scalar` is a
    // sentinel 1.0 set by the loader. Any fallback arm that ignores
    // `b_chscale` would multiply raw FP8 bytes by 1.0 and emit
    // garbage. Every arm below must either
    //   (a) consume `b_chscale` (the full row×col channelscale path),
    //   (b) be guaranteed unreachable when `b_chscale != 0`, or
    //   (c) fall through to the channelscale-preserving f32+scale_cols
    //       path at the tail of this function.
    // The cuBLASLt scalar `fp8_gemm(..., b_scale_scalar)` shortcut is
    // reserved for the `b_chscale == 0` case (true scalar-scale
    // weights).
    debug_assert!(
        b_chscale != 0 || b_scale_scalar != 0,
        "fp8_gemm_channelscale_or_fallback called with both b_chscale \
         and b_scale_scalar == 0 — no scale source",
    );
    let m1_gemv_enabled =
        m == 1 && b_chscale != 0 && std::env::var("RVLLM_FP8_GEMV_M1").ok().as_deref() == Some("1");
    if m1_gemv_enabled {
        return fp8_channelscale_gemv_m1(
            fn_gemv_ktiled,
            fn_gemv_splitk,
            fn_f32_to_f16,
            output_f16,
            a_fp8,
            b_fp8,
            a_scale,
            b_chscale,
            n,
            k,
            scratch_f32,
            stream,
        );
    }
    // CUTLASS SM120 blockwise path — default ON when a `SoSm120`
    // backend is loaded with all four prep symbols present (SFA bytes /
    // SFB bytes / prep SFA / prep SFB). Older `libcutlass_sm120.so`
    // builds without the prep helpers fall through to cuBLASLt.
    // The CUTLASS cooperative blockwise kernel is built with a 128×128
    // MMA tile and hard-asserts `M >= 128` (sm90_gemm_tma_warpspecialized_
    // cooperative.hpp:371). Gate the dispatch so small-batch decode
    // (M=num_seqs < 128) still gets routed through the cuBLASLt
    // fallback below.
    // CUTLASS blockwise path requires a proper 2-D b_blockscale. When
    // the loader fused a weight (QKV / gate_up) it cannot reconstruct
    // a consistent blockscale across parts (different shards ship
    // different block alignments), so blockscale_ptr is `None` →
    // b_blockscale == 0 here. In that case fall through to cuBLASLt
    // channelscale fallback — feeding the per-row channelscale to
    // CUTLASS's prep_sfb produces silently-wrong output (the kernel
    // interprets it as [n_blocks, k_blocks] and reads completely
    // unrelated row entries for the K/V output regions, which is
    // otherwise an invalid scale source can silently corrupt the output).
    let cutlass_sm120_enabled = m >= 128
        && b_blockscale != 0
        && std::env::var("RVLLM_FP8_GEMM_CUTLASS_SM120")
            .ok()
            .as_deref()
            != Some("0");
    if cutlass_sm120_enabled {
        if let CutlassBackend::SoSm120(lib) = cutlass {
            if lib.prep_sfa.is_some() && lib.prep_sfb.is_some() {
                let sfa_bytes = lib.sfa_bytes(m, k);
                let _sfb_bytes = lib.sfb_bytes(n, k);
                // 16-byte-align the SFB offset inside scratch_f32.
                let sfa_aligned = (sfa_bytes + 15) & !15;
                let sfa_ptr = scratch_f32;
                let sfb_ptr = scratch_f32 + sfa_aligned as u64;
                lib.launch_prep_sfa(a_scale, sfa_ptr, m, k, stream)?;
                lib.launch_prep_sfb(b_blockscale, sfb_ptr, n, k, stream)?;
                return cutlass.launch_fp8_gemm_blockscale_sm120(
                    output_f16,
                    a_fp8,
                    b_fp8,
                    sfa_ptr,
                    sfb_ptr,
                    m,
                    n,
                    k,
                    cutlass_workspace,
                    cutlass_workspace_bytes,
                    stream,
                );
            }
        }
    }
    // Channelscale-preserving fallback for the sm_121 / SoSm120
    // paths. cuBLASLt's scalar `fp8_gemm(..., b_scale_scalar)` is
    // the shortcut when there's no per-channel scale to apply; when
    // `b_chscale != 0` we can NOT use that shortcut (see the
    // safety rail at the top of this fn — raw FP8 × 1.0 would
    // land in the output). Instead route through:
    //   1. FP8 GEMM into the f32 scratch region, no per-channel
    //      scale yet → result has baked-in a_scale × 1.0.
    //   2. scale_cols_f32 multiplies each column by `b_chscale[n]`.
    //   3. cast f32 → f16 into the actual output buffer.
    // Slower than a fused CUTLASS channelscale GEMM but correct.
    // `scratch_f32` must be sized for `m * n * sizeof(f32)` — the
    // caller guarantees this (same `gemm_f32_tmp` region the
    // existing fp8_gemm_f32 path uses).
    let fallback_f32_scale_cast = || -> Result<()> {
        cublaslt.fp8_gemm_f32(
            a_fp8,
            b_fp8,
            scratch_f32,
            m,
            n,
            k,
            a_scale,
            b_scale_scalar,
            stream,
        )?;
        // cuBLASLt on sm_121 only supports SCALAR B_SCALE mode; the
        // FP8 GEMM above applied `a_scale[0]` uniformly to every
        // output row. For M>1 with per-token activation scales this
        // under/over-scales rows 1..M-1 by `a_scale[m] / a_scale[0]`.
        // Correct it with the ratio kernel. At M=1 the kernel is a
        // no-op (row 0 stays put) so decode stays bit-identical.
        if m > 1 {
            launch_scale_rows_f32_ratio(
                fn_scale_rows_f32_ratio,
                scratch_f32,
                a_scale,
                m as u32,
                n as u32,
                stream,
            )?;
        }
        if b_chscale != 0 {
            launch_scale_cols_f32(
                fn_scale_cols_f32,
                scratch_f32,
                b_chscale,
                m as u32,
                n as u32,
                stream,
            )?;
        }
        // `Bf16ToF16SatLaunch` has the (dst, src, n) ABI we need;
        // name refers to the historical caller, the launched kernel
        // is `f32_to_f16_sat`.
        gemma4_launcher::Bf16ToF16SatLaunch { n: (m * n) as u32 }.launch(
            fn_f32_to_f16,
            output_f16,
            scratch_f32,
            stream,
        )
    };

    // Small-M GEMMs default to cuBLASLt plus scale-and-cast, with a tunable
    // crossover. SM100 keeps this route disabled unless explicitly enabled.
    static LT_MAX_M: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    let lt_max_m = *LT_MAX_M.get_or_init(|| {
        std::env::var("RVLLM_FP8_GEMM_LT_MAX_M")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                // sm_100: cuBLASLt FP8 M=1 writes nothing (see
                // rvllm_cutlass::lt_fp8_default_off) — keep small-M
                // GEMMs on the CUTLASS/GEMV routes unless the operator
                // explicitly overrides for tuning.
                if rvllm_cutlass::lt_fp8_default_off() {
                    0
                } else {
                    64
                }
            })
    });
    static LT_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let force_lt_small_m = m <= lt_max_m
        && *LT_ENABLED
            .get_or_init(|| std::env::var("RVLLM_FP8_GEMM_LT_M1").ok().as_deref() != Some("0"));
    match cutlass {
        CutlassBackend::So(_) if force_lt_small_m => {
            // RVLLM_FP8_GEMM_LT_F16OUT=1 (m==1 only): let cuBLASLt write
            // f16 directly and apply the per-channel scale in-place with
            // ONE kernel — drops the f32 scratch round-trip and the cast
            // (240 launches/step at batch=1; also trims eager per-token
            // prefill). Numerics: output rounds to f16 BEFORE the channel
            // scale (double rounding) instead of once after it in f32 —
            // self-consistent/deterministic, but gate any default flip on
            // a ppl check. m>1 stays on the f32 route (the per-row ratio
            // correction kernel is f32-only).
            static LT_F16OUT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let lt_f16out = m == 1
                && *LT_F16OUT.get_or_init(|| {
                    std::env::var("RVLLM_FP8_GEMM_LT_F16OUT").ok().as_deref() == Some("1")
                });
            if b_chscale != 0 {
                if lt_f16out {
                    cublaslt.fp8_gemm(
                        a_fp8,
                        b_fp8,
                        output_f16,
                        m,
                        n,
                        k,
                        a_scale,
                        b_scale_scalar,
                        stream,
                    )?;
                    launch_scale_cols_f16(
                        fn_scale_cols_f16,
                        output_f16,
                        b_chscale,
                        m as u32,
                        n as u32,
                        stream,
                    )
                } else {
                    fallback_f32_scale_cast()
                }
            } else {
                cublaslt.fp8_gemm(
                    a_fp8,
                    b_fp8,
                    output_f16,
                    m,
                    n,
                    k,
                    a_scale,
                    b_scale_scalar,
                    stream,
                )
            }
        }
        CutlassBackend::So(_) => cutlass.launch_fp8_gemm_channelscale(
            output_f16,
            a_fp8,
            b_fp8,
            a_scale,
            b_chscale,
            m,
            n,
            k,
            cutlass_workspace,
            cutlass_workspace_bytes,
            stream,
        ),
        CutlassBackend::Absent => {
            if b_chscale != 0 {
                fallback_f32_scale_cast()
            } else {
                cublaslt.fp8_gemm(
                    a_fp8,
                    b_fp8,
                    output_f16,
                    m,
                    n,
                    k,
                    a_scale,
                    b_scale_scalar,
                    stream,
                )
            }
        }
        // CUTLASS SM120 blockwise kernel — the .so is loaded and
        // dispatchable, but its SFA ABI wants a [ceil(M/128), K/128] f32
        // tensor, not the per-M a_scale vector we have here. That
        // broadcast has to be staged by the caller (future: prefill
        // scratch region + a K/128-wide broadcast kernel), so for now
        // we route through the f32+scale_cols path which preserves
        // channelscale correctly. The `RVLLM_FP8_GEMM_CUTLASS_SM120`
        // opt-in at the top takes the SoSm120 fast path for M>=128;
        // below that we land here.
        CutlassBackend::SoSm120(_) => {
            if b_chscale != 0 {
                fallback_f32_scale_cast()
            } else {
                cublaslt.fp8_gemm(
                    a_fp8,
                    b_fp8,
                    output_f16,
                    m,
                    n,
                    k,
                    a_scale,
                    b_scale_scalar,
                    stream,
                )
            }
        }
        // Exhaustiveness for #[non_exhaustive] CutlassBackend — a
        // future variant lands here with a typed error rather than
        // aborting the process, so adding a new backend can never
        // silently panic in prod. The explicit arms above stay the
        // source of truth; this is the default-deny for unknowns.
        _ => Err(rvllm_core::RvllmError::cutlass(
            rvllm_core::CutlassError::FeatureNotAvailable {
                op: "fp8_gemm_channelscale (unknown CutlassBackend variant)",
            },
            rvllm_core::CutlassCtx {
                kernel: "fp8_gemm_channelscale",
                stream,
            },
        )),
    }
}

#[cfg(any(feature = "cuda", test))]
const fn fp8_channelscale_gemv_supported(major: i32, minor: i32) -> bool {
    major > 8 || (major == 8 && minor >= 9)
}

#[cfg(feature = "cuda")]
unsafe fn fp8_channelscale_gemv_m1(
    fn_ktiled: &KernelFn,
    fn_splitk: &KernelFn,
    fn_f32_to_f16: &KernelFn,
    output_f16: u64,
    a_fp8: u64,
    b_fp8: u64,
    a_scale: u64,
    b_chscale: u64,
    n: i32,
    k: i32,
    scratch_f32: u64,
    stream: u64,
) -> Result<()> {
    let context = fn_ktiled.context();
    let (major, minor) = context.compute_capability();
    if !fp8_channelscale_gemv_supported(major, minor) {
        return Err(rvllm_core::RvllmError::cuda(
            "fp8_channelscale_gemv_m1 requires compute capability 8.9 or newer",
            rvllm_core::CudaErrorKind::FeatureNotAvailable,
            rvllm_core::CudaCtx {
                stream,
                kernel: "fp8_channelscale_gemv_m1",
                launch: None,
                device: context.device(),
            },
        ));
    }
    let warps_per_block = 8u32;
    let block = (warps_per_block * 32, 1, 1);
    let grid_x = (n as u32).div_ceil(warps_per_block);
    let use_splitk = n <= 8192 && k >= 8192;

    if use_splitk {
        let split_k = std::env::var("RVLLM_SPLITK")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(8)
            .max(1);
        let rc = cudarc::driver::sys::cuMemsetD8Async(
            scratch_f32,
            0,
            (n as usize) * 4,
            stream as cudarc::driver::sys::CUstream,
        );
        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(rvllm_core::RvllmError::cuda(
                "cuMemsetD8Async",
                rvllm_core::CudaErrorKind::LaunchFailed,
                rvllm_core::CudaCtx {
                    stream,
                    kernel: "fp8_channelscale_gemv_splitk",
                    launch: None,
                    device: -1,
                },
            ));
        }
        let mut out = scratch_f32;
        let mut weight = b_fp8;
        let mut wscale = b_chscale;
        let mut act = a_fp8;
        let mut act_scale = a_scale;
        let mut ni = n;
        let mut ki = k;
        let mut split = split_k;
        let args = [
            (&mut out) as *mut u64 as *mut core::ffi::c_void,
            (&mut weight) as *mut u64 as *mut core::ffi::c_void,
            (&mut wscale) as *mut u64 as *mut core::ffi::c_void,
            (&mut act) as *mut u64 as *mut core::ffi::c_void,
            (&mut act_scale) as *mut u64 as *mut core::ffi::c_void,
            (&mut ni) as *mut i32 as *mut core::ffi::c_void,
            (&mut ki) as *mut i32 as *mut core::ffi::c_void,
            (&mut split) as *mut i32 as *mut core::ffi::c_void,
        ];
        rvllm_fused::launch_raw(
            fn_splitk,
            (grid_x, split_k as u32, 1),
            block,
            0,
            stream,
            &args,
        )?;
    } else {
        let tile_k = std::env::var("RVLLM_TILE_K")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(8192)
            .max(16);
        let tile_k = ((tile_k + 15) / 16) * 16;
        let smem = (tile_k.min(k).max(16) as u32) * 2;
        let mut out = scratch_f32;
        let mut weight = b_fp8;
        let mut wscale = b_chscale;
        let mut act = a_fp8;
        let mut act_scale = a_scale;
        let mut ni = n;
        let mut ki = k;
        let mut tki = tile_k;
        let args = [
            (&mut out) as *mut u64 as *mut core::ffi::c_void,
            (&mut weight) as *mut u64 as *mut core::ffi::c_void,
            (&mut wscale) as *mut u64 as *mut core::ffi::c_void,
            (&mut act) as *mut u64 as *mut core::ffi::c_void,
            (&mut act_scale) as *mut u64 as *mut core::ffi::c_void,
            (&mut ni) as *mut i32 as *mut core::ffi::c_void,
            (&mut ki) as *mut i32 as *mut core::ffi::c_void,
            (&mut tki) as *mut i32 as *mut core::ffi::c_void,
        ];
        rvllm_fused::launch_raw(fn_ktiled, (grid_x, 1, 1), block, smem, stream, &args)?;
    }

    gemma4_launcher::Bf16ToF16SatLaunch { n: n as u32 }.launch(
        fn_f32_to_f16,
        output_f16,
        scratch_f32,
        stream,
    )
}

#[cfg(feature = "cuda")]
unsafe fn rope_f16kv(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels<'_>,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    stream: u64,
) -> Result<()> {
    gemma4_launcher::FusedRopePartialF16KvLaunch {
        num_tokens: dims.num_tokens,
        num_heads: dims.num_heads,
        num_kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
        rotary_dim: dims.rotary_dim,
        rope_table_rows: dims.rope_table_rows,
        block_size: dims.block_size,
        num_blocks_total: dims.num_blocks_total,
    }
    .launch(
        kernels.fused_rope_partial_f16kv,
        scratch.q_normed,
        scratch.k_normed,
        scratch.v_normed,
        scratch.q_normed,
        scratch.k_cache,
        scratch.v_cache,
        meta.cos,
        meta.sin,
        meta.positions,
        meta.slot_mapping,
        stream,
    )
}

#[cfg(feature = "cuda")]
unsafe fn rope_fp8kv(
    dims: Gemma4LayerDims,
    kernels: &Gemma4LayerKernels<'_>,
    scratch: &Gemma4LayerScratch,
    meta: &Gemma4MetadataPtrs,
    stream: u64,
) -> Result<()> {
    gemma4_launcher::FusedRopePartialFp8KvLaunch {
        num_tokens: dims.num_tokens,
        num_heads: dims.num_heads,
        num_kv_heads: dims.num_kv_heads,
        head_dim: dims.head_dim,
        rotary_dim: dims.rotary_dim,
        rope_table_rows: dims.rope_table_rows,
        block_size: dims.block_size,
        num_blocks_total: dims.num_blocks_total,
    }
    .launch(
        kernels.fused_rope_partial_fp8kv,
        scratch.q_normed,
        scratch.k_normed,
        scratch.v_normed,
        scratch.q_fp8,
        scratch.k_cache,
        scratch.v_cache,
        scratch.k_scale_cache,
        scratch.v_scale_cache,
        scratch.q_scale_cache,
        meta.cos,
        meta.sin,
        meta.positions,
        meta.slot_mapping,
        scratch.q_scale_ptr,
        stream,
    )
}

pub unsafe fn logit_softcap(
    kernel: &KernelFn,
    logits_ptr: u64,
    num_tokens: u32,
    vocab: u32,
    cap: f32,
    stream: u64,
) -> Result<()> {
    gemma4_launcher::LogitSoftcapLaunch {
        num_tokens,
        vocab,
        cap,
    }
    .launch(kernel, logits_ptr, stream)
}

#[cfg(test)]
mod fp8_gemv_arch_tests {
    use super::fp8_channelscale_gemv_supported;

    #[test]
    fn rejects_sm80_noop_kernel() {
        assert!(!fp8_channelscale_gemv_supported(7, 5));
        assert!(!fp8_channelscale_gemv_supported(8, 0));
        assert!(!fp8_channelscale_gemv_supported(8, 6));
        assert!(!fp8_channelscale_gemv_supported(8, 7));
        assert!(fp8_channelscale_gemv_supported(8, 9));
        assert!(fp8_channelscale_gemv_supported(9, 0));
        assert!(fp8_channelscale_gemv_supported(12, 1));
    }
}
