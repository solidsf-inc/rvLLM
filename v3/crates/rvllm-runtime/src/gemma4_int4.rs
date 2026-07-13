//! INT4 decoder GEMMs and a pruned language-model-head greedy tail.
//!
//! Both paths are opt-in and have fail-closed `_REQUIRE` flags:
//!
//! 1. **Decoder GEMMs through w4a8.** The 4 decoder Linears (QKV, O, gate_up,
//!    down) are compressed-tensors *pack-quantized* (group 128, 4-bit signed
//!    symmetric). The Hopper INT4 path is `rvllm-cutlass`'s
//!    `W4a8Lib` (`rvllm_w4a8_gemm_run`), which expects a DIFFERENT on-disk
//!    layout: AWQ-reordered INT4 in the kernel's `LayoutB_Reordered` atom +
//!    packed FP8 E4M3 LUT group scales. We therefore convert the
//!    compressed-tensors pack -> w4a8 layout **offline at load** via
//!    `W4a8Lib::encode_fp16` (dequant the pack to FP16, hand it to the encoder).
//!    `route_decoder_gemm_w4a8` is the per-Linear call-site that mirrors the
//!    `CublasLt::fp8_gemm` shape so `gemma4_layer_exec` can swap by a flag.
//!
//! 2. **Pruned lm-head greedy tail.** Serve the lm-head as the *pruned* head
//!    of kept rows. For greedy decode we only need argmax over those rows,
//!    then remap the winning local index -> global token id via
//!    `keep_ids` (kernel `lmhead_prune_argmax.cu`). The full-vocab `-inf`
//!    scatter is logprobs-only and lives behind `RVLLM_LMHEAD_FULLVOCAB`.
//!    Scores are bf16-rounded with a left tie-break to preserve greedy identity.
//!
//! `WPacked` is the canonical pack-quantized weight view owned by `rvllm-loader`
//! (`WPacked { packed, scale, shape, packed_cols, scale_groups, group_size,
//! num_bits, symmetric }`). This module now re-exports it
//! (`pub use rvllm_loader::WPacked;`) so the loader's output flows straight
//! into the INT4 forward without a translation copy. The i32-typed kernel-ABI
//! views the dequant/encode call-sites need (`out_features`, `in_features`,
//! `group_size`, `packed_cols`, `num_groups`) plus the fail-closed `validate()`
//! live on the local `WPackedExt` extension trait below.
//! Field semantics:
//!   - `packed`: `[out_features, in_features/8]` i32, 8 signed int4 per i32,
//!     LSB-first (nibble j of lane p is logical col `p*8+j`), values in [-8,7].
//!   - `scale`:  `[out_features, in_features/group_size]` f16 per-group scale
//!     (channel strategy for lm_head => group_size == in_features => 1 col).
//!   - `shape`:  `[out_features, in_features]` (logical).

use rvllm_core::{ConfigError, Result, RvllmError};
#[cfg(feature = "cuda")]
use rvllm_kernels::KernelFn;

fn invalid(field: &'static str, reason: impl Into<String>) -> RvllmError {
    RvllmError::config(
        ConfigError::InvalidField {
            name: field,
            reason: reason.into(),
        },
        field,
    )
}

fn checked_i32(value: usize, field: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| invalid(field, format!("{value} exceeds i32::MAX")))
}

fn require_ptr(ptr: u64, field: &'static str) -> Result<()> {
    if ptr == 0 {
        return Err(invalid(field, "device pointer is null"));
    }
    Ok(())
}

// ===========================================================================
// WPacked — compressed-tensors pack-quantized weight view.
// ===========================================================================

pub use rvllm_loader::WPacked;

/// i32 kernel-ABI views + fail-closed layout validation for the canonical
/// loader `WPacked`. Kept local (extension trait) so the loader type stays a
/// plain data carrier while the INT4 forward gets the typed accessors its
/// `launch_raw` marshalling and the w4a8 encoder expect.
pub(crate) trait WPackedExt {
    fn out_features_i32(&self) -> Result<i32>;
    fn in_features_i32(&self) -> Result<i32>;
    fn group_size_i32(&self) -> Result<i32>;
    fn num_groups_i32(&self) -> Result<i32>;
    fn packed_cols_i32(&self) -> Result<i32>;
    /// Hard-fail on a layout that doesn't match the pack-quantized invariant.
    fn validate(&self) -> Result<()>;
}

impl WPackedExt for WPacked {
    fn out_features_i32(&self) -> Result<i32> {
        checked_i32(self.out_features(), "wpacked.out_features")
    }
    fn in_features_i32(&self) -> Result<i32> {
        checked_i32(self.in_features(), "wpacked.in_features")
    }
    fn group_size_i32(&self) -> Result<i32> {
        checked_i32(self.group_size, "wpacked.group_size")
    }
    fn num_groups_i32(&self) -> Result<i32> {
        checked_i32(self.scale_groups, "wpacked.scale_groups")
    }
    fn packed_cols_i32(&self) -> Result<i32> {
        checked_i32(self.packed_cols, "wpacked.packed_cols")
    }
    fn validate(&self) -> Result<()> {
        let inf = self.in_features();
        let out = self.out_features();
        if out == 0 || inf == 0 {
            return Err(invalid("wpacked.shape", "dimensions must be positive"));
        }
        self.out_features_i32()?;
        self.in_features_i32()?;
        require_ptr(self.packed, "wpacked.packed")?;
        require_ptr(self.scale, "wpacked.scale")?;
        if self.num_bits != 4 {
            return Err(invalid(
                "wpacked.num_bits",
                format!("expected 4, got {}", self.num_bits),
            ));
        }
        if !self.symmetric {
            return Err(invalid(
                "wpacked.symmetric",
                "asymmetric INT4 is unsupported",
            ));
        }
        if inf % 8 != 0 {
            return Err(invalid(
                "wpacked.in_features",
                "dimension is not divisible by 8",
            ));
        }
        let expected_packed_cols = inf / 8;
        if self.packed_cols != expected_packed_cols {
            return Err(invalid(
                "wpacked.packed_cols",
                format!("expected {expected_packed_cols}, got {}", self.packed_cols),
            ));
        }
        if self.group_size == 0 || inf % self.group_size != 0 {
            return Err(invalid(
                "wpacked.group_size",
                "must be positive and divide in_features",
            ));
        }
        let expected_scale_groups = inf / self.group_size;
        if self.scale_groups != expected_scale_groups {
            return Err(invalid(
                "wpacked.scale_groups",
                format!(
                    "expected {expected_scale_groups}, got {}",
                    self.scale_groups
                ),
            ));
        }
        self.group_size_i32()?;
        self.packed_cols_i32()?;
        self.num_groups_i32()?;
        out.checked_mul(self.packed_cols)
            .ok_or_else(|| invalid("wpacked.packed", "element count overflows usize"))?;
        out.checked_mul(self.scale_groups)
            .ok_or_else(|| invalid("wpacked.scale", "element count overflows usize"))?;
        Ok(())
    }
}

// ===========================================================================
// Environment flags with fail-closed _REQUIRE twins.
// ===========================================================================

/// Returns whether `name` is set to "1".
fn flag(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

/// `RVLLM_INT4` / `RVLLM_INT4_REQUIRE`. `engaged` is whether the w4a8 fast path
/// actually wired (lib loaded + weights encoded). If `_REQUIRE=1` and it did
/// not engage, hard-fail instead of silently falling back.
pub fn int4_requested() -> bool {
    flag("RVLLM_INT4")
}
pub fn int4_required() -> bool {
    flag("RVLLM_INT4_REQUIRE")
}
pub fn lmhead_prune_requested() -> bool {
    flag("RVLLM_LMHEAD_PRUNE")
}
pub fn lmhead_prune_required() -> bool {
    flag("RVLLM_LMHEAD_PRUNE_REQUIRE")
}
/// Logprobs-only full-vocab `-inf` scatter (off by default; greedy never needs it).
pub fn lmhead_fullvocab() -> bool {
    flag("RVLLM_LMHEAD_FULLVOCAB")
}

/// Assert a required fast path engaged, else hard-fail with a clear message.
pub fn enforce_required(flag_engaged: bool, required: bool, what: &str) -> Result<()> {
    if required && !flag_engaged {
        return Err(invalid(
            "int4_require",
            format!("REQUIRE set but {what} did not engage"),
        ));
    }
    Ok(())
}

// ===========================================================================
// Decoder GEMM through w4a8 — encoded handle + call-site.
// ===========================================================================

/// One decoder Linear, encoded into the w4a8 kernel's layout. Produced once at
/// load by `encode_from_pack`. Holds device pointers to the reordered INT4
/// weight + packed FP8 LUT scales sized for this `[N, K]`.
#[derive(Copy, Clone, Debug)]
pub struct Int4LinearW4a8 {
    /// `[K, N]` INT4 ColMajor AWQ-shuffled (LayoutB_Reordered). Device ptr.
    pub b_int4_reordered: u64,
    /// `[N, K/group]` packed FP8 LUT blocks. Device ptr.
    pub b_scales_packed: u64,
    pub n: i32,
    pub k: i32,
    pub group_size: i32,
}

/// Convert a compressed-tensors `WPacked` decoder weight into the w4a8 layout.
///
/// The `W4a8Lib::encode_fp16` entry takes FP16 `[N, K]` and emits the
/// reordered INT4 + packed-FP8-LUT-scale layout the kernel reads. The
/// compressed-tensors pack is a *different* encoding (i32 nibbles + f16 group
/// scales), so we first **dequant the pack to FP16 on device**
/// (`dequant_pack_to_f16` kernel), then hand that to `encode_fp16`. This is the
/// pack -> w4a8 reorder goes through the canonical
/// encoder rather than a bespoke shuffle, so the kernel-side atom layout stays
/// authoritative.
///
/// Buffers (`dst_int4`, `dst_scales_packed`, `f16_scratch`, `scales_f32_ws`)
/// must be pre-sized by the caller:
///   - `f16_scratch`:  N*K*2 bytes (dequant target, consumed by encode)
///   - `dst_int4`:     N*(K/2) bytes
///   - `dst_scales_packed`: N*(K/group)*8 bytes
///   - `scales_f32_ws`: N*(K/group)*4 bytes
///
/// # Safety
/// All device pointers valid for the call; dims match `wp`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn encode_from_pack(
    w4a8: &rvllm_cutlass::W4a8Lib,
    fn_dequant_pack_to_f16: &KernelFn,
    wp: &WPacked,
    f16_scratch: u64,
    dst_int4: u64,
    dst_scales_packed: u64,
    scales_f32_ws: u64,
    stream: u64,
) -> Result<Int4LinearW4a8> {
    wp.validate()?;
    require_ptr(f16_scratch, "int4_encode.f16_scratch")?;
    require_ptr(dst_int4, "int4_encode.dst_int4")?;
    require_ptr(dst_scales_packed, "int4_encode.dst_scales_packed")?;
    require_ptr(scales_f32_ws, "int4_encode.scales_f32_ws")?;
    let n = wp.out_features_i32()?;
    let k = wp.in_features_i32()?;
    let group_size = wp.group_size_i32()?;
    // 1) dequant compressed-tensors pack -> FP16 [N, K] (row-major).
    dequant_pack_to_f16(
        fn_dequant_pack_to_f16,
        wp.packed,
        wp.scale,
        f16_scratch,
        n,
        k,
        group_size,
        stream,
    )?;
    // 2) encode FP16 [N, K] -> reordered INT4 + packed FP8 LUT scales.
    w4a8.encode_fp16(
        f16_scratch,
        n,
        k,
        group_size,
        dst_int4,
        dst_scales_packed,
        scales_f32_ws,
        true, // shuffle into LayoutB_Reordered
        stream,
    )?;
    Ok(Int4LinearW4a8 {
        b_int4_reordered: dst_int4,
        b_scales_packed: dst_scales_packed,
        n,
        k,
        group_size,
    })
}

/// Route ONE decoder GEMM through the w4a8 Hopper INT4 path:
/// `D = act_fp8 * W_int4^T (+ residual)`. Mirrors `CublasLt::fp8_gemm` so the
/// dispatcher in `gemma4_layer_exec` swaps FP8<->INT4 by a per-linear flag
/// without changing the call-site shape.
///
/// - `a_fp8` / `a_is_residual`: M×K row-major E4M3 activations (already
///   per-token quantized by the same fused rmsnorm+fp8-quant kernel the FP8
///   path uses). w4a8 takes A in FP8 like the FP8 GEMM — the A side is
///   unchanged, only B is INT4.
/// - `c_f16` / `beta`: optional residual (O-proj and down-proj add residual;
///   pass `c=0, beta=0.0` for QKV / gate_up).
/// - `d_f16`: M×N row-major output.
///
/// # Safety
/// All device pointers valid; `workspace_bytes >= workspace_size(m, n, k)`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn route_decoder_gemm_w4a8(
    w4a8: &rvllm_cutlass::W4a8Lib,
    lin: &Int4LinearW4a8,
    a_fp8: u64,
    c_f16: u64,
    d_f16: u64,
    m: i32,
    alpha: f32,
    beta: f32,
    workspace: u64,
    workspace_bytes: usize,
    stream: u64,
) -> Result<()> {
    require_ptr(lin.b_int4_reordered, "int4_gemm.weight")?;
    require_ptr(lin.b_scales_packed, "int4_gemm.scales")?;
    require_ptr(a_fp8, "int4_gemm.activations")?;
    require_ptr(d_f16, "int4_gemm.output")?;
    if m <= 0 || lin.n <= 0 || lin.k <= 0 || lin.group_size <= 0 {
        return Err(invalid(
            "int4_gemm.shape",
            "all dimensions must be positive",
        ));
    }
    if lin.k % lin.group_size != 0 {
        return Err(invalid("int4_gemm.group_size", "group_size must divide k"));
    }
    if !alpha.is_finite() || !beta.is_finite() {
        return Err(invalid("int4_gemm.scale", "alpha and beta must be finite"));
    }
    if beta != 0.0 {
        require_ptr(c_f16, "int4_gemm.residual")?;
    }
    let needed = w4a8.workspace_size(m, lin.n, lin.k);
    if workspace_bytes < needed {
        return Err(invalid(
            "int4_gemm.workspace_bytes",
            format!("need {needed} bytes, got {workspace_bytes}"),
        ));
    }
    if needed != 0 {
        require_ptr(workspace, "int4_gemm.workspace")?;
    }
    w4a8.w4a8_gemm(
        a_fp8,
        lin.b_int4_reordered,
        lin.b_scales_packed,
        c_f16,
        d_f16,
        m,
        lin.n,
        lin.k,
        lin.group_size,
        alpha,
        beta,
        workspace,
        workspace_bytes,
        stream,
    )
}

// ===========================================================================
// Per-layer INT4 decoder weights — encoded handles + per-step dispatch.
//
// The E4B forward routes the 4 logical decoder GEMMs through w4a8. The
// pack-quantized checkpoint stores them as 7 separate Linears (q/k/v, o,
// gate/up, down); the FP8 skeleton FUSES q||k||v into one `qkv` matrix and
// gate||up into one `gate_up`. To match the FP8 layer body's buffer shapes
// (`scratch.q_out` carries the full `qkv_rows`, `scratch.gate_up_out` the
// `2*intermediate`), we ENCODE the fused `[N,K]` rows: the loader's f16
// dequant scratch writes q,k,v stacked along N then hands the fused matrix
// to `encode_fp16`. A KV-shared tail layer has no k/v Linear, so its "qkv"
// is Q-only (`k_proj`/`v_proj == None`); the encoded N is then just `q_dim`.
// ===========================================================================

/// The 4 encoded w4a8 decoder GEMMs for one E4B layer, sized to match the
/// FP8 skeleton's fused buffer shapes so the layer body swaps FP8<->INT4 by
/// a per-linear pointer without changing the kernel call shape.
///
/// `qkv` N = `q_dim + 2*kv_dim` (own-KV layer) or `q_dim` (KV-shared layer).
/// `gate_up` N = `2*intermediate`. `o`/`down` add residual via `beta=1`.
#[derive(Copy, Clone, Debug)]
pub struct Int4DecoderLayer {
    pub qkv: Int4LinearW4a8,
    pub o: Int4LinearW4a8,
    pub gate_up: Int4LinearW4a8,
    pub down: Int4LinearW4a8,
}

/// Whole-model INT4 decoder runtime: per-layer encoded GEMMs + the pruned
/// lm-head + a shared w4a8 GEMM workspace. `Some` on the bring-up only when
/// `RVLLM_E4B && RVLLM_INT4 && is_e4b()` AND the w4a8 `.so` loaded and the
/// per-layer weights encoded. The layer body reads `layers[layer_idx]`; the
/// lm-head tail reads `lm_head`. Fail-closed: if this is `None` and
/// `RVLLM_INT4_REQUIRE=1`, the bring-up hard-errors (no FP8 placeholder run).
#[derive(Clone, Debug)]
pub struct Gemma4Int4Runtime {
    pub layers: Vec<Int4DecoderLayer>,
    pub lm_head: LmHeadPruned,
    /// Shared GEMM workspace device ptr (sized for the largest decoder GEMM).
    pub workspace: u64,
    pub workspace_bytes: usize,
}

/// Per-layer INT4 dispatch handle threaded into the E4B layer forward so the
/// 4 GEMM sites route through w4a8 instead of the FP8 branches. Holds borrows
/// to the (arch-agnostic) `W4a8Lib` + this layer's encoded weights + the
/// shared GEMM workspace. The 31B path passes `None` so its forward is
/// byte-identical; on the E4B+INT4 path the caller passes `Some`.
#[derive(Copy, Clone)]
pub struct Int4LayerExec<'a> {
    pub w4a8: &'a rvllm_cutlass::W4a8Lib,
    pub layer: &'a Int4DecoderLayer,
    pub workspace: u64,
    pub workspace_bytes: usize,
}

// ===========================================================================
// Pruned lm-head greedy tail: GEMV(int4) -> bf16 score -> argmax -> remap.
// ===========================================================================

/// The pruned lm-head as served for greedy decode.
#[derive(Clone, Debug)]
pub struct LmHeadPruned {
    /// Pack-quantized kept-row weight: `[K_rows, hidden]`, channel strategy
    /// (group_size == hidden, one scale per row).
    pub w: WPacked,
    /// `[K_rows]` i32 device: local kept row -> global token id.
    pub keep_ids: u64,
    /// Number of kept rows.
    pub k_rows: i32,
    /// Full vocabulary size, used only for the logprobs scatter.
    pub full_vocab: i32,
}

/// Kernel symbol names in `lmhead_prune_argmax.cu`.
pub const FN_LMHEAD_INT4_GEMV: &str = "lmhead_int4_gemv_bf16score_kernel";
pub const FN_LMHEAD_ARGMAX_REMAP: &str = "lmhead_argmax_remap_kernel";
pub const FN_LMHEAD_SCATTER_FULLVOCAB: &str = "lmhead_scatter_full_vocab_kernel";
/// Dequant helper used by `encode_from_pack`.
pub const FN_DEQUANT_PACK_TO_F16: &str = "dequant_pack_to_f16_kernel";

#[cfg(feature = "cuda")]
const LMH_WARPS_PER_BLOCK: u32 = 8;
#[cfg(feature = "cuda")]
const LMH_ARGMAX_BLOCK: u32 = 1024;

#[cfg(feature = "cuda")]
fn validate_head(head: &LmHeadPruned) -> Result<()> {
    head.w.validate()?;
    require_ptr(head.keep_ids, "lmhead.keep_ids")?;
    if head.k_rows <= 0 || head.full_vocab <= 0 {
        return Err(invalid(
            "lmhead.shape",
            "k_rows and full_vocab must be positive",
        ));
    }
    if head.k_rows > head.full_vocab {
        return Err(invalid("lmhead.k_rows", "cannot exceed full_vocab"));
    }
    if head.w.out_features() != head.k_rows as usize {
        return Err(invalid(
            "lmhead.k_rows",
            format!(
                "expected {} rows from the weight, got {}",
                head.w.out_features(),
                head.k_rows
            ),
        ));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn validate_head_launch(head: &LmHeadPruned, m: u32, hidden: u32) -> Result<(i32, i32, u32)> {
    validate_head(head)?;
    if m == 0 || hidden == 0 {
        return Err(invalid("lmhead.shape", "m and hidden must be positive"));
    }
    let m_i = i32::try_from(m).map_err(|_| invalid("lmhead.m", "exceeds i32::MAX"))?;
    let hidden_i =
        i32::try_from(hidden).map_err(|_| invalid("lmhead.hidden", "exceeds i32::MAX"))?;
    if head.w.in_features() != hidden as usize {
        return Err(invalid(
            "lmhead.hidden",
            format!(
                "expected {} from the weight, got {hidden}",
                head.w.in_features()
            ),
        ));
    }
    if head.w.group_size != hidden as usize {
        return Err(invalid(
            "lmhead.group_size",
            "pruned head requires one quantization group per row",
        ));
    }
    let smem = hidden
        .checked_mul(4)
        .ok_or_else(|| invalid("lmhead.hidden", "shared-memory size overflows u32"))?;
    Ok((m_i, hidden_i, smem))
}

/// Pruned greedy lm-head: writes `out_token[m] = global token id of argmax over
/// kept rows`. `scores` is `[M, K_rows]` f32 scratch (values are bf16-rounded
/// row scores; argmax is identity). `residual` is `[M, hidden]` f16 final-normed.
///
/// # Safety
/// Caller owns all device pointers; `scores` sized `M*K_rows*4` bytes.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn lmhead_prune_argmax(
    fn_gemv: &KernelFn,
    fn_argmax_remap: &KernelFn,
    head: &LmHeadPruned,
    residual: u64,
    scores: u64,
    out_token: u64,
    m: u32,
    hidden: u32,
    stream: u64,
) -> Result<()> {
    require_ptr(residual, "lmhead.residual")?;
    require_ptr(scores, "lmhead.scores")?;
    require_ptr(out_token, "lmhead.out_token")?;
    let (m_i, hidden_i, smem) = validate_head_launch(head, m, hidden)?;
    // K1: int4 GEMV -> bf16-rounded scores.
    {
        let mut residual = residual;
        let mut packed = head.w.packed;
        let mut scale = head.w.scale;
        let mut scores = scores;
        let mut m_i = m_i;
        let mut k_rows = head.k_rows;
        let mut hidden_i = hidden_i;
        let mut group = head.w.group_size_i32()?;
        let args = [
            (&mut residual) as *mut u64 as *mut core::ffi::c_void,
            (&mut packed) as *mut u64 as *mut core::ffi::c_void,
            (&mut scale) as *mut u64 as *mut core::ffi::c_void,
            (&mut scores) as *mut u64 as *mut core::ffi::c_void,
            (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
            (&mut k_rows) as *mut i32 as *mut core::ffi::c_void,
            (&mut hidden_i) as *mut i32 as *mut core::ffi::c_void,
            (&mut group) as *mut i32 as *mut core::ffi::c_void,
        ];
        let grid = ((head.k_rows as u32).div_ceil(LMH_WARPS_PER_BLOCK), m, 1);
        let block = (LMH_WARPS_PER_BLOCK * 32, 1, 1);
        rvllm_fused::launch_raw(fn_gemv, grid, block, smem, stream, &args)?;
    }
    // K2: argmax over kept rows (left tie-break) + remap -> global token id.
    {
        let mut scores = scores;
        let mut keep_ids = head.keep_ids;
        let mut out_token = out_token;
        let mut m_i = m_i;
        let mut k_rows = head.k_rows;
        let args = [
            (&mut scores) as *mut u64 as *mut core::ffi::c_void,
            (&mut keep_ids) as *mut u64 as *mut core::ffi::c_void,
            (&mut out_token) as *mut u64 as *mut core::ffi::c_void,
            (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
            (&mut k_rows) as *mut i32 as *mut core::ffi::c_void,
        ];
        let grid = (m, 1, 1);
        let block = (LMH_ARGMAX_BLOCK, 1, 1);
        rvllm_fused::launch_raw(fn_argmax_remap, grid, block, 0, stream, &args)?;
    }
    Ok(())
}

/// INT4 pruned-head GEMV ONLY: `scores[M, K_rows] = bf16round(residual·W_row)`,
/// no argmax. Used by the PPL/logprobs path which needs the kept-row scores
/// then a full-vocab scatter (greedy decode uses `lmhead_prune_argmax`, which
/// fuses the argmax). Same K1 kernel as `lmhead_prune_argmax`.
///
/// # Safety
/// Caller owns all device pointers; `scores` sized `M*K_rows*4` bytes.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn lmhead_int4_scores(
    fn_gemv: &KernelFn,
    head: &LmHeadPruned,
    residual: u64,
    scores: u64,
    m: u32,
    hidden: u32,
    stream: u64,
) -> Result<()> {
    require_ptr(residual, "lmhead.residual")?;
    require_ptr(scores, "lmhead.scores")?;
    let (m_i, hidden_i, smem) = validate_head_launch(head, m, hidden)?;
    let mut residual = residual;
    let mut packed = head.w.packed;
    let mut scale = head.w.scale;
    let mut scores = scores;
    let mut m_i = m_i;
    let mut k_rows = head.k_rows;
    let mut hidden_i = hidden_i;
    let mut group = head.w.group_size_i32()?;
    let args = [
        (&mut residual) as *mut u64 as *mut core::ffi::c_void,
        (&mut packed) as *mut u64 as *mut core::ffi::c_void,
        (&mut scale) as *mut u64 as *mut core::ffi::c_void,
        (&mut scores) as *mut u64 as *mut core::ffi::c_void,
        (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
        (&mut k_rows) as *mut i32 as *mut core::ffi::c_void,
        (&mut hidden_i) as *mut i32 as *mut core::ffi::c_void,
        (&mut group) as *mut i32 as *mut core::ffi::c_void,
    ];
    let grid = ((head.k_rows as u32).div_ceil(LMH_WARPS_PER_BLOCK), m, 1);
    let block = (LMH_WARPS_PER_BLOCK * 32, 1, 1);
    rvllm_fused::launch_raw(fn_gemv, grid, block, smem, stream, &args)
}

/// Scatter pruned `scores[M, K_rows]` into a full-vocab f32 logits buffer
/// `[M, full_vocab]` with `-inf` at non-kept columns. Logprobs/PPL only.
/// Kernel: `lmhead_scatter_full_vocab_kernel(scores, keep_ids, out, M, K_rows,
/// full_vocab)`. The caller must pre-fill `out` with `-inf` OR the kernel does
/// (matches `scatter_full_vocab_ref`: every non-kept slot ends `-inf`).
///
/// # Safety
/// Caller owns all device pointers; `out` sized `M*full_vocab*4` bytes.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn lmhead_scatter_full_vocab(
    fn_scatter: &KernelFn,
    head: &LmHeadPruned,
    scores: u64,
    out_f32: u64,
    m: u32,
    stream: u64,
) -> Result<()> {
    require_ptr(scores, "lmhead.scores")?;
    require_ptr(out_f32, "lmhead.full_vocab_output")?;
    validate_head(head)?;
    if m == 0 {
        return Err(invalid("lmhead.m", "must be positive"));
    }
    let m_i = i32::try_from(m).map_err(|_| invalid("lmhead.m", "exceeds i32::MAX"))?;
    let mut scores = scores;
    let mut keep_ids = head.keep_ids;
    let mut out = out_f32;
    let mut m_i = m_i;
    let mut k_rows = head.k_rows;
    let mut full_vocab = head.full_vocab;
    let args = [
        (&mut scores) as *mut u64 as *mut core::ffi::c_void,
        (&mut keep_ids) as *mut u64 as *mut core::ffi::c_void,
        (&mut out) as *mut u64 as *mut core::ffi::c_void,
        (&mut m_i) as *mut i32 as *mut core::ffi::c_void,
        (&mut k_rows) as *mut i32 as *mut core::ffi::c_void,
        (&mut full_vocab) as *mut i32 as *mut core::ffi::c_void,
    ];
    // One block per (row); enough threads to cover full_vocab init + scatter.
    let grid = (m, 1, 1);
    let block = (LMH_ARGMAX_BLOCK, 1, 1);
    rvllm_fused::launch_raw(fn_scatter, grid, block, 0, stream, &args)
}

/// Dequant a compressed-tensors pack-quantized weight `[N, K]` to FP16 row-major
/// on device. Kernel: `dequant_pack_to_f16_kernel(packed, scale, out_f16, N, K,
/// group_size)`. Used by `encode_from_pack` before the w4a8 encoder.
///
/// # Safety
/// Caller owns pointers; `out_f16` sized `N*K*2` bytes.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dequant_pack_to_f16(
    kernel: &KernelFn,
    packed: u64,
    scale: u64,
    out_f16: u64,
    n: i32,
    k: i32,
    group_size: i32,
    stream: u64,
) -> Result<()> {
    require_ptr(packed, "int4_dequant.packed")?;
    require_ptr(scale, "int4_dequant.scale")?;
    require_ptr(out_f16, "int4_dequant.output")?;
    if n <= 0 || k <= 0 || group_size <= 0 {
        return Err(invalid(
            "int4_dequant.shape",
            "n, k, and group_size must be positive",
        ));
    }
    if k % 8 != 0 {
        return Err(invalid("int4_dequant.k", "must be divisible by 8"));
    }
    if k % group_size != 0 {
        return Err(invalid("int4_dequant.group_size", "must divide k"));
    }
    let mut packed = packed;
    let mut scale = scale;
    let mut out_f16 = out_f16;
    let mut n = n;
    let mut k = k;
    let mut group_size = group_size;
    let args = [
        (&mut packed) as *mut u64 as *mut core::ffi::c_void,
        (&mut scale) as *mut u64 as *mut core::ffi::c_void,
        (&mut out_f16) as *mut u64 as *mut core::ffi::c_void,
        (&mut n) as *mut i32 as *mut core::ffi::c_void,
        (&mut k) as *mut i32 as *mut core::ffi::c_void,
        (&mut group_size) as *mut i32 as *mut core::ffi::c_void,
    ];
    let grid = (n as u32, 1, 1);
    let block = ((k as u32).min(1024), 1, 1);
    rvllm_fused::launch_raw(kernel, grid, block, 0, stream, &args)
}

// ===========================================================================
// PURE-RUST REFERENCES (ground truth for the .cu kernels; no GPU).
// ===========================================================================

/// Round f32 -> bf16 (round-to-nearest-even) -> f32. Matches the torch bf16
/// cast applied to language-model-head GEMM scores.
pub fn bf16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    // round-to-nearest-even on the truncation boundary
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// Unpack one compressed-tensors int32 lane into 8 signed int4 (LSB-first).
/// Nibble `j` is logical column `lane_index*8 + j`. Values in `[-8, 7]`.
pub fn unpack_i32_to_int4(lane: u32) -> [i32; 8] {
    let mut out = [0i32; 8];
    for (j, o) in out.iter_mut().enumerate() {
        let nib = (lane >> (4 * j)) & 0xF;
        *o = if nib >= 8 {
            nib as i32 - 16
        } else {
            nib as i32
        };
    }
    out
}

/// Dequant a pack-quantized row to f32. `packed_row` is `in_features/8` i32
/// lanes; `scales` is `in_features/group_size` f32. Returns `in_features` f32.
pub fn dequant_pack_row_ref(
    packed_row: &[u32],
    scales: &[f32],
    in_features: usize,
    group_size: usize,
) -> Result<Vec<f32>> {
    if in_features == 0 || in_features % 8 != 0 {
        return Err(invalid(
            "dequant_ref.in_features",
            "must be positive and divisible by 8",
        ));
    }
    if group_size == 0 || in_features % group_size != 0 {
        return Err(invalid(
            "dequant_ref.group_size",
            "must be positive and divide in_features",
        ));
    }
    let packed_cols = in_features / 8;
    let scale_groups = in_features / group_size;
    if packed_row.len() != packed_cols {
        return Err(invalid(
            "dequant_ref.packed_row",
            format!("expected {packed_cols} lanes, got {}", packed_row.len()),
        ));
    }
    if scales.len() != scale_groups {
        return Err(invalid(
            "dequant_ref.scales",
            format!("expected {scale_groups} values, got {}", scales.len()),
        ));
    }
    if scales.iter().any(|scale| !scale.is_finite()) {
        return Err(invalid("dequant_ref.scales", "values must be finite"));
    }
    let mut w = Vec::new();
    w.try_reserve_exact(in_features)
        .map_err(|_| invalid("dequant_ref.in_features", "allocation failed"))?;
    w.resize(in_features, 0.0);
    for (p, &lane) in packed_row.iter().enumerate() {
        let nibs = unpack_i32_to_int4(lane);
        let base = p * 8;
        for (j, &q) in nibs.iter().enumerate() {
            let col = base + j;
            let g = col / group_size;
            w[col] = q as f32 * scales[g];
        }
    }
    Ok(w)
}

/// Reference for `lmhead_int4_gemv_bf16score_kernel`: per kept row, dequant the
/// pack-quantized weight, dot with the f32 residual, then bf16-round the score.
/// `residual` is `[M, hidden]`, `packed` is `[K_rows, hidden/8]`, `scale` is
/// `[K_rows, hidden/group]`. Writes `scores [M, K_rows]`.
#[allow(clippy::too_many_arguments)]
pub fn lmhead_int4_gemv_ref(
    residual: &[f32],
    packed: &[u32],
    scale: &[f32],
    m: usize,
    k_rows: usize,
    hidden: usize,
    group_size: usize,
    scores: &mut [f32],
) -> Result<()> {
    if m == 0 || k_rows == 0 {
        return Err(invalid("lmhead_ref.shape", "m and k_rows must be positive"));
    }
    if hidden == 0 || hidden % 8 != 0 {
        return Err(invalid(
            "lmhead_ref.hidden",
            "must be positive and divisible by 8",
        ));
    }
    if group_size == 0 || hidden % group_size != 0 {
        return Err(invalid(
            "lmhead_ref.group_size",
            "must be positive and divide hidden",
        ));
    }
    let packed_cols = hidden / 8;
    let num_groups = hidden / group_size;
    let residual_len = m
        .checked_mul(hidden)
        .ok_or_else(|| invalid("lmhead_ref.residual", "length overflows usize"))?;
    let packed_len = k_rows
        .checked_mul(packed_cols)
        .ok_or_else(|| invalid("lmhead_ref.packed", "length overflows usize"))?;
    let scale_len = k_rows
        .checked_mul(num_groups)
        .ok_or_else(|| invalid("lmhead_ref.scale", "length overflows usize"))?;
    let scores_len = m
        .checked_mul(k_rows)
        .ok_or_else(|| invalid("lmhead_ref.scores", "length overflows usize"))?;
    for (field, got, expected) in [
        ("lmhead_ref.residual", residual.len(), residual_len),
        ("lmhead_ref.packed", packed.len(), packed_len),
        ("lmhead_ref.scale", scale.len(), scale_len),
        ("lmhead_ref.scores", scores.len(), scores_len),
    ] {
        if got != expected {
            return Err(invalid(field, format!("expected {expected}, got {got}")));
        }
    }
    if residual.iter().any(|value| !value.is_finite()) {
        return Err(invalid("lmhead_ref.residual", "values must be finite"));
    }
    for mi in 0..m {
        let res = &residual[mi * hidden..(mi + 1) * hidden];
        for r in 0..k_rows {
            let prow = &packed[r * packed_cols..(r + 1) * packed_cols];
            let srow = &scale[r * num_groups..(r + 1) * num_groups];
            let w = dequant_pack_row_ref(prow, srow, hidden, group_size)?;
            let acc: f32 = w.iter().zip(res).map(|(a, b)| a * b).sum();
            scores[mi * k_rows + r] = bf16_round(acc);
        }
    }
    Ok(())
}

/// Reference for `lmhead_argmax_remap_kernel`: left-tie-break argmax over the
/// kept rows then remap the winning local row to the global token id. Returns
/// `[M]` global token ids.
pub fn argmax_remap_ref(
    scores: &[f32],
    keep_ids: &[i32],
    m: usize,
    k_rows: usize,
) -> Result<Vec<i32>> {
    if m == 0 || k_rows == 0 {
        return Err(invalid("argmax_ref.shape", "m and k_rows must be positive"));
    }
    let scores_len = m
        .checked_mul(k_rows)
        .ok_or_else(|| invalid("argmax_ref.scores", "length overflows usize"))?;
    if scores.len() != scores_len {
        return Err(invalid(
            "argmax_ref.scores",
            format!("expected {scores_len}, got {}", scores.len()),
        ));
    }
    if keep_ids.len() != k_rows {
        return Err(invalid(
            "argmax_ref.keep_ids",
            format!("expected {k_rows}, got {}", keep_ids.len()),
        ));
    }
    if keep_ids.iter().any(|id| *id < 0) {
        return Err(invalid(
            "argmax_ref.keep_ids",
            "token ids must be nonnegative",
        ));
    }
    if scores.iter().any(|score| score.is_nan()) {
        return Err(invalid("argmax_ref.scores", "NaN scores are unsupported"));
    }
    let mut out = vec![0i32; m];
    for mi in 0..m {
        let row = &scores[mi * k_rows..(mi + 1) * k_rows];
        let mut best = f32::NEG_INFINITY;
        let mut arg = 0usize;
        for (i, &v) in row.iter().enumerate() {
            if v > best {
                // strict > => smallest index wins on ties (left tie-break)
                best = v;
                arg = i;
            }
        }
        out[mi] = keep_ids[arg];
    }
    Ok(out)
}

/// Reference for `lmhead_scatter_full_vocab_kernel` (logprobs only): scatter the
/// `[M, K_rows]` pruned scores into `[M, full_vocab]` with `-inf` at non-kept
/// columns.
pub fn scatter_full_vocab_ref(
    scores: &[f32],
    keep_ids: &[i32],
    m: usize,
    k_rows: usize,
    full_vocab: usize,
) -> Result<Vec<f32>> {
    if m == 0 || k_rows == 0 || full_vocab == 0 {
        return Err(invalid(
            "scatter_ref.shape",
            "m, k_rows, and full_vocab must be positive",
        ));
    }
    let scores_len = m
        .checked_mul(k_rows)
        .ok_or_else(|| invalid("scatter_ref.scores", "length overflows usize"))?;
    if scores.len() != scores_len {
        return Err(invalid(
            "scatter_ref.scores",
            format!("expected {scores_len}, got {}", scores.len()),
        ));
    }
    if keep_ids.len() != k_rows {
        return Err(invalid(
            "scatter_ref.keep_ids",
            format!("expected {k_rows}, got {}", keep_ids.len()),
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(k_rows);
    for &gid in keep_ids {
        let gid = usize::try_from(gid)
            .map_err(|_| invalid("scatter_ref.keep_ids", "token ids must be nonnegative"))?;
        if gid >= full_vocab {
            return Err(invalid(
                "scatter_ref.keep_ids",
                format!("token id {gid} is outside full_vocab {full_vocab}"),
            ));
        }
        if !seen.insert(gid) {
            return Err(invalid(
                "scatter_ref.keep_ids",
                format!("duplicate token id {gid}"),
            ));
        }
    }
    let out_len = m
        .checked_mul(full_vocab)
        .ok_or_else(|| invalid("scatter_ref.output", "length overflows usize"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(out_len)
        .map_err(|_| invalid("scatter_ref.output", "allocation failed"))?;
    out.resize(out_len, f32::NEG_INFINITY);
    for mi in 0..m {
        for (local, &gid) in keep_ids.iter().enumerate() {
            out[mi * full_vocab + gid as usize] = scores[mi * k_rows + local];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_signed_int4_lsb_first() {
        // nibbles LSB-first: 0x7=7, 0x9=-7, 0x6=6, 0xA=-6.
        let lane = 0x7 | (0x9 << 4) | (0x6 << 8) | (0xA << 12); // [7,-7,6,-6,...]
        let v = unpack_i32_to_int4(lane);
        assert_eq!(&v[..4], &[7, -7, 6, -6]);
        // full signed range
        let lane2 = 0x8 | (0x7 << 4); // [-8, 7, 0, ...]
        let v2 = unpack_i32_to_int4(lane2);
        assert_eq!(v2[0], -8);
        assert_eq!(v2[1], 7);
    }

    #[test]
    fn dequant_pack_row_matches_manual() {
        // one group, group_size=8 (one lane), scale 0.5: int4 [-8..7]*0.5
        let lane: u32 = (0u32)
            | (1 << 4)
            | (2 << 8)
            | (3 << 12)
            | (4 << 16)
            | (5 << 20)
            | (6 << 24)
            | (7 << 28);
        let w = dequant_pack_row_ref(&[lane], &[0.5], 8, 8).unwrap();
        assert_eq!(w, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
    }

    #[test]
    fn dequant_two_groups_uses_per_group_scale() {
        // in=16, group=8 -> 2 lanes, 2 scales. lane0 all 1s *scale0, lane1 all 2 *scale1
        let lane_ones: u32 = (1u32) * 0x1111_1111; // all nibbles = 1
        let lane_twos: u32 = (2u32) * 0x1111_1111; // all nibbles = 2
        let w = dequant_pack_row_ref(&[lane_ones, lane_twos], &[10.0, 100.0], 16, 8).unwrap();
        assert_eq!(&w[..8], &[10.0; 8]);
        assert_eq!(&w[8..], &[200.0; 8]);
    }

    #[test]
    fn bf16_round_is_monotonic_and_truncating() {
        // exact bf16-representable values round to themselves
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(-2.0), -2.0);
        // round-to-nearest-even on a value with low mantissa bits
        let x = f32::from_bits(0x3f80_8001); // 1.0 + tiny
        let r = bf16_round(x);
        // bf16 keeps the top 7 mantissa bits; result is a valid bf16 value
        assert_eq!(r.to_bits() & 0xffff, 0);
        // monotonic: a < b => round(a) <= round(b)
        assert!(bf16_round(1.0) <= bf16_round(1.01));
        assert!(bf16_round(-1.01) <= bf16_round(-1.0));
    }

    /// GOLDEN: local argmax i -> keep_ids[i]. The whole point of the prune tail.
    #[test]
    fn argmax_remap_golden_local_to_global() {
        // M=2, K_rows=4. keep_ids maps kept rows to scattered global ids.
        let keep_ids = [100, 5, 42, 9999];
        // row0 winner is local 2 -> global 42; row1 winner local 0 -> global 100
        let scores = [
            0.1, 0.2, 0.9, 0.3, // m0 argmax = 2
            0.5, 0.4, 0.1, 0.2, // m1 argmax = 0
        ];
        let got = argmax_remap_ref(&scores, &keep_ids, 2, 4).unwrap();
        assert_eq!(got, vec![42, 100]);
    }

    /// Left tie-break: equal scores -> smallest local index -> its global id.
    #[test]
    fn argmax_remap_left_tie_break() {
        let keep_ids = [7, 8, 9, 10];
        let scores = [0.5, 0.5, 0.5, 0.1]; // three-way tie, smallest index 0 wins
        let got = argmax_remap_ref(&scores, &keep_ids, 1, 4).unwrap();
        assert_eq!(got, vec![7]);
    }

    /// End-to-end: int4 GEMV (bf16 score) then argmax-remap, against a fully
    /// independent manual computation.
    #[test]
    fn gemv_then_argmax_remap_end_to_end() {
        let hidden = 8usize;
        let group = 8usize;
        let k_rows = 3usize;
        let m = 1usize;
        // residual: distinct positive values
        let residual = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // row0: all int4=1, scale 1.0  -> dot = sum(res) = 36
        // row1: all int4=2, scale 1.0  -> dot = 2*36 = 72  (winner)
        // row2: all int4=-1, scale 1.0 -> dot = -36
        let lane1 = 1u32 * 0x1111_1111;
        let lane2 = 2u32 * 0x1111_1111;
        let lane_m1 = 0xFu32 * 0x1111_1111; // nibble 0xF = -1
        let packed = vec![lane1, lane2, lane_m1];
        let scale = vec![1.0, 1.0, 1.0];
        let keep_ids = [11, 22, 33];
        let mut scores = vec![0f32; m * k_rows];
        lmhead_int4_gemv_ref(
            &residual,
            &packed,
            &scale,
            m,
            k_rows,
            hidden,
            group,
            &mut scores,
        )
        .unwrap();
        assert_eq!(scores[0], bf16_round(36.0));
        assert_eq!(scores[1], bf16_round(72.0));
        assert_eq!(scores[2], bf16_round(-36.0));
        let tok = argmax_remap_ref(&scores, &keep_ids, m, k_rows).unwrap();
        assert_eq!(tok, vec![22]); // row1 wins -> global 22
    }

    #[test]
    fn scatter_full_vocab_fills_neg_inf() {
        let scores = [0.3, 0.7];
        let keep_ids = [2, 5];
        let full = scatter_full_vocab_ref(&scores, &keep_ids, 1, 2, 6).unwrap();
        assert_eq!(full[2], 0.3);
        assert_eq!(full[5], 0.7);
        assert!(full[0].is_infinite() && full[0] < 0.0);
        assert!(full[1].is_infinite());
        assert!(full[3].is_infinite());
    }

    #[test]
    fn references_reject_invalid_shapes_and_ids() {
        assert!(dequant_pack_row_ref(&[], &[1.0], 8, 8).is_err());
        assert!(dequant_pack_row_ref(&[0], &[1.0], 8, 0).is_err());
        assert!(argmax_remap_ref(&[], &[], 1, 0).is_err());
        assert!(argmax_remap_ref(&[f32::NAN], &[0], 1, 1).is_err());
        assert!(scatter_full_vocab_ref(&[1.0, 2.0], &[1, 1], 1, 2, 4).is_err());
        assert!(scatter_full_vocab_ref(&[1.0], &[4], 1, 1, 4).is_err());
    }

    #[test]
    fn wpacked_validate_rejects_bad_layout() {
        let bad = WPacked {
            packed: 0,
            scale: 0,
            shape: [16, 13], // in_features 13 not /8
            packed_cols: 1,
            scale_groups: 1,
            group_size: 13,
            num_bits: 4,
            symmetric: true,
        };
        assert!(bad.validate().is_err());
        let inconsistent = WPacked {
            packed: 1,
            scale: 2,
            shape: [16, 16],
            packed_cols: 1,
            scale_groups: 2,
            group_size: 8,
            num_bits: 4,
            symmetric: true,
        };
        assert!(inconsistent.validate().is_err());
        let good = WPacked {
            packed: 1,
            scale: 2,
            shape: [16, 16],
            packed_cols: 2,
            scale_groups: 2,
            group_size: 8,
            num_bits: 4,
            symmetric: true,
        };
        assert!(good.validate().is_ok());
        assert_eq!(good.num_groups_i32().unwrap(), 2);
        assert_eq!(good.packed_cols_i32().unwrap(), 2);
    }

    #[test]
    fn require_twin_hard_fails_when_not_engaged() {
        assert!(enforce_required(false, true, "int4").is_err());
        assert!(enforce_required(true, true, "int4").is_ok());
        assert!(enforce_required(false, false, "int4").is_ok());
    }
}
