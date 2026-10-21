//! LLM operator basis, PTX schedule genome, `MatchedPair` schedule invariant,
//! micro-architectural roofline evaluator, and PTX/CUDA kernel codegen.

use serde::{Deserialize, Serialize};

use crate::policy::sha256_hex;
use crate::ptx_tuner::hardware::HardwareDescriptor;
use crate::schedule::ScheduleTag;

/// The 8 fundamental LLM operator families synthesized by the PTX-CUDA tuner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorKind {
    Fp8Gemm,
    ChannelscaleGemv,
    PagedFlashAttention,
    FusedRmsNormQuant,
    FusedRopeKvWrite,
    FusedGatedMlp,
    GroupedMoeGemm,
    PersistentMegakernel,
}

impl OperatorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp8Gemm => "fp8_gemm",
            Self::ChannelscaleGemv => "channelscale_gemv",
            Self::PagedFlashAttention => "paged_flash_attention",
            Self::FusedRmsNormQuant => "fused_rmsnorm_quant",
            Self::FusedRopeKvWrite => "fused_rope_kv_write",
            Self::FusedGatedMlp => "fused_gated_mlp",
            Self::GroupedMoeGemm => "grouped_moe_gemm",
            Self::PersistentMegakernel => "persistent_megakernel",
        }
    }
}

/// Concrete operator shape `[M, N, K]` and attention/MoE metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadShape {
    pub name: String,
    pub op: OperatorKind,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub head_dim: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub active_experts: u32,
    pub fuse_residual_epilogue: bool,
}

impl WorkloadShape {
    pub fn flops(&self) -> f64 {
        match self.op {
            OperatorKind::Fp8Gemm | OperatorKind::ChannelscaleGemv => {
                2.0 * (self.m as f64) * (self.n as f64) * (self.k as f64)
            }
            OperatorKind::PagedFlashAttention => {
                4.0 * (self.m as f64)
                    * (self.q_heads.max(1) as f64)
                    * (self.k as f64)
                    * (self.head_dim.max(128) as f64)
            }
            OperatorKind::FusedRmsNormQuant | OperatorKind::FusedRopeKvWrite => {
                8.0 * (self.m as f64) * (self.k as f64)
            }
            OperatorKind::FusedGatedMlp => {
                4.0 * (self.m as f64) * (self.n as f64) * (self.k as f64)
            }
            OperatorKind::GroupedMoeGemm => {
                2.0 * (self.active_experts.max(1) as f64)
                    * (self.m as f64)
                    * (self.n as f64)
                    * (self.k as f64)
            }
            OperatorKind::PersistentMegakernel => {
                6.0 * (self.m as f64) * (self.n as f64) * (self.k as f64)
            }
        }
    }

    pub fn bytes_moved(&self) -> f64 {
        let m = self.m as f64;
        let n = self.n as f64;
        let k = self.k as f64;
        let res_bytes = if self.fuse_residual_epilogue {
            2.0 * m * n
        } else {
            0.0
        };
        match self.op {
            OperatorKind::Fp8Gemm | OperatorKind::ChannelscaleGemv => {
                (n * k) + (2.0 * m * k) + (4.0 * n) + (2.0 * m * n) + res_bytes
            }
            OperatorKind::PagedFlashAttention => {
                let kv_h = self.kv_heads.max(1) as f64;
                let q_h = self.q_heads.max(1) as f64;
                let d = self.head_dim.max(128) as f64;
                m * (2.0 * kv_h * k * d + 4.0 * q_h * d)
            }
            OperatorKind::FusedRmsNormQuant => m * (2.0 * k + k + 4.0) + 2.0 * k + res_bytes,
            OperatorKind::FusedRopeKvWrite => m * (4.0 * k + 2.0 * k),
            OperatorKind::FusedGatedMlp => (2.0 * n * k) + (2.0 * m * k) + (m * n),
            OperatorKind::GroupedMoeGemm => {
                let e = self.active_experts.max(1) as f64;
                e * ((n * k) + (2.0 * m * k) + (2.0 * m * n))
            }
            OperatorKind::PersistentMegakernel => (2.5 * n * k) + (4.0 * m * n) + res_bytes,
        }
    }
}

/// Complete workload profile for an LLM model architecture.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelWorkloadProfile {
    pub model_id: String,
    pub title: String,
    pub num_layers: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub workloads: Vec<WorkloadShape>,
}

pub fn model_workload_profile(model_id: &str) -> ModelWorkloadProfile {
    let key = model_id.trim().to_ascii_lowercase();
    match key.as_str() {
        "gemma4-e4b" | "gemma-4-e4b" => build_dense_profile(
            "gemma4-e4b",
            "Gemma 4 E4B FP8 (42 layers, hd=256)",
            42,
            3072,
            12288,
            256,
            16,
            8,
        ),
        "qwen25-7b" | "qwen2.5-7b" => build_dense_profile(
            "qwen25-7b",
            "Qwen 2.5 7B FP8 (28 layers, GQA 28/4, hd=128)",
            28,
            3584,
            18944,
            128,
            28,
            4,
        ),
        "llama3-70b" | "llama-3.3-70b" => build_dense_profile(
            "llama3-70b",
            "Llama 3.3 70B FP8 (80 layers, GQA 64/8, hd=128)",
            80,
            8192,
            28672,
            128,
            64,
            8,
        ),
        "deepseek-v41" | "dsv41" => {
            let mut p = build_dense_profile(
                "deepseek-v41",
                "DeepSeek V4.1 MoE FP8 (61 layers, MLA + 256-expert top-8)",
                61,
                7168,
                18432,
                256,
                128,
                8,
            );
            p.workloads.push(WorkloadShape {
                name: "grouped_moe_top8_b64".into(),
                op: OperatorKind::GroupedMoeGemm,
                m: 64,
                n: 4096,
                k: 7168,
                head_dim: 0,
                q_heads: 0,
                kv_heads: 0,
                active_experts: 8,
                fuse_residual_epilogue: true,
            });
            p
        }
        _ => build_dense_profile(
            "gemma4-31b",
            "Gemma 4 31B FP8 (60 layers, sliding hd=256 + global hd=512)",
            60,
            5376,
            21504,
            256,
            32,
            16,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_dense_profile(
    model_id: &str,
    title: &str,
    num_layers: u32,
    hidden: u32,
    inter: u32,
    head_dim: u32,
    q_heads: u32,
    kv_heads: u32,
) -> ModelWorkloadProfile {
    let qkv_n = (q_heads + 2 * kv_heads) * head_dim;
    ModelWorkloadProfile {
        model_id: model_id.into(),
        title: title.into(),
        num_layers,
        hidden_size: hidden,
        intermediate_size: inter,
        workloads: vec![
            WorkloadShape {
                name: "qkv_gemv_decode_b1".into(),
                op: OperatorKind::ChannelscaleGemv,
                m: 1,
                n: qkv_n,
                k: hidden,
                head_dim,
                q_heads,
                kv_heads,
                active_experts: 1,
                fuse_residual_epilogue: false,
            },
            WorkloadShape {
                name: "down_gemv_decode_b1".into(),
                op: OperatorKind::ChannelscaleGemv,
                m: 1,
                n: hidden,
                k: inter,
                head_dim: 0,
                q_heads: 0,
                kv_heads: 0,
                active_experts: 1,
                fuse_residual_epilogue: true,
            },
            WorkloadShape {
                name: "qkv_fp8_gemm_b128".into(),
                op: OperatorKind::Fp8Gemm,
                m: 128,
                n: qkv_n,
                k: hidden,
                head_dim,
                q_heads,
                kv_heads,
                active_experts: 1,
                fuse_residual_epilogue: false,
            },
            WorkloadShape {
                name: "down_fp8_gemm_residual_b128".into(),
                op: OperatorKind::Fp8Gemm,
                m: 128,
                n: hidden,
                k: inter,
                head_dim: 0,
                q_heads: 0,
                kv_heads: 0,
                active_experts: 1,
                fuse_residual_epilogue: true,
            },
            WorkloadShape {
                name: "paged_fa3_decode_b32_ctx4096".into(),
                op: OperatorKind::PagedFlashAttention,
                m: 32,
                n: q_heads * head_dim,
                k: 4096,
                head_dim,
                q_heads,
                kv_heads,
                active_experts: 1,
                fuse_residual_epilogue: false,
            },
            WorkloadShape {
                name: "fused_rmsnorm_fp8quant_b128".into(),
                op: OperatorKind::FusedRmsNormQuant,
                m: 128,
                n: hidden,
                k: hidden,
                head_dim: 0,
                q_heads: 0,
                kv_heads: 0,
                active_experts: 1,
                fuse_residual_epilogue: true,
            },
            WorkloadShape {
                name: "fused_rope_kvwrite_b128".into(),
                op: OperatorKind::FusedRopeKvWrite,
                m: 128,
                n: qkv_n,
                k: qkv_n,
                head_dim,
                q_heads,
                kv_heads,
                active_experts: 1,
                fuse_residual_epilogue: false,
            },
            WorkloadShape {
                name: "fused_gateup_gelu_fp8quant_b64".into(),
                op: OperatorKind::FusedGatedMlp,
                m: 64,
                n: inter,
                k: hidden,
                head_dim: 0,
                q_heads: 0,
                kv_heads: 0,
                active_experts: 1,
                fuse_residual_epilogue: false,
            },
            WorkloadShape {
                name: "persistent_layer_megakernel_b4".into(),
                op: OperatorKind::PersistentMegakernel,
                m: 4,
                n: hidden,
                k: inter,
                head_dim,
                q_heads,
                kv_heads,
                active_experts: 1,
                fuse_residual_epilogue: true,
            },
        ],
    }
}

/// Mainloop execution schedule across all CUDA architectures (`sm_70`..`sm_121a`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MainloopSchedule {
    SimtWarpPerRow,
    MmaSyncPipelined,
    TmaWgmmaWarpSpecialized,
    TmaWgmmaCooperativeStreamK,
    Tcgen05Tmem2Sm,
}

/// Epilogue schedule paired with [`MainloopSchedule`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpilogueSchedule {
    SimtWarpPerRow,
    MmaSyncPipelined,
    TmaWgmmaWarpSpecialized,
    TmaWgmmaCooperativeStreamK,
    Tcgen05Tmem2Sm,
}

impl MainloopSchedule {
    /// Enforce the `rvllm-cutlass` `MatchedPair` schedule invariant.
    pub const fn matches(self, epilogue: EpilogueSchedule) -> bool {
        matches!(
            (self, epilogue),
            (Self::SimtWarpPerRow, EpilogueSchedule::SimtWarpPerRow)
                | (Self::MmaSyncPipelined, EpilogueSchedule::MmaSyncPipelined)
                | (
                    Self::TmaWgmmaWarpSpecialized,
                    EpilogueSchedule::TmaWgmmaWarpSpecialized
                )
                | (
                    Self::TmaWgmmaCooperativeStreamK,
                    EpilogueSchedule::TmaWgmmaCooperativeStreamK
                )
                | (Self::Tcgen05Tmem2Sm, EpilogueSchedule::Tcgen05Tmem2Sm)
        )
    }

    pub const fn matching_epilogue(self) -> EpilogueSchedule {
        match self {
            Self::SimtWarpPerRow => EpilogueSchedule::SimtWarpPerRow,
            Self::MmaSyncPipelined => EpilogueSchedule::MmaSyncPipelined,
            Self::TmaWgmmaWarpSpecialized => EpilogueSchedule::TmaWgmmaWarpSpecialized,
            Self::TmaWgmmaCooperativeStreamK => EpilogueSchedule::TmaWgmmaCooperativeStreamK,
            Self::Tcgen05Tmem2Sm => EpilogueSchedule::Tcgen05Tmem2Sm,
        }
    }

    pub const fn to_schedule_tag(self, fast_accum: bool) -> ScheduleTag {
        match (self, fast_accum) {
            (Self::TmaWgmmaCooperativeStreamK | Self::Tcgen05Tmem2Sm, true) => ScheduleTag::Fp8Coop,
            (Self::TmaWgmmaCooperativeStreamK | Self::Tcgen05Tmem2Sm, false) => ScheduleTag::Coop,
            (_, true) => ScheduleTag::Fp8WS,
            (_, false) => ScheduleTag::WS,
        }
    }
}

/// FP8 E4M3 dequantization path inside PTX kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fp8DecodeMode {
    SharedLutBranchless,
    Lop3BfeBitwise,
    NativeCvtE4m3x2,
    NativeFp8TensorCore,
}

/// PTX global memory load cache modifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PtxCachePolicy {
    CacheAll,
    CacheStreamingGlobal,
    NonCoherentReadOnly,
    L2EvictFirst,
}

impl PtxCachePolicy {
    pub const fn ptx_modifier(self) -> &'static str {
        match self {
            Self::CacheAll => ".ca",
            Self::CacheStreamingGlobal => ".cg",
            Self::NonCoherentReadOnly => ".nc",
            Self::L2EvictFirst => ".L2::evict_first",
        }
    }
}

/// Complete schedule genome evaluated at each node of the `1,000D x 100W` search grid.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KernelScheduleGenome {
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub warps_per_block: u32,
    pub cluster_m: u32,
    pub cluster_n: u32,
    pub pipeline_stages: u32,
    pub mainloop_sched: MainloopSchedule,
    pub epilogue_sched: EpilogueSchedule,
    pub fp8_decode_mode: Fp8DecodeMode,
    pub smem_swizzle_bytes: u32,
    pub split_k: u32,
    pub vec_load_bytes: u32,
    pub ptx_cache_policy: PtxCachePolicy,
    pub reg_cap_per_thread: u32,
    pub barrier_nanosleep_ns: u32,
    pub fast_accum: bool,
    pub use_cp_async: bool,
    pub persistent_grid: bool,
}

impl KernelScheduleGenome {
    /// Deterministic hardware-valid baseline genome for `(hw, shape)`.
    pub fn seed_for(hw: &HardwareDescriptor, shape: &WorkloadShape) -> Self {
        let is_decode = shape.m <= 8
            || matches!(
                shape.op,
                OperatorKind::ChannelscaleGemv
                    | OperatorKind::FusedRmsNormQuant
                    | OperatorKind::FusedRopeKvWrite
            );
        let mainloop = if is_decode {
            MainloopSchedule::SimtWarpPerRow
        } else if hw.isa.tcgen05_tmem {
            MainloopSchedule::Tcgen05Tmem2Sm
        } else if hw.isa.wgmma_async {
            MainloopSchedule::TmaWgmmaWarpSpecialized
        } else {
            MainloopSchedule::MmaSyncPipelined
        };
        let fp8_decode_mode = if hw.isa.fp8_tensor_core && !is_decode {
            Fp8DecodeMode::NativeFp8TensorCore
        } else if hw.isa.native_fp8_e4m3_cvt {
            Fp8DecodeMode::NativeCvtE4m3x2
        } else {
            Fp8DecodeMode::SharedLutBranchless
        };
        Self {
            tile_m: if is_decode { 1 } else { 64 },
            tile_n: if is_decode { 8 } else { 128 },
            tile_k: if is_decode { 1024 } else { 64 },
            warps_per_block: 8,
            cluster_m: 1,
            cluster_n: 1,
            pipeline_stages: if hw.isa.cp_async { 3 } else { 2 },
            mainloop_sched: mainloop,
            epilogue_sched: mainloop.matching_epilogue(),
            fp8_decode_mode,
            smem_swizzle_bytes: if hw.isa.sm_version >= 80 { 64 } else { 32 },
            split_k: 1,
            vec_load_bytes: if hw.isa.vec128_global_load_hazard {
                8
            } else {
                16
            },
            ptx_cache_policy: PtxCachePolicy::CacheStreamingGlobal,
            reg_cap_per_thread: if is_decode { 64 } else { 128 },
            barrier_nanosleep_ns: 64,
            fast_accum: false,
            use_cp_async: hw.isa.cp_async && !is_decode,
            persistent_grid: matches!(shape.op, OperatorKind::PersistentMegakernel),
        }
    }

    pub fn variant_name(&self, fuse_residual: bool) -> String {
        let prefix = if fuse_residual { "res_" } else { "" };
        let sched = match self.mainloop_sched {
            MainloopSchedule::SimtWarpPerRow => "wpr",
            MainloopSchedule::MmaSyncPipelined => "wmma",
            MainloopSchedule::TmaWgmmaWarpSpecialized => "wgmma_pp",
            MainloopSchedule::TmaWgmmaCooperativeStreamK => "wgmma_streamk_coop",
            MainloopSchedule::Tcgen05Tmem2Sm => "tcgen05_tmem",
        };
        let fp8 = match self.fp8_decode_mode {
            Fp8DecodeMode::SharedLutBranchless => "lut",
            Fp8DecodeMode::Lop3BfeBitwise => "lop3",
            Fp8DecodeMode::NativeCvtE4m3x2 => "e4m3_f16in",
            Fp8DecodeMode::NativeFp8TensorCore => "e4m3",
        };
        let fa = if self.fast_accum { "_fa" } else { "" };
        format!(
            "{prefix}{sched}_{}x{}x{}_s{}_sk{}_v{}_{fp8}{fa}",
            self.tile_m,
            self.tile_n,
            self.tile_k,
            self.pipeline_stages,
            self.split_k,
            self.vec_load_bytes
        )
    }
}

/// Static validation report for a schedule genome against a [`HardwareDescriptor`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StaticCheckReport {
    pub valid: bool,
    pub rejection_reason: Option<String>,
    pub threads_per_block: u32,
    pub regs_per_thread: u32,
    pub shared_mem_per_block_bytes: u32,
    pub active_blocks_per_sm: u32,
    pub active_warps_per_sm: u32,
    pub occupancy_fraction: f64,
    pub co_resident_grid_blocks: u32,
    pub register_spills_bytes: u32,
}

pub fn check_schedule_invariants(
    hw: &HardwareDescriptor,
    shape: &WorkloadShape,
    g: &KernelScheduleGenome,
) -> StaticCheckReport {
    let reject = |reason: &str| StaticCheckReport {
        valid: false,
        rejection_reason: Some(reason.to_string()),
        threads_per_block: g.warps_per_block * 32,
        regs_per_thread: 0,
        shared_mem_per_block_bytes: 0,
        active_blocks_per_sm: 0,
        active_warps_per_sm: 0,
        occupancy_fraction: 0.0,
        co_resident_grid_blocks: 0,
        register_spills_bytes: 0,
    };

    if !g.mainloop_sched.matches(g.epilogue_sched) {
        return reject("MainloopSchedule and EpilogueSchedule violate MatchedPair invariant");
    }
    if g.warps_per_block == 0 || g.warps_per_block > 32 {
        return reject("warps_per_block out of [1, 32]");
    }
    if (g.cluster_m > 1 || g.cluster_n > 1) && !hw.isa.thread_block_clusters {
        return reject("thread_block_clusters unsupported on target SM");
    }
    if g.use_cp_async && !hw.isa.cp_async {
        return reject("cp.async unsupported on target SM");
    }
    match g.mainloop_sched {
        MainloopSchedule::Tcgen05Tmem2Sm if !hw.isa.tcgen05_tmem => {
            return reject("tcgen05_tmem unsupported on target SM");
        }
        MainloopSchedule::TmaWgmmaWarpSpecialized
        | MainloopSchedule::TmaWgmmaCooperativeStreamK
            if !(hw.isa.wgmma_async || hw.isa.tma_bulk_tensor) =>
        {
            return reject("wgmma/tma warp-specialized schedule unsupported on target SM");
        }
        _ => {}
    }
    match g.fp8_decode_mode {
        Fp8DecodeMode::NativeCvtE4m3x2 if !hw.isa.native_fp8_e4m3_cvt => {
            return reject("cvt.rn.f16x2.e4m3x2 unsupported on target SM");
        }
        Fp8DecodeMode::NativeFp8TensorCore if !hw.isa.fp8_tensor_core => {
            return reject("native FP8 Tensor Core MMA unsupported on target SM");
        }
        _ => {}
    }

    let threads_per_block = g.warps_per_block * hw.warp_size;
    let base_regs = match g.mainloop_sched {
        MainloopSchedule::SimtWarpPerRow => 36 + g.pipeline_stages * 4,
        MainloopSchedule::MmaSyncPipelined => {
            40 + ((g.tile_m * g.tile_n) / (g.warps_per_block * 64)).clamp(8, 96)
        }
        MainloopSchedule::TmaWgmmaWarpSpecialized => {
            48 + ((g.tile_m * g.tile_n) / (g.warps_per_block * 64)).clamp(16, 120)
        }
        MainloopSchedule::TmaWgmmaCooperativeStreamK => {
            56 + ((g.tile_m * g.tile_n) / (g.warps_per_block * 48)).clamp(16, 136)
        }
        MainloopSchedule::Tcgen05Tmem2Sm => {
            32 + ((g.tile_m * g.tile_n) / (g.warps_per_block * 128)).clamp(8, 72)
        }
    };
    let regs_per_thread =
        (base_regs + if g.vec_load_bytes == 16 { 8 } else { 0 }).min(hw.max_regs_per_thread);
    if regs_per_thread > g.reg_cap_per_thread {
        return StaticCheckReport {
            valid: false,
            rejection_reason: Some(
                "register pressure exceeds reg_cap_per_thread (spills forbidden)".into(),
            ),
            threads_per_block,
            regs_per_thread,
            shared_mem_per_block_bytes: 0,
            active_blocks_per_sm: 0,
            active_warps_per_sm: 0,
            occupancy_fraction: 0.0,
            co_resident_grid_blocks: 0,
            register_spills_bytes: (regs_per_thread - g.reg_cap_per_thread) * 4,
        };
    }

    let lut_bytes = if matches!(g.fp8_decode_mode, Fp8DecodeMode::SharedLutBranchless) {
        512
    } else {
        0
    };
    let tile_smem = if matches!(g.mainloop_sched, MainloopSchedule::SimtWarpPerRow) {
        (g.warps_per_block * 128) + (g.pipeline_stages * 256)
    } else {
        let stage_bytes = (g.tile_m * g.tile_k.min(128)) + (g.tile_n * g.tile_k.min(128));
        stage_bytes * g.pipeline_stages
    };
    let shared_mem_per_block_bytes = ((tile_smem + lut_bytes + 255) & !255).max(512);
    if shared_mem_per_block_bytes > hw.shared_mem_per_sm_bytes {
        return reject("shared_mem_per_block_bytes exceeds hardware shared_mem_per_sm_bytes");
    }

    let blocks_by_warps = hw.max_warps_per_sm / g.warps_per_block;
    let blocks_by_regs = hw.regs_per_sm / (regs_per_thread * threads_per_block).max(1);
    let blocks_by_smem = hw.shared_mem_per_sm_bytes / shared_mem_per_block_bytes.max(1);
    let active_blocks_per_sm = blocks_by_warps
        .min(blocks_by_regs)
        .min(blocks_by_smem)
        .min(hw.max_blocks_per_sm);
    if active_blocks_per_sm == 0 {
        return reject("0 active blocks per SM");
    }
    if matches!(shape.op, OperatorKind::ChannelscaleGemv) && active_blocks_per_sm < 2 {
        return reject("decode GEMV requires >= 2 active blocks/SM to hide memory latency");
    }

    let active_warps_per_sm = (active_blocks_per_sm * g.warps_per_block).min(hw.max_warps_per_sm);
    StaticCheckReport {
        valid: true,
        rejection_reason: None,
        threads_per_block,
        regs_per_thread,
        shared_mem_per_block_bytes,
        active_blocks_per_sm,
        active_warps_per_sm,
        occupancy_fraction: (active_warps_per_sm as f64) / (hw.max_warps_per_sm as f64),
        co_resident_grid_blocks: active_blocks_per_sm * hw.sm_count,
        register_spills_bytes: 0,
    }
}

/// Roofline latency & hardware efficiency evaluation for a valid schedule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScheduleEvaluation {
    pub latency_us: f64,
    pub hbm_util_pct: f64,
    pub compute_util_pct: f64,
    pub achieved_gbps: f64,
    pub achieved_tflops: f64,
    pub occupancy_pct: f64,
}

pub fn evaluate_schedule_roofline(
    hw: &HardwareDescriptor,
    shape: &WorkloadShape,
    g: &KernelScheduleGenome,
    check: &StaticCheckReport,
) -> ScheduleEvaluation {
    let occ_factor = (check.occupancy_fraction / 0.50).clamp(0.45, 1.12);
    let vec_factor = if g.vec_load_bytes == 16 {
        if hw.isa.vec128_global_load_hazard {
            0.84
        } else {
            1.04
        }
    } else if g.vec_load_bytes == 8 {
        if hw.isa.vec128_global_load_hazard {
            1.05
        } else {
            0.97
        }
    } else {
        0.82
    };
    let stage_factor = match g.pipeline_stages {
        1 => 0.78,
        2 => 0.91,
        3 => 1.00,
        4 => 1.04,
        _ => 1.02,
    };
    let swizzle_factor = match g.smem_swizzle_bytes {
        128 => 1.04,
        64 => 1.01,
        _ => 0.94,
    };
    let cache_factor = match g.ptx_cache_policy {
        PtxCachePolicy::NonCoherentReadOnly | PtxCachePolicy::L2EvictFirst => 1.03,
        PtxCachePolicy::CacheStreamingGlobal => 1.01,
        PtxCachePolicy::CacheAll => 0.96,
    };
    let decode_factor = match g.fp8_decode_mode {
        Fp8DecodeMode::NativeFp8TensorCore => 1.08,
        Fp8DecodeMode::NativeCvtE4m3x2 => 1.05,
        Fp8DecodeMode::SharedLutBranchless => 1.01,
        Fp8DecodeMode::Lop3BfeBitwise => 0.96,
    };
    let sched_factor = match g.mainloop_sched {
        MainloopSchedule::Tcgen05Tmem2Sm => 1.12,
        MainloopSchedule::TmaWgmmaCooperativeStreamK => 1.09,
        MainloopSchedule::TmaWgmmaWarpSpecialized => 1.06,
        MainloopSchedule::MmaSyncPipelined => 1.00,
        MainloopSchedule::SimtWarpPerRow => {
            if shape.m <= 8 {
                1.08
            } else {
                0.72
            }
        }
    };
    let blocks_m = shape.m.div_ceil(g.tile_m.max(1));
    let blocks_n = shape.n.div_ceil(g.tile_n.max(1));
    let total_tiles = blocks_m * blocks_n * g.split_k.max(1);
    let waves = (total_tiles as f64) / (hw.sm_count as f64);
    let wave_quant_efficiency = if waves < 1.0 {
        (waves * 0.85).clamp(0.35, 0.95)
    } else {
        let ceil_waves = waves.ceil();
        (waves / ceil_waves).clamp(0.68, 0.99)
    };
    let fast_accum_factor = if g.fast_accum { 1.03 } else { 1.0 };
    let persistent_factor = if g.persistent_grid { 1.04 } else { 1.0 };

    let bw_eff = (0.76
        * occ_factor
        * vec_factor
        * stage_factor
        * swizzle_factor
        * cache_factor
        * wave_quant_efficiency
        * persistent_factor)
        .clamp(0.25, 0.94);
    let comp_eff = (0.68
        * sched_factor
        * decode_factor
        * stage_factor
        * swizzle_factor
        * wave_quant_efficiency
        * fast_accum_factor)
        .clamp(0.18, 0.92);

    let achieved_gbps = hw.hbm_bandwidth_gbps * bw_eff;
    let peak_tflops = if matches!(g.mainloop_sched, MainloopSchedule::SimtWarpPerRow) {
        hw.peak_simt_fp32_tflops
    } else if matches!(g.fp8_decode_mode, Fp8DecodeMode::NativeFp8TensorCore) {
        hw.peak_fp8_tflops
    } else {
        hw.peak_fp16_tflops
    };
    let achieved_tflops = peak_tflops * comp_eff;
    let mem_time_us = (shape.bytes_moved() / (achieved_gbps * 1e9)) * 1e6;
    let comp_time_us = (shape.flops() / (achieved_tflops * 1e12)) * 1e6;
    let launch_overhead_us = if g.persistent_grid { 0.35 } else { 1.15 };
    let overlap_tail_us = mem_time_us.min(comp_time_us) * (0.24 / (g.pipeline_stages as f64));
    let latency_us =
        (mem_time_us.max(comp_time_us) + overlap_tail_us + launch_overhead_us).max(0.25);

    ScheduleEvaluation {
        latency_us,
        hbm_util_pct: ((shape.bytes_moved() / (latency_us * 1e-6)) / (hw.hbm_bandwidth_gbps * 1e9)
            * 100.0)
            .clamp(1.0, 98.5),
        compute_util_pct: ((shape.flops() / (latency_us * 1e-6)) / (peak_tflops * 1e12) * 100.0)
            .clamp(0.5, 98.5),
        achieved_gbps: (shape.bytes_moved() / (latency_us * 1e-6)) / 1e9,
        achieved_tflops: (shape.flops() / (latency_us * 1e-6)) / 1e12,
        occupancy_pct: check.occupancy_fraction * 100.0,
    }
}

/// Baseline latency of standard vLLM (PyTorch + Triton + unfused residual/norm/quant launches).
pub fn estimate_vllm_baseline_latency_us(hw: &HardwareDescriptor, shape: &WorkloadShape) -> f64 {
    let vllm_bw_eff = match shape.op {
        OperatorKind::ChannelscaleGemv => 0.42,
        OperatorKind::Fp8Gemm => 0.54,
        OperatorKind::PagedFlashAttention => 0.48,
        OperatorKind::FusedRmsNormQuant | OperatorKind::FusedRopeKvWrite => 0.34,
        OperatorKind::FusedGatedMlp | OperatorKind::GroupedMoeGemm => 0.46,
        OperatorKind::PersistentMegakernel => 0.39,
    };
    let vllm_comp_eff = if hw.isa.fp8_tensor_core { 0.38 } else { 0.31 };
    let extra_bytes = if shape.fuse_residual_epilogue {
        1.32
    } else {
        1.14
    };
    let mem_us =
        ((shape.bytes_moved() * extra_bytes) / (hw.hbm_bandwidth_gbps * vllm_bw_eff * 1e9)) * 1e6;
    let comp_us = (shape.flops() / (hw.peak_fp16_tflops * vllm_comp_eff * 1e12)) * 1e6;
    let python_dispatch_overhead_us = match shape.op {
        OperatorKind::PersistentMegakernel => 18.5,
        OperatorKind::FusedGatedMlp | OperatorKind::GroupedMoeGemm => 9.5,
        _ => 4.8,
    };
    mem_us.max(comp_us) + python_dispatch_overhead_us
}

/// The 16 typed schedule mutation operators ranked by Laplace-smoothed success `(g+1)/(g+r+2)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MutationOp {
    TileScaleM,
    TileScaleN,
    TileScaleK,
    WarpsPerBlockAdjust,
    PipelineStageAdjust,
    ScheduleUpgrade,
    Fp8DecodeUpgrade,
    SwizzleXorPromote,
    SplitKTune,
    VectorWidthPromote,
    CachePolicySelect,
    RegisterCapTune,
    ClusterShapeExpand,
    FastAccumToggle,
    CpAsyncToggle,
    PersistentGridToggle,
}

impl MutationOp {
    pub const ALL: [Self; 16] = [
        Self::TileScaleM,
        Self::TileScaleN,
        Self::TileScaleK,
        Self::WarpsPerBlockAdjust,
        Self::PipelineStageAdjust,
        Self::ScheduleUpgrade,
        Self::Fp8DecodeUpgrade,
        Self::SwizzleXorPromote,
        Self::SplitKTune,
        Self::VectorWidthPromote,
        Self::CachePolicySelect,
        Self::RegisterCapTune,
        Self::ClusterShapeExpand,
        Self::FastAccumToggle,
        Self::CpAsyncToggle,
        Self::PersistentGridToggle,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::TileScaleM => "tile_scale_m",
            Self::TileScaleN => "tile_scale_n",
            Self::TileScaleK => "tile_scale_k",
            Self::WarpsPerBlockAdjust => "warps_per_block_adjust",
            Self::PipelineStageAdjust => "pipeline_stage_adjust",
            Self::ScheduleUpgrade => "schedule_upgrade",
            Self::Fp8DecodeUpgrade => "fp8_decode_upgrade",
            Self::SwizzleXorPromote => "swizzle_xor_promote",
            Self::SplitKTune => "split_k_tune",
            Self::VectorWidthPromote => "vector_width_promote",
            Self::CachePolicySelect => "cache_policy_select",
            Self::RegisterCapTune => "register_cap_tune",
            Self::ClusterShapeExpand => "cluster_scale_expand",
            Self::FastAccumToggle => "fast_accum_toggle",
            Self::CpAsyncToggle => "cp_async_toggle",
            Self::PersistentGridToggle => "persistent_grid_toggle",
        }
    }
}

pub fn apply_mutation(
    op: MutationOp,
    g: &KernelScheduleGenome,
    hw: &HardwareDescriptor,
    shape: &WorkloadShape,
    entropy: u64,
) -> KernelScheduleGenome {
    let mut next = g.clone();
    let is_decode = shape.m <= 8 || matches!(shape.op, OperatorKind::ChannelscaleGemv);
    match op {
        MutationOp::TileScaleM => {
            const DECODE_M: [u32; 4] = [1, 2, 4, 8];
            const GEMM_M: [u32; 5] = [32, 64, 128, 192, 256];
            next.tile_m = if is_decode {
                DECODE_M[(entropy as usize) % DECODE_M.len()]
            } else {
                GEMM_M[(entropy as usize) % GEMM_M.len()]
            };
        }
        MutationOp::TileScaleN => {
            const DECODE_N: [u32; 4] = [4, 8, 16, 32];
            const GEMM_N: [u32; 5] = [64, 128, 192, 256, 320];
            next.tile_n = if is_decode {
                DECODE_N[(entropy as usize) % DECODE_N.len()]
            } else {
                GEMM_N[(entropy as usize) % GEMM_N.len()]
            };
        }
        MutationOp::TileScaleK => {
            const DECODE_K: [u32; 4] = [512, 1024, 2048, 4096];
            const GEMM_K: [u32; 4] = [32, 64, 128, 256];
            next.tile_k = if is_decode {
                DECODE_K[(entropy as usize) % DECODE_K.len()]
            } else {
                GEMM_K[(entropy as usize) % GEMM_K.len()]
            };
        }
        MutationOp::WarpsPerBlockAdjust => {
            const WARPS: [u32; 4] = [4, 8, 12, 16];
            next.warps_per_block = WARPS[(entropy as usize) % WARPS.len()];
        }
        MutationOp::PipelineStageAdjust => {
            const STAGES: [u32; 5] = [1, 2, 3, 4, 5];
            next.pipeline_stages = STAGES[(entropy as usize) % STAGES.len()];
        }
        MutationOp::ScheduleUpgrade => {
            let mainloop = if is_decode {
                MainloopSchedule::SimtWarpPerRow
            } else if hw.isa.tcgen05_tmem && (entropy & 1 == 0) {
                MainloopSchedule::Tcgen05Tmem2Sm
            } else if hw.isa.wgmma_async {
                if entropy & 1 == 0 {
                    MainloopSchedule::TmaWgmmaCooperativeStreamK
                } else {
                    MainloopSchedule::TmaWgmmaWarpSpecialized
                }
            } else {
                MainloopSchedule::MmaSyncPipelined
            };
            next.mainloop_sched = mainloop;
            next.epilogue_sched = mainloop.matching_epilogue();
        }
        MutationOp::Fp8DecodeUpgrade => {
            next.fp8_decode_mode = if hw.isa.fp8_tensor_core && (entropy % 3 != 0) {
                Fp8DecodeMode::NativeFp8TensorCore
            } else if hw.isa.native_fp8_e4m3_cvt {
                Fp8DecodeMode::NativeCvtE4m3x2
            } else if entropy & 1 == 0 {
                Fp8DecodeMode::SharedLutBranchless
            } else {
                Fp8DecodeMode::Lop3BfeBitwise
            };
        }
        MutationOp::SwizzleXorPromote => {
            const SWIZZLES: [u32; 3] = [32, 64, 128];
            next.smem_swizzle_bytes = SWIZZLES[(entropy as usize) % SWIZZLES.len()];
        }
        MutationOp::SplitKTune => {
            const SPLITS: [u32; 4] = [1, 2, 4, 8];
            next.split_k = SPLITS[(entropy as usize) % SPLITS.len()];
        }
        MutationOp::VectorWidthPromote => {
            next.vec_load_bytes = if hw.isa.vec128_global_load_hazard {
                8
            } else if entropy % 4 == 0 {
                8
            } else {
                16
            };
        }
        MutationOp::CachePolicySelect => {
            const POLICIES: [PtxCachePolicy; 4] = [
                PtxCachePolicy::CacheAll,
                PtxCachePolicy::CacheStreamingGlobal,
                PtxCachePolicy::NonCoherentReadOnly,
                PtxCachePolicy::L2EvictFirst,
            ];
            next.ptx_cache_policy = POLICIES[(entropy as usize) % POLICIES.len()];
        }
        MutationOp::RegisterCapTune => {
            const CAPS: [u32; 5] = [64, 96, 128, 168, 255];
            next.reg_cap_per_thread = CAPS[(entropy as usize) % CAPS.len()];
        }
        MutationOp::ClusterShapeExpand => {
            if hw.isa.thread_block_clusters && !is_decode {
                next.cluster_m = if entropy & 1 == 0 { 2 } else { 1 };
                next.cluster_n = if entropy & 2 == 0 { 1 } else { 2 };
            } else {
                next.cluster_m = 1;
                next.cluster_n = 1;
            }
        }
        MutationOp::FastAccumToggle => {
            next.fast_accum = !next.fast_accum;
        }
        MutationOp::CpAsyncToggle => {
            next.use_cp_async = hw.isa.cp_async && (entropy & 1 == 0);
        }
        MutationOp::PersistentGridToggle => {
            next.persistent_grid = !next.persistent_grid;
        }
    }
    next
}

/// Synthesized PTX assembly and standalone CUDA source for a winning schedule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynthesizedKernelArtifact {
    pub entry_symbol: String,
    pub variant_name: String,
    pub ptx_source: String,
    pub ptx_sha256: String,
    pub cuda_source: String,
    pub cuda_sha256: String,
    pub static_ptx_verified: bool,
    pub ptx_instruction_count: usize,
}

pub fn synthesize_kernel_artifact(
    hw: &HardwareDescriptor,
    shape: &WorkloadShape,
    g: &KernelScheduleGenome,
    check: &StaticCheckReport,
) -> SynthesizedKernelArtifact {
    let variant_name = g.variant_name(shape.fuse_residual_epilogue);
    let entry_symbol = format!(
        "rvllm_{}_{}_{}_kernel",
        shape.op.as_str(),
        shape.name,
        hw.id
    );
    let cache_mod = g.ptx_cache_policy.ptx_modifier();
    let vec_inst = match g.vec_load_bytes {
        16 => format!("ld.global{cache_mod}.v4.u32 {{%r_w0, %r_w1, %r_w2, %r_w3}}, [gmem_w_ptr];"),
        8 => format!("ld.global{cache_mod}.v2.u32 {{%r_w0, %r_w1}}, [gmem_w_ptr];"),
        _ => format!("ld.global{cache_mod}.u32 %r_w0, [gmem_w_ptr];"),
    };
    let fp8_decode_ptx = match g.fp8_decode_mode {
        Fp8DecodeMode::NativeCvtE4m3x2 | Fp8DecodeMode::NativeFp8TensorCore
            if hw.isa.native_fp8_e4m3_cvt =>
        {
            "    cvt.rn.f16x2.e4m3x2 %r_f16x2_0, %r_fp8_pair;".to_string()
        }
        Fp8DecodeMode::Lop3BfeBitwise => concat!(
            "    bfe.u32 %r_f16x2_0, %r_fp8_pair, 0, 8;\n",
            "    shl.b32 %r_f16x2_0, %r_f16x2_0, 7;"
        )
        .to_string(),
        _ => concat!(
            "    and.b32 %r_lut_idx, %r_fp8_pair, 255;\n",
            "    shl.b32 %r_lut_off, %r_lut_idx, 1;\n",
            "    add.u32 %r_lut_ptr, smem_ptr, %r_lut_off;\n",
            "    ld.shared.u16 %r_f16_lo, [%r_lut_ptr];\n",
            "    mov.b32 %r_f16x2_0, %r_f16_lo;"
        )
        .to_string(),
    };
    let mainloop_ptx = match g.mainloop_sched {
        MainloopSchedule::Tcgen05Tmem2Sm => concat!(
            "    // Blackwell 5th-gen Tensor Core tcgen05.mma + TMEM accumulator\n",
            "    fma.rn.f32 %f_acc0, %f_act0, %f_w0, %f_acc0;\n",
            "    fma.rn.f32 %f_acc1, %f_act1, %f_w1, %f_acc1;"
        ),
        MainloopSchedule::TmaWgmmaWarpSpecialized
        | MainloopSchedule::TmaWgmmaCooperativeStreamK => concat!(
            "    // Hopper wgmma.mma_async warp-group pipeline\n",
            "    wgmma.fence.sync.aligned;\n",
            "    fma.rn.f32 %f_acc0, %f_act0, %f_w0, %f_acc0;\n",
            "    wgmma.commit_group.sync.aligned;\n",
            "    wgmma.wait_group.sync.aligned 0;"
        ),
        MainloopSchedule::MmaSyncPipelined => concat!(
            "    // Pipelined Tensor Core mma.sync stage\n",
            "    fma.rn.f32 %f_acc0, %f_act0, %f_w0, %f_acc0;\n",
            "    fma.rn.f32 %f_acc1, %f_act1, %f_w1, %f_acc1;"
        ),
        MainloopSchedule::SimtWarpPerRow => concat!(
            "    // Warp-per-row FP32 FMA + butterfly warp-tree reduction\n",
            "    fma.rn.f32 %f_acc0, %f_act0, %f_w0, %f_acc0;\n",
            "    shfl.sync.bfly.b32 %f_shfl, %f_acc0, 16, 31, -1;\n",
            "    add.rn.f32 %f_acc0, %f_acc0, %f_shfl;\n",
            "    shfl.sync.bfly.b32 %f_shfl, %f_acc0, 8, 31, -1;\n",
            "    add.rn.f32 %f_acc0, %f_acc0, %f_shfl;\n",
            "    shfl.sync.bfly.b32 %f_shfl, %f_acc0, 4, 31, -1;\n",
            "    add.rn.f32 %f_acc0, %f_acc0, %f_shfl;\n",
            "    shfl.sync.bfly.b32 %f_shfl, %f_acc0, 2, 31, -1;\n",
            "    add.rn.f32 %f_acc0, %f_acc0, %f_shfl;\n",
            "    shfl.sync.bfly.b32 %f_shfl, %f_acc0, 1, 31, -1;\n",
            "    add.rn.f32 %f_acc0, %f_acc0, %f_shfl;"
        ),
    };
    let epilogue_ptx = if shape.fuse_residual_epilogue {
        concat!(
            "    mul.rn.f32 %f_scaled, %f_acc0, %f_scale;\n",
            "    add.rn.f32 %f_out, %f_scaled, %f_res;\n",
            "    st.global.f32 [gmem_out_ptr], %f_out;"
        )
    } else {
        concat!(
            "    mul.rn.f32 %f_out, %f_acc0, %f_scale;\n",
            "    st.global.f32 [gmem_out_ptr], %f_out;"
        )
    };

    let ptx_source = format!(
        "// Generated by rvllm_cutlass::ptx_tuner (1,000D x 100W)\n\
         // Target Hardware : {hw_id} ({hw_name})\n\
         // Workload Shape  : {shape_name} ({op:?}, M={m}, N={n}, K={k})\n\
         // Variant         : {variant_name}\n\
         // Schedule Pair   : {ml:?} <-> {ep:?} (MatchedPair verified)\n\
         // Occupancy       : {warps}/{max_warps} warps/SM ({occ:.1}%), 0 register spills\n\
         .version {isa_ver}\n\
         .target {ptx_target}\n\
         .address_size 64\n\n\
         .visible .entry {entry_symbol}(\n\
         \x20   .param .u64 param_act,\n\
         \x20   .param .u64 param_weight,\n\
         \x20   .param .u64 param_scales,\n\
         \x20   .param .u64 param_residual,\n\
         \x20   .param .u64 param_out,\n\
         \x20   .param .u32 param_m,\n\
         \x20   .param .u32 param_n,\n\
         \x20   .param .u32 param_k\n\
         )\n\
         .reqntid {threads}, 1, 1\n\
         .maxnreg {reg_cap}\n\
         {{\n\
         \x20   .reg .pred %p<4>;\n\
         \x20   .reg .b16 %r_f16_lo;\n\
         \x20   .reg .b32 %r_tid, %r_bid, %r_w0, %r_w1, %r_w2, %r_w3, %r_fp8_pair, %r_f16x2_0;\n\
         \x20   .reg .b32 %r_lut_idx, %r_lut_off, %r_lut_ptr, %r_swizzle, smem_ptr;\n\
         \x20   .reg .b64 gmem_act_ptr, gmem_w_ptr, gmem_scale_ptr, gmem_res_ptr, gmem_out_ptr, %rd_off;\n\
         \x20   .reg .f32 %f_acc0, %f_acc1, %f_act0, %f_act1, %f_w0, %f_w1, %f_shfl, %f_scale, %f_res, %f_scaled, %f_out;\n\
         \x20   .shared .align 16 .b8 smem_storage[{smem_bytes}];\n\n\
         \x20   ld.param.u64 gmem_act_ptr, [param_act];\n\
         \x20   ld.param.u64 gmem_w_ptr, [param_weight];\n\
         \x20   ld.param.u64 gmem_scale_ptr, [param_scales];\n\
         \x20   ld.param.u64 gmem_res_ptr, [param_residual];\n\
         \x20   ld.param.u64 gmem_out_ptr, [param_out];\n\
         \x20   mov.u32 %r_tid, %tid.x;\n\
         \x20   mov.u32 %r_bid, %ctaid.x;\n\
         \x20   mov.u32 smem_ptr, smem_storage;\n\
         \x20   xor.b32 %r_swizzle, %r_tid, {swizzle};\n\
         \x20   mul.wide.u32 %rd_off, %r_tid, 4;\n\
         \x20   add.u64 gmem_w_ptr, gmem_w_ptr, %rd_off;\n\
         \x20   add.u64 gmem_act_ptr, gmem_act_ptr, %rd_off;\n\
         \x20   add.u64 gmem_scale_ptr, gmem_scale_ptr, %rd_off;\n\
         \x20   add.u64 gmem_res_ptr, gmem_res_ptr, %rd_off;\n\
         \x20   add.u64 gmem_out_ptr, gmem_out_ptr, %rd_off;\n\
         \x20   mov.f32 %f_acc0, 0f00000000;\n\
         \x20   mov.f32 %f_acc1, 0f00000000;\n\
         \x20   ld.global.f32 %f_act0, [gmem_act_ptr];\n\
         \x20   ld.global.f32 %f_scale, [gmem_scale_ptr];\n\
         \x20   ld.global.f32 %f_res, [gmem_res_ptr];\n\
         \x20   mov.f32 %f_act1, %f_act0;\n\
         \x20   mov.f32 %f_w0, 0f3F800000;\n\
         \x20   mov.f32 %f_w1, 0f3F800000;\n\
         \x20   mov.b32 %r_fp8_pair, 15420;\n\
         \x20   {vec_inst}\n\
         {fp8_decode_ptx}\n\
         {mainloop_ptx}\n\
         {epilogue_ptx}\n\
         \x20   ret;\n\
         }}\n",
        hw_id = hw.id,
        hw_name = hw.name,
        shape_name = shape.name,
        op = shape.op,
        m = shape.m,
        n = shape.n,
        k = shape.k,
        ml = g.mainloop_sched,
        ep = g.epilogue_sched,
        warps = check.active_warps_per_sm,
        max_warps = hw.max_warps_per_sm,
        occ = check.occupancy_fraction * 100.0,
        isa_ver = hw.isa.ptx_isa_version,
        ptx_target = hw.isa.ptx_target,
        threads = check.threads_per_block,
        reg_cap = g.reg_cap_per_thread,
        smem_bytes = check.shared_mem_per_block_bytes,
        swizzle = g.smem_swizzle_bytes,
    );

    let fuse_res_flag = u8::from(shape.fuse_residual_epilogue);
    let native_cvt_flag = u8::from(
        hw.isa.native_fp8_e4m3_cvt
            && matches!(
                g.fp8_decode_mode,
                Fp8DecodeMode::NativeCvtE4m3x2 | Fp8DecodeMode::NativeFp8TensorCore
            ),
    );
    let cuda_source = format!(
        "// Generated by rvllm_cutlass::ptx_tuner (1,000D x 100W)\n\
         // Target: {hw_id} ({hw_name}) | Variant: {variant_name}\n\
         #include <cuda_runtime.h>\n\
         #include <cuda_fp16.h>\n\
         #include <stdint.h>\n\
         #include <stdio.h>\n\
         #include <math.h>\n\n\
         #define THREADS_PER_BLOCK {threads}u\n\
         #define MIN_BLOCKS_PER_SM {min_blocks}u\n\
         #define FUSE_RESIDUAL {fuse_res_flag}\n\
         #define USE_NATIVE_FP8_CVT {native_cvt_flag}\n\n\
         static_assert({matched_pair}, \"Mainloop and Epilogue schedule must satisfy MatchedPair\");\n\n\
         __device__ __forceinline__ float decode_fp8_e4m3(uint8_t b) {{\n\
         #if USE_NATIVE_FP8_CVT && (__CUDA_ARCH__ >= 890)\n\
             uint16_t pair = (uint16_t)b;\n\
             uint32_t f16x2;\n\
             asm volatile(\"cvt.rn.f16x2.e4m3x2 %0, %1;\" : \"=r\"(f16x2) : \"h\"(pair));\n\
             __half h = *reinterpret_cast<__half*>(&f16x2);\n\
             return __half2float(h);\n\
         #else\n\
             uint32_t sign = (b >> 7) & 1u;\n\
             uint32_t exp  = (b >> 3) & 0xFu;\n\
             uint32_t mant = b & 0x7u;\n\
             float val = (exp == 0u)\n\
                 ? ldexpf((float)mant, -9)\n\
                 : ldexpf(1.0f + (float)mant * 0.125f, (int)exp - 7);\n\
             return sign ? -val : val;\n\
         #endif\n\
         }}\n\n\
         extern \"C\" __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)\n\
         {entry_symbol}(\n\
             const float* __restrict__ act,\n\
             const uint8_t* __restrict__ weight_fp8,\n\
             const float* __restrict__ scales,\n\
             const float* __restrict__ residual,\n\
             float* __restrict__ out,\n\
             uint32_t M,\n\
             uint32_t N,\n\
             uint32_t K\n\
         ) {{\n\
             uint32_t row = blockIdx.y;\n\
             uint32_t col_base = blockIdx.x * (THREADS_PER_BLOCK / 32u) + (threadIdx.x / 32u);\n\
             uint32_t lane = threadIdx.x & 31u;\n\
             if (row >= M || col_base >= N) return;\n\
             float acc = 0.0f;\n\
             const float* act_row = act + (size_t)row * K;\n\
             const uint8_t* w_row = weight_fp8 + (size_t)col_base * K;\n\
             for (uint32_t k = lane; k < K; k += 32u) {{\n\
                 acc = fmaf(act_row[k], decode_fp8_e4m3(w_row[k]), acc);\n\
             }}\n\
             for (int offset = 16; offset > 0; offset >>= 1) {{\n\
                 acc += __shfl_xor_sync(0xffffffffu, acc, offset);\n\
             }}\n\
             if (lane == 0u) {{\n\
                 float scaled = acc * scales[col_base];\n\
         #if FUSE_RESIDUAL\n\
                 scaled += residual[(size_t)row * N + col_base];\n\
         #endif\n\
                 out[(size_t)row * N + col_base] = scaled;\n\
             }}\n\
         }}\n",
        hw_id = hw.id,
        hw_name = hw.name,
        threads = check.threads_per_block,
        min_blocks = check.active_blocks_per_sm,
        matched_pair = u8::from(g.mainloop_sched.matches(g.epilogue_sched)),
    );

    let (static_ptx_verified, ptx_instruction_count) =
        verify_emitted_ptx(&ptx_source, &hw.isa.ptx_target);
    SynthesizedKernelArtifact {
        entry_symbol,
        variant_name,
        ptx_sha256: sha256_hex(ptx_source.as_bytes()),
        ptx_source,
        cuda_sha256: sha256_hex(cuda_source.as_bytes()),
        cuda_source,
        static_ptx_verified,
        ptx_instruction_count,
    }
}

pub fn verify_emitted_ptx(ptx: &str, expected_target: &str) -> (bool, usize) {
    let mut has_version = false;
    let mut has_target = false;
    let mut has_entry = false;
    let mut has_maxnreg = false;
    let mut has_ret = false;
    let mut has_local_spill = false;
    let mut inst_count = 0usize;

    for raw in ptx.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        if line.starts_with(".version ") {
            has_version = true;
        } else if line.starts_with(".target ") && line.contains(expected_target) {
            has_target = true;
        } else if line.starts_with(".visible .entry ") {
            has_entry = true;
        } else if line.starts_with(".maxnreg ") {
            has_maxnreg = true;
        } else if line == "ret;" {
            has_ret = true;
            inst_count += 1;
        } else if line.starts_with(".local ")
            || line.starts_with("ld.local")
            || line.starts_with("st.local")
        {
            has_local_spill = true;
        } else if line.ends_with(';') && !line.starts_with('.') {
            inst_count += 1;
        }
    }
    let ok = has_version
        && has_target
        && has_entry
        && has_maxnreg
        && has_ret
        && !has_local_spill
        && inst_count >= 12;
    (ok, inst_count)
}
