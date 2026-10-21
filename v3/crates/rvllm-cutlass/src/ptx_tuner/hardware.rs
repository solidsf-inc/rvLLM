//! Uniform GPU hardware descriptor and PTX ISA capability flags.
//!
//! Every target GPU — from Volta (`sm_70`) through Blackwell Ultra (`sm_103a`)
//! and Grace-Blackwell Superchip (`sm_121a`), plus any arbitrary GPU probed at
//! runtime via `nvidia-smi` / `cuDeviceGetAttribute` — is represented by the
//! same declarative [`HardwareDescriptor`] and [`IsaCaps`] structs. The
//! `1,000D x 100W` schedule search contains zero SKU-specific branches.

use serde::{Deserialize, Serialize};

/// Declarative PTX & Tensor-Core ISA capability flags.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IsaCaps {
    /// Compute capability `major * 10 + minor` (e.g., `70`, `80`, `89`, `90`, `100`, `103`, `120`, `121`).
    pub sm_version: u32,
    /// Target directive for `.target` in PTX assembly (e.g., `sm_90a`, `sm_103a`, `sm_121a`).
    pub ptx_target: String,
    /// Minimum PTX ISA version string (e.g., `7.8`, `8.4`, `8.7`).
    pub ptx_isa_version: String,
    /// Supports `cp.async.ca.shared.global` / `cp.async.cg.shared.global` (`sm_80+`).
    pub cp_async: bool,
    /// Supports Hopper/Blackwell Tensor Memory Accelerator (`cp.async.bulk.tensor`, `sm_90+` datacenter).
    pub tma_bulk_tensor: bool,
    /// Supports Hopper warp-group MMA (`wgmma.mma_async.sync.aligned`, `sm_90a`).
    pub wgmma_async: bool,
    /// Supports Blackwell 5th-gen Tensor Core `tcgen05.mma` + Tensor Memory (`TMEM`, `sm_100a`/`sm_103a`).
    pub tcgen05_tmem: bool,
    /// Supports Thread Block Clusters and DSMEM (`cluster.ctarank`, `sm_90+`).
    pub thread_block_clusters: bool,
    /// Supports native hardware `cvt.rn.f16x2.e4m3x2` (`sm_89+`).
    pub native_fp8_e4m3_cvt: bool,
    /// Supports native FP8 Tensor Core MMA (`mma.sync.aligned.m16n8k32...e4m3`, `sm_89+`).
    pub fp8_tensor_core: bool,
    /// Supports native NVFP4 / MXFP4 block-scaled Tensor Core MMA (`sm_100+`).
    pub fp4_block_scale_mma: bool,
    /// Supports `redux.sync.add.s32` / `redux.sync.max.u32` (`sm_80+`).
    pub redux_sync: bool,
    /// True if 16-byte (`ld.global.nc.v4.u32`) vector loads inside tight unrolled K-loops
    /// trigger scoreboard stalls on unified LPDDR5X fabric (`sm_121` DGX Spark GB10),
    /// steering the tuner to 8-byte (`v2.u32`) vector width.
    pub vec128_global_load_hazard: bool,
}

/// Uniform physical hardware descriptor used by the `1,000D x 100W` search engine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardwareDescriptor {
    /// Canonical identifier (`sm_70`, `sm_80`, `sm_86`, `sm_89`, `sm_90a`, `sm_100`, `sm_103a`, `sm_120`, `sm_121a`).
    pub id: String,
    /// Human-readable device name.
    pub name: String,
    /// Vast.ai search query filter (`vastai search offers '<query>'`).
    pub vast_offer_query: String,
    /// Streaming Multiprocessor (SM) count.
    pub sm_count: u32,
    /// Memory bandwidth in GB/s (HBM2/HBM2e/HBM3/HBM3e/GDDR6X/GDDR7/LPDDR5X).
    pub hbm_bandwidth_gbps: f64,
    /// L2 cache capacity in bytes.
    pub l2_cache_bytes: u64,
    /// Peak dense FP16/BF16 Tensor Core throughput in TFLOPS.
    pub peak_fp16_tflops: f64,
    /// Peak dense FP8 E4M3 Tensor Core throughput in TFLOPS (or FP16 fallback when absent).
    pub peak_fp8_tflops: f64,
    /// Peak SIMT FP32 FMA throughput in TFLOPS.
    pub peak_simt_fp32_tflops: f64,
    /// 32-bit registers per SM (`65,536` on Volta through Blackwell).
    pub regs_per_sm: u32,
    /// Max 32-bit registers addressable per thread (`255`).
    pub max_regs_per_thread: u32,
    /// Configurable shared memory per SM in bytes.
    pub shared_mem_per_sm_bytes: u32,
    /// Tensor Memory (`TMEM`) per SM in bytes (`256 KiB` on `sm_100`/`sm_103a`, `0` elsewhere).
    pub tmem_per_sm_bytes: u32,
    /// Max resident warps per SM (`64` on datacenter SMs, `48` on GeForce/Ada/GB10 SMs).
    pub max_warps_per_sm: u32,
    /// Max resident thread blocks (CTAs) per SM.
    pub max_blocks_per_sm: u32,
    /// Warp size (`32`).
    pub warp_size: u32,
    /// Declarative PTX ISA capabilities.
    pub isa: IsaCaps,
}

impl HardwareDescriptor {
    /// Returns the canonical `rvllm_cutlass::Policy` architecture tag (`sm_80`, `sm_89`, `sm_90`, `sm_100`, `sm_121`, etc.).
    pub fn policy_arch_tag(&self) -> &str {
        self.id.as_str()
    }

    /// Construct a [`HardwareDescriptor`] from runtime device attributes probed on any
    /// Vast.ai or local GPU (`nvidia-smi` + `cuDeviceGetAttribute`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_device_probe(
        name: &str,
        cc_major: u32,
        cc_minor: u32,
        sm_count: u32,
        shared_mem_per_sm_bytes: u32,
        regs_per_sm: u32,
        max_warps_per_sm: u32,
        l2_cache_bytes: u64,
        hbm_bandwidth_gbps: f64,
        peak_fp16_tflops: f64,
    ) -> Self {
        let sm_version = cc_major * 10 + cc_minor;
        let has_arch_accel = matches!(sm_version, 90 | 100 | 103 | 121);
        let ptx_target = if has_arch_accel {
            format!("sm_{sm_version}a")
        } else {
            format!("sm_{sm_version}")
        };
        let ptx_isa_version = if sm_version >= 100 {
            "8.7".to_string()
        } else if sm_version >= 90 {
            "8.4".to_string()
        } else if sm_version >= 80 {
            "8.0".to_string()
        } else {
            "7.8".to_string()
        };
        let fp8_tensor_core = sm_version >= 89;
        let tcgen05_tmem = matches!(sm_version, 100 | 103);
        let wgmma_async = sm_version == 90;
        let tma_bulk_tensor = matches!(sm_version, 90 | 100 | 103 | 120);
        let vec128_global_load_hazard = sm_version == 121;

        Self {
            id: ptx_target.clone(),
            name: name.to_string(),
            vast_offer_query: format!("compute_cap>={sm_version}0 num_gpus=1"),
            sm_count: sm_count.max(1),
            hbm_bandwidth_gbps: hbm_bandwidth_gbps.max(100.0),
            l2_cache_bytes: l2_cache_bytes.max(4 * 1024 * 1024),
            peak_fp16_tflops: peak_fp16_tflops.max(10.0),
            peak_fp8_tflops: if fp8_tensor_core {
                peak_fp16_tflops * 2.0
            } else {
                peak_fp16_tflops
            },
            peak_simt_fp32_tflops: (peak_fp16_tflops * 0.25).max(5.0),
            regs_per_sm: regs_per_sm.max(65_536),
            max_regs_per_thread: 255,
            shared_mem_per_sm_bytes: shared_mem_per_sm_bytes.max(65_536),
            tmem_per_sm_bytes: if tcgen05_tmem { 262_144 } else { 0 },
            max_warps_per_sm: max_warps_per_sm.max(32),
            max_blocks_per_sm: if sm_version >= 89 { 24 } else { 32 },
            warp_size: 32,
            isa: IsaCaps {
                sm_version,
                ptx_target,
                ptx_isa_version,
                cp_async: sm_version >= 80,
                tma_bulk_tensor,
                wgmma_async,
                tcgen05_tmem,
                thread_block_clusters: sm_version >= 90,
                native_fp8_e4m3_cvt: sm_version >= 89,
                fp8_tensor_core,
                fp4_block_scale_mma: sm_version >= 100,
                redux_sync: sm_version >= 80,
                vec128_global_load_hazard,
            },
        }
    }

    /// Look up a hardware target by id or alias (`v100`, `a100`, `3090`, `4090`, `h100`, `b200`, `gb300`, `5090`, `gb10`, etc.).
    pub fn by_id(query: &str) -> Option<Self> {
        let q = query.trim().to_ascii_lowercase();
        let canonical = match q.as_str() {
            "sm_70" | "sm70" | "v100" | "volta" => "sm_70",
            "sm_80" | "sm80" | "a100" | "ampere" => "sm_80",
            "sm_86" | "sm86" | "rtx3090" | "3090" | "a10g" => "sm_86",
            "sm_89" | "sm89" | "rtx4090" | "4090" | "l40s" | "ada" => "sm_89",
            "sm_90" | "sm_90a" | "sm90" | "sm90a" | "h100" | "h200" | "hopper" => "sm_90a",
            "sm_100" | "sm_100a" | "sm100" | "b200" | "gb200" | "blackwell" => "sm_100",
            "sm_103" | "sm_103a" | "sm103" | "sm103a" | "b300" | "gb300" => "sm_103a",
            "sm_120" | "sm120" | "rtx5090" | "5090" | "rtx_pro_6000" => "sm_120",
            "sm_121" | "sm_121a" | "sm121" | "sm121a" | "gb10" | "dgx_spark" | "spark" => "sm_121a",
            other => other,
        };
        all_hardware_catalog()
            .into_iter()
            .find(|h| h.id.eq_ignore_ascii_case(canonical))
    }
}

/// Complete cross-architecture catalog from Volta (`sm_70`) to Blackwell Ultra (`sm_103a`) and DGX Spark (`sm_121a`).
pub fn all_hardware_catalog() -> Vec<HardwareDescriptor> {
    vec![
        HardwareDescriptor {
            id: "sm_70".into(),
            name: "NVIDIA V100 SXM2 32GB (Volta)".into(),
            vast_offer_query: "gpu_name=V100 num_gpus=1 reliability>=0.95 dph_total<0.60".into(),
            sm_count: 80,
            hbm_bandwidth_gbps: 900.0,
            l2_cache_bytes: 6 * 1024 * 1024,
            peak_fp16_tflops: 125.0,
            peak_fp8_tflops: 125.0,
            peak_simt_fp32_tflops: 15.7,
            regs_per_sm: 65_536,
            max_regs_per_thread: 255,
            shared_mem_per_sm_bytes: 98_304,
            tmem_per_sm_bytes: 0,
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            warp_size: 32,
            isa: IsaCaps {
                sm_version: 70,
                ptx_target: "sm_70".into(),
                ptx_isa_version: "7.8".into(),
                cp_async: false,
                tma_bulk_tensor: false,
                wgmma_async: false,
                tcgen05_tmem: false,
                thread_block_clusters: false,
                native_fp8_e4m3_cvt: false,
                fp8_tensor_core: false,
                fp4_block_scale_mma: false,
                redux_sync: false,
                vec128_global_load_hazard: false,
            },
        },
        HardwareDescriptor {
            id: "sm_80".into(),
            name: "NVIDIA A100 SXM4 80GB (Ampere)".into(),
            vast_offer_query: "gpu_name=A100_SXM4 num_gpus=1 reliability>=0.95 dph_total<1.50"
                .into(),
            sm_count: 108,
            hbm_bandwidth_gbps: 2_039.0,
            l2_cache_bytes: 40 * 1024 * 1024,
            peak_fp16_tflops: 312.0,
            peak_fp8_tflops: 312.0,
            peak_simt_fp32_tflops: 19.5,
            regs_per_sm: 65_536,
            max_regs_per_thread: 255,
            shared_mem_per_sm_bytes: 167_936,
            tmem_per_sm_bytes: 0,
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            warp_size: 32,
            isa: IsaCaps {
                sm_version: 80,
                ptx_target: "sm_80".into(),
                ptx_isa_version: "8.0".into(),
                cp_async: true,
                tma_bulk_tensor: false,
                wgmma_async: false,
                tcgen05_tmem: false,
                thread_block_clusters: false,
                native_fp8_e4m3_cvt: false,
                fp8_tensor_core: false,
                fp4_block_scale_mma: false,
                redux_sync: true,
                vec128_global_load_hazard: false,
            },
        },
        HardwareDescriptor {
            id: "sm_86".into(),
            name: "NVIDIA RTX 3090 / A10G 24GB (Ampere GA102)".into(),
            vast_offer_query: "gpu_name=RTX_3090 num_gpus=1 reliability>=0.95 dph_total<0.35"
                .into(),
            sm_count: 82,
            hbm_bandwidth_gbps: 936.0,
            l2_cache_bytes: 6 * 1024 * 1024,
            peak_fp16_tflops: 142.0,
            peak_fp8_tflops: 142.0,
            peak_simt_fp32_tflops: 35.6,
            regs_per_sm: 65_536,
            max_regs_per_thread: 255,
            shared_mem_per_sm_bytes: 102_400,
            tmem_per_sm_bytes: 0,
            max_warps_per_sm: 48,
            max_blocks_per_sm: 16,
            warp_size: 32,
            isa: IsaCaps {
                sm_version: 86,
                ptx_target: "sm_86".into(),
                ptx_isa_version: "8.0".into(),
                cp_async: true,
                tma_bulk_tensor: false,
                wgmma_async: false,
                tcgen05_tmem: false,
                thread_block_clusters: false,
                native_fp8_e4m3_cvt: false,
                fp8_tensor_core: false,
                fp4_block_scale_mma: false,
                redux_sync: true,
                vec128_global_load_hazard: false,
            },
        },
        HardwareDescriptor {
            id: "sm_89".into(),
            name: "NVIDIA RTX 4090 / L40S (Ada Lovelace)".into(),
            vast_offer_query: "gpu_name=RTX_4090 num_gpus=1 reliability>=0.95 dph_total<0.60"
                .into(),
            sm_count: 128,
            hbm_bandwidth_gbps: 1_008.0,
            l2_cache_bytes: 72 * 1024 * 1024,
            peak_fp16_tflops: 330.0,
            peak_fp8_tflops: 660.0,
            peak_simt_fp32_tflops: 82.6,
            regs_per_sm: 65_536,
            max_regs_per_thread: 255,
            shared_mem_per_sm_bytes: 102_400,
            tmem_per_sm_bytes: 0,
            max_warps_per_sm: 48,
            max_blocks_per_sm: 24,
            warp_size: 32,
            isa: IsaCaps {
                sm_version: 89,
                ptx_target: "sm_89".into(),
                ptx_isa_version: "8.4".into(),
                cp_async: true,
                tma_bulk_tensor: false,
                wgmma_async: false,
                tcgen05_tmem: false,
                thread_block_clusters: false,
                native_fp8_e4m3_cvt: true,
                fp8_tensor_core: true,
                fp4_block_scale_mma: false,
                redux_sync: true,
                vec128_global_load_hazard: false,
            },
        },
        HardwareDescriptor {
            id: "sm_90a".into(),
            name: "NVIDIA H100 SXM5 80GB / H200 141GB (Hopper)".into(),
            vast_offer_query: "gpu_name=H100_SXM num_gpus=1 reliability>=0.95 dph_total<2.80"
                .into(),
            sm_count: 132,
            hbm_bandwidth_gbps: 3_350.0,
            l2_cache_bytes: 50 * 1024 * 1024,
            peak_fp16_tflops: 989.0,
            peak_fp8_tflops: 1_979.0,
            peak_simt_fp32_tflops: 67.0,
            regs_per_sm: 65_536,
            max_regs_per_thread: 255,
            shared_mem_per_sm_bytes: 233_472,
            tmem_per_sm_bytes: 0,
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            warp_size: 32,
            isa: IsaCaps {
                sm_version: 90,
                ptx_target: "sm_90a".into(),
                ptx_isa_version: "8.4".into(),
                cp_async: true,
                tma_bulk_tensor: true,
                wgmma_async: true,
                tcgen05_tmem: false,
                thread_block_clusters: true,
                native_fp8_e4m3_cvt: true,
                fp8_tensor_core: true,
                fp4_block_scale_mma: false,
                redux_sync: true,
                vec128_global_load_hazard: false,
            },
        },
        HardwareDescriptor {
            id: "sm_100".into(),
            name: "NVIDIA B200 SXM6 192GB (Blackwell)".into(),
            vast_offer_query: "gpu_name=B200 num_gpus=1 reliability>=0.95 dph_total<5.50".into(),
            sm_count: 148,
            hbm_bandwidth_gbps: 8_000.0,
            l2_cache_bytes: 126 * 1024 * 1024,
            peak_fp16_tflops: 2_250.0,
            peak_fp8_tflops: 4_500.0,
            peak_simt_fp32_tflops: 90.0,
            regs_per_sm: 65_536,
            max_regs_per_thread: 255,
            shared_mem_per_sm_bytes: 233_472,
            tmem_per_sm_bytes: 262_144,
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            warp_size: 32,
            isa: IsaCaps {
                sm_version: 100,
                ptx_target: "sm_100a".into(),
                ptx_isa_version: "8.7".into(),
                cp_async: true,
                tma_bulk_tensor: true,
                wgmma_async: false,
                tcgen05_tmem: true,
                thread_block_clusters: true,
                native_fp8_e4m3_cvt: true,
                fp8_tensor_core: true,
                fp4_block_scale_mma: true,
                redux_sync: true,
                vec128_global_load_hazard: false,
            },
        },
        HardwareDescriptor {
            id: "sm_103a".into(),
            name: "NVIDIA B300 / GB300 288GB (Blackwell Ultra)".into(),
            vast_offer_query: "compute_cap>=1030 num_gpus=1 reliability>=0.95".into(),
            sm_count: 160,
            hbm_bandwidth_gbps: 9_600.0,
            l2_cache_bytes: 144 * 1024 * 1024,
            peak_fp16_tflops: 3_375.0,
            peak_fp8_tflops: 6_750.0,
            peak_simt_fp32_tflops: 110.0,
            regs_per_sm: 65_536,
            max_regs_per_thread: 255,
            shared_mem_per_sm_bytes: 233_472,
            tmem_per_sm_bytes: 262_144,
            max_warps_per_sm: 64,
            max_blocks_per_sm: 32,
            warp_size: 32,
            isa: IsaCaps {
                sm_version: 103,
                ptx_target: "sm_103a".into(),
                ptx_isa_version: "8.7".into(),
                cp_async: true,
                tma_bulk_tensor: true,
                wgmma_async: false,
                tcgen05_tmem: true,
                thread_block_clusters: true,
                native_fp8_e4m3_cvt: true,
                fp8_tensor_core: true,
                fp4_block_scale_mma: true,
                redux_sync: true,
                vec128_global_load_hazard: false,
            },
        },
        HardwareDescriptor {
            id: "sm_120".into(),
            name: "NVIDIA RTX 5090 32GB (Blackwell GeForce)".into(),
            vast_offer_query: "gpu_name=RTX_5090 num_gpus=1 reliability>=0.95 dph_total<1.20"
                .into(),
            sm_count: 170,
            hbm_bandwidth_gbps: 1_792.0,
            l2_cache_bytes: 96 * 1024 * 1024,
            peak_fp16_tflops: 419.0,
            peak_fp8_tflops: 838.0,
            peak_simt_fp32_tflops: 104.8,
            regs_per_sm: 65_536,
            max_regs_per_thread: 255,
            shared_mem_per_sm_bytes: 102_400,
            tmem_per_sm_bytes: 0,
            max_warps_per_sm: 48,
            max_blocks_per_sm: 24,
            warp_size: 32,
            isa: IsaCaps {
                sm_version: 120,
                ptx_target: "sm_120".into(),
                ptx_isa_version: "8.7".into(),
                cp_async: true,
                tma_bulk_tensor: true,
                wgmma_async: false,
                tcgen05_tmem: false,
                thread_block_clusters: true,
                native_fp8_e4m3_cvt: true,
                fp8_tensor_core: true,
                fp4_block_scale_mma: true,
                redux_sync: true,
                vec128_global_load_hazard: false,
            },
        },
        HardwareDescriptor {
            id: "sm_121a".into(),
            name: "NVIDIA GB10 Grace-Blackwell Superchip (DGX Spark 128GB)".into(),
            vast_offer_query: "compute_cap>=1210 num_gpus=1 reliability>=0.95".into(),
            sm_count: 48,
            hbm_bandwidth_gbps: 546.0,
            l2_cache_bytes: 48 * 1024 * 1024,
            peak_fp16_tflops: 250.0,
            peak_fp8_tflops: 500.0,
            peak_simt_fp32_tflops: 31.0,
            regs_per_sm: 65_536,
            max_regs_per_thread: 255,
            shared_mem_per_sm_bytes: 101_376,
            tmem_per_sm_bytes: 0,
            max_warps_per_sm: 48,
            max_blocks_per_sm: 24,
            warp_size: 32,
            isa: IsaCaps {
                sm_version: 121,
                ptx_target: "sm_121a".into(),
                ptx_isa_version: "8.7".into(),
                cp_async: true,
                tma_bulk_tensor: false,
                wgmma_async: false,
                tcgen05_tmem: false,
                thread_block_clusters: false,
                native_fp8_e4m3_cvt: true,
                fp8_tensor_core: true,
                fp4_block_scale_mma: true,
                redux_sync: true,
                vec128_global_load_hazard: true,
            },
        },
    ]
}
