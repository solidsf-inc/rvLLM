//! Frozen runtime configuration. Only constructible via
//! `RuntimeConfigBuilder::build(&model)` in `builder.rs`.

use std::path::{Path, PathBuf};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PreemptionMode {
    Recompute,
    Swap,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphMode {
    Off,
    Buckets(Vec<u32>),
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

/// Validated runtime configuration. Fields private to the config module
/// so callers can't skip the builder via struct-literal construction.
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub(super) device_id: u32,
    pub(super) max_batch: u32,
    pub(super) max_context: u32,
    pub(super) kv_block_size: u32,
    pub(super) num_gpu_blocks: u32,
    pub(super) num_cpu_blocks: u32,
    pub(super) gpu_memory_utilization: f32,
    pub(super) fp8_weights: bool,
    pub(super) fp8_kv_cache: bool,
    pub(super) graph_capture: GraphMode,
    pub(super) preemption: PreemptionMode,
    pub(super) log_level: LogLevel,
    pub(super) kernel_dir: Option<PathBuf>,
}

impl RuntimeConfig {
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
    pub fn max_batch(&self) -> u32 {
        self.max_batch
    }
    pub fn max_context(&self) -> u32 {
        self.max_context
    }
    pub fn kv_block_size(&self) -> u32 {
        self.kv_block_size
    }
    pub fn num_gpu_blocks(&self) -> u32 {
        self.num_gpu_blocks
    }
    pub fn num_cpu_blocks(&self) -> u32 {
        self.num_cpu_blocks
    }
    pub fn gpu_memory_utilization(&self) -> f32 {
        self.gpu_memory_utilization
    }
    pub fn fp8_weights(&self) -> bool {
        self.fp8_weights
    }
    pub fn fp8_kv_cache(&self) -> bool {
        self.fp8_kv_cache
    }
    pub fn graph_capture(&self) -> &GraphMode {
        &self.graph_capture
    }
    pub fn preemption(&self) -> PreemptionMode {
        self.preemption
    }
    pub fn log_level(&self) -> LogLevel {
        self.log_level
    }
    pub fn kernel_dir(&self) -> Option<&Path> {
        self.kernel_dir.as_deref()
    }
}
