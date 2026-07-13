// Copyright 2026 m0at
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::collections::HashMap;
use std::sync::Mutex;

use metal::{ComputePipelineState, FunctionConstantValues, Library, MTLDataType};

use crate::device::{MetalDevice, MetalKernelError};

/// Embedded Metal library produced by `build.rs`.
const KERNELS: &[u8] = include_bytes!(env!("RVLLM_METALLIB_PATH"));

/// Baseline entry points that must exist in every embedded metallib.
pub const KERNEL_NAMES: &[&str] = &[
    "fp8_perchannel_dequant_float",
    "fp8_perchannel_dequant_half",
    "fp8_perchannel_dequant_bfloat16_t",
    "dequant_fp8_blockwise_float",
    "dequant_fp8_blockwise_half",
    "dequant_fp8_blockwise_bfloat16_t",
    "paged_attention_bfloat16_t_cache_bfloat16_t_hs128_bs16_nt256_nsl32_ps0",
    "reshape_and_cache_kv_bfloat16_t_cache_bfloat16_t",
    "gather_kv_cache_cache_bfloat16_t_out_bfloat16_t",
    "rvllm_fp8_gemv_bf16scale_f32",
    "rvllm_fp8_gemv_f32scale_f32",
    "rvllm_fp8_many_gemv_bf16scale_f32",
    "rvllm_fp8_many_gemv_f32scale_f32",
    "rvllm_bf16_gemv_f32",
    "rvllm_gelu_tanh_mul_f32",
    "rvllm_fp8_gelu_down_bf16scale_f32",
    "rvllm_fp8_gelu_down_f32scale_f32",
    "rvllm_host_f32_attention",
    "rvllm_bf16_lm_head_argmax_gemv",
    "rvllm_lm_head_logsumexp_f32",
    "rvllm_lm_head_argmax_reduce",
];

/// Owner of the Metal `Library` plus a lazy cache of compiled
/// `ComputePipelineState`s keyed by kernel name.
pub struct MetalKernels {
    device: metal::Device,
    library: Library,
    pipelines: Mutex<HashMap<String, ComputePipelineState>>,
}

impl std::fmt::Debug for MetalKernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cached = self.pipelines.lock().map(|p| p.len()).unwrap_or(0);
        f.debug_struct("MetalKernels")
            .field("device", &self.device.name().to_string())
            .field("cached_pipelines", &cached)
            .finish()
    }
}

impl MetalKernels {
    /// Load the embedded `.metallib` and prepare an empty pipeline cache.
    ///
    /// Returns `KernelLoadFailed` if the bytes cannot be parsed by Metal.
    pub fn new(device: &MetalDevice) -> Result<Self, MetalKernelError> {
        let raw = device.device();
        let library = raw
            .new_library_with_data(KERNELS)
            .map_err(|e| MetalKernelError::KernelLoadFailed(e.to_string()))?;
        for name in KERNEL_NAMES {
            library.get_function(name, None).map_err(|error| {
                MetalKernelError::KernelLoadFailed(format!(
                    "embedded metallib is missing `{name}`: {error}"
                ))
            })?;
        }
        Ok(Self {
            device: raw.clone(),
            library,
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    /// Borrow the underlying Metal library (for advanced consumers that
    /// need to construct pipelines with `FunctionConstantValues`).
    pub fn library(&self) -> &Library {
        &self.library
    }

    /// Lazily build and cache a `ComputePipelineState` for `name`.
    ///
    /// On a cache hit the cached pipeline is cloned and returned. On a
    /// miss the function is looked up in the embedded library and a new
    /// pipeline state is compiled.
    pub fn pipeline(&self, name: &str) -> Result<ComputePipelineState, MetalKernelError> {
        if name.is_empty()
            || name.len() > 256
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(MetalKernelError::KernelLoadFailed(format!(
                "invalid Metal entry-point name `{name}`"
            )));
        }
        {
            let cache = self.pipelines.lock().map_err(|e| {
                MetalKernelError::KernelLoadFailed(format!("pipeline cache poisoned: {e}"))
            })?;
            if let Some(p) = cache.get(name) {
                return Ok(p.clone());
            }
        }

        let function = self
            .library
            .get_function(name, None)
            .map_err(|e| MetalKernelError::KernelLoadFailed(format!("function `{name}`: {e}")))?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalKernelError::KernelLoadFailed(format!("pipeline `{name}`: {e}")))?;

        let mut cache = self.pipelines.lock().map_err(|e| {
            MetalKernelError::KernelLoadFailed(format!("pipeline cache poisoned: {e}"))
        })?;
        cache.insert(name.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    pub(crate) fn pipeline_with_bool_constant(
        &self,
        name: &str,
        index: u64,
        value: bool,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        let cache_key = format!("{name}#bool{index}={value}");
        {
            let cache = self.pipelines.lock().map_err(|error| {
                MetalKernelError::KernelLoadFailed(format!("pipeline cache poisoned: {error}"))
            })?;
            if let Some(pipeline) = cache.get(&cache_key) {
                return Ok(pipeline.clone());
            }
        }
        let constants = FunctionConstantValues::new();
        constants.set_constant_value_at_index(
            (&value as *const bool).cast(),
            MTLDataType::Bool,
            index,
        );
        let function = self
            .library
            .get_function(name, Some(constants))
            .map_err(|error| {
                MetalKernelError::KernelLoadFailed(format!(
                    "function `{name}` with bool constant {index}={value}: {error}"
                ))
            })?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| {
                MetalKernelError::KernelLoadFailed(format!("pipeline `{name}`: {error}"))
            })?;
        let mut cache = self.pipelines.lock().map_err(|error| {
            MetalKernelError::KernelLoadFailed(format!("pipeline cache poisoned: {error}"))
        })?;
        cache.insert(cache_key, pipeline.clone());
        Ok(pipeline)
    }
}
