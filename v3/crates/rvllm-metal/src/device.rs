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

//! MTLDevice + MTLCommandQueue holder and crate-wide error type.
//!
//! All Metal types are feature-gated to `(feature = "metal",
//! target_os = "macos", target_arch = "aarch64")`. On other targets the
//! error enum is still exposed so consumer crates can compile their
//! feature-gated stubs without `cfg(feature = "metal")` everywhere.

#[derive(thiserror::Error, Debug)]
pub enum MetalKernelError {
    #[error("no Metal device is available on this system")]
    DeviceNotAvailable,
    #[error("feature not available: {0}")]
    FeatureNotAvailable(&'static str),
    #[error("kernel load failed: {0}")]
    KernelLoadFailed(String),
    #[error("dispatch failed: {0}")]
    DispatchFailed(String),
    #[error("invalid shape: {0}")]
    InvalidShape(String),
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod imp {
    use super::MetalKernelError;

    /// Thin owner of an `MTLDevice` + its primary `MTLCommandQueue`.
    ///
    /// `metal::Device` and `metal::CommandQueue` in `metal = "0.27"` are
    /// declared `Send + Sync` via `foreign_type!`, so this struct is
    /// naturally `Send + Sync`. Higher-level allocators (e.g. the KV
    /// page pool in `rvllm-mem`) wrap themselves in `Arc<Mutex<_>>` for
    /// per-instance mutation; the device itself is reference-counted by
    /// the underlying Metal runtime and is safe to share.
    pub struct MetalDevice {
        device: metal::Device,
        queue: metal::CommandQueue,
    }

    impl MetalDevice {
        /// Acquire the system-default `MTLDevice` and a fresh command queue.
        ///
        /// Returns `MetalKernelError::DeviceNotAvailable` if
        /// `MTLCreateSystemDefaultDevice` returns null (Metal not
        /// supported, headless VM without paravirt GPU, etc.).
        pub fn system_default() -> Result<Self, MetalKernelError> {
            let device =
                metal::Device::system_default().ok_or(MetalKernelError::DeviceNotAvailable)?;
            let queue = device.new_command_queue();
            Ok(Self { device, queue })
        }

        #[inline]
        pub fn device(&self) -> &metal::Device {
            &self.device
        }

        #[inline]
        pub fn queue(&self) -> &metal::CommandQueue {
            &self.queue
        }
    }

    impl std::fmt::Debug for MetalDevice {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MetalDevice")
                .field("name", &self.device.name())
                .field("registry_id", &self.device.registry_id())
                .finish()
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub use imp::MetalDevice;

#[cfg(all(
    feature = "metal",
    not(all(target_os = "macos", target_arch = "aarch64"))
))]
compile_error!(
    "feature = \"metal\" is only supported on macOS aarch64 (Apple Silicon). \
     Disable the `metal` feature on this target."
);
