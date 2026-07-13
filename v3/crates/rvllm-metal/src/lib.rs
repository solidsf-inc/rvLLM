// Copyright 2026 m0at <47344131+m0at@users.noreply.github.com>
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
//
// Portions of this crate are adapted from mistral.rs revision
// 31c13eb4587d3e4a5204870c98b70c05a1e5c943:
// https://github.com/EricLBuehler/mistral.rs
// licensed under the MIT License, Copyright (c) 2024 Eric Buehler.
// Per-file MIT attribution headers are preserved on every lifted source
// file; see LICENSES/MIT-mistralrs and LICENSES/Apache-2.0-mlx at the
// workspace root for the full upstream license texts.

//! Apple Silicon Metal backend for rvllm.
//!
//! This crate exposes Metal kernel wrappers used by rvllm-attention,
//! rvllm-mem, rvllm-loader, and rvllm-runtime on macOS aarch64. Every
//! public item is gated behind
//! `#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]`
//! so the crate is a no-op on CUDA/Linux builds.
//!
//! No silent fallbacks: if the `metal` feature is off or the target is
//! not Apple Silicon, consumer code that requests a Metal kernel must
//! receive `MetalKernelError::FeatureNotAvailable`.

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub mod device;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub mod kernels;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub mod fp8_dequant;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub mod paged_attention;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub mod kv_ops;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub mod sdpa;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub mod gemv;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub use device::{MetalDevice, MetalKernelError};

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub use kernels::{MetalKernels, KERNEL_NAMES};
