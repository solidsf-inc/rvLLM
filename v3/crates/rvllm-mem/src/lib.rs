//! rvllm-mem: HBM arena, tensor views, stream, event, capture scope,
//! pinned pool, and checked KV layout.
//!
//! The invariants this crate carries:
//! - Arena is fixed-size and non-relocating; once allocated, device
//!   pointers are stable for the arena's lifetime.
//! - Streams retain the primary CUDA context lease needed by their driver
//!   handles and remain pinned to the worker thread.
//! - `CaptureScope` binds only `GraphSafe` values; its unsafe record API makes
//!   the caller responsible for excluding captured allocators and side effects.
//! - CUDA handle teardown fences first and quarantines handles when a fence
//!   fails instead of destroying storage still referenced by device work.
//!
//! CUDA FFI is gated on `feature = "cuda"`. Without the feature, the
//! crate compiles with host stubs so invariant-level tests run on any
//! machine.

pub mod capture;
pub mod context;
pub mod cuda_owned;
pub mod event;
pub mod graph_safe;
pub mod hbm;
pub mod kv_layout;
#[cfg(feature = "metal")]
pub mod metal;
pub mod pinned;
pub mod stream;
pub mod tensor;
#[cfg(feature = "gb10")]
pub mod unified;
#[cfg(feature = "metal")]
pub use metal::{MetalBufferRegistry, MetalKvAllocator, METAL_SENTINEL_BASE};

pub use capture::{record, BoundHandle, CaptureScope, HasDevicePtr};
pub use context::CudaContextHandle;
pub use cuda_owned::CudaOwned;
pub use event::Event;
pub use graph_safe::GraphSafe;
pub use hbm::{HbmArena, Region};
pub use kv_layout::KvLayout;
pub use pinned::{DevicePod, PinnedBuf, PinnedPool};
pub use stream::Stream;
pub use tensor::Tensor;
#[cfg(feature = "gb10")]
pub use unified::UnifiedArena;
