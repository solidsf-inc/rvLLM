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

//! Metal-backed KV cache page allocator (Option A: single unified `MTLBuffer`
//! carved into fixed-size pages). Apple Silicon unified memory means the
//! pool is allocated `MTLStorageModeShared`; no explicit blit / barrier is
//! required for CPU<->GPU coherence.
//!
//! Mirrors the `HbmArena` invariant: once `MetalKvAllocator::new` returns,
//! the pool's base pointer is fixed for the allocator's lifetime. Pages are
//! handed out as `u32` indices; callers compute device offsets via
//! `page_offset(idx)` and pass `(pool_ptr, offset, page_bytes)` to Metal
//! kernels (paged_attention, reshape_and_cache, gather_kv_cache).

use crate::kv_layout::KvLayout;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod imp {
    use super::*;
    use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
    use std::collections::{HashMap, HashSet};
    use std::sync::{Arc, Mutex};

    use rvllm_metal::{MetalDevice, MetalKernelError};

    /// Fixed-size page pool over a single shared-storage `MTLBuffer`.
    ///
    /// Page size is derived from `KvLayout::block_bytes()` — the bytes for
    /// one K *or* V block at one layer. A "page" here is one K-block or one
    /// V-block; the per-sequence block table stores K and V page indices
    /// separately.
    pub struct MetalKvAllocator {
        device: Arc<MetalDevice>,
        pool: metal::Buffer,
        page_size_bytes: usize,
        num_pages_total: u32,
        free_list: Mutex<Vec<u32>>,
        allocated_pages: Mutex<HashSet<u32>>,
        allocated_count: AtomicU32,
    }

    impl MetalKvAllocator {
        /// Allocate a unified `MTLBuffer` of `num_pages * page_size_bytes`
        /// in `MTLStorageModeShared` and seed the free list LIFO with
        /// `[num_pages-1, num_pages-2, ..., 0]` so `allocate_blocks` hands
        /// out low indices first.
        pub fn new(
            device: Arc<MetalDevice>,
            layout: &KvLayout,
            num_pages: u32,
        ) -> Result<Self, MetalKernelError> {
            if num_pages == 0 {
                return Err(MetalKernelError::InvalidShape(
                    "num_pages must be > 0".into(),
                ));
            }
            let page_size_bytes = layout
                .block_bytes()
                .map_err(|e| MetalKernelError::InvalidShape(e.to_string()))?;
            let total_bytes = (num_pages as usize)
                .checked_mul(page_size_bytes)
                .ok_or_else(|| {
                    MetalKernelError::InvalidShape(
                        "num_pages * page_size_bytes overflows usize".into(),
                    )
                })?;

            let pool = device.device().new_buffer(
                total_bytes as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if pool.length() < total_bytes as u64 {
                return Err(MetalKernelError::DispatchFailed(format!(
                    "MTLBuffer alloc returned {} bytes, requested {}",
                    pool.length(),
                    total_bytes
                )));
            }

            let mut free_list: Vec<u32> = (0..num_pages).rev().collect();
            free_list.shrink_to_fit();

            Ok(Self {
                device,
                pool,
                page_size_bytes,
                num_pages_total: num_pages,
                free_list: Mutex::new(free_list),
                allocated_pages: Mutex::new(HashSet::new()),
                allocated_count: AtomicU32::new(0),
            })
        }

        #[inline]
        pub fn capacity(&self) -> u32 {
            self.num_pages_total
        }

        #[inline]
        pub fn allocated(&self) -> u32 {
            self.allocated_count.load(Ordering::Acquire)
        }

        #[inline]
        pub fn free(&self) -> u32 {
            self.num_pages_total - self.allocated()
        }

        /// Pop `num_blocks` page indices off the LIFO free list. On
        /// shortage, no pages are consumed and the error reports the
        /// requested-vs-available counts.
        pub fn allocate_blocks(&self, num_blocks: u32) -> Result<Vec<u32>, MetalKernelError> {
            if num_blocks == 0 {
                return Ok(Vec::new());
            }
            let mut guard = self.free_list.lock().map_err(|_| {
                MetalKernelError::DispatchFailed("MetalKvAllocator free_list poisoned".into())
            })?;
            let available = u32::try_from(guard.len()).map_err(|_| {
                MetalKernelError::InvalidShape("free-list length exceeds u32".into())
            })?;
            if num_blocks > available {
                return Err(MetalKernelError::DispatchFailed(format!(
                    "MetalKvAllocator: requested {} pages, only {} free",
                    num_blocks, available
                )));
            }
            let mut allocated = self.allocated_pages.lock().map_err(|_| {
                MetalKernelError::DispatchFailed("MetalKvAllocator allocated_pages poisoned".into())
            })?;
            let mut out = Vec::with_capacity(num_blocks as usize);
            for _ in 0..num_blocks {
                let Some(page) = guard.pop() else {
                    return Err(MetalKernelError::DispatchFailed(
                        "free-list changed while locked".into(),
                    ));
                };
                if !allocated.insert(page) {
                    return Err(MetalKernelError::DispatchFailed(
                        "free-list contained an allocated page".into(),
                    ));
                }
                out.push(page);
            }
            self.allocated_count.fetch_add(num_blocks, Ordering::AcqRel);
            Ok(out)
        }

        /// Return allocated pages. Invalid, duplicate, or already-free pages
        /// are rejected atomically so the free list can never alias a page.
        pub fn deallocate_blocks(&self, block_indices: &[u32]) -> Result<(), MetalKernelError> {
            if block_indices.is_empty() {
                return Ok(());
            }
            let mut guard = self.free_list.lock().map_err(|_| {
                MetalKernelError::DispatchFailed("MetalKvAllocator free_list poisoned".into())
            })?;
            let mut allocated = self.allocated_pages.lock().map_err(|_| {
                MetalKernelError::DispatchFailed("MetalKvAllocator allocated_pages poisoned".into())
            })?;
            let mut unique = HashSet::with_capacity(block_indices.len());
            for &idx in block_indices {
                if idx >= self.num_pages_total || !unique.insert(idx) || !allocated.contains(&idx) {
                    return Err(MetalKernelError::InvalidShape(format!(
                        "page {idx} is out of range, duplicated, or not allocated"
                    )));
                }
            }
            for &idx in block_indices {
                allocated.remove(&idx);
                guard.push(idx);
            }
            let returned = u32::try_from(block_indices.len()).map_err(|_| {
                MetalKernelError::InvalidShape("returned page count exceeds u32".into())
            })?;
            self.allocated_count.fetch_sub(returned, Ordering::AcqRel);
            Ok(())
        }

        #[inline]
        pub fn page_size_bytes(&self) -> usize {
            self.page_size_bytes
        }

        #[inline]
        pub fn pool(&self) -> &metal::Buffer {
            &self.pool
        }

        /// Byte offset of `page_index` within the unified pool buffer.
        /// Metal kernels do `(device u8*)pool + page_offset(idx)` and then
        /// cast to the KV element type.
        #[inline]
        pub fn page_offset(&self, page_index: u32) -> Result<usize, MetalKernelError> {
            if page_index >= self.num_pages_total {
                return Err(MetalKernelError::InvalidShape(format!(
                    "page index {page_index} exceeds capacity {}",
                    self.num_pages_total
                )));
            }
            (page_index as usize)
                .checked_mul(self.page_size_bytes)
                .ok_or_else(|| MetalKernelError::InvalidShape("page offset overflow".into()))
        }

        #[inline]
        pub fn device(&self) -> &MetalDevice {
            &self.device
        }
    }

    impl std::fmt::Debug for MetalKvAllocator {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MetalKvAllocator")
                .field("num_pages_total", &self.num_pages_total)
                .field("page_size_bytes", &self.page_size_bytes)
                .field("allocated", &self.allocated())
                .field("pool_len", &self.pool.length())
                .finish()
        }
    }

    // ------------------------------------------------------------------
    // MetalBufferRegistry — u64 sentinel ↔ (Arc<MTLBuffer>, byte offset)
    // ------------------------------------------------------------------
    //
    // rvllm-attention's launcher signatures take `u64` "device pointers"
    // (the CUDA paradigm). On Metal there are no addressable device
    // pointers — kernels bind `metal::Buffer` references with a byte
    // offset. The registry bridges the two: a launcher receives a `u64`
    // sentinel, looks it up here, and gets back `(Arc<Buffer>, offset,
    // size)` which it forwards to rvllm-metal wrappers.
    //
    // Sentinel range: we pick `0xFFFF_0000_0000_0000` upward. macOS user-
    // space pointers on Apple Silicon top out at 47 bits (0x0000_7FFF_...);
    // anything in the top 16 bits is guaranteed not to collide with a real
    // CUDA `CUdeviceptr` value. Each `register` increments by 64 so the
    // low 6 bits are always zero — that lets callers OR a sub-offset into
    // the sentinel if they want a "pointer + slot" addressing scheme
    // later (not used in v1).

    /// First sentinel handed out by `MetalBufferRegistry::register`.
    /// Values strictly below this are NOT Metal sentinels and a lookup
    /// against the registry will miss.
    pub const METAL_SENTINEL_BASE: u64 = 0xFFFF_0000_0000_0000;

    /// Step between successive sentinels. Keeps the low 6 bits free.
    const METAL_SENTINEL_STRIDE: u64 = 64;

    /// One entry in the registry. `buffer` is reference-counted so the
    /// registry can outlive the original allocator (defensive: nothing
    /// breaks if the allocator drops while the runtime still holds
    /// sentinels — the underlying MTLBuffer stays alive).
    #[derive(Clone)]
    struct MetalBufferEntry {
        buffer: Arc<metal::Buffer>,
        offset: usize,
        size: usize,
    }

    /// Process-wide registry mapping launcher-facing `u64` sentinels to
    /// the `(buffer, offset, size)` triples that rvllm-metal wrappers
    /// need. Construct one per `Engine`; pass an `Arc<Self>` to every
    /// crate that needs to translate.
    pub struct MetalBufferRegistry {
        entries: Mutex<HashMap<u64, MetalBufferEntry>>,
        next: AtomicU64,
    }

    impl MetalBufferRegistry {
        pub fn new() -> Self {
            Self {
                entries: Mutex::new(HashMap::new()),
                next: AtomicU64::new(METAL_SENTINEL_BASE),
            }
        }

        /// Register a (buffer, offset, size) triple and return its
        /// sentinel `u64`. Subsequent `lookup(sentinel)` calls return
        /// the same triple.
        pub fn register(
            &self,
            buffer: Arc<metal::Buffer>,
            offset: usize,
            size: usize,
        ) -> Result<u64, MetalKernelError> {
            if size == 0 {
                return Err(MetalKernelError::InvalidShape(
                    "registered buffer span must be non-empty".into(),
                ));
            }
            let end = offset.checked_add(size).ok_or_else(|| {
                MetalKernelError::InvalidShape("registered buffer span overflow".into())
            })?;
            if end > buffer.length() as usize {
                return Err(MetalKernelError::InvalidShape(format!(
                    "registered span {offset}..{end} exceeds buffer length {}",
                    buffer.length()
                )));
            }
            let sentinel = self
                .next
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    current
                        .checked_add(METAL_SENTINEL_STRIDE)
                        .filter(|next| *next >= METAL_SENTINEL_BASE)
                })
                .map_err(|_| {
                    MetalKernelError::DispatchFailed(
                        "MetalBufferRegistry: sentinel space exhausted".into(),
                    )
                })?;
            let mut g = self.entries.lock().map_err(|_| {
                MetalKernelError::DispatchFailed("MetalBufferRegistry entries lock poisoned".into())
            })?;
            if g.insert(
                sentinel,
                MetalBufferEntry {
                    buffer,
                    offset,
                    size,
                },
            )
            .is_some()
            {
                return Err(MetalKernelError::DispatchFailed(
                    "MetalBufferRegistry sentinel collision".into(),
                ));
            }
            Ok(sentinel)
        }

        /// True iff `ptr` is in the Metal sentinel range. Lets callers
        /// quickly reject obvious-CUDA pointers without taking the
        /// registry lock.
        #[inline]
        pub fn is_sentinel(ptr: u64) -> bool {
            ptr >= METAL_SENTINEL_BASE
        }

        /// Resolve a sentinel back to its `(buffer, offset, size)`.
        /// Returns `None` for non-sentinel `ptr` or for sentinels that
        /// were never registered / have been unregistered.
        pub fn lookup(&self, ptr: u64) -> Option<(Arc<metal::Buffer>, usize, usize)> {
            if !Self::is_sentinel(ptr) {
                return None;
            }
            let g = self.entries.lock().ok()?;
            g.get(&ptr)
                .map(|e| (Arc::clone(&e.buffer), e.offset, e.size))
        }

        /// Drop the registry entry for `ptr`. No-op for non-registered
        /// sentinels. The underlying MTLBuffer is dropped when its last
        /// `Arc` clone goes out of scope.
        pub fn unregister(&self, ptr: u64) {
            if !Self::is_sentinel(ptr) {
                return;
            }
            if let Ok(mut g) = self.entries.lock() {
                g.remove(&ptr);
            }
        }

        /// Number of live entries. Diagnostic only.
        pub fn len(&self) -> usize {
            self.entries.lock().map(|g| g.len()).unwrap_or(0)
        }

        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }
    }

    impl Default for MetalBufferRegistry {
        fn default() -> Self {
            Self::new()
        }
    }

    impl std::fmt::Debug for MetalBufferRegistry {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MetalBufferRegistry")
                .field("entries", &self.len())
                .field("next_sentinel", &self.next.load(Ordering::Relaxed))
                .finish()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use rvllm_core::DType;

        fn gemma4_layer_layout() -> KvLayout {
            // Gemma 4 31B: 4 KV heads, head_dim 256, block_size 16, BF16 (2B).
            KvLayout {
                num_blocks: 1,
                block_size: 16,
                num_kv_heads: 4,
                head_dim: 256,
                dtype: DType::Bf16,
            }
        }

        #[test]
        fn alloc_dealloc_round_trip() {
            let device = match MetalDevice::system_default() {
                Ok(d) => Arc::new(d),
                Err(_) => return, // no Metal on CI host, skip
            };
            let layout = gemma4_layer_layout();
            let alloc = MetalKvAllocator::new(device, &layout, 8).unwrap();
            assert_eq!(alloc.capacity(), 8);
            assert_eq!(alloc.allocated(), 0);
            assert_eq!(alloc.free(), 8);
            assert_eq!(alloc.page_size_bytes(), 16 * 4 * 256 * 2);

            let pages = alloc.allocate_blocks(3).unwrap();
            assert_eq!(pages.len(), 3);
            assert_eq!(alloc.allocated(), 3);
            assert_eq!(alloc.free(), 5);
            // LIFO: first allocation pops the lowest seeded index (0,1,2).
            assert_eq!(pages, vec![0, 1, 2]);

            // Page offsets are page_index * page_size_bytes.
            assert_eq!(alloc.page_offset(0).unwrap(), 0);
            assert_eq!(alloc.page_offset(1).unwrap(), alloc.page_size_bytes());
            assert_eq!(alloc.page_offset(7).unwrap(), 7 * alloc.page_size_bytes());

            alloc.deallocate_blocks(&pages).unwrap();
            assert_eq!(alloc.allocated(), 0);
            assert_eq!(alloc.free(), 8);
        }

        #[test]
        fn allocate_too_many_fails_atomically() {
            let device = match MetalDevice::system_default() {
                Ok(d) => Arc::new(d),
                Err(_) => return,
            };
            let layout = gemma4_layer_layout();
            let alloc = MetalKvAllocator::new(device, &layout, 4).unwrap();
            let err = alloc.allocate_blocks(5);
            assert!(err.is_err());
            assert_eq!(alloc.allocated(), 0);
            assert_eq!(alloc.free(), 4);
        }

        #[test]
        fn zero_request_is_noop() {
            let device = match MetalDevice::system_default() {
                Ok(d) => Arc::new(d),
                Err(_) => return,
            };
            let layout = gemma4_layer_layout();
            let alloc = MetalKvAllocator::new(device, &layout, 4).unwrap();
            assert!(alloc.allocate_blocks(0).unwrap().is_empty());
            alloc.deallocate_blocks(&[]).unwrap();
            assert_eq!(alloc.allocated(), 0);
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub use imp::{MetalBufferRegistry, MetalKvAllocator, METAL_SENTINEL_BASE};
