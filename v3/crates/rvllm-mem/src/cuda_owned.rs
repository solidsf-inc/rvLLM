//! `CudaOwned`: CUDA handle teardown safety.
//!
//! *"Any object owning a CUDA handle must guarantee its stream is idle
//! before destroying the handle."*
//!
//! Encoded as a trait: implementors provide the stream the handle is
//! tied to. The default drop helper (`fence_then_destroy`) can be called
//! from concrete `Drop` impls. This replaces the implicit ordering that
//! v2 encoded with a comment next to each `Drop`.

use crate::stream::Stream;

pub trait CudaOwned {
    /// The stream that synchronizes this handle's completion.
    fn stream_for_fence(&self) -> &Stream;

    /// Helper callable from `Drop`. A false result means the handle must not
    /// be destroyed because outstanding work may still reference it.
    fn fence_before_destroy(&self) -> bool {
        self.stream_for_fence().fence().is_ok()
    }
}
