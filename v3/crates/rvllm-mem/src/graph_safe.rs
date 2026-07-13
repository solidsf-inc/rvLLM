//! `GraphSafe` marker trait.
//!
//! A type is `GraphSafe` iff it is safe to bind by shared reference into
//! a CUDA graph capture region — that is, its device pointer will remain
//! valid across every replay of the captured graph.
//!
//! No realloc-capable wrapper may implement this trait. `capture::record`
//! remains unsafe because a closure can capture values that it does not bind
//! through `CaptureScope`.

/// # Safety
/// Implementors must guarantee that, for any live `&'a Self` held while
/// a captured graph is replayed, every device pointer derived from
/// `Self` remains valid and points at the same bytes it did at capture
/// time. Specifically:
/// - `Self` does not internally realloc or relocate.
/// - `Self` does not hand out pointers to memory owned by a reallocating
///   allocator.
/// - If `Self` borrows into an arena, the arena outlives the capture.
pub unsafe trait GraphSafe {}

// Primitive compile-time facts that help derive other implementations.
// Capture args are often small scalars copied into the graph's
// instantiated node — those are intrinsically safe.
unsafe impl GraphSafe for u8 {}
unsafe impl GraphSafe for u16 {}
unsafe impl GraphSafe for u32 {}
unsafe impl GraphSafe for u64 {}
unsafe impl GraphSafe for i8 {}
unsafe impl GraphSafe for i16 {}
unsafe impl GraphSafe for i32 {}
unsafe impl GraphSafe for i64 {}
unsafe impl GraphSafe for f32 {}
unsafe impl GraphSafe for f64 {}

#[cfg(test)]
mod tests {
    use super::*;

    fn accepts_graph_safe<T: GraphSafe>() {}

    #[test]
    fn primitives_are_graph_safe() {
        accepts_graph_safe::<u32>();
        accepts_graph_safe::<f32>();
        accepts_graph_safe::<u64>();
    }
}
