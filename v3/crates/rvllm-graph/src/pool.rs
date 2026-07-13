//! Per-bucket captured-graph pool.

use std::collections::BTreeMap;

#[cfg(feature = "cuda")]
use std::cell::Cell;

use rvllm_core::{CudaErrorKind, GraphError, MetaLayoutHash, Result, RvllmError};
use rvllm_mem::context::CudaContextHandle;
use rvllm_metadata::MetadataLayout;

/// SHA-256 of a captured graph's node types, kernel launch descriptors,
/// and dependency edges.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct GraphFingerprint(pub [u8; 32]);

/// One captured CUDA graph and its executable instance.
#[derive(Debug)]
pub struct CapturedGraph {
    bucket: u32,
    max_blocks: u32,
    layout_hash: MetaLayoutHash,
    fingerprint: GraphFingerprint,
    raw: u64,
    exec: u64,
    #[cfg(feature = "cuda")]
    context: Option<CudaContextHandle>,
    #[cfg(feature = "cuda")]
    last_stream: Cell<u64>,
}

impl CapturedGraph {
    /// Capture kernel launches issued by `body` on `stream`.
    ///
    /// # Safety
    /// `stream` must be a live, non-default stream belonging to `context`.
    /// The closure may issue only stream-capture-safe operations, and every
    /// device pointer captured by those operations must outlive this graph.
    #[cfg(feature = "cuda")]
    pub unsafe fn capture(
        context: &CudaContextHandle,
        bucket: u32,
        max_blocks: u32,
        layout_hash: MetaLayoutHash,
        stream: u64,
        body: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        use cudarc::driver::sys::*;

        validate_descriptor(bucket, max_blocks, layout_hash)?;
        if stream == 0 {
            return Err(graph_err(
                GraphError::InvalidCapture {
                    reason: "capture requires a non-default stream",
                },
                bucket,
            ));
        }

        let _guard = context.make_current()?;
        let mut stream_context: CUcontext = core::ptr::null_mut();
        let status = cuStreamGetCtx(stream as CUstream, &mut stream_context);
        if status != CUresult::CUDA_SUCCESS {
            return Err(inspect_err(status, bucket));
        }
        let mut current_context: CUcontext = core::ptr::null_mut();
        let status = cuCtxGetCurrent(&mut current_context);
        if status != CUresult::CUDA_SUCCESS {
            return Err(inspect_err(status, bucket));
        }
        if stream_context != current_context {
            return Err(graph_err(
                GraphError::InvalidCapture {
                    reason: "stream belongs to a different CUDA context",
                },
                bucket,
            ));
        }

        let status = cuStreamBeginCapture_v2(
            stream as CUstream,
            CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        );
        if status != CUresult::CUDA_SUCCESS {
            return Err(graph_err(GraphError::CaptureFailed, bucket));
        }

        let body_result = body();
        let mut raw: CUgraph = core::ptr::null_mut();
        let end_status = cuStreamEndCapture(stream as CUstream, &mut raw);

        if let Err(error) = body_result {
            if !raw.is_null() {
                let _ = cuGraphDestroy(raw);
            }
            return Err(error);
        }
        if end_status != CUresult::CUDA_SUCCESS || raw.is_null() {
            if !raw.is_null() {
                let _ = cuGraphDestroy(raw);
            }
            return Err(graph_err(GraphError::CaptureFailed, bucket));
        }

        let fingerprint = match fingerprint_graph(raw, bucket) {
            Ok(value) => value,
            Err(error) => {
                let _ = cuGraphDestroy(raw);
                return Err(error);
            }
        };

        let mut exec: CUgraphExec = core::ptr::null_mut();
        let status = cuGraphInstantiateWithFlags(&mut exec, raw, 0);
        if status != CUresult::CUDA_SUCCESS || exec.is_null() {
            let _ = cuGraphDestroy(raw);
            return Err(graph_err(GraphError::InstantiateFailed, bucket));
        }

        Ok(Self {
            bucket,
            max_blocks,
            layout_hash,
            fingerprint,
            raw: raw as u64,
            exec: exec as u64,
            context: Some(context.clone()),
            last_stream: Cell::new(0),
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub unsafe fn capture(
        _context: &CudaContextHandle,
        bucket: u32,
        max_blocks: u32,
        layout_hash: MetaLayoutHash,
        _stream: u64,
        _body: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        validate_descriptor(bucket, max_blocks, layout_hash)?;
        Err(graph_err(
            GraphError::FeatureNotAvailable { feature: "cuda" },
            bucket,
        ))
    }

    /// Launch the captured graph on a stream in the graph's context.
    ///
    /// # Safety
    /// Captured device pointers must still be valid. Callers should obtain
    /// this graph through `GraphPool::check_before_replay` so the metadata
    /// layout is validated before launch.
    pub unsafe fn replay(&self, stream: u64) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::sys::*;

            if stream == 0 || self.exec == 0 || self.raw == 0 {
                return Err(graph_err(
                    GraphError::InvalidCapture {
                        reason: "graph or replay stream is not live",
                    },
                    self.bucket,
                ));
            }
            let context = self.context.as_ref().ok_or_else(|| {
                graph_err(
                    GraphError::InvalidCapture {
                        reason: "graph has no CUDA context lease",
                    },
                    self.bucket,
                )
            })?;
            let _guard = context.make_current()?;

            let mut stream_context: CUcontext = core::ptr::null_mut();
            let status = cuStreamGetCtx(stream as CUstream, &mut stream_context);
            if status != CUresult::CUDA_SUCCESS {
                return Err(inspect_err(status, self.bucket));
            }
            let mut current_context: CUcontext = core::ptr::null_mut();
            let status = cuCtxGetCurrent(&mut current_context);
            if status != CUresult::CUDA_SUCCESS {
                return Err(inspect_err(status, self.bucket));
            }
            if stream_context != current_context {
                return Err(graph_err(
                    GraphError::InvalidCapture {
                        reason: "replay stream belongs to a different CUDA context",
                    },
                    self.bucket,
                ));
            }

            #[cfg(debug_assertions)]
            if fingerprint_graph(self.raw as CUgraph, self.bucket)? != self.fingerprint {
                return Err(graph_err(GraphError::FingerprintMismatch, self.bucket));
            }

            self.last_stream.set(stream);
            let status = cuGraphLaunch(self.exec as CUgraphExec, stream as CUstream);
            if status != CUresult::CUDA_SUCCESS {
                return Err(graph_err(
                    GraphError::ReplayFailed {
                        cuda: CudaErrorKind::DriverStatus(status as i32),
                        kernel_at_fault: None,
                    },
                    self.bucket,
                ));
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = stream;
            Err(graph_err(
                GraphError::FeatureNotAvailable { feature: "cuda" },
                self.bucket,
            ))
        }
    }

    #[must_use]
    pub fn bucket(&self) -> u32 {
        self.bucket
    }

    #[must_use]
    pub fn max_blocks(&self) -> u32 {
        self.max_blocks
    }

    #[must_use]
    pub fn fingerprint(&self) -> GraphFingerprint {
        self.fingerprint
    }
}

#[cfg(feature = "cuda")]
impl Drop for CapturedGraph {
    fn drop(&mut self) {
        use cudarc::driver::sys::*;

        let Some(context) = self.context.take() else {
            return;
        };
        let safe_to_release = match context.make_current() {
            Ok(_guard) => unsafe {
                let stream = self.last_stream.get();
                let synchronized = stream == 0
                    || cuStreamSynchronize(stream as CUstream) == CUresult::CUDA_SUCCESS;
                if !synchronized {
                    false
                } else {
                    let exec_ok = self.exec == 0
                        || cuGraphExecDestroy(self.exec as CUgraphExec) == CUresult::CUDA_SUCCESS;
                    let raw_ok = self.raw == 0
                        || cuGraphDestroy(self.raw as CUgraph) == CUresult::CUDA_SUCCESS;
                    exec_ok && raw_ok
                }
            },
            Err(_) => false,
        };

        self.exec = 0;
        self.raw = 0;
        if !safe_to_release {
            core::mem::forget(context);
        }
    }
}

#[cfg(not(feature = "cuda"))]
impl Drop for CapturedGraph {
    fn drop(&mut self) {}
}

fn validate_descriptor(bucket: u32, max_blocks: u32, layout_hash: MetaLayoutHash) -> Result<()> {
    let canonical = MetadataLayout::compute(bucket, max_blocks)?;
    if canonical.hash() != layout_hash {
        return Err(graph_err(
            GraphError::CaptureMetadataMismatch {
                captured: layout_hash,
                replay: canonical.hash(),
            },
            bucket,
        ));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
unsafe fn fingerprint_graph(
    raw: cudarc::driver::sys::CUgraph,
    bucket: u32,
) -> Result<GraphFingerprint> {
    use cudarc::driver::sys::*;
    use sha2::{Digest, Sha256};

    const MAX_GRAPH_NODES: usize = 1_000_000;
    const MAX_GRAPH_EDGES: usize = 4_000_000;

    let mut node_count = 0usize;
    let status = cuGraphGetNodes(raw, core::ptr::null_mut(), &mut node_count);
    if status != CUresult::CUDA_SUCCESS {
        return Err(inspect_err(status, bucket));
    }
    if node_count == 0 || node_count > MAX_GRAPH_NODES {
        return Err(graph_err(
            GraphError::InvalidCapture {
                reason: "captured graph has an invalid node count",
            },
            bucket,
        ));
    }

    let mut nodes = Vec::new();
    nodes.try_reserve_exact(node_count).map_err(|_| {
        graph_err(
            GraphError::InvalidCapture {
                reason: "captured graph node list is too large",
            },
            bucket,
        )
    })?;
    nodes.resize(node_count, core::ptr::null_mut());
    let capacity = node_count;
    let status = cuGraphGetNodes(raw, nodes.as_mut_ptr(), &mut node_count);
    if status != CUresult::CUDA_SUCCESS {
        return Err(inspect_err(status, bucket));
    }
    if node_count == 0 || node_count > capacity {
        return Err(graph_err(
            GraphError::InvalidCapture {
                reason: "captured graph node count changed during inspection",
            },
            bucket,
        ));
    }
    nodes.truncate(node_count);

    let mut hash = Sha256::new();
    hash.update((node_count as u64).to_le_bytes());
    for &node in &nodes {
        if node.is_null() {
            return Err(graph_err(
                GraphError::InvalidCapture {
                    reason: "captured graph contains a null node",
                },
                bucket,
            ));
        }
        let mut node_type = CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        let status = cuGraphNodeGetType(node, &mut node_type);
        if status != CUresult::CUDA_SUCCESS {
            return Err(inspect_err(status, bucket));
        }
        hash.update((node_type as u32).to_le_bytes());
        if node_type == CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            let mut params: CUDA_KERNEL_NODE_PARAMS = core::mem::zeroed();
            let status = cuGraphKernelNodeGetParams_v2(node, &mut params);
            if status != CUresult::CUDA_SUCCESS {
                return Err(inspect_err(status, bucket));
            }
            hash.update((params.func as usize as u64).to_le_bytes());
            hash.update(params.gridDimX.to_le_bytes());
            hash.update(params.gridDimY.to_le_bytes());
            hash.update(params.gridDimZ.to_le_bytes());
            hash.update(params.blockDimX.to_le_bytes());
            hash.update(params.blockDimY.to_le_bytes());
            hash.update(params.blockDimZ.to_le_bytes());
            hash.update(params.sharedMemBytes.to_le_bytes());
        }
    }

    let mut edge_count = 0usize;
    let status = cuGraphGetEdges_v2(
        raw,
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        &mut edge_count,
    );
    if status != CUresult::CUDA_SUCCESS {
        return Err(inspect_err(status, bucket));
    }
    if edge_count > MAX_GRAPH_EDGES {
        return Err(graph_err(
            GraphError::InvalidCapture {
                reason: "captured graph has too many dependency edges",
            },
            bucket,
        ));
    }

    let mut from = Vec::new();
    let mut to = Vec::new();
    from.try_reserve_exact(edge_count).map_err(|_| {
        graph_err(
            GraphError::InvalidCapture {
                reason: "captured graph edge list is too large",
            },
            bucket,
        )
    })?;
    to.try_reserve_exact(edge_count).map_err(|_| {
        graph_err(
            GraphError::InvalidCapture {
                reason: "captured graph edge list is too large",
            },
            bucket,
        )
    })?;
    from.resize(edge_count, core::ptr::null_mut());
    to.resize(edge_count, core::ptr::null_mut());
    let edge_capacity = edge_count;
    if edge_count != 0 {
        let status = cuGraphGetEdges_v2(
            raw,
            from.as_mut_ptr(),
            to.as_mut_ptr(),
            core::ptr::null_mut(),
            &mut edge_count,
        );
        if status != CUresult::CUDA_SUCCESS {
            return Err(inspect_err(status, bucket));
        }
        if edge_count > edge_capacity {
            return Err(graph_err(
                GraphError::InvalidCapture {
                    reason: "captured graph edge count changed during inspection",
                },
                bucket,
            ));
        }
        from.truncate(edge_count);
        to.truncate(edge_count);
    }

    let mut node_ids = Vec::new();
    node_ids.try_reserve_exact(nodes.len()).map_err(|_| {
        graph_err(
            GraphError::InvalidCapture {
                reason: "captured graph node index is too large",
            },
            bucket,
        )
    })?;
    for (index, &node) in nodes.iter().enumerate() {
        node_ids.push((node as usize, index as u32));
    }
    node_ids.sort_unstable_by_key(|entry| entry.0);

    let mut edges = Vec::new();
    edges.try_reserve_exact(edge_count).map_err(|_| {
        graph_err(
            GraphError::InvalidCapture {
                reason: "captured graph edge index is too large",
            },
            bucket,
        )
    })?;
    for (&source, &target) in from.iter().zip(&to) {
        let source = node_index(&node_ids, source).ok_or_else(|| {
            graph_err(
                GraphError::InvalidCapture {
                    reason: "dependency edge references an unknown source node",
                },
                bucket,
            )
        })?;
        let target = node_index(&node_ids, target).ok_or_else(|| {
            graph_err(
                GraphError::InvalidCapture {
                    reason: "dependency edge references an unknown target node",
                },
                bucket,
            )
        })?;
        edges.push((source, target));
    }
    edges.sort_unstable();
    hash.update((edge_count as u64).to_le_bytes());
    for (source, target) in edges {
        hash.update(source.to_le_bytes());
        hash.update(target.to_le_bytes());
    }

    let digest = hash.finalize();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&digest);
    Ok(GraphFingerprint(bytes))
}

#[cfg(feature = "cuda")]
fn node_index(node_ids: &[(usize, u32)], node: cudarc::driver::sys::CUgraphNode) -> Option<u32> {
    node_ids
        .binary_search_by_key(&(node as usize), |entry| entry.0)
        .ok()
        .map(|index| node_ids[index].1)
}

#[cfg(feature = "cuda")]
fn inspect_err(status: cudarc::driver::sys::CUresult, bucket: u32) -> RvllmError {
    graph_err(
        GraphError::InspectionFailed {
            cuda: CudaErrorKind::DriverStatus(status as i32),
        },
        bucket,
    )
}

fn graph_err(kind: GraphError, bucket: u32) -> RvllmError {
    RvllmError::graph(kind, bucket)
}

/// Pool of graphs keyed by `(bucket, max_blocks)`.
#[derive(Default, Debug)]
pub struct GraphPool {
    graphs: BTreeMap<(u32, u32), CapturedGraph>,
}

impl GraphPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, graph: CapturedGraph) -> Result<()> {
        let key = (graph.bucket, graph.max_blocks);
        if self.graphs.contains_key(&key) {
            return Err(graph_err(
                GraphError::DuplicateBucket {
                    max_blocks: graph.max_blocks,
                },
                graph.bucket,
            ));
        }
        self.graphs.insert(key, graph);
        Ok(())
    }

    pub fn get(&self, bucket: u32, max_blocks: u32) -> Option<&CapturedGraph> {
        self.graphs.get(&(bucket, max_blocks))
    }

    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    /// Verify that `current` is the canonical layout for this graph key and
    /// that its hash still matches the descriptor captured by the graph.
    pub fn check_before_replay(
        &self,
        bucket: u32,
        max_blocks: u32,
        current: &MetadataLayout,
    ) -> Result<&CapturedGraph> {
        let graph = self.get(bucket, max_blocks).ok_or_else(|| {
            graph_err(
                GraphError::BucketMissing {
                    padded_batch: bucket,
                },
                bucket,
            )
        })?;
        let canonical = MetadataLayout::compute(bucket, max_blocks)?;
        if *current != canonical || current.hash() != graph.layout_hash {
            return Err(graph_err(
                GraphError::CaptureMetadataMismatch {
                    captured: graph.layout_hash,
                    replay: current.hash(),
                },
                bucket,
            ));
        }
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_graph(bucket: u32, max_blocks: u32) -> CapturedGraph {
        let layout = MetadataLayout::compute(bucket, max_blocks).unwrap();
        CapturedGraph {
            bucket,
            max_blocks,
            layout_hash: layout.hash(),
            fingerprint: GraphFingerprint([1u8; 32]),
            raw: 0,
            exec: 0,
            #[cfg(feature = "cuda")]
            context: None,
            #[cfg(feature = "cuda")]
            last_stream: Cell::new(0),
        }
    }

    #[test]
    fn matching_layout_is_accepted() {
        let mut pool = GraphPool::new();
        pool.insert(fake_graph(128, 129)).unwrap();
        let layout = MetadataLayout::compute(128, 129).unwrap();
        assert!(pool.check_before_replay(128, 129, &layout).is_ok());
    }

    #[test]
    fn drift_returns_typed_error() {
        let mut pool = GraphPool::new();
        pool.insert(fake_graph(128, 129)).unwrap();
        let wrong = MetadataLayout::compute(128, 257).unwrap();
        let error = pool.check_before_replay(128, 129, &wrong).unwrap_err();
        assert!(format!("{error}").contains("CaptureMetadataMismatch"));
    }

    #[test]
    fn missing_bucket_returns_typed_error() {
        let pool = GraphPool::new();
        let layout = MetadataLayout::compute(1, 8).unwrap();
        let error = pool.check_before_replay(1, 8, &layout).unwrap_err();
        let message = format!("{error}");
        assert!(message.contains("BucketMissing"));
        assert!(message.contains("padded_batch: 1"));
    }

    #[test]
    fn duplicate_key_is_rejected() {
        let mut pool = GraphPool::new();
        pool.insert(fake_graph(8, 16)).unwrap();
        let error = pool.insert(fake_graph(8, 16)).unwrap_err();
        assert!(format!("{error}").contains("DuplicateBucket"));
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn noncanonical_layout_is_rejected_even_with_matching_key() {
        let mut pool = GraphPool::new();
        pool.insert(fake_graph(8, 16)).unwrap();
        let mut layout = MetadataLayout::compute(8, 16).unwrap();
        layout.positions_off += 1;
        assert!(pool.check_before_replay(8, 16, &layout).is_err());
    }
}
