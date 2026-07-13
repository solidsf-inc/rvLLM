# Attention

The attention crate validates head dimensions 128, 256, and 512 and dispatches
explicit backend variants. CUDA paths include FA3 integration and sm_121 FA2/
unified kernels; Apple builds include a Metal backend.

Every launch contract includes query/KV shapes, page table and context extents,
scale, block size, workspace, stream, and backend-specific artifact ABI.
Unsupported combinations must return `FeatureNotAvailable` or a typed launch
error.

The generic non-Metal `PagedPrefillLauncher` rejects its unimplemented route
with `FeatureNotAvailable`; no caller may treat that route as executed
attention. Metal FP8 paged attention and some sliding-window combinations are
also explicitly unavailable. End-to-end support requires pinning every
external kernel revision/license/ABI and publishing reference parity for each
enabled route.
