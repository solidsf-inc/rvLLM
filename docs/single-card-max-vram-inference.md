# Single-card memory planning

Peak device memory is not just checkpoint size. Budget for loaded weights,
KV-cache pages, graph-stable input/output buffers, temporary GEMM and attention
workspaces, allocator alignment, and driver/runtime overhead.

Use checked arithmetic when deriving each region. Reserve headroom before
loading, reject a plan that exceeds the queried free memory, and never assume
that unified or host memory is a transparent substitute for device memory.
Context length, batch size, KV dtype, and graph buckets must all be included in
the receipt for a capacity result.

This repository publishes no universal “maximum model” or context claim.
Measure the exact released source, artifact set, model revision, and card.
