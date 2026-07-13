# Contributor map

The current workspace is defined by `v3/Cargo.toml`, not by historical
milestones. Work normally follows these boundaries:

- model/config: `rvllm-core`;
- loading and memory: `rvllm-loader`, `rvllm-mem`;
- artifacts and execution: `rvllm-kernels`, `rvllm-cutlass`,
  `rvllm-attention`, `rvllm-fused`, `rvllm-graph`, `rvllm-sampling`;
- planning/runtime: `rvllm-metadata`, `rvllm-runtime`;
- interfaces: `rvllm-serve`, `rvllm-mcp`;
- validation: `rvllm-bench`, `rvllm-invariants`;
- optional Apple/vision: `rvllm-metal`, `rvllm-vision`, `rvllm-imageio`.

A feature is complete only when its unsupported cases fail explicitly and its
unit, reference, real-device, real-weight, and API gates appropriate to the
change pass. Compile-only Metal/vision paths and design-only optimizations must
remain labeled experimental.
